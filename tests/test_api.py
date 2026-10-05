"""API hardening tests — CPU-only, no network, no DB (fakes injected).

Covers: provider fallback, rate limiting, sanitized errors, single service build
under concurrency, and the pgvector reconnect + full-column upsert.
"""
from __future__ import annotations

import sys
import threading
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.assistant.serving.providers import HFInferenceProvider, UpstreamError

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from src.assistant.api import app as api  # noqa: E402


# ---------- provider fallback ----------
class _Resp:
    def __init__(self, text):
        msg = types.SimpleNamespace(content=text)
        self.choices = [types.SimpleNamespace(message=msg)]
        self.usage = types.SimpleNamespace(prompt_tokens=3, completion_tokens=2)


def _client(fail):
    def chat_completion(**_):
        if fail:
            raise RuntimeError("model_not_available")
        return _Resp(" ok [1]. ")
    return types.SimpleNamespace(chat_completion=chat_completion)


def test_provider_falls_through_to_next_live_provider():
    p = HFInferenceProvider("m", providers=["together", "featherless-ai"])
    p._clients = {"together": _client(True), "featherless-ai": _client(False)}
    out = p.generate([{"role": "user", "content": "q"}])
    assert out.text == "ok [1]." and out.prompt_tokens == 3


def test_provider_raises_upstream_error_when_all_fail():
    p = HFInferenceProvider("m", providers=["a", "b"])
    p._clients = {"a": _client(True), "b": _client(True)}
    with pytest.raises(UpstreamError, match="a: RuntimeError; b: RuntimeError"):
        p.generate([{"role": "user", "content": "q"}])


# ---------- rate limiter ----------
def test_rate_limiter_sliding_window():
    rl = api.RateLimiter(per_minute=2)
    assert rl.check("ip", now=0) == 0 and rl.check("ip", now=1) == 0
    assert rl.check("ip", now=2) == pytest.approx(58)     # blocked until t=60
    assert rl.check("other", now=2) == 0                   # per-client
    assert rl.check("ip", now=60.5) == 0                   # window slid
    assert api.RateLimiter(per_minute=0).check("ip") == 0  # disabled


# ---------- API behaviour ----------
class _Svc:
    def __init__(self, exc=None):
        self.exc = exc

    def answer(self, q):
        if self.exc:
            raise self.exc
        return types.SimpleNamespace(model_dump=lambda: {"answer": "a", "config": "base_rag"})


@pytest.fixture
def client(monkeypatch):
    with TestClient(api.app) as c:
        api._STATE["limiter"] = api.RateLimiter(per_minute=100)
        yield c


def test_query_ok_and_health_reports_last_outcome(client, monkeypatch):
    api._STATE["service"] = _Svc()
    r = client.post("/query", json={"question": "what is sepsis?"})
    assert r.status_code == 200 and r.json()["answer"] == "a"
    assert client.get("/health").json()["last_query"]["ok"] is True


def test_upstream_failure_is_502_without_internal_details(client):
    secret = "postgresql://user:pw@host/db"
    api._STATE["service"] = _Svc(UpstreamError(f"together blew up {secret}"))
    r = client.post("/query", json={"question": "what is sepsis?"},
                    headers={"x-request-id": "abc123"})
    assert r.status_code == 502
    assert "abc123" in r.json()["detail"] and secret not in r.text
    assert client.get("/health").json()["last_query"] == {
        "ok": False, "failed_stage": "model", "at": pytest.approx(0, abs=1e12)}
    assert api._STATE["service"] is not None   # don't rebuild on a model outage


def test_internal_failure_hides_exception_text(client):
    api._STATE["service"] = _Svc(ValueError("connection to postgresql://u:pw@h failed"))
    r = client.post("/query", json={"question": "what is sepsis?"})
    assert r.status_code == 502 and "pw@h" not in r.text


def test_rate_limit_returns_429_with_retry_after(client):
    api._STATE["service"] = _Svc()
    api._STATE["limiter"] = api.RateLimiter(per_minute=1)
    h = {"x-forwarded-for": "1.2.3.4, 10.0.0.1"}
    assert client.post("/query", json={"question": "abc"}, headers=h).status_code == 200
    r = client.post("/query", json={"question": "abc"}, headers=h)
    assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1


def test_service_built_once_under_concurrency(monkeypatch):
    calls = []

    def slow_build():
        calls.append(1)
        threading.Event().wait(0.05)
        return _Svc()
    monkeypatch.setattr(api, "_build_service", slow_build)
    api._STATE["service"] = None
    ts = [threading.Thread(target=api._get_service) for _ in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert len(calls) == 1


# ---------- pgvector store (fake psycopg) ----------
def test_pgvector_reconnects_once_and_upserts_all_columns(monkeypatch):
    class OpErr(Exception):
        pass
    monkeypatch.setitem(sys.modules, "psycopg", types.SimpleNamespace(OperationalError=OpErr))
    from src.assistant.rag import store as store_mod

    executed, conns = [], []

    class Cur:
        def __init__(self, conn): self.conn = conn
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, sql, params=None):
            if self.conn.dead:
                raise OpErr("server closed the connection")
            executed.append(sql)
        def executemany(self, sql, rows): executed.append(sql)
        def fetchone(self): return (7,)

    class Conn:
        def __init__(self): self.closed, self.dead = False, False
        def cursor(self): return Cur(self)
        def commit(self): pass
        def rollback(self): pass
        def close(self): self.closed = True

    def connect():
        conns.append(Conn())
        return conns[-1]
    monkeypatch.setattr(store_mod.PgVectorStore, "_connect", lambda self: connect())

    st = store_mod.PgVectorStore("dsn", 4)
    assert len(conns) == 1
    assert st.count() == 7 and len(conns) == 1        # connection reused
    conns[-1].dead = True                              # Neon dropped idle conn
    assert st.count() == 7 and len(conns) == 2        # one transparent reconnect

    from src.assistant.schema import Chunk, EmbeddedChunk
    ch = Chunk(chunk_id="c1", doc_id="d", source="pubmed", title="t", text="x",
               ordinal=0, url=None, metadata={})
    st.add([EmbeddedChunk(chunk=ch, embedding=[0.0] * 4)])
    assert "text = EXCLUDED.text" in executed[-1]


def test_pgvector_rebuild_loads_staging_then_swaps_atomically(monkeypatch):
    class OpErr(Exception):
        pass
    monkeypatch.setitem(sys.modules, "psycopg", types.SimpleNamespace(OperationalError=OpErr))
    from src.assistant.rag import store as store_mod

    log, commits = [], []

    class Cur:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, sql, params=None): log.append(" ".join(sql.split()))
        def executemany(self, sql, rows): log.append(f"MANY {len(rows)}")

    class Conn:
        closed = False
        def cursor(self): return Cur()
        def commit(self): commits.append(len(log))
        def rollback(self): pass
    monkeypatch.setattr(store_mod.PgVectorStore, "_connect", lambda self: Conn())

    from src.assistant.schema import Chunk, EmbeddedChunk
    st = store_mod.PgVectorStore("dsn", 4)
    log.clear(); commits.clear()
    emb = [EmbeddedChunk(chunk=Chunk(chunk_id=f"c{i}", doc_id="d", source="pubmed",
                                     title="t", text="x", ordinal=i), embedding=[0.0] * 4)
           for i in range(5)]
    st.rebuild(emb, batch=2)

    joined = "\n".join(log)
    assert "MANY 2" in joined and "MANY 1" in joined               # batched load
    i_index = next(i for i, s in enumerate(log) if s.startswith("CREATE INDEX biomed_chunks_new_emb_idx"))
    i_live = log.index("ALTER TABLE biomed_chunks RENAME TO biomed_chunks_prev")
    i_new = log.index("ALTER TABLE biomed_chunks_new RENAME TO biomed_chunks")
    assert max(i for i, s in enumerate(log) if s.startswith("MANY")) < i_index < i_live < i_new
    # Index build and both renames commit together: no window with no live table.
    swap_commit = next(c for c in commits if c > i_new)
    assert not any(i_index <= c - 1 < i_new for c in commits if c != swap_commit)
