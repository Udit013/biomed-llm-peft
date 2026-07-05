"""Structured JSON logging (no third-party dependency).

Every log line is a single JSON object so logs are grep-able and machine-parsable
in production (Render, Docker). Use `get_logger(__name__)` and pass structured
fields as `extra={"stage": "retrieve", "latency_ms": 42}`.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": round(time.time(), 3),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Fold any structured `extra=...` fields into the JSON object.
        for key, val in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = val
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(os.environ.get("BIOMED_LOG_LEVEL", "INFO").upper())
        logger.propagate = False
    return logger


class timed:
    """Context manager that logs the wall-clock duration of a stage.

    with timed(logger, "retrieve", query=q) as t:
        ...
    # logs {"stage": "retrieve", "latency_ms": ...}; t.ms available after.
    """

    def __init__(self, logger: logging.Logger, stage: str, **fields):
        self.logger, self.stage, self.fields = logger, stage, fields
        self.ms: float = 0.0

    def __enter__(self) -> "timed":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.ms = round((time.perf_counter() - self._t0) * 1000, 2)
        self.logger.info(f"{self.stage} done", extra={"stage": self.stage,
                                                       "latency_ms": self.ms, **self.fields})
