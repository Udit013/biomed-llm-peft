### MedMCQA per-subject error analysis

| Subject | N | Base acc | Fine-tuned acc | Δ (pp) | Verdict |
|---|---:|---:|---:|---:|---|
| Anaesthesia | 5 | 40.0 | 40.0 | +0.0 | neutral |
| Anatomy | 11 | 54.5 | 54.5 | +0.0 | neutral |
| Biochemistry | 12 | 66.7 | 58.3 | -8.3 | worsened |
| Dental | 57 | 43.9 | 47.4 | +3.5 | improved |
| ENT | 5 | 20.0 | 20.0 | +0.0 | neutral |
| Forensic Medicine | 3 | 0.0 | 0.0 | +0.0 | neutral |
| Gynaecology & Obstetrics | 9 | 66.7 | 55.6 | -11.1 | worsened |
| Medicine | 12 | 50.0 | 58.3 | +8.3 | improved |
| Microbiology | 10 | 20.0 | 30.0 | +10.0 | improved |
| Ophthalmology | 6 | 16.7 | 16.7 | +0.0 | neutral |
| Pathology | 12 | 50.0 | 50.0 | +0.0 | neutral |
| Pediatrics | 13 | 61.5 | 61.5 | +0.0 | neutral |
| Pharmacology | 11 | 72.7 | 72.7 | +0.0 | neutral |
| Physiology | 13 | 61.5 | 69.2 | +7.7 | improved |
| Radiology | 1 | 0.0 | 0.0 | +0.0 | neutral |
| Social & Preventive Medicine | 3 | 66.7 | 66.7 | +0.0 | neutral |
| Surgery | 17 | 35.3 | 47.1 | +11.8 | improved |

#### Interpretation (auto-generated from measured deltas)

- **Improved** (5): Dental, Medicine, Microbiology, Physiology, Surgery
- **Neutral** (10): Anaesthesia, Anatomy, ENT, Forensic Medicine, Ophthalmology, Pathology, Pediatrics, Pharmacology, Radiology, Social & Preventive Medicine
- **Worsened** (2): Biochemistry, Gynaecology & Obstetrics

> Hypothesis to confirm/refine in the README: PEFT on MedMCQA tends to strengthen fact-recall-heavy subjects more than reasoning-heavy ones. Replace this line with the interpretation supported by the table above.
