# confirmed_only_v2 - katana benchmark

Built from `training/data_v9`. The primary `test.jsonl` benchmark keeps the family-disjoint test policy: clean rows and confirmed trainable attacks from `data_v9/splits/test.jsonl`, plus held-out gold rows whose family assignment is `test`.

| slice | rows |
|---|---:|
| clean test rows | 207 |
| confirmed trainable test attacks | 392 |
| test-family held-out gold attacks | 30 |
| **primary total** | **629** |

## All-Gold Stress Eval

`all_gold_stress.jsonl` keeps the same clean test rows but includes every held-out gold attack, regardless of family split. It is an attack-recall stress slice, not a family-disjoint leaderboard replacement.

| slice | rows |
|---|---:|
| clean test rows | 207 |
| all held-out gold attacks | 438 |
| **stress total** | **645** |

## Attack Labels - Primary

- `cognitive_state_attack`: 71
- `content_injection`: 68
- `semantic_manipulation`: 63
- `jailbreak`: 62
- `exfiltration_attempt`: 58
- `behavioral_control`: 48
- `encoding_evasion`: 35
- `persona_jailbreak`: 17

## Attack Labels - All-Gold Stress

- `content_injection`: 94
- `encoding_evasion`: 56
- `persona_jailbreak`: 56
- `behavioral_control`: 55
- `jailbreak`: 49
- `semantic_manipulation`: 46
- `cognitive_state_attack`: 44
- `exfiltration_attempt`: 38

## Rebuild

```bash
python evals/benchmarks/confirmed_only_v2/build.py
```
