# Parser fixtures

Real-world stdout captures from the agent CLIs the proving-ground evaluates. Each fixture pins one specific parser behavior so regressions are detected.

## Layout

```
fixtures/
├── codex_cli/
│   ├── text_only_response.txt
│   └── tool_with_reasoning_preamble.txt
├── gemini_cli/
│   ├── empty_response.txt
│   └── model_garbage_2_5_flash.txt
└── hermes_cli/
    ├── ok_with_tool_calls.txt
    ├── or_arcee_spark_empty.txt
    └── or_deepseek_v3_free_empty.txt
```

## Adding a fixture

1. Capture a real CLI run that exhibits a specific format / failure mode.
2. Redact any personal data, API keys, or workspace paths (`/home/<user>/...` → `/tmp/work/...`).
3. Save raw stdout (not stderr) as `<scenario>.txt` in the appropriate driver folder.
4. Add an expected `calls` list to `tests/parsers/test_agent_parsers.py` and replace the placeholder assertion.

## Why these specific scenarios

The 2026-05-04 reliability audit (see [`LIMITATIONS.md`](../../LIMITATIONS.md#parser-reliability-5-agents-below-85)) identified the failing scenarios. Pinning fixtures here means a future parser change must explicitly choose to break behavior on these known cases — no silent regression.

## Test status

| driver / scenario | fixture present? | expected output set? |
|---|---|---|
| codex_cli/text_only_response | TODO | TODO |
| codex_cli/tool_with_reasoning_preamble | TODO | TODO |
| gemini_cli/empty_response | TODO | TODO |
| gemini_cli/model_garbage_2_5_flash | TODO | TODO |
| hermes_cli/ok_with_tool_calls | TODO | TODO |
| hermes_cli/or_arcee_spark_empty | TODO | TODO |
| hermes_cli/or_deepseek_v3_free_empty | TODO | TODO |

Tests with `pytest.skip` if the fixture is missing — adding them flips them green.
