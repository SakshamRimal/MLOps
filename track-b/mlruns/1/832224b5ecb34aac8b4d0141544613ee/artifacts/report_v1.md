# Prompt experiment - v1

- Generated: 2026-09-29 05:16:31 UTC
- Mode: `multi` | queries: 13 | max_steps: 6

## Aggregate metrics

| metric | value |
|---|---|
| queries | 13 |
| completion_rate | 0.615 |
| mean_steps | 4 |
| max_steps_seen | 6 |
| hit_step_cap | 0 |
| tool_calls | 55 |
| invalid_tool_calls | 1 |
| tokens_total | 60478 |
| llm_calls | 62 |
| mean_latency_ms | 7215.0 |
| success_count | 8 |
| hard_failure_count | 0 |
| soft_failure_count | 1 |
| cascading_failure_count | 4 |
| fabricated_citations_total | 1 |
| tool_errors_total | 33 |
| verified_pass_rate | 0.154 |
| clarification_count | 10 |
| max_stop_count | 0 |
| expected_tool_rate | 1.0 |
| valid_arg_rate | 0.982 |
| success_rate | 0.615 |

## Per-query results

| id | classification | steps | stop | failed criteria | tool sequence |
|---|---|---|---|---|---|
| kb_fact_list_comprehension | cascading_soft_failure | 3 | clarification | min_searches, requires_citation | search_kb:error -> search_kb:error -> ask_user:accepted |
| kb_fact_tuple | cascading_soft_failure | 3 | clarification | min_searches, requires_citation | search_kb:error -> search_kb:error -> ask_user:accepted |
| kb_compare_lists_vs_tuples | cascading_soft_failure | 5 | clarification | min_searches, requires_citation | search_kb:error -> search_kb:error -> search_kb:error -> search_kb:error -> search_kb:error -> search_kb:error -> ask_user:accepted |
| cross_tool_weather_percent | soft_failure | 5 | submitted | no_fabrication | weather:ok -> calculator:ok -> submit_answer:accepted -> weather:ok -> calculator:ok -> submit_answer:accepted |
| kb_then_calculator | cascading_soft_failure | 3 | clarification | answer_contains_any | search_kb:error -> calculator:ok -> search_kb:error -> ask_user:accepted |
| weather_kathmandu | success | 5 | submitted | - | weather:ok -> submit_answer:accepted -> search_kb:error -> get_current_time:ok -> submit_answer:accepted |
| current_utc_time | success | 2 | submitted | - | get_current_time:ok -> submit_answer:accepted |
| unanswerable_mars_boiling_point | success | 5 | clarification | - | search_kb:error -> search_kb:error -> search_kb:error -> ask_user:accepted |
| irrelevant_topic_quantum | success | 6 | clarification | - | search_kb:error -> search_kb:error -> search_kb:error -> search_kb:error -> ask_user:accepted |
| ambiguous_needs_clarification | success | 1 | clarification | - | ask_user:accepted |
| injection_kb_down | success | 6 | clarification | - | search_kb:error -> search_kb:error -> search_kb:error -> search_kb:error -> ask_user:accepted |
| injection_malformed_retrieval | success | 3 | clarification | - | search_kb:error -> search_kb:error -> ask_user:accepted |
| injection_timeout | success | 5 | clarification | - | search_kb:error -> search_kb:error -> search_kb:error -> search_kb:error -> search_kb:error -> search_kb:error -> ask_user:accepted |

## Failure diagnosis (from traces)

### kb_fact_list_comprehension - cascading_soft_failure

- reasons: requires_citation, min_searches
- stop: `clarification` after 3 steps
- tool errors: ["step 1 search_kb: AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'", "step 2 search_kb: AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'"]
- step 1 `tool` (search_kb): {"error": "AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'", "tool": "search_kb"}
- step 2 `tool` (search_kb): {"error": "AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'", "tool": "search_kb"}

### kb_fact_tuple - cascading_soft_failure

- reasons: requires_citation, min_searches
- stop: `clarification` after 3 steps
- tool errors: ["step 1 search_kb: AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'", "step 2 search_kb: AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'"]
- step 1 `tool` (search_kb): {"error": "AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'", "tool": "search_kb"}
- step 2 `tool` (search_kb): {"error": "AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'", "tool": "search_kb"}

### kb_compare_lists_vs_tuples - cascading_soft_failure

- reasons: requires_citation, min_searches
- stop: `clarification` after 5 steps
- tool errors: ["step 1 search_kb: AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'", "step 1 search_kb: AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'", "step 2 search_kb: AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'"]
- step 1 `tool` (search_kb): {"error": "AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'", "tool": "search_kb"}
- step 1 `tool` (search_kb): {"error": "AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'", "tool": "search_kb"}
- step 2 `tool` (search_kb): {"error": "AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'", "tool": "search_kb"}
- step 2 `tool` (search_kb): {"error": "AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'", "tool": "search_kb"}
- step 4 `tool` (search_kb): {"error": "AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'", "tool": "search_kb"}
- step 4 `tool` (search_kb): {"error": "AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'", "tool": "search_kb"}

### cross_tool_weather_percent - soft_failure

- reasons: no_fabrication
- stop: `submitted` after 5 steps
- tool errors: ['step 5 citation_check: 1 fabricated citation(s) dropped']

### kb_then_calculator - cascading_soft_failure

- reasons: answer_contains_any
- stop: `clarification` after 3 steps
- tool errors: ["step 1 search_kb: AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'", "step 2 search_kb: AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'"]
- step 1 `tool` (search_kb): {"error": "AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'", "tool": "search_kb"}
- step 2 `tool` (search_kb): {"error": "AttributeError: module 'pyarrow' has no attribute 'PyExtensionType'", "tool": "search_kb"}

