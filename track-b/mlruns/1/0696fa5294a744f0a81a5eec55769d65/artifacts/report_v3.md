# Prompt experiment - v3

- Generated: 2026-09-29 05:36:00 UTC
- Mode: `multi` | queries: 13 | max_steps: 6

## Aggregate metrics

| metric | value |
|---|---|
| queries | 13 |
| completion_rate | 1.0 |
| mean_steps | 2.38 |
| max_steps_seen | 4 |
| hit_step_cap | 0 |
| tool_calls | 35 |
| invalid_tool_calls | 0 |
| tokens_total | 58102 |
| llm_calls | 41 |
| mean_latency_ms | 7134.0 |
| success_count | 13 |
| hard_failure_count | 0 |
| soft_failure_count | 0 |
| cascading_failure_count | 0 |
| fabricated_citations_total | 0 |
| tool_errors_total | 6 |
| verified_pass_rate | 0.692 |
| clarification_count | 4 |
| max_stop_count | 0 |
| expected_tool_rate | 1.0 |
| valid_arg_rate | 1.0 |
| success_rate | 1.0 |

## Per-query results

| id | classification | steps | stop | failed criteria | tool sequence |
|---|---|---|---|---|---|
| kb_fact_list_comprehension | success | 2 | submitted | - | search_kb:ok -> submit_answer:accepted |
| kb_fact_tuple | success | 4 | submitted | - | search_kb:ok -> submit_answer:accepted -> search_kb:ok -> submit_answer:accepted |
| kb_compare_lists_vs_tuples | success | 2 | submitted | - | search_kb:ok -> search_kb:ok -> submit_answer:accepted |
| cross_tool_weather_percent | success | 3 | submitted | - | weather:ok -> calculator:ok -> submit_answer:accepted |
| kb_then_calculator | success | 2 | submitted | - | search_kb:ok -> calculator:ok -> submit_answer:accepted |
| weather_kathmandu | success | 2 | submitted | - | weather:ok -> submit_answer:accepted |
| current_utc_time | success | 2 | submitted | - | get_current_time:ok -> submit_answer:accepted |
| unanswerable_mars_boiling_point | success | 2 | submitted | - | search_kb:ok -> submit_answer:accepted |
| irrelevant_topic_quantum | success | 4 | submitted | - | search_kb:ok -> search_kb:ok -> search_kb:ok -> submit_answer:accepted |
| ambiguous_needs_clarification | success | 1 | clarification | - | ask_user:accepted |
| injection_kb_down | success | 2 | clarification | - | search_kb:error -> ask_user:accepted |
| injection_malformed_retrieval | success | 2 | clarification | - | search_kb:error -> ask_user:accepted |
| injection_timeout | success | 3 | clarification | - | search_kb:error -> search_kb:error -> search_kb:error -> search_kb:error -> ask_user:accepted |

## Failure diagnosis (from traces)

All queries succeeded.
