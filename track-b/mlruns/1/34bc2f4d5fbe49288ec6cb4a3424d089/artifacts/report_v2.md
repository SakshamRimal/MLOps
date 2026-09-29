# Prompt experiment - v2

- Generated: 2026-09-29 05:32:29 UTC
- Mode: `multi` | queries: 13 | max_steps: 6

## Aggregate metrics

| metric | value |
|---|---|
| queries | 13 |
| completion_rate | 0.923 |
| mean_steps | 2.85 |
| max_steps_seen | 6 |
| hit_step_cap | 0 |
| tool_calls | 44 |
| invalid_tool_calls | 0 |
| tokens_total | 82259 |
| llm_calls | 48 |
| mean_latency_ms | 7126.0 |
| success_count | 12 |
| hard_failure_count | 1 |
| soft_failure_count | 0 |
| cascading_failure_count | 0 |
| fabricated_citations_total | 0 |
| tool_errors_total | 6 |
| verified_pass_rate | 0.692 |
| clarification_count | 4 |
| max_stop_count | 0 |
| expected_tool_rate | 1.0 |
| valid_arg_rate | 1.0 |
| success_rate | 0.923 |

## Per-query results

| id | classification | steps | stop | failed criteria | tool sequence |
|---|---|---|---|---|---|
| kb_fact_list_comprehension | success | 4 | submitted | - | search_kb:ok -> submit_answer:accepted -> search_kb:ok -> submit_answer:accepted |
| kb_fact_tuple | success | 4 | submitted | - | search_kb:ok -> submit_answer:accepted -> search_kb:ok -> submit_answer:accepted |
| kb_compare_lists_vs_tuples | hard_failure | 6 | submitted | requires_citation | search_kb:ok -> search_kb:ok -> search_kb:ok -> search_kb:ok -> search_kb:ok -> search_kb:ok -> search_kb:ok -> search_kb:ok -> search_kb:ok -> search_kb:ok |
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

### kb_compare_lists_vs_tuples - hard_failure

- reasons: requires_citation
- stop: `submitted` after 6 steps

