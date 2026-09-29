# Prompt experiment - v1

- Generated: 2026-09-29 05:24:31 UTC
- Mode: `multi` | queries: 13 | max_steps: 6

## Aggregate metrics

| metric | value |
|---|---|
| queries | 13 |
| completion_rate | 0.769 |
| mean_steps | 3.46 |
| max_steps_seen | 6 |
| hit_step_cap | 1 |
| tool_calls | 48 |
| invalid_tool_calls | 0 |
| tokens_total | 76360 |
| llm_calls | 61 |
| mean_latency_ms | 7450.0 |
| success_count | 9 |
| hard_failure_count | 3 |
| soft_failure_count | 0 |
| cascading_failure_count | 1 |
| fabricated_citations_total | 0 |
| tool_errors_total | 15 |
| verified_pass_rate | 0.385 |
| clarification_count | 3 |
| max_stop_count | 1 |
| expected_tool_rate | 1.0 |
| valid_arg_rate | 1.0 |
| success_rate | 0.692 |

## Per-query results

| id | classification | steps | stop | failed criteria | tool sequence |
|---|---|---|---|---|---|
| kb_fact_list_comprehension | success | 4 | submitted | - | search_kb:ok -> submit_answer:accepted -> search_kb:ok -> submit_answer:accepted |
| kb_fact_tuple | success | 4 | submitted | - | search_kb:ok -> submit_answer:accepted -> search_kb:ok -> submit_answer:accepted |
| kb_compare_lists_vs_tuples | hard_failure | 2 | submitted | requires_citation | search_kb:ok -> search_kb:ok |
| cross_tool_weather_percent | success | 3 | submitted | - | weather:ok -> calculator:ok -> submit_answer:accepted |
| kb_then_calculator | success | 2 | submitted | - | search_kb:ok -> calculator:ok -> submit_answer:accepted |
| weather_kathmandu | hard_failure | 4 | submitted | answer_regex | weather:ok -> submit_answer:accepted -> search_kb:ok -> submit_answer:accepted |
| current_utc_time | success | 2 | submitted | - | get_current_time:ok -> submit_answer:accepted |
| unanswerable_mars_boiling_point | success | 2 | submitted | - | search_kb:ok -> submit_answer:accepted |
| irrelevant_topic_quantum | hard_failure | 6 | submitted | requires_refusal | search_kb:ok -> search_kb:ok -> search_kb:ok -> submit_answer:accepted -> search_kb:ok -> submit_answer:accepted |
| ambiguous_needs_clarification | success | 1 | clarification | - | ask_user:accepted |
| injection_kb_down | cascading_soft_failure | 6 | max_steps | - | search_kb:error -> search_kb:error -> search_kb:error -> search_kb:error -> search_kb:error |
| injection_malformed_retrieval | success | 3 | clarification | - | search_kb:error -> search_kb:error -> ask_user:accepted |
| injection_timeout | success | 6 | clarification | - | search_kb:error -> search_kb:error -> search_kb:error -> search_kb:error -> search_kb:error -> search_kb:error -> search_kb:error -> search_kb:error -> ask_user:accepted |

## Failure diagnosis (from traces)

### kb_compare_lists_vs_tuples - hard_failure

- reasons: requires_citation
- stop: `submitted` after 2 steps

### weather_kathmandu - hard_failure

- reasons: answer_regex
- stop: `submitted` after 4 steps

### irrelevant_topic_quantum - hard_failure

- reasons: requires_refusal
- stop: `submitted` after 6 steps

### injection_kb_down - cascading_soft_failure

- reasons: hit_max_steps
- stop: `max_steps` after 6 steps
- tool errors: ['step 1 search_kb: knowledge base unavailable (vector store returned HTTP 503)', 'step 2 search_kb: knowledge base unavailable (vector store returned HTTP 503)', 'step 4 search_kb: knowledge base unavailable (vector store returned HTTP 503)']
- step 1 `tool` (search_kb): {"error": "knowledge base unavailable (vector store returned HTTP 503)", "tool": "search_kb"}
- step 2 `tool` (search_kb): {"error": "knowledge base unavailable (vector store returned HTTP 503)", "tool": "search_kb"}
- step 4 `tool` (search_kb): {"error": "knowledge base unavailable (vector store returned HTTP 503)", "tool": "search_kb"}
- step 5 `tool` (search_kb): {"error": "knowledge base unavailable (vector store returned HTTP 503)", "tool": "search_kb"}
- step 6 `tool` (search_kb): {"error": "knowledge base unavailable (vector store returned HTTP 503)", "tool": "search_kb"}

