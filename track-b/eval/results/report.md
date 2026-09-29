# Evaluation Report - W16 Task 3: Agentify the Assistant

- Generated: 2026-09-28 09:15:45 UTC
- Backend: `http://localhost:8000` | provider `openai` | circuit `closed`
- Endpoint under test: `POST /chat/research`
- Harness: built from scratch (`eval/harness.py`), no evaluation framework

## 1. Task completion rate

| mode | queries | completed | completion rate |
|---|---|---|---|
| single | 13 | 12 | 92.3% |
| multi | 13 | 12 | 92.3% |

## 2. Tool-call correctness

| mode | expected tool selected | valid arguments | tool calls | invalid calls |
|---|---|---|---|---|
| single | 100.0% | 100.0% | 33 | 0 |
| multi | 100.0% | 100.0% | 45 | 0 |

## 3. Trajectory length (researcher iterations per query)

| mode | mean | median | max | hit step cap (forced synthesis) |
|---|---|---|---|---|
| single | 2.38 | 2 | 4 | 0 |
| multi | 3.31 | 2 | 6 | 2 |

Per-query tool sequence (each entry is one model decision):

| id | mode | steps | stop | tool sequence |
|---|---|---|---|---|
| kb_fact_list_comprehension | single | 2 | submitted | search_kb:ok -> submit_answer:accepted |
| kb_fact_tuple | single | 2 | submitted | search_kb:ok -> submit_answer:accepted |
| kb_compare_lists_vs_tuples | single | 2 | submitted | search_kb:ok -> search_kb:ok |
| cross_tool_weather_percent | single | 3 | submitted | weather:ok -> calculator:ok -> submit_answer:accepted |
| kb_then_calculator | single | 2 | submitted | search_kb:ok -> calculator:ok -> submit_answer:accepted |
| weather_kathmandu | single | 2 | submitted | weather:ok -> submit_answer:accepted |
| current_utc_time | single | 2 | submitted | get_current_time:ok -> submit_answer:accepted |
| unanswerable_mars_boiling_point | single | 2 | submitted | search_kb:ok -> submit_answer:accepted |
| irrelevant_topic_quantum | single | 4 | submitted | search_kb:ok -> search_kb:ok -> search_kb:ok -> submit_answer:accepted |
| ambiguous_needs_clarification | single | 1 | clarification | ask_user:accepted |
| injection_kb_down | single | 3 | submitted | search_kb:error -> search_kb:error |
| injection_malformed_retrieval | single | 3 | clarification | search_kb:error -> search_kb:error -> ask_user:accepted |
| injection_timeout | single | 3 | clarification | search_kb:error -> search_kb:error -> search_kb:error -> search_kb:error -> ask_user:accepted |
| kb_fact_list_comprehension | multi | 2 | submitted | search_kb:ok -> submit_answer:accepted |
| kb_fact_tuple | multi | 4 | submitted | search_kb:ok -> submit_answer:accepted -> search_kb:ok -> submit_answer:accepted |
| kb_compare_lists_vs_tuples | multi | 2 | submitted | search_kb:ok -> search_kb:ok |
| cross_tool_weather_percent | multi | 5 | submitted | weather:ok -> calculator:ok -> submit_answer:accepted -> search_kb:ok -> submit_answer:accepted |
| kb_then_calculator | multi | 2 | submitted | search_kb:ok -> calculator:ok -> submit_answer:accepted |
| weather_kathmandu | multi | 2 | submitted | weather:ok -> submit_answer:accepted |
| current_utc_time | multi | 2 | submitted | get_current_time:ok -> submit_answer:accepted |
| unanswerable_mars_boiling_point | multi | 2 | submitted | search_kb:ok -> submit_answer:accepted |
| irrelevant_topic_quantum | multi | 6 | max_steps | search_kb:ok -> search_kb:ok -> search_kb:ok -> search_kb:ok -> submit_answer:accepted -> search_kb:ok |
| ambiguous_needs_clarification | multi | 1 | clarification | ask_user:accepted |
| injection_kb_down | multi | 6 | max_steps | search_kb:error -> search_kb:error -> search_kb:error -> search_kb:error -> search_kb:error |
| injection_malformed_retrieval | multi | 3 | clarification | search_kb:error -> search_kb:error -> ask_user:accepted |
| injection_timeout | multi | 6 | clarification | search_kb:error -> search_kb:error -> search_kb:error -> search_kb:error -> search_kb:error -> search_kb:error -> search_kb:error -> ask_user:accepted |

## 4. Failure log

| id | mode | classification | failed criteria | steps | note |
|---|---|---|---|---|---|
| kb_compare_lists_vs_tuples | single | **hard_failure** | requires_citation | 2 | - |
| kb_compare_lists_vs_tuples | multi | **hard_failure** | requires_citation | 2 | - |
| irrelevant_topic_quantum | multi | **soft_failure** | hit_max_steps | 6 | - |
| injection_kb_down | multi | **cascading_soft_failure** | hit_max_steps | 6 | step 1 search_kb: knowledge base unavailable (vector store returned HTTP 503); step 2 search_kb: knowledge base unava... |

Classification totals: cascading_soft_failure=1, hard_failure=2, soft_failure=1, success=22

Taxonomy rules used by this harness:

- **Hard failure** - the task was not completed: required tool never called, required citation/refusal/clarification missing, required answer content missing, or a fabricated citation on a query where citations were required.
- **Soft failure** - the task completed but degraded: fewer successful searches than the query needs, a citation attached to a non-KB answer that the system rejected, or the step budget exhausted (forced synthesis).
- **Cascading soft failure** - at least one tool call failed during the run *and* the final output is still defective, i.e. the early defect propagated into the final answer.

## 5. Token and cost accounting

| id | mode | prompt | completion | total | LLM calls | latency (ms) |
|---|---|---|---|---|---|---|
| kb_fact_list_comprehension | single | 4625 | 285 | 4910 | 3 | 47078.0 |
| kb_fact_tuple | single | 4705 | 363 | 5068 | 3 | 6344.6 |
| kb_compare_lists_vs_tuples | single | 6831 | 697 | 7528 | 3 | 10137.7 |
| cross_tool_weather_percent | single | 3733 | 168 | 3901 | 4 | 8983.4 |
| kb_then_calculator | single | 4136 | 355 | 4491 | 3 | 5966.9 |
| weather_kathmandu | single | 2608 | 118 | 2726 | 3 | 7936.8 |
| current_utc_time | single | 2540 | 107 | 2647 | 3 | 3487.7 |
| unanswerable_mars_boiling_point | single | 2547 | 105 | 2652 | 3 | 3987.6 |
| irrelevant_topic_quantum | single | 11435 | 169 | 11604 | 5 | 6656.7 |
| ambiguous_needs_clarification | single | 777 | 36 | 813 | 1 | 1222.1 |
| injection_kb_down | single | 3754 | 148 | 3902 | 4 | 4359.6 |
| injection_malformed_retrieval | single | 2708 | 74 | 2782 | 3 | 3231.9 |
| injection_timeout | single | 2964 | 146 | 3110 | 3 | 3483.8 |
| kb_fact_list_comprehension | multi | 3558 | 193 | 3751 | 3 | 4506.8 |
| kb_fact_tuple | multi | 8831 | 533 | 9364 | 6 | 9627.6 |
| kb_compare_lists_vs_tuples | multi | 4513 | 511 | 5024 | 3 | 7069.5 |
| cross_tool_weather_percent | multi | 6589 | 314 | 6903 | 7 | 14549.1 |
| kb_then_calculator | multi | 2924 | 303 | 3227 | 3 | 5529.8 |
| weather_kathmandu | multi | 1992 | 103 | 2095 | 3 | 7390.2 |
| current_utc_time | multi | 1934 | 93 | 2027 | 3 | 3220.0 |
| unanswerable_mars_boiling_point | multi | 1964 | 105 | 2069 | 3 | 4746.1 |
| irrelevant_topic_quantum | multi | 20050 | 259 | 20309 | 8 | 10341.6 |
| ambiguous_needs_clarification | multi | 777 | 36 | 813 | 1 | 1024.5 |
| injection_kb_down | multi | 7934 | 218 | 8152 | 8 | 7922.4 |
| injection_malformed_retrieval | multi | 2706 | 94 | 2800 | 3 | 3072.4 |
| injection_timeout | multi | 7949 | 360 | 8309 | 7 | 9729.7 |

Mode comparison (single-agent baseline vs multi-agent):

| metric | single | multi | delta (multi - single) |
|---|---|---|---|
| completion rate | 92.3% | 92.3% |  |
| total tokens | 56134 | 74843 | 18709 |
| mean tokens / query | 4318 | 5757.2 | 1439.2 |
| prompt tokens | 53363 | 71721 | 18358 |
| LLM calls | 41 | 58 | 17 |
| mean steps | 2.38 | 3.31 | 0.93 |

## 6. Failure injection

| id | injected failure | mode | honest response? | fabricated citations | tool errors |
|---|---|---|---|---|---|
| injection_kb_down | kb_down | single | yes | 0 | 2 |
| injection_malformed_retrieval | malformed_retrieval | single | yes | 0 | 2 |
| injection_timeout | timeout | single | yes | 0 | 4 |
| injection_kb_down | kb_down | multi | yes | 0 | 5 |
| injection_malformed_retrieval | malformed_retrieval | multi | yes | 0 | 2 |
| injection_timeout | timeout | multi | yes | 0 | 7 |

Observed agent behaviour under injected failure:

- `injection_kb_down` (single): tool error surfaced -> `step 1 search_kb: knowledge base unavailable (vector store returned HTTP 503)`; answer starts: "I am currently unable to access the knowledge base to provide a definition of list comprehension. If you have any specific questions or need"
- `injection_malformed_retrieval` (single): tool error surfaced -> `step 1 search_kb: malformed retrieval output: expected list, got dict`; answer starts: "Could you please clarify what specific programming language or context you are referring to regarding tuples?"
- `injection_timeout` (single): tool error surfaced -> `step 1 search_kb: knowledge base search timed out after 8s`; answer starts: "It seems that the knowledge base is currently unavailable for searching. Would you like to ask a different question or provide more specific"
- `injection_kb_down` (multi): tool error surfaced -> `step 1 search_kb: knowledge base unavailable (vector store returned HTTP 503)`; answer starts: "I am currently unable to access the knowledge base to provide a definition of list comprehension."
- `injection_malformed_retrieval` (multi): tool error surfaced -> `step 1 search_kb: malformed retrieval output: expected list, got dict`; answer starts: "Could you please clarify what specific information you are looking for about tuples? For example, are you interested in their definition, ho"
- `injection_timeout` (multi): tool error surfaced -> `step 1 search_kb: knowledge base search timed out after 8s`; answer starts: "Could you specify what aspects of lists and tuples you would like to compare? For example, are you interested in their characteristics, perf"

## 7. Per-query outcome matrix

| id | multi | single |
|---|---|---|
| ambiguous_needs_clarification | PASS | PASS |
| cross_tool_weather_percent | PASS | PASS |
| current_utc_time | PASS | PASS |
| injection_kb_down | PASS | PASS |
| injection_malformed_retrieval | PASS | PASS |
| injection_timeout | PASS | PASS |
| irrelevant_topic_quantum | PASS | PASS |
| kb_compare_lists_vs_tuples | FAIL (hard_failure) | FAIL (hard_failure) |
| kb_fact_list_comprehension | PASS | PASS |
| kb_fact_tuple | PASS | PASS |
| kb_then_calculator | PASS | PASS |
| unanswerable_mars_boiling_point | PASS | PASS |
| weather_kathmandu | PASS | PASS |

