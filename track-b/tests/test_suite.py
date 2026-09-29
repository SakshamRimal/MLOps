import unittest
import time
import json
from unittest.mock import patch, AsyncMock, MagicMock
from starlette.testclient import TestClient

from app.cache import ResponseCache
from app.circuit_breaker import CircuitBreaker, CircuitState
from app.tools import execute_tool, calculator, get_current_time, web_search, TOOL_DEFINITIONS
from app.schemas import ChatRequest, ChatMessage, AssistantAnswer, ChatResponse
from app.main import app


class TestResponseCache(unittest.TestCase):
    def setUp(self):
        self.cache = ResponseCache(max_size=3, ttl_seconds=1)

    def test_set_and_get(self):
        msgs = [{"role": "user", "content": "hello"}]
        self.cache.set(msgs, "model-a", "reply-a")
        res = self.cache.get(msgs, "model-a")
        self.assertEqual(res, "reply-a")
        self.assertEqual(self.cache.stats["hits"], 1)

    def test_miss_and_stats(self):
        msgs = [{"role": "user", "content": "not found"}]
        res = self.cache.get(msgs, "model-a")
        self.assertIsNone(res)
        self.assertEqual(self.cache.stats["misses"], 1)

    def test_lru_eviction(self):
        for i in range(4):
            self.cache.set([{"role": "user", "content": f"msg-{i}"}], "model", f"reply-{i}")
        self.assertEqual(self.cache.stats["size"], 3)
        # First item should be evicted
        res = self.cache.get([{"role": "user", "content": "msg-0"}], "model")
        self.assertIsNone(res)
        # Latest items should be present
        res = self.cache.get([{"role": "user", "content": "msg-3"}], "model")
        self.assertEqual(res, "reply-3")

    def test_ttl_expiration(self):
        msgs = [{"role": "user", "content": "expire"}]
        self.cache.set(msgs, "model", "quick", ttl_seconds=0.1)
        time.sleep(0.15)
        res = self.cache.get(msgs, "model")
        self.assertIsNone(res)

    def test_invalidate_all(self):
        self.cache.set([{"role": "user", "content": "a"}], "model", "ans")
        count = self.cache.invalidate_all()
        self.assertEqual(count, 1)
        self.assertEqual(self.cache.stats["size"], 0)


class TestCircuitBreaker(unittest.TestCase):
    def setUp(self):
        self.cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.2)

    def test_initial_state_closed(self):
        self.assertEqual(self.cb.state, CircuitState.CLOSED)
        self.assertTrue(self.cb.allow_request())

    def test_trip_to_open_after_threshold(self):
        self.cb.record_failure()
        self.cb.record_failure()
        self.assertEqual(self.cb.state, CircuitState.CLOSED)
        self.assertTrue(self.cb.allow_request())

        self.cb.record_failure()  # 3rd failure
        self.assertEqual(self.cb.state, CircuitState.OPEN)
        self.assertFalse(self.cb.allow_request())

    def test_half_open_recovery(self):
        for _ in range(3):
            self.cb.record_failure()
        self.assertEqual(self.cb.state, CircuitState.OPEN)

        # Wait for recovery timeout
        time.sleep(0.25)
        self.assertTrue(self.cb.allow_request())
        self.assertEqual(self.cb.state, CircuitState.HALF_OPEN)

        # Success in half open closes circuit
        self.cb.record_success()
        self.assertEqual(self.cb.state, CircuitState.CLOSED)
        self.assertEqual(self.cb.failure_count, 0)

    def test_half_open_failure_reopens(self):
        for _ in range(3):
            self.cb.record_failure()
        time.sleep(0.25)
        self.cb.allow_request()
        self.assertEqual(self.cb.state, CircuitState.HALF_OPEN)

        # Probe failure reopens circuit
        self.cb.record_failure()
        self.assertEqual(self.cb.state, CircuitState.OPEN)


class TestTools(unittest.TestCase):
    def test_calculator_valid(self):
        res = calculator("sqrt(144) + pow(2, 3)")
        self.assertEqual(res, {"result": 20.0})

    def test_calculator_error(self):
        res = calculator("1 / 0")
        self.assertIn("error", res)

    def test_get_current_time(self):
        res = get_current_time()
        self.assertIn("iso", res)
        self.assertIn("utc_time", res)

    def test_web_search(self):
        res = web_search("python 3.12")
        self.assertIn("results", res)
        self.assertTrue(len(res["results"]) > 0)

    def test_execute_tool_dispatch(self):
        out = execute_tool("calculator", '{"expression": "10 * 5"}')
        data = json.loads(out)
        self.assertEqual(data["result"], 50)

    def test_execute_tool_missing(self):
        out = execute_tool("non_existent_tool", "{}")
        data = json.loads(out)
        self.assertIn("error", data)

    def test_tool_definitions_complete(self):
        names = [t["function"]["name"] for t in TOOL_DEFINITIONS]
        self.assertIn("calculator", names)
        self.assertIn("query_knowledge_base", names)
        self.assertIn("web_search", names)
        self.assertIn("weather", names)
        self.assertIn("get_current_time", names)


class TestSchemas(unittest.TestCase):
    def test_chat_request_normalized_history(self):
        req = ChatRequest(
            message="hi",
            history=[
                ChatMessage(role="user", content="msg1"),
                {"role": "assistant", "content": "reply1"},
            ],
            temperature=0.5,
            top_p=0.9,
            max_tokens=256,
        )
        history = req.get_normalized_history()
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0], {"role": "user", "content": "msg1"})
        self.assertEqual(history[1], {"role": "assistant", "content": "reply1"})

    def test_assistant_answer_validation(self):
        ans = AssistantAnswer(
            answer="Test answer",
            sources=[{"document": "test.pdf", "chunk_id": "c1"}],
            confidence=0.95,
        )
        self.assertEqual(ans.confidence, 0.95)
        self.assertEqual(len(ans.sources), 1)

    def test_assistant_answer_confidence_bounds(self):
        with self.assertRaises(Exception):
            AssistantAnswer(answer="Bad", sources=[], confidence=1.5)


class TestFastAPIEndpoints(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_health_endpoint(self):
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ok")
        self.assertIn("circuit_breaker", data)
        self.assertIn("cache", data)
        # Check tracking headers added by middleware
        self.assertIn("x-request-id", resp.headers)
        self.assertIn("x-response-time", resp.headers)

    def test_cache_stats_and_invalidate(self):
        resp = self.client.get("/cache/stats")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("size", resp.json())

        resp = self.client.post("/cache/invalidate")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("invalidated", resp.json())

    def test_rag_documents_endpoint(self):
        resp = self.client.get("/rag/documents")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("total_chunks", data)
        self.assertIn("documents", data)

    @patch("app.llm_client.llm_client.chat", new_callable=AsyncMock)
    def test_chat_endpoint(self, mock_chat):
        mock_chat.return_value = "Paris is the capital of France."
        resp = self.client.post(
            "/chat",
            json={"message": "What is France's capital?", "temperature": 0.2},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["answer"], "Paris is the capital of France.")
        self.assertEqual(data["confidence"], 1.0)

    @patch("app.llm_client.llm_client.chat_with_tools", new_callable=AsyncMock)
    def test_chat_tools_endpoint(self, mock_chat_tools):
        mock_chat_tools.return_value = {
            "answer": "The result is 42.",
            "tool_calls_made": ["calculator"],
        }
        resp = self.client.post(
            "/chat/tools",
            json={"message": "Compute 6 * 7"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["answer"], "The result is 42.")
        self.assertEqual(data["tool_calls_made"], ["calculator"])

    @patch("app.llm_client.llm_client.chat_structured", new_callable=AsyncMock)
    def test_chat_structured_endpoint(self, mock_chat_structured):
        mock_chat_structured.return_value = {
            "answer": "Structured response.",
            "sources": [{"document": "guide.pdf", "chunk_id": "1"}],
            "confidence": 0.98,
        }
        resp = self.client.post(
            "/chat/structured",
            json={"message": "Give me structured info"},
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["answer"], "Structured response.")
        self.assertEqual(data["confidence"], 0.98)
        self.assertEqual(len(data["sources"]), 1)


if __name__ == "__main__":
    unittest.main()

