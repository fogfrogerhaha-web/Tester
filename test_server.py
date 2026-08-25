"""server.py 单元测试：会话存储、API 校验、SSE 流式端点（mock LLM 与 MCP）"""
import asyncio
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient

import server
from server import SessionStore, create_app


def parse_sse_events(text):
    events = []
    current_event, data_lines = None, []
    for line in text.splitlines():
        if line.startswith("event:"):
            current_event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())
        elif not line.strip() and current_event is not None:
            payload = "\n".join(data_lines)
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                pass
            events.append((current_event, payload))
            current_event, data_lines = None, []
    return events


class ServerTestBase(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.client = TestClient(create_app(sessions_dir=self.tmpdir))

    def tearDown(self):
        for name in os.listdir(self.tmpdir):
            os.unlink(os.path.join(self.tmpdir, name))
        os.rmdir(self.tmpdir)


class TestSessionStore(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.store = SessionStore(self.tmpdir)

    def tearDown(self):
        for name in os.listdir(self.tmpdir):
            os.unlink(os.path.join(self.tmpdir, name))
        os.rmdir(self.tmpdir)

    def test_create_and_load(self):
        sid = self.store.create()
        self.assertTrue(self.store.exists(sid))
        self.assertEqual(self.store.load(sid), [])
        history = [{"role": "user", "content": "去杭州"}]
        self.store.save(sid, history)
        self.assertEqual(self.store.load(sid), history)

    def test_load_missing_returns_empty(self):
        self.assertEqual(self.store.load("no-such-id"), [])

    def test_load_corrupt_returns_empty(self):
        sid = self.store.create()
        with open(self.store._path(sid), "w", encoding="utf-8") as f:
            f.write("{broken json!!")
        self.assertEqual(self.store.load(sid), [])

    def test_delete(self):
        sid = self.store.create()
        self.assertTrue(self.store.delete(sid))
        self.assertFalse(self.store.exists(sid))
        self.assertFalse(self.store.delete(sid))

    def test_id_is_filename_safe(self):
        sid = self.store.create()
        self.assertTrue(sid.replace("-", "").isalnum())


class TestSessionApi(ServerTestBase):
    def test_create_get_delete_cycle(self):
        resp = self.client.post("/api/session")
        self.assertEqual(resp.status_code, 200)
        sid = resp.json()["session_id"]
        self.assertTrue(sid)

        resp = self.client.get(f"/api/session/{sid}")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["history"], [])

        resp = self.client.delete(f"/api/session/{sid}")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["ok"])

        resp = self.client.get(f"/api/session/{sid}")
        self.assertEqual(resp.status_code, 404)

    def test_get_unknown_session_404(self):
        self.assertEqual(self.client.get("/api/session/deadbeef").status_code, 404)

    def test_delete_unknown_session_404(self):
        self.assertEqual(self.client.delete("/api/session/deadbeef").status_code, 404)


class TestChatValidation(ServerTestBase):
    def test_unknown_session_404(self):
        resp = self.client.post(
            "/api/chat", json={"session_id": "deadbeef", "query": "你好"}
        )
        self.assertEqual(resp.status_code, 404)

    def test_missing_query_and_trip_request_422(self):
        sid = self.client.post("/api/session").json()["session_id"]
        resp = self.client.post("/api/chat", json={"session_id": sid})
        self.assertEqual(resp.status_code, 422)

    def test_invalid_days_422(self):
        sid = self.client.post("/api/session").json()["session_id"]
        resp = self.client.post(
            "/api/chat",
            json={"session_id": sid, "trip_request": {"days": 0, "destination": "杭州"}},
        )
        self.assertEqual(resp.status_code, 422)


class TestChatSse(ServerTestBase):
    def _patch_phases(self, tool_events, tokens, agent_text="搜索到的信息"):
        async def fake_agent_phase(client, llm, query, history, on_tool=None, **kw):
            for name, args in tool_events:
                if on_tool:
                    on_tool(name, args)
            return agent_text, [f"{n}(...)" for n, _ in tool_events]

        async def fake_stream(llm, query, history, agent_text):
            for t in tokens:
                yield t

        server.run_agent_phase = fake_agent_phase
        server.run_final_phase_stream = fake_stream
        server.get_llm = lambda: object()

    def tearDown(self):
        importlib_reload_server()
        super().tearDown()

    def _stream(self, sid, body):
        chunks = []
        with self.client.stream("POST", "/api/chat", json=body) as resp:
            self.assertEqual(resp.status_code, 200)
            for chunk in resp.iter_text():
                chunks.append(chunk)
        return parse_sse_events("".join(chunks))

    def test_trip_request_round(self):
        self._patch_phases(
            [("maps_weather", {"city": "杭州"}), ("maps_geo", {"address": "西湖"})],
            ["第一天", "游西湖"],
        )
        sid = self.client.post("/api/session").json()["session_id"]
        events = self._stream(
            sid,
            {
                "session_id": sid,
                "trip_request": {
                    "days": 3,
                    "destination": "杭州",
                    "people": "2大1小",
                },
            },
        )
        names = [e for e, _ in events]
        self.assertEqual(names.count("tool"), 2)
        tool_payloads = [p for e, p in events if e == "tool"]
        self.assertEqual(tool_payloads[0]["name"], "maps_weather")
        self.assertEqual(tool_payloads[1]["args"], {"address": "西湖"})
        tokens = "".join(p["text"] for e, p in events if e == "token")
        self.assertEqual(tokens, "第一天游西湖")
        self.assertIn("done", names)
        done = [p for e, p in events if e == "done"][0]
        self.assertEqual(done["session_id"], sid)

        history = self.client.get(f"/api/session/{sid}").json()["history"]
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["role"], "user")
        self.assertIn("杭州", history[0]["content"])
        self.assertIn("3天", history[0]["content"])
        self.assertEqual(history[1]["content"], "第一天游西湖")

    def test_followup_query_round(self):
        self._patch_phases([], ["调整好了"])
        sid = self.client.post("/api/session").json()["session_id"]
        first = self._stream(sid, {"session_id": sid, "query": "规划成都两日游"})
        self.assertIn("done", [e for e, _ in first])
        second = self._stream(sid, {"session_id": sid, "query": "第二天改成博物馆"})
        tokens = "".join(p["text"] for e, p in second if e == "token")
        self.assertEqual(tokens, "调整好了")
        history = self.client.get(f"/api/session/{sid}").json()["history"]
        self.assertEqual(len(history), 4)
        self.assertEqual(history[2]["content"], "第二天改成博物馆")

    def test_error_event_leaves_history_unchanged(self):
        async def failing_agent_phase(client, llm, query, history, on_tool=None, **kw):
            raise RuntimeError("mcp down")

        async def fake_stream(llm, query, history, agent_text):
            yield "不该出现"

        server.run_agent_phase = failing_agent_phase
        server.run_final_phase_stream = fake_stream
        server.get_llm = lambda: object()

        sid = self.client.post("/api/session").json()["session_id"]
        events = self._stream(sid, {"session_id": sid, "query": "去杭州"})
        names = [e for e, _ in events]
        self.assertIn("error", names)
        self.assertNotIn("done", names)
        err = [p for e, p in events if e == "error"][0]
        self.assertIn("mcp down", err["message"])
        history = self.client.get(f"/api/session/{sid}").json()["history"]
        self.assertEqual(history, [])


def importlib_reload_server():
    import importlib

    importlib.reload(server)


if __name__ == "__main__":
    unittest.main()
