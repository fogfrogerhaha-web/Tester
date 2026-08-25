"""travel_helper 单元测试（stdlib unittest，零额外依赖）"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.types import CallToolResult, TextContent, Tool

from travel_helper import (
    AGENT_QUERY_SUFFIX,
    SYSTEM_PROMPT_TRAVEL,
    TripRequest,
    build_agent_messages,
    build_final_messages,
    build_trip_query,
    collect_trip_request,
    extract_tool_result_text,
    mcp_tool_to_openai_schema,
    parse_attractions,
    parse_budget,
    parse_days,
    parse_env_file,
    parse_people,
    truncate_text,
)


class TestParseEnvFile(unittest.TestCase):
    def test_parses_key_value_pairs(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".env", delete=False, encoding="utf-8"
        ) as f:
            f.write("DEEPSEEK_API_KEY=sk-test-123\n")
            f.write("# 注释行应被忽略\n")
            f.write("OTHER_KEY = 带空格的值\n")
            path = f.name
        try:
            result = parse_env_file(path)
        finally:
            os.unlink(path)
        self.assertEqual(result["DEEPSEEK_API_KEY"], "sk-test-123")
        self.assertEqual(result["OTHER_KEY"], "带空格的值")
        self.assertNotIn("# 注释行应被忽略", result)

    def test_missing_file_returns_empty_dict(self):
        self.assertEqual(parse_env_file(r"Z:\no\such\file.env"), {})


class TestMcpToolToOpenaiSchema(unittest.TestCase):
    def test_converts_tool(self):
        tool = Tool(
            name="maps_geo",
            description="将详细的结构化地址转换为经纬度坐标",
            inputSchema={
                "type": "object",
                "properties": {"address": {"type": "string"}},
                "required": ["address"],
            },
        )
        schema = mcp_tool_to_openai_schema(tool)
        self.assertEqual(schema["type"], "function")
        self.assertEqual(schema["function"]["name"], "maps_geo")
        self.assertEqual(schema["function"]["description"], "将详细的结构化地址转换为经纬度坐标")
        self.assertEqual(schema["function"]["parameters"], tool.input_schema)


class TestTruncateText(unittest.TestCase):
    def test_short_text_unchanged(self):
        self.assertEqual(truncate_text("短文本"), "短文本")

    def test_long_text_truncated_with_marker(self):
        text = "x" * 5000
        result = truncate_text(text, limit=1000)
        self.assertLessEqual(len(result), 1100)
        self.assertIn("…", result)
        self.assertTrue(result.startswith("x"))


class TestBuildAgentMessages(unittest.TestCase):
    def test_first_turn_structure(self):
        messages = build_agent_messages("我想去杭州玩三天", [])
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("工具", messages[0]["content"])
        self.assertEqual(messages[1]["role"], "user")
        self.assertIn("我想去杭州玩三天", messages[1]["content"])
        self.assertTrue(messages[1]["content"].endswith(AGENT_QUERY_SUFFIX))


class TestBuildFinalMessages(unittest.TestCase):
    def test_structure_with_history(self):
        history = [
            {"role": "user", "content": "帮我规划杭州三日游"},
            {"role": "assistant", "content": "第一天游览西湖……"},
        ]
        messages = build_final_messages(
            "第二天改成博物馆", history, "搜索到：浙江博物馆位于西湖孤山……"
        )
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn("你是一名专业的旅游助手", messages[0]["content"])
        self.assertIn("严禁包含任何XML标签", messages[0]["content"])
        # 历史消息应被保留
        self.assertEqual(messages[1]["content"], "帮我规划杭州三日游")
        self.assertEqual(messages[2]["content"], "第一天游览西湖……")
        # 最后一条 user 消息包含上下文标签与搜索信息
        last = messages[-1]
        self.assertEqual(last["role"], "user")
        self.assertIn("<user_context>", last["content"])
        self.assertIn("第二天改成博物馆", last["content"])
        self.assertIn("<search_information>", last["content"])
        self.assertIn("浙江博物馆位于西湖孤山", last["content"])

    def test_first_turn_no_history(self):
        messages = build_final_messages("成都两日游", [], "搜索到：宽窄巷子……")
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]["role"], "system")
        self.assertEqual(messages[-1]["role"], "user")


class TestExtractToolResultText(unittest.TestCase):
    def test_extracts_text_content(self):
        result = CallToolResult(
            content=[TextContent(type="text", text="第一段"), TextContent(type="text", text="第二段")]
        )
        self.assertEqual(extract_tool_result_text(result), "第一段\n第二段")

    def test_empty_content_returns_placeholder(self):
        result = CallToolResult(content=[])
        self.assertIn("空", extract_tool_result_text(result))


class TestPromptConstants(unittest.TestCase):
    def test_travel_prompt_keeps_example(self):
        self.assertIn("苏堤春晓", SYSTEM_PROMPT_TRAVEL)
        self.assertIn("龙井村", SYSTEM_PROMPT_TRAVEL)


class TestTripRequest(unittest.TestCase):
    def test_required_only_and_tuple(self):
        req = TripRequest(3, "杭州")
        self.assertIsInstance(req, tuple)
        self.assertEqual((req.days, req.destination), (3, "杭州"))
        self.assertIsNone(req.people)
        self.assertIsNone(req.budget)
        self.assertIsNone(req.attractions)
        self.assertIsNone(req.date)


class TestParseHelpers(unittest.TestCase):
    def test_parse_days(self):
        self.assertEqual(parse_days("3"), 3)
        self.assertEqual(parse_days(" 5 "), 5)
        self.assertIsNone(parse_days("abc"))
        self.assertIsNone(parse_days("0"))
        self.assertIsNone(parse_days("-2"))
        self.assertIsNone(parse_days(""))
        self.assertIsNone(parse_days("3.5"))

    def test_parse_attractions(self):
        self.assertEqual(
            parse_attractions("西湖，灵隐寺、雷峰塔"), ["西湖", "灵隐寺", "雷峰塔"]
        )
        self.assertEqual(parse_attractions(" 西湖 "), ["西湖"])
        self.assertEqual(parse_attractions(""), [])
        self.assertEqual(parse_attractions("   "), [])

    def test_parse_people_budget(self):
        self.assertEqual(parse_people(" 3 "), "3")
        self.assertEqual(parse_people("2大1小"), "2大1小")
        self.assertIsNone(parse_people(""))
        self.assertEqual(parse_budget(" 5000元 "), "5000元")
        self.assertIsNone(parse_budget(""))


class TestBuildTripQuery(unittest.TestCase):
    def test_minimal(self):
        q = build_trip_query(TripRequest(3, "杭州"))
        self.assertIn("杭州", q)
        self.assertIn("3天", q)
        self.assertNotIn("预算", q)
        self.assertNotIn("人数", q)
        self.assertNotIn("游览", q)
        self.assertNotIn("日期", q)

    def test_full(self):
        req = TripRequest(
            days=3,
            destination="杭州",
            people="2大1小",
            budget="5000元",
            attractions=["西湖", "灵隐寺"],
            date="十月初",
        )
        q = build_trip_query(req)
        self.assertIn("出行日期为十月初", q)
        self.assertIn("出行人数为2大1小", q)
        self.assertIn("预算5000元", q)
        self.assertIn("西湖、灵隐寺", q)


class TestCollectTripRequest(unittest.TestCase):
    @staticmethod
    def _fake_input(responses):
        it = iter(responses)
        return lambda prompt="": next(it)

    def test_collect_with_reprompts(self):
        responses = ["", "杭州", "abc", "0", "3", "十一假期", "2大1小", "5000元", "西湖、灵隐寺"]
        req = collect_trip_request(
            input_fn=self._fake_input(responses), print_fn=lambda *a, **k: None
        )
        self.assertEqual(req.destination, "杭州")
        self.assertEqual(req.days, 3)
        self.assertEqual(req.date, "十一假期")
        self.assertEqual(req.people, "2大1小")
        self.assertEqual(req.budget, "5000元")
        self.assertEqual(req.attractions, ["西湖", "灵隐寺"])

    def test_collect_optionals_skipped(self):
        responses = ["成都", "2", "", "", "", ""]
        req = collect_trip_request(
            input_fn=self._fake_input(responses), print_fn=lambda *a, **k: None
        )
        self.assertEqual((req.days, req.destination), (2, "成都"))
        self.assertIsNone(req.date)
        self.assertIsNone(req.people)
        self.assertIsNone(req.budget)
        self.assertEqual(req.attractions, [])


if __name__ == "__main__":
    unittest.main()
