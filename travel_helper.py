"""travel helper —— 复现 Dify travel helper 工作流

流程：用户输入 → Agent 检索（DeepSeek function calling + 高德地图 MCP）
→ 旅游助手生成路线推荐（原工作流 system prompt）→ 交互式多轮对话
"""
import asyncio
import json
import os
import re
import sys
from collections import namedtuple

from mcp import ClientSession
from mcp.client.sse import sse_client
from openai import OpenAI

AMAP_MCP_URL = "https://mcp.api-inference.modelscope.net/6cc76c760dc742/sse"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"
MAX_ITERATIONS = 15
TOOL_RESULT_LIMIT = 4000

AGENT_QUERY_SUFFIX = "根据用户给出的信息，调用工具搜索需要的内容"

AGENT_SYSTEM_PROMPT = """你是旅游信息检索助手，可以调用高德地图 MCP 工具集，工具包括：
- maps_geo / maps_regeocode：地址与经纬度坐标互转
- maps_text_search / maps_around_search / maps_search_detail：POI 关键词搜索、周边搜索与详情查询
- maps_weather：查询指定城市天气
- maps_direction_driving / maps_direction_transit_integrated / maps_direction_walking / maps_bicycling：驾车/公交/步行/骑行路线规划
- maps_distance：两地距离测量
- maps_ip_location：IP 定位

工作要求：
1. 根据用户需求，先用 maps_geo 把地名转为经纬度，再用 maps_text_search / maps_around_search 搜索景点、餐厅、酒店等 POI，必要时用 maps_search_detail 获取详情。
2. 安排行程顺序时，用 maps_distance 或路线规划工具核实各地点间的实际距离与耗时。
3. 出行日期明确时，用 maps_weather 查询目的地天气。
4. 工具参数不足时合理推断（如城市名、adcode），不要向用户追问。
5. 完成检索后，用中文条理清晰地总结所有搜索到的关键信息（景点名称与位置、距离耗时、天气、开放信息等），供下游行程规划使用。不要编造未搜索到的信息。"""

SYSTEM_PROMPT_TRAVEL = """```xml
<instruction>
你是一名专业的旅游助手。请根据提供的用户对话上下文和搜索到的相关信息，为用户规划一份合理的旅游路线推荐。

在执行任务时，请严格遵守以下步骤：
1. 分析 <user_context> 以准确理解用户的旅行偏好、时间预算、出行人数及特殊需求。
2. 结合 <search_information> 中的景点信息、交通状况、开放时间及实时数据，筛选出最匹配的目的地和活动。
3. 规划行程时务必确保逻辑合理，避免路线折返或时间安排过紧，充分考虑地理距离与游玩体验的平衡。
4. 使用亲切、热情且专业的语气撰写回复，让用户感受到贴心的服务。
5. 输出内容必须为纯文本格式，严禁包含任何XML标签或其他标记语言符号。
6. 若搜索结果与用户需求存在冲突，请优先基于搜索结果给出调整建议并说明原因。
</instruction>

<example>
用户上下文：我和父母打算五月初去杭州玩三天，他们腿脚不太方便，不想太累，喜欢自然风光和茶文化。
搜索信息：西湖景区无障碍设施完善；龙井村有平缓步道适合老年人；灵隐寺台阶较多但可乘坐电瓶车至半山；五月杭州气温适宜但早晚温差大。
推荐路线：
亲爱的朋友，您好！很高兴为您和家人规划杭州之旅～考虑到叔叔阿姨的舒适度，我为您精心安排了一条轻松惬意的三日游路线：
第一天：上午漫步苏堤春晓（全程平坦无台阶），午后乘船游湖赏三潭印月，傍晚在湖滨银泰享用清淡杭帮菜。
第二天：前往龙井村体验采茶乐趣，沿茶园缓坡散步约1小时，中午品尝地道农家茶宴，下午回酒店休憩。
第三天：乘电瓶车游览灵隐寺外围园林，避开陡峭台阶区域，午后返程前可在河坊街选购伴手礼。
温馨提示：五月杭州早晚微凉，记得为长辈备一件薄外套；各景点间已预留充足休息时间，轮椅也可顺畅通行哦～祝您全家旅途愉快！
</example>
```"""


def parse_env_file(path):
    result = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                result[key.strip()] = value.strip().strip('"').strip("'")
    except OSError:
        return {}
    return result


def load_api_key():
    for name in ("DEEPSEEK_API_KEY", "LLM_API_KEY"):
        key = os.environ.get(name)
        if key:
            return key
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    env = parse_env_file(env_path)
    return env.get("DEEPSEEK_API_KEY") or env.get("LLM_API_KEY")


def mcp_tool_to_openai_schema(tool):
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or tool.name,
            "parameters": tool.input_schema,
        },
    }


def truncate_text(text, limit=TOOL_RESULT_LIMIT):
    if len(text) <= limit:
        return text
    return text[:limit] + f"…（已截断，原文共{len(text)}字符）"


def extract_tool_result_text(result):
    parts = [c.text for c in result.content if getattr(c, "text", None)]
    text = "\n".join(parts)
    if getattr(result, "isError", False):
        return f"工具返回错误: {text or '（空结果）'}"
    return text or "（工具返回空结果）"


def build_agent_messages(query, history):
    context = ""
    if history:
        lines = []
        for h in history[-6:]:
            who = "用户" if h["role"] == "user" else "助手"
            lines.append(f"{who}：{h['content'][:200]}")
        context = "对话历史（供参考）：\n" + "\n".join(lines) + "\n\n"
    return [
        {"role": "system", "content": AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": f"{context}{query}{AGENT_QUERY_SUFFIX}"},
    ]


def build_final_messages(query, history, agent_text):
    messages = [{"role": "system", "content": SYSTEM_PROMPT_TRAVEL}]
    messages.extend({"role": h["role"], "content": h["content"]} for h in history)
    if history:
        lines = ["之前的对话："]
        for h in history[-6:]:
            who = "用户" if h["role"] == "user" else "助手"
            lines.append(f"{who}：{h['content'][:300]}")
        lines.append(f"当前需求：{query}")
        user_context = "\n".join(lines)
    else:
        user_context = query
    user_content = (
        "<input>\n"
        f"<user_context>\n{user_context}\n</user_context>\n\n"
        f"<search_information>\n{agent_text}\n</search_information>\n"
        "</input>"
    )
    messages.append({"role": "user", "content": user_content})
    return messages


TripRequest = namedtuple(
    "TripRequest",
    ["days", "destination", "people", "budget", "attractions", "date"],
    defaults=(None, None, None, None),
)


def parse_days(value):
    try:
        days = int(str(value).strip())
    except ValueError:
        return None
    return days if days >= 1 else None


def parse_people(value):
    value = (value or "").strip()
    return value or None


def parse_budget(value):
    value = (value or "").strip()
    return value or None


def parse_attractions(value):
    value = (value or "").strip()
    if not value:
        return []
    return [p.strip() for p in re.split(r"[，,、;；/]+", value) if p.strip()]


def collect_trip_request(input_fn=input, print_fn=print):
    print_fn("—— 请填写旅行需求（必填项留空或无效会重新询问）——")
    destination = ""
    while not destination:
        destination = input_fn("目的地（必填，如：杭州）：").strip()
    days = None
    while days is None:
        days = parse_days(input_fn("天数（必填，如：3）："))
        if days is None:
            print_fn("天数需为正整数，请重新输入。")
    date = input_fn("出行日期（选填，如：十月初，回车跳过）：").strip() or None
    people = parse_people(input_fn("人数（选填，如：3 或 2大1小，回车跳过）："))
    budget = parse_budget(input_fn("预算（选填，如：5000元，回车跳过）："))
    attractions = parse_attractions(
        input_fn("想去的景点（选填，逗号分隔，如：西湖、灵隐寺，回车跳过）：")
    )
    return TripRequest(
        days=days,
        destination=destination,
        people=people,
        budget=budget,
        attractions=attractions,
        date=date,
    )


def build_trip_query(req):
    parts = [f"计划去{req.destination}旅游{req.days}天"]
    if req.date:
        parts.append(f"出行日期为{req.date}")
    if req.people:
        parts.append(f"出行人数为{req.people}")
    if req.budget:
        parts.append(f"预算{req.budget}")
    if req.attractions:
        parts.append(f"希望游览{'、'.join(req.attractions)}")
    return "，".join(parts) + "，请为我规划一份合理的旅游路线"


class AmapMcpClient:
    def __init__(self, url=AMAP_MCP_URL):
        self.url = url
        self._sse_cm = None
        self._session_cm = None
        self._session = None

    async def _ensure_session(self):
        if self._session is None:
            self._sse_cm = sse_client(url=self.url)
            read, write = await self._sse_cm.__aenter__()
            self._session_cm = ClientSession(read, write)
            self._session = await self._session_cm.__aenter__()
            await self._session.initialize()
        return self._session

    async def reset(self):
        if self._session_cm is not None:
            try:
                await self._session_cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._session_cm = None
            self._session = None
        if self._sse_cm is not None:
            try:
                await self._sse_cm.__aexit__(None, None, None)
            except Exception:
                pass
            self._sse_cm = None

    async def close(self):
        await self.reset()

    async def list_tools(self):
        async def _do():
            session = await self._ensure_session()
            return (await session.list_tools()).tools

        try:
            return await _do()
        except Exception:
            await self.reset()
            return await _do()

    async def call_tool(self, name, arguments):
        async def _do():
            session = await self._ensure_session()
            return extract_tool_result_text(await session.call_tool(name, arguments))

        try:
            return await _do()
        except Exception as e:
            await self.reset()
            try:
                return await _do()
            except Exception as e2:
                return f"工具调用失败: {e2}"


async def run_agent_phase(
    client, llm, query, history, max_iterations=MAX_ITERATIONS, on_tool=None
):
    mcp_tools = await client.list_tools()
    tools = [mcp_tool_to_openai_schema(t) for t in mcp_tools]
    messages = build_agent_messages(query, history)
    tool_log = []

    for _ in range(max_iterations):
        resp = llm.chat.completions.create(
            model=DEEPSEEK_MODEL, messages=messages, tools=tools
        )
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return msg.content or "", tool_log

        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ],
        })
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            result_text = truncate_text(await client.call_tool(tc.function.name, args))
            tool_log.append(f"{tc.function.name}({json.dumps(args, ensure_ascii=False)})")
            if on_tool:
                try:
                    on_tool(tc.function.name, args)
                except Exception:
                    pass
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result_text})

    resp = llm.chat.completions.create(model=DEEPSEEK_MODEL, messages=messages)
    return resp.choices[0].message.content or "", tool_log


def run_final_phase(llm, query, history, agent_text):
    messages = build_final_messages(query, history, agent_text)
    resp = llm.chat.completions.create(model=DEEPSEEK_MODEL, messages=messages)
    return resp.choices[0].message.content or ""


async def run_final_phase_stream(llm, query, history, agent_text):
    messages = build_final_messages(query, history, agent_text)
    stream = llm.chat.completions.create(
        model=DEEPSEEK_MODEL, messages=messages, stream=True
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


async def chat_loop_async(api_key):
    llm = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    client = AmapMcpClient()
    history = []
    print("=== travel helper（高德地图 MCP 版）===")
    print("首次输入为结构化旅行需求：目的地、天数为必填；出行日期、人数、预算、想去的景点为选填。")
    print("生成路线后可继续追问调整（如：第二天换成博物馆），输入 退出/exit 结束。\n")

    first_round = True
    while True:
        if first_round:
            try:
                request = await asyncio.to_thread(collect_trip_request)
            except (EOFError, KeyboardInterrupt):
                break
            query = build_trip_query(request)
            print(f"\n你的需求：{query}")
            first_round = False
        else:
            try:
                query = (await asyncio.to_thread(input, "\n你（追问/调整）> ")).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not query:
                continue
            if query.lower() in {"exit", "quit", "退出"}:
                break

        print("\n[正在调用高德地图工具检索信息……]")
        try:
            agent_text, tool_log = await run_agent_phase(client, llm, query, history)
        except Exception as e:
            await client.reset()
            print(f"[检索阶段出错，已重置 MCP 连接] {e}\n请重试或换个问法。\n")
            continue
        for entry in tool_log:
            print(f"  [工具] {entry}")

        try:
            answer = run_final_phase(llm, query, history, agent_text)
        except Exception as e:
            print(f"[生成阶段出错] {e}\n")
            continue
        print(f"\n{answer}\n")
        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": answer})

    await client.close()
    print("再见，祝旅途愉快！")


def main():
    api_key = load_api_key()
    if not api_key:
        print(
            "未找到 API Key。请设置环境变量 DEEPSEEK_API_KEY 或 LLM_API_KEY，"
            "或在脚本同目录创建 .env 文件：\nLLM_API_KEY=sk-xxxxxxxx"
        )
        sys.exit(1)
    asyncio.run(chat_loop_async(api_key))


if __name__ == "__main__":
    main()
