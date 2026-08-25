# Implementation Plan: travel helper（高德 MCP 复现）

## Overview
用 Python 复现 Dify travel helper 工作流：交互式聊天程序。每轮对话分两阶段——
1. Agent 检索：DeepSeek 原生 function calling 循环调用高德地图 MCP（12 个工具），上限 15 次
2. 路线生成：按原 YAML 的"专业旅游助手"system prompt，注入 user_context 与 search_information

## Prerequisites
- [x] 设计已确认（方案 A：原生 function calling + DeepSeek + 交互式聊天）
- [x] venv 已有 mcp 2.0.0、openai 3.3.1，零新增依赖
- [x] 高德 MCP SSE 端点连通验证通过（12 个工具）
- [ ] DEEPSEEK_API_KEY（用户提供，环境变量或 .env）

## Implementation Steps

### Step 1: 单元测试（红灯）
**Complexity:** Low
用 stdlib unittest（不装 pytest）。覆盖：
- `parse_env_file`：解析 .env 为 dict
- `mcp_tool_to_openai_schema`：MCP Tool → OpenAI function schema
- `build_agent_messages`：拼装 Agent 阶段消息（含 query 后缀）
- `build_final_messages`：拼装生成阶段消息（原 YAML system prompt + user_context + search_information）
- `truncate_text`：工具结果截断

**Verification:** `python -m unittest` 全部失败（模块不存在）

### Step 2: 实现 travel_helper.py（绿灯）
**Complexity:** Medium
结构：
- `AMAP_MCP_URL`、`SYSTEM_PROMPT_TRAVEL`（原 YAML 提示词，含示例）、`AGENT_SYSTEM_PROMPT`
- `AmapMcpClient`：懒连接 + 断线重连 + `call_tool(name, args)` 返回文本
- `run_agent_phase(client, query, history)`：DeepSeek tool_calls 循环，≤15 轮，返回 (agent_text, tool_log)
- `run_final_phase(query, history, agent_text)`：生成路线推荐
- `chat_loop()`：交互循环，维护 (user, assistant) 历史

**Verification:** 单测全绿；`python travel_helper.py` 进入交互

### Step 3: 端到端验证
**Complexity:** Low
- 用户提供 key 后跑真实对话（示例：杭州三日游）
- 验证：确实调用了高德工具（打印工具调用日志）、输出符合原提示词风格（纯文本、亲切语气、按天分行程）

## Checkpoints
- Step 1 后：测试存在且失败
- Step 2 后：测试全绿
- Step 3 后：真实对话输出合理路线

## Risks
- DeepSeek tool_calls 参数为 JSON 字符串，解析失败需容错 → try/except 后把错误回给 LLM
- MCP 会话超时/掉线 → 每轮 Agent 阶段前重置连接 + 失败重连一次
- 工具返回大 JSON 撑爆上下文 → 截断到 4000 字符
