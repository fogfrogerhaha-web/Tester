"""travel helper 后端服务：FastAPI + 高德 MCP + DeepSeek

接口：
- POST   /api/session        新建会话
- GET    /api/session/{id}   取回历史
- DELETE /api/session/{id}   结束会话并清理历史文件
- POST   /api/chat           SSE 流式对话（tool/token/done/error 事件）

运行：python server.py → http://127.0.0.1:8000
"""
import asyncio
import json
import os
import re
import secrets

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from sse_starlette.sse import EventSourceResponse

from travel_helper import (
    DEEPSEEK_BASE_URL,
    AmapMcpClient,
    OpenAI,
    TripRequest,
    build_trip_query,
    load_api_key,
    run_agent_phase,
    run_final_phase_stream,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SESSION_ID_RE = re.compile(r"[0-9a-f-]+")


class SessionStore:
    def __init__(self, directory):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

    def _path(self, session_id):
        if not SESSION_ID_RE.fullmatch(session_id):
            raise HTTPException(status_code=404, detail="会话不存在")
        return os.path.join(self.directory, f"{session_id}.json")

    def create(self):
        session_id = secrets.token_hex(8)
        self.save(session_id, [])
        return session_id

    def exists(self, session_id):
        try:
            return os.path.isfile(self._path(session_id))
        except HTTPException:
            return False

    def load(self, session_id):
        try:
            path = self._path(session_id)
        except HTTPException:
            return []
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except (OSError, json.JSONDecodeError):
            return []

    def save(self, session_id, history):
        with open(self._path(session_id), "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    def delete(self, session_id):
        path = self._path(session_id)
        if os.path.isfile(path):
            os.unlink(path)
            return True
        return False


class TripRequestModel(BaseModel):
    days: int = Field(ge=1)
    destination: str = Field(min_length=1)
    date: str | None = None
    people: str | None = None
    budget: str | None = None
    attractions: list[str] | None = None


class ChatRequest(BaseModel):
    session_id: str
    trip_request: TripRequestModel | None = None
    query: str | None = None

    @model_validator(mode="after")
    def check_payload(self):
        if not self.trip_request and not (self.query and self.query.strip()):
            raise ValueError("trip_request 与 query 必须提供其一")
        return self


_llm = None


def get_llm():
    global _llm
    if _llm is None:
        key = load_api_key()
        if not key:
            raise RuntimeError("未找到 API Key（DEEPSEEK_API_KEY / LLM_API_KEY 或 .env）")
        _llm = OpenAI(api_key=key, base_url=DEEPSEEK_BASE_URL)
    return _llm


def create_app(sessions_dir=None):
    store = SessionStore(sessions_dir or os.path.join(BASE_DIR, "sessions"))
    app = FastAPI(title="travel helper")
    app.state.store = store
    app.state.mcp = AmapMcpClient()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/api/session")
    async def create_session():
        return {"session_id": store.create()}

    @app.get("/api/session/{session_id}")
    async def get_session(session_id: str):
        if not store.exists(session_id):
            raise HTTPException(status_code=404, detail="会话不存在")
        return {"session_id": session_id, "history": store.load(session_id)}

    @app.delete("/api/session/{session_id}")
    async def delete_session(session_id: str):
        if not store.delete(session_id):
            raise HTTPException(status_code=404, detail="会话不存在")
        return {"ok": True}

    @app.post("/api/chat")
    async def chat(body: ChatRequest):
        if not store.exists(body.session_id):
            raise HTTPException(status_code=404, detail="会话不存在")
        try:
            llm = get_llm()
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))

        if body.trip_request:
            t = body.trip_request
            query = build_trip_query(
                TripRequest(
                    days=t.days,
                    destination=t.destination,
                    people=t.people,
                    budget=t.budget,
                    attractions=t.attractions or [],
                    date=t.date,
                )
            )
        else:
            query = body.query.strip()

        history = store.load(body.session_id)
        mcp = app.state.mcp

        async def event_stream():
            queue: asyncio.Queue = asyncio.Queue()

            def on_tool(name, args):
                queue.put_nowait(("tool", {"name": name, "args": args}))

            async def agent_worker():
                try:
                    result = await run_agent_phase(
                        mcp, llm, query, history, on_tool=on_tool
                    )
                    queue.put_nowait(("agent_done", result))
                except Exception as e:
                    queue.put_nowait(("agent_error", e))

            task = asyncio.create_task(agent_worker())
            try:
                while True:
                    kind, payload = await queue.get()
                    if kind == "tool":
                        yield {
                            "event": "tool",
                            "data": json.dumps(payload, ensure_ascii=False),
                        }
                    elif kind == "agent_error":
                        yield {
                            "event": "error",
                            "data": json.dumps(
                                {"message": f"检索阶段出错: {payload}"},
                                ensure_ascii=False,
                            ),
                        }
                        return
                    else:
                        agent_text, _tool_log = payload
                        break

                answer = ""
                async for token in run_final_phase_stream(llm, query, history, agent_text):
                    answer += token
                    yield {
                        "event": "token",
                        "data": json.dumps({"text": token}, ensure_ascii=False),
                    }

                history.append({"role": "user", "content": query})
                history.append({"role": "assistant", "content": answer})
                store.save(body.session_id, history)
                yield {
                    "event": "done",
                    "data": json.dumps(
                        {"session_id": body.session_id, "role": "assistant"},
                        ensure_ascii=False,
                    ),
                }
            except Exception as e:
                yield {
                    "event": "error",
                    "data": json.dumps(
                        {"message": f"生成阶段出错: {e}"}, ensure_ascii=False
                    ),
                }
            finally:
                if not task.done():
                    task.cancel()

        return EventSourceResponse(event_stream())

    static_dir = os.path.join(BASE_DIR, "static")
    if os.path.isdir(static_dir):
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
