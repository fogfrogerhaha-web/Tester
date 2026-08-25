"""测试高德地图 MCP SSE 端点连通性，列出可用工具"""
import asyncio
from mcp import ClientSession
from mcp.client.sse import sse_client

AMAP_MCP_URL = "https://mcp.api-inference.modelscope.net/6cc76c760dc742/sse"


async def main():
    async with sse_client(url=AMAP_MCP_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print(f"共 {len(tools.tools)} 个工具:")
            for t in tools.tools:
                print(f"  - {t.name}: {(t.description or '').strip()[:80]}")


if __name__ == "__main__":
    asyncio.run(main())
