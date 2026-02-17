# Agent.AI → OpenAI API 代理服务

把 Agent.AI 包装成 OpenAI 兼容的 `/v1/chat/completions` 接口，支持多种后端模式。

## 三种服务模式

| 文件 | 后端 | 认证方式 | 适用场景 |
|---|---|---|---|
| `server.js` | MCP 协议 | OAuth 账号池（自动刷新） | **推荐**，稳定，支持日志 |
| `server.py` | invoke_llm / invoke_agent API | API Key 池 | 多 key 轮换，Toolify 风格 FC |
| `server_llm.py` | invoke_llm API | 单 API Key | 最简版，快速测试 |

## 快速启动

### Node.js 版（server.js，推荐）

```bash
# 安装依赖（无外部依赖，纯 Node.js）
node server.js

# 环境变量
PORT=9090                      # 监听端口
API_KEY=sk-agentai-proxy       # 访问密钥
AGENT_MCP_BASE=https://api.agent.ai/api/v2/agents/<agent_id>  # Agent MCP 地址
ACCOUNTS_FILE=accounts.json    # 账号文件路径
LOG_DIR=logs                   # 日志目录
```

账号通过 `accounts.json` 管理，支持多账号 least-error 轮换：

```json
[
  {
    "name": "account-1",
    "access_token": "eyJ...",
    "refresh_token": "xxx",
    "client_id": "xxx",
    "agent_mcp_base": "https://api.agent.ai/api/v2/agents/<agent_id>",
    "enabled": true
  }
]
```

### Python invoke_llm 版（server.py）

```bash
pip install fastapi uvicorn httpx
python server.py

# 环境变量
PORT=9090
API_KEY=sk-agentai-proxy
DEFAULT_ENGINE=claude-sonnet-4
MAX_RETRIES=3
USE_AGENT=false                # true 则走 invoke_agent 而非 invoke_llm
AGENTAI_AGENT_ID=h252a58ehz9qlcwp  # invoke_agent 模式的 agent ID
```

API Key 通过 `api_keys.json` 管理，支持多 key 轮换。

### Python 精简版（server_llm.py）

```bash
python server_llm.py

# 环境变量
AGENTAI_API_KEY=your_key       # 单个 API Key
DEFAULT_ENGINE=claude-sonnet-4
```

## 使用方式

三种模式对外接口完全一致，跟调 OpenAI 一样：

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:9090/v1",
    api_key="sk-agentai-proxy"
)

# 普通对话
resp = client.chat.completions.create(
    model="claude-sonnet-4",
    messages=[{"role": "user", "content": "你好"}]
)
print(resp.choices[0].message.content)

# 流式输出
for chunk in client.chat.completions.create(
    model="claude-sonnet-4",
    messages=[{"role": "user", "content": "讲个故事"}],
    stream=True
):
    print(chunk.choices[0].delta.content or "", end="")

# Function Calling（工具调用）
resp = client.chat.completions.create(
    model="claude-sonnet-4",
    messages=[{"role": "user", "content": "北京天气"}],
    tools=[{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取天气",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"]
            }
        }
    }]
)
```

## 智能上下文压缩（server.py）

当对话历史过长时，自动压缩上下文避免超出模型 token 限制。

### 工作流程

```
客户端请求 (messages)
    │
    ▼
① Tool 结果按距离递减截断
    │  最近 3 轮：完整保留
    │  3-10 轮前：截断到 500 字符
    │  10 轮以前：截断到 100 字符
    │
    ▼
② 检查 prompt 长度
    │
    ├─ < 256K chars ──→ 正常发送给 Agent.AI
    │
    └─ ≥ 256K chars ──→ 主动压缩 ③
                            │
                            ▼
                    ③ 拆分消息：system + 旧对话 + 最近3轮
                            │
                            ▼
                    ④ 调 DeepSeek 总结旧对话
                       (支持滚动压缩：旧摘要 + 新增对话 → 新摘要)
                            │
                            ▼
                    ⑤ 重建 prompt：system + [摘要] + 最近3轮
                            │
                            ├─ < 256K ──→ 发送
                            └─ ≥ 256K ──→ 强制重新生成摘要再发送
```

### 滚动压缩

不是一次性压缩，而是**增量式**：
1. 首次超阈值 → 调 DeepSeek 生成摘要 A
2. 对话继续增长，再次超阈值 → 把 **摘要 A + 新增对话** 发给 DeepSeek → 生成摘要 B
3. 重复，摘要越来越完整，信息不丢失

### 被动压缩（兜底）

当 Agent.AI 返回空响应时：
1. 先重试 3 次（1 秒间隔）
2. 仍然失败 → 触发压缩，重建更短的 prompt
3. 压缩后再重试 3 次

### 相关环境变量

```bash
PROACTIVE_SUMMARY_THRESHOLD=256000   # 主动压缩阈值（字符数）
SUMMARY_ENGINE=deepseek-chat         # 总结模型
SUMMARY_API_BASE=https://newapi.haomo.de/v1/chat/completions  # 总结 API 地址
SUMMARY_API_KEY=sk-xxx               # 总结 API Key
SUMMARY_KEEP_ROUNDS=3                # 压缩时保留最近几轮对话
SUMMARY_CACHE_SIZE=200               # 摘要缓存大小（LRU）

# Tool 结果截断
TOOL_TRUNCATE_RECENT_ROUNDS=3        # 完整保留最近几轮
TOOL_TRUNCATE_MID_ROUNDS=10          # 中期轮数边界
TOOL_TRUNCATE_MID_CHARS=500          # 中期截断字符数
TOOL_TRUNCATE_OLD_CHARS=100          # 远期截断字符数
```

### System 提示词保护

压缩过程**不会动 system 提示词**，只压缩 user/assistant 对话历史。

## API 端点

| 端点 | 说明 | 所有模式 |
|---|---|---|
| `POST /v1/chat/completions` | 聊天补全（兼容 OpenAI） | ✅ |
| `GET /v1/models` | 模型列表 | ✅ |
| `GET /health` | 健康检查 + 账号状态 | ✅ |
| `GET /admin` | 管理面板（Web UI） | ✅ |

### 管理端点

**server.js（Node.js 版）：**

| 端点 | 说明 |
|---|---|
| `GET /admin/accounts` | 查看所有账号 |
| `POST /admin/accounts` | 添加账号 |
| `PUT /admin/accounts/:index` | 更新账号 |
| `DELETE /admin/accounts/:index` | 删除账号 |
| `POST /admin/oauth/start` | 启动 OAuth 授权流程 |
| `POST /admin/oauth/exchange` | 完成 OAuth 授权（换 token） |
| `POST /admin/reload` | 重新加载账号文件 |
| `GET /admin/logs?date=YYYY-MM-DD&limit=50` | 查询请求日志 |

**server.py（Python 版）：**

| 端点 | 说明 |
|---|---|
| `GET /admin/keys` | 查看 API Key 池 |
| `POST /admin/keys` | 添加 Key |
| `DELETE /admin/keys/:index` | 删除 Key |
| `POST /admin/keys/:index/toggle` | 启用/禁用 Key |

## 支持的模型

server.py 支持模型名映射（OpenAI 风格 → Agent.AI 引擎名）：

```
gpt-4o, gpt-4o-mini, gpt-4.1, gpt-4.1-mini, gpt-4.1-nano
gpt-5, gpt-5.2, gpt-5.2-codex
o1, o3, o3-mini, o3-pro
claude-opus-4, claude-opus-4-5, claude-opus-4-6, claude-sonnet-4
gemini-2.0-flash, perplexity
```

server.js 使用 MCP 调用，模型由 Agent 本身决定。

## Function Calling 实现

两种不同的实现方式：

- **server.py / server_llm.py**：Toolify 风格，用 XML trigger signal（`<Function_XXXX_Start/>`）+ `<function_calls>` XML 格式
- **server.js**：JSON 提示词风格，让模型输出 `{"tool_calls": [...]}` JSON 格式

## OAuth 授权流程（server.js）

通过管理面板或 API 完成 PKCE 授权：

```bash
# 1. 启动授权
curl -X POST http://localhost:9090/admin/oauth/start \
  -H "Content-Type: application/json" \
  -d '{"agent_id": "h252a58ehz9qlcwp"}'
# 返回 auth_url，在浏览器打开授权

# 2. 用回调的 code 换 token
curl -X POST http://localhost:9090/admin/oauth/exchange \
  -H "Content-Type: application/json" \
  -d '{"session_id": "...", "code": "..."}'
```

也可以通过 `get-token.js` 辅助获取 token。

## 日志（server.js）

请求日志按天存储在 `logs/` 目录（JSONL 格式），记录：
- 账号、模型、消息、prompt、响应、工具调用、耗时、错误

查询：`GET /admin/logs?date=2026-02-15&limit=50`

## 文件结构

```
├── server.js          # Node.js MCP 版（推荐）
├── server.py          # Python invoke_llm 版（多 key + invoke_agent + 上下文压缩）
├── server_llm.py      # Python 精简版（单 key）
├── main.py            # MCP 客户端库（Python，server.py 旧版用）
├── accounts.json      # OAuth 账号池（server.js 用）
├── api_keys.json      # API Key 池（server.py 用）
├── admin.html         # 管理面板 Web UI
├── get-token.js       # OAuth token 获取工具
└── logs/              # 请求日志（server.js）
```

## systemd 服务

```bash
# 查看状态
systemctl status agentai-proxy

# 重启
systemctl restart agentai-proxy
```
