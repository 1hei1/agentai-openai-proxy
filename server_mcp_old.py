"""OpenAI-compatible proxy using Agent.AI invoke_llm API."""

import asyncio
import json
import logging
import logging.handlers
import math
import os
import re
import secrets
import string
import time
import uuid
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

# ── Config ───────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AGENTAI_API_KEY = os.getenv("AGENTAI_API_KEY", "YUG7luaZIfAREDro03K8Qr87S65Gc5W5WscWCAJVLyBsuiEoDBtiW1w0GFt9QMRZ")
AGENTAI_URL = os.getenv("AGENTAI_URL", "https://api-lr.agent.ai/v1/action/invoke_llm")
DEFAULT_ENGINE = os.getenv("DEFAULT_ENGINE", "claude-sonnet-4")
API_KEY = os.getenv("API_KEY", "sk-agentai-proxy")
PORT = int(os.getenv("PORT", "9090"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# Model name mapping: OpenAI-style names -> Agent.AI engine names
ENGINE_MAP = {
    "gpt-4o": "gpt4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "o1": "o1",
    "o3-mini": "o3-mini",
    "claude-opus": "claude_opus",
    "claude-3-opus": "claude-3-opus",
    "claude-opus-4": "claude-opus-4",
    "claude-opus-4-5": "claude-opus-4-5",
    "claude-opus-4-5-20251101": "claude-opus-4-5-20251101",
    "claude-3-sonnet": "claude-3-sonnet",
    "claude-sonnet-4": "claude-sonnet-4",
    "gemini-2.0-flash": "gemini-2.0-flash",
    "perplexity": "perplexity",
}

# ── Logging ──────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
logger = logging.getLogger("agentai.proxy")
fh = logging.handlers.RotatingFileHandler(os.path.join(BASE_DIR, "server.log"), maxBytes=10*1024*1024, backupCount=3)
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
logging.getLogger().addHandler(fh)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# ── Auth ─────────────────────────────────────────────────────────────

def verify_api_key(authorization: str | None) -> None:
    if not API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing API key")
    if authorization[7:] != API_KEY:
        raise HTTPException(401, "Invalid API key")

# ── Toolify-style function calling ───────────────────────────────────

def _generate_trigger_signal() -> str:
    chars = string.ascii_letters + string.digits
    return f"<Function_{''.join(secrets.choice(chars) for _ in range(4))}_Start/>"

GLOBAL_TRIGGER_SIGNAL = _generate_trigger_signal()
_TRIGGER_PATTERN = re.compile(r"<Function_[A-Za-z0-9]{4}_Start/>")

def _extract_text(content) -> str:
    if isinstance(content, str): return content
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text").strip()
    return str(content) if content else ""

def _build_tool_call_index(messages):
    idx = {}
    for msg in messages:
        if msg.get("role") != "assistant": continue
        for tc in (msg.get("tool_calls") or []):
            if not isinstance(tc, dict): continue
            fn = tc.get("function", {})
            args = fn.get("arguments", "{}")
            if not isinstance(args, str):
                try: args = json.dumps(args, ensure_ascii=False)
                except: args = "{}"
            if tc.get("id") and fn.get("name"):
                idx[tc["id"]] = {"name": fn["name"], "arguments": args}
    return idx

def _preprocess_messages(messages):
    tool_idx = _build_tool_call_index(messages)
    out = []
    for msg in messages:
        if not isinstance(msg, dict): continue
        role = msg.get("role")
        if role == "tool":
            info = tool_idx.get(msg.get("tool_call_id"), {"name": msg.get("name", "unknown"), "arguments": "{}"})
            out.append({"role": "user", "content": f"<tool_result><tool_name>{info['name']}</tool_name><tool_arguments>{info['arguments']}</tool_arguments><tool_output>{_extract_text(msg.get('content', ''))}</tool_output></tool_result>"})
        elif role == "assistant" and isinstance(msg.get("tool_calls"), list):
            blocks = []
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                name = fn.get("name", "")
                args = fn.get("arguments", "{}")
                if not isinstance(args, str):
                    try: args = json.dumps(args, ensure_ascii=False)
                    except: args = "{}"
                if name:
                    blocks.append(f"<function_call><name>{name}</name><args_json>{args}</args_json></function_call>")
            xml = f"{GLOBAL_TRIGGER_SIGNAL}\n<function_calls>\n" + "\n".join(blocks) + "\n</function_calls>" if blocks else ""
            out.append({"role": "assistant", "content": (_extract_text(msg.get("content", "")) + "\n" + xml).strip()})
        elif role == "developer":
            out.append({**msg, "role": "system"})
        else:
            out.append(msg)
    return out

def _flatten_messages(messages):
    parts = []
    for msg in messages:
        role = str(msg.get("role", "user")).upper()
        parts.append(f"<{role}>{_extract_text(msg.get('content', ''))}</{role}>")
    return "\n".join(parts)

def _generate_function_prompt(tools, trigger):
    lines = []
    for i, t in enumerate(tools):
        if t.get("type") != "function": continue
        fn = t.get("function", {})
        name = fn.get("name", "")
        if not name: continue
        desc = str(fn.get("description", ""))[:80]
        lines.append(f"{i+1}. {name}: {desc}")
    return (
        f"You have tools. To call them, output:\n{trigger}\n"
        "<function_calls>\n<function_call><name>NAME</name><args_json>{{\"arg\":\"val\"}}</args_json></function_call>\n</function_calls>\n\n"
        f"Tools:\n" + "\n".join(lines)
    )

def _parse_function_calls(text):
    last = -1
    for m in _TRIGGER_PATTERN.finditer(text): last = m.start()
    if last == -1: return []
    sub = text[last:]
    m = re.search(r"<function_calls>([\s\S]*?)</function_calls>", sub)
    if not m: return []
    out = []
    for c in re.findall(r"<function_call>([\s\S]*?)</function_call>", m.group(1)):
        nm = re.search(r"<name>([\s\S]*?)</name>", c)
        am = re.search(r"<args_json>([\s\S]*?)</args_json>", c)
        if not nm: continue
        args_raw = am.group(1).strip() if am else "{}"
        try:
            parsed = json.loads(args_raw)
            if not isinstance(parsed, dict): parsed = {"value": parsed}
        except: parsed = {"raw": args_raw}
        out.append({"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                     "function": {"name": nm.group(1).strip(), "arguments": json.dumps(parsed, ensure_ascii=False)}})
    return out

def _find_trigger_pos(text):
    last = -1
    for m in _TRIGGER_PATTERN.finditer(text): last = m.start()
    return last

# ── Helpers ──────────────────────────────────────────────────────────

def _make_id(): return f"chatcmpl-{uuid.uuid4().hex[:29]}"

def _openai_chunk(cid, model, *, content=None, finish_reason=None):
    delta = {}
    if content is not None: delta["content"] = content
    return {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()),
            "model": model, "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}

def _simulate_stream(text, chunk_size=20):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)] if text else [""]

def _estimate_tokens(text): return max(1, len(text) // 2) if text else 0

# ── App ──────────────────────────────────────────────────────────────

http_client: httpx.AsyncClient | None = None

@asynccontextmanager
async def lifespan(_app):
    global http_client
    http_client = httpx.AsyncClient(timeout=httpx.Timeout(connect=10, read=120, write=10, pool=5))
    yield
    await http_client.aclose()

app = FastAPI(lifespan=lifespan)

async def _call_llm(prompt: str, engine: str) -> str:
    resp = await http_client.post(AGENTAI_URL,
        headers={"Authorization": f"Bearer {AGENTAI_API_KEY}", "Content-Type": "application/json"},
        json={"instructions": prompt, "llm_engine": engine})
    if resp.status_code != 200:
        raise RuntimeError(f"Agent.AI HTTP {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    text = data.get("response", "")
    if text.startswith("Error:"):
        raise RuntimeError(f"Agent.AI error: {text[:300]}")
    return text

@app.get("/health")
async def health():
    return {"status": "ok", "mode": "invoke_llm", "default_engine": DEFAULT_ENGINE}

@app.get("/v1/models")
async def list_models(authorization: str = Header(None)):
    verify_api_key(authorization)
    models = list(ENGINE_MAP.keys()) + [DEFAULT_ENGINE]
    return {"object": "list", "data": [{"id": m, "object": "model", "created": int(time.time()), "owned_by": "agent-ai"} for m in set(models)]}

@app.get("/v1/models/{model_id}")
async def get_model(model_id: str, authorization: str = Header(None)):
    verify_api_key(authorization)
    return {"id": model_id, "object": "model", "created": int(time.time()), "owned_by": "agent-ai"}

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    verify_api_key(request.headers.get("authorization"))
    body = await request.json()
    model = body.get("model", DEFAULT_ENGINE)
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    tools = body.get("tools")
    tool_choice = body.get("tool_choice")

    engine = ENGINE_MAP.get(model, model)
    processed = _preprocess_messages(messages)

    has_fc = bool(tools)
    if has_fc:
        fc_prompt = _generate_function_prompt(tools, GLOBAL_TRIGGER_SIGNAL)
        if isinstance(tool_choice, str) and tool_choice == "required":
            fc_prompt += "\nYou MUST call at least one tool."
        elif isinstance(tool_choice, str) and tool_choice == "none":
            fc_prompt += "\nDo NOT call tools. Answer directly."
        elif isinstance(tool_choice, dict):
            name = tool_choice.get("function", {}).get("name")
            if name: fc_prompt += f"\nYou MUST call: {name}"
        processed.insert(0, {"role": "system", "content": fc_prompt})

    prompt_text = _flatten_messages(processed)
    req_id = f"req_{uuid.uuid4().hex[:10]}"
    logger.info("[entry][%s] model=%s engine=%s stream=%s tools=%d msgs=%d prompt_len=%d",
                req_id, model, engine, stream, len(tools or []), len(messages), len(prompt_text))

    if stream:
        async def gen_sse():
            cid = _make_id()
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    text = await _call_llm(prompt_text, engine)
                    if not text.strip() and attempt < MAX_RETRIES:
                        logger.warning("[stream][%s] empty, retry %d", cid, attempt)
                        await asyncio.sleep(2 ** attempt)
                        continue

                    yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"

                    parsed_tcs = _parse_function_calls(text) if has_fc else []
                    if parsed_tcs:
                        logger.info("[stream][%s] tool_calls=%d", cid, len(parsed_tcs))
                        prefix_pos = _find_trigger_pos(text)
                        if prefix_pos > 0:
                            for chunk in _simulate_stream(text[:prefix_pos].rstrip()):
                                yield f"data: {json.dumps(_openai_chunk(cid, model, content=chunk), ensure_ascii=False)}\n\n"
                        for i, tc in enumerate(parsed_tcs):
                            yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model, 'choices': [{'index': 0, 'delta': {'tool_calls': [{'index': i, **tc}]}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"
                        yield f"data: {json.dumps(_openai_chunk(cid, model, finish_reason='tool_calls'), ensure_ascii=False)}\n\n"
                    else:
                        for chunk in _simulate_stream(text):
                            yield f"data: {json.dumps(_openai_chunk(cid, model, content=chunk), ensure_ascii=False)}\n\n"
                        yield f"data: {json.dumps(_openai_chunk(cid, model, finish_reason='stop'), ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

                except Exception as e:
                    logger.error("[stream][%s] attempt %d error: %s", cid, attempt, e)
                    if attempt < MAX_RETRIES:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    yield f"data: {json.dumps({'error': {'message': str(e)[:200], 'type': 'server_error'}}, ensure_ascii=False)}\n\n"
                    yield "data: [DONE]\n\n"
                    return

        return StreamingResponse(gen_sse(), media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"})

    # Non-streaming
    cid = _make_id()
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            text = await _call_llm(prompt_text, engine)
            if not text.strip() and attempt < MAX_RETRIES:
                await asyncio.sleep(2 ** attempt)
                continue
            parsed_tcs = _parse_function_calls(text) if has_fc else []
            if parsed_tcs:
                prefix_pos = _find_trigger_pos(text)
                prefix = text[:prefix_pos].rstrip() if prefix_pos > 0 else None
                msg = {"role": "assistant", "content": prefix, "tool_calls": parsed_tcs}
                fr = "tool_calls"
            else:
                msg = {"role": "assistant", "content": text}
                fr = "stop"
            p, c = _estimate_tokens(prompt_text), _estimate_tokens(text)
            return {"id": cid, "object": "chat.completion", "created": int(time.time()), "model": model,
                    "choices": [{"index": 0, "message": msg, "finish_reason": fr}],
                    "usage": {"prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c}}
        except Exception as e:
            logger.exception("[sync][%s] attempt %d: %s", cid, attempt, e)
            if attempt < MAX_RETRIES:
                await asyncio.sleep(2 ** attempt)
                continue
            return JSONResponse(status_code=502, content={"error": {"message": str(e)[:200], "type": "server_error"}})

# ── Admin (keep compatible) ──────────────────────────────────────────

admin_html = os.path.join(BASE_DIR, "admin.html")
if os.path.exists(admin_html):
    from fastapi.responses import FileResponse
    @app.get("/admin")
    async def admin_page(): return FileResponse(admin_html)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
