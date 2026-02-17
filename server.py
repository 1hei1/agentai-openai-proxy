"""OpenAI-compatible proxy using Agent.AI invoke_llm API."""

import asyncio
import hashlib
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
from collections import OrderedDict, defaultdict
from datetime import datetime
from contextlib import asynccontextmanager

import httpx
import uvicorn
from fastapi import FastAPI, Request, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

# ── Config ───────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KEYS_FILE = os.path.join(BASE_DIR, "api_keys.json")
AGENTAI_LLM_URL = os.getenv("AGENTAI_LLM_URL", "https://api-lr.agent.ai/v1/action/invoke_llm")
AGENTAI_AGENT_URL = os.getenv("AGENTAI_AGENT_URL", "https://api-lr.agent.ai/v1/action/invoke_agent")
AGENTAI_AGENT_ID = os.getenv("AGENTAI_AGENT_ID", "h252a58ehz9qlcwp")
DEFAULT_ENGINE = os.getenv("DEFAULT_ENGINE", "claude-sonnet-4")
USE_AGENT = os.getenv("USE_AGENT", "false").lower() == "true"
API_KEY = os.getenv("API_KEY", "sk-agentai-proxy")
PORT = int(os.getenv("PORT", "9090"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
SUMMARY_ENGINE = os.getenv("SUMMARY_ENGINE", "deepseek-chat")
SUMMARY_API_BASE = os.getenv("SUMMARY_API_BASE", "https://newapi.haomo.de/v1/chat/completions")
SUMMARY_API_KEY = os.getenv("SUMMARY_API_KEY", "sk-PvUDadEMEH7FPMYd9BNZsS0sJoa6TJ0z0riSimfSjLp9AIjS")
SUMMARY_KEEP_ROUNDS = int(os.getenv("SUMMARY_KEEP_ROUNDS", "3"))
SUMMARY_CACHE_SIZE = int(os.getenv("SUMMARY_CACHE_SIZE", "200"))
PROACTIVE_SUMMARY_THRESHOLD = int(os.getenv("PROACTIVE_SUMMARY_THRESHOLD", "256000"))

# ── Summary Cache (LRU) ─────────────────────────────────────────────

class LRUCache:
    def __init__(self, maxsize=200):
        self._cache: OrderedDict[str, str] = OrderedDict()
        self._maxsize = maxsize

    def get(self, key: str) -> str | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, key: str, value: str):
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = value
        while len(self._cache) > self._maxsize:
            self._cache.popitem(last=False)

summary_cache = LRUCache(SUMMARY_CACHE_SIZE)

# ── Usage Statistics ─────────────────────────────────────────────────

usage_stats: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(lambda: {
    "requests": 0, "errors": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0
}))

def _record_usage(model: str, prompt_tokens: int = 0, completion_tokens: int = 0, error: bool = False):
    date_str = datetime.now().strftime("%Y-%m-%d")
    s = usage_stats[date_str][model]
    s["requests"] += 1
    if error:
        s["errors"] += 1
    else:
        s["prompt_tokens"] += prompt_tokens
        s["completion_tokens"] += completion_tokens
        s["total_tokens"] += prompt_tokens + completion_tokens

# Track the latest summary per conversation for rolling summarization
# Key: conversation_id (hash of first system+user), Value: latest summary text
_rolling_summaries: dict[str, str] = {}

def _make_conversation_id(messages: list[dict]) -> str:
    """Stable conversation identity based on first system + first user message."""
    parts = []
    for m in messages:
        role = m.get("role", "")
        content = _extract_text(m.get("content", ""))
        if role == "system" and not parts:
            parts.append(f"s:{content[:500]}")
        elif role == "user" and len(parts) < 2:
            parts.append(f"u:{content[:500]}")
        if len(parts) >= 2:
            break
    return hashlib.md5("|".join(parts).encode()).hexdigest()

def _make_cache_key(messages: list[dict]) -> str:
    """Cache key includes conversation id + message count for rolling summarization."""
    conv_id = _make_conversation_id(messages)
    msg_count = len(messages)
    # Round to nearest 10 to avoid regenerating on every single message
    rounded = (msg_count // 10) * 10
    return f"{conv_id}:{rounded}"

def _split_messages_for_summary(messages: list[dict], keep_rounds: int = SUMMARY_KEEP_ROUNDS):
    """Split messages into (system_msgs, old_msgs_to_summarize, recent_msgs_to_keep)."""
    system_msgs = []
    conversation = []
    for m in messages:
        if m.get("role") == "system":
            system_msgs.append(m)
        else:
            conversation.append(m)

    # Count rounds (a round = one user + one assistant)
    round_starts = []
    for i, m in enumerate(conversation):
        if m.get("role") == "user":
            round_starts.append(i)

    if len(round_starts) <= keep_rounds:
        return system_msgs, [], conversation

    cut = round_starts[-keep_rounds]
    return system_msgs, conversation[:cut], conversation[cut:]

async def _summarize_messages(old_messages: list[dict], cache_key: str, conv_id: str = "") -> str:
    """Summarize old messages using a cheap model. Supports rolling summarization."""
    cached = summary_cache.get(cache_key)
    if cached:
        logger.info("[summary] cache hit: %s", cache_key)
        return cached

    # Build conversation text from old messages
    parts = []
    for m in old_messages:
        role = m.get("role", "user")
        content = _extract_text(m.get("content", ""))
        if not content:
            continue
        parts.append(f"{role}: {content}")

    conversation_text = "\n".join(parts)

    # Rolling summarization: include previous summary as context
    prev_summary = _rolling_summaries.get(conv_id, "")
    if prev_summary:
        summary_prompt = (
            "以下是之前对话的摘要，以及之后新增的对话内容。"
            "请将两部分合并，生成一份更新的简洁总结，保留所有重要的事实、决定和上下文。"
            "直接输出总结，不要加前缀。\n\n"
            f"【之前的摘要】\n{prev_summary}\n\n"
            f"【新增对话】\n{conversation_text}"
        )
    else:
        summary_prompt = (
            "请用中文简洁总结以下对话的关键信息，保留重要的事实、决定和上下文，去掉寒暄和重复内容。"
            "直接输出总结，不要加前缀。\n\n"
            f"{conversation_text}"
        )

    logger.info("[summary] generating for %s, input_len=%d", cache_key[:12], len(summary_prompt))
    try:
        resp = await http_client.post(SUMMARY_API_BASE,
            headers={"Authorization": f"Bearer {SUMMARY_API_KEY}", "Content-Type": "application/json"},
            json={"model": SUMMARY_ENGINE, "messages": [{"role": "user", "content": summary_prompt}], "max_tokens": 8192},
            timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if text:
                summary_cache.put(cache_key, text)
                if conv_id:
                    _rolling_summaries[conv_id] = text
                logger.info("[summary] generated, len=%d, rolling=%s, content: %s", len(text), bool(prev_summary), text[:500])
                return text
        logger.warning("[summary] failed: HTTP %d %s", resp.status_code, resp.text[:200])
    except Exception as e:
        logger.error("[summary] error: %s", e)
    return ""

async def _summarize_messages_nocache(old_messages: list[dict], conv_id: str) -> str:
    """Force regenerate summary without cache, for rolling re-summarization."""
    parts = []
    for m in old_messages:
        role = m.get("role", "user")
        content = _extract_text(m.get("content", ""))
        if not content:
            continue
        parts.append(f"{role}: {content}")

    conversation_text = "\n".join(parts)

    prev_summary = _rolling_summaries.get(conv_id, "")
    if prev_summary:
        summary_prompt = (
            "以下是之前对话的摘要，以及之后新增的对话内容。"
            "请将两部分合并，生成一份更新的简洁总结，保留所有重要的事实、决定和上下文。"
            "直接输出总结，不要加前缀。\n\n"
            f"【之前的摘要】\n{prev_summary}\n\n"
            f"【新增对话】\n{conversation_text}"
        )
    else:
        summary_prompt = (
            "请用中文简洁总结以下对话的关键信息，保留重要的事实、决定和上下文，去掉寒暄和重复内容。"
            "直接输出总结，不要加前缀。\n\n"
            f"{conversation_text}"
        )

    logger.info("[summary] force generating for conv %s, input_len=%d, has_prev=%s",
                conv_id[:12], len(summary_prompt), bool(prev_summary))
    try:
        resp = await http_client.post(SUMMARY_API_BASE,
            headers={"Authorization": f"Bearer {SUMMARY_API_KEY}", "Content-Type": "application/json"},
            json={"model": SUMMARY_ENGINE, "messages": [{"role": "user", "content": summary_prompt}], "max_tokens": 8192},
            timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if text:
                _rolling_summaries[conv_id] = text
                logger.info("[summary] force generated, len=%d, content: %s", len(text), text[:500])
                return text
        logger.warning("[summary] force failed: HTTP %d %s", resp.status_code, resp.text[:200])
    except Exception as e:
        logger.error("[summary] force error: %s", e)
    return ""

def _rebuild_with_summary(system_msgs, summary_text, recent_msgs, tools=None):
    """Rebuild message list with summary injected."""
    rebuilt = list(system_msgs)
    if summary_text:
        rebuilt.append({"role": "system", "content": f"[之前对话的摘要]\n{summary_text}"})
    rebuilt.extend(recent_msgs)
    return rebuilt

# ── API Key Pool ─────────────────────────────────────────────────────

class KeyPool:
    def __init__(self):
        self._keys: list[dict] = []
        self._idx = 0
        self._lock = asyncio.Lock()

    def load(self):
        if os.path.exists(KEYS_FILE):
            with open(KEYS_FILE) as f:
                self._keys = json.load(f)
        else:
            # Migrate from single key
            self._keys = [{"name": "default", "key": "YUG7luaZIfAREDro03K8Qr87S65Gc5W5WscWCAJVLyBsuiEoDBtiW1w0GFt9QMRZ", "enabled": True, "requests": 0, "errors": 0}]
            self.save()
        logger.info("Key pool: %d keys loaded (%d enabled)", len(self._keys), sum(1 for k in self._keys if k.get("enabled", True)))

    def save(self):
        with open(KEYS_FILE, "w") as f:
            json.dump(self._keys, f, indent=2, ensure_ascii=False)

    async def next_key(self) -> str:
        async with self._lock:
            enabled = [k for k in self._keys if k.get("enabled", True)]
            if not enabled:
                raise RuntimeError("No enabled API keys")
            k = enabled[self._idx % len(enabled)]
            self._idx += 1
            k["requests"] = k.get("requests", 0) + 1
            return k["key"]

    def mark_error(self, key: str):
        for k in self._keys:
            if k["key"] == key:
                k["errors"] = k.get("errors", 0) + 1
                break

    @property
    def keys(self): return self._keys

key_pool = KeyPool()

# Model name mapping: OpenAI-style names -> Agent.AI engine names
ENGINE_MAP = {
    "gpt-4o": "gpt4o",
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4.1": "gpt-4.1",
    "gpt-4.1-mini": "gpt-4.1-mini",
    "gpt-4.1-nano": "gpt-4.1-nano",
    "gpt-5": "gpt-5",
    "gpt-5.2": "gpt-5.2",
    "gpt-5.2-codex": "gpt-5.2-codex",
    "o1": "o1",
    "o3": "o3",
    "o3-mini": "o3-mini",
    "o3-pro": "o3-pro",
    "claude-opus": "claude_opus",
    "claude-3-opus": "claude-3-opus",
    "claude-opus-4": "claude-opus-4",
    "claude-opus-4-5": "claude-opus-4-5",
    "claude-opus-4-5-20251101": "claude-opus-4-5-20251101",
    "claude-opus-4-6": "claude-opus-4-6",
    "claude-3-sonnet": "claude-3-sonnet",
    "claude-sonnet-4": "claude-sonnet-4",
    "gemini-2.0-flash": "gemini-2.0-flash",
    "perplexity": "perplexity",
}

# ── Logging ──────────────────────────────────────────────────────────

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
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

# ── Tool Result Truncation (distance-based) ─────────────────────────

TOOL_TRUNCATE_NEAR = int(os.getenv("TOOL_TRUNCATE_NEAR", "500"))    # 3-10 rounds ago
TOOL_TRUNCATE_FAR = int(os.getenv("TOOL_TRUNCATE_FAR", "100"))      # 10+ rounds ago

def _truncate_tool_results(messages: list[dict]) -> list[dict]:
    """Truncate tool call results based on distance from the end of conversation.
    - Last 3 rounds: keep full
    - 3-10 rounds ago: truncate tool results to TOOL_TRUNCATE_NEAR chars
    - 10+ rounds ago: truncate tool results to TOOL_TRUNCATE_FAR chars
    """
    # Find round boundaries (each user message starts a new round)
    round_starts = []
    for i, m in enumerate(messages):
        if m.get("role") == "user":
            round_starts.append(i)
    
    total_rounds = len(round_starts)
    if total_rounds <= 3:
        return messages  # Nothing to truncate
    
    # Map each message index to its round number (from the end)
    msg_round_from_end = {}
    for r_idx, start in enumerate(round_starts):
        end = round_starts[r_idx + 1] if r_idx + 1 < len(round_starts) else len(messages)
        rounds_from_end = total_rounds - r_idx
        for i in range(start, end):
            msg_round_from_end[i] = rounds_from_end
    
    result = []
    truncated_count = 0
    saved_chars = 0
    for i, m in enumerate(messages):
        rounds_ago = msg_round_from_end.get(i, 0)
        role = m.get("role", "")
        
        if role == "tool" and rounds_ago > 3:
            content = _extract_text(m.get("content", ""))
            limit = TOOL_TRUNCATE_NEAR if rounds_ago <= 10 else TOOL_TRUNCATE_FAR
            if len(content) > limit:
                saved_chars += len(content) - limit
                truncated_count += 1
                truncated = content[:limit] + f"\n... [截断，原始 {len(content)} 字符]"
                result.append({**m, "content": truncated})
                continue
        
        # Also truncate tool_call arguments in old assistant messages
        if role == "assistant" and rounds_ago > 3 and isinstance(m.get("tool_calls"), list):
            new_tcs = []
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    limit = TOOL_TRUNCATE_NEAR if rounds_ago <= 10 else TOOL_TRUNCATE_FAR
                    if len(args) > limit:
                        saved_chars += len(args) - limit
                        args = args[:limit] + "..."
                        tc = {**tc, "function": {**fn, "arguments": args}}
                new_tcs.append(tc)
            result.append({**m, "tool_calls": new_tcs})
            continue
        
        result.append(m)
    
    if truncated_count > 0:
        logger.info("[tool-truncate] truncated %d tool messages, saved ~%d chars", truncated_count, saved_chars)
    return result

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
        desc = str(fn.get("description", ""))[:100]
        params = fn.get("parameters", {})
        required = params.get("required", []) if isinstance(params, dict) else []
        props = params.get("properties", {}) if isinstance(params, dict) else {}
        # Show required param names + types
        param_info = ""
        if props:
            param_parts = []
            for pname, pdef in props.items():
                ptype = pdef.get("type", "string") if isinstance(pdef, dict) else "string"
                marker = " (required)" if pname in required else ""
                param_parts.append(f"{pname}:{ptype}{marker}")
            param_info = " | params: " + ", ".join(param_parts)
        lines.append(f"{i+1}. {name}: {desc}{param_info}")
    return (
        f"You have tools. To call them, output EXACTLY ONE tool call at a time:\n{trigger}\n"
        "<function_calls>\n<function_call><name>NAME</name><args_json>{{\"arg\":\"val\"}}</args_json></function_call>\n</function_calls>\n\n"
        "IMPORTANT: Call only ONE tool per response. Wait for the result before calling the next tool.\n\n"
        f"Tools:\n" + "\n".join(lines)
    )

def _parse_function_calls(text):
    # Try trigger signal first
    m = _TRIGGER_PATTERN.search(text)
    if m:
        sub = text[m.start():]
    else:
        # Fallback: match <function_calls> directly (Agent.AI built-in format)
        sub = text
    m2 = re.search(r"<function_calls>([\s\S]*?)</function_calls>", sub)
    if not m2: return []
    out = []
    for c in re.findall(r"<function_call>([\s\S]*?)</function_call>", m2.group(1)):
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
    m = _TRIGGER_PATTERN.search(text)
    if m: return m.start()
    # Fallback: find <function_calls> directly
    m2 = re.search(r"<function_calls>", text)
    return m2.start() if m2 else -1

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
    key_pool.load()
    yield
    await http_client.aclose()

app = FastAPI(lifespan=lifespan)

async def _call_llm(prompt: str, engine: str, model: str = "") -> str:
    api_key = await key_pool.next_key()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if USE_AGENT:
        resp = await http_client.post(AGENTAI_AGENT_URL, headers=headers,
            json={"id": AGENTAI_AGENT_ID, "user_input": prompt})
    else:
        resp = await http_client.post(AGENTAI_LLM_URL, headers=headers,
            json={"instructions": prompt, "llm_engine": engine})
    if resp.status_code != 200:
        key_pool.mark_error(api_key)
        _record_usage(model or engine, error=True)
        raise RuntimeError(f"Agent.AI HTTP {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    text = data.get("response", "")
    if text.startswith("Error:"):
        key_pool.mark_error(api_key)
        _record_usage(model or engine, error=True)
        raise RuntimeError(f"Agent.AI error: {text[:300]}")
    # Record token usage from metadata
    meta_usage = (data.get("metadata") or {}).get("usage") or {}
    pt = meta_usage.get("input_tokens", 0) or _estimate_tokens(prompt)
    ct = meta_usage.get("output_tokens", 0) or _estimate_tokens(text)
    _record_usage(model or engine, prompt_tokens=pt, completion_tokens=ct)
    return text

@app.get("/health")
async def health():
    return {"status": "ok", "mode": "invoke_agent" if USE_AGENT else "invoke_llm", "default_engine": DEFAULT_ENGINE, "agent_id": AGENTAI_AGENT_ID if USE_AGENT else None}

@app.get("/v1/models")
async def list_models(authorization: str = Header(None)):
    verify_api_key(authorization)
    models = list(ENGINE_MAP.keys()) + [DEFAULT_ENGINE]
    return {"object": "list", "data": [{"id": m, "object": "model", "created": int(time.time()), "owned_by": "agent-ai"} for m in set(models)]}

@app.get("/v1/models/{model_id}")
async def get_model(model_id: str, authorization: str = Header(None)):
    verify_api_key(authorization)
    return {"id": model_id, "object": "model", "created": int(time.time()), "owned_by": "agent-ai"}

def _prepare_prompt(messages, tools, tool_choice):
    """Preprocess messages and build the flattened prompt text. Returns (processed, prompt_text, has_fc)."""
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
    return processed, _flatten_messages(processed), has_fc

async def _try_summarize_and_rebuild(messages, tools, tool_choice, force_regenerate=False):
    """Summarize old messages and rebuild a shorter prompt.
    If force_regenerate=True, skip cache and regenerate summary.
    Returns (new_prompt_text, has_fc) or None if not applicable."""
    system_msgs, old_msgs, recent_msgs = _split_messages_for_summary(messages)
    if not old_msgs:
        logger.info("[summary] no old messages to summarize, skip")
        return None
    conv_id = _make_conversation_id(messages)

    if force_regenerate:
        # Bypass cache, force a new rolling summary
        logger.info("[summary] force regenerate for conv %s", conv_id[:12])
        summary = await _summarize_messages_nocache(old_msgs, conv_id)
    else:
        cache_key = _make_cache_key(messages)
        summary = await _summarize_messages(old_msgs, cache_key, conv_id)

    if not summary:
        logger.warning("[summary] failed to generate summary, keeping original messages (no truncation)")
        return None
    rebuilt = _rebuild_with_summary(system_msgs, summary, recent_msgs)
    logger.info("[summary] rebuilt: %d system + summary + %d recent (was %d total)",
                len(system_msgs), len(recent_msgs), len(messages))
    _, prompt_text, has_fc = _prepare_prompt(rebuilt, tools, tool_choice)
    return prompt_text, has_fc

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
    messages = _truncate_tool_results(messages)
    _, prompt_text, has_fc = _prepare_prompt(messages, tools, tool_choice)

    req_id = f"req_{uuid.uuid4().hex[:10]}"
    logger.info("[entry][%s] model=%s engine=%s stream=%s tools=%d msgs=%d prompt_len=%d",
                req_id, model, engine, stream, len(tools or []), len(messages), len(prompt_text))

    # Proactive summarization: compress before hitting the LLM if prompt is too long
    summarized = False
    if len(prompt_text) > PROACTIVE_SUMMARY_THRESHOLD:
        logger.info("[proactive-summary][%s] prompt_len=%d > threshold=%d, summarizing",
                     req_id, len(prompt_text), PROACTIVE_SUMMARY_THRESHOLD)
        result = await _try_summarize_and_rebuild(messages, tools, tool_choice)
        if result:
            prompt_text, has_fc = result
            summarized = True
            logger.info("[proactive-summary][%s] compressed to prompt_len=%d", req_id, len(prompt_text))
            # If still over threshold after using cached summary, force regenerate
            if len(prompt_text) > PROACTIVE_SUMMARY_THRESHOLD:
                logger.info("[proactive-summary][%s] still over threshold (%d), force regenerating summary",
                             req_id, len(prompt_text))
                result2 = await _try_summarize_and_rebuild(messages, tools, tool_choice, force_regenerate=True)
                if result2:
                    prompt_text, has_fc = result2
                    logger.info("[proactive-summary][%s] re-compressed to prompt_len=%d", req_id, len(prompt_text))

    if stream:
        async def gen_sse():
            nonlocal prompt_text, has_fc, summarized
            cid = _make_id()
            post_summary_retries = 0
            for attempt in range(1, MAX_RETRIES * 2 + 2):  # enough room for post-summary retries
                try:
                    text = await _call_llm(prompt_text, engine, model)
                    if not text.strip():
                        if not summarized and attempt < MAX_RETRIES:
                            # Retry first (pre-summary)
                            logger.warning("[stream][%s] empty, retry %d/%d", cid, attempt, MAX_RETRIES)
                            await asyncio.sleep(1)
                            continue
                        # Try summarization if not done yet
                        if not summarized:
                            logger.warning("[stream][%s] retries exhausted, attempting summarization", cid)
                            result = await _try_summarize_and_rebuild(messages, tools, tool_choice)
                            if result:
                                prompt_text, has_fc = result
                                summarized = True
                                post_summary_retries = 0
                                logger.info("[stream][%s] summarized, new prompt_len=%d, retrying", cid, len(prompt_text))
                                continue
                        # Post-summary retries
                        if summarized:
                            post_summary_retries += 1
                            if post_summary_retries <= MAX_RETRIES:
                                logger.warning("[stream][%s] empty after summary, retry %d/%d", cid, post_summary_retries, MAX_RETRIES)
                                await asyncio.sleep(1)
                                continue
                        # Nothing left to try
                        logger.error("[stream][%s] empty after all retries + summary", cid)
                        yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"
                        yield f"data: {json.dumps(_openai_chunk(cid, model, content='[Error] Agent.AI 返回空响应，已重试多次仍失败，请稍后再试'), ensure_ascii=False)}\n\n"
                        yield f"data: {json.dumps(_openai_chunk(cid, model, finish_reason='stop'), ensure_ascii=False)}\n\n"
                        yield "data: [DONE]\n\n"
                        return

                    yield f"data: {json.dumps({'id': cid, 'object': 'chat.completion.chunk', 'created': int(time.time()), 'model': model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]}, ensure_ascii=False)}\n\n"

                    parsed_tcs = _parse_function_calls(text) if has_fc else []
                    if has_fc:
                        logger.debug("[stream][%s] raw_response: %s", cid, text[:2000])
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
                    if not summarized:
                        logger.warning("[stream][%s] error, attempting summarization", cid)
                        result = await _try_summarize_and_rebuild(messages, tools, tool_choice)
                        if result:
                            prompt_text, has_fc = result
                            summarized = True
                            logger.info("[stream][%s] summarized after error, new prompt_len=%d", cid, len(prompt_text))
                            continue
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
    sync_summary_attempt = 0
    for attempt in range(1, MAX_RETRIES * 2 + 2):
        try:
            text = await _call_llm(prompt_text, engine, model)
            if not text.strip():
                if not summarized and attempt < MAX_RETRIES:
                    logger.warning("[sync][%s] empty, retry %d/%d", cid, attempt, MAX_RETRIES)
                    await asyncio.sleep(1)
                    continue
                if not summarized:
                    logger.warning("[sync][%s] retries exhausted, attempting summarization", cid)
                    result = await _try_summarize_and_rebuild(messages, tools, tool_choice)
                    if result:
                        prompt_text, has_fc = result
                        summarized = True
                        sync_summary_attempt = 0
                        logger.info("[sync][%s] summarized, new prompt_len=%d, retrying", cid, len(prompt_text))
                        continue
                if summarized:
                    sync_summary_attempt += 1
                    if sync_summary_attempt <= MAX_RETRIES:
                        logger.warning("[sync][%s] empty after summary, retry %d/%d", cid, sync_summary_attempt, MAX_RETRIES)
                        await asyncio.sleep(1)
                        continue
                return JSONResponse(status_code=502, content={"error": {"message": "Empty response after retries", "type": "server_error"}})
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
            if not summarized:
                logger.warning("[sync][%s] error, attempting summarization", cid)
                result = await _try_summarize_and_rebuild(messages, tools, tool_choice)
                if result:
                    prompt_text, has_fc = result
                    summarized = True
                    logger.info("[sync][%s] summarized after error, new prompt_len=%d", cid, len(prompt_text))
                    continue
            if attempt < MAX_RETRIES:
                await asyncio.sleep(2 ** attempt)
                continue
            return JSONResponse(status_code=502, content={"error": {"message": str(e)[:200], "type": "server_error"}})

# ── Admin ────────────────────────────────────────────────────────────

@app.get("/admin/usage")
async def admin_usage(date: str = None, model: str = None):
    result = {}
    for d, models in usage_stats.items():
        if date and d != date:
            continue
        for m, s in models.items():
            if model and m != model:
                continue
            result.setdefault(d, {})[m] = dict(s)
    # Compute totals
    totals = {"requests": 0, "errors": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for d in result.values():
        for s in d.values():
            for k in totals:
                totals[k] += s.get(k, 0)
    return {"totals": totals, "by_date": result}

@app.get("/admin/keys")
async def admin_list_keys():
    keys = []
    for i, k in enumerate(key_pool.keys):
        keys.append({"index": i, "name": k.get("name", ""), "enabled": k.get("enabled", True),
                      "key_preview": k["key"][:8] + "...", "requests": k.get("requests", 0), "errors": k.get("errors", 0)})
    return {"total": len(keys), "keys": keys}

@app.post("/admin/keys")
async def admin_add_key(request: Request):
    body = await request.json()
    key_pool.keys.append({"name": body.get("name", f"key-{len(key_pool.keys)+1}"), "key": body["key"], "enabled": True, "requests": 0, "errors": 0})
    key_pool.save()
    return {"status": "ok", "total": len(key_pool.keys)}

@app.delete("/admin/keys/{index}")
async def admin_delete_key(index: int):
    if index < 0 or index >= len(key_pool.keys): raise HTTPException(404)
    removed = key_pool.keys.pop(index)
    key_pool.save()
    return {"status": "ok", "removed": removed.get("name")}

@app.post("/admin/keys/{index}/toggle")
async def admin_toggle_key(index: int):
    if index < 0 or index >= len(key_pool.keys): raise HTTPException(404)
    k = key_pool.keys[index]
    k["enabled"] = not k.get("enabled", True)
    key_pool.save()
    return {"name": k.get("name"), "enabled": k["enabled"]}

@app.get("/admin/usage")
async def admin_usage(date: str = None, model: str = None):
    result = {}
    for d, models in usage_stats.items():
        if date and d != date:
            continue
        for m, stats in models.items():
            if model and m != model:
                continue
            result.setdefault(d, {})[m] = dict(stats)
    totals = {"requests": 0, "errors": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for d in result.values():
        for stats in d.values():
            for k in totals:
                totals[k] += stats[k]
    return {"usage": result, "totals": totals}

admin_html = os.path.join(BASE_DIR, "admin.html")
if os.path.exists(admin_html):
    from fastapi.responses import FileResponse
    @app.get("/admin")
    async def admin_page(): return FileResponse(admin_html)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
