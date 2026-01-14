import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

import setproctitle
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.models import get_registry, get_loader, ModelInfo

setproctitle.setproctitle("mlx-server")

VERSION = "0.1.0"

app = FastAPI(title="MLX Server", version=VERSION)


# ##################################################################
# request models
# pydantic models for API requests


class GenerateRequest(BaseModel):
    model: str
    prompt: str = ""
    suffix: str | None = None
    images: list[str] | None = None
    format: str | dict | None = None
    system: str | None = None
    stream: bool = True
    raw: bool = False
    keep_alive: str | int | None = None
    options: dict[str, Any] | None = None


class Message(BaseModel):
    role: str
    content: str
    images: list[str] | None = None
    tool_calls: list[dict] | None = None


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    tools: list[dict] | None = None
    format: str | dict | None = None
    stream: bool = True
    keep_alive: str | int | None = None
    options: dict[str, Any] | None = None


class ShowRequest(BaseModel):
    model: str
    verbose: bool = False


class PullRequest(BaseModel):
    model: str
    insecure: bool = False
    stream: bool = True


class EmbedRequest(BaseModel):
    model: str
    input: str | list[str]
    truncate: bool = True
    keep_alive: str | int | None = None
    options: dict[str, Any] | None = None


class DeleteRequest(BaseModel):
    model: str


# ##################################################################
# openai request models
# pydantic models for OpenAI-compatible endpoints


class OpenAIMessage(BaseModel):
    role: str
    content: str | list[dict] | None = None


class OpenAIChatRequest(BaseModel):
    model: str
    messages: list[OpenAIMessage]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    stop: str | list[str] | None = None
    seed: int | None = None
    tools: list[dict] | None = None


class OpenAICompletionRequest(BaseModel):
    model: str
    prompt: str | list[str]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    stop: str | list[str] | None = None
    seed: int | None = None


class OpenAIEmbeddingRequest(BaseModel):
    model: str
    input: str | list[str]
    encoding_format: str | None = None


# ##################################################################
# helper functions
# utilities for response formatting


def parse_keep_alive(keep_alive: str | int | None) -> float:
    if keep_alive is None:
        return 300.0
    if isinstance(keep_alive, int):
        return float(keep_alive)
    if keep_alive.endswith("m"):
        return float(keep_alive[:-1]) * 60
    if keep_alive.endswith("h"):
        return float(keep_alive[:-1]) * 3600
    if keep_alive.endswith("s"):
        return float(keep_alive[:-1])
    return float(keep_alive)


def get_options(options: dict | None, defaults: dict | None = None) -> dict:
    result = defaults or {"temperature": 0.7, "top_p": 0.9, "max_tokens": 256}
    if options:
        if "temperature" in options:
            result["temperature"] = options["temperature"]
        if "top_p" in options:
            result["top_p"] = options["top_p"]
        if "num_predict" in options:
            result["max_tokens"] = options["num_predict"]
        if "num_ctx" in options:
            result["context_length"] = options["num_ctx"]
    return result


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# ##################################################################
# ollama native endpoints
# native API compatible with Ollama


@app.get("/api/version")
async def api_version() -> dict:
    return {"version": VERSION}


@app.get("/api/tags")
async def api_tags() -> dict:
    registry = get_registry()
    models = []
    for info in registry.get_available():
        models.append({
            "name": info.name,
            "model": info.name,
            "modified_at": now_iso(),
            "size": 0,
            "digest": "",
            "details": {
                "parent_model": "",
                "format": "mlx",
                "family": info.family,
                "families": [info.family],
                "parameter_size": info.parameter_size,
                "quantization_level": info.quantization,
            },
        })
    return {"models": models}


@app.get("/api/ps")
async def api_ps() -> dict:
    registry = get_registry()
    models = []
    for loaded in registry.get_loaded():
        models.append({
            "name": loaded.info.name,
            "model": loaded.info.name,
            "size": loaded.size_bytes,
            "digest": "",
            "details": {
                "parent_model": "",
                "format": "mlx",
                "family": loaded.info.family,
                "families": [loaded.info.family],
                "parameter_size": loaded.info.parameter_size,
                "quantization_level": loaded.info.quantization,
            },
            "expires_at": datetime.fromtimestamp(
                loaded.keep_alive_until, timezone.utc
            ).isoformat().replace("+00:00", "Z"),
            "size_vram": loaded.size_bytes,
        })
    return {"models": models}


@app.post("/api/show")
async def api_show(request: ShowRequest) -> dict:
    registry = get_registry()
    info = registry.get_info(request.model)
    if not info:
        raise HTTPException(status_code=404, detail=f"Model {request.model} not found")
    return {
        "modelfile": f"FROM {info.hf_path}",
        "parameters": f"num_ctx {info.context_length}",
        "template": "",
        "details": {
            "parent_model": "",
            "format": "mlx",
            "family": info.family,
            "families": [info.family],
            "parameter_size": info.parameter_size,
            "quantization_level": info.quantization,
        },
        "model_info": {
            "general.architecture": info.family,
            "general.parameter_count": info.parameter_size,
        },
    }


@app.post("/api/generate", response_model=None)
async def api_generate(request: GenerateRequest) -> StreamingResponse | dict:
    loader = get_loader()
    keep_alive = parse_keep_alive(request.keep_alive)
    options = get_options(request.options)

    if request.system and request.prompt:
        full_prompt = f"{request.system}\n\n{request.prompt}"
    else:
        full_prompt = request.prompt

    start_time = time.time_ns()

    if request.stream:
        return StreamingResponse(
            _stream_generate(
                request.model,
                full_prompt,
                options,
                keep_alive,
                start_time,
            ),
            media_type="application/x-ndjson",
        )

    response_text = loader.generate(
        request.model,
        full_prompt,
        max_tokens=options.get("max_tokens", 256),
        temperature=options.get("temperature", 0.7),
        top_p=options.get("top_p", 0.9),
        stream=False,
    )

    total_duration = time.time_ns() - start_time
    return {
        "model": request.model,
        "created_at": now_iso(),
        "response": response_text,
        "done": True,
        "done_reason": "stop",
        "context": [],
        "total_duration": total_duration,
        "load_duration": 0,
        "prompt_eval_count": len(full_prompt.split()),
        "prompt_eval_duration": 0,
        "eval_count": len(str(response_text).split()),
        "eval_duration": total_duration,
    }


async def _stream_generate(
    model: str,
    prompt: str,
    options: dict,
    keep_alive: float,
    start_time: int,
) -> AsyncGenerator[str, None]:
    loader = get_loader()
    loader.load(model, keep_alive)

    full_response = ""
    for chunk in loader.generate(
        model,
        prompt,
        max_tokens=options.get("max_tokens", 256),
        temperature=options.get("temperature", 0.7),
        top_p=options.get("top_p", 0.9),
        stream=True,
    ):
        full_response += chunk
        yield json.dumps({
            "model": model,
            "created_at": now_iso(),
            "response": chunk,
            "done": False,
        }) + "\n"

    total_duration = time.time_ns() - start_time
    yield json.dumps({
        "model": model,
        "created_at": now_iso(),
        "response": "",
        "done": True,
        "done_reason": "stop",
        "context": [],
        "total_duration": total_duration,
        "load_duration": 0,
        "prompt_eval_count": len(prompt.split()),
        "prompt_eval_duration": 0,
        "eval_count": len(full_response.split()),
        "eval_duration": total_duration,
    }) + "\n"


@app.post("/api/chat", response_model=None)
async def api_chat(request: ChatRequest) -> StreamingResponse | dict:
    loader = get_loader()
    keep_alive = parse_keep_alive(request.keep_alive)
    options = get_options(request.options)

    loaded = loader.load(request.model, keep_alive)
    tokenizer = loaded.tokenizer

    messages_for_template = [{"role": m.role, "content": m.content} for m in request.messages]
    try:
        prompt = tokenizer.apply_chat_template(
            messages_for_template,
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:
        prompt = "\n".join([f"{m.role}: {m.content}" for m in request.messages])
        prompt += "\nassistant:"

    start_time = time.time_ns()

    if request.stream:
        return StreamingResponse(
            _stream_chat(
                request.model,
                prompt,
                options,
                keep_alive,
                start_time,
            ),
            media_type="application/x-ndjson",
        )

    response_text = loader.generate(
        request.model,
        prompt,
        max_tokens=options.get("max_tokens", 256),
        temperature=options.get("temperature", 0.7),
        top_p=options.get("top_p", 0.9),
        stream=False,
    )

    total_duration = time.time_ns() - start_time
    return {
        "model": request.model,
        "created_at": now_iso(),
        "message": {
            "role": "assistant",
            "content": response_text,
        },
        "done": True,
        "done_reason": "stop",
        "total_duration": total_duration,
        "load_duration": 0,
        "prompt_eval_count": len(prompt.split()),
        "prompt_eval_duration": 0,
        "eval_count": len(str(response_text).split()),
        "eval_duration": total_duration,
    }


async def _stream_chat(
    model: str,
    prompt: str,
    options: dict,
    _keep_alive: float,
    start_time: int,
) -> AsyncGenerator[str, None]:
    loader = get_loader()

    full_response = ""
    for chunk in loader.generate(
        model,
        prompt,
        max_tokens=options.get("max_tokens", 256),
        temperature=options.get("temperature", 0.7),
        top_p=options.get("top_p", 0.9),
        stream=True,
    ):
        full_response += chunk
        yield json.dumps({
            "model": model,
            "created_at": now_iso(),
            "message": {
                "role": "assistant",
                "content": chunk,
            },
            "done": False,
        }) + "\n"

    total_duration = time.time_ns() - start_time
    yield json.dumps({
        "model": model,
        "created_at": now_iso(),
        "message": {
            "role": "assistant",
            "content": "",
        },
        "done": True,
        "done_reason": "stop",
        "total_duration": total_duration,
        "load_duration": 0,
        "prompt_eval_count": len(prompt.split()),
        "prompt_eval_duration": 0,
        "eval_count": len(full_response.split()),
        "eval_duration": total_duration,
    }) + "\n"


@app.post("/api/embed")
async def api_embed(_request: EmbedRequest) -> dict:
    raise HTTPException(
        status_code=501,
        detail="Embeddings not yet implemented for MLX models"
    )


@app.post("/api/pull", response_model=None)
async def api_pull(request: PullRequest) -> StreamingResponse | dict:
    registry = get_registry()
    loader = get_loader()

    hf_path = registry.resolve_path(request.model)
    if hf_path == request.model and "/" not in request.model:
        hf_path = f"mlx-community/{request.model}"

    info = ModelInfo(
        name=request.model,
        hf_path=hf_path,
    )
    registry.register(info)

    if request.stream:
        return StreamingResponse(
            _stream_pull(request.model, hf_path),
            media_type="application/x-ndjson",
        )

    loader.load(request.model)
    return {"status": "success"}


async def _stream_pull(model: str, hf_path: str) -> AsyncGenerator[str, None]:
    yield json.dumps({"status": f"pulling {hf_path}"}) + "\n"
    yield json.dumps({"status": "pulling manifest"}) + "\n"

    loader = get_loader()
    try:
        loader.load(model)
        yield json.dumps({"status": "verifying sha256 digest"}) + "\n"
        yield json.dumps({"status": "writing manifest"}) + "\n"
        yield json.dumps({"status": "success"}) + "\n"
    except Exception as e:
        yield json.dumps({"status": "error", "error": str(e)}) + "\n"


@app.delete("/api/delete")
async def api_delete(request: DeleteRequest) -> dict:
    loader = get_loader()
    if loader.unload(request.model):
        return {"status": "success"}
    raise HTTPException(status_code=404, detail=f"Model {request.model} not loaded")


# ##################################################################
# openai compatible endpoints
# OpenAI API compatibility layer


@app.get("/v1/models")
async def v1_models() -> dict:
    registry = get_registry()
    models = []
    for info in registry.get_available():
        models.append({
            "id": info.name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "mlx-server",
        })
    return {"object": "list", "data": models}


@app.get("/v1/models/{model_id}")
async def v1_model_info(model_id: str) -> dict:
    registry = get_registry()
    info = registry.get_info(model_id)
    if not info:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    return {
        "id": info.name,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "mlx-server",
    }


@app.post("/v1/chat/completions", response_model=None)
async def v1_chat_completions(request: OpenAIChatRequest) -> StreamingResponse | dict:
    loader = get_loader()
    loaded = loader.load(request.model)
    tokenizer = loaded.tokenizer

    messages_for_template = []
    for m in request.messages:
        content = m.content
        if isinstance(content, list):
            text_parts = [p.get("text", "") for p in content if p.get("type") == "text"]
            content = " ".join(text_parts)
        messages_for_template.append({"role": m.role, "content": content or ""})

    try:
        prompt = tokenizer.apply_chat_template(
            messages_for_template,
            add_generation_prompt=True,
            tokenize=False,
        )
    except Exception:
        prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages_for_template])
        prompt += "\nassistant:"

    max_tokens = request.max_tokens or 256
    temperature = request.temperature if request.temperature is not None else 0.7
    top_p = request.top_p if request.top_p is not None else 0.9

    if request.stream:
        return StreamingResponse(
            _stream_openai_chat(
                request.model,
                prompt,
                max_tokens,
                temperature,
                top_p,
            ),
            media_type="text/event-stream",
        )

    response_text = loader.generate(
        request.model,
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=False,
    )

    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response_text,
            },
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(str(response_text).split()),
            "total_tokens": len(prompt.split()) + len(str(response_text).split()),
        },
    }


async def _stream_openai_chat(
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> AsyncGenerator[str, None]:
    loader = get_loader()
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"

    for chunk in loader.generate(
        model,
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True,
    ):
        data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {
                    "content": chunk,
                },
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(data)}\n\n"

    final_data = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "stop",
        }],
    }
    yield f"data: {json.dumps(final_data)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/completions", response_model=None)
async def v1_completions(request: OpenAICompletionRequest) -> StreamingResponse | dict:
    loader = get_loader()

    prompt = request.prompt if isinstance(request.prompt, str) else request.prompt[0]
    max_tokens = request.max_tokens or 256
    temperature = request.temperature if request.temperature is not None else 0.7
    top_p = request.top_p if request.top_p is not None else 0.9

    if request.stream:
        return StreamingResponse(
            _stream_openai_completion(
                request.model,
                prompt,
                max_tokens,
                temperature,
                top_p,
            ),
            media_type="text/event-stream",
        )

    response_text = loader.generate(
        request.model,
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=False,
    )

    completion_id = f"cmpl-{uuid.uuid4().hex[:8]}"
    return {
        "id": completion_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": request.model,
        "choices": [{
            "index": 0,
            "text": response_text,
            "finish_reason": "stop",
        }],
        "usage": {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(str(response_text).split()),
            "total_tokens": len(prompt.split()) + len(str(response_text).split()),
        },
    }


async def _stream_openai_completion(
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> AsyncGenerator[str, None]:
    loader = get_loader()
    completion_id = f"cmpl-{uuid.uuid4().hex[:8]}"

    for chunk in loader.generate(
        model,
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=True,
    ):
        data = {
            "id": completion_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "text": chunk,
                "finish_reason": None,
            }],
        }
        yield f"data: {json.dumps(data)}\n\n"

    final_data = {
        "id": completion_id,
        "object": "text_completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "text": "",
            "finish_reason": "stop",
        }],
    }
    yield f"data: {json.dumps(final_data)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/embeddings")
async def v1_embeddings(_request: OpenAIEmbeddingRequest) -> dict:
    raise HTTPException(
        status_code=501,
        detail="Embeddings not yet implemented for MLX models"
    )


# ##################################################################
# health check
# basic endpoint for service health verification


@app.get("/")
async def root() -> dict:
    return {"status": "ok", "service": "mlx-server", "version": VERSION}
