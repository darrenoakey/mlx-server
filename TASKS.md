# MLX Server - Ollama-Compatible Server for MLX Models

## Project Goal
Build an Ollama API-compatible server that uses mlx-lm for inference on Apple Silicon,
with a CLI tool (`mlx`) for management commands.

**Default Port**: 11435 (Ollama default 11434 + 1)
**Default Model**: gpt-oss-20b-16k (GPT-OSS 20B with 16384 context length)

---

## Task List

### Phase 1: Project Setup
- [x] 1.1 Create project structure (src/, output/, local/, run file)
- [x] 1.2 Create requirements.txt with dependencies (mlx-lm, fastapi, uvicorn, httpx, psutil)
- [x] 1.3 Create .gitignore
- [x] 1.4 Initialize basic run file with argparse

### Phase 2: Core Server Infrastructure
- [x] 2.1 Create server module with FastAPI app skeleton
- [x] 2.2 Implement model registry (track loaded models, memory usage)
- [x] 2.3 Implement model loader using mlx_lm.load()
- [x] 2.4 Implement model unloader with memory cleanup

### Phase 3: Ollama Native API Endpoints
- [x] 3.1 GET /api/tags - List available models
- [x] 3.2 GET /api/ps - List running models with memory usage
- [x] 3.3 POST /api/show - Show model details
- [x] 3.4 POST /api/generate - Generate completion (streaming + non-streaming)
- [x] 3.5 POST /api/chat - Chat completion (streaming + non-streaming)
- [x] 3.6 POST /api/embed - Generate embeddings (returns 501 - not implemented)
- [x] 3.7 POST /api/pull - Pull model from HuggingFace (map to mlx-community)
- [x] 3.8 DELETE /api/delete - Unload/remove model
- [x] 3.9 GET /api/version - Return server version

### Phase 4: OpenAI Compatibility Endpoints
- [x] 4.1 GET /v1/models - List models (OpenAI format)
- [x] 4.2 GET /v1/models/{model} - Get model info
- [x] 4.3 POST /v1/chat/completions - Chat completions (streaming + non-streaming)
- [x] 4.4 POST /v1/completions - Text completions
- [x] 4.5 POST /v1/embeddings - Embeddings (returns 501 - not implemented)

### Phase 5: CLI Tool (mlx command)
- [x] 5.1 Create cli module with subcommands
- [x] 5.2 `mlx serve` - Start the server
- [x] 5.3 `mlx ps` - Show running models and memory (calls GET /api/ps)
- [x] 5.4 `mlx list` - Show available models (calls GET /api/tags)
- [x] 5.5 `mlx run <model>` - Load and interact with model
- [x] 5.6 `mlx pull <model>` - Download model
- [x] 5.7 `mlx stop <model>` - Unload model

### Phase 6: Model Configuration
- [x] 6.1 Define model registry with aliases (gpt-oss-20b-16k -> actual HF path)
- [x] 6.2 Implement context length configuration per model
- [x] 6.3 Implement keep_alive timeout handling

### Phase 7: Testing
- [x] 7.1 Write tests for model registry
- [x] 7.2 Write tests for API endpoints
- [x] 7.3 Write tests for CLI commands
- [ ] 7.4 Integration tests with actual model loading (requires GPU/model download)

### Phase 8: Final Integration
- [x] 8.1 Wire up default model (gpt-oss-20b-16k)
- [ ] 8.2 Test full compatibility with Ollama clients
- [ ] 8.3 Final verification and cleanup

---

## API Reference (Ollama Compatibility Target)

### Native Endpoints (Port 11435)
| Endpoint | Method | Description |
|----------|--------|-------------|
| /api/generate | POST | Generate completion |
| /api/chat | POST | Chat completion |
| /api/tags | GET | List available models |
| /api/ps | GET | List running models |
| /api/show | POST | Show model info |
| /api/embed | POST | Generate embeddings |
| /api/pull | POST | Pull model |
| /api/delete | DELETE | Delete/unload model |
| /api/version | GET | Server version |

### OpenAI Compatible Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| /v1/models | GET | List models |
| /v1/models/{model} | GET | Get model info |
| /v1/chat/completions | POST | Chat completions |
| /v1/completions | POST | Text completions |
| /v1/embeddings | POST | Embeddings |

---

## Model Registry

```
gpt-oss-20b-16k: mlx-community/gpt-oss-20b-16k (default)
```

Additional models can be added to the registry or pulled directly from HuggingFace.

---

## Usage

### Start the server
```bash
./mlx serve
# or
./mlx serve --port 11435
```

### CLI Commands
```bash
./mlx ps          # Show running models
./mlx list        # Show available models
./mlx run <model> # Interactive session with model
./mlx pull <model> # Download a model
./mlx stop <model> # Unload a model
```

### API Usage (curl examples)
```bash
# Generate text
curl http://localhost:11435/api/generate -d '{
  "model": "gpt-oss-20b-16k",
  "prompt": "Hello, world!"
}'

# Chat
curl http://localhost:11435/api/chat -d '{
  "model": "gpt-oss-20b-16k",
  "messages": [{"role": "user", "content": "Hello!"}]
}'

# OpenAI compatible
curl http://localhost:11435/v1/chat/completions -d '{
  "model": "gpt-oss-20b-16k",
  "messages": [{"role": "user", "content": "Hello!"}]
}'
```

---

## Sources
- [MLX-LM GitHub](https://github.com/ml-explore/mlx-lm)
- [Ollama API Documentation](https://docs.ollama.com/api/introduction)
- [Ollama OpenAI Compatibility](https://docs.ollama.com/api/openai-compatibility)
