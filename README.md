![](banner.jpg)

# MLX Server

An Ollama API-compatible server for running MLX language models on Apple Silicon.

## Purpose

MLX Server provides a drop-in replacement for Ollama that uses Apple's MLX framework for efficient inference on M-series Macs. It exposes both Ollama-native and OpenAI-compatible API endpoints, allowing you to use existing Ollama clients and tools without modification.

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Starting the Server

```bash
./mlx serve
```

Or specify a custom port:
```bash
./mlx serve --port 8080
```

The server runs on port 11435 by default (Ollama's default port + 1).

### CLI Commands

```bash
./mlx serve           # Start the server
./mlx ps              # Show running models and memory usage
./mlx list            # Show available models
./mlx run <model>     # Load and interact with a model
./mlx pull <model>    # Download a model from HuggingFace
./mlx stop <model>    # Unload a model from memory
```

### API Examples

**Generate text (Ollama API):**
```bash
curl http://localhost:11435/api/generate -d '{
  "model": "gpt-oss-20b-16k",
  "prompt": "Explain quantum computing in simple terms"
}'
```

**Chat completion (Ollama API):**
```bash
curl http://localhost:11435/api/chat -d '{
  "model": "gpt-oss-20b-16k",
  "messages": [{"role": "user", "content": "Hello!"}]
}'
```

**Chat completion (OpenAI-compatible API):**
```bash
curl http://localhost:11435/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-oss-20b-16k",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

**List available models:**
```bash
curl http://localhost:11435/api/tags
```

**List running models:**
```bash
curl http://localhost:11435/api/ps
```

**Pull a model:**
```bash
curl http://localhost:11435/api/pull -d '{
  "name": "gpt-oss-20b-16k"
}'
```

### Development Commands

```bash
./run check    # Run linter and full test suite
./run lint     # Run linter only
./run format   # Format code
./run test <target>  # Run a specific test
```

## API Endpoints

### Ollama Native API
- `POST /api/generate` - Generate text completion
- `POST /api/chat` - Chat completion
- `GET /api/tags` - List available models
- `GET /api/ps` - List running models
- `POST /api/show` - Show model details
- `POST /api/pull` - Pull model from HuggingFace
- `DELETE /api/delete` - Unload model
- `GET /api/version` - Server version

### OpenAI Compatible API
- `GET /v1/models` - List models
- `GET /v1/models/{model}` - Get model info
- `POST /v1/chat/completions` - Chat completions
- `POST /v1/completions` - Text completions

## License

This project is licensed under [CC BY-NC 4.0](https://darren-static.waft.dev/license) - free to use and modify, but no commercial use without permission.
