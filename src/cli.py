#!/usr/bin/env python3
import argparse
import subprocess
import sys

import httpx
import setproctitle


DEFAULT_HOST = "http://localhost:11435"


# ##################################################################
# format bytes
# converts bytes to human-readable format
def format_bytes(size: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size //= 1024
    return f"{size:.1f} PB"


# ##################################################################
# get client
# creates httpx client for server communication
def get_client(host: str) -> httpx.Client:
    return httpx.Client(base_url=host, timeout=30.0)


# ##################################################################
# check server
# verifies server is running and accessible
def check_server(host: str) -> bool:
    try:
        client = get_client(host)
        response = client.get("/")
        return response.status_code == 200
    except httpx.ConnectError:
        return False


# ##################################################################
# ps command
# shows running models and memory usage
def cmd_ps(args: argparse.Namespace) -> int:
    host = args.host or DEFAULT_HOST
    if not check_server(host):
        print(f"Error: Cannot connect to server at {host}")
        print("Is the server running? Start it with: mlx serve")
        return 1

    client = get_client(host)
    response = client.get("/api/ps")
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return 1

    data = response.json()
    models = data.get("models", [])

    if not models:
        print("No models currently loaded")
        return 0

    print(f"{'NAME':<30} {'SIZE':<12} {'EXPIRES':<25}")
    print("-" * 70)
    for model in models:
        name = model.get("name", "unknown")
        size = format_bytes(model.get("size_vram", 0))
        expires = model.get("expires_at", "never")
        print(f"{name:<30} {size:<12} {expires:<25}")

    return 0


# ##################################################################
# list command
# shows available models
def cmd_list(args: argparse.Namespace) -> int:
    host = args.host or DEFAULT_HOST
    if not check_server(host):
        print(f"Error: Cannot connect to server at {host}")
        print("Is the server running? Start it with: mlx serve")
        return 1

    client = get_client(host)
    response = client.get("/api/tags")
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return 1

    data = response.json()
    models = data.get("models", [])

    if not models:
        print("No models available")
        return 0

    print(f"{'NAME':<30} {'FAMILY':<15} {'SIZE':<10} {'QUANTIZATION':<12}")
    print("-" * 70)
    for model in models:
        name = model.get("name", "unknown")
        details = model.get("details", {})
        family = details.get("family", "unknown")
        param_size = details.get("parameter_size", "unknown")
        quant = details.get("quantization_level", "unknown")
        print(f"{name:<30} {family:<15} {param_size:<10} {quant:<12}")

    return 0


# ##################################################################
# serve command
# starts the mlx server
def cmd_serve(args: argparse.Namespace) -> int:
    setproctitle.setproctitle("mlx-server")
    port = args.port
    return subprocess.call([
        "uvicorn",
        "src.server:app",
        "--host", "0.0.0.0",
        "--port", str(port),
    ])


# ##################################################################
# run command
# loads model and starts interactive session
def cmd_run(args: argparse.Namespace) -> int:
    host = args.host or DEFAULT_HOST
    if not check_server(host):
        print(f"Error: Cannot connect to server at {host}")
        print("Is the server running? Start it with: mlx serve")
        return 1

    model = args.model
    client = get_client(host)

    print(f"Loading {model}...")

    while True:
        try:
            prompt = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            return 0

        if not prompt.strip():
            continue

        if prompt.strip().lower() in ["/bye", "/exit", "/quit"]:
            print("Goodbye!")
            return 0

        try:
            with client.stream(
                "POST",
                "/api/generate",
                json={"model": model, "prompt": prompt, "stream": True},
                timeout=300.0,
            ) as response:
                for line in response.iter_lines():
                    if line:
                        import json
                        data = json.loads(line)
                        text = data.get("response", "")
                        print(text, end="", flush=True)
                print()
        except httpx.TimeoutException:
            print("\nError: Request timed out")
        except Exception as e:
            print(f"\nError: {e}")


# ##################################################################
# pull command
# downloads a model from huggingface
def cmd_pull(args: argparse.Namespace) -> int:
    host = args.host or DEFAULT_HOST
    if not check_server(host):
        print(f"Error: Cannot connect to server at {host}")
        print("Is the server running? Start it with: mlx serve")
        return 1

    model = args.model
    client = get_client(host)

    print(f"Pulling {model}...")

    try:
        with client.stream(
            "POST",
            "/api/pull",
            json={"model": model, "stream": True},
            timeout=600.0,
        ) as response:
            import json
            for line in response.iter_lines():
                if line:
                    data = json.loads(line)
                    status = data.get("status", "")
                    print(status)
                    if "error" in data:
                        print(f"Error: {data['error']}")
                        return 1
    except httpx.TimeoutException:
        print("Error: Request timed out")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


# ##################################################################
# stop command
# unloads a model from memory
def cmd_stop(args: argparse.Namespace) -> int:
    host = args.host or DEFAULT_HOST
    if not check_server(host):
        print(f"Error: Cannot connect to server at {host}")
        print("Is the server running? Start it with: mlx serve")
        return 1

    model = args.model
    client = get_client(host)

    response = client.request("DELETE", "/api/delete", json={"model": model})
    if response.status_code == 200:
        print(f"Model {model} unloaded")
        return 0
    elif response.status_code == 404:
        print(f"Model {model} not found or not loaded")
        return 1
    else:
        print(f"Error: {response.status_code}")
        return 1


# ##################################################################
# main
# parses arguments and dispatches to appropriate command
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="mlx",
        description="MLX Server CLI - Ollama-compatible server for MLX models",
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_HOST,
        help=f"Server host (default: {DEFAULT_HOST})",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_serve = sub.add_parser("serve", help="Start the MLX server")
    p_serve.add_argument(
        "--port", "-p",
        type=int,
        default=11435,
        help="Port to listen on (default: 11435)",
    )
    p_serve.set_defaults(func=cmd_serve)

    p_ps = sub.add_parser("ps", help="Show running models and memory usage")
    p_ps.set_defaults(func=cmd_ps)

    p_list = sub.add_parser("list", help="Show available models")
    p_list.set_defaults(func=cmd_list)

    p_run = sub.add_parser("run", help="Load and interact with a model")
    p_run.add_argument("model", help="Model name to run")
    p_run.set_defaults(func=cmd_run)

    p_pull = sub.add_parser("pull", help="Download a model")
    p_pull.add_argument("model", help="Model name to pull")
    p_pull.set_defaults(func=cmd_pull)

    p_stop = sub.add_parser("stop", help="Unload a model from memory")
    p_stop.add_argument("model", help="Model name to stop")
    p_stop.set_defaults(func=cmd_stop)

    args = parser.parse_args(argv)
    return args.func(args)


# ##################################################################
# entry point
# standard python dispatch
if __name__ == "__main__":
    sys.exit(main())
