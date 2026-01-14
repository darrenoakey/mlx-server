import time
import threading
from dataclasses import dataclass, field
from typing import Any, Generator
import psutil


# ##################################################################
# model info
# holds metadata about a model in the registry
@dataclass
class ModelInfo:
    name: str
    hf_path: str
    context_length: int = 4096
    family: str = "unknown"
    parameter_size: str = "unknown"
    quantization: str = "unknown"


# ##################################################################
# loaded model
# represents a model that is currently loaded in memory
@dataclass
class LoadedModel:
    info: ModelInfo
    model: Any
    tokenizer: Any
    loaded_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)
    keep_alive_until: float = field(default_factory=lambda: time.time() + 300)
    size_bytes: int = 0


# ##################################################################
# model registry
# manages available and loaded models
class ModelRegistry:
    # default model aliases mapped to huggingface paths
    DEFAULT_MODELS: dict[str, ModelInfo] = {
        "gpt-oss-20b-16k": ModelInfo(
            name="gpt-oss-20b-16k",
            hf_path="mlx-community/gpt-oss-20b-16k",
            context_length=16384,
            family="gpt-oss",
            parameter_size="20B",
            quantization="4bit",
        ),
    }

    def __init__(self) -> None:
        self._available: dict[str, ModelInfo] = dict(self.DEFAULT_MODELS)
        self._loaded: dict[str, LoadedModel] = {}
        self._lock = threading.Lock()

    # ##################################################################
    # register model
    # adds a model to the available registry
    def register(self, info: ModelInfo) -> None:
        with self._lock:
            self._available[info.name] = info

    # ##################################################################
    # get available
    # returns list of available models
    def get_available(self) -> list[ModelInfo]:
        with self._lock:
            return list(self._available.values())

    # ##################################################################
    # get loaded
    # returns list of currently loaded models
    def get_loaded(self) -> list[LoadedModel]:
        with self._lock:
            return list(self._loaded.values())

    # ##################################################################
    # get model info
    # retrieves info for a model by name
    def get_info(self, name: str) -> ModelInfo | None:
        with self._lock:
            return self._available.get(name)

    # ##################################################################
    # resolve model path
    # converts model name to huggingface path
    def resolve_path(self, name: str) -> str:
        with self._lock:
            if name in self._available:
                return self._available[name].hf_path
            return name

    # ##################################################################
    # is loaded
    # checks if model is currently loaded
    def is_loaded(self, name: str) -> bool:
        with self._lock:
            return name in self._loaded

    # ##################################################################
    # get loaded model
    # retrieves a loaded model instance
    def get_loaded_model(self, name: str) -> LoadedModel | None:
        with self._lock:
            model = self._loaded.get(name)
            if model:
                model.last_used = time.time()
            return model

    # ##################################################################
    # set loaded
    # stores a loaded model in the registry
    def set_loaded(self, name: str, loaded: LoadedModel) -> None:
        with self._lock:
            self._loaded[name] = loaded

    # ##################################################################
    # unload
    # removes a model from loaded models
    def unload(self, name: str) -> bool:
        with self._lock:
            if name in self._loaded:
                del self._loaded[name]
                return True
            return False

    # ##################################################################
    # get memory usage
    # returns current process memory usage in bytes
    @staticmethod
    def get_memory_usage() -> int:
        process = psutil.Process()
        return process.memory_info().rss


# ##################################################################
# model loader
# handles loading and unloading mlx models
class ModelLoader:
    def __init__(self, registry: ModelRegistry) -> None:
        self._registry = registry
        self._load_lock = threading.Lock()

    # ##################################################################
    # load model
    # loads a model into memory using mlx_lm
    def load(self, name: str, keep_alive: float = 300.0) -> LoadedModel:
        with self._load_lock:
            existing = self._registry.get_loaded_model(name)
            if existing:
                existing.keep_alive_until = time.time() + keep_alive
                return existing

            hf_path = self._registry.resolve_path(name)
            info = self._registry.get_info(name)
            if not info:
                info = ModelInfo(name=name, hf_path=hf_path)
                self._registry.register(info)

            from mlx_lm import load as mlx_load
            result = mlx_load(hf_path)
            model = result[0]
            tokenizer = result[1]

            loaded = LoadedModel(
                info=info,
                model=model,
                tokenizer=tokenizer,
                keep_alive_until=time.time() + keep_alive,
                size_bytes=self._registry.get_memory_usage(),
            )
            self._registry.set_loaded(name, loaded)
            return loaded

    # ##################################################################
    # unload model
    # removes model from memory
    def unload(self, name: str) -> bool:
        with self._load_lock:
            import gc
            result = self._registry.unload(name)
            if result:
                gc.collect()
            return result

    # ##################################################################
    # generate
    # generates text using loaded model
    def generate(
        self,
        name: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stream: bool = False,
    ) -> str | Generator[str, None, None]:
        loaded = self.load(name)
        loaded.last_used = time.time()

        from mlx_lm import generate as mlx_generate

        if stream:
            return self._stream_generate(loaded, prompt, max_tokens, temperature, top_p)
        else:
            return mlx_generate(
                loaded.model,
                loaded.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temp=temperature,
                top_p=top_p,
            )

    # ##################################################################
    # stream generate
    # yields generated tokens one at a time
    def _stream_generate(
        self,
        loaded: LoadedModel,
        prompt: str,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Generator[str, None, None]:
        from mlx_lm import stream_generate

        for response in stream_generate(
            loaded.model,
            loaded.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temperature,
            top_p=top_p,
        ):
            yield response.text


# ##################################################################
# global registry instance
# singleton for use across the application
_registry: ModelRegistry | None = None
_loader: ModelLoader | None = None


# ##################################################################
# get registry
# returns or creates the global registry
def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


# ##################################################################
# get loader
# returns or creates the global loader
def get_loader() -> ModelLoader:
    global _loader
    if _loader is None:
        _loader = ModelLoader(get_registry())
    return _loader
