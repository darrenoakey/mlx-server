from src.models import ModelInfo, ModelRegistry, get_registry


# ##################################################################
# test model info creation
# verifies model info dataclass can be created with defaults
def test_model_info_creation() -> None:
    info = ModelInfo(name="test-model", hf_path="test/model")
    assert info.name == "test-model"
    assert info.hf_path == "test/model"
    assert info.context_length == 4096
    assert info.family == "unknown"


# ##################################################################
# test model info with all fields
# verifies model info can be created with all fields specified
def test_model_info_all_fields() -> None:
    info = ModelInfo(
        name="custom-model",
        hf_path="org/custom-model",
        context_length=8192,
        family="llama",
        parameter_size="7B",
        quantization="4bit",
    )
    assert info.name == "custom-model"
    assert info.hf_path == "org/custom-model"
    assert info.context_length == 8192
    assert info.family == "llama"
    assert info.parameter_size == "7B"
    assert info.quantization == "4bit"


# ##################################################################
# test registry initialization
# verifies registry starts with default models
def test_registry_initialization() -> None:
    registry = ModelRegistry()
    available = registry.get_available()
    assert len(available) >= 1
    names = [m.name for m in available]
    assert "gpt-oss-20b-16k" in names


# ##################################################################
# test registry register
# verifies new models can be registered
def test_registry_register() -> None:
    registry = ModelRegistry()
    info = ModelInfo(name="new-model", hf_path="test/new-model")
    registry.register(info)
    retrieved = registry.get_info("new-model")
    assert retrieved is not None
    assert retrieved.name == "new-model"
    assert retrieved.hf_path == "test/new-model"


# ##################################################################
# test registry resolve path
# verifies model paths are resolved correctly
def test_registry_resolve_path() -> None:
    registry = ModelRegistry()
    path = registry.resolve_path("gpt-oss-20b-16k")
    assert path == "mlx-community/gpt-oss-20b-16k"

    unknown_path = registry.resolve_path("unknown-model")
    assert unknown_path == "unknown-model"


# ##################################################################
# test registry is loaded
# verifies loaded status is tracked correctly
def test_registry_is_loaded() -> None:
    registry = ModelRegistry()
    assert not registry.is_loaded("any-model")


# ##################################################################
# test registry get loaded empty
# verifies empty list when no models loaded
def test_registry_get_loaded_empty() -> None:
    registry = ModelRegistry()
    loaded = registry.get_loaded()
    assert loaded == []


# ##################################################################
# test get registry singleton
# verifies global registry returns same instance
def test_get_registry_singleton() -> None:
    r1 = get_registry()
    r2 = get_registry()
    assert r1 is r2


# ##################################################################
# test registry unload not loaded
# verifies unload returns false for non-loaded model
def test_registry_unload_not_loaded() -> None:
    registry = ModelRegistry()
    result = registry.unload("not-loaded-model")
    assert result is False


# ##################################################################
# test registry memory usage
# verifies memory usage returns a positive integer
def test_registry_memory_usage() -> None:
    memory = ModelRegistry.get_memory_usage()
    assert isinstance(memory, int)
    assert memory > 0
