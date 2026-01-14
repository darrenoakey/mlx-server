from fastapi.testclient import TestClient
from src.server import app, parse_keep_alive, get_options, now_iso


client = TestClient(app)


# ##################################################################
# test root endpoint
# verifies health check returns ok status
def test_root_endpoint() -> None:
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "mlx-server"
    assert "version" in data


# ##################################################################
# test api version
# verifies version endpoint returns version info
def test_api_version() -> None:
    response = client.get("/api/version")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data


# ##################################################################
# test api tags
# verifies tags endpoint returns available models
def test_api_tags() -> None:
    response = client.get("/api/tags")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)
    assert len(data["models"]) >= 1
    model = data["models"][0]
    assert "name" in model
    assert "details" in model


# ##################################################################
# test api ps empty
# verifies ps returns empty list when no models loaded
def test_api_ps_empty() -> None:
    response = client.get("/api/ps")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)


# ##################################################################
# test api show
# verifies show returns model info for known model
def test_api_show() -> None:
    response = client.post("/api/show", json={"model": "gpt-oss-20b-16k"})
    assert response.status_code == 200
    data = response.json()
    assert "modelfile" in data
    assert "details" in data


# ##################################################################
# test api show not found
# verifies show returns 404 for unknown model
def test_api_show_not_found() -> None:
    response = client.post("/api/show", json={"model": "nonexistent-model"})
    assert response.status_code == 404


# ##################################################################
# test v1 models
# verifies openai compatible models endpoint
def test_v1_models() -> None:
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert "data" in data
    assert isinstance(data["data"], list)
    if data["data"]:
        model = data["data"][0]
        assert "id" in model
        assert model["object"] == "model"


# ##################################################################
# test v1 model info
# verifies openai compatible model info endpoint
def test_v1_model_info() -> None:
    response = client.get("/v1/models/gpt-oss-20b-16k")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == "gpt-oss-20b-16k"
    assert data["object"] == "model"


# ##################################################################
# test v1 model info not found
# verifies model info returns 404 for unknown model
def test_v1_model_info_not_found() -> None:
    response = client.get("/v1/models/nonexistent-model")
    assert response.status_code == 404


# ##################################################################
# test parse keep alive minutes
# verifies keep alive parsing for minute format
def test_parse_keep_alive_minutes() -> None:
    result = parse_keep_alive("5m")
    assert result == 300.0


# ##################################################################
# test parse keep alive hours
# verifies keep alive parsing for hour format
def test_parse_keep_alive_hours() -> None:
    result = parse_keep_alive("2h")
    assert result == 7200.0


# ##################################################################
# test parse keep alive seconds
# verifies keep alive parsing for seconds format
def test_parse_keep_alive_seconds() -> None:
    result = parse_keep_alive("30s")
    assert result == 30.0


# ##################################################################
# test parse keep alive integer
# verifies keep alive parsing for integer
def test_parse_keep_alive_integer() -> None:
    result = parse_keep_alive(120)
    assert result == 120.0


# ##################################################################
# test parse keep alive none
# verifies keep alive parsing returns default for none
def test_parse_keep_alive_none() -> None:
    result = parse_keep_alive(None)
    assert result == 300.0


# ##################################################################
# test get options default
# verifies default options are returned
def test_get_options_default() -> None:
    result = get_options(None)
    assert "temperature" in result
    assert "top_p" in result
    assert "max_tokens" in result


# ##################################################################
# test get options override
# verifies options can be overridden
def test_get_options_override() -> None:
    result = get_options({"temperature": 0.5, "top_p": 0.8})
    assert result["temperature"] == 0.5
    assert result["top_p"] == 0.8


# ##################################################################
# test get options num predict
# verifies num_predict maps to max_tokens
def test_get_options_num_predict() -> None:
    result = get_options({"num_predict": 512})
    assert result["max_tokens"] == 512


# ##################################################################
# test now iso format
# verifies iso timestamp is returned in correct format
def test_now_iso_format() -> None:
    result = now_iso()
    assert "T" in result
    assert result.endswith("Z")


# ##################################################################
# test api embed not implemented
# verifies embed endpoint returns 501
def test_api_embed_not_implemented() -> None:
    response = client.post("/api/embed", json={"model": "test", "input": "test"})
    assert response.status_code == 501


# ##################################################################
# test v1 embeddings not implemented
# verifies openai embeddings endpoint returns 501
def test_v1_embeddings_not_implemented() -> None:
    response = client.post("/v1/embeddings", json={"model": "test", "input": "test"})
    assert response.status_code == 501


# ##################################################################
# test api delete not loaded
# verifies delete returns 404 for non-loaded model
def test_api_delete_not_loaded() -> None:
    response = client.request("DELETE", "/api/delete", json={"model": "not-loaded"})
    assert response.status_code == 404
