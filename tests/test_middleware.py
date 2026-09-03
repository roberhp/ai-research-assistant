from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_research_assistant.observability.middleware import (
    request_logging_middleware,
)


def create_test_app():
    app = FastAPI()

    app.middleware("http")(
        request_logging_middleware
    )

    @app.get("/test")
    def test_endpoint():
        return {"status": "ok"}

    return app


def test_middleware_generates_request_id():
    app = create_test_app()

    client = TestClient(app)

    response = client.get("/test")

    assert response.status_code == 200

    request_id = response.headers.get(
        "X-Request-ID"
    )

    assert request_id is not None
    assert len(request_id) > 0


def test_middleware_preserves_existing_request_id():
    app = create_test_app()

    client = TestClient(app)

    response = client.get(
        "/test",
        headers={
            "X-Request-ID": "test-request-id"
        },
    )

    assert response.status_code == 200

    assert response.headers[
        "X-Request-ID"
    ] == "test-request-id"