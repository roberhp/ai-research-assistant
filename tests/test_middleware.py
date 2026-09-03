import logging

from fastapi import FastAPI
from fastapi.testclient import TestClient

from ai_research_assistant.observability.middleware import (
    request_logging_middleware,
)


def create_test_app() -> FastAPI:
    app = FastAPI()
    app.middleware("http")(request_logging_middleware)

    @app.get("/test")
    def test_endpoint():
        return {"status": "ok"}

    @app.get("/error")
    def error_endpoint():
        raise RuntimeError("test error")

    return app


def test_request_id_is_preserved():
    client = TestClient(create_test_app())

    response = client.get(
        "/test",
        headers={"X-Request-ID": "test-request-id"},
    )

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-request-id"


def test_request_id_is_generated():
    client = TestClient(create_test_app())

    response = client.get("/test")

    assert response.status_code == 200

    request_id = response.headers["X-Request-ID"]

    assert request_id
    assert len(request_id) > 0


def test_request_id_is_included_in_logs(caplog):
    client = TestClient(create_test_app())

    with caplog.at_level(
        logging.INFO,
        logger="ai_research_assistant.http",
    ):
        response = client.get(
            "/test",
            headers={"X-Request-ID": "test-request-id"},
        )

    assert response.status_code == 200

    records = [
        record
        for record in caplog.records
        if record.name == "ai_research_assistant.http"
    ]

    assert records
    assert records[-1].request_id == "test-request-id"


def test_failed_request_is_logged(caplog):
    client = TestClient(create_test_app())

    with caplog.at_level(
        logging.ERROR,
        logger="ai_research_assistant.http",
    ):
        try:
            client.get(
                "/error",
                headers={"X-Request-ID": "test-request-id"},
            )
        except RuntimeError:
            pass

    records = [
        record
        for record in caplog.records
        if record.name == "ai_research_assistant.http"
    ]

    assert records
    assert records[-1].request_id == "test-request-id"
    assert records[-1].message.startswith("request_failed")