from fastapi import Request

import pytest

from ai_research_assistant.exceptions import ApplicationError
from ai_research_assistant.main import (
    application_error_handler,
    unexpected_error_handler,
)


def create_request():
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/test",
        "headers": [],
    }

    return Request(scope)


@pytest.mark.anyio
async def test_application_error_handler_returns_503():
    request = create_request()

    response = await application_error_handler(
        request,
        ApplicationError("Dependency unavailable"),
    )

    assert response.status_code == 503


@pytest.mark.anyio
async def test_unexpected_error_handler_returns_500():
    request = create_request()

    response = await unexpected_error_handler(
        request,
        RuntimeError("Unexpected failure"),
    )

    assert response.status_code == 500