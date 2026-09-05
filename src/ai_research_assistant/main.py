import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ai_research_assistant.api.documents import router as documents_router
from ai_research_assistant.api.health import router as health_router
from ai_research_assistant.api.rag import router as rag_router
from ai_research_assistant.api.search import router as search_router
from ai_research_assistant.exceptions import ApplicationError
from ai_research_assistant.observability.logging import configure_logging
from ai_research_assistant.observability.middleware import (
    request_logging_middleware,
)

configure_logging()

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Research Assistant",
    version="1.0.0",
)


@app.exception_handler(ApplicationError)
async def application_error_handler(
    request: Request,
    exc: ApplicationError,
):
    logger.warning(
        "application_error "
        "method=%s "
        "path=%s "
        "error=%s",
        request.method,
        request.url.path,
        type(exc).__name__,
    )

    return JSONResponse(
        status_code=503,
        content={
            "error": "service_unavailable",
            "message": "A required service is currently unavailable.",
        },
    )


@app.exception_handler(Exception)
async def unexpected_error_handler(
    request: Request,
    exc: Exception,
):
    logger.exception(
        "unexpected_application_error "
        "method=%s "
        "path=%s",
        request.method,
        request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "message": "An unexpected error occurred.",
        },
    )


app.middleware("http")(request_logging_middleware)

app.include_router(documents_router)
app.include_router(search_router)
app.include_router(rag_router)
app.include_router(health_router)


@app.get("/health")
def health():
    return {"status": "ok"}