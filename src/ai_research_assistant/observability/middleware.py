import logging
import time
import uuid

from fastapi import Request

from ai_research_assistant.observability.context import request_id_var


logger = logging.getLogger("ai_research_assistant.http")


async def request_logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

    request.state.request_id = request_id

    token = request_id_var.set(request_id)

    start_time = time.perf_counter()

    try:
        try:
            response = await call_next(request)
            status_code = response.status_code

        except Exception:
            status_code = 500
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            logger.exception(
                "request_failed "
                "method=%s path=%s "
                "status=%s latency_ms=%.2f",
                request.method,
                request.url.path,
                status_code,
                elapsed_ms,
            )

            raise

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        response.headers["X-Request-ID"] = request_id

        logger.info(
            "request_completed "
            "method=%s path=%s "
            "status=%s latency_ms=%.2f",
            request.method,
            request.url.path,
            status_code,
            elapsed_ms,
        )

        return response

    finally:
        request_id_var.reset(token)