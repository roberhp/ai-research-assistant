from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from ai_research_assistant.dependencies import get_db
from ai_research_assistant.settings import Settings


router = APIRouter(
    prefix="/health",
    tags=["health"],
)


@router.get("/live")
def liveness():
    return {"status": "ok"}


@router.get("/ready")
def readiness(
    db: Session = Depends(get_db),
):
    checks = {}

    # PostgreSQL
    try:
        db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception:
        checks["database"] = "error"

    # Redis
    settings = Settings()

    try:
        import redis

        client = redis.Redis.from_url(settings.redis_url)
        client.ping()

        checks["redis"] = "ok"
    except Exception:
        checks["redis"] = "error"

    is_ready = all(
        status == "ok"
        for status in checks.values()
    )

    return {
        "status": "ready" if is_ready else "not_ready",
        "checks": checks,
    }