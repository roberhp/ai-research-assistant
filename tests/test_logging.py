import logging

from ai_research_assistant.observability.logging import (
    configure_logging,
)


def test_configure_logging():
    configure_logging()

    logger = logging.getLogger(
        "ai_research_assistant"
    )

    assert logger.isEnabledFor(
        logging.INFO
    )