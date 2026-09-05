class ApplicationError(Exception):
    """Base exception for expected application errors."""


class ExternalServiceError(ApplicationError):
    """Raised when an external dependency is unavailable."""


class DatabaseError(ExternalServiceError):
    """Raised when the database cannot complete an operation."""


class CacheError(ExternalServiceError):
    """Raised when the cache cannot complete an operation."""


class LLMProviderError(ExternalServiceError):
    """Raised when the LLM provider cannot complete a request."""