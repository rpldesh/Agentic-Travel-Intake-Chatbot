class TravelBotError(Exception):
    """Base application error."""


class SessionNotFoundError(TravelBotError):
    pass


class SessionUnauthorizedError(TravelBotError):
    pass


class SessionExpiredError(TravelBotError):
    pass


class ConcurrencyConflictError(TravelBotError):
    """The session's row_version changed between read and write."""


class PersistenceError(TravelBotError):
    """A write to the entity store failed after retries."""


class RequestValidationError(TravelBotError):
    def __init__(self, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.reason_code = reason_code


class RateLimitedError(TravelBotError):
    pass


class ModelConfigurationError(TravelBotError):
    pass
