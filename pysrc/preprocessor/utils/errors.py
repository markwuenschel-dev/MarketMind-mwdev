class OOMRetry(Exception):
    def __init__(self, message="OOM: Retry with smaller batch", retry_hint=None):
        super().__init__(message)
        self.retry_hint = retry_hint or {"reduce_factor": 2}


class UnsupportedAST(Exception):
    pass


class SchemaMismatch(Exception):
    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or {}

    def __str__(self):
        return f"{super().__str__()} (details: {self.details})"
