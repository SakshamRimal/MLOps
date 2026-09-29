import time
from collections import defaultdict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from app.config import settings


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory sliding-window rate limiter per client IP."""

    def __init__(self, app, max_requests: int = None, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests or settings.RATE_LIMIT_PER_MINUTE
        self.window = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        now = time.time()

        # Prune old entries outside the window
        self._requests[client_ip] = [
            t for t in self._requests[client_ip] if now - t < self.window
        ]

        if len(self._requests[client_ip]) >= self.max_requests:
            retry_after = self._requests[client_ip][0] + self.window - now
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "detail": f"Rate limit of {self.max_requests} requests per minute exceeded.",
                    "retry_after": round(retry_after, 1),
                },
                headers={"Retry-After": str(int(retry_after))},
            )

        self._requests[client_ip].append(now)
        return await call_next(request)


class RequestTrackingMiddleware(BaseHTTPMiddleware):
    """Adds X-Request-ID and X-Response-Time headers to all responses."""

    async def dispatch(self, request: Request, call_next):
        import uuid
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        start = time.time()

        response = await call_next(request)
        elapsed_ms = round((time.time() - start) * 1000, 2)

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{elapsed_ms}ms"
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Catches unhandled exceptions and returns structured JSON errors."""

    async def dispatch(self, request: Request, call_next):
        try:
            return await call_next(request)
        except Exception as e:
            import uuid
            request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
            return JSONResponse(
                status_code=500,
                content={
                    "error": "internal_server_error",
                    "detail": "An unexpected error occurred. Please try again.",
                    "request_id": request_id,
                },
            )
