"""
Security configuration for Quantum Dengue STPP API.

Provides security best practices for production deployment.
"""
import os
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class SecurityConfig:
    """Security configuration settings."""
    
    # CORS settings
    cors_origins: List[str] = None
    
    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds
    
    # Authentication
    api_key_enabled: bool = False
    jwt_enabled: bool = False
    
    # API security
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    request_timeout: int = 30
    
    # Headers
    security_headers: dict = None
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = ["*"]
        
        if self.security_headers is None:
            self.security_headers = {
                "X-Content-Type-Options": "nosniff",
                "X-Frame-Options": "DENY",
                "X-XSS-Protection": "1; mode=block",
                "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
                "Content-Security-Policy": "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'",
            }


# Environment-based configuration
def get_security_config() -> SecurityConfig:
    """Get security configuration from environment variables."""
    return SecurityConfig(
        cors_origins=os.getenv("CORS_ORIGINS", "*").split(","),
        rate_limit_requests=int(os.getenv("RATE_LIMIT_REQUESTS", "100")),
        rate_limit_window=int(os.getenv("RATE_LIMIT_WINDOW", "60")),
        api_key_enabled=os.getenv("API_KEY_ENABLED", "false").lower() == "true",
        jwt_enabled=os.getenv("JWT_ENABLED", "false").lower() == "true",
        max_request_size=int(os.getenv("MAX_REQUEST_SIZE", str(10 * 1024 * 1024))),
        request_timeout=int(os.getenv("REQUEST_TIMEOUT", "30")),
    )


# Rate limiter implementation
class RateLimiter:
    """
    Simple in-memory rate limiter.
    
    For production, use Redis-based rate limiting.
    """
    
    def __init__(self, requests: int = 100, window: int = 60):
        self.requests = requests
        self.window = window
        self._clients: dict = {}
    
    def is_allowed(self, client_id: str) -> bool:
        """Check if request is allowed."""
        import time
        
        current_time = time.time()
        
        if client_id not in self._clients:
            self._clients[client_id] = []
        
        # Remove expired requests
        self._clients[client_id] = [
            t for t in self._clients[client_id]
            if current_time - t < self.window
        ]
        
        # Check limit
        if len(self._clients[client_id]) >= self.requests:
            return False
        
        # Record request
        self._clients[client_id].append(current_time)
        return True
    
    def get_remaining(self, client_id: str) -> int:
        """Get remaining requests for client."""
        import time
        
        current_time = time.time()
        
        if client_id not in self._clients:
            return self.requests
        
        active_requests = [
            t for t in self._clients[client_id]
            if current_time - t < self.window
        ]
        
        return max(0, self.requests - len(active_requests))


# API Key validator
class APIKeyValidator:
    """
    Simple API key validation.
    
    For production, use proper key management (AWS Secrets Manager, HashiCorp Vault, etc.)
    """
    
    def __init__(self, valid_keys: Optional[List[str]] = None):
        self.valid_keys = set(valid_keys or [])
        self._env_keys = os.getenv("API_KEYS", "").split(",")
        for key in self._env_keys:
            if key:
                self.valid_keys.add(key.strip())
    
    def is_valid(self, api_key: str) -> bool:
        """Validate API key."""
        return api_key in self.valid_keys
    
    def add_key(self, key: str):
        """Add a valid API key."""
        self.valid_keys.add(key)
    
    def remove_key(self, key: str):
        """Remove an API key."""
        self.valid_keys.discard(key)


# Security middleware for FastAPI
def setup_security_middleware(app, config: Optional[SecurityConfig] = None):
    """
    Set up security middleware for FastAPI application.
    
    Args:
        app: FastAPI application instance
        config: Security configuration
    """
    from fastapi import FastAPI, Request, HTTPException, Depends
    from fastapi.middleware.cors import CORSMiddleware
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import Response
    import time
    
    if config is None:
        config = get_security_config()
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Rate limiting middleware
    rate_limiter = RateLimiter(
        requests=config.rate_limit_requests,
        window=config.rate_limit_window
    )
    
    class RateLimitMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            client_id = request.client.host if request.client else "unknown"
            
            if not rate_limiter.is_allowed(client_id):
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Please try again later."
                )
            
            response = await call_next(request)
            
            # Add rate limit headers
            remaining = rate_limiter.get_remaining(client_id)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Limit"] = str(config.rate_limit_requests)
            
            return response
    
    app.add_middleware(RateLimitMiddleware)
    
    # Security headers middleware
    class SecurityHeadersMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            response = await call_next(request)
            
            for header, value in config.security_headers.items():
                response.headers[header] = value
            
            return response
    
    app.add_middleware(SecurityHeadersMiddleware)
    
    return app


# Input validation
def validate_coordinates(lat: float, lon: float) -> tuple[bool, str]:
    """
    Validate geographic coordinates.
    
    Returns:
        (is_valid, error_message)
    """
    import math
    
    if math.isnan(lat) or math.isnan(lon):
        return False, "Coordinates cannot be NaN"
    
    if lat < -90 or lat > 90:
        return False, f"Latitude must be between -90 and 90, got {lat}"
    
    if lon < -180 or lon > 180:
        return False, f"Longitude must be between -180 and 180, got {lon}"
    
    return True, ""


def sanitize_input(value: str, max_length: int = 1000) -> str:
    """
    Sanitize user input to prevent injection attacks.
    
    Args:
        value: Input string
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
    """
    import html
    
    # Escape HTML entities
    value = html.escape(value)
    
    # Remove control characters
    value = ''.join(char for char in value if ord(char) >= 32 or char in '\n\r\t')
    
    # Truncate
    return value[:max_length]
