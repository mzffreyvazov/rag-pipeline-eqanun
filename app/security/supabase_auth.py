"""Supabase JWT authentication and verification."""
import os
import time
from typing import Dict, Any, Optional
from functools import lru_cache
import redis
import requests
from jose import jwt, jwk
from jose.exceptions import JWTError, JWKError
from fastapi import HTTPException, status

from app.config.security import security_settings


# Cache for JWKS (JSON Web Key Set) to avoid repeated network calls
_jwks_cache: Dict[str, Any] = {}
_jwks_cache_timestamp: float = 0
JWKS_CACHE_TTL = 3600  # Cache JWKs for 1 hour


@lru_cache(maxsize=1)
def get_jwks_url() -> str:
    """Get the JWKS URL for Supabase."""
    if not security_settings.supabase_url:
        raise ValueError("SUPABASE_URL is not configured")
    
    # Remove trailing slash if present
    base_url = security_settings.supabase_url.rstrip('/')
    # Supabase JWKS endpoint is at /.well-known/jwks.json (NOT /auth/v1/jwks)
    return f"{base_url}/auth/v1/.well-known/jwks.json"


def fetch_supabase_jwks(force_refresh: bool = False) -> Dict[str, Any]:
    """
    Fetch Supabase's public keys (JWKS) for JWT verification.
    
    Args:
        force_refresh: Force fetching new keys even if cache is valid
        
    Returns:
        JWKS dictionary containing public keys
        
    Raises:
        HTTPException: If unable to fetch JWKS
    """
    global _jwks_cache, _jwks_cache_timestamp
    
    current_time = time.time()
    cache_age = current_time - _jwks_cache_timestamp
    
    # Return cached JWKS if still valid
    if not force_refresh and _jwks_cache and cache_age < JWKS_CACHE_TTL:
        return _jwks_cache
    
    try:
        jwks_url = get_jwks_url()
        
        # JWKS endpoint is public and doesn't require authentication
        # but we can optionally include the apikey header for consistency
        headers = {}
        if security_settings.supabase_anon_key:
            headers["apikey"] = security_settings.supabase_anon_key
        
        response = requests.get(jwks_url, headers=headers, timeout=10)
        response.raise_for_status()
        
        jwks = response.json()
        
        # Update cache
        _jwks_cache = jwks
        _jwks_cache_timestamp = current_time
        
        return jwks
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Unable to fetch Supabase JWKS: {str(e)}"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Invalid JWKS response: {str(e)}"
        )


def get_signing_key(token: str) -> Any:
    """
    Extract the signing key from JWKS based on the token's key ID.
    
    Args:
        token: JWT token string
        
    Returns:
        Public key for verification
        
    Raises:
        HTTPException: If key ID not found or invalid
    """
    try:
        # Get the key ID from token header (without verification)
        unverified_header = jwt.get_unverified_header(token)
        key_id = unverified_header.get("kid")
        
        if not key_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing key ID (kid) in header"
            )
        
        # Fetch JWKS
        jwks = fetch_supabase_jwks()
        
        # Find matching key
        for key_data in jwks.get("keys", []):
            if key_data.get("kid") == key_id:
                try:
                    # Construct RSA public key from JWK
                    return jwk.construct(key_data)
                except JWKError as e:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Failed to construct public key: {str(e)}"
                    )
        
        # Key ID not found - try refreshing JWKS cache
        jwks = fetch_supabase_jwks(force_refresh=True)
        for key_data in jwks.get("keys", []):
            if key_data.get("kid") == key_id:
                try:
                    return jwk.construct(key_data)
                except JWKError as e:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail=f"Failed to construct public key: {str(e)}"
                    )
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Unable to find matching public key for kid: {key_id}"
        )
        
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token header: {str(e)}"
        )


def verify_supabase_token(token: str) -> Dict[str, Any]:
    """
    Verify and decode a Supabase-issued JWT token.
    
    This function supports two verification methods:
    1. HS256 (symmetric) - uses JWT secret from Supabase
    2. RS256 (asymmetric) - uses JWKS public keys
    
    The method is auto-detected from the token's algorithm header.
    
    Args:
        token: JWT token string from Authorization header
        
    Returns:
        Decoded token payload containing user information:
        {
            "sub": "user-uuid",
            "email": "user@example.com",
            "role": "authenticated",
            "aud": "authenticated",
            "exp": 1234567890,
            "iat": 1234567890,
            "app_metadata": {...},
            "user_metadata": {...}
        }
        
    Raises:
        HTTPException: If token is invalid, expired, or verification fails
    """
    try:
        # Check the algorithm used in the token
        unverified_header = jwt.get_unverified_header(token)
        algorithm = unverified_header.get("alg", "RS256")
        
        # Choose verification method based on algorithm
        if algorithm == "HS256":
            # Use JWT secret for symmetric key verification
            if not security_settings.supabase_jwt_secret and not security_settings.supabase_anon_key:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Supabase JWT secret not configured for HS256 verification"
                )
            
            # Try JWT secret first, fall back to anon key
            secret = security_settings.supabase_jwt_secret or security_settings.supabase_anon_key
            
            payload = jwt.decode(
                token,
                secret,
                algorithms=["HS256"],
                audience=security_settings.supabase_jwt_audience,
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_aud": True,
                    "require_exp": True,
                    "require_iat": True
                }
            )
        else:
            # Use JWKS for RS256/asymmetric verification
            public_key = get_signing_key(token)
            
            payload = jwt.decode(
                token,
                public_key,
                algorithms=["RS256", "RS384", "RS512"],
                audience=security_settings.supabase_jwt_audience,
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_aud": True,
                    "require_exp": True,
                    "require_iat": True
                }
            )
        
        return payload
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except jwt.JWTClaimsError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid token claims: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"}
        )
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not validate token: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"}
        )


def extract_user_info(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract and normalize user information from Supabase token payload.
    
    Args:
        payload: Decoded JWT payload from Supabase
        
    Returns:
        Normalized user information dictionary
    """
    return {
        "user_id": payload.get("sub"),
        "email": payload.get("email"),
        "role": payload.get("role", "authenticated"),
        "aud": payload.get("aud"),
        "exp": payload.get("exp"),
        "iat": payload.get("iat"),
        "app_metadata": payload.get("app_metadata", {}),
        "user_metadata": payload.get("user_metadata", {}),
        "phone": payload.get("phone"),
        "email_verified": payload.get("email_confirmed_at") is not None,
        "phone_verified": payload.get("phone_confirmed_at") is not None
    }


def validate_supabase_config() -> bool:
    """
    Validate that required Supabase configuration is present.
    
    Returns:
        True if configuration is valid, False otherwise
    """
    if not security_settings.supabase_url:
        return False
    
    # Validate URL format
    url = security_settings.supabase_url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        return False
    
    return True


# Optional: Redis-based token blacklist for logout functionality
class TokenBlacklist:
    """
    Optional token blacklist implementation for logout/revocation.
    
    Requires Redis to be installed and configured:
        pip install redis
    
    Usage:
        blacklist = TokenBlacklist()
        blacklist.add_token(token, expiry_seconds=1800)
        if blacklist.is_blacklisted(token):
            raise HTTPException(401, "Token has been revoked")
    """
    
    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize token blacklist with Redis connection.
        
        Args:
            redis_url: Redis connection URL (e.g., redis://localhost:6379/0)
                      If None, uses REDIS_URL from environment
        """
        self.redis_client = None
        self.enabled = False
        
        try:
            
            
            url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
            self.redis_client = redis.from_url(url, decode_responses=True)
            
            # Test connection
            self.redis_client.ping()
            self.enabled = True
            
        except ImportError:
            print("⚠️ Redis not installed. Token blacklist disabled.")
        except Exception as e:
            print(f"⚠️ Redis connection failed: {e}. Token blacklist disabled.")
    
    def add_token(self, token: str, expiry_seconds: int):
        """Add a token to the blacklist with expiration."""
        if not self.enabled:
            return
        
        try:
            # Store token hash to save space
            import hashlib
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            self.redis_client.setex(
                f"blacklist:{token_hash}",
                expiry_seconds,
                "1"
            )
        except Exception as e:
            print(f"⚠️ Failed to blacklist token: {e}")
    
    def is_blacklisted(self, token: str) -> bool:
        """Check if a token is blacklisted."""
        if not self.enabled:
            return False
        
        try:
            import hashlib
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            return self.redis_client.exists(f"blacklist:{token_hash}") > 0
        except Exception as e:
            print(f"⚠️ Failed to check blacklist: {e}")
            return False
    
    def remove_token(self, token: str):
        """Remove a token from the blacklist."""
        if not self.enabled:
            return
        
        try:
            import hashlib
            token_hash = hashlib.sha256(token.encode()).hexdigest()
            self.redis_client.delete(f"blacklist:{token_hash}")
        except Exception as e:
            print(f"⚠️ Failed to remove token from blacklist: {e}")


# Global blacklist instance (optional)
token_blacklist: Optional[TokenBlacklist] = None

def initialize_token_blacklist():
    """Initialize the global token blacklist (call on app startup)."""
    global token_blacklist
    if os.getenv("ENABLE_TOKEN_BLACKLIST", "false").lower() == "true":
        token_blacklist = TokenBlacklist()
