"""JWT-based authentication for RAG Pipeline."""
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

from app.config.security import security_settings


# HTTP Bearer token scheme
security_scheme = HTTPBearer(auto_error=False)


class TokenData(BaseModel):
    """Token payload data model."""
    username: Optional[str] = None
    email: Optional[str] = None
    user_id: Optional[str] = None


class Token(BaseModel):
    """Token response model."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class RefreshToken(BaseModel):
    """Refresh token response model."""
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenPair(BaseModel):
    """Access and refresh token pair."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    access_expires_in: int
    refresh_expires_in: int


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash."""
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))


def get_password_hash(password: str) -> str:
    """Hash a password."""
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
    return hashed.decode('utf-8')


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Data to encode in the token (e.g., {"sub": user_id, "email": email})
        expires_delta: Optional custom expiration time
        
    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=security_settings.jwt_access_token_expire_minutes
        )
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "access"
    })
    
    encoded_jwt = jwt.encode(
        to_encode,
        security_settings.jwt_secret_key,
        algorithm=security_settings.jwt_algorithm
    )
    
    return encoded_jwt


def create_refresh_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT refresh token.
    
    Args:
        data: Data to encode in the token (e.g., {"sub": user_id})
        expires_delta: Optional custom expiration time
        
    Returns:
        Encoded JWT refresh token string
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            days=security_settings.jwt_refresh_token_expire_days
        )
    
    to_encode.update({
        "exp": expire,
        "iat": datetime.utcnow(),
        "type": "refresh"
    })
    
    encoded_jwt = jwt.encode(
        to_encode,
        security_settings.jwt_secret_key,
        algorithm=security_settings.jwt_algorithm
    )
    
    return encoded_jwt


def create_token_pair(data: Dict[str, Any]) -> TokenPair:
    """
    Create both access and refresh tokens.
    
    Args:
        data: Data to encode in the tokens
        
    Returns:
        TokenPair with access and refresh tokens
    """
    access_token = create_access_token(data)
    refresh_token = create_refresh_token(data)
    
    return TokenPair(
        access_token=access_token,
        refresh_token=refresh_token,
        access_expires_in=security_settings.jwt_access_token_expire_minutes * 60,
        refresh_expires_in=security_settings.jwt_refresh_token_expire_days * 24 * 60 * 60
    )


def decode_token(token: str) -> Dict[str, Any]:
    """
    Decode and verify a JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded token payload
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(
            token,
            security_settings.jwt_secret_key,
            algorithms=[security_settings.jwt_algorithm]
        )
        return payload
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Could not validate credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme)
) -> Dict[str, Any]:
    """
    FastAPI dependency to get the current authenticated user from JWT token.
    
    Supports both local and Supabase authentication based on AUTH_PROVIDER setting.
    
    Args:
        credentials: HTTP Bearer credentials from request
        
    Returns:
        User data from token payload
        
    Raises:
        HTTPException: If authentication fails
    """
    # If JWT auth is disabled, allow all requests
    if not security_settings.enable_jwt_auth:
        return {"sub": "anonymous", "email": "anonymous@localhost"}
    
    # Check if credentials were provided
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    
    # Route to appropriate authentication provider
    auth_provider = security_settings.auth_provider.lower()
    
    if auth_provider == "supabase":
        # Use Supabase JWT verification
        from app.security.supabase_auth import (
            verify_supabase_token,
            extract_user_info,
            validate_supabase_config,
            token_blacklist
        )
        
        # Validate Supabase configuration
        if not validate_supabase_config():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Supabase authentication is not properly configured"
            )
        
        # Check if token is blacklisted (optional)
        if token_blacklist and token_blacklist.is_blacklisted(token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has been revoked",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        # Verify Supabase token
        payload = verify_supabase_token(token)
        user_info = extract_user_info(payload)
        
        # Validate required fields
        if not user_info.get("user_id"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing user ID",
                headers={"WWW-Authenticate": "Bearer"}
            )
        
        return user_info
    
    else:
        # Use local JWT verification (default)
        payload = decode_token(token)
        
        # Verify token type
        token_type = payload.get("type")
        if token_type != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type. Access token required.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Extract user info
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return payload


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme)
) -> Optional[Dict[str, Any]]:
    """
    Optional authentication - returns user if authenticated, None otherwise.
    Useful for endpoints that have different behavior for authenticated users.
    """
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials)
    except HTTPException:
        return None


async def verify_refresh_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme)
) -> Dict[str, Any]:
    """
    Verify a refresh token and return its payload.
    
    Args:
        credentials: HTTP Bearer credentials from request
        
    Returns:
        Token payload
        
    Raises:
        HTTPException: If token is invalid or not a refresh token
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    payload = decode_token(token)
    
    # Verify it's a refresh token
    token_type = payload.get("type")
    if token_type != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token type. Refresh token required.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return payload
