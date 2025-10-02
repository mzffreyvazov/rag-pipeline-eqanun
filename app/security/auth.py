"""Authentication module for API key-based security."""
from typing import Optional
from fastapi import Header, HTTPException, status
from app.config.security import security_settings


async def get_current_api_key(
    x_api_key: Optional[str] = Header(None, alias=security_settings.api_key_header)
) -> Optional[str]:
    """
    Extract API key from request headers.
    
    Args:
        x_api_key: The API key from the request header
        
    Returns:
        The API key if present, None otherwise
    """
    return x_api_key


async def verify_api_key(
    x_api_key: Optional[str] = Header(None, alias=security_settings.api_key_header)
) -> str:
    """
    Verify that the request contains a valid API key.
    
    Args:
        x_api_key: The API key from the request header
        
    Returns:
        The validated API key
        
    Raises:
        HTTPException: If API key is missing or invalid
    """
    # If API key authentication is disabled, allow all requests
    if not security_settings.enable_api_key_auth:
        return "auth-disabled"
    
    # Check if API key is provided
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Please provide a valid API key in the X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    # Validate the API key
    if not security_settings.is_valid_api_key(x_api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key. Please check your API key and try again.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    
    return x_api_key
