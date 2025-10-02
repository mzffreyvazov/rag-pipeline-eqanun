"""Supabase authentication routes for user login, registration, and management."""
from fastapi import APIRouter, HTTPException, status, Depends
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
import os
import requests

from app.config.security import security_settings
from app.security.jwt_auth import get_current_user


router = APIRouter(prefix="/auth/supabase", tags=["supabase-authentication"])


# Request/Response Models
class SupabaseLogin(BaseModel):
    """Supabase login credentials."""
    email: EmailStr
    password: str = Field(..., min_length=6)


class SupabaseRegister(BaseModel):
    """Supabase user registration data."""
    email: EmailStr
    password: str = Field(..., min_length=6)
    full_name: Optional[str] = None


class SupabaseTokenResponse(BaseModel):
    """Supabase authentication response."""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: dict


class SupabaseUserResponse(BaseModel):
    """Supabase user information."""
    id: str
    email: str
    created_at: str
    last_sign_in_at: Optional[str] = None


class PasswordResetRequest(BaseModel):
    """Password reset request."""
    email: EmailStr


class RefreshTokenRequest(BaseModel):
    """Refresh token request."""
    refresh_token: str = Field(..., description="Refresh token from login")


def get_supabase_auth_url() -> str:
    """Get Supabase Auth API URL."""
    if not security_settings.supabase_url:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Supabase URL not configured. Set SUPABASE_URL in .env"
        )
    base_url = security_settings.supabase_url.rstrip('/')
    return f"{base_url}/auth/v1"


def get_auth_headers() -> dict:
    """Get headers for Supabase Auth API requests."""
    if not security_settings.supabase_anon_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Supabase anon key not configured. Set SUPABASE_ANON_KEY in .env"
        )
    return {
        "apikey": security_settings.supabase_anon_key,
        "Content-Type": "application/json"
    }


@router.post("/login", response_model=SupabaseTokenResponse)
async def supabase_login(credentials: SupabaseLogin):
    """
    Login with Supabase email and password.
    
    This endpoint authenticates users via Supabase Auth and returns:
    - Access token (JWT) - Use this in Authorization header for API requests
    - Refresh token - Use to get new access tokens when they expire
    - User information
    
    **Example:**
    ```bash
    curl -X POST http://localhost:8000/auth/supabase/login \\
      -H "Content-Type: application/json" \\
      -d '{"email":"user@example.com","password":"yourpassword"}'
    ```
    
    **Returns:**
    ```json
    {
      "access_token": "eyJhbGc...",
      "refresh_token": "eyJhbGc...",
      "token_type": "bearer",
      "expires_in": 3600,
      "user": {
        "id": "user-uuid",
        "email": "user@example.com"
      }
    }
    ```
    """
    try:
        auth_url = get_supabase_auth_url()
        headers = get_auth_headers()
        
        response = requests.post(
            f"{auth_url}/token?grant_type=password",
            headers=headers,
            json={
                "email": credentials.email,
                "password": credentials.password
            },
            timeout=10
        )
        
        if response.status_code == 400:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        response.raise_for_status()
        data = response.json()
        
        return SupabaseTokenResponse(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            token_type="bearer",
            expires_in=data.get("expires_in", 3600),
            user=data.get("user", {})
        )
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to connect to Supabase: {str(e)}"
        )
    except KeyError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected response from Supabase: missing {str(e)}"
        )


@router.post("/register", response_model=SupabaseTokenResponse, status_code=status.HTTP_201_CREATED)
async def supabase_register(user_data: SupabaseRegister):
    """
    Register a new user with Supabase.
    
    Creates a new user account in Supabase and returns authentication tokens.
    
    **Note:** Depending on your Supabase settings:
    - Email confirmation might be required
    - User might be auto-confirmed
    
    **Example:**
    ```bash
    curl -X POST http://localhost:8000/auth/supabase/register \\
      -H "Content-Type: application/json" \\
      -d '{
        "email":"newuser@example.com",
        "password":"securepassword123",
        "full_name":"John Doe"
      }'
    ```
    """
    try:
        auth_url = get_supabase_auth_url()
        headers = get_auth_headers()
        
        payload = {
            "email": user_data.email,
            "password": user_data.password
        }
        
        if user_data.full_name:
            payload["data"] = {"full_name": user_data.full_name}
        
        response = requests.post(
            f"{auth_url}/signup",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 400:
            error_msg = response.json().get("msg", "Registration failed")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=error_msg
            )
        
        response.raise_for_status()
        data = response.json()
        
        return SupabaseTokenResponse(
            access_token=data.get("access_token", ""),
            refresh_token=data.get("refresh_token", ""),
            token_type="bearer",
            expires_in=data.get("expires_in", 3600),
            user=data.get("user", {})
        )
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to connect to Supabase: {str(e)}"
        )


@router.post("/refresh", response_model=SupabaseTokenResponse)
async def supabase_refresh_token(request: RefreshTokenRequest):
    """
    Get a new access token using a refresh token.
    
    **Example:**
    ```bash
    curl -X POST http://localhost:8000/auth/supabase/refresh \\
      -H "Content-Type: application/json" \\
      -d '{"refresh_token":"your-refresh-token"}'
    ```
    """
    try:
        auth_url = get_supabase_auth_url()
        headers = get_auth_headers()
        
        response = requests.post(
            f"{auth_url}/token?grant_type=refresh_token",
            headers=headers,
            json={"refresh_token": request.refresh_token},
            timeout=10
        )
        
        if response.status_code == 400:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired refresh token"
            )
        
        response.raise_for_status()
        data = response.json()
        
        return SupabaseTokenResponse(
            access_token=data["access_token"],
            refresh_token=data.get("refresh_token", request.refresh_token),
            token_type="bearer",
            expires_in=data.get("expires_in", 3600),
            user=data.get("user", {})
        )
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to connect to Supabase: {str(e)}"
        )


@router.get("/me", response_model=SupabaseUserResponse)
async def get_supabase_user(current_user: dict = Depends(get_current_user)):
    """
    Get current user information from Supabase token.
    
    **Headers:**
    ```
    Authorization: Bearer <access_token>
    ```
    
    **Example:**
    ```bash
    curl -X GET http://localhost:8000/auth/supabase/me \\
      -H "Authorization: Bearer your-access-token"
    ```
    """
    user_id = current_user.get("user_id") or current_user.get("sub")
    email = current_user.get("email", "")
    
    return SupabaseUserResponse(
        id=user_id or "unknown",
        email=email,
        created_at=current_user.get("created_at", ""),
        last_sign_in_at=current_user.get("last_sign_in_at")
    )


@router.post("/logout")
async def supabase_logout(current_user: dict = Depends(get_current_user)):
    """
    Logout current user (frontend should clear tokens).
    
    **Headers:**
    ```
    Authorization: Bearer <access_token>
    ```
    
    **Note:** This endpoint validates the token and confirms logout.
    Your frontend should:
    1. Call this endpoint
    2. Clear access_token and refresh_token from storage
    3. Redirect to login page
    
    For immediate token revocation, configure Redis and set ENABLE_TOKEN_BLACKLIST=true
    """
    return {
        "message": "Logout successful",
        "note": "Please clear tokens from your frontend storage"
    }


@router.post("/reset-password")
async def request_password_reset(request: PasswordResetRequest):
    """
    Send password reset email via Supabase.
    
    **Example:**
    ```bash
    curl -X POST http://localhost:8000/auth/supabase/reset-password \\
      -H "Content-Type: application/json" \\
      -d '{"email":"user@example.com"}'
    ```
    
    **Note:** Configure your Supabase email templates and redirect URLs
    in the Supabase dashboard under Authentication > Email Templates
    """
    try:
        auth_url = get_supabase_auth_url()
        headers = get_auth_headers()
        
        response = requests.post(
            f"{auth_url}/recover",
            headers=headers,
            json={"email": request.email},
            timeout=10
        )
        
        # Supabase returns 200 even if email doesn't exist (security best practice)
        response.raise_for_status()
        
        return {
            "message": "If the email exists, a password reset link has been sent",
            "note": "Check your email inbox and spam folder"
        }
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Failed to send reset email: {str(e)}"
        )
