"""Authentication routes for user login and token management."""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field
from typing import Optional

from app.security.jwt_auth import (
    verify_password,
    get_password_hash,
    create_token_pair,
    create_access_token,
    verify_refresh_token,
    get_current_user,
    TokenPair,
    Token
)
from app.config.security import security_settings


router = APIRouter(prefix="/auth", tags=["authentication"])
security_scheme = HTTPBearer(auto_error=False)


# Request/Response Models
class UserLogin(BaseModel):
    """User login credentials."""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)


class UserRegister(BaseModel):
    """User registration data."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=8)
    full_name: Optional[str] = None


class UserResponse(BaseModel):
    """User data response."""
    user_id: str
    username: str
    email: str
    full_name: Optional[str] = None


# In-memory user store (replace with database in production!)
# For demo purposes only - you should integrate with your actual user database
DEMO_USERS = {
    "demo": {
        "user_id": "demo-user-001",
        "username": "demo",
        "email": "demo@example.com",
        "full_name": "Demo User",
        # Password: "demo1234" (hashed with bcrypt)
        "hashed_password": "$2b$12$LIIOc/Q.2zfJcnjabjcEMep7vso9.vEAaNqqXg4tYZ8odCHoIMBey"
    }
}


@router.post("/login", response_model=TokenPair)
async def login(credentials: UserLogin):
    """
    Authenticate user and return JWT access and refresh tokens.
    
    **Demo Credentials:**
    - Username: `demo`
    - Password: `demo1234`
    
    **For Production:**
    Replace the in-memory user store with your database lookup.
    
    Returns:
        - access_token: Short-lived token for API requests (30 min)
        - refresh_token: Long-lived token for getting new access tokens (7 days)
    """
    # Look up user (replace with database query)
    user = DEMO_USERS.get(credentials.username)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify password
    if not verify_password(credentials.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create token pair
    token_data = {
        "sub": user["user_id"],
        "username": user["username"],
        "email": user["email"]
    }
    
    tokens = create_token_pair(token_data)
    
    return tokens


@router.post("/refresh", response_model=Token)
async def refresh_access_token(payload: dict = Depends(verify_refresh_token)):
    """
    Generate a new access token using a refresh token.
    
    **Headers:**
    ```
    Authorization: Bearer <refresh_token>
    ```
    
    Returns:
        New access token
    """
    # Create new access token with same user data
    token_data = {
        "sub": payload.get("sub"),
        "username": payload.get("username"),
        "email": payload.get("email")
    }
    
    access_token = create_access_token(token_data)
    
    from app.config.security import security_settings
    
    return Token(
        access_token=access_token,
        expires_in=security_settings.jwt_access_token_expire_minutes * 60
    )


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(user_data: UserRegister):
    """
    Register a new user (demo endpoint).
    
    **Note:** This is a demo implementation. In production:
    1. Store users in a database
    2. Add email verification
    3. Implement proper validation
    4. Add rate limiting
    5. Check for duplicate usernames/emails
    
    **For your Next.js app:** You'll likely have your own user management,
    so you can remove this endpoint or integrate with your existing system.
    """
    # Check if username exists (in real app, check database)
    if user_data.username in DEMO_USERS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already exists"
        )
    
    # Hash password
    hashed_password = get_password_hash(user_data.password)
    
    # Create user ID (in real app, use database auto-increment or UUID)
    import uuid
    user_id = f"user-{uuid.uuid4()}"
    
    # Store user (in real app, save to database)
    DEMO_USERS[user_data.username] = {
        "user_id": user_id,
        "username": user_data.username,
        "email": user_data.email,
        "full_name": user_data.full_name,
        "hashed_password": hashed_password
    }
    
    return UserResponse(
        user_id=user_id,
        username=user_data.username,
        email=user_data.email,
        full_name=user_data.full_name
    )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """
    Get current authenticated user information.
    
    **Headers:**
    ```
    Authorization: Bearer <access_token>
    ```
    
    Returns:
        Current user data from token
    """
    # Get user_id from token (works for both local and Supabase auth)
    # For Supabase: user_id is extracted from "sub" claim
    # For local: sub is the user_id
    user_id = current_user.get("user_id") or current_user.get("sub")
    
    # For demo, find in our in-memory store
    for user in DEMO_USERS.values():
        if user["user_id"] == user_id:
            return UserResponse(
                user_id=user["user_id"],
                username=user["username"],
                email=user["email"],
                full_name=user.get("full_name")
            )
    
    # If not found, return data from token
    return UserResponse(
        user_id=user_id or "unknown",
        username=current_user.get("username", "unknown"),
        email=current_user.get("email", "unknown@example.com")
    )


@router.post("/logout")
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
    current_user: dict = Depends(get_current_user)
):
    """
    Logout endpoint - blacklists the current token (requires Redis).
    
    **Headers:**
    ```
    Authorization: Bearer <access_token>
    ```
    
    **Note:** This endpoint is only functional when:
    1. AUTH_PROVIDER=supabase (Supabase authentication)
    2. ENABLE_TOKEN_BLACKLIST=true (Redis is configured)
    
    For local authentication, tokens will expire naturally after 30 minutes.
    For Supabase, you should also call Supabase's sign-out endpoint from your frontend.
    
    Returns:
        Success message
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    
    token = credentials.credentials
    
    # Only blacklist tokens if using Supabase authentication
    if security_settings.auth_provider.lower() == "supabase":
        from app.security.supabase_auth import token_blacklist
        
        if token_blacklist and token_blacklist.enabled:
            # Calculate remaining TTL for the token
            exp = current_user.get("exp")
            if exp:
                import time
                ttl = exp - int(time.time())
                if ttl > 0:
                    token_blacklist.add_token(token, ttl)
            
            return {
                "message": "Successfully logged out. Token has been revoked.",
                "note": "Please also clear the token from your frontend storage."
            }
        else:
            return {
                "message": "Logout processed (token blacklist not enabled)",
                "note": "Token will expire naturally. Configure Redis and set ENABLE_TOKEN_BLACKLIST=true for immediate revocation."
            }
    else:
        return {
            "message": "Logout processed",
            "note": "Token will expire naturally after 30 minutes. For immediate revocation, use Supabase authentication with Redis."
        }
