"""Security configuration for RAG Pipeline."""
import os
from typing import List, Set, Union, Optional
from pydantic_settings import BaseSettings
from pydantic import field_validator, Field


class SecuritySettings(BaseSettings):
    """Security configuration settings loaded from environment variables."""
    
    # CORS Configuration
    allowed_origins: str = "http://localhost:3000,http://localhost:8000"
    
    # API Key Authentication (Legacy - for backward compatibility)
    api_key_header: str = "X-API-Key"
    service_api_keys: str = ""
    
    # JWT Authentication
    jwt_secret_key: str = Field(default="")  # Must be set in production!
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_minutes: int = 30
    jwt_refresh_token_expire_days: int = 7
    
    # Authentication Provider Selection
    auth_provider: str = Field(default="local")  # "local" or "supabase"
    
    # Supabase Configuration
    supabase_url: str = Field(default="")  # e.g., https://xxxxx.supabase.co
    supabase_anon_key: str = Field(default="")
    supabase_jwt_secret: str = Field(default="")  # Optional: for additional validation
    supabase_jwt_audience: str = Field(default="authenticated")  # Default Supabase audience
    
    # Admin Configuration (for restricted endpoints like /upload)
    admin_emails: str = Field(default="")  # Comma-separated list of admin emails
    admin_user_ids: str = Field(default="")  # Comma-separated list of admin user IDs (Supabase UUIDs)
    
    # File Upload Limits
    upload_max_megabytes: int = 50
    allowed_file_extensions: Set[str] = {".pdf", ".md", ".markdown"}
    allowed_mime_types: Set[str] = {"application/pdf", "text/markdown", "text/plain"}
    
    # Feature Flags
    enable_cors: bool = True
    enable_api_key_auth: bool = False  # Disabled by default, use JWT instead
    enable_jwt_auth: bool = True
    enable_file_validation: bool = True
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"
    
    def get_api_keys(self) -> Set[str]:
        """Parse and return API keys as a set."""
        if not self.service_api_keys:
            return set()
        return {k.strip() for k in self.service_api_keys.split(",") if k.strip()}
    
    def get_allowed_origins(self) -> List[str]:
        """Parse and return allowed origins as a list."""
        if not self.allowed_origins:
            return ["http://localhost:3000", "http://localhost:8000"]
        return [o.strip() for o in self.allowed_origins.split(",") if o.strip()]
    
    def is_valid_api_key(self, api_key: str) -> bool:
        """Check if the provided API key is valid."""
        if not self.enable_api_key_auth:
            return True
        return api_key in self.get_api_keys()
    
    def get_max_upload_bytes(self) -> int:
        """Get maximum upload size in bytes."""
        return self.upload_max_megabytes * 1024 * 1024
    
    def get_admin_emails(self) -> Set[str]:
        """Parse and return admin emails as a set."""
        if not self.admin_emails:
            return set()
        return {e.strip().lower() for e in self.admin_emails.split(",") if e.strip()}
    
    def get_admin_user_ids(self) -> Set[str]:
        """Parse and return admin user IDs as a set."""
        if not self.admin_user_ids:
            return set()
        return {uid.strip() for uid in self.admin_user_ids.split(",") if uid.strip()}
    
    def is_admin_user(self, email: Optional[str] = None, user_id: Optional[str] = None) -> bool:
        """
        Check if the user is an admin based on email or user ID.
        
        Args:
            email: User's email address
            user_id: User's unique ID (Supabase UUID)
            
        Returns:
            True if user is an admin, False otherwise
        """
        admin_emails = self.get_admin_emails()
        admin_user_ids = self.get_admin_user_ids()
        
        # If no admins are configured, deny access for security
        if not admin_emails and not admin_user_ids:
            return False
        
        # Check email match (case-insensitive)
        if email and admin_emails:
            if email.strip().lower() in admin_emails:
                return True
        
        # Check user ID match
        if user_id and admin_user_ids:
            if user_id.strip() in admin_user_ids:
                return True
        
        return False


# Global instance
security_settings = SecuritySettings()
