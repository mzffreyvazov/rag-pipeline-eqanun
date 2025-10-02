"""File upload validation and security utilities."""
import secrets
import mimetypes
from pathlib import Path
from typing import Optional
from fastapi import UploadFile, HTTPException, status
from app.config.security import security_settings


async def validate_upload_file(file: UploadFile) -> None:
    """
    Validate uploaded file for security compliance.
    
    Checks:
    - File extension is allowed
    - Content type is allowed
    - File size is within limits
    
    Args:
        file: The uploaded file to validate
        
    Raises:
        HTTPException: If file fails validation
    """
    if not security_settings.enable_file_validation:
        return
    
    # Check file extension
    file_ext = Path(file.filename).suffix.lower() if file.filename else ""
    if file_ext not in security_settings.allowed_file_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type '{file_ext}' not allowed. Allowed types: {', '.join(security_settings.allowed_file_extensions)}"
        )
    
    # Check content type
    content_type = file.content_type or ""
    if content_type and content_type not in security_settings.allowed_mime_types:
        # Try to infer MIME type from filename
        inferred_type, _ = mimetypes.guess_type(file.filename or "")
        if not inferred_type or inferred_type not in security_settings.allowed_mime_types:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Content type '{content_type}' not allowed. Allowed types: {', '.join(security_settings.allowed_mime_types)}"
            )
    
    # Check file size (read first chunk to estimate)
    max_size = security_settings.get_max_upload_bytes()
    
    # Read the file to check size
    file_content = await file.read()
    file_size = len(file_content)
    
    # Reset file pointer for later processing
    await file.seek(0)
    
    if file_size > max_size:
        max_mb = security_settings.upload_max_megabytes
        actual_mb = file_size / (1024 * 1024)
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size ({actual_mb:.2f}MB) exceeds maximum allowed size ({max_mb}MB)"
        )


def sanitize_filename(filename: str) -> str:
    """
    Sanitize a filename to prevent path traversal and other security issues.
    
    Args:
        filename: The original filename
        
    Returns:
        A sanitized filename with random prefix
    """
    # Get the file extension
    file_ext = Path(filename).suffix.lower()
    
    # Generate a random secure filename
    random_name = secrets.token_hex(16)
    
    # Combine random name with original extension
    return f"{random_name}{file_ext}"
