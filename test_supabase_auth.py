#!/usr/bin/env python3
"""
Test script for Supabase authentication integration.

Usage:
    python test_supabase_auth.py
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.config.security import security_settings


def test_configuration():
    """Test if configuration is properly loaded."""
    print("🔍 Testing Configuration...")
    print(f"  AUTH_PROVIDER: {security_settings.auth_provider}")
    print(f"  ENABLE_JWT_AUTH: {security_settings.enable_jwt_auth}")
    
    if security_settings.auth_provider.lower() == "supabase":
        print(f"  SUPABASE_URL: {security_settings.supabase_url or '❌ NOT SET'}")
        print(f"  SUPABASE_JWT_AUDIENCE: {security_settings.supabase_jwt_audience}")
        
        if not security_settings.supabase_url:
            print("⚠️  WARNING: SUPABASE_URL is not configured!")
            return False
    
    print("✅ Configuration OK\n")
    return True


def test_supabase_module():
    """Test if Supabase module can be imported."""
    print("🔍 Testing Supabase Module Import...")
    
    try:
        from app.security.supabase_auth import (
            fetch_supabase_jwks,
            verify_supabase_token,
            extract_user_info,
            validate_supabase_config
        )
        print("✅ Supabase module imported successfully\n")
        return True
    except ImportError as e:
        print(f"❌ Failed to import Supabase module: {e}\n")
        return False


def test_supabase_config_validation():
    """Test Supabase configuration validation."""
    print("🔍 Testing Supabase Configuration Validation...")
    
    if security_settings.auth_provider.lower() != "supabase":
        print("⏭️  Skipping (AUTH_PROVIDER is not 'supabase')\n")
        return True
    
    try:
        from app.security.supabase_auth import validate_supabase_config
        
        is_valid = validate_supabase_config()
        if is_valid:
            print("✅ Supabase configuration is valid\n")
            return True
        else:
            print("❌ Supabase configuration is invalid\n")
            return False
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}\n")
        return False


def test_jwks_fetch():
    """Test JWKS fetching from Supabase."""
    print("🔍 Testing JWKS Fetching...")
    
    if security_settings.auth_provider.lower() != "supabase":
        print("⏭️  Skipping (AUTH_PROVIDER is not 'supabase')\n")
        return True
    
    if not security_settings.supabase_url:
        print("⏭️  Skipping (SUPABASE_URL not configured)\n")
        return True
    
    try:
        from app.security.supabase_auth import fetch_supabase_jwks, get_jwks_url
        
        jwks_url = get_jwks_url()
        print(f"  JWKS URL: {jwks_url}")
        
        jwks = fetch_supabase_jwks()
        keys = jwks.get("keys", [])
        print(f"  Found {len(keys)} public keys")
        
        if keys:
            print(f"  Sample key ID: {keys[0].get('kid', 'N/A')}")
        
        print("✅ JWKS fetch successful\n")
        return True
    except Exception as e:
        print(f"❌ JWKS fetch failed: {e}\n")
        return False


def test_jwt_auth_module():
    """Test JWT auth module with multi-provider support."""
    print("🔍 Testing JWT Auth Module...")
    
    try:
        from app.security.jwt_auth import get_current_user
        print("✅ JWT auth module imported successfully\n")
        return True
    except ImportError as e:
        print(f"❌ Failed to import JWT auth module: {e}\n")
        return False


def test_dependencies():
    """Test required dependencies."""
    print("🔍 Testing Dependencies...")
    
    missing_deps = []
    
    try:
        import requests
        print("  ✅ requests")
    except ImportError:
        print("  ❌ requests")
        missing_deps.append("requests")
    
    try:
        from jose import jwt, jwk
        print("  ✅ python-jose")
    except ImportError:
        print("  ❌ python-jose")
        missing_deps.append("python-jose[cryptography]")
    
    try:
        import bcrypt
        print("  ✅ bcrypt")
    except ImportError:
        print("  ❌ bcrypt")
        missing_deps.append("bcrypt")
    
    # Optional: Redis
    try:
        import redis
        print("  ✅ redis (optional)")
    except ImportError:
        print("  ⚠️  redis (optional - needed for token blacklist)")
    
    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        print(f"Install with: pip install {' '.join(missing_deps)}\n")
        return False
    
    print("\n✅ All required dependencies installed\n")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Supabase Authentication Integration Test")
    print("=" * 60)
    print()
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Configuration", test_configuration),
        ("Supabase Module", test_supabase_module),
        ("JWT Auth Module", test_jwt_auth_module),
        ("Config Validation", test_supabase_config_validation),
        ("JWKS Fetch", test_jwks_fetch),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}\n")
            results.append((test_name, False))
    
    print("=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {test_name}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Supabase integration is ready.")
        print("\nNext steps:")
        print("1. Configure your Supabase credentials in .env")
        print("2. Set AUTH_PROVIDER=supabase")
        print("3. Start the server: python app/main.py")
        print("4. Test with a real Supabase token")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
