# Test Admin Upload Access
# Save this as: test_admin_upload.py

import requests
import json

BASE_URL = "http://localhost:8000"

def test_admin_upload():
    """Test that only admin users can upload documents."""
    
    print("=" * 60)
    print("TESTING ADMIN UPLOAD ACCESS")
    print("=" * 60)
    
    # Step 1: Login as admin
    print("\n1️⃣ Logging in as admin...")
    admin_email = input("Enter your admin email: ")
    admin_password = input("Enter your password: ")
    
    login_response = requests.post(
        f"{BASE_URL}/auth/supabase/login",
        json={
            "email": admin_email,
            "password": admin_password
        }
    )
    
    if login_response.status_code != 200:
        print(f"❌ Login failed: {login_response.text}")
        return
    
    token_data = login_response.json()
    access_token = token_data["access_token"]
    user_email = token_data["user"]["email"]
    
    print(f"✅ Login successful!")
    print(f"   User: {user_email}")
    print(f"   Token: {access_token[:50]}...")
    
    # Step 2: Check user info
    print("\n2️⃣ Checking user info...")
    me_response = requests.get(
        f"{BASE_URL}/auth/supabase/me",
        headers={"Authorization": f"Bearer {access_token}"}
    )
    
    if me_response.status_code == 200:
        user_info = me_response.json()
        print(f"✅ User info retrieved:")
        print(f"   Email: {user_info.get('email')}")
        print(f"   ID: {user_info.get('id')}")
    
    # Step 3: Try to upload a test file
    print("\n3️⃣ Attempting to upload document...")
    print("   Creating test file...")
    
    # Create a test file
    test_content = b"This is a test document for admin upload verification."
    files = {
        'files': ('test_admin_upload.txt', test_content, 'text/plain')
    }
    
    upload_response = requests.post(
        f"{BASE_URL}/upload",
        headers={"Authorization": f"Bearer {access_token}"},
        files=files
    )
    
    print(f"\n4️⃣ Upload Response (Status: {upload_response.status_code}):")
    print("-" * 60)
    
    if upload_response.status_code == 200:
        result = upload_response.json()
        print("✅ SUCCESS! Upload allowed (you are an admin)")
        print(f"\n   Message: {result.get('message')}")
        print(f"   Files processed: {result.get('files_processed')}")
        print(f"   Total documents: {result.get('total_documents')}")
        print("\n" + "=" * 60)
        print("✅ ADMIN VERIFICATION PASSED")
        print("=" * 60)
        
    elif upload_response.status_code == 403:
        print("❌ FORBIDDEN! Upload denied")
        error = upload_response.json()
        print(f"\n   Error: {error.get('detail')}")
        print("\n" + "=" * 60)
        print("❌ YOU ARE NOT CONFIGURED AS ADMIN")
        print("=" * 60)
        print("\nTo fix this:")
        print(f"1. Add this to your .env file:")
        print(f"   ADMIN_EMAILS={user_email}")
        print(f"\n2. Restart your server")
        print(f"3. Run this test again")
        
    elif upload_response.status_code == 401:
        print("❌ UNAUTHORIZED! Token invalid or expired")
        error = upload_response.json()
        print(f"\n   Error: {error.get('detail')}")
        
    else:
        print(f"⚠️  Unexpected response:")
        print(upload_response.text)
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    try:
        test_admin_upload()
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Cannot connect to server")
        print("   Make sure the server is running:")
        print("   uvicorn app.main:app --reload")
    except KeyboardInterrupt:
        print("\n\nTest cancelled by user")
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
