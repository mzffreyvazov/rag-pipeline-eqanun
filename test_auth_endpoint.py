"""
Test script to verify Supabase JWT authentication
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Get Supabase configuration
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
API_BASE_URL = "http://localhost:8000"

print("=" * 60)
print("Supabase Authentication Test")
print("=" * 60)
print(f"Supabase URL: {SUPABASE_URL}")
print(f"API Base URL: {API_BASE_URL}")
print()

# Step 1: Test JWKS endpoint accessibility
print("Step 1: Testing JWKS endpoint...")
jwks_url = f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json"
print(f"JWKS URL: {jwks_url}")

try:
    response = requests.get(jwks_url, timeout=10)
    if response.status_code == 200:
        jwks = response.json()
        print(f"✅ JWKS endpoint is accessible")
        print(f"   Found {len(jwks.get('keys', []))} keys")
    else:
        print(f"❌ JWKS endpoint returned status: {response.status_code}")
        print(f"   Response: {response.text}")
except Exception as e:
    print(f"❌ Failed to fetch JWKS: {e}")

print()
print("=" * 60)
print("How to get a valid Supabase JWT token:")
print("=" * 60)
print("""
1. Sign up or sign in a user through Supabase:
   
   Using JavaScript (in your frontend or Node.js):
   ```javascript
   import { createClient } from '@supabase/supabase-js'
   
   const supabase = createClient(
     'https://tppessosancihrmfvgme.supabase.co',
     'YOUR_ANON_KEY'
   )
   
   // Sign up a new user
   const { data, error } = await supabase.auth.signUp({
     email: 'user@example.com',
     password: 'yourpassword'
   })
   
   // Or sign in existing user
   const { data, error } = await supabase.auth.signInWithPassword({
     email: 'user@example.com',
     password: 'yourpassword'
   })
   
   // Get the access token
   const token = data.session.access_token
   console.log('Access Token:', token)
   ```

2. Using Python with supabase-py:
   ```python
   from supabase import create_client, Client
   
   url = "https://tppessosancihrmfvgme.supabase.co"
   key = "YOUR_ANON_KEY"
   supabase: Client = create_client(url, key)
   
   # Sign in
   response = supabase.auth.sign_in_with_password({
       "email": "user@example.com",
       "password": "yourpassword"
   })
   
   token = response.session.access_token
   print(f"Access Token: {token}")
   ```

3. Using cURL to sign in directly:
   ```bash
   curl -X POST 'https://tppessosancihrmfvgme.supabase.co/auth/v1/token?grant_type=password' \\
     -H "apikey: YOUR_ANON_KEY" \\
     -H "Content-Type: application/json" \\
     -d '{
       "email": "user@example.com",
       "password": "yourpassword"
     }'
   ```

4. Once you have a token, test the /auth/me endpoint:
   ```bash
   curl -X GET 'http://localhost:8000/auth/me' \\
     -H "Authorization: Bearer YOUR_ACTUAL_TOKEN"
   ```

⚠️  IMPORTANT: Replace $TOKEN with the actual token value!
   The literal string "$TOKEN" won't work - you need the real JWT token.
""")

print()
print("=" * 60)
print("Quick Test with Supabase")
print("=" * 60)
print("""
If you want to quickly test, you need to:

1. Go to your Supabase Dashboard (https://supabase.com/dashboard)
2. Navigate to Authentication > Users
3. Create a test user or use existing one
4. Get the user's JWT token

OR use the Supabase JavaScript client to authenticate and get the token.
""")
