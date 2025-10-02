"""Debug script to analyze JWT token and JWKS"""
import requests
import json
from jose import jwt

# Your Supabase URL
SUPABASE_URL = "https://tppessosancihrmfvgme.supabase.co"

print("=" * 70)
print("JWT Token Debug Analysis")
print("=" * 70)

# Paste your token here
TOKEN = input("\nPaste your access_token here: ").strip()

if not TOKEN:
    print("❌ No token provided!")
    exit(1)

print("\n" + "=" * 70)
print("Step 1: Analyzing Token Header")
print("=" * 70)

try:
    # Decode header without verification
    header = jwt.get_unverified_header(TOKEN)
    print(json.dumps(header, indent=2))
    
    kid = header.get('kid')
    alg = header.get('alg')
    
    print(f"\n🔑 Key ID (kid): {kid}")
    print(f"🔒 Algorithm: {alg}")
    
except Exception as e:
    print(f"❌ Error decoding token header: {e}")
    exit(1)

print("\n" + "=" * 70)
print("Step 2: Fetching JWKS from Supabase")
print("=" * 70)

jwks_url = f"{SUPABASE_URL}/auth/v1/.well-known/jwks.json"
print(f"JWKS URL: {jwks_url}")

try:
    response = requests.get(jwks_url, timeout=10)
    response.raise_for_status()
    jwks = response.json()
    
    print(f"\n✅ JWKS fetched successfully")
    print(f"Number of keys: {len(jwks.get('keys', []))}")
    
    print("\nAvailable Key IDs:")
    for key in jwks.get('keys', []):
        print(f"  - kid: {key.get('kid')} | alg: {key.get('alg')} | kty: {key.get('kty')}")
    
except Exception as e:
    print(f"❌ Error fetching JWKS: {e}")
    exit(1)

print("\n" + "=" * 70)
print("Step 3: Checking for Key Match")
print("=" * 70)

matching_key = None
for key in jwks.get('keys', []):
    if key.get('kid') == kid:
        matching_key = key
        break

if matching_key:
    print(f"✅ Found matching key!")
    print(json.dumps(matching_key, indent=2))
else:
    print(f"❌ No matching key found for kid: {kid}")
    print("\n⚠️  This means:")
    print("   1. Your JWT was signed with a different key")
    print("   2. The token might be using HS256 (symmetric) instead of RS256")
    print("   3. You might need to use JWT_SECRET instead of JWKS verification")

print("\n" + "=" * 70)
print("Step 4: Decoding Token Claims (unverified)")
print("=" * 70)

try:
    claims = jwt.get_unverified_claims(TOKEN)
    print(json.dumps(claims, indent=2, default=str))
except Exception as e:
    print(f"❌ Error decoding claims: {e}")

print("\n" + "=" * 70)
print("Recommendation:")
print("=" * 70)

if alg == "HS256":
    print("""
⚠️  Your token uses HS256 (symmetric key) algorithm!

Supabase tokens are typically signed with HS256 using the JWT_SECRET.
You need to verify using the JWT secret, not JWKS (public keys).

Solution:
1. Get your JWT secret from Supabase Dashboard:
   Settings > API > JWT Secret
   
2. Update your .env file:
   SUPABASE_JWT_SECRET=your-actual-jwt-secret
   
3. The verification code needs to use the secret instead of JWKS for HS256 tokens.
""")
elif alg == "RS256" and not matching_key:
    print("""
⚠️  Token uses RS256 but the key ID doesn't match JWKS!

This could mean:
1. The JWKS cache needs to be refreshed
2. Token was issued by a different auth server
3. Supabase project settings have changed

Try signing in again to get a fresh token.
""")
else:
    print("✅ Token configuration looks good!")
