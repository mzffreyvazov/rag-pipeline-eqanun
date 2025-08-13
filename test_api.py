# Testing script for the RAG Pipeline API
import requests
import json
import time
from pathlib import Path

# Configuration
BASE_URL = "http://localhost:8000"
TEST_PDF_PATH = "assets/emek-mecellesi-1-2.pdf"  # Adjust path as needed

def test_health_check():
    """Test the health check endpoint"""
    print("🏥 Testing health check...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def test_status():
    """Test the status endpoint"""
    print("📊 Testing status endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/status")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Status check failed: {e}")
        return False

def test_upload_document():
    """Test document upload"""
    print("📄 Testing document upload...")
    
    # Check if test file exists
    if not Path(TEST_PDF_PATH).exists():
        print(f"❌ Test file not found: {TEST_PDF_PATH}")
        print("Please ensure you have a PDF file in the assets folder or update the path")
        return False
    
    try:
        with open(TEST_PDF_PATH, 'rb') as f:
            files = {'files': (TEST_PDF_PATH, f, 'application/pdf')}
            response = requests.post(f"{BASE_URL}/upload", files=files)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False

def test_chat():
    """Test chat functionality"""
    print("💬 Testing chat...")
    
    test_messages = [
        "Salam!",
        "Əmək məcəlləsinə əsasən Müəssisə anlayışı nədir?",
        "İşəgötürən kim hesab olunur?",
        "Əmək müqaviləsi nədir?"
    ]
    
    session_id = "test_session_123"
    
    for message in test_messages:
        try:
            data = {
                "message": message,
                "session_id": session_id
            }
            
            print(f"\nUser: {message}")
            response = requests.post(f"{BASE_URL}/chat", json=data)
            
            if response.status_code == 200:
                result = response.json()
                print(f"Assistant: {result['response']}")
                session_id = result['session_id']  # Use the returned session_id
            else:
                print(f"❌ Chat failed with status: {response.status_code}")
                print(f"Error: {response.text}")
                return False
                
            time.sleep(1)  # Small delay between messages
            
        except Exception as e:
            print(f"❌ Chat message failed: {e}")
            return False
    
    return True

def test_clear_documents():
    """Test clearing documents"""
    print("🗑️ Testing document clearing...")
    try:
        response = requests.delete(f"{BASE_URL}/documents")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Clear documents failed: {e}")
        return False

def run_all_tests():
    """Run all tests"""
    print("🧪 Starting RAG Pipeline API Tests")
    print("=" * 50)
    
    tests = [
        ("Health Check", test_health_check),
        ("Status Check", test_status),
        ("Document Upload", test_upload_document),
        ("Chat Functionality", test_chat),
        ("Clear Documents", test_clear_documents),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        success = test_func()
        results.append((test_name, success))
        
        if success:
            print(f"✅ {test_name} passed")
        else:
            print(f"❌ {test_name} failed")
    
    print("\n" + "="*50)
    print("📋 TEST SUMMARY")
    print("="*50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed. Please check the API and try again.")

if __name__ == "__main__":
    run_all_tests()
