#!/usr/bin/env python3
"""
Test script to verify that the evaluation endpoint works
"""

import requests
import json

def test_evaluation_endpoint():
    """Test the /evaluate endpoint"""
    
    url = "http://localhost:8001/evaluate"
    test_data = {
        "text": "Hey everyone! Today I'm going to show you an amazing life hack that will save you so much time. This simple trick will change your morning routine forever!",
        "max_iterations": 1,  # Use 1 to make test faster
        "target_score": 7
    }
    
    try:
        print("🧪 Testing evaluation endpoint...")
        print(f"📤 Sending request to {url}")
        print(f"📄 Data: {json.dumps(test_data, indent=2)}")
        
        response = requests.post(url, json=test_data, timeout=30)
        
        print(f"📊 Response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Evaluation successful!")
            print(f"📈 Final score: {result.get('final_score', 'N/A')}")
            print(f"🔄 Iterations: {result.get('iterations_completed', 'N/A')}")
            print(f"⏱️  Time: {result.get('total_processing_time', 'N/A')}s")
        else:
            print(f"❌ Request failed: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    test_evaluation_endpoint()
