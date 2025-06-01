#!/usr/bin/env python3
"""
Test script to verify OpenAI structured outputs implementation.
"""

import asyncio
import os
import json
from backend.app.score_new_rubric import AsyncOpenAIClient

async def test_structured_outputs():
    """Test the structured outputs functionality."""
    
    # Check if API key is available
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("❌ No OPENAI_API_KEY environment variable found.")
        print("Set your API key with: export OPENAI_API_KEY='your-key-here'")
        return False
    
    print("🔑 API key found in environment")
    
    # Test sample content
    test_content = """
    Did you know that honey never spoils? Archaeologists have found edible honey in ancient Egyptian tombs 
    that's over 3,000 years old! The low moisture content and acidic pH of honey create an environment 
    where bacteria can't survive. This is why honey has been used as a natural preservative for thousands of years.
    """
    
    try:
        # Initialize client
        print("🚀 Initializing OpenAI client...")
        client = AsyncOpenAIClient()
        await client.initialize()
        
        if not client.is_available:
            print("❌ OpenAI client is not available")
            return False
            
        print("✅ OpenAI client initialized successfully")
        
        # Test comprehensive analysis
        print("🧠 Testing structured output analysis...")
        result = await client.analyze_content_comprehensive(test_content)
        
        if not result:
            print("❌ No result returned from analysis")
            return False
            
        print("✅ Analysis completed successfully!")
        print("\n📊 Results:")
        print(json.dumps(result, indent=2))
        
        # Verify required fields
        required_fields = [
            'hook_score', 'virality_score', 'clarity_score', 
            'self_sufficiency_score', 'engagement_score',
            'has_strong_hook', 'has_one_claim', 'is_self_sufficient',
            'hook_explanation', 'virality_explanation', 'overall_analysis'
        ]
        
        missing_fields = [field for field in required_fields if field not in result]
        if missing_fields:
            print(f"❌ Missing required fields: {missing_fields}")
            return False
            
        print("✅ All required fields present")
        
        # Verify data types
        score_fields = ['hook_score', 'virality_score', 'clarity_score', 'self_sufficiency_score', 'engagement_score']
        bool_fields = ['has_strong_hook', 'has_one_claim', 'is_self_sufficient']
        string_fields = ['hook_explanation', 'virality_explanation', 'overall_analysis']
        
        for field in score_fields:
            if not isinstance(result[field], int) or not (1 <= result[field] <= 10):
                print(f"❌ {field} should be integer 1-10, got {result[field]}")
                return False
                
        for field in bool_fields:
            if not isinstance(result[field], bool):
                print(f"❌ {field} should be boolean, got {type(result[field])}")
                return False
                
        for field in string_fields:
            if not isinstance(result[field], str) or len(result[field]) < 1:
                print(f"❌ {field} should be non-empty string, got {result[field]}")
                return False
                
        print("✅ All field types are correct")
        
        # Test caching
        print("🔄 Testing caching functionality...")
        cached_result = await client.analyze_content_comprehensive(test_content)
        
        if cached_result != result:
            print("❌ Cached result doesn't match original")
            return False
            
        print("✅ Caching works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function."""
    print("🧪 Testing OpenAI Structured Outputs Implementation")
    print("=" * 50)
    
    success = await test_structured_outputs()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 All tests passed! OpenAI structured outputs are working correctly.")
        print("\n💡 You can now run the main script with:")
        print("   python score_new_rubric.py --text 'Your content here'")
    else:
        print("💥 Tests failed. Please check the implementation.")
        
if __name__ == "__main__":
    asyncio.run(main())
