#!/usr/bin/env python3
"""
Test script to verify the OpenAI Responses API implementation with structured outputs.
"""

import asyncio
import json
import os
from backend.app.score_new_rubric import AsyncOpenAIClient

async def test_responses_api():
    """Test the Responses API implementation with structured outputs."""
    
    # Sample text to analyze
    test_content = """
    Wait until you hear this shocking discovery! Scientists just found out that 
    drinking water before meals can actually make you feel full faster. 
    This simple trick could change how we think about weight loss forever!
    """
    
    try:
        # Initialize the client
        client = AsyncOpenAIClient()
        
        # Initialize and check availability
        await client.initialize()
        
        if not client.is_available:
            print("❌ OpenAI Responses API is not available")
            return
        
        print("✅ OpenAI Responses API is available")
        print(f"Using model: {client.model_name}")
        
        # Test comprehensive analysis
        print("\n🔍 Testing structured output analysis...")
        result = await client.analyze_content_comprehensive(test_content)
        
        if result:
            print("✅ Analysis completed successfully!")
            print("\n📊 Results:")
            print(f"Hook Score: {result.get('hook_score', 'N/A')}")
            print(f"Virality Score: {result.get('virality_score', 'N/A')}")
            print(f"Clarity Score: {result.get('clarity_score', 'N/A')}")
            print(f"Self-Sufficiency Score: {result.get('self_sufficiency_score', 'N/A')}")
            print(f"Engagement Score: {result.get('engagement_score', 'N/A')}")
            print(f"\nHook Explanation: {result.get('hook_explanation', 'N/A')}")
            print(f"Virality Explanation: {result.get('virality_explanation', 'N/A')}")
            print(f"Overall Analysis: {result.get('overall_analysis', 'N/A')}")
            
            # Verify all expected fields are present
            expected_fields = [
                'hook_score', 'virality_score', 'clarity_score', 
                'self_sufficiency_score', 'engagement_score',
                'has_strong_hook', 'has_one_claim', 'is_self_sufficient',
                'hook_explanation', 'virality_explanation', 'overall_analysis'
            ]
            
            missing_fields = [field for field in expected_fields if field not in result]
            if missing_fields:
                print(f"\n⚠️  Missing fields: {missing_fields}")
            else:
                print("\n✅ All expected fields present in structured output!")
                
        else:
            print("❌ Analysis failed - no results returned")
            
    except Exception as e:
        print(f"❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Check if API key is set
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Please set OPENAI_API_KEY environment variable")
        exit(1)
    
    print("🚀 Testing OpenAI Responses API with Structured Outputs")
    print("=" * 60)
    
    asyncio.run(test_responses_api())
