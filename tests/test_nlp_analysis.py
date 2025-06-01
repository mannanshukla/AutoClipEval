#!/usr/bin/env python3
"""
Test script for the new OpenAI-powered script analysis.
Tests both the script analysis functions and the API endpoint.
"""

import asyncio
import json
import time
import os
from backend.app.score_new_rubric import process_text_async, analyze_hook_nlp, analyze_one_claim_nlp, analyze_self_sufficient_nlp
import requests
from typing import Dict, Any

# Test script content
TEST_SCRIPT = """Did you hear what AOC said? She said the
people that they're coming in to this country, most of them it's because of
climate change. Often people want to say, "Why aren't you talking about the
border crisis or why aren't you talking about it in this way?" Because it's not
a border crisis. It's an imperialism crisis. It's a climate crisis. It's a
trade crisis. It's amazing. As I have already said, even during this term and
this president, our immigration system is based and designed on our carceral
system. It is straight out of South Park. It's straight out of South Park.
Like what? I know. The US has disproportionately contributed to the
total amount of emissions that is causing a planetary climate crisis right
now."""

def test_script_analysis_function():
    """Test the script analysis function from score_new_rubric.py"""
    print("\n=== Testing script analysis function ===")
    
    # Set API key from environment or use your key here
    api_key = os.environ.get("OPENAI_API_KEY")
    
    # Process the test script
    result = asyncio.run(process_text_async(TEST_SCRIPT, "gpt-4.1-nano", api_key))
    
    # Print results
    print(f"Analysis complete: Overall score: {result.get('overall_shorts_score', 0)}/10")
    print(f"Viral potential: {result.get('viral_potential', 'Unknown')}")
    print(f"Hook: {result.get('hook', False)}")
    print(f"Explanations available: {'Yes' if 'explanations' in result else 'No'}")
    
    return result

def test_api_endpoint():
    """Test the FastAPI endpoint for script scoring"""
    print("\n=== Testing API endpoint ===")
    
    try:
        # Make sure the server is running at this URL
        api_url = "http://localhost:8000/score"
        
        # Prepare request data
        request_data = {
            "script": TEST_SCRIPT,
            "model": "gpt-4.1-nano"
        }
        
        # Send request to API
        print("Sending request to API...")
        start_time = time.time()
        response = requests.post(api_url, json=request_data)
        duration = time.time() - start_time
        
        # Check if request was successful
        if response.status_code == 200:
            result = response.json()
            print(f"API response received in {duration:.2f} seconds")
            print(f"Analysis complete: Overall score: {result['rubric'].get('overall_shorts_score', 0)}/10")
            print(f"Viral potential: {result['rubric'].get('viral_potential', 'Unknown')}")
            print(f"Hook: {result['rubric'].get('hook', False)}")
            print(f"Explanations available: {'Yes' if 'analysis_explanations' in result else 'No'}")
            return result
        else:
            print(f"API request failed with status code {response.status_code}")
            print(f"Error: {response.text}")
            return None
    except Exception as e:
        print(f"Error testing API endpoint: {str(e)}")
        return None

def compare_results(func_result: Dict[str, Any], api_result: Dict[str, Any]):
    """Compare results from function and API endpoint"""
    if not api_result:
        print("\n=== Cannot compare results: API test failed ===")
        return
    
    print("\n=== Comparing results ===")
    
    # Extract rubric from API result
    api_rubric = api_result['rubric']
    
    # Compare overall scores
    func_score = func_result.get('overall_shorts_score', 0)
    api_score = api_rubric.get('overall_shorts_score', 0)
    print(f"Overall score: Function={func_score}/10, API={api_score}/10")
    
    # Compare boolean values
    boolean_fields = ['hook', 'oneClaim', 'selfSufficient']
    for field in boolean_fields:
        func_value = func_result.get(field, False)
        api_value = api_rubric.get(field, False)
        print(f"{field}: Function={func_value}, API={api_value}")
    
    # Compare important numeric values
    numeric_fields = ['early_engagement', 'virality_potential', 'engagement_potential']
    for field in numeric_fields:
        func_value = func_result.get(field, 0)
        api_value = api_rubric.get(field, 0)
        print(f"{field}: Function={func_value}/10, API={api_value}/10")

def test_specific_scripts():
    """Test NLP functions on specific scripts that showed changes."""
    
    # Load the scripts
    with open('scores.json', 'r') as f:
        scripts = json.load(f)
    
    # Find specific test cases
    test_cases = [
        'Joe_Rogan_Reacts_to_Macron_SLAP',
        'Joe_Rogan_Reacts_to_AOC_MELTDOWN', 
        'Joe_Rogan_Reacts_to_Bernie_Sanders_ROASTING_Donald_Trump'
    ]
    
    for script_id in test_cases:
        script = next((s for s in scripts if s['id'] == script_id), None)
        if not script:
            continue
            
        print(f"\n" + "="*60)
        print(f"ANALYZING: {script_id}")
        print("="*60)
        
        script_text = script['script']
        current_rubric = script['rubric']
        
        # Show first 150 characters 
        print(f"\nOPENING TEXT (first 150 chars):")
        print(f'"{script_text[:150]}..."')
        
        # Test each NLP function
        print(f"\nNLP ANALYSIS RESULTS:")
        hook_result = asyncio.run(analyze_hook_nlp(script_text))
        claim_result = asyncio.run(analyze_one_claim_nlp(script_text))
        sufficient_result = asyncio.run(analyze_self_sufficient_nlp(script_text))
        
        print(f"Hook (NLP): {hook_result} | Current: {current_rubric['hook']}")
        print(f"OneClaim (NLP): {claim_result} | Current: {current_rubric['oneClaim']}")
        print(f"SelfSufficient (NLP): {sufficient_result} | Current: {current_rubric['selfSufficient']}")
        
        # Show any mismatches
        mismatches = []
        if hook_result != current_rubric['hook']:
            mismatches.append(f"Hook: Manual={current_rubric['hook']} vs NLP={hook_result}")
        if claim_result != current_rubric['oneClaim']:
            mismatches.append(f"OneClaim: Manual={current_rubric['oneClaim']} vs NLP={claim_result}")
        if sufficient_result != current_rubric['selfSufficient']:
            mismatches.append(f"SelfSufficient: Manual={current_rubric['selfSufficient']} vs NLP={sufficient_result}")
        
        if mismatches:
            print(f"\nMISMATCHES:")
            for mismatch in mismatches:
                print(f"  • {mismatch}")
        else:
            print(f"\n✓ All criteria match between manual and NLP analysis")

def analyze_hook_details():
    """Analyze hook detection in detail."""
    print(f"\n" + "="*60)
    print("HOOK ANALYSIS DEEP DIVE")
    print("="*60)
    
    test_openings = [
        "Someone sent me a video of Bridget",  # Macron SLAP - NLP says False
        "Did you hear what AOC said?",  # AOC MELTDOWN - NLP says True  
        "what a crazy time all these politicians"  # Bernie Sanders - NLP says True
    ]
    
    for opening in test_openings:
        print(f"\nTesting: '{opening}'")
        
        # Manual hook score calculation
        opening_lower = opening.lower()
        hook_score = 0
        
        # Strong hook indicators
        hook_words = ['did you hear', 'crazy', 'amazing', 'wild', 'shocking', 'unbelievable', 
                      'what', 'there was', 'someone sent', 'breaking', 'just happened']
        
        matched_words = []
        for word in hook_words:
            if word in opening_lower:
                hook_score += 2
                matched_words.append(word)
        
        # Questions engage immediately
        questions = opening_lower.count('?')
        hook_score += questions * 1.5
        
        # Direct address
        audience_words = ['you', 'your', 'have you', 'did you']
        matched_audience = []
        for word in audience_words:
            if word in opening_lower:
                hook_score += 1
                matched_audience.append(word)
        
        print(f"  Hook words found: {matched_words}")
        print(f"  Audience words found: {matched_audience}")
        print(f"  Questions: {questions}")
        print(f"  Total hook score: {hook_score}")
        print(f"  Hook detected: {hook_score >= 4}")

if __name__ == "__main__":
    # Test the script analysis function
    func_result = test_script_analysis_function()
    
    # Test the API endpoint
    api_result = test_api_endpoint()
    
    # Compare results
    compare_results(func_result, api_result)
    
    # Run the original tests
    print("\n=== Running original NLP tests ===")
    test_specific_scripts()
    analyze_hook_details()