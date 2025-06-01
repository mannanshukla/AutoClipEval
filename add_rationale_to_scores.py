#!/usr/bin/env python3
"""
Script to add rationale explanations to existing scores.json entries.
This provides concise explanations for scoring decisions based on the existing rubric scores.
"""

import json
import os
from typing import Dict, Any


def generate_rationale(entry: Dict[str, Any]) -> str:
    """Generate a concise rationale based on the rubric scores."""
    rubric = entry.get("rubric", {})
    
    # Extract key scores
    hook = rubric.get("hook", False)
    one_claim = rubric.get("oneClaim", False)
    self_sufficient = rubric.get("selfSufficient", False)
    overall_score = rubric.get("overall_shorts_score", 0)
    viral_potential = rubric.get("viral_potential", "Low")
    
    # Numerical scores for more detailed analysis
    early_engagement = rubric.get("early_engagement", 0)
    main_idea_clarity = rubric.get("main_idea_clarity", 0)
    no_lulls = rubric.get("no_lulls", 0)
    payoff_strength = rubric.get("payoff_strength", 0)
    context_free_understanding = rubric.get("context_free_understanding", 0)
    distinctive_twist = rubric.get("distinctive_twist", 0)
    virality_potential_score = rubric.get("virality_potential", 0)
    engagement_potential = rubric.get("engagement_potential", 0)
    
    # Build rationale based on strengths and weaknesses
    rationale_parts = []
    
    # Overall assessment
    if overall_score >= 7:
        rationale_parts.append("Strong shorts potential")
    elif overall_score >= 5:
        rationale_parts.append("Moderate shorts potential")
    else:
        rationale_parts.append("Limited shorts potential")
    
    # Key strengths
    strengths = []
    if hook:
        strengths.append("strong hook")
    if early_engagement >= 8:
        strengths.append("high early engagement")
    if main_idea_clarity >= 8:
        strengths.append("clear main idea")
    if context_free_understanding >= 8:
        strengths.append("self-contained content")
    if distinctive_twist >= 7:
        strengths.append("distinctive angle")
    if virality_potential_score >= 7:
        strengths.append("viral elements")
    
    # Key weaknesses  
    weaknesses = []
    if not hook:
        weaknesses.append("weak hook")
    if not one_claim:
        weaknesses.append("multiple topics")
    if not self_sufficient:
        weaknesses.append("requires context")
    if no_lulls <= 3:
        weaknesses.append("pacing issues")
    if payoff_strength <= 4:
        weaknesses.append("weak payoff")
    if engagement_potential <= 4:
        weaknesses.append("low engagement potential")
    
    # Construct rationale
    if strengths:
        rationale_parts.append("due to " + ", ".join(strengths[:3]))  # Limit to top 3 strengths
    
    if weaknesses and overall_score < 7:
        rationale_parts.append("but limited by " + ", ".join(weaknesses[:2]))  # Limit to top 2 weaknesses
    
    # Add viral potential context
    if viral_potential == "High":
        rationale_parts.append("High viral potential from controversial/engaging topic")
    elif viral_potential == "Medium":
        rationale_parts.append("Medium viral potential")
    elif overall_score < 5:
        rationale_parts.append("Low viral potential")
    
    return ". ".join(rationale_parts) + "."


def add_rationale_to_scores_file(scores_file_path: str):
    """Add rationale to each entry in the scores.json file."""
    try:
        # Read existing scores
        with open(scores_file_path, 'r', encoding='utf-8') as f:
            scores_data = json.load(f)
        
        print(f"Processing {len(scores_data)} entries...")
        
        # Add rationale to each entry
        for entry in scores_data:
            if "rationale" not in entry.get("rubric", {}):
                rationale = generate_rationale(entry)
                entry["rubric"]["rationale"] = rationale
                print(f"Added rationale for: {entry.get('id', 'Unknown')}")
            else:
                print(f"Rationale already exists for: {entry.get('id', 'Unknown')}")
        
        # Write back to file with proper formatting
        with open(scores_file_path, 'w', encoding='utf-8') as f:
            json.dump(scores_data, f, indent=2, ensure_ascii=False)
        
        print(f"Successfully updated {scores_file_path}")
        
    except FileNotFoundError:
        print(f"Error: File {scores_file_path} not found")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in {scores_file_path}: {e}")
    except Exception as e:
        print(f"Error processing file: {e}")


def main():
    """Main function to add rationale to scores.json."""
    # Try both possible locations for scores.json
    possible_paths = [
        "/home/rohit/sandbox/AutoClipEval/backend/scores.json",
        "/home/rohit/sandbox/AutoClipEval/scores.json"
    ]
    
    scores_file = None
    for path in possible_paths:
        if os.path.exists(path):
            scores_file = path
            break
    
    if not scores_file:
        print("Error: scores.json file not found in expected locations")
        print("Searched paths:")
        for path in possible_paths:
            print(f"  - {path}")
        return
    
    print(f"Found scores.json at: {scores_file}")
    add_rationale_to_scores_file(scores_file)


if __name__ == "__main__":
    main()
