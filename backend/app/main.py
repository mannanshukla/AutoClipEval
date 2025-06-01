from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import json
import os
from openai import AsyncOpenAI, OpenAI

app = FastAPI(title="AutoClipEval Server", description="API for evaluating YouTube Shorts scripts")

# Initialize OpenAI client
client = OpenAI()

class ScoreRequest(BaseModel):
    """Request model for scoring a script"""
    script: str
    model: Optional[str] = "gpt-4.1-nano"


class ScoreResponse(BaseModel):
    """Response model for a scored script"""
    rubric: Dict[str, Any]
    analysis_explanations: Optional[Dict[str, str]] = None


@app.get("/")
async def root():
    """Root endpoint to check server status"""
    return {"status": "online", "message": "AutoClipEval API is running"}


@app.get("/scores")
async def get_scores():
    """Get all scores from the scores.json file"""
    try:
        with open("scores.json", "r") as f:
            scores = json.load(f)
        return scores
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="scores.json not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading scores: {str(e)}")


@app.post("/score", response_model=ScoreResponse)
async def score_script(request: ScoreRequest = Body(...)):
    """Score a script using OpenAI's API with structured output"""
    try:
        # Define the structured output schema
        structured_output_schema = {
            "type": "object",
            "properties": {
                "hook_score": {
                    "type": "integer",
                    "description": "Score from 1-10 for how well it grabs attention in first 10 seconds",
                    "minimum": 1,
                    "maximum": 10
                },
                "virality_score": {
                    "type": "integer",
                    "description": "Score from 1-10 for emotional impact, surprising elements, shareability",
                    "minimum": 1,
                    "maximum": 10
                },
                "clarity_score": {
                    "type": "integer",
                    "description": "Score from 1-10 for focus on one clear message/claim vs multiple topics",
                    "minimum": 1,
                    "maximum": 10
                },
                "self_sufficiency_score": {
                    "type": "integer",
                    "description": "Score from 1-10 for ability to be understood without prior context",
                    "minimum": 1,
                    "maximum": 10
                },
                "engagement_score": {
                    "type": "integer",
                    "description": "Score from 1-10 for likelihood to get likes, comments, shares",
                    "minimum": 1,
                    "maximum": 10
                },
                "no_lulls_score": {
                    "type": "integer",
                    "description": "Score from 1-10 for maintaining consistent energy without boring parts",
                    "minimum": 1,
                    "maximum": 10
                },
                "payoff_strength_score": {
                    "type": "integer",
                    "description": "Score from 1-10 for how satisfying the conclusion or key moment is",
                    "minimum": 1,
                    "maximum": 10
                },
                "distinctive_twist_score": {
                    "type": "integer",
                    "description": "Score from 1-10 for having unique/unexpected elements that set it apart",
                    "minimum": 1,
                    "maximum": 10
                },
                "has_strong_hook": {
                    "type": "boolean",
                    "description": "Whether the content has a strong hook in the beginning"
                },
                "has_one_claim": {
                    "type": "boolean",
                    "description": "Whether the content focuses on one clear claim"
                },
                "is_self_sufficient": {
                    "type": "boolean",
                    "description": "Whether the content can be understood without prior context"
                },
                "hook_explanation": {
                    "type": "string",
                    "description": "Brief explanation of the hook score"
                },
                "virality_explanation": {
                    "type": "string",
                    "description": "Brief explanation of the virality score"
                },
                "overall_analysis": {
                    "type": "string",
                    "description": "Brief overall assessment of the content"
                }
            },
            "required": [
                "hook_score", "virality_score", "clarity_score", "self_sufficiency_score", 
                "engagement_score", "no_lulls_score", "payoff_strength_score", "distinctive_twist_score",
                "has_strong_hook", "has_one_claim", "is_self_sufficient",
                "hook_explanation", "virality_explanation", "overall_analysis"
            ]
        }
        
        # System prompt for the model
        system_prompt = """You are an expert content analyst specializing in YouTube Shorts and viral social media content.
Your job is to analyze transcript text and evaluate its potential as a YouTube Short.
Focus on engagement factors like hook strength, emotional impact, clarity, and viral potential.
Provide numerical scores (1-10) and brief explanations for your ratings."""

        # Sending request to OpenAI
        response = client.chat.completions.create(
            model=request.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"""Analyze this transcript for YouTube Shorts potential across multiple dimensions:

Transcript: "{request.script}"

Please evaluate and provide scores (1-10) for each aspect:

1. HOOK STRENGTH: How well does it grab attention in first 10 seconds?
2. VIRALITY POTENTIAL: Emotional impact, surprising elements, shareability
3. CLARITY: Focus on one clear message/claim vs multiple topics
4. SELF-SUFFICIENCY: Can be understood without prior context
5. ENGAGEMENT: Likely to get likes, comments, shares
6. NO LULLS: Maintains consistent energy without boring parts
7. PAYOFF STRENGTH: How satisfying the conclusion or key moment is
8. DISTINCTIVE TWIST: Has unique/unexpected elements that set it apart

Also determine these boolean values:
- Has strong hook (true/false)
- Focuses on one claim (true/false) 
- Is self-sufficient (true/false)

Provide brief explanations for hook strength and virality potential, and an overall analysis."""}
            ],
            temperature=0.1,
            response_format={"type": "json_object", "schema": structured_output_schema}
        )
        
        # Parse the response
        result = json.loads(response.choices[0].message.content)
        
        # Calculate overall score (weighted average)
        weights = {
            'hook_score': 0.15,      # Critical for Shorts
            'has_strong_hook': 0.15,  # Critical for Shorts
            'virality_score': 0.15,    # Key metric
            'distinctive_twist_score': 0.10,     # Important for standing out
            'payoff_strength_score': 0.10,       # Important for completion
            'clarity_score': 0.10,     # Important for comprehension
            'engagement_score': 0.10,  # Important for shares/likes
            'no_lulls_score': 0.05,    # Good for retention
            'self_sufficiency_score': 0.05, # Good for new viewers
            'has_one_claim': 0.03,     # Helpful but less critical
            'is_self_sufficient': 0.02  # Helpful but less critical
        }
        
        # Calculate weighted score - convert booleans to 10/0
        weighted_score = 0
        for criterion, weight in weights.items():
            if criterion in result:
                if isinstance(result[criterion], bool):
                    value = 10 if result[criterion] else 0
                else:
                    value = result[criterion]
                weighted_score += value * weight
        
        # Format the response
        rubric = {
            # Boolean criteria
            "hook": result["has_strong_hook"],
            "oneClaim": result["has_one_claim"],
            "selfSufficient": result["is_self_sufficient"],
            
            # Numeric criteria (1-10 scale)
            "early_engagement": result["hook_score"],
            "main_idea_clarity": result["clarity_score"],
            "no_lulls": result["no_lulls_score"],
            "payoff_strength": result["payoff_strength_score"],
            "context_free_understanding": result["self_sufficiency_score"],
            "distinctive_twist": result["distinctive_twist_score"],
            "virality_potential": result["virality_score"],
            "engagement_potential": result["engagement_score"],
            
            # Overall score
            "overall_shorts_score": round(weighted_score, 1),
            
            # Viral classification
            "viral_potential": "High" if weighted_score >= 7.5 else "Medium" if weighted_score >= 5.5 else "Low"
        }
        
        # Add explanations
        analysis_explanations = {
            "hook": result["hook_explanation"],
            "virality": result["virality_explanation"],
            "overall": result["overall_analysis"]
        }
        
        return {
            "rubric": rubric,
            "analysis_explanations": analysis_explanations
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scoring script: {str(e)}")


@app.get("/scores/{script_id}")
async def get_score(script_id: str):
    """Get the score for a specific script by ID"""
    try:
        with open("scores.json", "r") as f:
            scores = json.load(f)
        
        for score in scores:
            if score.get("id") == script_id:
                return score
                
        raise HTTPException(status_code=404, detail=f"Script with ID {script_id} not found")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="scores.json not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading score: {str(e)}")


@app.get("/tools/get_shorts_scores")
async def get_shorts_scores(scores_path: str):
    """Get the scores from a JSON file - compatible with MCP server"""
    try:
        with open(scores_path, "r") as f:
            return f.read()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"File {scores_path} not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)