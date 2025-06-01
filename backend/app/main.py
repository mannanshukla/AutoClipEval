#!/usr/bin/env python3
"""
FastAPI application for AutoClipEval - YouTube Shorts Content Evaluation API
Integrates with the agent-based evaluation system for comprehensive content analysis.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import json
from datetime import datetime
import traceback
import os
from pathlib import Path

# Import Pydantic models
from .models import (
    EvaluationRequest, 
    EvaluationResponse, 
    HealthResponse
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AutoClipEval API",
    description="YouTube Shorts Content Evaluation API using AI Agents",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Endpoints
@app.get("/", response_model=dict)
async def root():
    """Root endpoint to check server status"""
    return {
        "status": "online", 
        "message": "AutoClipEval API is running",
        "version": "1.0.0",
        "endpoints": ["/evaluate", "/health", "/scores"]
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with system status"""
    try:
        # Test agent system availability
        agent_status = "operational"
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now(),
            version="1.0.0",
            agent_status=agent_status
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now(),
            version="1.0.0",
            agent_status="error"
        )

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate_content(request: EvaluationRequest):
    """
    Content evaluation using the agent system.
    Processes the provided text and returns comprehensive evaluation results.
    """
    try:
        # Import agent evaluation system only when needed
        from .script_eval_agent import evaluate_script_api
        
        logger.info(f"Starting evaluation for text of length {len(request.text)}")
        
        # Run evaluation
        result = evaluate_script_api(
            text=request.text,
            max_iterations=request.max_iterations,
            target_score=request.target_score
        )
        
        # Convert to response format
        response_data = EvaluationResponse(**result)
        
        logger.info(f"Evaluation completed successfully. Final score: {result['final_score']}")
        return response_data
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        logger.error(traceback.format_exc())
        
        return EvaluationResponse(
            success=False,
            final_score=0,
            initial_score=0,
            iterations_completed=0,
            total_processing_time=0.0,
            iteration_history=[],
            final_analysis=f"Evaluation failed: {str(e)}",
            recommendations=["Check input text and try again"],
            target_achieved=False,
            error_message=str(e)
        )

@app.get("/scores")
async def get_scores():
    """Get all scores from the scores.json file for reference"""
    try:
        scores_path = Path(__file__).parent.parent.parent / "scores.json"
        with open(scores_path, "r") as f:
            scores = json.load(f)
        return {"scores": scores, "count": len(scores)}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="scores.json not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading scores: {str(e)}")

@app.get("/scores/{script_id}")
async def get_score(script_id: str):
    """Get the score for a specific script by ID from scores.json"""
    try:
        scores_path = Path(__file__).parent.parent.parent / "scores.json"
        with open(scores_path, "r") as f:
            scores = json.load(f)
        
        for score in scores:
            if score.get("id") == script_id:
                return score
                
        raise HTTPException(status_code=404, detail=f"Script with ID {script_id} not found")
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="scores.json not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading score: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)