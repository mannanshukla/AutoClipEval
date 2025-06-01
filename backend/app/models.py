from typing import Optional, List
from bson import ObjectId
from pydantic import BaseModel, Field, field_validator
from pydantic.functional_validators import BeforeValidator
from typing_extensions import Annotated
from datetime import datetime

# Pydantic models for MongoDB integration
def validate_object_id(v):
    if isinstance(v, ObjectId):
        return v
    if isinstance(v, str) and ObjectId.is_valid(v):
        return ObjectId(v)
    raise ValueError("Invalid ObjectId")

PyObjectId = Annotated[ObjectId, BeforeValidator(validate_object_id)]

class ItemModel(BaseModel):
    id: Optional[PyObjectId] = Field(default=None, alias="_id")
    name: str
    description: Optional[str] = None
    
    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
        "json_encoders": {ObjectId: str}
    }

class UpdateItemModel(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    
    model_config = {
        "arbitrary_types_allowed": True,
        "json_encoders": {ObjectId: str}
    }

# FastAPI models for content evaluation
class EvaluationRequest(BaseModel):
    """Request model for content evaluation"""
    text: str = Field(
        ..., 
        min_length=10,
        max_length=10000,
        description="The script/content text to evaluate"
    )
    max_iterations: Optional[int] = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum improvement iterations"
    )
    target_score: Optional[int] = Field(
        default=8,
        ge=1,
        le=10,
        description="Target score threshold for iterations"
    )
    
    @field_validator('text')
    def validate_text_content(cls, v):
        """Ensure text contains meaningful content"""
        if not v.strip():
            raise ValueError("Text cannot be empty or whitespace only")
        return v.strip()

class IterationResult(BaseModel):
    """Single iteration result"""
    iteration: int
    score: int
    rationale: str
    improvements: List[str] = []
    timestamp: str
    processing_time: Optional[float] = None
    result_output: Optional[str] = None
    rubric_score: Optional[int] = None

class EvaluationResponse(BaseModel):
    """Response model for content evaluation"""
    success: bool
    final_score: int
    initial_score: int
    iterations_completed: int
    total_processing_time: float
    iteration_history: List[IterationResult]
    final_analysis: str
    recommendations: List[str] = []
    final_result_content: Optional[str] = None
    rubric_final_score: Optional[int] = None
    target_achieved: bool = False
    error_message: Optional[str] = None

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: datetime
    version: str
    agent_status: str = "operational"
