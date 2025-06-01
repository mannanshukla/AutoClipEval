import os
import re
from dotenv import load_dotenv
from pathlib import Path
from pydantic import BaseModel

from rich.markdown import Markdown
from rich.console import Console

console = Console()

load_dotenv()

class Evaluation(BaseModel):
    score: int = 0
    rationale: str = ""


def score_rubric(text: str) -> str:
    """
    Score content using the enhanced rubric criteria for YouTube Shorts with fast configuration.
    Uses rule-based analysis combined with LLM inference for comprehensive evaluation.
    
    Args:
        text (str): The script/content text to analyze and score
        
    Returns:
        str: JSON-formatted scoring results including rubric scores, rationale, and analysis
    """
    import asyncio
    import json
    from score_rubric import process_text_async
    
    # Use fast configuration with default model
    result = asyncio.run(process_text_async(
        text=text,
        openai_model="gpt-4.1-nano",  # Fast, optimized model
        api_key=os.getenv("OPENAI_API_KEY")
    ))
    
    return json.dumps(result, indent=2)

def get_shorts_scores(file_path: str) -> str:
    """
    Returns the contents of the scores.json file.
    Args:
        file_path (str): Path to the scores.json file.
    Returns:
        str: Contents of the scores.json file.
    """
    with open(file_path, "r") as f:
        return f.read()

def get_score_rubric_source() -> str:
    """
    Returns the complete source code of the score_rubric.py file for analysis and evaluation.
    
    Returns:
        str: The complete source code of score_rubric.py
    """
    score_rubric_path = Path(__file__).parent / 'score_rubric.py'
    try:
        with open(score_rubric_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "Error: score_rubric.py file not found"
    except Exception as e:
        return f"Error reading score_rubric.py: {str(e)}"

# Global agent instances - will be created lazily
_agent = None
_validator_agent = None
_model = None
_validator_model = None

def get_model():
    """Lazily create and return the main model instance"""
    global _model
    if _model is None:
        from smolagents import LiteLLMModel
        _model = LiteLLMModel(
            model_id="openai/gpt-4.1",
            api_key=os.getenv("OPENAI_API_KEY"), 
        )
    return _model

def get_validator_model():
    """Lazily create and return the validator model instance"""
    global _validator_model
    if _validator_model is None:
        from smolagents import LiteLLMModel
        _validator_model = LiteLLMModel(
            model_id="openai/o4-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    return _validator_model

def get_agent():
    """Lazily create and return the main agent instance"""
    global _agent
    if _agent is None:
        from smolagents import CodeAgent, WebSearchTool, tool
        
        # Apply tool decorator to functions
        global score_rubric, get_shorts_scores, get_score_rubric_source
        score_rubric = tool(score_rubric)
        get_shorts_scores = tool(get_shorts_scores)
        get_score_rubric_source = tool(get_score_rubric_source)
        
        _agent = CodeAgent(
            tools=[WebSearchTool(), get_shorts_scores, score_rubric, get_score_rubric_source], 
            model=get_model(), 
            stream_outputs=True,
            additional_authorized_imports=[
                "json",
                "os",
                "ast",
                "pathlib",
                "dotenv",
                "pydantic"
            ],
            use_structured_outputs_internally=True
        )
    return _agent

def get_validator_agent():
    """Lazily create and return the validator agent instance"""
    global _validator_agent
    if _validator_agent is None:
        from smolagents import CodeAgent
        _validator_agent = CodeAgent(
            tools=[], 
            model=get_validator_model(), 
            stream_outputs=True,
            additional_authorized_imports=[
                "json",
                "os",
                "ast",
                "pathlib",
                "dotenv",
                "pydantic"
            ],
            use_structured_outputs_internally=True
        )
    return _validator_agent


rubric_evaluator = """
SYSTEM:
You are an expert evaluation architect and rubric analyst. You specialize in critically analyzing, validating, and improving evaluation frameworks to ensure they are comprehensive, fair, and effective.

USER:
[BEGIN_RUBRIC_ANALYSIS]
**Rubric to Evaluate:** {{Insert rubric structure}}
**Evaluation Context:** {{Insert domain/purpose}}
**Sample Evaluations:** {{Insert examples if available}}
[END_RUBRIC_ANALYSIS]

TASK:
Thoroughly critique and evaluate the provided rubric across these dimensions:

## 1. CONSTRUCT VALIDITY
- **Criterion Coverage**: Does the rubric capture all essential aspects of the evaluation domain?
- **Logical Coherence**: Are criteria logically related and non-contradictory?
- **Domain Alignment**: How well do metrics align with the stated evaluation purpose?

## 2. MEASUREMENT QUALITY
- **Scale Appropriateness**: Are scoring scales (boolean, numeric, categorical) optimal for each criterion?
- **Granularity**: Is the level of detail appropriate (not too coarse/fine-grained)?
- **Discriminatory Power**: Can the rubric effectively differentiate between high/low performance?
- **Inter-rater Reliability**: Would different evaluators reach similar conclusions?

## 3. PRACTICAL USABILITY
- **Clarity**: Are criteria definitions unambiguous and actionable?
- **Efficiency**: Is the rubric feasible to apply without excessive time/effort?
- **Weighting Logic**: Are relative importance weights justified and balanced?
- **Edge Case Handling**: How well does it handle outliers or unusual cases?

## 4. BIAS AND FAIRNESS
- **Cultural Bias**: Are criteria culturally neutral or appropriately contextualized?
- **Systematic Bias**: Do any criteria systematically favor certain approaches/styles?
- **Accessibility**: Are evaluation standards accessible across different skill levels?

## 5. IMPROVEMENT RECOMMENDATIONS
Provide specific, actionable suggestions for:
- **Missing Criteria**: What important aspects are not captured?
- **Redundant Elements**: What can be consolidated or removed?
- **Scale Optimization**: How can scoring methods be improved?
- **Weight Rebalancing**: Should relative importance be adjusted?

## OUTPUT FORMAT:
```json
{
  "validity_score": 1-10,
  "measurement_quality": 1-10,
  "usability_score": 1-10,
  "fairness_score": 1-10,
  "overall_rating": 1-10,
  "strengths": ["list", "of", "key", "strengths"],
  "critical_weaknesses": ["list", "of", "major", "issues"],
  "improvement_recommendations": [
    {
      "category": "missing_criteria|redundancy|scaling|weighting|clarity",
      "priority": "high|medium|low",
      "suggestion": "specific recommendation",
      "rationale": "justification for change"
    }
  ],
  "revised_rubric_outline": "optional improved version"
}
```

## ANALYSIS PRINCIPLES:
- Be thorough but concise in your critique
- Ground recommendations in evaluation theory and best practices
- Consider the specific domain context (e.g., content evaluation, performance assessment)
- Balance comprehensiveness with practical usability
- Identify both strengths to preserve and weaknesses to address
"""

test_script = """
🎙️ The Joe Rogan Experience #2999: Donald Duck
[INTRO MUSIC PLAYS – same thumping bass intro]

Joe Rogan:
What's up freak bitches. My guest today is someone I’ve been wanting to talk to for a long time… cultural icon, war veteran, possible MK Ultra experiment... the one and only — Donald freaking Duck.

Donald, what’s up man?
Donald Duck:
(incoherent angry quacking)

Joe (laughing):
Dude I love it. I can't understand a goddamn word you're saying but it’s powerful. You’ve got presence. There’s something primal about it.

Joe:
So listen, you fought in World War II, right?

Donald Duck:
(squawking proudly, pulls out a crumpled propaganda poster of himself in uniform)

Joe:
This is wild. People don’t talk about this — but Disney actually weaponized their characters during wartime. You were in Der Fuehrer’s Face. That shit was full-blown psychological warfare.

Donald Duck:
(nods furiously, pulls out flask labeled "Acme Ether")

Joe:
I gotta ask… were you on something during that era? Some of that early government LSD? Be honest, Don.

Donald Duck:
(furious muffled rant)
[subtitles: “WAS I? THEY FED ME DUCKSPEED FOR FOUR YEARS!”]

Joe:
laughs hysterically DUCKSPEED. Bro that’s better than Alpha Brain.
"""

rubric = None
rubric_summary = ""
result_score = 0
rubric_score = 0

while result_score < 5 or rubric_score < 5:
    goal = f"Using the youtube shorts scores as a reference and the rubric - {rubric} - if the rubric is not non-existent, apply the same or better reasoning to the test script: {test_script}. Format your scoring as a markdown table in the GitHub markdown spec. Only return the Markdown table."
    optimizer = "effectiveness in shortform reel performance"

    result = get_agent().run(goal ,additional_args={'file_path': Path(__file__).parent.parent.parent / 'scores.json'})

    result_evaluation = get_validator_agent().run(f"Evaluate the scoring: {result} in measuring {optimizer} in a range from 1-10 w/ a concise rationale in the format `Score: <score> | Rationale: <rationale>`.")

    if not rubric:
        rubric_summary = get_agent().run("Retrieve the scoring rubric source code and provide an outlined summary of the rubric")

    instructions = "Evaluate the existing scoring rubric implementation" + (str(rubric) if rubric else str(rubric_summary)) + f"and suggest improvements using the rubric evaluator:- {rubric_evaluator}. Analyze and provide a score from 1-10 with a concise rationale in the format `Score: <score> | Rationale: <rationale>`."
    
    rubric_evaluation = get_validator_agent().run(instructions)


    result_score_match = re.search(r'Score:\s*(\d+)', str(result_evaluation))
    result_score = int(result_score_match.group(1)) if result_score_match else 0
    
    rubric_score_match = re.search(r'Score:\s*(\d+)', str(rubric_evaluation))
    rubric_score = int(rubric_score_match.group(1)) if rubric_score_match else 0

    if result_score < 5 and rubric_score >= 5:
        print("Summary: The generated script did not meet the quality standards.")
        console.print(Markdown(str(result)))
        break

    elif result_score >= 5 and rubric_score >= 5:
        print("Summary: The generated script met the quality standards.")
        console.print(Markdown(str(result)))
        break

    if rubric_score < 5:
        response = get_agent().run(f"Based on this assessment - {rubric_evaluation} - improve the existing rubric - {rubric} - to better evaluate the script. Provide a revised rubric outline")
        rubric = str(response)

# Add API integration function at the end of the file
def evaluate_script_api(text: str, max_iterations: int = 3, target_score: int = 8) -> dict:
    """
    API wrapper function for agent evaluation system.
    
    Args:
        text (str): The script/content text to evaluate
        max_iterations (int): Maximum improvement iterations (default: 3)
        target_score (int): Target score threshold for iterations (default: 8)
    
    Returns:
        dict: Comprehensive evaluation results including iterations, scores, and analysis
    """
    import time
    import json
    from datetime import datetime
    
    start_time = time.time()
    iterations = []
    initial_score = 0
    final_score = 0
    rubric = None
    rubric_summary = ""
    result_score = 0
    rubric_score = 0
    iteration_count = 0
    
    # Score patterns for regex parsing
    score_pattern = r'Score:\s*(\d+)'
    rationale_pattern = r'Rationale:\s*(.+?)(?:\n|$)'
    
    try:
        while (result_score < target_score or rubric_score < 5) and iteration_count < max_iterations:
            iteration_count += 1
            iteration_start = time.time()
            
            goal = f"Using the youtube shorts scores as a reference and the rubric - {rubric} - if the rubric is not non-existent, apply the same or better reasoning to the test script: {text}. Format your scoring as a markdown table in the GitHub markdown spec. Only return the Markdown table."
            optimizer = "effectiveness in shortform reel performance"

            # Run agent evaluation
            result = get_agent().run(goal, additional_args={'file_path': Path(__file__).parent.parent.parent / 'scores.json'})

            # Validate the result
            result_evaluation = get_validator_agent().run(f"Evaluate the scoring: {result} in measuring {optimizer} in a range from 1-10 w/ a concise rationale in the format `Score: <score> | Rationale: <rationale>`.")

            # Get rubric summary on first iteration
            if not rubric and iteration_count == 1:
                rubric_summary = get_agent().run("Retrieve the scoring rubric source code and provide an outlined summary of the rubric")

            # Evaluate rubric
            instructions = "Evaluate the existing scoring rubric implementation" + (str(rubric) if rubric else str(rubric_summary)) + f"and suggest improvements using the rubric evaluator:- {rubric_evaluator}. Analyze and provide a score from 1-10 with a concise rationale in the format `Score: <score> | Rationale: <rationale>`."
            
            rubric_evaluation = get_validator_agent().run(instructions)

            # Parse scores using regex
            result_score_match = re.search(score_pattern, str(result_evaluation))
            result_score = int(result_score_match.group(1)) if result_score_match else 0
            
            rubric_score_match = re.search(score_pattern, str(rubric_evaluation))
            rubric_score = int(rubric_score_match.group(1)) if rubric_score_match else 0
            
            # Parse rationale
            result_rationale_match = re.search(rationale_pattern, str(result_evaluation))
            result_rationale = result_rationale_match.group(1).strip() if result_rationale_match else "No rationale provided"
            
            # Store iteration results
            iteration_result = {
                "iteration": iteration_count,
                "score": result_score,
                "rationale": result_rationale,
                "improvements": [],
                "timestamp": datetime.now().isoformat(),
                "processing_time": time.time() - iteration_start,
                "result_output": str(result),
                "rubric_score": rubric_score
            }
            
            # Set initial score on first iteration
            if iteration_count == 1:
                initial_score = result_score
            
            iterations.append(iteration_result)
            
            # Check completion conditions
            if result_score < target_score and rubric_score >= 5:
                final_score = result_score
                break
            elif result_score >= target_score and rubric_score >= 5:
                final_score = result_score
                break
            
            # Improve rubric if needed
            if rubric_score < 5:
                response = get_agent().run(f"Based on this assessment - {rubric_evaluation} - improve the existing rubric - {rubric} - to better evaluate the script. Provide a revised rubric outline")
                rubric = str(response)
        
        final_score = result_score if result_score > 0 else initial_score
        total_time = time.time() - start_time
        
        # Generate final analysis and recommendations
        final_analysis = f"Evaluation completed after {iteration_count} iterations. Final score: {final_score}/{target_score}"
        recommendations = []
        
        if final_score < target_score:
            recommendations.append(f"Consider improving content to reach target score of {target_score}")
        if rubric_score < 5:
            recommendations.append("Rubric evaluation indicates scoring framework needs improvement")
        
        # Extract the final result content for analysis
        final_result_content = iterations[-1]["result_output"] if iterations else ""
        
        return {
            "success": True,
            "final_score": final_score,
            "initial_score": initial_score,
            "iterations_completed": iteration_count,
            "total_processing_time": round(total_time, 2),
            "iteration_history": iterations,
            "final_analysis": final_analysis,
            "recommendations": recommendations,
            "final_result_content": final_result_content,
            "rubric_final_score": rubric_score,
            "target_achieved": final_score >= target_score,
            "error_message": None
        }
        
    except Exception as e:
        return {
            "success": False,
            "final_score": 0,
            "initial_score": 0,
            "iterations_completed": iteration_count,
            "total_processing_time": round(time.time() - start_time, 2),
            "iteration_history": iterations,
            "final_analysis": f"Evaluation failed: {str(e)}",
            "recommendations": ["Check input text and try again"],
            "final_result_content": "",
            "rubric_final_score": 0,
            "target_achieved": False,
            "error_message": str(e)
        }