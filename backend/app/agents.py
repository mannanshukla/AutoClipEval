from smolagents import CodeAgent, WebSearchTool, InferenceClientModel, LiteLLMModel, tool
import os
import json
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

@tool
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

model = LiteLLMModel(
    model_id="openai/gpt-4.1",
    api_key=os.getenv("OPENAI_API_KEY"), 
)

validator_model = LiteLLMModel(
    model_id="openai/o4-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

agent = CodeAgent(
    tools=[WebSearchTool(), get_shorts_scores], 
    model=model, 
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

validator_agent = CodeAgent(
    tools=[WebSearchTool()], 
    model=validator_model, 
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


rubric_creator = """
SYSTEM:
You are an expert evaluation architect. Convert any problem description
and proposed solution into a measurable, weighted checklist.

USER:
[BEGIN_PROBLEM]
**Problem:** {{Insert text}}
[END_PROBLEM]

[BEGIN_SOLUTION]
**Proposed Solution:** {{Insert text}}
[END_SOLUTION]

TASK:
1. Extract 3-7 core *Objectives* the solution must satisfy.
2. For each objective, create 1-3 *Criteria* that are:
   • Specific & observable  
   • Measurable with a numeric or boolean indicator  
   • Achievable & relevant to the stated problem  
   • Time-bound or scoped (if applicable)
3. For every Criterion output:
   • `metric_name` (concise)  
   • `indicator` (how to measure)  
   • `scale` (e.g. 1-5, % success, ms latency)  
   • `target` (value that counts as “pass”)  
   • `weight` (low, medium, high)
4. Return a JSON object with keys:
   - `objectives` : list of strings
   - `rubric` : list of Criterion objects
5. After the JSON, render the same rubric as a Markdown
   table for human readability.

Ensure the JSON is valid and the sums of weights are correct.
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

goal = f"Using the youtube shorts scores as a reference, apply the same or better reasoning to the test script: {test_script}"
optimizer = "effectiveness in shortform reel performance"

result = agent.run(goal + f" | Generate a better scoring rubric for {optimizer} and evaluate the test script using the rubric creator: {rubric_creator}",additional_args={'file_path': Path(__file__).parent.parent.parent / 'scores.json'})

evaluation = validator_agent.run(f"Rate the scoring of the test script: {test_script} | Scoring: {result}" )
