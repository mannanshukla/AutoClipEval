from smolagents import CodeAgent, WebSearchTool, InferenceClientModel, tool
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

model = InferenceClientModel(token=os.getenv("HF_TOKEN"))

agent = CodeAgent(
    tools=[WebSearchTool(), get_shorts_scores], 
    model=model, 
    stream_outputs=True
)

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

agent.run("Get the shorts scores for the following script: " + test_script, additional_args={"file_path": Path(__file__).parent.parent.parent / "scores.json"})
