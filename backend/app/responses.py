from openai import OpenAI
import json
from pathlib import Path

client = OpenAI()

def get_shorts_scores(scores_path: str) -> str:
    with open(scores_path, "r") as f:
        return f.read()

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


tools = [{
    "type": "function",
    "name": "get_shorts_scores",
    "description": "Get a scoring of sample script snippets on their effectiveness in getting views.",
    "parameters": {
        "type": "object",
        "properties": {
            "scores_path": {
                "type": "string",
                "description": "Path to the scores.json file."
            }
        },
        "required": [
            "scores_path"
        ],
        "additionalProperties": False
    }
}]

input_messages = [
    {
        "role": "user",
        "content": "Get the shorts scores for the following script: " + test_script
    }
]

response = client.responses.create(
    model="gpt-4.1",
    tools=tools,
    input=input_messages
)

tool_call = response.output[0]
args = json.loads(tool_call.arguments)

scores = get_shorts_scores(args["scores_path"])

input_messages.append(tool_call)
input_messages.append({                               # append result message
    "type": "function_call_output",
    "call_id": tool_call.call_id,
    "output": str(scores)
})

tool_response = client.responses.create(
    model="gpt-4.1",
    tools=tools,
    input=input_messages,
)
    
print(tool_response.output_text)