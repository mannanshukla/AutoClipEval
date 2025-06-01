import asyncio
from mcp import Tool, Server, run

# Define a simple echo tool
class EchoTool(Tool):
    name = "echo"
    description = "Echoes the input text."
    input_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to echo back."}
        },
        "required": ["text"]
    }

    async def call(self, input, context=None):
        return {"echo": input["text"]}

# Create the server and register the tool
server = Server()
server.add_tool(EchoTool())

if __name__ == "__main__":
    run(server) 