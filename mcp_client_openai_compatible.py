"""
mcp_client_openai_compatible.py

A command-line MCP client that connects to a local MCP server, retrieves
available tools, and uses any OpenAI-compatible API to execute them.

What is "OpenAI-compatible"?
-----------------------------
OpenAI published a REST API specification for chat completions. Many services
implement the exact same HTTP endpoints and request/response formats, which
means you can point the official openai Python library at them just by changing
the base_url parameter. Examples:
  - Real OpenAI (api.openai.com)
  - GPUStack (a self-hosted GPU cluster manager with an OpenAI-compatible API)
  - LiteLLM, Ollama's OpenAI mode, vLLM, LocalAI, ...

This script works with all of them. Configuration is done entirely through
environment variables in a .env file so no code needs to change when you
switch providers.

Required .env variables:
  OPENAI_API_KEY  - The API key (for real OpenAI) or service token
  API_BASE_URL    - The base URL of the API  (omit for real OpenAI)
  MODEL_NAME      - The model to use (e.g. gpt-4o, gpt-oss-120b, ...)

Usage:
  python mcp_client_openai_compatible.py
"""

import asyncio
import os
import sys
import json
from typing import List, Dict, Any

# python-dotenv reads the .env file and injects its key=value pairs into
# os.environ so we can use os.getenv() throughout the script.
from dotenv import load_dotenv

# The official openai Python library. When you set openai.base_url to a
# custom endpoint it sends requests there instead of api.openai.com, making
# it work with GPUStack and every other compatible service.
from openai import OpenAI

# MCP client utilities - same as in the Ollama version.
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Load variables from .env into the process environment.
# override=False means existing environment variables take priority.
load_dotenv(override=False)

# Read each setting from the environment. Provide clear error messages when
# required values are missing so the user knows exactly what to fix.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("[Error] OPENAI_API_KEY is not set in your .env file.")
    print("        For real OpenAI: use your sk-... key from platform.openai.com")
    print("        For GPUStack / other: use the service token provided by the host")
    sys.exit(1)

# API_BASE_URL is optional for real OpenAI (the library defaults to
# https://api.openai.com/v1).  For self-hosted or alternative endpoints,
# set this to the root URL of the OpenAI-compatible service.
API_BASE_URL = os.getenv("API_BASE_URL")  # e.g. https://gpustack.unibe.ch/v1

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o")  # fallback to gpt-4o if not set


# ---------------------------------------------------------------------------
# OpenAI client setup
# ---------------------------------------------------------------------------

# Build keyword arguments for the OpenAI client.
# If API_BASE_URL is present we pass it as base_url; otherwise the library
# will use the official OpenAI endpoint automatically.
client_kwargs: Dict[str, Any] = {"api_key": OPENAI_API_KEY}
if API_BASE_URL:
    # Inform the user which endpoint is being used to avoid confusion.
    print(f"[Client] Using custom API endpoint: {API_BASE_URL}")
    client_kwargs["base_url"] = API_BASE_URL
else:
    print("[Client] Using the official OpenAI API (api.openai.com)")

# Create a single reusable OpenAI client instance.
openai_client = OpenAI(**client_kwargs)


# ---------------------------------------------------------------------------
# Helper: convert MCP tool definitions -> OpenAI tool schema format
# ---------------------------------------------------------------------------

def convert_mcp_tools_to_openai(mcp_tools) -> List[Dict[str, Any]]:
    """
    Convert the list of MCP Tool objects into the format the OpenAI API expects.

    OpenAI expects tools to look like this:
      {
        "type": "function",
        "function": {
          "name": "...",
          "description": "...",
          "parameters": { <JSON Schema object> }
        }
      }

    MCP Tool objects (mcp 2.x) expose:
      tool.name         - string
      tool.description  - string
      tool.input_schema - dict following JSON Schema (type, properties, required)
    """
    openai_tools = []
    for tool in mcp_tools:
        openai_tools.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                # input_schema is already a JSON-Schema dict - pass it directly.
                "parameters": tool.input_schema,
            }
        })
    return openai_tools


# ---------------------------------------------------------------------------
# Chat loop
# ---------------------------------------------------------------------------

async def chat_loop(session: ClientSession, openai_tools: List[Dict[str, Any]]) -> None:
    """
    Run the interactive chat loop.

    Flow for each user message:
      1. Append the user message to the conversation history.
      2. Send the full history + available tools to the model.
      3. If the model responds with tool_calls:
           a. Execute each requested tool via the MCP session.
           b. Append tool results to the history.
           c. Send the updated history back to the model for a final answer.
      4. If the model responds with plain text, print it directly.

    Args:
        session       - Active MCP ClientSession (used to call tools)
        openai_tools  - Tool schemas in OpenAI format (passed on every request)
    """
    # The messages list is the conversation history.  We keep appending to it
    # so the model has full context of the conversation on every turn.
    messages: List[Dict[str, Any]] = []

    print(f"\n[Client] Using model: {MODEL_NAME}")
    print("[Client] Ready! Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("quit", "exit"):
                print("[Client] Goodbye.")
                break

            # Step 1: Add the user's message to the history.
            messages.append({"role": "user", "content": user_input})

            print(f"[Client] Querying {MODEL_NAME}...")

            # Step 2: Call the OpenAI-compatible chat completions endpoint.
            # tool_choice="auto" lets the model decide whether to call a tool
            # or reply directly with text.
            response = openai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=openai_tools,
                tool_choice="auto",
            )

            # The API returns a list of choices; we always take the first one.
            choice = response.choices[0]
            assistant_message = choice.message

            # Step 3: Check whether the model wants to call tools.
            if assistant_message.tool_calls:
                print(f"[Client] Model requested {len(assistant_message.tool_calls)} tool call(s).")

                # Append the assistant turn (including the tool_calls field)
                # to the history BEFORE adding tool results.  The model needs
                # to see its own tool_calls turn when we send results back.
                messages.append(assistant_message)

                # Execute each tool call sequentially.
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name

                    # The model sends arguments as a JSON string; parse it into
                    # a Python dict before passing to the MCP session.
                    try:
                        tool_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        tool_args = {}

                    print(f"  [Tool Execution] Calling '{tool_name}' with args: {tool_args}")

                    try:
                        # Call the MCP server tool.  The session handles the
                        # JSON-RPC communication with the server process over stdio.
                        result = await session.call_tool(tool_name, arguments=tool_args)

                        # MCP returns a list of content blocks; join any text ones.
                        tool_result_text = "\n".join(
                            block.text for block in result.content if block.type == "text"
                        )
                    except Exception as exc:
                        tool_result_text = f"Error executing tool '{tool_name}': {exc}"

                    print(f"  [Tool Result] {tool_result_text}")

                    # Append the tool result to the history.
                    # The role must be "tool" and tool_call_id must match the
                    # id from the model's request so the model can correlate them.
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_name,
                        "content": tool_result_text,
                    })

                # Step 3c: Send the updated history (with tool results) back to
                # the model so it can compose a final human-readable response.
                print("[Client] Sending tool results back to the model...")
                follow_up = openai_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                )
                final_message = follow_up.choices[0].message
                messages.append(final_message)
                print(f"\nAssistant: {final_message.content}\n")

            else:
                # Step 4: Plain text response - no tools were called.
                messages.append(assistant_message)
                print(f"\nAssistant: {assistant_message.content}\n")

        except KeyboardInterrupt:
            print("\n[Client] Interrupted. Goodbye.")
            break
        except Exception as exc:
            print(f"\n[Client] An error occurred: {exc}\n")
            # We break on unexpected errors to avoid an infinite error loop.
            # Remove the 'break' if you prefer to continue the conversation
            # even after errors (e.g. transient network failures).
            break


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    """
    Locate the MCP server script, launch it as a subprocess over stdio,
    retrieve its tools, and start the chat loop.

    The MCP stdio transport works by spawning a child process (the server)
    and communicating with it through stdin/stdout using JSON-RPC messages.
    This keeps the server and client decoupled: the server could be written
    in any language, not just Python.
    """
    # Resolve the server script path relative to this file so the client
    # works regardless of the current working directory.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    server_script = os.path.join(current_dir, "mcp_server.py")

    if not os.path.exists(server_script):
        print(f"[Error] Could not find server script at: {server_script}")
        print("        Please ensure 'mcp_server.py' is in the same directory.")
        sys.exit(1)

    # StdioServerParameters tells the MCP library how to spawn the server.
    # We use sys.executable so the server runs inside the same virtualenv
    # as this client, ensuring it has access to all installed packages.
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_script],
        env=None,  # Inherit the current process environment
    )

    print(f"[Client] Connecting to MCP server at '{server_script}'...")

    # stdio_client is an async context manager that spawns the server process
    # and gives us read/write streams for the JSON-RPC communication.
    async with stdio_client(server_params) as (read, write):

        # ClientSession wraps the streams with the MCP protocol logic:
        # initialize handshake, tool listing, tool calling, etc.
        async with ClientSession(read, write) as session:

            # The initialize call performs the MCP handshake.  The server
            # sends its capabilities; the client acknowledges them.
            await session.initialize()

            # Ask the server to list all registered tools.
            tools_response = await session.list_tools()
            mcp_tools = tools_response.tools

            # Convert MCP tool definitions to OpenAI's expected format.
            openai_tools = convert_mcp_tools_to_openai(mcp_tools)

            print(f"[Client] Connected! Available tools: {[t.name for t in mcp_tools]}")

            # Hand off to the interactive chat loop.
            await chat_loop(session, openai_tools)


if __name__ == "__main__":
    asyncio.run(main())
