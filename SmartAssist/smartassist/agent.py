import json
from openai import APIError, RateLimitError, APIConnectionError

from smartassist.config import client, MODEL, MAX_TOKENS
from smartassist.tools import TOOL_SCHEMAS, TOOL_MAP
from smartassist.models import AgentResponse
from smartassist.renderer import render_response


def handle_tool_calls(response_message, messages):
    """Executes each tool the model requested and appends the results to the message history."""
    for idx, tool_call in enumerate(response_message.tool_calls):
        fn_name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments) if tool_call.function.arguments else {}
        except json.JSONDecodeError as e:
            args = {}
            print(f"  [Warning] Could not parse arguments for '{fn_name}': {e}")

        if fn_name not in TOOL_MAP:
            result = f"Error: unknown tool '{fn_name}'"
            print(f"  [Warning] Model called unknown tool: {fn_name}")
        else:
            try:
                result = str(TOOL_MAP[fn_name](args))
            except Exception as e:
                result = f"Error executing '{fn_name}': {e}"
                print(f"  [Warning] Tool '{fn_name}' failed: {e}")

        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        })


def run_turn(i: int, messages: list):
    """Runs a single conversation turn: calls the model, handles tool calls if any, and renders the response."""
    try:
        # First call — may result in tool use or a direct structured response
        response = client.beta.chat.completions.parse(
            model=MODEL,
            messages=messages,
            tools=TOOL_SCHEMAS,
            response_format=AgentResponse,
            max_tokens=MAX_TOKENS,
        )
        msg = response.choices[0].message

        if msg.tool_calls:
            messages.append(msg)
            handle_tool_calls(msg, messages)
            # Follow-up call after tool results so the model can produce the final structured response
            followup = client.beta.chat.completions.parse(
                model=MODEL,
                messages=messages,
                response_format=AgentResponse,
                max_tokens=MAX_TOKENS,
            )
            parsed = followup.choices[0].message.parsed
            messages.append({"role": "assistant", "content": followup.choices[0].message.content})
        else:
            parsed = msg.parsed
            messages.append({"role": "assistant", "content": msg.content})

        if parsed is None:
            print("Assistant: [Error] Model returned empty structured response.")
        else:
            render_response(parsed)

    except RateLimitError:
        print(f"  [Error] Rate limit hit on turn {i + 1} — skipping.")
    except APIConnectionError as e:
        print(f"  [Error] Connection failed on turn {i + 1}: {e} — skipping.")
    except APIError as e:
        print(f"  [Error] OpenAI API error on turn {i + 1}: {e} — skipping.")
