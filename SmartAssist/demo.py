from smartassist.prompts import system_prompt
from smartassist.agent import run_turn

DEMO_PROMPTS = [
    "What is the date today?",
    """Station A and Station B are exactly 540 km apart.
    Train X leaves Station A at 8:00 AM traveling towards Station B at a speed of 60 km/hour.
    Train Y leaves Station A at 9:00 AM traveling towards Station A at a speed of 45 km/hour.
    At what time will the two trains cross each other, and exactly how far is the meeting point from Station A?
    Think step by step.""",
    "What are three benefits of drinking enough water every day?",
]

messages = [{"role": "system", "content": system_prompt}]

for i, prompt in enumerate(DEMO_PROMPTS):
    messages.append({"role": "user", "content": prompt})
    print("--------------------------------------------")
    print(f"Turn {i + 1}")
    print(f"User     : {prompt}")
    run_turn(i, messages)

print("--------------------------------------------")
