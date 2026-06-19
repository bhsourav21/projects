from smartassist.prompts import system_prompt, user_prompts
from smartassist.agent import run_turn

messages = [{"role": "system", "content": system_prompt}]

for i, prompt in enumerate(user_prompts):
    messages.append({"role": "user", "content": prompt})
    print("--------------------------------------------")
    print(f"Turn {i + 1}")
    print(f"User     : {prompt}")
    run_turn(i, messages)

print("--------------------------------------------")
