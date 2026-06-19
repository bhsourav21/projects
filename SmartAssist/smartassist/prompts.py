# Instructs the model to signal response type via the discriminator field so rendering is dynamic
system_prompt = """
    Your responsibility is to respond to user queries based on your own knowledge or by leveraging the available tools.
    For specific query types like creating a task list, return structured JSON as:
    [
        {
            "task_number": <number>,
            "task_description": "<string>"
        }
    ]
    and set response_type to "task_list" with the tasks field populated.
    For all other queries, set response_type to "text" and populate the text field with your answer as a string.
"""

# The queries processed in sequence, each building on the shared conversation history
user_prompts = [
    # Tool: get_current_date
    "What is the date today?",
    # # Tool: calculate
    # "A boy has 10 mangoes. After 2 days he got 2 more mangoes but 5 mangoes in the previous list got rotten. How many good mangoes are there? Think step by step.",
    # # Structured output: task list
    # "Prepare a task list for visiting a Gynaecologist",
    # # Tool: web_search
    # "Search the web and tell me about the latest OpenAI models.",
    # # No tool: general knowledge
    # "What is the difference between a virus and a bacterium? Keep it simple.",
    # # Tool: calculate
    # "A pizza costs $14.50 and I am splitting it equally among 4 people. How much does each person pay?",
    # # No tool: general knowledge
    # "What are three benefits of drinking enough water every day?",
    # # Tool: web_search
    # "Search the web and give me a brief overview of the Python programming language.",
    # # Tool: get_current_date
    # "What year are we in right now? Use a tool to check.",
    # # No tool: opinion / advice
    # "Give me a one-paragraph study plan for learning machine learning from scratch.",
]
