from smartassist.models import Task, AgentResponse


def print_task_list(tasks: list[Task]):
    """Renders a list of Tasks as a formatted numbered table in the console."""
    print(f"  {'#':<5} {'Task'}")
    print(f"  {'-'*5} {'-'*50}")
    for task in tasks:
        print(f"  {task.task_number:<5} {task.task_description}")


def render_response(parsed: AgentResponse):
    """Dispatches to the right renderer based on the response_type discriminator."""
    if parsed.response_type == "task_list" and parsed.tasks:
        print_task_list(parsed.tasks)
    elif parsed.text:
        print(parsed.text)
    else:
        print("(No response was returned.)")
