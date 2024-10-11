import uuid

from text_utils import trace_function_calls
from LLM_api import call_groq_api


class TaskNode:
    def __init__(self, task_name, parent=None, status="pending"):
        self.task_name = task_name
        self.node_name = str(uuid.uuid4())
        self.parent = parent
        self.children = []
        self.status = status
        self.context = []
    @trace_function_calls
    def add_child(self, child_node):
        self.children.append(child_node)
        child_node.parent = self

    # TODO implement. Search on what's the best prompt to give to gpt to reason about a task
    @trace_function_calls
    def add_reasoning(self, reasoning_step):
        self.context.append(reasoning_step)
        print(f"Context for task '{self.task_name}': {self.context}")
   
   
    @trace_function_calls
    def reason_through_task(self):
    # Generate reasoning for the task and add it to the context
            reasoning_prompt = f"Reason through the task: '{self.task_name}' given the current context: '{self.context}' , give reasoning step by step."
            response = call_groq_api(reasoning_prompt)
            reasoning_step = response.choices[0].message.content.strip()
            self.add_reasoning(reasoning_step)  # Add reasoning to the task context

    @trace_function_calls
    def remove_child(self, child_node):
        if child_node in self.children:
            self.children.remove(child_node)
            child_node.parent = None

    @trace_function_calls
    def update_task_name(self, task_name):
        self.task_name = task_name

    def all_children_succeeded(self):
        return all(child.status == 'succeeded' for child in self.children)

    def mark_as_succeeded(self):
        if self.all_children_succeeded():
            self.status = 'succeeded'
        else:
            print(f"Cannot mark {self.task_name} as succeeded because some children tasks are still pending.")