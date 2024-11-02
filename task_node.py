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
        self.reasoning = "" 
    @trace_function_calls
    def add_child(self, child_node):
        self.children.append(child_node)
        child_node.parent = self

    # TODO implement. Search on what's the best prompt to give to gpt to reason about a task
    @trace_function_calls
    def add_reasoning(self, reasoning_step):
        """
        Adds reasoning for a specific subtask to the context.
        Each entry in the context will include the subtask name and the associated reasoning.
        """
        # Append reasoning as a dictionary with subtask name and reasoning
        self.reasoning = reasoning_step
        self.context.append({"subtask": self.task_name, "reasoning": reasoning_step})
        # print(f"Context for task '{self.task_name}': {self.context}")
    
    @trace_function_calls
    def reason_through_task(self):
    # Generate reasoning for the task and add it to the context
            reasoning_prompt = f"Reason through the task: '{self.task_name}' given the current context: '{self.context}' , give reasoning step by step but make it 2-3 sentences each."
            response = call_groq_api(reasoning_prompt)
            reasoning_step = response.choices[0].message.content.strip()
            self.add_reasoning(reasoning_step)
              # Add reasoning to the task context

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
    def can_reuse_previous_reasoning(self):
        """
        Check if a task similar to the current one has already been reasoned through.
        If found, reuse the reasoning but proceed with execution.
        """
        cleaned_task_name = self.task_name.strip().lower()  # Normalize the task name
        for context_step in self.context:
            if cleaned_task_name in context_step.lower():
                print(f"Reusing previous reasoning for task: {cleaned_task_name}")
                return True
        return False
    

    def successful_execution(self):
        """
        Executes the task and returns True if the task completed successfully,
        False otherwise.
        """
        try:
            # Reasoning or subtask breakdown based on HTN planner logic
            self.reason_through_task()  # For generating reasoning steps
            if self.status == "succeeded" or "completed":  # Assuming status reflects task completion
                return True
            else:
                # If task is incomplete or failed, mark as failed
                self.status = "failed"
                return False
        except Exception as e:
            print(f"Error executing task '{self.task_name}': {e}")
            self.status = "failed"
            return False
        
    def flatten(self):
        """Recursively flattens all tasks and subtasks into a single list."""
        all_tasks = [self]
        for child in self.children:
            all_tasks.extend(child.flatten())
        return all_tasks    
    

    def get_context_score(self):
        """Returns a score representing how aligned the task context is with its execution."""
        if self.context:
            return min(1.0, len(self.context) / 10)  # Example score based on context size
        return 0.5  # Default score if no context exists

    def calculate_depth(self):
        """Calculates the depth of this task node within the tree structure."""
        depth = 0
        current_node = self
        while current_node.parent:
            depth += 1
            current_node = current_node.parent
        return depth

    def successful_execution(self):
        """Returns True if the task completed successfully, False otherwise."""
        try:
            if self.status == "succeeded" or self.status == "completed":
                return True
            else:
                self.status = "failed"
                return False
        except Exception as e:
            print(f"Error executing task '{self.task_name}': {e}")
            self.status = "failed"
            return False
        

     