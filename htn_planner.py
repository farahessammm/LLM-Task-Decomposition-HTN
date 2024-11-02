
from pyexpat import model
from LLM_utils import calculate_similarity, groq_is_goal, is_task_primitive, can_execute, log_state_change
from LLM_api import call_groq_api, log_response
from task_node import TaskNode 
from text_utils import extract_lists, trace_function_calls
from htn_prompts import *
from vector_db import VectorDB
import json
import os
from task_history_db import TaskHistoryDB
import time
import statistics


class HTNPlanner:

    def __init__(self, goal_input, initial_state, goal_task, capabilities_input, max_depth=7, send_update_callback=None, task_history_file = 'task_history_file.json' , threshold=0.3):
        self.goal_input = goal_input
        self.initial_state = initial_state
        self.goal_task = goal_task
        self.capabilities_input = capabilities_input
        self.max_depth = max_depth
        self.send_update_callback = send_update_callback
        self.task_history_db = TaskHistoryDB()
        self.current_run_tasks = []
        self.repetition_count = {}
        self.threshold = threshold 
        self.context_loss_instances = 0
        self.branches_children_count = []
        self.branch_depths = []
        self.context_loss_depths = []
        self.success_count = 0
        self.total_runs = 0 

    def add_to_current_run_tasks(self, task_node):
        """
        Adds the task to the current run task list to track tasks executed in this run.
        Ensures each task is added only once. If the task is repeated, retrieves and applies previous context.
        """
        task_name_lower = task_node.task_name.strip().lower()
        
        if task_name_lower in self.current_run_tasks:
            # Add the task to current run tasks if not already present
            print(f"Task '{task_name_lower}' is already in the current run tasks list. Retrieving and applying previous context.")
            # Retrieve and apply previous context for the repeated task
            # self.retrieve_and_apply_previous_context(task_name_lower, task_node)
            self.repetition_count[task_name_lower] = self.repetition_count.get(task_name_lower, 0) + 1

            
        else:
            self.current_run_tasks.append(task_name_lower)
            self.repetition_count[task_name_lower] = 1
            # print(f"Current run tasks: {', '.join(self.current_run_tasks)}")
            print(f"Task '{task_name_lower}' added to current run tasks list with repetition count {self.repetition_count[task_name_lower]}")
            # Task is repeated within the same run, handle repeated task context retrieval
            


    def is_task_repeated_in_current_run(self, task_name):
        """
        Checks if the task has already been executed in the current run.
        :param task_name: Name of the task to check.
        :return: Boolean indicating if the task is repeated within the current run.
        """
        return task_name.lower() in self.current_run_tasks
    
    # def detect_repeated_task_within_run(self, task_node, db):
    #     """
    #     Detect if a task is repeated within the current run and apply previous learning context if so.
    #     :param task_node: The current TaskNode object.
    #     :param db: Task history database to retrieve context from.
    #     :return: Boolean indicating whether the task was repeated and previous context applied.
    #     """
    #     task_name = task_node.task_name  # Extract task name as a string
    #     if self.is_task_repeated_in_current_run(task_name):
    #         # print(f"Task '{task_name}' is repeated within the same run.")
    #         # self.retrieve_and_apply_previous_context(task_name, task_node)  # Apply previous context if found
    #             self.add_to_current_run_tasks(task_node)
    #             return True
    #     else:
    #         self.add_to_current_run_tasks(task_node)  # Pass task_node here
    #         return False

    def retrieve_and_apply_previous_context(self, task_name, repeated_task_node, max_context_size=10):
            """
            Retrieves the previous context and state of a task with the same name from task_history_db,
            and applies it to the repeated task's context if the previous state was successful.
            :param task_name: The name of the task being repeated.
            :param repeated_task_node: The current repeated task node.
            :param max_context_size: The maximum length of the context allowed after merging.
            """
            if self.is_task_repeated_in_current_run(task_name):
             print(f"Task '{task_name}' is repeated within the same run.")
        
            # Step 1: Retrieve tasks with the same name from task_history_db
            previous_tasks = self.task_history_db.get_similar_tasks(task_name)

            if not previous_tasks:
                print(f"No previous tasks found for '{task_name}'")
                return

            # Step 2: Find a successful task and retrieve context and state
            for previous_task in previous_tasks:
                previous_state = previous_task.get("status")  # Adjusted to check 'outcome' instead of 'status'
                if previous_state == "success":
                    previous_context = previous_task.get("context", [])
                    print(f"Found previous successful task for '{task_name}' with context: {previous_context}")
                    
                    # Step 3: Limit context size and merge it
                    #the repeated_task_node.context is not its name 
                    merged_context = self.merge_previous_context_with_repeated(
                        previous_context, repeated_task_node.context, max_context_size
                    )
                    repeated_task_node.context = merged_context
                    print(f"Updated context for repeated task '{task_name}': {repeated_task_node.context}")
                    return  # Once a successful context is applied, stop further search

            print(f"No successful previous tasks found for '{task_name}'")

    def merge_previous_context_with_repeated(self, previous_context, repeated_context, max_context_size):
                """
                Merges previous context with the repeated task's context, ensuring the total size does not exceed max_context_size.
                The previous context is added before the repeated context.
                :param previous_context: The context retrieved from the previous successful task.
                :param repeated_context: The current context of the repeated task.
                :param max_context_size: The maximum allowed context size.
                :return: The merged context list, truncated to max_context_size.
                """
                # Combine previous context with the current repeated task's context
                #the repeated_context is not a valid name 
                merged_context = previous_context + repeated_context
                
                # Ensure the length doesn't exceed the max_context_size
                if len(merged_context) > max_context_size:
                    merged_context = merged_context[-max_context_size:]  # Keep the last `max_context_size` elements
                
                return merged_context  


    def htn_planning(self):
        db = VectorDB()
        root_node = TaskNode(self.goal_input)
        root_node.context = [f"Initial task reasoning for goal: {self.goal_task}"]
        root_node.reason_through_task()
        # task_history = [] 
        max_iterations = 100    
        self.total_runs += 1   

        while not self.is_goal_achieved(root_node, self.initial_state, self.goal_task):
            success, _ , _= self.decompose(root_node, self.initial_state, 0, self.max_depth, 
                                        self.capabilities_input, self.goal_task, db, self.send_update_callback)
            if not success:
                return None

        print("Plan found successfully!")
        self.success_count += 1
        print(f"Run success count: {self.success_count} / {self.total_runs} runs completed successfully.")

        redundant_tasks = {task: count for task, count in self.repetition_count.items() if count > 1}
        total_redundant_tasks = len(redundant_tasks)
        print(f"Total redundant tasks in this run: {total_redundant_tasks}")
        print(f"Redundant tasks and their counts: {redundant_tasks}")


        if len(self.branches_children_count) > 1:  # Std deviation requires at least 2 values
            std_dev_subtasks = statistics.stdev(self.branches_children_count)
            print(f"Standard Deviation of Subtasks Generated Per Task: {std_dev_subtasks}")
            avg_subtasks_per_branch = sum(self.branches_children_count) / len(self.branches_children_count)
            print(f"Average Number of Subtasks per Branch: {avg_subtasks_per_branch}")
        
        else:
            print("Not enough data points to calculate standard deviation.")
        
        return root_node

    def is_goal_achieved(self, node, state, goal_task):
        return groq_is_goal(state, goal_task) or node.status == "completed"

    @trace_function_calls
    def htn_planning_recursive(self, state, goal_task, root_node, max_depth, capabilities_input, db, send_update_callback=None, task_history=None):
        if groq_is_goal(state, goal_task):
            return root_node

        if send_update_callback:
            send_update_callback(root_node)

        success, updated_state = self.decompose(root_node, state, 0, max_depth, capabilities_input, goal_task,
                                        db, send_update_callback, task_history)
        if success:
            root_node.status = "succeeded"
            state = updated_state
            return root_node
        else:
            root_node.status = "failed"

        return root_node

    @trace_function_calls
    def replan_required(self, state, goal_task, task_node):
        if groq_is_goal(state, goal_task):
            return False
        if task_node is None or task_node.children == []:
            return True
        return False


    @trace_function_calls
    def translate_task(self, task, capabilities_input):
        response = translate(self.goal_input, task, capabilities_input)
        translated_task = response.strip()
        log_response("translate_task", translated_task)
        return translated_task


    @trace_function_calls
    def check_subtasks(self, task, subtasks, capabilities_input, task_history):
        result = check_subtasks(task, subtasks, capabilities_input, task_history)
        log_response("check_subtasks", result)
        return result == 'true'

    @trace_function_calls
    def decompose(self, task_node, state, depth, max_depth, capabilities_input, goal_state, db, send_update_callback=None, task_history=None):
        task = task_node.task_name

        # if self.detect_repeated_task_within_run(task_node, db):
        #     print(f"Task '{task}' is repeated within the same run , finding solution.")
        #     self.retrieve_and_apply_previous_context(task, task_node)
            
        
        # self.add_to_current_run_tasks(task_node)
        # print("this run task in decompose" .join(self.current_run_tasks))

        decompose_state = state
        start_time = time.time()
        self.branches_children_count = self.branches_children_count if hasattr(self, 'branches_children_count') else []
        self.branch_depths = self.branch_depths if hasattr(self, 'branch_depths') else []

        branch_depth = depth
        
        if is_granular(task, capabilities_input):
            print(f"Task '{task}' is granular enough to execute.")
            if can_execute(task, capabilities_input, state):
                print(f"Executing granular task: {task}")
                updated_state = self.execute_task(decompose_state, task_node)
                task_node.status = "completed"
                self.task_history_db.add_task(
                    task_name=subtask_node.task_name,
                    goal_task=self.goal_task,
                    reasoning=subtask_node.reasoning,  # Save only the specific reasoning here
                    status=subtask_node.status
                )
                print("status return", task_node.status)
                return True, updated_state, task_node.status
            else:
                print(f"Cannot execute task '{task}' at this time.")
                task_node.status = "failed"
                success, updated_state, subtask_status = self.decompose(
                    subtask_node, decompose_state, depth + 1, max_depth,
                    capabilities_input, goal_state, db, send_update_callback, task_history
                )
                print("status return", task_node.status)
                return False, state, task_node.status
        
        print(f"Decomposing task (depth {depth}/{max_depth}): {task}")

    
        if depth > 2:  # Set a limit to stop overly deep decomposition
            print(f"Task '{task}' has reached a sufficient level of abstraction.")
            task_node.status = "in progress"
            print("status return", task_node.status)
            return True, decompose_state, task_node.status

        if depth > max_depth:
            print(f"Max depth reached for task: {task}")
            task_node.status = "failed"
            if send_update_callback:
                send_update_callback(task_node)
            print("status return", task_node.status)    
            return False, decompose_state, task_node.status

        subtasks_list = self.get_subtasks(task, decompose_state, max_depth - depth, capabilities_input)
        print(f"Subtasks for {task}: {subtasks_list}")

        if not subtasks_list:
            print(f"No valid subtasks found for {task}")
            task_node.status = "failed"
            if send_update_callback:
                send_update_callback(task_node)
            print("status return", task_node.status)
            return False, decompose_state, task_node.status
        

        children_count = len(subtasks_list)
        self.branches_children_count.append(children_count)
    
    # Update the branch depth for this task node if it's a new branch
        self.branch_depths.append(branch_depth)
    


    # Step 2: Calculate similarity scores with the hierarchy embedding
        context_loss_detected = False
        similarity_scores = []
        for subtask in subtasks_list:
            score = calculate_similarity(task, subtask)
            similarity_scores.append(score)
            if score < self.threshold:
                self.context_loss_instances += 1
                self.context_loss_depths.append(depth)  # Increment context loss count
                print(f"Context loss detected f   or subtask '{subtask}' with score {score}")
                print(f"Total context loss instances detected so far: {self.context_loss_instances}")
                context_loss_detected = True
        # if task not in task_history:
        #     task_history.append(task)
        #     print("task historyyy" , task_history)

        average_score = sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0
        print(f"Average similarity score for task '{task}': {average_score}")
        print(f"Total context loss instances detected so far: {self.context_loss_instances}")


        task_node.status = "in-progress"
        if send_update_callback:
            send_update_callback(task_node)

        for subtask in subtasks_list:
            subtask_node = TaskNode(subtask, parent=task_node)
            print("subtask node", subtask_node.task_name)
            # print(f"Parent context before passing to subtask: {task_node.context}")
            subtask_node.context = task_node.context.copy()
            # print(f"Subtask context after copying: {subtask_node.context}")
            subtask_node.reason_through_task()
            # print(f"Context size for subtask '{subtask_node.task_name}': {len(subtask_node.context)} items")
            # print(f"Context contents: {subtask_node.context}")
            task_node.add_child(subtask_node)
        

            if is_task_primitive(subtask):
                # print("is task primitive result " , is_task_primitive(subtask))
                # print(f"Checking can_execute for task '{subtask}' with state '{decompose_state}'")
                if can_execute(subtask, capabilities_input, decompose_state):
                    # print("can execute result", can_execute(subtask, capabilities_input, decompose_state))
                    print(f"Executing task: {subtask}")
                    updated_state = self.execute_task(decompose_state, subtask_node)
                    decompose_state = updated_state
                    self.task_history_db.add_task(
                    task_name=subtask_node.task_name,
                    goal_task=self.goal_task,
                    reasoning=subtask_node.reasoning,  # Save only the specific reasoning here
                    status=subtask_node.status
                )
                    success = True
                    subtask_status = "completed"

                    
                else:
                    # If can_execute fails, attempt further decomposition instead of failing
                    print(f"Cannot execute task '{subtask}', attempting further decomposition.")
                    # Initialize success and others as False/None in case decomposition fails
                    success, updated_state, subtask_status = self.decompose(
                    subtask_node, decompose_state, depth + 1, max_depth,
                    capabilities_input, goal_state, db, send_update_callback, task_history
                )

            else:
            # Task is not primitive, so attempt further decomposition
                success, updated_state, subtask_status = self.decompose(
                    subtask_node, decompose_state, depth + 1, max_depth,
                    capabilities_input, goal_state, db, send_update_callback, task_history
                )        

            if success:
                decompose_state = updated_state
                subtask_node.status = subtask_status

            if send_update_callback:
                send_update_callback(subtask_node)

            if subtask_node.status == "failed":
                task_node.status = "failed"
                if send_update_callback:
                    send_update_callback(task_node)
                return False, decompose_state, task_node.status

        task_node.status = "completed"
        if send_update_callback:
            send_update_callback(task_node)

        if task_node.status == "completed":
            db.add_task_node(task_node)
            self.task_history_db.add_task(
            task_name=subtask_node.task_name,
            goal_task=self.goal_task,
            reasoning=subtask_node.reasoning,  # Save only the specific reasoning here
            status=subtask_node.status
        )
            
        self.branches_children_count.append(children_count)
        print(f"Task completed: {task}")
        print("status return", task_node.status)
        execution_time = time.time() - start_time
        print(f"Execution time for task '{task}': {execution_time} seconds")
        print(f"Task '{task}' - Average Score: {average_score}, Execution Time: {execution_time} seconds")
        return True, decompose_state, task_node.status

    @trace_function_calls
    def evaluate_candidate(self, task, subtasks, capabilities_input, task_history):
        max_retries = 3
        retries = 0
        while retries < max_retries:
            # Max 10 token or 8 digits after the decimal 0.99999999
            response = evaluate_candidate(self.goal_input, task, subtasks, capabilities_input, task_history)
            try:
                score = float(response.strip())
                log_response("evaluate_candidate", score)
                return score
            except ValueError:
                retries += 1
                if retries >= max_retries:
                    raise ValueError("Failed to convert response to float after multiple retries.")


    @trace_function_calls
    def get_subtasks(self, task, state, remaining_decompositions, capabilities_input, task_history=None):
        # print("task history in get subtasks ", task_history)

        # Granularity check to avoid excessive decomposition
        if is_granular(task, capabilities_input):
            print(f"Task '{task}' is granular and cannot be decomposed further.")
            return []  # Do not decompose further
        

        # Adding depth control: if the task is already manageable, stop further decomposition
        if remaining_decompositions < 3:  # Adjust this number based on the desired abstraction level
            print(f"Task '{task}' has reached an appropriate level of detail.")
            return []  # No further decomposition

        recent_history = self.task_history_db.get_task_history(limit=10)
        task_history_str = ", ".join([entry['task_name'] for entry in recent_history])

        # Get subtasks as before
        subtasks_with_types = get_subtasks(task, state, remaining_decompositions, capabilities_input,  task_history_str)
        # filtered_subtasks = [subtask for subtask in subtasks_with_types if subtask != task]
        # print(f"Decomposing task {task} into candidates:\n{subtasks_with_types}")
        # print("subtasks with types", subtasks_with_types)
        return subtasks_with_types


    @trace_function_calls
    def execute_task(self, state, task_node):
        # if self.detect_repeated_task_within_run(task_node, self.task_history_db):
        #     print(f"Skipping execution for repeated task: {task_node.task_name}")
        #     self.retrieve_and_apply_previous_context(task_node.task_name, task_node)
        #     return state 
        # print("the state", state)
        prompt = (f"Given the current state '{state}' and the task '{task_node.task_name}', "
          f"update the state after executing the task corresponding to the task and state. "
          f"Provide both the **updated state** and the reasoning for CoT based on the following context: '{task_node.context}'.")
        response = call_groq_api(prompt)
        updated_state_raw = response.choices[0].message.content.strip()  # Raw response with state + reasoning

    # Separate state and reasoning
        if "Reasoning" in updated_state_raw:
                updated_state, reasoning = updated_state_raw.split("Reasoning", 1)
                updated_state = updated_state.strip()
                reasoning = reasoning.strip()
        else:
                updated_state = updated_state_raw
                reasoning = ""

         
        if task_node.status == "failed":
            print(f"Task '{task_node.task_name}' failed. Reason: {reasoning}")
            task_node.context.append(f"Failed task: {task_node.task_name}, Reasoning: {reasoning}")
            # print("contextttt", task_node.context)
            return state 
            
        self.add_to_current_run_tasks(task_node)

            # Append reasoning to task context (important for CoT)
        task_node.context.append(f"Executed task: {task_node.task_name}, State: {updated_state}, Reasoning: {reasoning}")
        task_node.outcome = updated_state
        log_state_change(state, updated_state, task_node)
            
            # Return only the concise state for task execution
        # print(f"Updated state after task '{task_node.task_name}': {updated_state}")
        return updated_state
    