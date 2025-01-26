
from LLM_utils import calculate_similarity, generate_hierarchy_embedding, groq_is_goal, is_task_primitive, can_execute, log_state_change
from LLM_api import call_groq_api, log_response
from task_node import TaskNode 
from text_utils import extract_lists, trace_function_calls
from htn_prompts import * 
from vector_db import VectorDB
import json

from task_history_db import TaskHistoryDB
import time
import statistics

from datetime import datetime

class HTNPlanner:

    def __init__(self, goal_input, initial_state, goal_task, capabilities_input, max_depth=7, send_update_callback=None, task_history_file = 'task_history_file.json' , threshold=0.7):
        self.goal_input = goal_input
        self.initial_state = initial_state
        self.goal_task = goal_task
        self.capabilities_input = capabilities_input
        self.max_depth = max_depth
        self.send_update_callback = send_update_callback
        self.task_history_db = TaskHistoryDB()
        self.current_run_tasks = []
        self.threshold = threshold 
        self.context_loss_instances = 0
        self.saved_tasks = set()
        self.generated_tasks = []
        self.depth_of_context_loss = 0
        self.successful_context_applications = 0

        
    
    
    def add_to_current_run_tasks(self, task_node):
        """
        Adds the task to the current run task list to track tasks executed in this run.
        Ensures each task is added only once.
        """
        task_name_lower = task_node.task_name.strip().lower()
        if task_name_lower not in self.current_run_tasks:
            self.current_run_tasks.append(task_name_lower)
            # print(f"Task '{task_name_lower}' added to current run.")
                     


    def is_task_repeated_in_current_run(self, task_name):
        """
        Checks if the task has already been executed in the current run.
        :param task_name: Name of the task to check.
        :return: Boolean indicating if the task is repeated within the current run.
        """
        return is_task_similar(task_name, self.current_run_tasks, thresholdofsimilar=0.97)

    def retrieve_and_apply_previous_context(self, task_name, repeated_task_node, max_context_size=10, similarity_threshold=0.7, fallback_threshold=0.66):
        """
        Retrieves the previous context and state of a task with a similar name from task_history_db,
        and applies it to the repeated task's context if a successful state is found.
        """

        previous_tasks = self.task_history_db.get_similar_tasks(task_name)

        if not previous_tasks:
            print(f"No previous tasks found for '{task_name}'")
            return

        similar_tasks = []
        for previous_task in previous_tasks:
            previous_task_name = previous_task.get("task_name", "")
            previous_goal_task = previous_task.get("goal_task", "")
            similarity_score = calculate_similarity(str(task_name), str(previous_task_name))
            goal_similarity_score = calculate_similarity(self.goal_task, previous_goal_task)
            
            if similarity_score >= similarity_threshold and goal_similarity_score >= similarity_threshold:
                similar_tasks.append((previous_task, similarity_score))
            else:
                print(f"Task '{previous_task_name}' with goal '{previous_goal_task}' has low similarity ({similarity_score}), not initially considered.")
        
        if not similar_tasks:
            for previous_task in previous_tasks:
                previous_task_name = previous_task.get("task_name", "")
                similarity_score = calculate_similarity(str(task_name), str(previous_task_name))
                
                if similarity_score >= fallback_threshold:
                    similar_tasks.append((previous_task, similarity_score))

        similar_tasks = sorted(similar_tasks, key=lambda x: x[1], reverse=True)
        for previous_task, score in similar_tasks:
            raw_context = previous_task.get("context", "[]")

            if isinstance(raw_context, str):
                try:
                    context_used = json.loads(raw_context)
                except json.JSONDecodeError:
                    context_used = []
            elif isinstance(raw_context, list):
                context_used = raw_context

            if isinstance(context_used, list):
                previous_context = [
                item.get("reasoning", "") for item in context_used
                if isinstance(item, dict) and item.get("subtask") == task_name and "reasoning" in item
                ]
            else:
                previous_context = []

            if previous_context:
                merged_context = self.merge_previous_context_with_repeated(
                    previous_context, repeated_task_node.context, max_context_size, task_name
                )
                repeated_task_node.context = merged_context
                return True

        print(f"No successful previous tasks found for '{task_name}'")
        return False


    def merge_previous_context_with_repeated(self, previous_context, repeated_context, max_context_size, task_name=None):
        """
        Merges previous context with the repeated task's context, ensuring the total size does not exceed max_context_size.
        Only the most relevant items (related to the task_name) are retained.
        :param previous_context: The context retrieved from the previous successful task.
        :param repeated_context: The current context of the repeated task.
        :param max_context_size: The maximum allowed context size.
        :param task_name: Optional task name for relevance scoring.
        :return: The merged context list.
        """
        merged_context = previous_context + repeated_context
        merged_context = [str(item) for item in merged_context if isinstance(item, str)]
        
        if task_name and merged_context:
            try:
                for context_item in merged_context:
                    print(f"Comparing task_name: '{task_name}'")
                    merged_context = sorted(
                    merged_context,
                    key=lambda context_item: calculate_similarity(task_name, context_item),
                    reverse=True
                    )
            except Exception as e:
                print(f"Error calculating similarity: {e}")
                return merged_context[:max_context_size]

        if len(merged_context) > max_context_size:
            merged_context = merged_context[:max_context_size]
        return merged_context




    def htn_planning(self):
        db = VectorDB()
        root_node = TaskNode(self.goal_input)
        root_node.context = [f"Initial task reasoning for goal: {self.goal_task}"]
        root_node.reason_through_task()
        max_iterations = 100    
        start_time = time.time()
        execution_start_time = time.time()

        while not self.is_goal_achieved(root_node, self.initial_state, self.goal_task):
            success, _ , _= self.decompose(root_node, self.initial_state, 0, self.max_depth, 
                                        self.capabilities_input, self.goal_task, db, self.send_update_callback)
            if not success:
                return None

        print("Plan found successfully!")
        return root_node
        # execution_time = time.time() - start_time
        # # print(f"Execution time for task '{root_node.task_name}': {execution_time} seconds")

        # execution_time = time.time() - execution_start_time
        # self.execution_times = getattr(self, "execution_times", [])
        # self.execution_times.append(execution_time)
        # print(f"Execution time for the plan: {execution_time:.2f} seconds")


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
    def decompose(self, task_node, state, depth, max_depth, capabilities_input, goal_state, db, send_update_callback=None, task_history=None):
        task = task_node.task_name
        decompose_state = state

        if task in self.saved_tasks:
            print(f"Task '{task}' already saved, skipping duplicate save.")
            return True, state, task_node.status

        print("Initial state:", "Standing in the kitchen")
        print(f"Decomposing task (depth {depth}/{max_depth}): {task}")

    
        if depth > 4:  # Set a limit to stop overly deep decomposition
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

        subtasks_list = self.get_subtasks(task, decompose_state, max_depth - depth, capabilities_input, self.current_run_tasks, task_node.context, self.generated_tasks)
        subtasks_list = [subtask for subtask in subtasks_list if subtask not in self.current_run_tasks]
    
        
        if not subtasks_list:
            print(f"No valid subtasks found for {task}")
            task_node.status = "failed"
            if send_update_callback:
                send_update_callback(task_node)
            print("status return", task_node.status)
            return False, decompose_state, task_node.status
        self.generated_tasks.extend(subtasks_list)

        if not self.feedback_fallback(task_node, subtasks_list):
            print(f"Regenerating subtasks for task: {task_node.task_name}")
            return self.decompose(task_node, state, depth, max_depth, capabilities_input, goal_state, db, send_update_callback, task_history)


        initial_similarity_scores = []
        context_loss_detected = False
        context_applied = False
        context_loss_index = None

        for index, subtask in enumerate(subtasks_list):
            score = calculate_similarity(task, subtask)
            initial_similarity_scores.append(score)
            # print(f"Initial similarity score for subtask '{subtask}': {score}" )
        initial_avg_similarity_score = round(sum(initial_similarity_scores) / len(initial_similarity_scores), 3)
        # print(f"Initial Average Similarity Score for task '{task}': {initial_avg_similarity_score}")

        current_threshold = self.adjust_threshold(task)
        for index, subtask in enumerate(subtasks_list):
            if initial_similarity_scores[index] < current_threshold:
                self.context_loss_instances += 1

                print("Depth for context loss" ,  self.depth_of_context_loss)
                print(f"Context loss detected for subtask '{subtask}' with score :{initial_similarity_scores[index]}")
                context_loss_detected = True

                if index == len(subtasks_list) - 1:
                    print(f"Context loss detected at the last subtask '{subtask}'. Skipping context retrieval.")
                    continue

                context_applied = self.retrieve_and_apply_previous_context(subtask, task_node)
                if context_applied and context_loss_detected:
                    self.successful_context_applications = getattr(self, "successful_context_applications", 0) + 1
                    context_loss_index = index + 1
                    print(f"Context applied. Regenerating subtasks from index {context_loss_index} onwards.")
                    break
                
                if context_applied and index != len(subtasks_list) - 1:
                    context_loss_index = index + 1
                    break

        if context_applied and context_loss_index is not None:
            updated_context = task_node.context
            new_subtasks_after_loss = get_subtasks_with_context(
                task, state, max_depth - depth, capabilities_input, updated_context, task_history, self.generated_tasks
            )
            

            unique_new_subtasks = [
            subtask for subtask in new_subtasks_after_loss if subtask not in subtasks_list[:context_loss_index]
            ]
            updated_subtasks = subtasks_list[:context_loss_index] + unique_new_subtasks

            print(f"Updated subtasks for task '{task}': {updated_subtasks}")
                    
            regenerated_similarity_scores = [
                calculate_similarity(task, subtask) for subtask in new_subtasks_after_loss
            ]
            regenerated_avg_similarity_score = round(
            sum(regenerated_similarity_scores) / len(regenerated_similarity_scores), 3
            ) if regenerated_similarity_scores else 0

            
            print(f"New Average Similarity Score for task after context application: {regenerated_avg_similarity_score}")
            
            
            if regenerated_avg_similarity_score > initial_avg_similarity_score:
                subtasks_list = updated_subtasks
                self.successful_context_applications = getattr(self, "successful_context_applications", 0) + 1
                print(f"New subtasks accepted for task '{task}': {subtasks_list}")
            else:
                subtasks_list = subtasks_list
                print(f"New subtasks rejected for task '{task}'. Retaining original subtasks: {subtasks_list}")
 

        if context_applied and regenerated_avg_similarity_score >= self.threshold:
            for score, subtask in zip(regenerated_similarity_scores, new_subtasks_after_loss):
                print(f"New similarity score for regenerated subtask '{subtask}': {score}")

                if score < self.threshold:
                    self.context_loss_instances += 1
                    print(f"Context loss detected for regenerated subtask '{subtask}' with score {score}")

        print(f"Total context loss instances detected so far: {self.context_loss_instances}")
        print("Skipping previous learning context.")

        task_node.status = "in-progress"
        if send_update_callback:
            send_update_callback(task_node)


        for subtask in subtasks_list:
            subtask_node = TaskNode(subtask, parent=task_node)
            subtask_node.context = task_node.context.copy()
            subtask_node.reason_through_task()
            task_node.add_child(subtask_node)
        
            if is_task_similar(subtask_node.task_name, self.current_run_tasks):
                print(f"Subtask '{subtask_node.task_name}' is detected as repeated within the same run. Executing directly.")
                updated_state = self.execute_task(self.initial_state, subtask_node)
                subtask_node.status = "completed"
                self.task_history_db.add_task(
                    task_name= subtask_node.task_name,
                    goal_task=self.goal_task,
                    reasoning=subtask_node.reasoning,
                    context=subtask_node.context,  
                    status=subtask_node.status
                    )
                continue

            if is_task_primitive(subtask):
                if can_execute(subtask, capabilities_input, decompose_state):
                    updated_state = self.execute_task(decompose_state, subtask_node)
                    decompose_state = updated_state
                    subtask_node.status = "completed"
                    success = True


                    
                else:
                    print(f"Cannot execute task '{subtask}', attempting further decomposition.")
                    success, updated_state, subtask_status = self.decompose(
                    subtask_node, decompose_state, depth + 1, max_depth,
                    capabilities_input, goal_state, db, send_update_callback, task_history
                )

            else:
                success, updated_state, subtask_status = self.decompose(
                    subtask_node, decompose_state, depth + 1, max_depth,
                    capabilities_input, goal_state, db, send_update_callback, task_history
                )        

                if success:
                    decompose_state = updated_state
                    subtask_node.status = "completed"


            if subtask_node and subtask_node.status == "completed":
                self.task_history_db.add_task(
                    task_name=subtask_node.task_name,
                    goal_task=self.goal_task,
                    reasoning=subtask_node.context,
                    context=subtask_node.context,
                    status=subtask_node.status
                )
                self.saved_tasks.add(task)

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
            reasoning=subtask_node.reasoning,
            context=subtask_node.context,
            status=subtask_node.status
        )
        print(f"Task completed by the robot :)")
        # print(f"Task completed: {task}")
        print("status return", task_node.status)
        return True, decompose_state, task_node.status


    @trace_function_calls
    def get_subtasks(self, task, state, remaining_decompositions, capabilities_input, current_run_tasks, context, generated_tasks):

        if remaining_decompositions < 3: 
            print(f"Task '{task}' has reached an appropriate level of detail.")
            return []
        task_history_str = ", ".join(current_run_tasks)
        subtasks_with_types = get_subtasks(task, state, remaining_decompositions, capabilities_input, task_history_str, context, generated_tasks)
        return subtasks_with_types


    @trace_function_calls
    def execute_task(self, state, task_node):
        start_time = time.time()
        print(f"Executing Task : {task_node.task_name}")
        prompt = (f"Given the current state '{state}' and the task '{task_node.task_name}', "
                  f"consider the following reasoning context:{task_node.context}"
                  f"update the state after executing the task corresponding to the task and state. "
                  f"Provide both the **updated state** and the reasoning for CoT based on the following context: '{task_node.context}'.")
        response = call_groq_api(prompt)
        updated_state_raw = response.choices[0].message.content.strip()
        # print("updated state", updated_state_raw)  # Raw response with state + reasoning
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
        # print("state", state , "updated_state" , updated_state)
        log_state_change(state, updated_state, task_node)
        return updated_state
    
    
    def feedback_fallback(self, task_node, subtasks_list):
        """
        Prompts the user to validate the generated subtasks.
        """
        print(f"Generated subtasks for '{task_node.task_name}': {subtasks_list}")
        user_input = input(f"Are the generated subtasks for '{task_node.task_name}' accurate and aligned with the task requirements? (yes/no):").strip().lower()
        
        if user_input == "yes":
            print("User validated the subtasks. Proceeding with the flow.")
            return True
        elif user_input == "no":
            print("User rejected the subtasks. Regenerating subtasks.")
            return False
        else:
            print("Invalid input. Assuming 'yes' and Proceeding with the flow. ")
            return True


    def calculate_average_execution_time(self):
        """
        Calculate the average execution time for task decomposition and execution.
        """
        total_time = sum(getattr(self, "execution_times", []))
        return total_time / len(self.execution_times) if self.execution_times else 0.0

    def adjust_threshold(self, task_name):
        """
        Adjusts the similarity threshold dynamically based on task complexity.
        """
        complex_keywords = ["prepare", "organize", "manage", "inspect", "identify", "Do"]
        specific_keywords = ["turn on", "activate", "operate", "flip"]

        # Default threshold using the instance's threshold attribute
        threshold = self.threshold

        # Adjust based on task complexity
        if any(keyword in task_name.lower() for keyword in complex_keywords):
            threshold = 0.8
            # threshold = 0.84  # More lenient for complex tasks
        elif any(keyword in task_name.lower() for keyword in specific_keywords):
            threshold = 0.8
            # threshold = 0.82  # Stricter for specific tasks

        # print(f"Adjusted threshold for '{task_name}': {threshold}")
        return threshold