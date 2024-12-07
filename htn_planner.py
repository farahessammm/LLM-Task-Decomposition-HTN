
from pyexpat import model
from LLM_utils import calculate_similarity, generate_hierarchy_embedding, groq_is_goal, is_task_primitive, can_execute, log_state_change
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
import re
from datetime import datetime

class HTNPlanner:

    def __init__(self, goal_input, initial_state, goal_task, capabilities_input, max_depth=7, send_update_callback=None, task_history_file = 'task_history_file.json' , threshold=0.8):
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
        self.saved_tasks = set()
        self.task_attempts = {}      # Dictionary to track the number of attempts for each task
        self.task_completions = {}
        self.total_task_count = 0 
        self.generated_tasks = []
        
    
    
    def add_to_current_run_tasks(self, task_node):
        """
        Adds the task to the current run task list to track tasks executed in this run.
        Ensures each task is added only once. If the task is repeated, retrieves and applies previous context.
        """
        task_name_lower = task_node.task_name.strip().lower()
        if task_name_lower not in self.current_run_tasks:
                self.current_run_tasks.append(task_name_lower)
                # self.repetition_count[task_name_lower] = self.repetition_count.get(task_name_lower, 0) + 1
                # print(f"[DEBUG] Task repetition count for '{task_name_lower}': {self.repetition_count[task_name_lower]}")
                    
    def adjust_threshold(task_name):
            """
            Adjusts the similarity threshold dynamically based on task complexity.
            """
            complex_keywords = ["prepare", "organize", "manage", "inspect", "identify"]
            specific_keywords = ["turn on", "activate", "operate", "flip"]

            # Default threshold
            threshold = threshold # This can also be set as a default value if not passed to the class

            # Adjust based on task complexity
            if any(keyword in task_name.lower() for keyword in complex_keywords):
                threshold = 0.7  # More lenient for complex tasks
            elif any(keyword in task_name.lower() for keyword in specific_keywords):
                threshold = 0.9  # Stricter for specific tasks

            # print(f"Adjusted threshold for '{task_name}': {threshold}")
            return threshold    


    def is_task_repeated_in_current_run(self, task_name):
        """
        Checks if the task has already been executed in the current run.
        :param task_name: Name of the task to check.
        :return: Boolean indicating if the task is repeated within the current run.
        """
        # rint(f"[DEBUG] Checking if tpask '{task_name}' is repeated in the current run.")
        return is_task_similar(task_name, self.current_run_tasks, thresholdofsimilar=0.97)
    
    def detect_repeated_task_within_run(self, task_node):
        """
        Detect if a task is repeated within the current run and apply previous learning context if so.
        :param task_node: The current TaskNode object.
        :param db: Task history database to retrieve context from.
        :return: Boolean indicating whether the task was repeated and previous context applied.
        """
        task_name = task_node.task_name  # Extract task name as a string
        if self.is_task_repeated_in_current_run(task_name):
            print(f"Task '{task_name}' is detected as repeated.")
            self.add_to_current_run_tasks(task_node)
            return True
        else:
            self.add_to_current_run_tasks(task_node)
            return False

    def retrieve_and_apply_previous_context(self, task_name, repeated_task_node, max_context_size=10, similarity_threshold=0.7, fallback_threshold=0.66):
        """
        Retrieves the previous context and state of a task with a similar name from task_history_db,
        and applies it to the repeated task's context if a successful state is found.
        """
        start_time = time.time()

        previous_tasks = self.task_history_db.get_similar_tasks(task_name)

        if not previous_tasks:
            print(f"No previous tasks found for '{task_name}'")
            return

        similar_tasks = []
        for previous_task in previous_tasks:
            previous_task_name = previous_task.get("task_name", "")
            previous_goal_task = previous_task.get("goal_task", "")
            # print(f"Comparing task '{task_name}' with previous task '{previous_task_name}' under goal '{previous_goal_task}'")
            similarity_score = calculate_similarity(str(task_name), str(previous_task_name))
            goal_similarity_score = calculate_similarity(self.goal_task, previous_goal_task)
            
            if similarity_score >= similarity_threshold and goal_similarity_score >= similarity_threshold:
                similar_tasks.append((previous_task, similarity_score))
            else:
                print(f"Task '{previous_task_name}' with goal '{previous_goal_task}' has low similarity ({similarity_score}), not initially considered.")
        
        if not similar_tasks:
            # print(f"No tasks met the similarity threshold of {similarity_threshold}. Retrying with fallback threshold of {fallback_threshold}.")
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
                    # print(f"Failed to decode context for task '{task_name}'.")
                    context_used = []
            elif isinstance(raw_context, list):
                context_used = raw_context
            else:
                # print(f"Unexpected format for context in task '{task_name}': {type(raw_context).__name__}.")
                context_used = []

            if isinstance(context_used, list):
                previous_context = [
                item.get("reasoning", "") for item in context_used
                if isinstance(item, dict) and item.get("subtask") == task_name and "reasoning" in item
                ]
            else:
                # print(f"Unexpected format in reasoning for task '{task_name}'. Expected list, found {type(context_used).__name__}.")
                previous_context = []

            if previous_context:
                # print(f"Found previous successful task for '{task_name}' with context: {previous_context}")
                
                # Step 4: Limit context size and merge it
                merged_context = self.merge_previous_context_with_repeated(
                    previous_context, repeated_task_node.context, max_context_size, task_name
                )
                repeated_task_node.context = merged_context
                print(f"Updated context for task '{task_name}': {repeated_task_node.context}")
                return True  # Stop further search once a successful context is applied

        print(f"No successful previous tasks found for '{task_name}'")
        # elapsed_time = time.time() - start_time
        # print(f"Elapsed time for operation: {elapsed_time} seconds")
        return False


    def merge_previous_context_with_repeated(self, previous_context, repeated_context, max_context_size, task_name=None):
        """
        Merges previous context with the repeated task's context, ensuring the total size does not exceed max_context_size.
        Only the most relevant items (related to the task_name) are retained.
        :param previous_context: The context retrieved from the previous successful task.
        :param repeated_context: The current context of the repeated task.
        :param max_context_size: The maximum allowed context size.
        :param task_name: Optional task name for relevance scoring.
        :return: The merged context list, truncated to max_context_size.
        """
        # Combine previous context with the current repeated task's context
        start_time = time.time()
        merged_context = previous_context + repeated_context

        # Ensure all context items are strings; filter out any non-string items
        merged_context = [str(item) for item in merged_context if isinstance(item, str)]
        
        # Relevance scoring (optional) if task_name is provided
        if task_name and merged_context:
            try:
                # Log items being compared for debugging
                for context_item in merged_context:
                    print(f"Comparing task_name: '{task_name}' with context_item: '{context_item}'")

                # Sort based on similarity to the task_name
                    merged_context = sorted(
                    merged_context,
                    key=lambda context_item: calculate_similarity(task_name, context_item),
                    reverse=True
                    )
                    print("similarity of merged context" , merged_context)
            except Exception as e:
                print(f"Error calculating similarity: {e}")
                return merged_context[:max_context_size]  # Return truncated list if error occurs

        # Ensure the length doesn't exceed the max_context_size
        if len(merged_context) > max_context_size:
            merged_context = merged_context[:max_context_size]  # Keep only the most relevant `max_context_size` elements
        # elapsed_time = time.time() - start_time
        # print(f"Elapsed time for operation: {elapsed_time} seconds")
        return merged_context




    def htn_planning(self):
        db = VectorDB()
        root_node = TaskNode(self.goal_input)
        root_node.context = [f"Initial task reasoning for goal: {self.goal_task}"]
        root_node.reason_through_task()
        # task_history = [] 
        max_iterations = 100    
        self.total_runs += 1
        start_time = time.time()
        execution_start_time = time.time()

        while not self.is_goal_achieved(root_node, self.initial_state, self.goal_task):
            success, _ , _= self.decompose(root_node, self.initial_state, 0, self.max_depth, 
                                        self.capabilities_input, self.goal_task, db, self.send_update_callback)
            if not success:
                return None

        print("Plan found successfully!")
        execution_time = time.time() - start_time
        print(f"Execution time for task '{root_node.task_name}': {execution_time} seconds")
        execution_success_rate = self.calculate_execution_success_rate()
        print(f"Execution Success Rate: {execution_success_rate * 100:.2f}%")

        redundant_tasks = {task: count for task, count in self.repetition_count.items() if count > 1}
        total_redundant_tasks = len(redundant_tasks)
        print(f"Total redundant tasks in this run: {total_redundant_tasks}")
        print(f"Redundant tasks and their counts: {redundant_tasks}")
        max_depth_reached = max(self.branch_depths) if self.branch_depths else 0
        print(f"Max Depth Reached: {max_depth_reached}")
        print(f"Total Number of Tasks (including subtasks): {self.total_task_count}")   
        context_retention_rate = self.calculate_context_retention_rate()
        print(f"Context Retention Rate: {context_retention_rate}") 

        execution_time = time.time() - execution_start_time
        self.execution_times = getattr(self, "execution_times", [])
        self.execution_times.append(execution_time)
        print(f"Execution time for the plan: {execution_time:.2f} seconds")
        average_execution_time = self.calculate_average_execution_time()
        print(f"Average Execution Time: {average_execution_time:.2f} seconds")


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
        decompose_state = state
        self.task_attempts[task] = self.task_attempts.get(task, 0) + 1
        

       

        if task in self.saved_tasks:
            print(f"Task '{task}' already saved, skipping duplicate save.")
            return True, state, task_node.status
        
        
        self.branches_children_count = self.branches_children_count if hasattr(self, 'branches_children_count') else []
        self.branch_depths = self.branch_depths if hasattr(self, 'branch_depths') else []
        branch_depth = depth
        
        if is_granular(task, capabilities_input):
            print(f"[DEBUG] Task '{task}' is granular enough.")
            if can_execute(task, capabilities_input, state):
                print(f"[DEBUG] Task '{task}' can be executed.")
                updated_state = self.execute_task(decompose_state, task_node)
                task_node.status = "completed"
                self.task_completions[task] = self.task_completions.get(task, 0) + 1
                # print("THE REASONING ", subtask_node.reasoning)
                # print("THE CONTEXT " , subtask_node.context)
                self.task_history_db.add_task(
                    task_name=task_node.task_name,
                    goal_task=self.goal_task,
                    reasoning=task_node.reasoning,
                    context=task_node.context,  # Save only the specific reasoning here
                    status=task_node.status
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
        
        # print(f"Initial subtasks for {task}: {subtasks_list}")



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
                # print(f"[DEBUG] Generated tasks so far: {self.generated_tasks}")

        # redundancy_count, redundancy_ratio = self.calculate_subtask_redundancy(subtasks_list)
        # print(f"Subtask Redundancy: {redundancy_count} subtasks ({redundancy_ratio * 100:.2f}% redundancy)")
        children_count = len(subtasks_list)
        self.branches_children_count.append(children_count)
        self.branch_depths.append(branch_depth)
   

    # Step 2: Calculate similarity scores with the hierarchy embedding
        initial_similarity_scores = []
        context_loss_detected = False
        context_applied = False
        context_loss_index = None

        for index, subtask in enumerate(subtasks_list):
            score = calculate_similarity(task, subtask)
            initial_similarity_scores.append(score)
            print(f"Initial similarity score for subtask '{subtask}': {score}" )
        initial_avg_similarity_score = sum(initial_similarity_scores) / len(initial_similarity_scores)
        print(f"Initial Average Similarity Score for task '{task}': {initial_avg_similarity_score}")

        # goal_embedding = generate_hierarchy_embedding(task, subtasks_list)
        # self_similarity = calculate_similarity(task, task, goal_embedding=goal_embedding)
        # dynamic_threshold = self_similarity * 0.8 

        current_threshold = self.adjust_threshold(task)
        for index, subtask in enumerate(subtasks_list):
            if initial_similarity_scores[index] < current_threshold:
                self.context_loss_instances += 1
                self.context_loss_depths.append(depth)
                print(f"Context loss detected for subtask '{subtask}' with score :{initial_similarity_scores[index]}")
                context_loss_detected = True

                if index == len(subtasks_list) - 1:
                    print(f"Context loss detected at the last subtask '{subtask}'. Skipping context retrieval.")
                    continue

                context_applied = self.retrieve_and_apply_previous_context(subtask, task_node)
                if context_applied and context_loss_detected:
                    self.successful_context_applications = getattr(self, "successful_context_applications", 0) + 1
                    # Regenerate subtasks with the updated context and exit loop
                    context_loss_index = index + 1
                    print(f"Context applied. Regenerating subtasks from index {context_loss_index} onwards.")
                    break
                
                if context_applied and index != len(subtasks_list) - 1:
                    context_loss_index = index + 1
                    break

        if context_applied and context_loss_index is not None:
            updated_context = task_node.context
            # Get subtasks starting only from the context_loss_index
            new_subtasks_after_loss = get_subtasks_with_context(
                task, state, max_depth - depth, capabilities_input, updated_context, task_history, self.generated_tasks
            )
            

            unique_new_subtasks = [
            subtask for subtask in new_subtasks_after_loss if subtask not in subtasks_list[:context_loss_index]
            ]
            updated_subtasks = subtasks_list[:context_loss_index] + unique_new_subtasks

            print(f"Updated subtasks for task '{task}': {updated_subtasks}")


            # for i, new_subtask in enumerate(new_subtasks_after_loss):
            #     # Check similarity with the original subtasks to avoid duplication
            #     # if calculate_similarity(subtasks_list[context_loss_index + i], new_subtask) < self.threshold:
            #     #     updated_subtasks.append(new_subtask)
            #     #     print(f"Added new unique subtask '{new_subtask}' with low similarity to original subtask.")
            #     # else:
            #     #     print(f"Skipping regeneration for '{new_subtask}' due to high similarity with existing subtasks.")
            #     updated_subtasks.append(new_subtask)
            #     # subtasks_list = updated_subtasks
            # print(f"Updated subtasks for {task} after selective regeneration: {updated_subtasks}")

                    
            regenerated_similarity_scores = [
                calculate_similarity(task, subtask) for subtask in new_subtasks_after_loss
            ]
            regenerated_avg_similarity_score = (
                sum(regenerated_similarity_scores) / len(regenerated_similarity_scores)
                if regenerated_similarity_scores else 0
            )
            
            print(f"New Average Similarity Score for task '{task}' after context application: {regenerated_avg_similarity_score}")
            
            
            if regenerated_avg_similarity_score > initial_avg_similarity_score:
                subtasks_list = updated_subtasks
                print(f"New subtasks accepted for task '{task}': {subtasks_list}")
            else:
                subtasks_list = subtasks_list
                print(f"New subtasks rejected for task '{task}'. Retaining original subtasks: {subtasks_list}")


        if context_applied and regenerated_avg_similarity_score >= self.threshold:
            for score, subtask in zip(regenerated_similarity_scores, new_subtasks_after_loss):
                print(f"New similarity score for regenerated subtask '{subtask}': {score}")

                if score < self.threshold:
                    # Detect further context loss in regenerated subtasks
                    self.context_loss_instances += 1
                    self.context_loss_depths.append(depth)
                    print(f"Context loss detected for regenerated subtask '{subtask}' with score {score}")

        print(f"Total context loss instances detected so far: {self.context_loss_instances}")





        task_node.status = "in-progress"
        if send_update_callback:
            send_update_callback(task_node)


        for subtask in subtasks_list:
            subtask_node = TaskNode(subtask, parent=task_node)
            self.total_task_count += 1
            # print("subtask node", subtask_node.task_name)
            # print(f"Parent context before passing to subtask: {task_node.context}")
            subtask_node.context = task_node.context.copy()
            # print(f"Subtask context after copying: {subtask_node.context}")
            subtask_node.reason_through_task()
            # print(f"Context size for subtask '{subtask_node.task_name}': {len(subtask_node.context)} items")
            # print(f"Context contents: {subtask_node.context}")
            task_node.add_child(subtask_node)
        
            if self.detect_repeated_task_within_run(subtask_node):
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
                # print("is task primitive result " , is_task_primitive(subtask))
                # print(f"Checking can_execute for task '{subtask}' with state '{decompose_state}'")
                if can_execute(subtask, capabilities_input, decompose_state):
                    # print("can execute result", can_execute(subtask, capabilities_input, decompose_state))
                    # print(f"Executing task: {subtask}")
                    updated_state = self.execute_task(decompose_state, subtask_node)
                    decompose_state = updated_state
                    subtask_node.status = "completed"
                #     self.task_history_db.add_task(
                #     task_name=subtask_node.task_name,
                #     goal_task=self.goal_task,
                #     reasoning=subtask_node.context,  # Save full context as reasoning
                #     status=subtask_node.status
                # )
                #     print(f"Successfully saved task '{subtask_node.task_name}' to the database.")
                    success = True


                    
                else:
                    # If can_execute fails, attempt further decomposition instead of failing
                    print(f"Cannot execute task '{subtask}', attempting further decomposition.")
                    print(f"[DEBUG] Generated tasks before recursion: {self.generated_tasks}")
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
                    subtask_node.status = "completed"


            if subtask_node and subtask_node.status == "completed":
                # print("THE REASONING ", subtask_node.reasoning)
                # print("THE CONTEXT " , subtask_node.context)
                self.task_history_db.add_task(
                    task_name=subtask_node.task_name,
                    goal_task=self.goal_task,
                    reasoning=subtask_node.context,
                    context=subtask_node.context,
                    status=subtask_node.status
                )
                self.saved_tasks.add(task)
                # print(f"Successfully saved decomposed task '{subtask_node.task_name}' to the database.")    




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
            # print("THE REASONING ", subtask_node.reasoning)
            # print("THE CONTEXT " , subtask_node.context)
            db.add_task_node(task_node)
            self.task_history_db.add_task(
            task_name=subtask_node.task_name,
            goal_task=self.goal_task,
            reasoning=subtask_node.reasoning,
            context=subtask_node.context,  # Save only the specific reasoning here
            status=subtask_node.status
        )
            
        self.branches_children_count.append(children_count)
        print(f"Task completed: {task}")
        print("status return", task_node.status)
        
        # print(f"Task '{task}' - Average Score: {regenerated_avg_similarity_score}, Execution Time: {execution_time} seconds")
        average_depth = self.calculate_average_context_retention_depth()
        print(f"Final Average Depth of Context Retention for this run: {average_depth}")
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
    def get_subtasks(self, task, state, remaining_decompositions, capabilities_input, current_run_tasks, context, generated_tasks):
        # print("task history in get subtasks ", task_history)
        start_time = time.time()
        # Granularity check to avoid excessive decomposition
        if is_granular(task, capabilities_input):
            print(f"Task '{task}' is granular and cannot be decomposed further.")
            return []  # Do not decompose further
        

        # Adding depth control: if the task is already manageable, stop further decomposition
        if remaining_decompositions < 3:  # Adjust this number based on the desired abstraction level
            print(f"Task '{task}' has reached an appropriate level of detail.")
            return []  # No further decomposition

        task_history_str = ", ".join(current_run_tasks)

        # Get subtasks as before
        subtasks_with_types = get_subtasks(task, state, remaining_decompositions, capabilities_input, task_history_str, context, generated_tasks)
        # filtered_subtasks = [subtask for subtask in subtasks_with_types if subtask != task]
        # print(f"Decomposing task {task} into candidates:\n{subtasks_with_types}")
        # print("subtasks with types", subtasks_with_types)
        # elapsed_time = time.time() - start_time
        # print(f"Elapsed time for operation: {elapsed_time} seconds")
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
        # elapsed_time = time.time() - start_time
        # print(f"Elapsed time for operation: {elapsed_time} seconds")
        return updated_state
    
    def calculate_task_completion_rate(self):
        completion_rates = {}
        for task, attempts in self.task_attempts.items():
            completions = self.task_completions.get(task, 0)
            completion_rate = (completions / attempts) * 100 if attempts > 0 else 0
            completion_rates[task] = completion_rate
            print(f"Task: '{task}', Completion Rate: {completion_rate}%")
        return completion_rates
    
    def calculate_average_context_retention_depth(self):
        if self.context_loss_depths:
            average_depth = sum(self.context_loss_depths) / len(self.context_loss_depths)
            print(f"Average Depth of Context Retention: {average_depth}")
            return average_depth
        else:
            print("No context loss detected.")
            return 0
        


    
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
            print("Invalid input. Assuming 'no' and regenerating subtasks.")
            return False    
        
    def calculate_context_retention_rate(self):
        """
        Calculate the context retention rate as the ratio of successful context applications
        to the total detected context loss instances.
        """
        if self.context_loss_instances == 0:
            return 1.0  # No context loss detected, full retention
        successful_contexts = getattr(self, "successful_context_applications", 0)
        return successful_contexts / self.context_loss_instances

    def calculate_subtask_redundancy(self, subtasks_list):
        """
        Calculate the redundancy in subtask generation by identifying duplicate subtasks in the list.
        Returns the redundancy count and redundancy ratio.
        """
        total_subtasks = len(subtasks_list)
        unique_subtasks = len(set(subtasks_list))
        redundancy_count = total_subtasks - unique_subtasks
        redundancy_ratio = redundancy_count / total_subtasks if total_subtasks > 0 else 0
        return redundancy_count, redundancy_ratio

    def calculate_execution_success_rate(self):
        """
        Calculate the execution success rate as the ratio of completed tasks
        to the total attempted tasks.
        """
        total_attempts = sum(self.task_attempts.values())
        total_completions = sum(self.task_completions.values())
        return total_completions / total_attempts if total_attempts > 0 else 0.0
    
    def calculate_average_execution_time(self):
        """
        Calculate the average execution time for task decomposition and execution.
        """
        total_time = sum(getattr(self, "execution_times", []))
        return total_time / len(self.execution_times) if self.execution_times else 0.0

