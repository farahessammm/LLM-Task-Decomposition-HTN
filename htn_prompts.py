import json
import os
from LLM_api import call_groq_api
import re
from openai.embeddings_utils import cosine_similarity
import time

# def is_granular(task, capabilities_input):
#     prompt = f"""Given the capabilities {capabilities_input}, is the task '{task}' granular enough to be directly executed or considered a primitive action by a robot based on 
#     the capabilities given? For instance, if a task includes the word scan, place then it's granular because scanning can't be further broken down.
#     Answer with "Yes" or "No" only."""
#     response = call_groq_api(prompt, strip=True)
#     # print(f"Prompt: {prompt}")
#     # print("is the task granular enough? " , response)
#     if(response == "yes"):
#         return True
#     elif(response == "No"):
#         return False
#     return response == "Yes"

# def translate(goal_input, original_task, capabilities_input, context):
#     reasoning_context = "\n".join(
#         f"- {entry['reasoning']}" for entry in context if isinstance(entry, dict) and 'reasoning' in entry
#     )
#     prompt = f"""Given this parent goal {goal_input}, translate the task '{original_task}', 
#     use the following reasoning context:\n{reasoning_context}\n\n
#     Translate the task into a form that can be executed by a robot using the following capabilities:
#     '{capabilities_input}'. Provide the executable form in a single line without any commentary
#     or superfluous text.
    
#     When translated to use the specified capabilities the result is:"""

#     response = call_groq_api(prompt, strip=True)
#     return response


def get_subtasks(task, state, remaining_decompositions, capabilities_input, current_run_tasks, context, generated_subtasks):
   
    prompt = f"""
        You are a robot tasked with achieving the goal by executing **sequential and critical steps** in a structured and efficient manner. Your goal is to decompose the task into primitive, actionable subtasks that strictly adhere to the provided capabilities.

        **Input Details:**
        - **Task:** '{task}'
        - **Current State:** '{state}'
        - **Completed Tasks History:** '{current_run_tasks}'
        - **Remaining Decompositions Before Failure:** {remaining_decompositions}
        - **Capabilities:** '{capabilities_input}'
        - **Context:** '{context}'
        - **Generated Subtasks So Far:** '{generated_subtasks}'

        Leverage the context: '{context}' to guide the decomposition process. Ensure that each generated subtask is **critical**, **unique**, and directly contributes to achieving the task's goal. Avoid including unnecessary, overly detailed, redundant, or vague steps.

        **Guidelines for Subtask Generation:**
        1. **Strict Adherence to Capabilities:**
        # - ***Every subtask **must start with an action** from the provided list of capabilities: {capabilities_input}.
        - Avoid using generic terms such as "categorize" or "group" or "read" unless they explicitly match the capability list.
        - Avoid generating subtasks that repeat or closely resemble each other or the original task.

        2. **Focus on Primitive Actions:**
        - Subtasks must be granular and executable directly by the robot without requiring further decomposition.
        - Do not include high-level tasks similar to the main task or previously completed tasks.
        - Avoid generating subtasks for verification (verify, confirm , etc)
        - ****do not generate subtasks like "repeat for all", "repeat" etc.****

        3. **Uniqueness and Relevance:**
        - Each subtask must be distinct, new, and necessary to achieve the main task.
        - Avoid redundant, overly detailed, or irrelevant steps.
        - Ensure subtasks cooperate to achieve the original task without duplicating efforts.
        - Avoid generating the task provided again.

        4. **Conciseness and Precision:**
        - Provide only the list of subtasks as a Python list of strings.
        - Do not add any explanations, summaries, or non-critical information.

        5. **Ensure Full Coverage**:
        - Subtasks must collectively address **all aspects of the task** described in the goal, including preparation, execution, and finalization phases.
        - Review the list to ensure that no critical steps are omitted. Add steps as necessary to cover all phases.


        **Formatting Requirements:**
        - Example Output: ['subtask1', 'subtask2', 'subtask3']
        - Subtasks must be clear, concise, and actionable.

        **Keep in Mind:**
        - The robot's behavior must mimic real-world execution, focusing on what a robot can practically perform using the specified capabilities.
        - Avoid generating subtasks that repeat or closely resemble each other or the original task.

        Decompose the task '{task}' into an actionable step-by-step plan following these guidelines and constraints.
        """

    response = call_groq_api(prompt, strip=True)
    # print("subtasks of response", response)
    try:
        # Use regex to find the first valid Python list in the response
        match = re.search(r"\[.*?\]", response, re.DOTALL)
        if match:
            subtasks = eval(match.group(0))  # Evaluate the extracted list
            if isinstance(subtasks, list):
                return subtasks
    except Exception as e:
        print(f"Error parsing subtasks: {e}")
    
    print(f"Error parsing subtasks: {response}")
    return []
    
def get_subtasks_with_context(task, state, remaining_decompositions, capabilities_input, context, current_run_tasks, generated_subtasks):
    

    prompt = f"""
            You are a robot tasked with achieving the goal of decomposing tasks into actionable subtasks using the provided updated context.

            **Input Details:**
            - **Task:** '{task}'
            - **Current State:** '{state}'
            - **Completed Tasks History:** '{current_run_tasks}'
            - **Remaining Decompositions Before Failure:** {remaining_decompositions}
            - **Capabilities:** '{capabilities_input}'
            - **Updated Context:** '{context}'
            - **Generated Subtasks:** '{generated_subtasks}'

            **Guidelines for Subtask Generation:**
            1. **Leverage Updated Context:**
            - Ensure the generated subtasks utilize and align with the provided updated context.
            - Prioritize actions that directly address the insights and requirements specified in the context.

            2. **Avoid Redundancy:**
            - Exclude subtasks that duplicate previously completed tasks or closely resemble already generated subtasks.
            - Do not include subtasks that restate the main task.

            3. **Focus on Actionable and Relevant Steps:**
            - Subtasks must begin with an action from the provided capabilities: {capabilities_input}.
            - Ensure subtasks are granular, directly executable, and necessary for achieving the task.

            4. **Optimize for Efficiency:**
            - Generate only the essential subtasks required to achieve the goal efficiently.
            - Avoid overly detailed decomposition that adds unnecessary steps or complexity.

            5. **Limit to Critical Steps:**
            - Restrict the number of subtasks to a maximum of **3-15 actionable steps**.
            - Each subtask must be distinct, contributing meaningfully to the overall goal.
            - Avoid generating subtasks for verification (verify, confirm , etc)
            - Avoid generating subtasks like "repeat for all"

            **Critical Output Requirement:**
            - Respond ONLY with a valid Python list of strings in the format: `['subtask1', 'subtask2', 'subtask3']`.
            - DO NOT include any additional text, explanations, or formatting beyond the Python list.

            **Output Example:**
            ['subtask1', 'subtask2', 'subtask3']

            Decompose the task '{task}' into subtasks following these guidelines and constraints.
            """
    response = call_groq_api(prompt, strip=True)
    try:
        subtasks = eval(response)
        return subtasks if isinstance(subtasks, list) else []
    except:
        print(f"Error parsing subtasks: {response}")
        return []

def is_task_similar(task_name, current_run_tasks, thresholdofsimilar=0.97):
    print("is the task repeated?")
    # start_time = time.time()
    # print("inside is task similar")
    if not current_run_tasks:
        print("No tasks to compare against.")
        return False
    tasks_to_compare = [task for task in current_run_tasks]

    prompt = f"""
        Compare the following task with multiple tasks and determine their similarity on a scale from 0 to 1 with a precision of up to three decimal places:
        - Task to Compare: "{task_name}"
        - Tasks to Compare Against: {json.dumps(tasks_to_compare)}

        Return a JSON object in the following format:
        {{
        "task_name_1": similarity_score,
        "task_name_2": similarity_score,
        ...
        }}

        Do not include any explanations, reasoning, or additional text in the response.
    """

    try:
        # Call the API
        response = call_groq_api(prompt, strip=True)

        # Parse the response as JSON
        similarity_scores = json.loads(response)
        if not isinstance(similarity_scores, dict):
            raise ValueError(f"Unexpected format in API response: {response}")

        # print("Similarity scores:", similarity_scores)

        # Check for similarity against the threshold
        for run_task, score in similarity_scores.items():
            if float(score) >= thresholdofsimilar:
                print(f"Task '{task_name}' is similar to '{run_task}' with score {score} (threshold: {thresholdofsimilar})")
                
                # elapsed_time = time.time() - start_time
                # print(f"Elapsed time for operation: {elapsed_time:.2f} seconds")
                return True

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON response: {e}")
        print(f"Raw response: {response}")
    except Exception as e:
        print(f"Error during similarity check: {e}")

    # elapsed_time = time.time() - start_time
    # print(f"Elapsed time for operation: {elapsed_time:.2f} seconds")
    print(f"Task '{task_name}' is not similar to any of the current run tasks.")
    return False



