import json
import os
from LLM_api import call_groq_api
import re
from openai.embeddings_utils import cosine_similarity
import time

def is_granular(task, capabilities_input):
    prompt = f"""Given the capabilities {capabilities_input}, is the task '{task}' granular enough to be directly executed or considered a primitive action by a robot based on 
    the capabilities given? For instance, if a task includes the word scan, place then it's granular because scanning can't be further broken down.
    Answer with "Yes" or "No" only."""
    response = call_groq_api(prompt, strip=True)
    # print(f"Prompt: {prompt}")
    # print("is the task granular enough? " , response)
    if(response == "yes"):
        return True
    elif(response == "No"):
        return False
    return response == "Yes"

def translate(goal_input, original_task, capabilities_input, context):
    reasoning_context = "\n".join(
        f"- {entry['reasoning']}" for entry in context if isinstance(entry, dict) and 'reasoning' in entry
    )
    prompt = f"""Given this parent goal {goal_input}, translate the task '{original_task}', 
    use the following reasoning context:\n{reasoning_context}\n\n
    Translate the task into a form that can be executed by a robot using the following capabilities:
    '{capabilities_input}'. Provide the executable form in a single line without any commentary
    or superfluous text.
    
    When translated to use the specified capabilities the result is:"""

    response = call_groq_api(prompt, strip=True)
    return response

def evaluate_candidate(goal_input, task, subtasks, capabilities_input, task_history):
    prompt = f"""Given the parent goal {goal_input}, and the parent task {task}, and its subtasks {subtasks}, 
    evaluate how well these subtasks address the requirements 
    of the parent task without any gaps or redundancies, using the following capabilities: 
    {capabilities_input}
    Return a score between 0 and 1, where 1 is the best possible score.
    
    Consider the following task history to avoid repetition:
    {', '.join(task_history)}

    Please follow this regex expression: ^[0]\.\d{{8}}$
    Provide only the score without any additional text.
    """

    response = call_groq_api(prompt, strip=True)
    print("scoreeeee", response)
    return response

def check_subtasks(task, subtasks, capabilities_input, task_history):
    prompt = f"""Given the parent task {task}, and its subtasks {', '.join(subtasks)},
    check if these subtasks effectively and comprehensively address the requirements
    of the parent task without any gaps or redundancies, using the following capabilities:
    {capabilities_input}. Return 'True' if they meet the requirements or 'False' otherwise.
    
    Consider the following task history to avoid repetition:
    {', '.join(task_history)}
    """

    response = call_groq_api(prompt, strip=True)
    return response.lower() == 'true'

def get_subtasks(task, state, remaining_decompositions, capabilities_input, current_run_tasks, context, generated_subtasks):
   
    prompt = f""" you are a robot trying to achieve the goal through sequential steps
    Given the task '{task}', the current state '{state}',
    and the following history of completed tasks: '{current_run_tasks}',
    {remaining_decompositions} decompositions remaining before failing,
    and the following capabilities: '{capabilities_input}',
    Consider the context: '{context}' to guide the decomposition process. Focus **strictly** on the **critical and unique steps** necessary to achieve the task's goal while avoiding unnecessary, overly detailed, or redundant subtasks.
    Avoid generating subtasks that simply repeat or closely resemble the main task or any tasks in '{generated_subtasks}'

    ** DO NOT START THE SENTENCE with something rather than the {capabilities_input}**
    *** Make sure each task start with one of the given capabilities input lis {capabilities_input}***
    example: words like "categorize, group,etc" should not be used instead words that are present in the list {capabilities_input} should be used

    *** Keep in mind that: You are a robot trying to achieve the goal, give tasks that mimic the robot behavior    

    decompose the task into a step-by-step plan that are a bit detailed but NOT overly detailed or decomposed.
    Provide ONLY the subtasks as a Python list of strings, without any additional text or explanations.
    Ensure that the subtasks are granular enough to be executed directly without further decomposition.
    Do not explain or summarize, and ensure the task starts directly with one of the provided capabilities list {capabilities_input}
    - **Do not generate subtasks that simply repeat or closely resemble the task given* Focus on new, unique actions.    
     
    -DO NOT GENERATE REPEATED SUBTASKS , make sure each one is distinct and cooperate into achieving the original task
    - Only provide **primitive actions** (e.g., 'move', 'grab', 'dust') and **avoid repeating** high-level tasks like the one already given. 


    Example format: ['subtask1', 'subtask2', 'subtask3']"""


    
   
   
    # prompt = f"""
    #         Given the task: '{task}' 
    #         and the current state: '{state}',

    #         With the following history of completed tasks: '{current_run_tasks}',

    #         {remaining_decompositions} decompositions remaining before failing,
    #         and the following capabilities: '{capabilities_input}',

    #         Consider the context: '{context}' to guide the decomposition process. Focus **strictly** on the **critical and unique steps** necessary to achieve the task's goal while avoiding unnecessary, overly detailed, or redundant subtasks.
    #         decompose the task into a step-by-step plan that are a bit detailed but NOT overly detailed or decomposed.

    #         ### Rules for Subtasks:
    #         1. Generate **no more than 3-5 essential subtasks** that are strictly necessary for achieving the task.
    #         2. Each subtask must:
    #             - Begin with an **actionable verb** strictly listed in '{capabilities_input}'.
    #             - Be concise and unique, avoiding duplication of completed tasks or subtasks in '{generated_subtasks}'.
    #             - Directly align with the main task and the provided context.
    #         3. Reject subtasks that:
    #             - Restate or paraphrase the main task.
    #             - Are overly fine-grained, trivial, or out of scope.
    #             - Use verbs or actions not explicitly listed in '{capabilities_input}'.

    #         ### Formatting Instructions:
    #         - Respond only with a **Python list of strings**.
    #         - Do not include explanations, extra details, or formatting beyond the list.
    #         - Example response format:
    #         ['subtask1', 'subtask2', 'subtask3']
    #     """

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
    """
    Generate subtasks using task, state, and context to ensure subtasks are aligned with updated context.
    """
    
    print(f"Given the task '{task}', the current state '{state}', and the following history of completed tasks: '{current_run_tasks}'",

    f"you have {remaining_decompositions} decomposition steps remaining to achieve the goal."

    f"Capabilities available: '{capabilities_input}' "

    f"Context to guide the decomposition: '{context}'")


    prompt = f"""
    Given the task '{task}', the current state '{state}', and the following history of completed tasks: '{current_run_tasks}',

    you have {remaining_decompositions} decomposition steps remaining to achieve the goal.

    Capabilities available: '{capabilities_input}' 

    Context to guide the decomposition: '{context}'

    Decompose the task into actionable subtasks that effectively utilize the updated context. Ensure the steps are:
    
    - Essential to accomplish the task efficiently, avoiding redundant or unnecessary actions.
    - Actionable and aligned with the specified capabilities, starting with one of the provided capability keywords.
    - Prioritized based on the updated context to improve relevance and execution quality.
    - Avoiding excessive detail or repetition.
    - Limited to **3-7 high-priority subtasks** that directly contribute to achieving the task given the context and the main goal.

    **Output Format**: Provide ONLY a Python list of subtasks as strings. Do not include explanations or additional text.
    **give the output only as the desired format and no  any additional texts.**

    Example:
    ['subtask1', 'subtask2', 'subtask3']
    """
    response = call_groq_api(prompt, strip=True)
    print("get subtasks with context" , response)
    try:
        subtasks = eval(response)
        return subtasks if isinstance(subtasks, list) else []
    except:
        print(f"Error parsing subtasks: {response}")
        return []

def is_task_similar(task_name, current_run_tasks, thresholdofsimilar=0.97):
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
                print(f"[INFO] Task '{task_name}' is similar to '{run_task}' with score {score} (threshold: {thresholdofsimilar})")
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
    # print(f"Task '{task_name}' is not similar to any of the current run tasks.")
    return False


