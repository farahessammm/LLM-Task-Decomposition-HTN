import os
from LLM_api import call_groq_api

def is_granular(task, capabilities_input):
    prompt = f"""Given the capabilities {capabilities_input}, is the task '{task}' granular enough to be directly executed or considered a primitive action by a robot based on 
    the capabilities given? For instance, if a task includes the word scan, then it's granular because scanning can't be further broken down. Answer with "Yes" or "No" only."""
    response = call_groq_api(prompt, strip=True)
    # print(f"Prompt: {prompt}")
    # print("is the task granular enough? " , response)
    if(response == "yes"):
        return True
    elif(response == "No"):
        return False
    return response == "Yes"

def translate(goal_input, original_task, capabilities_input):
    prompt = f"""Given this parent goal {goal_input}, translate the task '{original_task}' into a form that can be executed by a robot using the following capabilities:
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

def get_subtasks(task, state, remaining_decompositions, capabilities_input, task_history_str):
    # print("capabilities", {capabilities_input})
    prompt = f"""Given the task '{task}', the current state '{state}',
    and the following history of completed tasks: '{task_history_str}',
    {remaining_decompositions} decompositions remaining before failing,
    and the following capabilities: '{capabilities_input}',
    decompose the task into a step-by-step plan that are a bit detailed but NOT overly detailed or decomposed.
    Provide ONLY the subtasks as a Python list of strings, without any additional text or explanations.
    Ensure that the subtasks are granular enough to be executed directly without further decomposition.
    Do not explain or summarize, and ensure the task starts directly with one of the provided capabilities and the subtask
    - **Do not generate subtasks that simply repeat or closely resemble the task given* Focus on new, unique actions.   
     ** DO NOT START THE SENTENCE with something rather than the {capabilities_input}**
    -DO NOT GENERATE REPEATED SUBTASKS , make sure each one is distinct and cooperate into achieving the original task
    - Only provide **primitive actions** (e.g., 'move', 'grab', 'dust') and **avoid repeating** high-level tasks like the one already given. 


    Example format: ['subtask1', 'subtask2', 'subtask3']"""

    response = call_groq_api(prompt, strip=True)
    try:
        subtasks = eval(response)
        return subtasks if isinstance(subtasks, list) else []
    except:
        # print(f"Error parsing subtasks: {response}")
        return []