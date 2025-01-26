import datetime
import os
from LLM_api import call_groq_api, log_response
from text_utils import trace_function_calls
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util


model = SentenceTransformer('paraphrase-MiniLM-L12-v2')


def generate_hierarchy_embedding(goal, subtasks, goal_weight=0.8):
    """
    Generate a cumulative embedding for the goal and each contextualized subtask.
    """
    goal_embedding = model.encode(goal, convert_to_tensor=True)
    
    # Re-contextualize each subtask with the goal to improve alignment
    contextualized_subtasks = [f"{goal}. Step: {subtask}" for subtask in subtasks]
    subtask_embeddings = model.encode(contextualized_subtasks, convert_to_tensor=True)
    # print(goal_weight ,"gw")
    # print(goal_embedding, "ge")
    # print(subtask_embeddings ,"se")
    # Calculate a cumulative embedding that weighs the goal and average subtask embeddings
    cumulative_embedding = goal_weight * goal_embedding + (1 - goal_weight) * subtask_embeddings.mean(axis=0)
    
    return cumulative_embedding

def calculate_similarity(parent_task, subtask,  goal_embedding=None):
    """
    Calculate similarity between the parent task and each subtask.
    """
    # Generate embeddings for the parent task and subtask, ensuring they are tensors
    contextualized_subtask = f"{parent_task}: {subtask}"
    parent_embedding = model.encode(parent_task, convert_to_tensor=True) if goal_embedding is None else goal_embedding
    subtask_embedding = model.encode(contextualized_subtask, convert_to_tensor=True)
    
    # Compute similarity
    similarity_score = util.pytorch_cos_sim(parent_embedding, subtask_embedding).item()
    return round(similarity_score, 3)


@trace_function_calls
def groq_is_goal(state, goal_task):
    prompt = (f"Given the current state '{state}' and the goal '{goal_task}', "
              f"determine if the current state satisfies the goal. "
              f"Please provide the answer as 'True' or 'False':")

    response = call_groq_api(prompt, strip=True)

    log_response("groq_is_goal", response)
    return response.lower() == "true"

@trace_function_calls
def get_initial_task(goal):
    prompt = f"Given the goal '{goal}', suggest a high level task that will complete it:"

    response = call_groq_api(prompt, strip=True)
    log_response("get_initial_task", response)
    return response

@trace_function_calls
def is_task_primitive(task_name):
    lemmatizer = WordNetLemmatizer()
    task_words = task_name.lower().split()
    
    primitive_actions_keywords = [
        'grab', 'reach', 'twist', 'move', 'push', 'pull', 'lift', 'hold',
        'release', 'turn', 'rotate', 'locate', 'identify', 'find', 'pick', 'book',
        'place', 'put', 'insert', 'remove', 'open', 'close', 'clean',
        'wipe', 'sweep', 'mop', 'vacuum', 'dust', 'wash', 'rinse', 'cook', 'heat',
        'boil', 'fry', 'bake', 'microwave', 'cut', 'slice', 'dice', 'chop', 'examine',
        'grate', 'peel', 'mix', 'blend', 'stir', 'pour', 'serve', 'stop', 'scan', 'activate','measure', 'read'
        ]

    for word in task_words:
        lemma = lemmatizer.lemmatize(word)
        if lemma in primitive_actions_keywords:
            # print("is task primtive?", "yes")
            return True
    
    # print("is task primtive?", "No")
    return False

@trace_function_calls
def compress_capabilities(text):
    # Remove newline characters and strip any extra whitespace
    text_cleaned = " ".join(text.split())
    
    # Updated prompt with single-line format
    prompt = f"From the following capabilities '{text_cleaned}', return just the action verbs without any explanatory text. Return them as a comma-separated list of primitive actions."
    
    response = call_groq_api(prompt, strip=True)
    return response

def can_execute(task, capabilities, state):
    print("Can the task be executed?")
    prompt = (
        f"Task: '{task}'\n"
        f"Current State: '{state}'\n"
        f"Available Capabilities: {capabilities}\n"
        "Determine whether the task can be executed directly or if it requires further decomposition based on:\n"
        "- Available capabilities (e.g., does the task align directly with the listed capabilities?).\n"
        "- Execution granularity (e.g., is the task a high-level goal or a low-level action?).\n"
        "\n"
        "Guidelines:\n"
        "- If the task is **simple** and matches available capabilities, answer 'True'.\n"
        "- If the task is **complex**, requires multiple steps, or involves actions not directly supported by the capabilities, answer 'False'.\n"
        "\n"
        "Provide your answer as 'True' or 'False' only."
    )
    # print("inside can_execute")
    # print("prompt for can_execute" , prompt)
    response = call_groq_api(prompt, strip=True)
    # print("can execute response", response)

    log_response("can_execute", response)   
    print(f"{response}")
    return response.lower() == "true"
    

def log_state_change(prev_state, new_state, task):
    log_dir = "../state_changes"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_path = f"{log_dir}/state_changes.log"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(log_file_path, "a") as log_file:
        log_file.write(f"{timestamp}: Executing task '{task}'\n")
        log_file.write(f"{timestamp}: Previous state: '{prev_state}'\n")
        log_file.write(f"{timestamp}: New state: '{new_state}'\n\n")