import chromadb


class VectorDB:
    def __init__(self):
        # Create a chroma client
        self.client = chromadb.Client()

        # Create a collection
        self.collection = self.client.create_collection("task_nodes")

    def add_task_node(self, task_node):
        # metadata = serialize_task_node(task_node)
        metadata = serialize_task_node(task_node)
        print("metadata: ", metadata)
      #  self.collection.upsert(documents=[task_node.task_name], ids=[task_node.node_name], metadatas=[metadata])
        self.collection.upsert(documents=[task_node.task_name], ids=[task_node.node_name])

    def get_task_node(self, task_node):
        return self.collection.get(ids=[task_node.node_name])[0]

    # def query_by_name(self, task_name):
    #     task_nodes = self.collection.query(query_texts=[task_name], n_results=1)
    #     return task_nodes['metadatas'][0]



    def add_task_node(self, task_node):
        # metadata = serialize_task_node(task_node)
        metadata = serialize_task_node(task_node)
        print("metadata: ", metadata)
      #  self.collection.upsert(documents=[task_node.task_name], ids=[task_node.node_name], metadatas=[metadata])
        self.collection.upsert(documents=[task_node.task_name], ids=[task_node.node_name])

    # def get_task_node(self, task_node):
    #     return self.collection.get(ids=[task_node.node_name])['metadatas'][0]

    # def query_by_name(self, task_name):
    #     task_nodes = self.collection.query(query_texts=[task_name], n_results=1)
    #     return task_nodes['metadatas'][0]

def serialize_task_node(task_node):
    # Create a dictionary with all attributes that need to be serialized
    serialized_data = {
        'task_name': task_node.task_name,
        'node_name': task_node.node_name,
        'status': task_node.status,
        'children': [child.node_name for child in task_node.children]  # assuming children is a list of TaskNode objects
    }
    # Optionally, include parent's node_name if it exists and is necessary
    if hasattr(task_node, 'parent') and task_node.parent is not None:
        serialized_data['parent'] = task_node.parent.node_name
    else:
        serialized_data['parent'] = None

    return serialized_data

def serialize_task_node(task_node):
    # Prepare the dictionary with all necessary fields
    serialized_data = {
        'task_name': task_node.task_name,
        'node_name': task_node.node_name,
        'status': task_node.status,
        # Convert parent node to its node_name or None if no parent
        'parent': task_node.parent.node_name if task_node.parent else None,
        # Convert children nodes to a list of their node_names
        'children': [child.node_name for child in task_node.children] if task_node.children else []
    }
    return serialized_data