import sqlite3
import json

class VectorDB:
    def __init__(self, db_path="task_history.db"):
        # Initialize SQLite connection and create table if not exists
        self.conn = sqlite3.connect(db_path)
        self.create_table()

    def create_table(self):
        # Create a table to store task nodes if it doesn't already exist
        query = '''
        CREATE TABLE IF NOT EXISTS task_nodes (
            node_name TEXT PRIMARY KEY,
            task TEXT,
            context TEXT,
            state TEXT
        )
        '''
        self.conn.execute(query)
        self.conn.commit()

    def add_task_node(self, task_node, task=None, context=None, state=None):
        task = task or "Unknown Task"  # Default to 'Unknown Task' if task is None
        context = json.dumps(context or [])  # Serialize the context to store as text
        state = state or "No state available"

        # Insert or replace the task node data in SQLite
        query = '''
        INSERT OR REPLACE INTO task_nodes (node_name, task, context, state)
        VALUES (?, ?, ?, ?)
        '''
        self.conn.execute(query, (task_node.node_name, task, context, state))
        self.conn.commit()

        print(f"Storing metadata for task '{task_node.node_name}': Task='{task}', State='{state}'")

    def get_task_node(self, task_node):
        query = "SELECT * FROM task_nodes WHERE node_name = ?"
        cursor = self.conn.execute(query, (task_node.node_name,))
        row = cursor.fetchone()

        if row:
            metadata = {
                'task': row[1],
                'context': json.loads(row[2]),  # Deserialize the JSON context
                'state': row[3]
            }
            return metadata
        else:
            raise ValueError(f"Task '{task_node.node_name}' not found in the database.")

    def get_all_task_nodes(self):
        # Fetch all task nodes from the SQLite database
        query = "SELECT * FROM task_nodes"
        cursor = self.conn.execute(query)
        rows = cursor.fetchall()

        tasks = []
        for row in rows:
            tasks.append({
                'node_name': row[0],
                'task': row[1],
                'context': json.loads(row[2]),
                'state': row[3]
            })
        return tasks

    def delete_task_node(self, task_node):
        # Delete a task node from the SQLite database
        query = "DELETE FROM task_nodes WHERE node_name = ?"
        self.conn.execute(query, (task_node.node_name,))
        self.conn.commit()

        print(f"Task '{task_node.node_name}' deleted from the database.")
