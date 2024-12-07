import sqlite3
import json
from datetime import datetime

class TaskHistoryDB:
    def __init__(self, db_path="task_history.db"):
        self.conn = sqlite3.connect(db_path)
        self.create_table()

    def create_table(self):
        query = '''
       CREATE TABLE IF NOT EXISTS task_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_name TEXT NOT NULL,
    goal_task TEXT,
    reasoning TEXT,
    context TEXT,
    status TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
'''
        self.conn.execute(query)
        self.conn.commit()

    def add_task(self, task_name, goal_task,reasoning, context, status):
        query = '''
        INSERT INTO task_history (task_name, goal_task, reasoning, context, status, timestamp)
        VALUES (?, ?, ?, ?,?,?)
        '''
        reasoning_json = json.dumps(reasoning) if isinstance(reasoning, (dict, list)) else reasoning
        context_json = json.dumps(context) if isinstance(context, (dict, list)) else context
        status_json = json.dumps(status) if isinstance(status, (dict, list)) else status
        timestamp = datetime.now().isoformat()
        self.conn.execute(query, (task_name, goal_task, reasoning_json, context_json, status_json, timestamp))
        self.conn.commit()

    def get_task_history(self, limit=100):
        query = '''
        SELECT task_name,goal_task,reasoning, context, status, timestamp
        FROM task_history
        ORDER BY timestamp DESC
        LIMIT ?
        '''
        cursor = self.conn.execute(query, (limit,))
        rows = cursor.fetchall()
        return [
            {
                'task_name': row[0],
                'goal_task': row[1],
                'reasoning': json.loads(row[2]),
                'context':json.loads(row[3]), 
                'status': row[4],
                'timestamp': row[5]
            }
            for row in rows
        ]
    def get_similar_tasks(self, task, limit=5):
        """
        Retrieves similar tasks from the task history, filtering by task name.
        :param task: The name of the task to search for similar tasks.
        :param limit: The maximum number of similar tasks to retrieve.
        :return: A list of task dictionaries with keys 'task_name', 'goal_task', 'reasoning', and 'status'.
        """
        query = '''
        SELECT task_name, goal_task, reasoning, context, status, timestamp
        FROM task_history
        WHERE task_name LIKE ?
        ORDER BY timestamp DESC
        LIMIT ?
        '''
        
        cursor = self.conn.execute(query, (f'%{task}%', limit))
        rows = cursor.fetchall()

        similar_tasks = []
        for row in rows:
            try:
                # Attempt to load reasoning as JSON
                reasoning = json.loads(row[2]) if row[2] else None
                context = json.loads(row[3]) if row[3] else None
                    
                similar_tasks.append({
                'task_name': row[0],
                'goal_task': row[1],
                'reasoning': reasoning,
                'context': context,
                'status': row[4],
                'timestamp': row[5]
            })
            except json.JSONDecodeError:
                # Handle JSON decode errors
                print(f"Error decoding JSON for task '{row[0]}', skipping this entry.")
                continue

        return similar_tasks


    def delete_all_tasks(self):
        """Delete all records from the task_history table."""
        query = "DELETE FROM task_history"
        self.conn.execute(query)
        self.conn.commit()
        print("All tasks have been deleted from the database.")
    def close(self):
        self.conn.close()