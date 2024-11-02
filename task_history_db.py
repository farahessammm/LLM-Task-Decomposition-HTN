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
    status TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
'''
        self.conn.execute(query)
        self.conn.commit()

    def add_task(self, task_name, goal_task,reasoning, status):
        query = '''
        INSERT INTO task_history (task_name, goal_task, reasoning, status,timestamp)
        VALUES (?, ?, ?, ?,?)
        '''
        reasoning_json = json.dumps(reasoning)
        status_json = json.dumps(status)
        timestamp = datetime.now().isoformat()
        self.conn.execute(query, (task_name,goal_task, reasoning_json,  status_json , timestamp))
        self.conn.commit()

    def get_task_history(self, limit=100):
        query = '''
        SELECT task_name,goal_task,reasoning, status, timestamp
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
                'status': row[3],
                'timestamp': row[4]
            }
            for row in rows
        ]

    def get_similar_tasks(self, task, limit=5):
        query = '''
        SELECT task_name,goal_task,reasoning, status
        FROM task_history
        WHERE task_name LIKE ?
        ORDER BY timestamp DESC
        LIMIT ?
        '''
        cursor = self.conn.execute(query, (f'%{task}%', limit))
        rows = cursor.fetchall()
        return [
            {
                'task_name': row[0],
                'goal_task': row[1],
                'reasoning': json.loads(row[2]),
                'status': row[3]
            }
            for row in rows
        ]
    def delete_all_tasks(self):
        """Delete all records from the task_history table."""
        query = "DELETE FROM task_history"
        self.conn.execute(query)
        self.conn.commit()
        print("All tasks have been deleted from the database.")
    def close(self):
        self.conn.close()