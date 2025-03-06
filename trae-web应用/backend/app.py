from flask import Flask, request, jsonify
from flask_cors import CORS
import pyodbc
from datetime import datetime

app = Flask(__name__)
CORS(app)

# 数据库连接配置
DB_CONFIG = {
    'driver': 'SQL Server',
    'server': 'localhost',
    'database': 'TaskManagement',
    'trusted_connection': 'yes'
}

def get_db_connection():
    conn_str = (
        f"DRIVER={{{DB_CONFIG['driver']}}};"
        f"SERVER={DB_CONFIG['server']};"
        f"DATABASE={DB_CONFIG['database']};"
        f"Trusted_Connection={DB_CONFIG['trusted_connection']};"
    )
    return pyodbc.connect(conn_str)

@app.route('/api/tasks', methods=['GET'])
def get_tasks():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM Tasks')
    tasks = [dict(zip([column[0] for column in cursor.description], row)) 
             for row in cursor.fetchall()]
    conn.close()
    return jsonify(tasks)

@app.route('/api/tasks', methods=['POST'])
def create_task():
    task_data = request.json
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO Tasks (title, priority, due_date, status)
        VALUES (?, ?, ?, ?)
    ''', (task_data['title'], task_data['priority'], 
          task_data['dueDate'], 'active'))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Task created successfully'})

@app.route('/api/tasks/<int:task_id>', methods=['PUT'])
def update_task(task_id):
    task_data = request.json
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        UPDATE Tasks 
        SET title=?, priority=?, due_date=?, status=?
        WHERE id=?
    ''', (task_data['title'], task_data['priority'], 
          task_data['dueDate'], task_data['status'], task_id))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Task updated successfully'})

@app.route('/api/tasks/<int:task_id>', methods=['DELETE'])
def delete_task(task_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('DELETE FROM Tasks WHERE id=?', (task_id,))
    conn.commit()
    conn.close()
    return jsonify({'message': 'Task deleted successfully'})

if __name__ == '__main__':
    app.run(debug=True)