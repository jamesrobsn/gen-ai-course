import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('student.db')

# Create a cursor object to insert record, create table
cursor = conn.cursor()

# Create table
table_info = """
CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    age INTEGER NOT NULL,
    class TEXT NOT NULL,
    grade TEXT NOT NULL
);
"""

# Execute the table creation command
cursor.execute(table_info)

## Insert some records
insert_query = """
INSERT INTO students (name, age, class, grade) VALUES (?, ?, ?, ?);
"""
students_data = [
    ('Alice', 20, 'Physics', 'A'),
    ('Bob', 22, 'Chemistry', 'B'),
    ('Charlie', 21, 'Mathematics', 'A'),
    ('David', 23, 'Biology', 'C'),  
    ('Eve', 20, 'Computer Science', 'B')
]


# Insert multiple records
cursor.executemany(insert_query, students_data)
conn.commit()

# Display the records
select_query = "SELECT * FROM students;"
cursor.execute(select_query)
records = cursor.fetchall()

# Print the records
for record in records:
    print(record)
conn.close()