import sqlite3
conn = sqlite3.connect("artworks.db")
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS artworks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        title TEXT,
        artist TEXT,
        medium TEXT,
        style TEXT,
        theme TEXT,
        embedding TEXT
    )
''')

conn.commit()
conn.close()
