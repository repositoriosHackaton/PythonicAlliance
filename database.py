import sqlite3
import numpy as np

def create_tables():
    conn = sqlite3.connect('iris_database.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 username TEXT UNIQUE NOT NULL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS embeddings (
                 id INTEGER PRIMARY KEY AUTOINCREMENT,
                 user_id INTEGER,
                 embedding BLOB,
                 FOREIGN KEY (user_id) REFERENCES users (id))''')
    conn.commit()
    conn.close()

def save_user_embeddings(username, embeddings):
    conn = sqlite3.connect('iris_database.db')
    c = conn.cursor()
    c.execute("INSERT INTO users (username) VALUES (?)", (username,))
    user_id = c.lastrowid
    for emb in embeddings:
        c.execute("INSERT INTO embeddings (user_id, embedding) VALUES (?, ?)", (user_id, emb.tobytes()))
    conn.commit()
    conn.close()

def find_best_match(new_embedding, threshold=0.85):
    conn = sqlite3.connect('iris_database.db')
    c = conn.cursor()
    c.execute("SELECT user_id, embedding FROM embeddings")
    results = c.fetchall()
    best_similarity = 0
    best_user_id = None

    for user_id, emb_blob in results:
        embedding = np.frombuffer(emb_blob, dtype=np.float32)
        similarity = cosine_similarity(new_embedding, embedding)
        if similarity > best_similarity:
            best_similarity = similarity
            best_user_id = user_id

    if best_similarity > threshold:
        c.execute("SELECT username FROM users WHERE id=?", (best_user_id,))
        user = c.fetchone()[0]
        return user, best_similarity
    else:
        return None, best_similarity

def cosine_similarity(embedding1, embedding2):
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    return dot_product / (norm1 * norm2)
