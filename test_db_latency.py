import time
import sqlite3

def test_get_latest_session_id(session_name="session1"):
    conn = sqlite3.connect("textnexus.db")
    cursor = conn.cursor()

    start = time.perf_counter()
    cursor.execute("SELECT id FROM chat_sessions WHERE session_name = ? ORDER BY id DESC LIMIT 1", (session_name,))
    row = cursor.fetchone()
    end = time.perf_counter()

    conn.close()

    elapsed_ms = (end - start) * 1000
    print(f"Session ID: {row[0] if row else 'None'}")
    print(f"Query Time: {elapsed_ms:.2f} ms")

    if elapsed_ms > 200:
        print("❌ FAILED: Query took longer than 200 ms")
    else:
        print("✅ PASSED: Query completed within 200 ms")

if __name__ == "__main__":
    test_get_latest_session_id()