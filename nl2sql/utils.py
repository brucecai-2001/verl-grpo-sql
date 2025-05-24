import sqlite3

def query_database(db_path: str, sql: str):
    """
        Query Sqlite3
    """
    try:
        # connect db
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # execute sql
        cursor.execute(sql)
        results = cursor.fetchall()

        conn.close()
        return results
        
    except sqlite3.Error as e:
        print(f"数据库操作出错: {e}")