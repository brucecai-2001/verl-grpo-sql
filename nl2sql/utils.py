import sqlite3

def is_single_statement(sql):
    stripped = sql.strip()
    return not stripped.endswith(';') or stripped.count(';') == 0

def query_database(db_path: str, sql: str):
    """
        Query Sqlite3
    """
    #print(f"querying dabase {sql}")
    if not is_single_statement(sql):
        return None

    try:
        # connect db
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # execute sql
        cursor.executescript(sql)
        results = cursor.fetchall()

        conn.close()
        return results
        
    except sqlite3.Error as e:
        print(f"数据库操作出错: {e}")
