import os

import psycopg2

DB_CONFIG = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': '123456',
    'host': 'localhost',
    'port': 5432
}
def test_connection():
    try:
        conn = get_connection()
        with conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1;")
                result = cur.fetchone()
                print("✅ Database connection successful! Result:", result)
    except Exception as e:
        print("❌ Database connection failed:", e)
    finally:
        if conn:
            conn.close()

def get_connection():
    print(f"Connecting to: dbname={DB_CONFIG['dbname']} user={DB_CONFIG['user']}")
    return psycopg2.connect(**DB_CONFIG)

def insert_anomaly(camera_id, timestamp, anomaly_type, frame_path):
    conn = get_connection()
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO surveillance_alerts (camera_id, timestamp, anomaly_type, frame_path)
                    VALUES (%s, %s, %s, %s)
                    ON CONFLICT DO NOTHING;
                """, (camera_id, timestamp, anomaly_type, os.path.dirname(frame_path)))
    finally:
            conn.close()

def fetch_alerts(camera_id=None, anomaly_type=None, timestamp=None):
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            query = "SELECT * FROM surveillance_alerts WHERE 1=1"
            params = []

            if camera_id:
                query += " AND camera_id = %s"
                params.append(camera_id)
            if anomaly_type:
                query += " AND anomaly_type = %s"
                params.append(anomaly_type)
            if timestamp:
                query += " AND timestamp::text LIKE %s"
                params.append(timestamp + "%")

            cur.execute(query, params)
            return cur.fetchall()
    finally:
        conn.close()
