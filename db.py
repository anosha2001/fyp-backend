import psycopg2

DB_CONFIG = {
    'dbname': 'surveillance_system',
    'user': 'your_user',
    'password': 'your_password',
    'host': 'localhost',
    'port': 5432
}

def get_connection():
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
                """, (camera_id, timestamp, anomaly_type, frame_path))
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
