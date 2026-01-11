# create_gm.py
from dotenv import load_dotenv
from db import get_conn, ensure_schema, create_user

load_dotenv()  # lädt .env aus dem aktuellen Ordner

username = input("GM username: ").strip()
password = input("GM password: ").strip()

conn = get_conn()
ensure_schema(conn)

create_user(conn, username=username, password=password, role="gm", country=None)
conn.close()

print("✅ GM user created/updated.")
