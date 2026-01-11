import sqlite3

# Datenbank erstellen
conn = sqlite3.connect("game.db")
cursor = conn.cursor()

# Tabellen erstellen
cursor.execute("""
CREATE TABLE IF NOT EXISTS countries (
    name TEXT PRIMARY KEY,
    military INTEGER,
    stability INTEGER,
    economy INTEGER,
    diplomatic_influence INTEGER,
    public_approval INTEGER,
    ambition TEXT
);
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS turn_history (
    turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
    country TEXT,
    military INTEGER,
    stability INTEGER,
    economy INTEGER,
    diplomatic_influence INTEGER,
    public_approval INTEGER,
    action_public TEXT,
    action_private TEXT,
    action_internal TEXT,
    global_context TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(country) REFERENCES countries(name)
);
""")

# Startdaten für Deutschland einfügen
cursor.execute("""
INSERT OR IGNORE INTO countries VALUES
('Germany', 80, 90, 95, 95, 85, 'Weaken far-right, lead EU, energy transition');
""")

conn.commit()
conn.close()
print("Datenbank erstellt!")
