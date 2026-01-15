# countries.py

COUNTRY_DEFS = {
    "Germany": {
        "display_name": "Deutschland",
        "military": 70,
        "stability": 90,
        "economy": 95,
        "diplomatic_influence": 85,
        "public_approval": 75,
        "ambition": "EU führen, wirtschaftliche Stärke sichern, politische Extreme eindämmen",
        "win_conditions": [
            {"metric": "economy", "op": ">=", "value": 97, "label": "Starke Wirtschaft (≥ 97)"},
            {"metric": "stability", "op": ">=", "value": 92, "label": "Hohe innenpolitische Stabilität (≥ 92)"},
            {"metric": "diplomatic_influence", "op": ">=", "value": 88, "label": "Führende diplomatische Rolle (≥ 88)"},
            {"metric": "eu_cohesion", "op": ">=", "value": 80, "label": "EU-Kohäsion stabil (≥ 80)"},
        ],
    },
    "Italy": {
        "display_name": "Italien",
        "military": 65,
        "stability": 60,
        "economy": 70,
        "diplomatic_influence": 65,
        "public_approval": 55,
        "ambition": "Wirtschaft stabilisieren, Migration kontrollieren, Einfluss im Mittelmeerraum stärken",
        "win_conditions": [
            {"metric": "economy", "op": ">=", "value": 80, "label": "Wirtschaft erholt (≥ 80)"},
            {"metric": "stability", "op": ">=", "value": 70, "label": "Politische Stabilisierung (≥ 70)"},
            {"metric": "public_approval", "op": ">=", "value": 65, "label": "Öffentliche Zustimmung gesichert (≥ 65)"},
            {"metric": "eu_cohesion", "op": ">=", "value": 60, "label": "EU-Unterstützung vorhanden (≥ 60)"},
        ],
    },
    "France": {
        "display_name": "Frankreich",
        "military": 85,
        "stability": 75,
        "economy": 80,
        "diplomatic_influence": 90,
        "public_approval": 65,
        "ambition": "Strategische Autonomie Europas, militärische Führungsrolle, Einfluss in Afrika sichern",
        "win_conditions": [
            {"metric": "military", "op": ">=", "value": 90, "label": "Militärische Schlagkraft (≥ 90)"},
            {"metric": "diplomatic_influence", "op": ">=", "value": 92, "label": "Globaler diplomatischer Einfluss (≥ 92)"},
            {"metric": "stability", "op": ">=", "value": 78, "label": "Ausreichende innenpolitische Stabilität (≥ 78)"},
            {"metric": "eu_cohesion", "op": ">=", "value": 75, "label": "EU handlungsfähig (≥ 75)"},
        ],
    },
    "Poland": {
        "display_name": "Polen",
        "military": 90,
        "stability": 70,
        "economy": 75,
        "diplomatic_influence": 70,
        "public_approval": 80,
        "ambition": "Abschreckung gegen Russland, Führungsrolle in Osteuropa, starke NATO-Anbindung",
        "win_conditions": [
            {"metric": "military", "op": ">=", "value": 95, "label": "Maximale militärische Abschreckung (≥ 95)"},
            {"metric": "public_approval", "op": ">=", "value": 85, "label": "Starker Rückhalt in der Bevölkerung (≥ 85)"},
            {"metric": "stability", "op": ">=", "value": 75, "label": "Innenpolitisch stabil (≥ 75)"},
            {"metric": "eu_cohesion", "op": ">=", "value": 70, "label": "EU ausreichend geschlossen (≥ 70)"},
        ],
    },
    "Hungary": {
        "display_name": "Ungarn",
        "military": 60,
        "stability": 80,
        "economy": 65,
        "diplomatic_influence": 55,
        "public_approval": 85,
        "ambition": "Nationale Souveränität bewahren, EU-Einfluss begrenzen, wirtschaftliche Vorteile sichern",
        "win_conditions": [
            {"metric": "public_approval", "op": ">=", "value": 80, "label": "Sehr hohe öffentliche Zustimmung (≥ 80)"},
            {"metric": "stability", "op": ">=", "value": 75, "label": "Starke innenpolitische Kontrolle (≥ 75)"},
            {"metric": "economy", "op": ">=", "value": 65, "label": "Ausreichende wirtschaftliche Lage (≥ 65)"},
            {"metric": "eu_cohesion", "op": "<=", "value": 45, "label": "Begrenzter EU-Zusammenhalt (≤ 45)"},
        ],
    },
}

EU_DEFAULT = {
    "cohesion": 75,
    "global_context": (
        "Russland droht mit Gaskürzungen und eskaliert den Krieg in der Ukraine weiter. "
        "USA drohen mit Übernahme Grönlands. "
        "China liebäugelt mit Invasion Taiwans."
    ),
}

# ----------------------------
# NEW: Außenmächte-Flavor (Crazy + Quote / Soundbite)
# ----------------------------

# Hinweis: Das sind "Stil-Rollen" fürs Spiel (fiktive Soundbites), keine echten Zitate.
EXTERNAL_LEADER_STYLE = {
    "USA": {
        "leader_label": "US-Präsident (Trump-ähnlicher Ton)",
        "style": "kurz, markig, superlativ-lastig, deal-orientiert, 'wir gewinnen' vibe",
    },
    "Russia": {
        "leader_label": "Kremlchef (Putin-ähnlicher Ton)",
        "style": "kalt, kontrolliert, drohend mit Unterton, spricht von 'roten Linien' und 'Souveränität'",
    },
    "China": {
        "leader_label": "Staatspräsident (Xi-ähnlicher Ton)",
        "style": "bürokratisch-höflich, aber unmissverständlich, spricht von 'Stabilität', 'Harmonie', 'Souveränität'",
    },
}

# Baseline-Tendenzen (du kannst die ranges jederzeit tweaken)
EXTERNAL_CRAZY_BASELINE_RANGES = {
    "USA": (30, 90),
    "Russia": (20, 85),
    "China": (15, 80),
}
