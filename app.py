# app.py
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

import streamlit as st
from dotenv import load_dotenv

from countries import COUNTRY_DEFS, EU_DEFAULT

from db import (
    get_conn,
    ensure_schema,
    seed_countries_if_missing,
    reset_all_countries,
    load_country_metrics,
    load_all_country_metrics,
    load_recent_history,
    get_eu_state,
    set_eu_state,
    get_game_meta,
    set_game_meta,
    clear_round_data,
    upsert_round_actions,
    get_round_actions,
    get_round_action_impacts,
    lock_choice,
    get_locks,
    all_locked,
    apply_country_deltas,
    insert_turn_history,
    get_recent_round_summaries,
    upsert_round_summary,
    clear_all_round_summaries,
    # external
    clear_external_events,
    upsert_external_event,
    get_external_events,
    # auth
    create_user,
    verify_user,
    list_users,
    delete_user,
)

from ai_round import generate_actions_for_country, resolve_round_all_countries, generate_round_summary
from ai_external import generate_external_moves

# Optional: win.py (falls vorhanden)
try:
    from win import evaluate_all_countries, evaluate_country_win_conditions
except Exception:
    evaluate_all_countries = None
    evaluate_country_win_conditions = None


def load_env():
    env_path = Path(__file__).with_name(".env")
    load_dotenv(env_path)


def summarize_recent_actions(rows) -> str:
    if not rows:
        return "Keine."
    items = []
    for r in rows[:6]:
        items.append(f"R{r[0]}: {r[1]}")
    return " | ".join(items)


def pressure_badge(label: str, value: int) -> str:
    if value >= 70:
        icon = "🔴"
        lvl = "hoch"
    elif value >= 40:
        icon = "🟡"
        lvl = "mittel"
    else:
        icon = "🟢"
        lvl = "niedrig"
    return f"{icon} {label}: {lvl}"


def render_metrics(metrics: Dict[str, Any]):
    c1, c2, c3 = st.columns(3)
    c1.metric("Wirtschaft", metrics["economy"])
    c2.metric("Stabilität", metrics["stability"])
    c3.metric("Militär", metrics["military"])
    c4, c5 = st.columns(2)
    c4.metric("Diplomatie", metrics["diplomatic_influence"])
    c5.metric("Öffentliche Zustimmung", metrics["public_approval"])
    with st.expander("Ambition"):
        st.write(metrics["ambition"])


def format_external_events(events: List[Dict[str, Any]]) -> str:
    if not events:
        return "Keine."
    lines = []
    for e in events:
        lines.append(f"- {e.get('actor')}: {e.get('headline')}")
    return "\n".join(lines)


def _arrow(delta: int) -> str:
    if delta >= 3:
        return "⬆️"
    if delta <= -3:
        return "⬇️"
    return "➖"


def impact_preview_text(folgen: Dict[str, Any]) -> str:
    land = (folgen or {}).get("land", {}) or {}
    eu = (folgen or {}).get("eu", {}) or {}

    dm = int(land.get("militär", 0))
    ds = int(land.get("stabilität", 0))
    de = int(land.get("wirtschaft", 0))
    dd = int(land.get("diplomatie", 0))
    dp = int(land.get("öffentliche_zustimmung", 0))
    dcoh = int(eu.get("kohäsion", 0))

    max_abs = max(abs(dm), abs(ds), abs(de), abs(dd), abs(dp), abs(dcoh))
    if max_abs >= 9:
        risk = "Risiko: 🔥 hoch"
    elif max_abs >= 6:
        risk = "Risiko: ⚠️ mittel"
    else:
        risk = "Risiko: ✅ niedrig"

    return (
        f"Mil {_arrow(dm)}  Sta {_arrow(ds)}  Wir {_arrow(de)}  Dip {_arrow(dd)}  Zust {_arrow(dp)}  "
        f"EU {_arrow(dcoh)}  •  {risk}"
    )


def build_action_prompt(
    *,
    country_display: str,
    metrics: Dict[str, Any],
    eu_state: Dict[str, Any],
    external_events: List[Dict[str, Any]],
    recent_actions_summary: str,
) -> str:
    external_str = format_external_events(external_events)

    return f"""
Du bist ein Spielleiter in einem EU-Geopolitik-Spiel.
Erzeuge drei öffentliche Aktionsoptionen für {country_display}: aggressiv, moderate, passiv.

Kontext:
- {country_display} Metriken: Militär={metrics["military"]}, Stabilität={metrics["stability"]}, Wirtschaft={metrics["economy"]}, Diplomatie={metrics["diplomatic_influence"]}, Öffentliche Zustimmung={metrics["public_approval"]}.
- Ambition: {metrics["ambition"]}.

EU-/Weltlage:
- EU-Kohäsion={eu_state["cohesion"]}%
- Threat Level={eu_state["threat_level"]}/100, Frontline Pressure={eu_state["frontline_pressure"]}/100
- Energy={eu_state["energy_pressure"]}/100, Migration={eu_state["migration_pressure"]}/100
- Disinfo={eu_state["disinfo_pressure"]}/100, TradeWar={eu_state["trade_war_pressure"]}/100
- Globaler Kontext: {eu_state["global_context"]}

Außenmächte-Moves dieser Runde:
{external_str}

Letzte Aktionen (für Variation, nicht wiederholen):
{recent_actions_summary}

Format:
Gib NUR gültiges JSON zurück (kein Markdown, keine Erklärungen).
Schema (genau so):
{{
  "aggressiv": {{
    "aktion": "...",
    "folgen": {{
      "land": {{"militär": 0, "stabilität": 0, "wirtschaft": 0, "diplomatie": 0, "öffentliche_zustimmung": 0}},
      "eu": {{"kohäsion": 0}},
      "global_context": "kurzer Satz zur Reaktion"
    }}
  }},
  "moderate": {{ ... }},
  "passiv": {{ ... }}
}}

Regeln:
- Folgen sind kleine, realistische Ganzzahlen (z.B. -12 bis +12).
- global_context ist ein kurzer Satz (max. 1 Zeile).
- Die drei Optionen sollen sich klar unterscheiden (Risiko/Ertrag).
- Baue öfter Sicherheitsdruck, innenpolitische Gegenreaktionen und diplomatische Deals ein.
""".strip()


def apply_external_modifiers_to_eu(eu_before: Dict[str, Any], moves_obj: Dict[str, Any]) -> Dict[str, Any]:
    eu = dict(eu_before)
    moves = moves_obj.get("moves", [])

    d_coh = d_threat = d_front = d_energy = d_migr = d_disinfo = d_trade = 0
    for m in moves:
        mods = m.get("modifiers", {}) or {}
        d_coh += int(mods.get("eu_cohesion_delta", 0))
        d_threat += int(mods.get("threat_delta", 0))
        d_front += int(mods.get("frontline_delta", 0))
        d_energy += int(mods.get("energy_delta", 0))
        d_migr += int(mods.get("migration_delta", 0))
        d_disinfo += int(mods.get("disinfo_delta", 0))
        d_trade += int(mods.get("trade_war_delta", 0))

    eu["cohesion"] = eu["cohesion"] + d_coh
    eu["threat_level"] = eu["threat_level"] + d_threat
    eu["frontline_pressure"] = eu["frontline_pressure"] + d_front
    eu["energy_pressure"] = eu["energy_pressure"] + d_energy
    eu["migration_pressure"] = eu["migration_pressure"] + d_migr
    eu["disinfo_pressure"] = eu["disinfo_pressure"] + d_disinfo
    eu["trade_war_pressure"] = eu["trade_war_pressure"] + d_trade

    if moves_obj.get("global_context"):
        eu["global_context"] = str(moves_obj["global_context"])

    return eu


def decay_pressures(eu: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(eu)
    out["threat_level"] = out["threat_level"] - 2
    out["frontline_pressure"] = out["frontline_pressure"] - 2
    out["energy_pressure"] = out["energy_pressure"] - 3
    out["migration_pressure"] = out["migration_pressure"] - 3
    out["disinfo_pressure"] = out["disinfo_pressure"] - 3
    out["trade_war_pressure"] = out["trade_war_pressure"] - 3
    return out


def render_player_view(
    *,
    conn,
    round_no: int,
    phase: str,
    eu: Dict[str, Any],
    countries_display: Dict[str, str],
    my_country: str,
    is_lock_disabled: bool,
):
    actions_texts = get_round_actions(conn, round_no)
    action_impacts = get_round_action_impacts(conn, round_no)
    locks_now = get_locks(conn, round_no)

    st.subheader(f"🏳️ Land: {countries_display[my_country]}")
    my_metrics = load_country_metrics(conn, my_country)
    if not my_metrics:
        st.error("Land konnte nicht geladen werden.")
        return

    render_metrics(my_metrics)

    if evaluate_country_win_conditions is not None:
        eu_now = get_eu_state(conn)
        is_winner, cond_results = evaluate_country_win_conditions(
            my_country,
            country_metrics=my_metrics,
            eu_state=eu_now,
            country_defs=COUNTRY_DEFS,
        )
        st.write("---")
        st.subheader("🏁 Siegfortschritt")
        if not cond_results:
            st.warning("Für dieses Land sind noch keine Siegbedingungen definiert (countries.py: win_conditions).")
        else:
            if is_winner:
                st.success("✅ Siegbedingungen erfüllt! Du hast gewonnen.")
            else:
                st.info("Noch nicht gewonnen — Bedingungen:")
            for r in cond_results:
                st.write(("✅ " if r.ok else "❌ ") + f"{r.label} (aktuell: {r.current})")

    st.write("---")

    if phase != "actions_published":
        st.info("Optionen sind noch nicht veröffentlicht. Warte auf den Game Master.")
    else:
        a = actions_texts.get(my_country, {})
        if not a or len(a) < 3:
            st.warning("Optionen fehlen noch (GM muss Aktionen generieren und veröffentlichen).")
        else:
            st.subheader("Öffentliche Aktion wählen")

            if my_country in locks_now:
                st.success("✅ Eingelockt. (Welche Variante bleibt für andere verborgen.)")
            else:
                st.warning("⏳ Noch nicht eingelockt.")

            options = {
                "aggressiv": a["aggressiv"],
                "moderate": a["moderate"],
                "passiv": a["passiv"],
            }
            labels = [options["aggressiv"], options["moderate"], options["passiv"]]
            choice_label = st.radio("Option:", labels, index=1)
            chosen_variant = next(k for k, v in options.items() if v == choice_label)

            folgen = (action_impacts.get(my_country, {}) or {}).get(chosen_variant, {}) or {}
            if folgen:
                st.caption("**Voraussichtliche Wirkung:** " + impact_preview_text(folgen))
            else:
                st.caption("Voraussichtliche Wirkung: (noch keine Daten / alte Runde ohne Impact gespeichert)")

            with st.expander("Alle Wirkungen vergleichen", expanded=False):
                for v in ("aggressiv", "moderate", "passiv"):
                    folgen_v = (action_impacts.get(my_country, {}) or {}).get(v, {}) or {}
                    st.write(f"**{v.capitalize()}**")
                    st.write(options[v])
                    if folgen_v:
                        st.caption(impact_preview_text(folgen_v))
                    else:
                        st.caption("(keine Impact-Daten)")
                    st.write("---")

            if st.button("✅ Auswahl einlocken", use_container_width=True, disabled=is_lock_disabled):
                lock_choice(conn, round_no, my_country, chosen_variant)
                st.rerun()

            if is_lock_disabled:
                st.caption("GM-Hinweis: In der simulierten Spieleransicht ist Einlocken deaktiviert.")

    with st.expander("Turn-History (Mein Land)"):
        rows = load_recent_history(conn, my_country, limit=12)
        if not rows:
            st.write("Noch keine Runden gespielt.")
        else:
            for r in rows:
                st.markdown(
                    f"""
**Runde {r[0]}**  
Aktion: {r[1]}  
Δ Militär {r[2]}, Δ Stabilität {r[3]}, Δ Wirtschaft {r[4]}, Δ Diplomatie {r[5]}, Δ Zustimmung {r[6]}  
Kontext: {r[7]}
"""
                )


# ----------------------------
# App start
# ----------------------------
st.set_page_config(page_title="eugenia", layout="wide")
st.title("eugenia - save europe, save yourself")

load_env()

api_key = (os.getenv("MISTRAL_API_KEY") or "").strip()
if not api_key:
    st.error("MISTRAL_API_KEY fehlt. Lege eine .env neben app.py an: MISTRAL_API_KEY=... ")
    st.stop()

gm_pin = (os.getenv("GM_PIN") or "").strip()

conn = get_conn()
ensure_schema(conn)
seed_countries_if_missing(conn, COUNTRY_DEFS)

countries = list(COUNTRY_DEFS.keys())
countries_display = {k: COUNTRY_DEFS[k]["display_name"] for k in countries}

# ----------------------------
# Auth gate
# ----------------------------
if "auth" not in st.session_state:
    st.session_state.auth = None

if st.session_state.auth is None:
    st.subheader("🔐 Login")
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Passwort", type="password")
        submitted = st.form_submit_button("Einloggen")

    if submitted:
        user = verify_user(conn, username=username, password=password)
        if not user:
            st.error("Login fehlgeschlagen.")
        else:
            st.session_state.auth = user
            st.rerun()

    st.info("Bitte einloggen. (User werden vom Game Master erstellt.)")
    conn.close()
    st.stop()

auth = st.session_state.auth
is_gm = auth["role"] == "gm"
assigned_country = auth.get("country")

st.sidebar.write(f"👤 **{auth['username']}**")
st.sidebar.write(f"Rolle: **{auth['role']}**")
if assigned_country:
    st.sidebar.write(f"Land: **{assigned_country}**")

if st.sidebar.button("🚪 Logout"):
    st.session_state.auth = None
    st.rerun()

if is_gm and gm_pin:
    entered = st.sidebar.text_input("GM PIN", type="password")
    if entered != gm_pin:
        st.sidebar.warning("PIN erforderlich.")
        conn.close()
        st.stop()

# ----------------------------
# GM: Spieleransicht simulieren
# ----------------------------
if "gm_view_enabled" not in st.session_state:
    st.session_state.gm_view_enabled = False
if "gm_view_country" not in st.session_state:
    st.session_state.gm_view_country = None

if is_gm:
    with st.sidebar.expander("🕵️ Spieleransicht simulieren", expanded=False):
        st.session_state.gm_view_enabled = st.checkbox(
            "Spieleransicht aktivieren",
            value=st.session_state.gm_view_enabled,
        )
        if st.session_state.gm_view_enabled:
            opts = list(COUNTRY_DEFS.keys())
            if st.session_state.gm_view_country not in opts:
                st.session_state.gm_view_country = opts[0]
            st.session_state.gm_view_country = st.selectbox(
                "Als Spieler ansehen (Land)",
                options=opts,
                index=opts.index(st.session_state.gm_view_country),
            )
        else:
            st.session_state.gm_view_country = None

# Effective country for player UI
effective_country = None
is_simulating_player_view = False
if is_gm and st.session_state.get("gm_view_enabled") and st.session_state.get("gm_view_country"):
    effective_country = st.session_state.gm_view_country
    is_simulating_player_view = True
elif not is_gm:
    effective_country = assigned_country

if not is_gm and not effective_country:
    st.error("Kein Land zugewiesen. GM muss dir ein Land zuweisen.")
    conn.close()
    st.stop()

# ----------------------------
# Sidebar: Quick guide
# ----------------------------
with st.sidebar.expander("📘 Werte erklärt (Kurz)", expanded=False):
    st.markdown("""
**Militär**: Abschreckung/Verteidigung. Hilft bei hohem Threat/Frontline, kann innenpolitisch polarisieren.  
**Stabilität**: Regierungsfähigkeit/Protestresistenz. Niedrig → Krisenanfälligkeit.  
**Wirtschaft**: Wachstum/Inflation/Haushalt. Niedrig → Zustimmung fällt schneller.  
**Diplomatie**: Fähigkeit zu Deals/Koalitionen/Sanktionen. Hoch → bessere Kompromisse.  
**Öffentliche Zustimmung**: Rückendeckung. Niedrig → riskante Entscheidungen “kosten” stärker.

**Druckwerte (EU):**  
**Threat/Frontline** = Kriegsrisiko & Ostflanken-Spannung.  
**Energy/Migration/Disinfo/TradeWar** erhöhen innenpolitischen Stress & Spaltung.
""")

# ----------------------------
# DB states
# ----------------------------
meta = get_game_meta(conn)
round_no = meta["round"]
phase = meta["phase"]  # setup -> external_generated -> actions_generated -> actions_published

eu = get_eu_state(conn)
if not eu["global_context"]:
    set_eu_state(
        conn,
        cohesion=EU_DEFAULT.get("cohesion", eu["cohesion"]),
        global_context=EU_DEFAULT.get("global_context", ""),
        threat_level=eu["threat_level"],
        frontline_pressure=eu["frontline_pressure"],
        energy_pressure=eu["energy_pressure"],
        migration_pressure=eu["migration_pressure"],
        disinfo_pressure=eu["disinfo_pressure"],
        trade_war_pressure=eu["trade_war_pressure"],
    )
    eu = get_eu_state(conn)

# ----------------------------
# GM: User management
# ----------------------------
if is_gm:
    with st.sidebar.expander("👥 User verwalten", expanded=False):
        st.caption("User in SQLite. Passwörter: PBKDF2 + Salt + Pepper (.env).")
        with st.form("create_user_form"):
            new_u = st.text_input("Neuer Username")
            new_p = st.text_input("Neues Passwort", type="password")
            new_role = st.selectbox("Rolle", ["player", "gm"], index=0)
            new_country = None
            if new_role == "player":
                new_country = st.selectbox("Land zuweisen", list(COUNTRY_DEFS.keys()))
            submitted = st.form_submit_button("User anlegen/aktualisieren")
        if submitted:
            try:
                create_user(conn, username=new_u, password=new_p, role=new_role, country=new_country)
                st.success("User gespeichert.")
                st.rerun()
            except Exception as e:
                st.error(f"Fehler: {e}")

        st.write("---")
        st.write("**Bestehende User**")
        for u in list_users(conn):
            st.write(f"- {u['username']} ({u['role']}) {('→ ' + u['country']) if u['country'] else ''}")

        del_u = st.text_input("Username löschen")
        if st.button("User löschen"):
            delete_user(conn, del_u)
            st.success("Gelöscht.")
            st.rerun()

# ----------------------------
# Sidebar: reset (GM only)
# ----------------------------
if is_gm:
    st.sidebar.write("---")
    st.sidebar.subheader("Reset")
    if st.sidebar.button("💣 Reset alle"):
        reset_all_countries(conn, COUNTRY_DEFS)
        clear_round_data(conn, round_no)
        clear_all_round_summaries(conn)
        clear_external_events(conn, round_no)
        set_eu_state(
            conn,
            cohesion=EU_DEFAULT.get("cohesion", 75),
            global_context=EU_DEFAULT.get("global_context", ""),
            threat_level=35,
            frontline_pressure=30,
            energy_pressure=25,
            migration_pressure=25,
            disinfo_pressure=25,
            trade_war_pressure=25,
        )
        set_game_meta(conn, 1, "setup")
        st.rerun()

# ----------------------------
# Layout
# ----------------------------
left, right = st.columns([0.62, 0.38], gap="large")

# ----------------------------
# RIGHT: Status + GM controls (GM) / Status (Player)
# ----------------------------
with right:
    st.subheader("📊 Rundenstatus")
    st.write(f"**Runde:** {round_no}  |  **Phase:** {phase}")

    locks = get_locks(conn, round_no)
    st.write("**Lock-Status (diese Runde)**")
    for c in countries:
        name = countries_display[c]
        if c in locks:
            # GM sieht Variante, Spieler nicht
            if is_gm:
                st.success(f"{name}: ✅ eingelockt ({locks[c]})")
            else:
                st.success(f"{name}: ✅ eingelockt")
        else:
            st.warning(f"{name}: ⏳ nicht eingelockt")

    st.write("---")
    st.write("**EU & Druckwerte**")
    st.write(f"Kohäsion: **{eu['cohesion']}%**")
    c1, c2 = st.columns(2)
    c1.write(pressure_badge("Threat", eu["threat_level"]))
    c1.write(pressure_badge("Frontline", eu["frontline_pressure"]))
    c2.write(pressure_badge("Energy", eu["energy_pressure"]))
    c2.write(pressure_badge("Migration", eu["migration_pressure"]))
    with st.expander("Mehr Details (Druckwerte)"):
        st.write(f"Disinfo: {eu['disinfo_pressure']} / 100")
        st.write(f"TradeWar: {eu['trade_war_pressure']} / 100")
    st.caption(eu["global_context"])

    st.write("---")
    ext_events = get_external_events(conn, round_no)
    with st.expander("🌐 Außenmächte-Moves (öffentlich)"):
        if not ext_events:
            st.write("Noch keine Moves generiert.")
        else:
            for e in ext_events:
                st.markdown(f"**{e['actor']}**: {e['headline']}")

    st.write("---")
    with st.expander("🧠 Letzte Runden (Memory)"):
        mem = get_recent_round_summaries(conn, limit=5)
        if not mem:
            st.write("Noch keine Runden-Summaries vorhanden.")
        else:
            for r, s in reversed(mem):
                st.markdown(f"**Runde {r}**\n\n{s}")

    st.write("---")

    if evaluate_all_countries is not None:
        all_metrics_now = load_all_country_metrics(conn, countries)
        eu_now = get_eu_state(conn)
        win_eval = evaluate_all_countries(
            all_country_metrics=all_metrics_now,
            eu_state=eu_now,
            country_defs=COUNTRY_DEFS,
        )
        winners = [countries_display[c] for c in countries if win_eval.get(c, {}).get("is_winner")]
        if winners:
            st.success("🏆 Gewinner erreicht: " + ", ".join(winners))
        else:
            st.caption("Noch kein Land hat die Siegbedingungen vollständig erfüllt.")
        if is_gm:
            with st.expander("📈 Siegfortschritt (GM Detail)"):
                for c in countries:
                    st.markdown(f"### {countries_display[c]}")
                    res = win_eval[c]["results"]
                    if not res:
                        st.caption("Keine Bedingungen definiert.")
                        continue
                    for r in res:
                        st.write(("✅ " if r.ok else "❌ ") + f"{r.label} (aktuell: {r.current})")
        st.write("---")

    # GM controls
    if is_gm:
        st.subheader("🎛️ Game Master Steuerung (sequenziell)")

        actions_in_db = get_round_actions(conn, round_no)
        have_all_actions = all((c in actions_in_db and len(actions_in_db[c]) == 3) for c in countries)
        have_all_locks = all_locked(conn, round_no, countries)
        have_external = len(get_external_events(conn, round_no)) == 3

        # 1) External moves
        external_disabled = (phase == "actions_published")
        if st.button("⚠️ Außenmächte-Züge generieren (USA/China/Russland)", disabled=external_disabled, use_container_width=True):
            with st.spinner("Generiere Außenmächte-Moves..."):
                recent_summaries = get_recent_round_summaries(conn, limit=3)
                eu_before = get_eu_state(conn)

                moves_obj = generate_external_moves(
                    api_key=api_key,
                    model="mistral-small",
                    round_no=round_no,
                    eu_state=eu_before,
                    recent_round_summaries=recent_summaries,
                    temperature=0.8,
                    top_p=0.95,
                    max_tokens=900,
                )

                clear_external_events(conn, round_no)
                for m in moves_obj["moves"]:
                    upsert_external_event(
                        conn,
                        round_no,
                        actor=m["actor"],
                        headline=m["headline"],
                        modifiers=m.get("modifiers", {}),
                    )

                eu_after = apply_external_modifiers_to_eu(eu_before, moves_obj)
                set_eu_state(
                    conn,
                    cohesion=eu_after["cohesion"],
                    global_context=eu_after["global_context"],
                    threat_level=eu_after["threat_level"],
                    frontline_pressure=eu_after["frontline_pressure"],
                    energy_pressure=eu_after["energy_pressure"],
                    migration_pressure=eu_after["migration_pressure"],
                    disinfo_pressure=eu_after["disinfo_pressure"],
                    trade_war_pressure=eu_after["trade_war_pressure"],
                )

                set_game_meta(conn, round_no, "external_generated")
            st.rerun()

        # 2) Generate actions for all (only after external generated)
        gen_disabled = not (phase in ("external_generated", "actions_generated") and have_external) or (phase == "actions_published")
        if st.button("⚙️ Aktionen für alle generieren", disabled=gen_disabled, use_container_width=True):
            with st.spinner("Generiere Aktionen für alle Länder..."):
                eu_now = get_eu_state(conn)
                ext_now = get_external_events(conn, round_no)

                all_metrics = load_all_country_metrics(conn, countries)
                for c in countries:
                    m = all_metrics[c]
                    recent = load_recent_history(conn, c, limit=12)

                    prompt = build_action_prompt(
                        country_display=countries_display[c],
                        metrics=m,
                        eu_state=eu_now,
                        external_events=ext_now,
                        recent_actions_summary=summarize_recent_actions(recent),
                    )
                    actions_obj, _raw_first, _used_repair = generate_actions_for_country(
                        api_key=api_key,
                        model="mistral-small",
                        prompt=prompt,
                        temperature=0.9,
                        top_p=0.95,
                        max_tokens=900,
                    )
                    upsert_round_actions(conn, round_no, c, actions_obj)

                set_game_meta(conn, round_no, "actions_generated")
            st.rerun()

        actions_in_db = get_round_actions(conn, round_no)
        if actions_in_db:
            with st.expander("👀 Vorschau: Generierte Länderaktionen (GM)"):
                for c in countries:
                    st.markdown(f"### {countries_display[c]}")
                    a = actions_in_db.get(c, {})
                    if not a:
                        st.caption("Noch keine Aktionen generiert.")
                        continue
                    st.write(f"**Aggressiv:** {a.get('aggressiv','')}")
                    st.write(f"**Moderate:** {a.get('moderate','')}")
                    st.write(f"**Passiv:** {a.get('passiv','')}")
                    st.write("---")

        # 3) Publish
        publish_disabled = not (phase == "actions_generated" and have_all_actions and have_external)
        if st.button("🚦 Runde starten (Optionen veröffentlichen)", disabled=publish_disabled, use_container_width=True):
            set_game_meta(conn, round_no, "actions_published")
            st.rerun()

        # 4) Resolve
        resolve_disabled = not (phase == "actions_published" and have_all_locks)
        if st.button("🧮 Ergebnis der Runde kalkulieren", disabled=resolve_disabled, use_container_width=True):
            with st.spinner("KI kalkuliert Gesamtergebnis der Runde..."):
                recent_summaries = get_recent_round_summaries(conn, limit=3)
                eu_before = get_eu_state(conn)
                ext_now = get_external_events(conn, round_no)

                actions_texts = get_round_actions(conn, round_no)
                locks_now = get_locks(conn, round_no)
                all_metrics = load_all_country_metrics(conn, countries)

                chosen_actions_lines = []
                for c in countries:
                    v = locks_now[c]
                    chosen_actions_lines.append(f"- {countries_display[c]} ({c}): {v} -> {actions_texts[c][v]}")
                chosen_actions_str = "\n".join(chosen_actions_lines)

                result = resolve_round_all_countries(
                    api_key=api_key,
                    model="mistral-small",
                    round_no=round_no,
                    eu_state=eu_before,
                    countries_metrics=all_metrics,
                    countries_display=countries_display,
                    actions_texts=actions_texts,
                    locked_choices=locks_now,
                    recent_round_summaries=recent_summaries,
                    external_events=ext_now,
                    temperature=0.6,
                    top_p=0.95,
                    max_tokens=1700,
                )

                eu_after = dict(eu_before)
                eu_after["cohesion"] = eu_before["cohesion"] + int(result["eu"].get("kohäsion_delta", 0))
                eu_after["global_context"] = str(result["eu"].get("global_context", eu_before["global_context"]))
                eu_after = decay_pressures(eu_after)

                set_eu_state(
                    conn,
                    cohesion=eu_after["cohesion"],
                    global_context=eu_after["global_context"],
                    threat_level=eu_after["threat_level"],
                    frontline_pressure=eu_after["frontline_pressure"],
                    energy_pressure=eu_after["energy_pressure"],
                    migration_pressure=eu_after["migration_pressure"],
                    disinfo_pressure=eu_after["disinfo_pressure"],
                    trade_war_pressure=eu_after["trade_war_pressure"],
                )

                for c in countries:
                    d = result["länder"][c] or {}
                    apply_country_deltas(conn, c, d)

                    chosen_variant = locks_now[c]
                    chosen_action_text = actions_texts[c][chosen_variant]
                    insert_turn_history(
                        conn,
                        country=c,
                        round_no=round_no,
                        action_public=chosen_action_text,
                        global_context=eu_after["global_context"],
                        deltas=d,
                    )

                eu_after_fresh = get_eu_state(conn)
                summary_text = generate_round_summary(
                    api_key=api_key,
                    model="mistral-small",
                    round_no=round_no,
                    memory_in=recent_summaries,
                    eu_before=eu_before,
                    eu_after=eu_after_fresh,
                    external_events=ext_now,
                    chosen_actions_str=chosen_actions_str,
                    result_obj=result,
                    temperature=0.4,
                    top_p=0.95,
                    max_tokens=520,
                )
                upsert_round_summary(conn, round_no, summary_text)

                clear_round_data(conn, round_no)
                clear_external_events(conn, round_no)
                set_game_meta(conn, round_no + 1, "setup")

            st.success("Runde aufgelöst, Werte gesetzt, nächste Runde gestartet.")
            st.rerun()

        st.caption("Flow: Außenmächte → Aktionen generieren → Veröffentlichen → Lock → Resolve")

# ----------------------------
# LEFT: Player UI (Players always; GM only if simulating)
# ----------------------------
with left:
    if is_gm:
        st.subheader("🎮 Spieleransicht (GM Simulation)")
        if not is_simulating_player_view or not effective_country:
            st.info("Aktiviere in der Sidebar 'Spieleransicht simulieren' und wähle ein Land.")
        else:
            render_player_view(
                conn=conn,
                round_no=round_no,
                phase=phase,
                eu=eu,
                countries_display=countries_display,
                my_country=effective_country,
                is_lock_disabled=False,  # GM can lock in simulation
            )
    else:
        st.subheader("🎮 Spielerbereich")
        render_player_view(
            conn=conn,
            round_no=round_no,
            phase=phase,
            eu=eu,
            countries_display=countries_display,
            my_country=effective_country,
            is_lock_disabled=False,
        )

conn.close()
