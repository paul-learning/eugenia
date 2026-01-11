# app.py
import os
from pathlib import Path
from typing import Dict, Any

import streamlit as st
from dotenv import load_dotenv

from countries import COUNTRY_DEFS, EU_DEFAULT
from db import (
    get_conn,
    ensure_schema,
    seed_countries_if_missing,
    reset_all_countries,
    reset_country_to_defaults,
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
    lock_choice,
    get_locks,
    all_locked,
    apply_country_deltas,
    insert_turn_history,
    get_recent_round_summaries,
    upsert_round_summary,
    clear_all_round_summaries,
)
from ai_round import generate_actions_for_country, resolve_round_all_countries, generate_round_summary

# Optional: wenn du win.py bereits nutzt
try:
    from win import evaluate_all_countries, evaluate_country_win_conditions
except Exception:
    evaluate_all_countries = None
    evaluate_country_win_conditions = None


def load_env():
    env_path = Path(__file__).with_name(".env")
    load_dotenv(env_path)


def build_action_prompt(
    *,
    country_display: str,
    metrics: Dict[str, Any],
    eu_cohesion: int,
    global_context: str,
    recent_actions_summary: str,
) -> str:
    return f"""
Du bist ein Spielleiter in einem EU-Geopolitik-Spiel.
Erzeuge drei öffentliche Aktionsoptionen für {country_display}: aggressiv, moderate, passiv.

Kontext:
- {country_display} Metriken: Militär={metrics["military"]}, Stabilität={metrics["stability"]}, Wirtschaft={metrics["economy"]}, Diplomatie={metrics["diplomatic_influence"]}, Öffentliche Zustimmung={metrics["public_approval"]}.
- Ambition: {metrics["ambition"]}.
- EU: Kohäsion={eu_cohesion}%.
- Globaler Kontext: {global_context}
- Letzte Aktionen (für Variation, nicht wiederholen): {recent_actions_summary}

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
- Folgen sind kleine, realistische Ganzzahlen (z.B. -10 bis +10).
- global_context ist ein kurzer Satz (max. 1 Zeile).
- Die drei Optionen sollen sich klar unterscheiden (Risiko/Ertrag).
- Vermeide wiederkehrende Standardfloskeln; sei spezifisch zum Land und Kontext.
""".strip()


def summarize_recent_actions(rows) -> str:
    if not rows:
        return "Keine."
    items = []
    for r in rows[:6]:
        items.append(f"R{r[0]}: {r[1]}")
    return " | ".join(items)


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


# ----------------------------
# App start
# ----------------------------
st.set_page_config(page_title="EU Geopolitik (GM Rundensteuerung)", layout="wide")
st.title("EU Geopolitik-Prototyp — Game Master Flow")

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

# DB states
meta = get_game_meta(conn)
round_no = meta["round"]
phase = meta["phase"]  # setup | actions_generated | actions_published

eu = get_eu_state(conn)
if not eu["global_context"]:
    set_eu_state(conn, eu["cohesion"], EU_DEFAULT["global_context"])
    eu = get_eu_state(conn)

# ----------------------------
# Sidebar: role + status + reset
# ----------------------------
st.sidebar.header("Rolle")
role = st.sidebar.selectbox("Ansicht", ["Spieler", "Game Master"], index=0)
is_gm = (role == "Game Master")

if is_gm and gm_pin:
    entered = st.sidebar.text_input("GM PIN", type="password")
    if entered != gm_pin:
        st.sidebar.warning("PIN erforderlich.")
        st.stop()

st.sidebar.write("---")
st.sidebar.write(f"**Runde:** {round_no}")
st.sidebar.write(f"**Phase:** {phase}")
st.sidebar.write(f"**EU-Kohäsion:** {eu['cohesion']}%")
st.sidebar.caption(eu["global_context"])

st.sidebar.write("---")
st.sidebar.subheader("Reset")
colA, colB = st.sidebar.columns(2)
if colB.button("💣 Reset alle"):
    reset_all_countries(conn, COUNTRY_DEFS)
    clear_round_data(conn, round_no)
    clear_all_round_summaries(conn)
    set_eu_state(conn, EU_DEFAULT["cohesion"], EU_DEFAULT["global_context"])
    set_game_meta(conn, 1, "setup")
    st.rerun()

# -------------- Layout --------------
left, right = st.columns([0.62, 0.38], gap="large")

# ----------------------------
# RIGHT: Status/Overview (GM + Spieler)
# ----------------------------
with right:
    st.subheader("📊 Rundenstatus")

    locks = get_locks(conn, round_no)

    st.write("**Lock-Status (diese Runde)**")
    for c in countries:
        name = countries_display[c]
        if c in locks:
            if is_gm:
                st.success(f"{name}: ✅ eingelockt ({locks[c]})")
            else:
                st.success(f"{name}: ✅ eingelockt")
        else:
            st.warning(f"{name}: ⏳ nicht eingelockt")

    st.write("---")
    st.write("**EU**")
    st.write(f"Kohäsion: **{eu['cohesion']}%**")
    st.caption(eu["global_context"])

    st.write("---")

    # Memory anzeigen (für alle ok, enthält keine geheimen Varianten)
    with st.expander("🧠 Letzte Runden (Memory)"):
        mem = get_recent_round_summaries(conn, limit=5)
        if not mem:
            st.write("Noch keine Runden-Summaries vorhanden.")
        else:
            for r, s in reversed(mem):
                st.markdown(f"**Runde {r}**\n\n{s}")

    st.write("---")

    # Optional: Winners overview (falls win.py da ist)
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

        # GM Detail
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

    # GM controls below
    if not is_gm:
        st.info("Für Rundensteuerung: in der Sidebar zu **Game Master** wechseln.")
    else:
        st.subheader("🎛️ Game Master Steuerung")

        actions_in_db = get_round_actions(conn, round_no)
        have_all_actions = all((c in actions_in_db and len(actions_in_db[c]) == 3) for c in countries)
        have_all_locks = all_locked(conn, round_no, countries)

        # Generate: erlaubt bis veröffentlicht (setup + actions_generated)
        gen_disabled = (phase == "actions_published")
        if st.button("⚙️ Aktionen für alle generieren", disabled=gen_disabled, use_container_width=True):
            with st.spinner("Generiere Aktionen für alle Länder..."):
                all_metrics = load_all_country_metrics(conn, countries)
                for c in countries:
                    m = all_metrics[c]
                    recent = load_recent_history(conn, c, limit=12)
                    prompt = build_action_prompt(
                        country_display=countries_display[c],
                        metrics=m,
                        eu_cohesion=eu["cohesion"],
                        global_context=eu["global_context"],
                        recent_actions_summary=summarize_recent_actions(recent),
                    )
                    actions_obj, raw_first, used_repair = generate_actions_for_country(
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

        # GM preview
        actions_in_db = get_round_actions(conn, round_no)
        if actions_in_db:
            with st.expander("👀 Vorschau: Generierte Aktionen (alle Länder)"):
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

        # Publish: nur wenn actions_generated + alles da
        publish_disabled = not (phase == "actions_generated" and have_all_actions)
        if st.button("🚦 Runde starten (Optionen veröffentlichen)", disabled=publish_disabled, use_container_width=True):
            set_game_meta(conn, round_no, "actions_published")
            st.rerun()

        # Resolve: nur wenn published + alle gelockt
        resolve_disabled = not (phase == "actions_published" and have_all_locks)
        if st.button("🧮 Ergebnis der Runde kalkulieren", disabled=resolve_disabled, use_container_width=True):
            with st.spinner("KI kalkuliert Gesamtergebnis der Runde..."):
                # Stufe 1 Memory: letzte 3 Summaries
                recent_summaries = get_recent_round_summaries(conn, limit=3)

                actions_texts = get_round_actions(conn, round_no)
                locks_now = get_locks(conn, round_no)
                all_metrics = load_all_country_metrics(conn, countries)

                # chosen actions string (für summary + KI)
                chosen_actions_lines = []
                for c in countries:
                    v = locks_now[c]
                    chosen_actions_lines.append(
                        f"- {countries_display[c]} ({c}): {v} -> {actions_texts[c][v]}"
                    )
                chosen_actions_str = "\n".join(chosen_actions_lines)

                eu_before = get_eu_state(conn)

                result = resolve_round_all_countries(
                    api_key=api_key,
                    model="mistral-small",
                    round_no=round_no,
                    eu_state=eu_before,
                    countries_metrics=all_metrics,
                    countries_display=countries_display,
                    actions_texts=actions_texts,
                    locked_choices=locks_now,
                    recent_round_summaries=recent_summaries,  # ✅ Stufe 1
                    temperature=0.6,
                    top_p=0.95,
                    max_tokens=1500,
                )

                # Apply EU
                eu_delta = int(result["eu"].get("kohäsion_delta", 0))
                new_global = str(result["eu"].get("global_context", eu_before["global_context"]))
                set_eu_state(conn, eu_before["cohesion"] + eu_delta, new_global)

                # Apply countries + history
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
                        global_context=new_global,
                        deltas=d,
                    )

                eu_after = get_eu_state(conn)

                # Stufe 2 Summary schreiben + speichern
                summary_text = generate_round_summary(
                    api_key=api_key,
                    model="mistral-small",
                    round_no=round_no,
                    memory_in=recent_summaries,
                    eu_before=eu_before,
                    eu_after=eu_after,
                    chosen_actions_str=chosen_actions_str,
                    result_obj=result,
                    temperature=0.4,
                    top_p=0.95,
                    max_tokens=500,
                )
                upsert_round_summary(conn, round_no, summary_text)

                # Next round
                clear_round_data(conn, round_no)
                set_game_meta(conn, round_no + 1, "setup")

            st.success("Runde aufgelöst, Werte gesetzt, nächste Runde gestartet.")
            st.rerun()

        st.caption("Hinweis: Bis zur Veröffentlichung können Aktionen neu generiert werden. Nach Veröffentlichung sind Optionen fix.")


# ----------------------------
# LEFT: Player View (default: own country)
# ----------------------------
with left:
    st.subheader("🎮 Spielerbereich")

    # Persistent player country selection (rerun-safe)
    if "my_country" not in st.session_state:
        st.session_state.my_country = countries[0]

    country_keys = countries
    country_labels = [countries_display[k] for k in country_keys]

    def _on_country_change():
        label = st.session_state.my_country_label
        st.session_state.my_country = country_keys[country_labels.index(label)]

    if "my_country_label" not in st.session_state:
        st.session_state.my_country_label = countries_display[st.session_state.my_country]

    st.selectbox(
        "Ich spiele:",
        country_labels,
        key="my_country_label",
        on_change=_on_country_change,
    )

    my_country = st.session_state.my_country

    actions_texts = get_round_actions(conn, round_no)
    locks_now = get_locks(conn, round_no)

    st.write("---")
    st.subheader(f"🏳️ Mein Land: {countries_display[my_country]}")

    my_metrics = load_country_metrics(conn, my_country)
    if not my_metrics:
        st.error("Mein Land konnte nicht geladen werden.")
    else:
        render_metrics(my_metrics)

        # Optional: eigener Siegfortschritt
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

        # Actions: only own country
        if phase != "actions_published":
            st.info("Optionen sind noch nicht veröffentlicht. Warte auf den Game Master.")
        else:
            a = actions_texts.get(my_country, {})
            if not a or len(a) < 3:
                st.warning("Optionen fehlen noch (GM muss Aktionen generieren und veröffentlichen).")
            else:
                st.subheader("Öffentliche Aktion wählen")

                if my_country in locks_now:
                    st.success("Du bist eingelockt. (Welche Variante bleibt für andere verborgen.)")
                else:
                    st.warning("Du bist noch nicht eingelockt.")

                options = {
                    "aggressiv": a["aggressiv"],
                    "moderate": a["moderate"],
                    "passiv": a["passiv"],
                }
                labels = [options["aggressiv"], options["moderate"], options["passiv"]]
                choice_label = st.radio("Option:", labels, index=1)

                chosen_variant = next(k for k, v in options.items() if v == choice_label)

                if st.button("✅ Auswahl einlocken", use_container_width=True):
                    lock_choice(conn, round_no, my_country, chosen_variant)
                    st.rerun()

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

    st.write("---")
    with st.expander("🌍 Andere Länder (nur Metriken/History)"):
        other_countries = [c for c in countries if c != my_country]
        tabs = st.tabs([countries_display[c] for c in other_countries])

        for idx, c in enumerate(other_countries):
            with tabs[idx]:
                m = load_country_metrics(conn, c)
                if not m:
                    st.error("Land konnte nicht geladen werden.")
                    continue

                render_metrics(m)

                with st.expander("Turn-History"):
                    rows = load_recent_history(conn, c, limit=12)
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

conn.close()
