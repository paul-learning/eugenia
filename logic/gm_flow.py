import random
from typing import Dict, Any, List, Callable

import streamlit as st

from db import (
    get_round_actions,
    all_locked,
    get_external_events,
    get_recent_round_summaries,
    get_eu_state,
    clear_external_events,
    upsert_external_event,
    set_eu_state,
    clear_domestic_events,
    load_all_country_metrics,
    load_recent_history,
    get_domestic_events,
    upsert_domestic_event,
    set_game_meta,
    get_locks,
    apply_country_deltas,
    insert_turn_history,
    upsert_round_summary,
    upsert_country_snapshot,
    get_max_snapshot_round,
    clear_round_data,
    set_game_over,
    upsert_round_actions,
)

from ai_external import generate_external_moves, generate_domestic_events
from ai_round import (
    generate_actions_for_country,
    resolve_round_all_countries,
    generate_round_summary,
)


def render_gm_controls(
    *,
    conn,
    api_key: str,
    round_no: int,
    phase: str,
    countries: List[str],
    countries_display: Dict[str, str],
    country_defs: Dict[str, Dict[str, Any]],
    external_crazy_baseline_ranges: Dict[str, tuple],
    # Injected from app.py to keep this step 100% behavior-identical
    build_action_prompt: Callable[..., str],
    summarize_recent_actions: Callable[[Any], str],
    apply_external_modifiers_to_eu: Callable[[Dict[str, Any], Dict[str, Any]], Dict[str, Any]],
    decay_pressures: Callable[[Dict[str, Any]], Dict[str, Any]],
    progress_from_conditions: Callable[[Any], float],
    evaluate_all_countries,  # may be None
) -> None:
    """
    Extracted GM control flow from app.py.
    No intended behavior changes (same buttons, same DB calls, same phases).
    """

    with st.expander("🎛️ Game Master Steuerung (sequenziell)", expanded=False):
        actions_in_db = get_round_actions(conn, round_no)
        have_all_actions = all((c in actions_in_db and len(actions_in_db[c]) == 3) for c in countries)
        have_all_locks = all_locked(conn, round_no, countries)
        have_external = len(get_external_events(conn, round_no)) == 3

        if phase == "game_over":
            st.warning("Game Over – nur Reset möglich.")
            st.stop()

        # 1) External moves + Domestic events
        external_disabled = (phase == "actions_published")
        if st.button(
            "⚠️ Außenmächte-Moves und Innenpolitik-Headlines generieren",
            disabled=external_disabled,
            use_container_width=True,
        ):
            with st.spinner("Generiere Außenmächte-Moves..."):
                recent_summaries = get_recent_round_summaries(conn, limit=3)
                eu_before = get_eu_state(conn)

                usa_min, usa_max = external_crazy_baseline_ranges["USA"]
                rus_min, rus_max = external_crazy_baseline_ranges["Russia"]
                chi_min, chi_max = external_crazy_baseline_ranges["China"]

                craziness_by_actor = {
                    "USA": random.randint(usa_min, usa_max),
                    "Russia": random.randint(rus_min, rus_max),
                    "China": random.randint(chi_min, chi_max),
                }

                moves_obj = generate_external_moves(
                    api_key=api_key,
                    model="mistral-small",
                    round_no=round_no,
                    eu_state=eu_before,
                    recent_round_summaries=recent_summaries,
                    craziness_by_actor=craziness_by_actor,
                    temperature=0.8,
                    top_p=0.95,
                    max_tokens=1200,
                )

                clear_external_events(conn, round_no)

                for m in moves_obj["moves"]:
                    upsert_external_event(
                        conn,
                        round_no,
                        actor=m["actor"],
                        headline=m["headline"],
                        modifiers=m.get("modifiers", {}),
                        quote=m.get("quote", ""),
                        craziness=int(m.get("craziness", 0) or 0),
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

                # Domestic headlines
                clear_domestic_events(conn, round_no)

                all_metrics = load_all_country_metrics(conn, countries)

                recent_actions_by_country = {}
                for c in countries:
                    recent = load_recent_history(conn, c, limit=6)
                    recent_actions_by_country[c] = [r[1] for r in recent if r and r[1]]

                dom_obj = generate_domestic_events(
                    api_key=api_key,
                    model="mistral-small",
                    round_no=round_no,
                    eu_state=eu_after,
                    countries=countries,
                    countries_metrics=all_metrics,
                    recent_round_summaries=recent_summaries,
                    recent_actions_by_country=recent_actions_by_country,
                    temperature=0.85,
                    top_p=0.95,
                    max_tokens=1400,
                )

                for c in countries:
                    e = (dom_obj.get("events", {}) or {}).get(c, {}) or {}
                    upsert_domestic_event(
                        conn,
                        round_no,
                        c,
                        e.get("headline", ""),
                        details=e.get("details", ""),
                        craziness=int(e.get("craziness", 0) or 0),
                    )

                set_game_meta(conn, round_no, "external_generated")
            st.rerun()

        # 2) Generate actions for all
        gen_disabled = not (phase in ("external_generated", "actions_generated") and have_external) or (phase == "actions_published")
        if st.button("⚙️ Aktionen für alle generieren", disabled=gen_disabled, use_container_width=True):
            with st.spinner("Generiere Aktionen für alle Länder..."):
                eu_now = get_eu_state(conn)
                ext_now = get_external_events(conn, round_no)

                all_metrics = load_all_country_metrics(conn, countries)
                dom_map = {e["country"]: e for e in get_domestic_events(conn, round_no)}

                for c in countries:
                    m = all_metrics[c]
                    recent = load_recent_history(conn, c, limit=12)
                    domestic_headline = (dom_map.get(c, {}) or {}).get("headline", "Keine auffälligen Ereignisse gemeldet.")

                    prompt = build_action_prompt(
                        country_display=countries_display[c],
                        metrics=m,
                        eu_state=eu_now,
                        external_events=ext_now,
                        recent_actions_summary=summarize_recent_actions(recent),
                        domestic_headline=domestic_headline,
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
                dom_now = get_domestic_events(conn, round_no)
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
                    domestic_events=dom_now,
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

                all_metrics_before = load_all_country_metrics(conn, countries)
                eu_before_for_progress = get_eu_state(conn)

                max_snap = get_max_snapshot_round(conn)
                need_baseline = (max_snap is None) and (round_no >= 1)

                if need_baseline:
                    if evaluate_all_countries is not None:
                        win_eval_before = evaluate_all_countries(
                            all_country_metrics=all_metrics_before,
                            eu_state=eu_before_for_progress,
                            country_defs=country_defs,
                        )
                        for c in countries:
                            res = win_eval_before.get(c, {})
                            progress_before = progress_from_conditions(res.get("results") or [])
                            upsert_country_snapshot(
                                conn,
                                round_no=round_no - 1,
                                country=c,
                                metrics=all_metrics_before[c],
                                victory_progress=progress_before,
                                is_winner=bool(res.get("is_winner")),
                            )
                    else:
                        for c in countries:
                            upsert_country_snapshot(
                                conn,
                                round_no=round_no - 1,
                                country=c,
                                metrics=all_metrics_before[c],
                                victory_progress=0.0,
                                is_winner=False,
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
                    domestic_events=dom_now,
                    chosen_actions_str=chosen_actions_str,
                    result_obj=result,
                    temperature=0.4,
                    top_p=0.95,
                    max_tokens=520,
                )
                upsert_round_summary(conn, round_no, summary_text)

                winners: List[str] = []
                all_metrics_now = load_all_country_metrics(conn, countries)
                eu_now = get_eu_state(conn)

                if evaluate_all_countries is not None:
                    win_eval = evaluate_all_countries(
                        all_country_metrics=all_metrics_now,
                        eu_state=eu_now,
                        country_defs=country_defs,
                    )
                    for c in countries:
                        res = win_eval.get(c, {})
                        is_winner_now = bool(res.get("is_winner"))
                        progress = progress_from_conditions(res.get("results") or [])
                        upsert_country_snapshot(
                            conn,
                            round_no=round_no,
                            country=c,
                            metrics=all_metrics_now[c],
                            victory_progress=progress,
                            is_winner=is_winner_now,
                        )
                        if is_winner_now:
                            winners.append(c)
                else:
                    for c in countries:
                        upsert_country_snapshot(
                            conn,
                            round_no=round_no,
                            country=c,
                            metrics=all_metrics_now[c],
                            victory_progress=0.0,
                            is_winner=False,
                        )

                clear_round_data(conn, round_no)

                if winners:
                    set_game_over(conn, winner_country=winners[0], winner_round=round_no, reason="win_conditions")
                else:
                    set_game_meta(conn, round_no + 1, "setup")

            st.success("Runde aufgelöst.")
            st.rerun()

        st.caption("Flow: Außenmächte → Aktionen generieren → Veröffentlichen → Lock → Resolve")
