# ai_external.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple, Optional
from mistralai import Mistral

from utils import content_to_text, parse_json_maybe


def _chat(client: Mistral, model: str, messages, temperature: float, top_p: float, max_tokens: int) -> str:
    resp = client.chat.complete(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return content_to_text(resp.choices[0].message.content)


def _repair_to_valid_json(client: Mistral, model: str, bad_text: str, schema_hint: str) -> Dict[str, Any]:
    repair_prompt = f"""
Du bist ein Validator/Formatter. Wandle die folgende Ausgabe in **gültiges JSON** um.

Wichtig:
- Gib **NUR** JSON zurück (keine Erklärungen, kein Markdown).
- Nutze **nur** doppelte Anführungszeichen.
- Keine trailing commas.
- Schema MUSS exakt passen.

Schema:
{schema_hint}

Hier ist die zu reparierende Ausgabe:
{bad_text}
""".strip()

    fixed_raw = _chat(
        client,
        model,
        messages=[
            {"role": "system", "content": "Du gibst ausschließlich gültiges JSON zurück. Kein Markdown."},
            {"role": "user", "content": repair_prompt},
        ],
        temperature=0.2,
        top_p=1.0,
        max_tokens=1200,
    )
    return parse_json_maybe(fixed_raw)


def generate_external_moves(
    *,
    api_key: str,
    model: str,
    round_no: int,
    eu_state: Dict[str, Any],
    recent_round_summaries: List[Tuple[int, str]] | None = None,
    # NEW:
    craziness_by_actor: Optional[Dict[str, int]] = None,
    temperature: float = 0.8,
    top_p: float = 0.95,
    max_tokens: int = 1200,
) -> Dict[str, Any]:
    """
    Output schema:
    {
      "global_context": "1 Zeile",
      "moves": [
        {
          "actor":"Russia",
          "craziness": 0,
          "headline":"...",
          "quote":"...",
          "modifiers":{
             "eu_cohesion_delta": 0,
             "threat_delta": 0,
             "frontline_delta": 0,
             "energy_delta": 0,
             "migration_delta": 0,
             "disinfo_delta": 0,
             "trade_war_delta": 0
          }
        },
        {"actor":"USA",...},
        {"actor":"China",...}
      ]
    }
    """
    client = Mistral(api_key=api_key)

    memory_str = "Keine."
    if recent_round_summaries:
        rev = list(reversed(recent_round_summaries))
        memory_str = "\n".join([f"- Runde {r}: {s}" for r, s in rev])

    # Default craziness if not provided
    cb = craziness_by_actor or {}
    usa_c = int(cb.get("USA", 50))
    rus_c = int(cb.get("Russia", 50))
    chi_c = int(cb.get("China", 50))

    schema_hint = """
{
  "global_context": "1 Zeile",
  "moves": [
    {
      "actor": "Russia",
      "craziness": 0,
      "headline": "...",
      "quote": "...",
      "modifiers": {
        "eu_cohesion_delta": 0,
        "threat_delta": 0,
        "frontline_delta": 0,
        "energy_delta": 0,
        "migration_delta": 0,
        "disinfo_delta": 0,
        "trade_war_delta": 0
      }
    },
    {
      "actor":"USA",
      "craziness": 0,
      "headline":"...",
      "quote":"...",
      "modifiers": {
        "eu_cohesion_delta": 0,
        "threat_delta": 0,
        "frontline_delta": 0,
        "energy_delta": 0,
        "migration_delta": 0,
        "disinfo_delta": 0,
        "trade_war_delta": 0
      }
    },
    {
      "actor":"China",
      "craziness": 0,
      "headline":"...",
      "quote":"...",
      "modifiers": {
        "eu_cohesion_delta": 0,
        "threat_delta": 0,
        "frontline_delta": 0,
        "energy_delta": 0,
        "migration_delta": 0,
        "disinfo_delta": 0,
        "trade_war_delta": 0
      }
    }
  ]
}
""".strip()

    prompt = f"""
Du bist die Weltlage-Engine eines EU-Geopolitik-Spiels.
Erzeuge für Runde {round_no} GENAU 3 Außenmacht-Züge: USA, China, Russland.

NEU: Crazy-Faktor je Außenmacht
- USA: {usa_c}/100
- Russia: {rus_c}/100
- China: {chi_c}/100

Interpretation des Crazy-Faktors (0..100):
- 0–30: rational/realpolitisch
- 31–70: provokativ/unberechenbar
- 71–100: sehr "wild" / überzogen (aber bitte trotzdem innerhalb geopolitischer Plausibilität: keine Fantasy)

Ziel:
- Mehr Kriegs-/Sicherheitsdruck (threat/frontline) realistisch eskalieren oder deeskalieren.
- Mehr Innenpolitik/Populismus triggern (migration/disinfo/energy wirken indirekt auf Zustimmung/Stabilität).
- Mehr Diplomatie/Deals ermöglichen (USA/China-Angebote oder Druck).

Aktueller EU-Status:
- EU-Kohäsion: {eu_state["cohesion"]}%
- Threat Level: {eu_state["threat_level"]} / 100
- Frontline Pressure: {eu_state["frontline_pressure"]} / 100
- Energy Pressure: {eu_state["energy_pressure"]} / 100
- Migration Pressure: {eu_state["migration_pressure"]} / 100
- Disinfo Pressure: {eu_state["disinfo_pressure"]} / 100
- Trade War Pressure: {eu_state["trade_war_pressure"]} / 100
- Globaler Kontext: {eu_state["global_context"]}

Memory (letzte Runden):
{memory_str}

WICHTIG: Quote/Soundbite Regeln
- Gib pro Move zusätzlich "quote" aus: ein KURZES, fiktives Soundbite (1–2 Sätze).
- Die Quote ist NUR stilistisch inspiriert von öffentlicher Rhetorik:
  * USA: Trump-ähnlich (kurz, superlativ, deal/pressure)
  * Russia: Putin-ähnlich (kalt, "rote Linien", Souveränität)
  * China: Xi-ähnlich (höflich, "Harmonie", aber unmissverständlich)
- KEINE echten Zitate, KEINE Behauptung es wäre wörtlich, KEINE Jahreszahlen/Quellen.
- Inhalt muss zur headline passen.

Regeln:
- Gib NUR gültiges JSON zurück, kein Markdown.
- actor muss exakt "USA", "China", "Russia" sein (jeweils einmal).
- craziness muss exakt den oben genannten Werten entsprechen.
- headline ist öffentlich (1 Satz).
- quote ist öffentlich (1–2 Sätze).
- modifiers sind Ganzzahlen in etwa -12..+12 (eu_cohesion_delta eher -4..+4).
- global_context ist eine neue 1-Zeilen-Lagebeschreibung, die die drei Moves widerspiegelt.
- Moves sollen sich unterscheiden und plausible Folgeketten nahelegen.

Schema:
{schema_hint}
""".strip()

    raw = _chat(
        client,
        model,
        messages=[
            {"role": "system", "content": "Antworte ausschließlich mit gültigem JSON. Kein Markdown."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    try:
        obj = parse_json_maybe(raw)
    except Exception:
        obj = _repair_to_valid_json(client, model, raw, schema_hint)

    # minimal validate
    moves = obj.get("moves", [])
    actors = {m.get("actor") for m in moves}
    need = {"USA", "China", "Russia"}
    if actors != need:
        raise ValueError(f"External moves: actors müssen genau {need} sein, bekommen: {actors}")

    # enforce craziness values
    wanted = {"USA": usa_c, "Russia": rus_c, "China": chi_c}
    for m in moves:
        a = m.get("actor")
        if a in wanted:
            m["craziness"] = int(wanted[a])

        # harden fields
        m["headline"] = str(m.get("headline", "")).strip()
        m["quote"] = str(m.get("quote", "")).strip()
        if not m["headline"]:
            raise ValueError(f"External move ({a}) hat leere headline.")
        if not m["quote"]:
            # fallback: not fatal, but keep non-empty to avoid UI weirdness
            m["quote"] = "—"

    return obj
