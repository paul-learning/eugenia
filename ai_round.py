# ai_round.py
from __future__ import annotations
from typing import Dict, Any, Tuple, List
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
        max_tokens=1400,
    )
    return parse_json_maybe(fixed_raw)


def generate_actions_for_country(
    *,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float = 0.9,
    top_p: float = 0.95,
    max_tokens: int = 900,
) -> Tuple[Dict[str, Any], str, bool]:
    client = Mistral(api_key=api_key)

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

    schema_hint = """
{
  "aggressiv": {
    "aktion": "...",
    "folgen": {
      "land": {"militär": 0, "stabilität": 0, "wirtschaft": 0, "diplomatie": 0, "öffentliche_zustimmung": 0},
      "eu": {"kohäsion": 0},
      "global_context": "..."
    }
  },
  "moderate": { ... },
  "passiv": { ... }
}
""".strip()

    used_repair = False
    try:
        obj = parse_json_maybe(raw)
    except Exception:
        used_repair = True
        obj = _repair_to_valid_json(client, model, raw, schema_hint)

    for k in ("aggressiv", "moderate", "passiv"):
        if k not in obj:
            raise ValueError(f"Fehlender Key im JSON: {k}")
        if "aktion" not in obj[k] or "folgen" not in obj[k]:
            raise ValueError(f"Key '{k}' muss 'aktion' und 'folgen' enthalten.")
        folgen = obj[k].get("folgen") or {}
        if "land" not in folgen or "eu" not in folgen or "global_context" not in folgen:
            raise ValueError(f"'{k}.folgen' muss land/eu/global_context enthalten.")

    return obj, raw, used_repair


def resolve_round_all_countries(
    *,
    api_key: str,
    model: str,
    round_no: int,
    eu_state: Dict[str, Any],
    countries_metrics: Dict[str, Dict[str, Any]],
    countries_display: Dict[str, str],
    actions_texts: Dict[str, Dict[str, str]],
    locked_choices: Dict[str, str],
    recent_round_summaries: List[Tuple[int, str]] | None = None,
    external_events: List[Dict[str, Any]] | None = None,
    temperature: float = 0.6,
    top_p: float = 0.95,
    max_tokens: int = 1700,
) -> Dict[str, Any]:
    client = Mistral(api_key=api_key)

    chosen_actions_block = []
    for c, variant in locked_choices.items():
        display = countries_display.get(c, c)
        text = actions_texts.get(c, {}).get(variant, "")
        chosen_actions_block.append(f"- {display} ({c}): {variant} -> {text}")
    chosen_actions_str = "\n".join(chosen_actions_block)

    metrics_block = []
    for c, m in countries_metrics.items():
        display = countries_display.get(c, c)
        metrics_block.append(
            f"- {display} ({c}): Militär={m['military']}, Stabilität={m['stability']}, Wirtschaft={m['economy']}, "
            f"Diplomatie={m['diplomatic_influence']}, Zustimmung={m['public_approval']}. Ambition: {m['ambition']}"
        )
    metrics_str = "\n".join(metrics_block)

    memory_str = "Keine."
    if recent_round_summaries:
        rev = list(reversed(recent_round_summaries))
        memory_str = "\n".join([f"- Runde {r}: {s}" for r, s in rev])

    external_str = "Keine."
    if external_events:
        lines = []
        for e in external_events:
            mods = e.get("modifiers", {})
            lines.append(f"- {e.get('actor')}: {e.get('headline')} | mods={mods}")
        external_str = "\n".join(lines)

    schema_hint = """
{
  "eu": {"kohäsion_delta": 0, "global_context": "..."},
  "länder": {
    "Germany": {"militär": 0, "stabilität": 0, "wirtschaft": 0, "diplomatie": 0, "öffentliche_zustimmung": 0}
  },
  "notizen": "kurz"
}
""".strip()

    prompt = f"""
Du bist Spielleiter und Simulations-Engine für ein EU-Geopolitik-Spiel.

Aufgabe:
Berechne das Ergebnis der Runde {round_no} für ALLE Länder gemeinsam.
Effekte dürfen sich gegenseitig beeinflussen (z.B. Ungarn-Aktion wirkt auf Frankreich).

WICHTIG (Druckmechanik):
- Hoher Threat/Frontline erhöht Wert von Militär/Abschreckung, aber kann Zustimmung/Stabilität kosten.
- Hoher Energy/Migration/Disinfo/TradeWar-Druck verstärkt innenpolitische Risiken (Zustimmung/Stabilität) und macht Deals/Diplomatie wichtiger.
- Nutze Memory für wiederkehrende Konflikte/Kooperationen.

Story-/Memory-Kontext (letzte Runden):
{memory_str}

Außenmächte-Moves dieser Runde (USA/China/Russia):
{external_str}

Aktueller EU-Status:
- Kohäsion={eu_state["cohesion"]}%
- Threat Level={eu_state["threat_level"]}/100
- Frontline Pressure={eu_state["frontline_pressure"]}/100
- Energy Pressure={eu_state["energy_pressure"]}/100
- Migration Pressure={eu_state["migration_pressure"]}/100
- Disinfo Pressure={eu_state["disinfo_pressure"]}/100
- Trade War Pressure={eu_state["trade_war_pressure"]}/100
- Globaler Kontext: {eu_state["global_context"]}

Aktuelle Länderwerte:
{metrics_str}

Gewählte Aktionen dieser Runde:
{chosen_actions_str}

Output:
- Gib NUR gültiges JSON zurück (kein Markdown).
- Gib nur Netto-DELTAS je Land aus (Ganzzahlen, typischerweise -12..+12).
- Zusätzlich EU-Kohäsions-Delta und neuen global_context (1 Zeile).
- Keys in "länder" müssen exakt die internen Country-Keys sein: {list(countries_metrics.keys())}
- Alle Länder müssen enthalten sein.

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

    if "eu" not in obj or "länder" not in obj:
        raise ValueError("Resolve-JSON muss 'eu' und 'länder' enthalten.")
    for c in countries_metrics.keys():
        if c not in obj["länder"]:
            raise ValueError(f"Resolve-JSON: fehlendes Land in 'länder': {c}")

    return obj


def generate_round_summary(
    *,
    api_key: str,
    model: str,
    round_no: int,
    memory_in: List[Tuple[int, str]] | None,
    eu_before: Dict[str, Any],
    eu_after: Dict[str, Any],
    external_events: List[Dict[str, Any]] | None,
    chosen_actions_str: str,
    result_obj: Dict[str, Any],
    temperature: float = 0.4,
    top_p: float = 0.95,
    max_tokens: int = 520,
) -> str:
    client = Mistral(api_key=api_key)

    memory_str = "Keine."
    if memory_in:
        rev = list(reversed(memory_in))
        memory_str = "\n".join([f"- Runde {r}: {s}" for r, s in rev])

    external_str = "Keine."
    if external_events:
        lines = []
        for e in external_events:
            lines.append(f"- {e.get('actor')}: {e.get('headline')}")
        external_str = "\n".join(lines)

    schema_hint = """{ "summary": "..." }"""

    prompt = f"""
Du bist Chronist eines EU-Geopolitik-Spiels.
Erstelle eine sehr kurze Zusammenfassung der Runde {round_no} als 2–4 Bulletpoints.

Inputs:
- Memory (letzte Runden):
{memory_str}

- Außenmächte-Moves:
{external_str}

- EU vorher: Kohäsion={eu_before["cohesion"]}%, Threat={eu_before["threat_level"]}, Frontline={eu_before["frontline_pressure"]},
  Energy={eu_before["energy_pressure"]}, Migration={eu_before["migration_pressure"]}, Disinfo={eu_before["disinfo_pressure"]}, TradeWar={eu_before["trade_war_pressure"]}
- EU nachher: Kohäsion={eu_after["cohesion"]}%, Threat={eu_after["threat_level"]}, Frontline={eu_after["frontline_pressure"]},
  Energy={eu_after["energy_pressure"]}, Migration={eu_after["migration_pressure"]}, Disinfo={eu_after["disinfo_pressure"]}, TradeWar={eu_after["trade_war_pressure"]}

- Gewählte Aktionen:
{chosen_actions_str}

- Ergebnis (Deltas):
{result_obj}

Regeln:
- Gib NUR gültiges JSON zurück, Schema: {schema_hint}
- "summary" ist ein String mit 2–4 Bulletpoints (jede Zeile beginnt mit "- ").
- Maximal ~520 Zeichen.
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

    summary = str(obj.get("summary", "")).strip()
    if not summary:
        summary = "- (Keine Summary generiert)"
    return summary
