"""
AI Report Engine - Generazione report AI via GitHub Models (GPT-4.1).

Supporta report multipli, ciascuno con contesto e prompt specifici.
Usa GPT-4.1 tramite GitHub Models API (gratuito con GitHub token).
Report disponibili definiti in ai_report_definitions.py.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from config.settings import config
from utils.logger import get_logger

logger = get_logger("analysis.ai_daily_report")

# Directory per salvare i report (cache)
REPORTS_DIR = Path(__file__).resolve().parent.parent / "data" / "ai_reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# File per i prompt personalizzati salvati come preimpostati
CUSTOM_PRESETS_FILE = REPORTS_DIR / "_custom_presets.json"


def load_custom_presets() -> dict:
    """Carica i prompt personalizzati salvati come preimpostati."""
    if CUSTOM_PRESETS_FILE.exists():
        try:
            return json.loads(CUSTOM_PRESETS_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_custom_preset(preset_id: str, name: str, icon: str, description: str,
                       prompt_template: str, include_market_data: bool = True) -> bool:
    """Salva un nuovo prompt personalizzato come preimpostato.

    Args:
        preset_id: ID univoco (slug)
        name: Nome visualizzato
        icon: Emoji
        description: Descrizione breve
        prompt_template: Il testo del prompt
        include_market_data: Se includere dati di mercato nel contesto

    Returns:
        True se salvato con successo
    """
    try:
        presets = load_custom_presets()
        presets[preset_id] = {
            "id": preset_id,
            "name": name,
            "icon": icon,
            "description": description,
            "prompt_template": prompt_template,
            "include_market_data": include_market_data,
            "created_at": datetime.now().isoformat(),
        }
        CUSTOM_PRESETS_FILE.write_text(
            json.dumps(presets, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(f"Preset personalizzato salvato: {preset_id} ({name})")
        return True
    except Exception as e:
        logger.error(f"Errore salvataggio preset: {e}")
        return False


def delete_custom_preset(preset_id: str) -> bool:
    """Elimina un preset personalizzato."""
    try:
        presets = load_custom_presets()
        if preset_id in presets:
            del presets[preset_id]
            CUSTOM_PRESETS_FILE.write_text(
                json.dumps(presets, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            logger.info(f"Preset eliminato: {preset_id}")
            return True
        return False
    except Exception as e:
        logger.error(f"Errore eliminazione preset: {e}")
        return False


def generate_custom_preset_report(api_key: str, preset_id: str, force: bool = False) -> Optional[str]:
    """Genera un report da un preset personalizzato."""
    presets = load_custom_presets()
    if preset_id not in presets:
        logger.error(f"Preset '{preset_id}' non trovato")
        return None

    preset = presets[preset_id]
    today = datetime.now().strftime("%Y-%m-%d")
    cache_file = REPORTS_DIR / f"custom_{preset_id}_{today}.md"

    # Cache check
    if not force and cache_file.exists():
        logger.info(f"Preset '{preset_id}' di oggi in cache")
        return cache_file.read_text(encoding="utf-8")

    # Build prompt
    full_prompt = ""
    if preset.get("include_market_data", True):
        try:
            from analysis.ai_report_definitions import get_daily_market_context
            context = get_daily_market_context()
            full_prompt += f"DATI DI MERCATO AGGIORNATI:\n{context}\n\n---\n\n"
        except Exception as e:
            logger.warning(f"Errore raccolta contesto: {e}")

    full_prompt += preset["prompt_template"]

    report = _call_github_models(api_key, full_prompt, f"custom_{preset_id}", today)
    if report is None and api_key.startswith("sk-ant-"):
        report = _call_anthropic(api_key, full_prompt, f"custom_{preset_id}", today)

    if report:
        cache_file.write_text(report, encoding="utf-8")
        logger.info(f"Report preset '{preset_id}' salvato in {cache_file}")

    return report


def generate_report(api_key: str, report_id: str = "daily_market", force: bool = False) -> Optional[str]:
    """
    Genera un report AI specifico.

    Args:
        api_key: GitHub PAT token o Anthropic key
        report_id: ID del report (vedi AI_REPORTS in ai_report_definitions.py)
        force: Se True, rigenera anche se esiste già il report di oggi

    Returns:
        Report markdown o None se errore
    """
    from analysis.ai_report_definitions import AI_REPORTS

    if report_id not in AI_REPORTS:
        logger.error(f"Report '{report_id}' non trovato. Disponibili: {list(AI_REPORTS.keys())}")
        return None

    report_def = AI_REPORTS[report_id]
    today = datetime.now().strftime("%Y-%m-%d")
    cache_file = REPORTS_DIR / f"{report_id}_{today}.md"

    # Controlla cache
    if not force and cache_file.exists():
        logger.info(f"Report '{report_id}' di oggi già in cache: {cache_file}")
        return cache_file.read_text(encoding="utf-8")

    if not api_key:
        logger.warning("API key non configurata")
        return None

    # Raccogli contesto specifico per questo report
    logger.info(f"Raccolta dati per report '{report_def['name']}'...")
    context = report_def["get_context"]()
    prompt = report_def["get_prompt"](context, today)

    # Genera con GitHub Models (GPT-4.1)
    report = _call_github_models(api_key, prompt, report_id, today)

    # Fallback Anthropic
    if report is None and api_key.startswith("sk-ant-"):
        report = _call_anthropic(api_key, prompt, report_id, today)

    if report:
        cache_file.write_text(report, encoding="utf-8")
        logger.info(f"Report '{report_id}' salvato in {cache_file}")

    return report


# Backward compatibility alias
def generate_daily_report(api_key: str, force: bool = False) -> Optional[str]:
    """Alias per compatibilità: genera il report giornaliero mercati."""
    return generate_report(api_key, report_id="daily_market", force=force)


def _call_github_models(token: str, prompt: str, report_id: str, today: str) -> Optional[str]:
    """Genera report usando GPT-4.1 via GitHub Models API."""
    try:
        from openai import OpenAI

        client = OpenAI(
            base_url="https://models.inference.ai.azure.com",
            api_key=token,
        )

        logger.info(f"Invio richiesta a GPT-4.1 per report '{report_id}'...")

        response = client.chat.completions.create(
            model="gpt-4.1",
            max_tokens=8000,
            temperature=0.7,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Sei un analista finanziario senior con 25 anni di esperienza "
                        "nei mercati globali, specializzato in azionario USA, ETF europei, "
                        "derivati, macro e crypto. Scrivi report dettagliati e professionali "
                        "in italiano. Usa i dati numerici forniti e integra con la tua "
                        "conoscenza. Sii sempre preciso sui numeri e onesto quando non "
                        "hai dati aggiornati."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )

        report = response.choices[0].message.content

        # Salva metadata
        meta_file = REPORTS_DIR / f"{report_id}_{today}_meta.json"
        meta_file.write_text(json.dumps({
            "date": today,
            "report_id": report_id,
            "model": "gpt-4.1 (GitHub Models)",
            "generated_at": datetime.now().isoformat(),
            "tokens_input": response.usage.prompt_tokens if response.usage else 0,
            "tokens_output": response.usage.completion_tokens if response.usage else 0,
        }, indent=2), encoding="utf-8")

        logger.info(f"Report '{report_id}' generato con GPT-4.1")
        return report

    except ImportError:
        logger.error("Pacchetto 'openai' non installato. Installa con: pip install openai")
        return None
    except Exception as e:
        logger.error(f"Errore GitHub Models per '{report_id}': {e}")
        return None


def _call_anthropic(api_key: str, prompt: str, report_id: str, today: str) -> Optional[str]:
    """Fallback: genera report usando Claude (Anthropic)."""
    try:
        import anthropic

        logger.info(f"Fallback Anthropic per report '{report_id}'...")
        client = anthropic.Anthropic(api_key=api_key)

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=8000,
            messages=[{"role": "user", "content": prompt}],
        )

        report = message.content[0].text

        meta_file = REPORTS_DIR / f"{report_id}_{today}_meta.json"
        meta_file.write_text(json.dumps({
            "date": today,
            "report_id": report_id,
            "model": "claude-sonnet-4 (Anthropic)",
            "generated_at": datetime.now().isoformat(),
            "tokens_input": message.usage.input_tokens,
            "tokens_output": message.usage.output_tokens,
        }, indent=2), encoding="utf-8")

        return report

    except Exception as e:
        logger.error(f"Errore Anthropic per '{report_id}': {e}")
        return None


def generate_freeform_report(api_key: str, user_prompt: str, include_market_data: bool = True,
                             save: bool = True) -> Optional[str]:
    """
    Genera un report AI da un prompt libero dell'utente.

    Args:
        api_key: GitHub PAT token o Anthropic key
        user_prompt: Prompt scritto dall'utente
        include_market_data: Se True, aggiunge contesto dati di mercato al prompt
        save: Se True, salva il report su disco

    Returns:
        Report markdown o None se errore
    """
    if not api_key:
        logger.warning("API key non configurata")
        return None

    if not user_prompt.strip():
        logger.warning("Prompt vuoto")
        return None

    # Costruisci il prompt completo
    full_prompt = ""
    if include_market_data:
        try:
            from analysis.ai_report_definitions import get_daily_market_context
            context = get_daily_market_context()
            full_prompt += f"DATI DI MERCATO AGGIORNATI:\n{context}\n\n---\n\n"
        except Exception as e:
            logger.warning(f"Impossibile raccogliere dati di mercato: {e}")

    full_prompt += f"RICHIESTA DELL'UTENTE:\n{user_prompt}"

    today = datetime.now().strftime("%Y-%m-%d")
    now_ts = datetime.now().strftime("%H%M%S")

    # Genera con GitHub Models
    report = _call_github_models(api_key, full_prompt, "freeform", today)

    # Fallback Anthropic
    if report is None and api_key.startswith("sk-ant-"):
        report = _call_anthropic(api_key, full_prompt, "freeform", today)

    if report and save:
        # Salva con timestamp per permettere multipli report nello stesso giorno
        save_file = REPORTS_DIR / f"freeform_{today}_{now_ts}.md"
        # Aggiungi header con il prompt originale
        header = f"<!-- prompt: {user_prompt[:200]} -->\n"
        header += f"<!-- generated: {datetime.now().isoformat()} -->\n\n"
        save_file.write_text(header + report, encoding="utf-8")

        # Salva metadata
        meta_file = REPORTS_DIR / f"freeform_{today}_{now_ts}_meta.json"
        meta_file.write_text(json.dumps({
            "date": today,
            "timestamp": now_ts,
            "report_id": "freeform",
            "user_prompt": user_prompt,
            "include_market_data": include_market_data,
            "generated_at": datetime.now().isoformat(),
        }, indent=2, ensure_ascii=False), encoding="utf-8")

        logger.info(f"Report freeform salvato in {save_file}")

    return report


def list_freeform_reports() -> list[dict]:
    """Lista tutti i report freeform salvati, dal più recente."""
    meta_files = sorted(REPORTS_DIR.glob("freeform_*_meta.json"), reverse=True)
    reports = []
    for mf in meta_files:
        try:
            meta = json.loads(mf.read_text(encoding="utf-8"))
            # Trova il file .md corrispondente
            md_name = mf.name.replace("_meta.json", ".md")
            md_file = REPORTS_DIR / md_name
            if md_file.exists():
                reports.append({
                    "file": str(md_file),
                    "meta_file": str(mf),
                    "date": meta.get("date", ""),
                    "timestamp": meta.get("timestamp", ""),
                    "prompt": meta.get("user_prompt", ""),
                    "generated_at": meta.get("generated_at", ""),
                })
        except Exception:
            continue
    return reports


def get_freeform_report(file_path: str) -> Optional[str]:
    """Legge un report freeform dal suo path."""
    p = Path(file_path)
    if p.exists():
        content = p.read_text(encoding="utf-8")
        # Rimuovi le righe di commento HTML in cima
        lines = content.split("\n")
        clean_lines = [l for l in lines if not l.strip().startswith("<!-- ")]
        return "\n".join(clean_lines).strip()
    return None


def delete_report(file_path: str) -> bool:
    """Elimina un report e il relativo metadata."""
    try:
        p = Path(file_path)
        if p.exists():
            p.unlink()
        # Elimina anche il meta
        meta_path = p.with_name(p.stem + "_meta.json")
        if meta_path.exists():
            meta_path.unlink()
        logger.info(f"Report eliminato: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Errore eliminazione report: {e}")
        return False


def get_cached_report(report_id: str = "daily_market", date: str = None) -> Optional[str]:
    """Recupera un report dalla cache."""
    if date is None:
        date = datetime.now().strftime("%Y-%m-%d")
    cache_file = REPORTS_DIR / f"{report_id}_{date}.md"
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    # Backward compatibility: prova vecchio formato
    old_file = REPORTS_DIR / f"report_{date}.md"
    if report_id == "daily_market" and old_file.exists():
        return old_file.read_text(encoding="utf-8")
    return None


def list_available_reports(report_id: str = None) -> list[str]:
    """Lista le date dei report disponibili per un tipo specifico o tutti."""
    if report_id:
        pattern = f"{report_id}_????-??-??.md"
    else:
        pattern = "*_????-??-??.md"
    reports = sorted(REPORTS_DIR.glob(pattern), reverse=True)
    dates = []
    for f in reports:
        # Estrai data dal nome file: report_id_YYYY-MM-DD.md
        parts = f.stem.rsplit("_", 3)
        if len(parts) >= 4:
            date = f"{parts[-3]}-{parts[-2]}-{parts[-1]}"
            if date not in dates:
                dates.append(date)
    # Fallback per vecchio formato
    if report_id == "daily_market" or report_id is None:
        old_reports = sorted(REPORTS_DIR.glob("report_????-??-??.md"), reverse=True)
        for f in old_reports:
            date = f.stem.replace("report_", "")
            if date not in dates:
                dates.append(date)
    return sorted(set(dates), reverse=True)
