import os
import json
import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path

import gradio as gr
from PIL import Image
from PIL.ExifTags import TAGS
import c2pa

# =========================
# SETUP
# =========================

DB_PATH = "detector.db"
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# C2PA IPTC digital source type URIs that mean AI-generated
# https://cv.iptc.org/newscodes/digitalsourcetype/
AI_DIGITAL_SOURCE_TYPES = {
    "trainedAlgorithmicMedia",             # Fully AI-generated (DALL·E, Firefly, Imagen, gpt-image)
    "compositeWithTrainedAlgorithmicMedia", # Human + AI composite
    "algorithmicMedia",                    # Older algorithmic generation
    "dataDrivenMedia",                     # Data-driven generation
}

# Assertion labels that signal AI generation
AI_ASSERTION_LABELS = {
    "c2pa.ai.generative",
    "c2pa.ai.training",
    "com.adobe.generative-ai",
    "com.openai.dall-e",
    "com.microsoft.copilot",
    "com.google.imagen",
}

# Known AI tool name substrings (case-insensitive)
AI_TOOL_NAMES = [
    "openai", "gpt-image", "dall-e", "dalle",
    "google", "gemini", "imagen", "media processing",
    "adobe", "firefly", "generative-ai",
    "midjourney",
    "stable diffusion", "sdxl",
    "microsoft copilot", "copilot designer",
    "meta ai", "emu",
    "runway", "kling", "sora",
    "ideogram", "leonardo", "canva ai",
]

# =========================
# DATABASE
# =========================

def init_db():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            filename        TEXT NOT NULL,
            file_hash       TEXT NOT NULL,
            file_size       INTEGER,
            verdict         TEXT NOT NULL,
            confidence      TEXT NOT NULL,
            has_c2pa        INTEGER NOT NULL DEFAULT 0,
            c2pa_trusted    INTEGER NOT NULL DEFAULT 0,
            sig_issuer      TEXT,
            generator_tool  TEXT,
            source_type     TEXT,
            action_recorded TEXT,
            scanned_at      TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn

DB = init_db()

# =========================
# C2PA READING
# =========================

def read_c2pa_manifest(filepath: str) -> dict:
    """Read and fully parse the C2PA manifest store from a file."""
    result = {
        "has_manifest": False,
        "trusted": False,
        "manifest_store": None,
        "active_manifest": None,
        "validation_state": None,
        "error": None,
    }

    ext = Path(filepath).suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png",  ".webp": "image/webp",
        ".avif": "image/avif", ".heic": "image/heic",
        ".mp4": "video/mp4",  ".mov": "video/quicktime",
        ".tiff": "image/tiff", ".tif": "image/tiff",
        ".gif": "image/gif",
    }
    mime = mime_map.get(ext, "image/jpeg")

    try:
        with open(filepath, "rb") as f:
            reader = c2pa.Reader(mime, f)
            raw_json = reader.json()

        store = json.loads(raw_json)
        result["has_manifest"] = True
        result["manifest_store"] = store

        # Validation state from top-level
        result["validation_state"] = store.get("validation_state", "Unknown")
        result["trusted"] = store.get("validation_state") == "Trusted"

        # Resolve active manifest
        active_label = store.get("active_manifest")
        manifests = store.get("manifests", {})
        if active_label and active_label in manifests:
            result["active_manifest"] = manifests[active_label]

    except c2pa.C2paError.ManifestNotFound:
        result["error"] = "no_manifest"
    except c2pa.C2paError.NotSupported:
        result["error"] = "no_manifest"
    except c2pa.C2paError.Decoding:
        result["error"] = "no_manifest"
    except c2pa.C2paError.Verify as e:
        # Has manifest but tampered
        result["has_manifest"] = True
        result["trusted"] = False
        result["error"] = f"signature_invalid: {e}"
    except c2pa.C2paError as e:
        result["error"] = f"c2pa_error: {e}"
    except Exception as e:
        result["error"] = f"read_error: {e}"

    return result


# =========================
# AI ANALYSIS
# =========================

def match_ai_tool(text: str) -> str | None:
    """Return matched AI tool name or None."""
    t = text.lower()
    for name in AI_TOOL_NAMES:
        if name in t:
            return text  # return original casing
    return None


def is_ai_source_type(url: str) -> bool:
    """
    Check IPTC digitalSourceType URI.
    e.g. http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia
    """
    for ai_type in AI_DIGITAL_SOURCE_TYPES:
        if ai_type.lower() in url.lower():
            return True
    return False


def analyze_manifest_for_ai(active: dict) -> dict:
    """
    Parse the active manifest dict (as returned by c2pa-python) for AI signals.

    Key paths found in real OpenAI manifest:
      active["claim_generator_info"][0]["name"]          → "OpenAI Media Service API"
      active["assertions"][*]["label"]                   → "c2pa.actions.v2"
      active["assertions"][*]["data"]["actions"][*]["softwareAgent"]["name"] → "gpt-image"
      active["assertions"][*]["data"]["actions"][*]["digitalSourceType"]     → "trainedAlgorithmicMedia" URL
      active["signature_info"]["issuer"]                 → "OpenAI OpCo, LLC"
    """
    result = {
        "is_ai": False,
        "confidence": "none",
        "generator_tool": None,
        "source_type": None,
        "ai_action": None,
        "sig_issuer": None,
        "signals": [],           # human-readable list of signals found
    }

    if not active:
        return result

    # ── 1. claim_generator_info (OpenAI, Adobe pattern) ──────────────────────
    for cgi in active.get("claim_generator_info", []):
        name = cgi.get("name", "")
        matched = match_ai_tool(name)
        if matched:
            result["generator_tool"] = name
            result["signals"].append(f"Claim generator: **{name}**")
            result["is_ai"] = True
            result["confidence"] = "high"

    # ── 2. signature_info (issuer + common_name) ──────────────────────────────
    sig = active.get("signature_info", {})
    issuer = sig.get("issuer", "")
    common_name = sig.get("common_name", "")
    result["sig_issuer"] = issuer
    
    if issuer:
        matched = match_ai_tool(issuer)
        if matched:
            result["signals"].append(f"Certificate issuer: **{issuer}**")
            if not result["generator_tool"]:
                result["generator_tool"] = issuer
            result["is_ai"] = True
            if result["confidence"] != "high":
                result["confidence"] = "high"
    
    # Also check common_name (certificate subject) for generator info
    # Google pattern: "Google Media Processing Services", Adobe pattern: "Adobe Media Processor"
    if common_name and not result["generator_tool"]:
        matched = match_ai_tool(common_name)
        if matched:
            result["signals"].append(f"Certificate subject: **{common_name}**")
            result["generator_tool"] = common_name
            result["is_ai"] = True
            if result["confidence"] == "none":
                result["confidence"] = "medium"

    # ── 3. assertions ─────────────────────────────────────────────────────────
    for assertion in active.get("assertions", []):
        label = assertion.get("label", "")
        data  = assertion.get("data", {}) or {}

        # 3a. AI-specific assertion label
        for ai_label in AI_ASSERTION_LABELS:
            if ai_label in label:
                result["is_ai"] = True
                result["confidence"] = "high"
                result["signals"].append(f"AI assertion label: **{label}**")

        # 3b. c2pa.actions / c2pa.actions.v2
        if "c2pa.actions" in label:
            for action in data.get("actions", []):
                action_name = action.get("action", "")

                # digitalSourceType in the action (real OpenAI pattern)
                dst = action.get("digitalSourceType", "")
                if dst and is_ai_source_type(dst):
                    result["is_ai"] = True
                    result["confidence"] = "high"
                    result["source_type"] = dst
                    # Extract the trailing token for display
                    display_dst = dst.rstrip("/").split("/")[-1]
                    result["signals"].append(
                        f"Digital source type: **{display_dst}** (action: `{action_name}`)"
                    )
                    if not result["ai_action"]:
                        result["ai_action"] = action_name

                # softwareAgent name (real OpenAI pattern: {"name": "gpt-image", "version": "2.0"})
                sa = action.get("softwareAgent", {})
                agent_name = ""
                if isinstance(sa, dict):
                    agent_name = sa.get("name", "")
                elif isinstance(sa, str):
                    agent_name = sa

                if agent_name and match_ai_tool(agent_name):
                    result["is_ai"] = True
                    result["signals"].append(f"Software agent: **{agent_name}** (action: `{action_name}`)")
                    if not result["generator_tool"]:
                        result["generator_tool"] = agent_name
                    if result["confidence"] != "high":
                        result["confidence"] = "medium"

        # 3c. stds.schema-org.CreativeWork / c2pa.metadata
        if "schema-org" in label or "c2pa.metadata" in label:
            dst = data.get("digitalSourceType", "")
            if dst and is_ai_source_type(dst):
                result["is_ai"] = True
                result["confidence"] = "high"
                result["source_type"] = dst
                display_dst = dst.rstrip("/").split("/")[-1]
                result["signals"].append(f"Creative Work digital source type: **{display_dst}**")

    # ── 4. Legacy claim_generator string ─────────────────────────────────────
    cg = active.get("claim_generator", "")
    if cg and match_ai_tool(cg):
        result["is_ai"] = True
        result["signals"].append(f"Claim generator string: **{cg}**")
        if not result["generator_tool"]:
            result["generator_tool"] = cg
        if result["confidence"] == "none":
            result["confidence"] = "medium"

    return result


# =========================
# EXIF FALLBACK
# =========================

def extract_exif(filepath: str) -> dict:
    meta = {}
    try:
        img = Image.open(filepath)
        exif = img.getexif()
        if exif:
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, str(tag_id))
                meta[tag] = str(value)
        if hasattr(img, "info"):
            for k, v in (img.info or {}).items():
                if isinstance(v, (str, int, float)):
                    meta[f"info_{k}"] = str(v)
    except Exception as e:
        meta["_error"] = str(e)
    return meta


def exif_heuristic(exif: dict) -> tuple[bool, str]:
    """Last-resort: scan all EXIF text for AI tool names."""
    combined = json.dumps(exif).lower()
    for name in AI_TOOL_NAMES:
        if name in combined:
            return True, name
    return False, ""


# =========================
# VERDICT BUILDER
# =========================

def build_verdict(c2pa_result: dict, ai_analysis: dict, exif: dict) -> dict:
    verdict = {
        "status": "unknown",
        "confidence": "none",
        "headline": "",
        "details": [],
        "generator": None,
        "source_type": None,
        "issuer": None,
        "validation": None,
    }

    has = c2pa_result.get("has_manifest", False)
    trusted = c2pa_result.get("trusted", False)
    error = c2pa_result.get("error", "") or ""
    validation = c2pa_result.get("validation_state")

    verdict["validation"] = validation
    verdict["issuer"] = ai_analysis.get("sig_issuer")
    verdict["generator"] = ai_analysis.get("generator_tool")
    verdict["source_type"] = ai_analysis.get("source_type")

    # ── Case 1: Tampered signature ────────────────────────────────────────────
    if has and "signature_invalid" in error:
        verdict["status"] = "tampered"
        verdict["confidence"] = "high"
        verdict["headline"] = "⚠️ C2PA Manifest Present — Signature INVALID"
        verdict["details"].append(
            "The file contains a C2PA manifest but the cryptographic signature "
            "failed verification. The file may have been modified after signing."
        )
        return verdict

    # ── Case 2: Valid C2PA manifest ───────────────────────────────────────────
    if has:
        if ai_analysis["is_ai"]:
            verdict["status"] = "ai_detected"
            verdict["confidence"] = ai_analysis["confidence"]
            verdict["headline"] = "🤖 AI-Generated Content Detected"
            for signal in ai_analysis["signals"]:
                verdict["details"].append(signal)
            if trusted:
                verdict["details"].append(
                    f"✅ Signature **trusted** — validation state: `{validation}`"
                )
            else:
                verdict["details"].append(
                    f"⚠️ Signature **untrusted** — validation state: `{validation}`"
                )
        else:
            verdict["status"] = "no_ai"
            verdict["confidence"] = "high"
            verdict["headline"] = "✅ C2PA Credential Found — No AI Signal"
            verdict["details"].append(
                "A valid C2PA manifest was found but it contains no AI generation signals."
            )
            if trusted:
                verdict["details"].append(f"Signature trusted — validation state: `{validation}`")
        return verdict

    # ── Case 3: No C2PA — EXIF heuristic ─────────────────────────────────────
    is_ai_exif, tool = exif_heuristic(exif)
    if is_ai_exif:
        verdict["status"] = "ai_detected"
        verdict["confidence"] = "low"
        verdict["headline"] = "⚠️ Possible AI Content (EXIF Heuristic Only)"
        verdict["generator"] = tool
        verdict["details"].append(
            f"No C2PA credential found, but EXIF/metadata contains a reference to: **{tool}**"
        )
        verdict["details"].append(
            "⚠️ This is a weak heuristic signal — not cryptographically verified."
        )
    else:
        verdict["status"] = "no_credential"
        verdict["confidence"] = "none"
        verdict["headline"] = "❓ No AI Credential Found"
        verdict["details"].append(
            "No C2PA manifest and no AI tool references found in metadata."
        )
        verdict["details"].append(
            "This does **not** prove the content is human-made — many AI tools don't embed credentials."
        )

    return verdict


# =========================
# FILE HASH
# =========================

def hash_file(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# =========================
# MAIN PROCESSING
# =========================

def process_image(file_obj):
    if file_obj is None:
        return "### Please upload an image or video file.", {}, {}, []

    filepath = file_obj if isinstance(file_obj, str) else file_obj.name
    filename = Path(filepath).name
    file_size = Path(filepath).stat().st_size
    file_hash = hash_file(filepath)

    save_path = UPLOAD_DIR / filename
    with open(filepath, "rb") as src, open(save_path, "wb") as dst:
        dst.write(src.read())

    # Pipeline
    c2pa_result  = read_c2pa_manifest(str(save_path))
    ai_analysis  = analyze_manifest_for_ai(c2pa_result.get("active_manifest") or {})
    exif_meta    = extract_exif(str(save_path))
    verdict      = build_verdict(c2pa_result, ai_analysis, exif_meta)

    # Persist
    DB.execute("""
        INSERT INTO scans
            (filename, file_hash, file_size, verdict, confidence,
             has_c2pa, c2pa_trusted, sig_issuer, generator_tool,
             source_type, action_recorded, scanned_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        filename, file_hash, file_size,
        verdict["status"], verdict["confidence"],
        int(c2pa_result.get("has_manifest", False)),
        int(c2pa_result.get("trusted", False)),
        verdict.get("issuer", ""),
        verdict.get("generator", ""),
        verdict.get("source_type", ""),
        ai_analysis.get("ai_action", ""),
        datetime.utcnow().isoformat(),
    ))
    DB.commit()

    # ── Format markdown result ────────────────────────────────────────────────
    md = f"## {verdict['headline']}\n\n"
    md += f"**Confidence:** `{verdict['confidence'].upper()}`  \n"
    md += f"**File:** `{filename}` ({file_size:,} bytes)  \n"
    md += f"**SHA-256:** `{file_hash[:20]}...`\n\n"

    if verdict.get("generator"):
        md += f"**Generator:** {verdict['generator']}  \n"
    if verdict.get("issuer"):
        md += f"**Certificate Issuer:** {verdict['issuer']}  \n"
    if verdict.get("source_type"):
        short = verdict["source_type"].rstrip("/").split("/")[-1]
        md += f"**Digital Source Type:** `{short}`  \n"
    if verdict.get("validation"):
        md += f"**C2PA Validation:** `{verdict['validation']}`  \n"

    md += "\n### Detection Signals\n"
    for detail in verdict["details"]:
        md += f"- {detail}\n"

    # C2PA display
    c2pa_display = {}
    if c2pa_result.get("manifest_store"):
        # Strip the bulky ocspVals from display
        store = json.loads(json.dumps(c2pa_result["manifest_store"]))
        for mkey, mval in store.get("manifests", {}).items():
            for assertion in mval.get("assertions", []):
                if assertion.get("label") == "c2pa.certificate-status":
                    assertion["data"] = {"ocspVals": ["<omitted for display>"] }
        c2pa_display = store

    return md, exif_meta, c2pa_display, get_history()


def get_history():
    rows = DB.execute("""
        SELECT id, filename, verdict, confidence, has_c2pa,
               c2pa_trusted, sig_issuer, generator_tool, scanned_at
        FROM scans ORDER BY id DESC LIMIT 50
    """).fetchall()
    return [dict(r) for r in rows]


# =========================
# GRADIO UI
# =========================

CSS = """
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=DM+Sans:ital,wght@0,300;0,400;0,600;1,400&display=swap');

:root {
    --bg:       #06070d;
    --surface:  #0d0e1a;
    --border:   #1a1b2e;
    --accent:   #7c6eff;
    --green:    #23d18b;
    --red:      #ff4f70;
    --yellow:   #ffd166;
    --text:     #d8d9f0;
    --muted:    #4a4b6a;
    --radius:   6px;
    --mono:     'IBM Plex Mono', monospace;
    --sans:     'DM Sans', sans-serif;
}

body, .gradio-container { background: var(--bg) !important; font-family: var(--sans) !important; color: var(--text) !important; }
h1,h2,h3,h4 { font-family: var(--mono) !important; letter-spacing: -0.02em; }
.block { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: var(--radius) !important; }
label span { font-family: var(--mono) !important; font-size: 0.72rem !important; color: var(--muted) !important; text-transform: uppercase; letter-spacing: 0.1em; }
button.primary { background: var(--accent) !important; color: #fff !important; font-family: var(--mono) !important; font-weight: 600 !important; border: none !important; border-radius: 4px !important; letter-spacing: 0.04em; transition: box-shadow 0.2s; }
button.primary:hover { box-shadow: 0 0 24px rgba(124,110,255,0.5) !important; }
button.secondary { font-family: var(--mono) !important; font-size: 0.8rem !important; color: var(--muted) !important; border: 1px solid var(--border) !important; background: transparent !important; }
.tab-nav button { font-family: var(--mono) !important; font-size: 0.75rem !important; color: var(--muted) !important; }
.tab-nav button.selected { color: var(--accent) !important; border-bottom: 2px solid var(--accent) !important; }
textarea, input { font-family: var(--mono) !important; background: var(--bg) !important; color: var(--text) !important; border-color: var(--border) !important; }
"""

INTRO = """
# C2PA AI Content Detector

Detects AI-generated content using the **C2PA (Coalition for Content Provenance and Authenticity)** standard —
the same cryptographic provenance system used by LinkedIn, Instagram, and TikTok.

Reads the signed manifest embedded by tools like **OpenAI gpt-image / DALL·E 3 · Adobe Firefly · Google Imagen · Microsoft Copilot**.
No classifiers. No guessing. Pure cryptographic provenance.
"""

with gr.Blocks(title="C2PA AI Detector", css=CSS) as demo:
    gr.Markdown(INTRO)

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Upload Image / Video",
                file_types=[".jpg",".jpeg",".png",".webp",".avif",".mp4",".mov",".tiff",".gif"],
            )
            scan_btn = gr.Button("🔍  Scan for AI Content", variant="primary", size="lg")
            gr.Markdown("""
**Formats:** JPEG · PNG · WebP · AVIF · MP4 · MOV · TIFF · GIF

**Note:** Files without C2PA credentials are not proven human-made — many AI tools don't embed them yet.
            """)

        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab("🎯 Result"):
                    result_md = gr.Markdown("*Upload a file and click **Scan** to begin.*")

                with gr.Tab("📋 C2PA Manifest"):
                    c2pa_output = gr.JSON(label="Parsed Manifest Store")

                with gr.Tab("🗂️ EXIF Metadata"):
                    exif_output = gr.JSON(label="Raw EXIF / Image Metadata")

                with gr.Tab("🕓 History"):
                    history_output = gr.JSON(label="Recent Scans (last 50)")
                    refresh_btn = gr.Button("↻ Refresh", variant="secondary")

    scan_btn.click(
        fn=process_image,
        inputs=[file_input],
        outputs=[result_md, exif_output, c2pa_output, history_output],
    )
    refresh_btn.click(fn=get_history, outputs=[history_output])

if __name__ == "__main__":
    demo.launch(server_port=7860, show_error=True)