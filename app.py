"""
VitalIA — Dr. Digital
Análisis inteligente de exámenes médicos con IA
"""
import os, sys, json, sqlite3, base64, re, uuid, threading, smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from datetime import datetime
from functools import wraps

# ── In-memory job store ───────────────────────────────────────────────────────
_jobs: dict = {}   # job_id → {"status": "pending|done|error", "result": {...}}
_jobs_lock = threading.Lock()

from flask import (Flask, render_template, request, jsonify,
                   Response, stream_with_context, redirect, url_for, session)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from google import genai
from google.genai import types as genai_types
import PIL.Image
import stripe

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent

def _resolve_data_dir() -> Path:
    """Elige el directorio de datos: /data (Render disk), /tmp (Render free), o local."""
    for candidate in [os.environ.get("RENDER_DATA_DIR"), "/data", str(BASE_DIR)]:
        if not candidate:
            continue
        p = Path(candidate)
        try:
            p.mkdir(parents=True, exist_ok=True)
            test = p / ".write_test"
            test.write_text("ok")
            test.unlink()
            return p
        except Exception:
            continue
    return Path("/tmp")

DATA_DIR   = _resolve_data_dir()
DB_PATH    = DATA_DIR / "vitalia.db"
UPLOAD_DIR = DATA_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
print(f"[VitalIA] DATA_DIR={DATA_DIR}")
PORT         = int(os.environ.get("PORT", 5002))
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY","") or os.environ.get("GEMINI_API_KEY","")
ALLOWED_EXT  = {".jpg", ".jpeg", ".png", ".webp", ".pdf"}

# Modelos candidatos en orden de preferencia (más nuevos primero)
_GEMINI_CANDIDATES = [
    "gemini-2.5-flash-preview-04-17",
    "gemini-2.5-pro-preview-03-25",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash-latest",
    "gemini-1.5-pro-latest",
]

def _detect_gemini_model(api_key: str) -> str:
    """Prueba cada modelo con una llamada real hasta encontrar uno que funcione."""
    forced = os.environ.get("GEMINI_MODEL","")
    if forced:
        print(f"[VitalIA] Modelo forzado: {forced}")
        return forced
    if not api_key:
        return _GEMINI_CANDIDATES[0]
    gc = genai.Client(api_key=api_key, http_options={"api_version": "v1"})
    for candidate in _GEMINI_CANDIDATES:
        try:
            gc.models.generate_content(model=candidate, contents="hola")
            print(f"[VitalIA] Modelo activo: {candidate}")
            return candidate
        except Exception as e:
            msg = str(e)
            if ("no longer available" in msg or "NOT_FOUND" in msg
                    or "not found" in msg.lower() or "UNAVAILABLE" in msg
                    or "503" in msg or "high demand" in msg):
                continue   # probar siguiente
            # Otro error (cuota, auth) — detener búsqueda
            print(f"[VitalIA] Error al probar {candidate}: {msg[:80]}")
            break
    print("[VitalIA] Usando candidato por defecto")
    return "gemini-2.0-flash"

GEMINI_MODEL = _detect_gemini_model(GOOGLE_API_KEY)
MAX_MB = 20

# Stripe
STRIPE_SECRET_KEY      = os.environ.get("STRIPE_SECRET_KEY", "")
STRIPE_PUBLISHABLE_KEY = os.environ.get("STRIPE_PUBLISHABLE_KEY", "")
STRIPE_WEBHOOK_SECRET  = os.environ.get("STRIPE_WEBHOOK_SECRET", "")
EXAM_PRICE_CENTS       = 300  # $3.00 USD

# Admin
ADMIN_EMAIL    = os.environ.get("ADMIN_EMAIL", "admin@vitalia.app")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "VitalIA#Admin2025")

if STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("SECRET_KEY", "vitalia-secret-2025-CHANGE-IN-PROD")
app.config["MAX_CONTENT_LENGTH"] = MAX_MB * 1024 * 1024

# ── DB ────────────────────────────────────────────────────────────────────────
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn

def init_db():
    with get_db() as db:
        db.executescript("""
        -- ── Usuarios ─────────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS usuarios (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            email        TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            nombre       TEXT NOT NULL,
            creado_en    TEXT DEFAULT (datetime('now','localtime')),
            activo       INTEGER DEFAULT 1
        );

        -- ── Pacientes (uno por usuario) ───────────────────────────────────────
        CREATE TABLE IF NOT EXISTS pacientes (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario_id  INTEGER NOT NULL,
            nombre      TEXT NOT NULL DEFAULT 'Paciente',
            fecha_nac   TEXT DEFAULT '',
            genero      TEXT DEFAULT '',
            email       TEXT DEFAULT '',
            notas       TEXT DEFAULT '',
            creado_en   TEXT DEFAULT (datetime('now','localtime')),
            FOREIGN KEY (usuario_id) REFERENCES usuarios(id)
        );

        -- ── Exámenes ──────────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS examenes (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            paciente_id     INTEGER NOT NULL,
            titulo          TEXT NOT NULL,
            tipo            TEXT DEFAULT 'sangre',
            fecha_examen    TEXT DEFAULT '',
            laboratorio     TEXT DEFAULT '',
            imagen_path     TEXT DEFAULT '',
            texto_extraido  TEXT DEFAULT '',
            interpretacion  TEXT DEFAULT '',
            resumen         TEXT DEFAULT '',
            estado          TEXT DEFAULT 'pendiente',
            riesgo          TEXT DEFAULT 'normal',
            creado_en       TEXT DEFAULT (datetime('now','localtime')),
            FOREIGN KEY (paciente_id) REFERENCES pacientes(id)
        );

        -- ── Indicadores ───────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS indicadores (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            examen_id   INTEGER NOT NULL,
            nombre      TEXT NOT NULL,
            valor       TEXT NOT NULL,
            unidad      TEXT DEFAULT '',
            rango_ref   TEXT DEFAULT '',
            rango_min   REAL,
            rango_max   REAL,
            estado      TEXT DEFAULT 'normal',
            descripcion TEXT DEFAULT '',
            FOREIGN KEY (examen_id) REFERENCES examenes(id) ON DELETE CASCADE
        );

        -- ── Conversaciones ────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS conversaciones (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            examen_id   INTEGER,
            titulo      TEXT DEFAULT 'Consulta médica',
            creado_en   TEXT DEFAULT (datetime('now','localtime')),
            FOREIGN KEY (examen_id) REFERENCES examenes(id)
        );

        -- ── Mensajes ──────────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS mensajes (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            conversacion_id     INTEGER NOT NULL,
            rol                 TEXT NOT NULL,
            contenido           TEXT NOT NULL,
            creado_en           TEXT DEFAULT (datetime('now','localtime')),
            FOREIGN KEY (conversacion_id) REFERENCES conversaciones(id) ON DELETE CASCADE
        );

        -- ── Recomendaciones ───────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS recomendaciones (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            examen_id   INTEGER NOT NULL,
            tipo        TEXT DEFAULT 'general',
            titulo      TEXT NOT NULL,
            descripcion TEXT NOT NULL,
            prioridad   TEXT DEFAULT 'media',
            creado_en   TEXT DEFAULT (datetime('now','localtime')),
            FOREIGN KEY (examen_id) REFERENCES examenes(id) ON DELETE CASCADE
        );

        -- ── Pagos ─────────────────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS pagos (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario_id              INTEGER NOT NULL,
            stripe_payment_intent_id TEXT UNIQUE NOT NULL,
            monto                   INTEGER NOT NULL,
            moneda                  TEXT DEFAULT 'usd',
            estado                  TEXT DEFAULT 'pendiente',
            examen_id               INTEGER,
            creado_en               TEXT DEFAULT (datetime('now','localtime')),
            FOREIGN KEY (usuario_id) REFERENCES usuarios(id),
            FOREIGN KEY (examen_id) REFERENCES examenes(id)
        );
        """)
        db.commit()

    # ── Migraciones ───────────────────────────────────────────────────────────
    with get_db() as db:
        # pacientes.usuario_id
        cols = [r[1] for r in db.execute("PRAGMA table_info(pacientes)").fetchall()]
        if "usuario_id" not in cols:
            db.execute("ALTER TABLE pacientes ADD COLUMN usuario_id INTEGER DEFAULT 0")
            db.commit()
        # usuarios.es_admin
        ucols = [r[1] for r in db.execute("PRAGMA table_info(usuarios)").fetchall()]
        if "es_admin" not in ucols:
            db.execute("ALTER TABLE usuarios ADD COLUMN es_admin INTEGER DEFAULT 0")
            db.commit()
        # examenes.preguntas_doctor
        ecols = [r[1] for r in db.execute("PRAGMA table_info(examenes)").fetchall()]
        if "preguntas_doctor" not in ecols:
            db.execute("ALTER TABLE examenes ADD COLUMN preguntas_doctor TEXT DEFAULT '[]'")
            db.commit()

    # ── Superusuario admin ────────────────────────────────────────────────────
    _ensure_admin()

def _ensure_admin():
    """Crea o actualiza el superusuario admin en cada arranque."""
    pwd_hash = generate_password_hash(ADMIN_PASSWORD)
    with get_db() as db:
        existing = db.execute("SELECT id FROM usuarios WHERE email=?", (ADMIN_EMAIL,)).fetchone()
        if existing:
            # Actualizar contraseña y garantizar flag admin
            db.execute(
                "UPDATE usuarios SET password_hash=?, es_admin=1, activo=1 WHERE email=?",
                (pwd_hash, ADMIN_EMAIL)
            )
            admin_id = existing["id"]
        else:
            cur = db.execute(
                "INSERT INTO usuarios (email, password_hash, nombre, es_admin) VALUES (?,?,?,1)",
                (ADMIN_EMAIL, pwd_hash, "Administrador VitalIA")
            )
            admin_id = cur.lastrowid
        db.commit()

        # Crear perfil de paciente admin si no existe
        pac = db.execute("SELECT id FROM pacientes WHERE usuario_id=?", (admin_id,)).fetchone()
        if not pac:
            db.execute(
                "INSERT INTO pacientes (usuario_id, nombre, email) VALUES (?,?,?)",
                (admin_id, "Administrador VitalIA", ADMIN_EMAIL)
            )
            db.commit()

init_db()

# ── Helpers de autenticación ──────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if "usuario_id" not in session:
            if request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return jsonify({"error": "No autenticado", "login_required": True}), 401
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated

def get_current_user():
    uid = session.get("usuario_id")
    if not uid:
        return None
    with get_db() as db:
        return db.execute("SELECT * FROM usuarios WHERE id=?", (uid,)).fetchone()

def is_admin():
    return bool(session.get("es_admin"))

def get_current_paciente():
    uid = session.get("usuario_id")
    if not uid:
        return None
    with get_db() as db:
        pac = db.execute("SELECT * FROM pacientes WHERE usuario_id=?", (uid,)).fetchone()
        if not pac:
            # Auto-crear paciente si no existe (p.ej. tras reinicio de DB en Render)
            usuario = db.execute("SELECT * FROM usuarios WHERE id=?", (uid,)).fetchone()
            if not usuario:
                return None
            db.execute(
                "INSERT INTO pacientes (usuario_id, nombre, email) VALUES (?,?,?)",
                (uid, usuario["nombre"], usuario["email"])
            )
            db.commit()
            pac = db.execute("SELECT * FROM pacientes WHERE usuario_id=?", (uid,)).fetchone()
        return dict(pac) if pac else None

# ── Error handlers ────────────────────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e): return jsonify({"error": "No encontrado"}), 404
@app.errorhandler(500)
def internal(e): return jsonify({"error": str(e)}), 500
@app.errorhandler(413)
def too_large(e): return jsonify({"error": f"Archivo muy grande (máx {MAX_MB}MB)"}), 413
@app.errorhandler(Exception)
def handle_exc(e):
    import traceback; print(traceback.format_exc())
    return jsonify({"error": str(e)}), 500

# ── Cache-control ─────────────────────────────────────────────────────────────
@app.after_request
def no_cache(r):
    r.headers["Cache-Control"] = "no-store"
    return r

# ── Helpers ───────────────────────────────────────────────────────────────────
def _google_key():
    return (os.environ.get("GOOGLE_API_KEY","") or
            os.environ.get("GEMINI_API_KEY","") or
            request.headers.get("X-API-Key",""))

def get_client():
    """Retorna la API key de Google validada."""
    key = _google_key()
    if not key:
        raise ValueError("GOOGLE_API_KEY no configurada.")
    return key

def _gemini_client(api_key):
    return genai.Client(api_key=api_key, http_options={"api_version": "v1"})

def _generate(gc, contents):
    """generate_content con fallback automático ante 503/NOT_FOUND."""
    last_err = None
    for model in _GEMINI_CANDIDATES:
        try:
            return gc.models.generate_content(model=model, contents=contents)
        except Exception as e:
            msg = str(e)
            if ("UNAVAILABLE" in msg or "503" in msg or "high demand" in msg
                    or "NOT_FOUND" in msg or "not found" in msg.lower()):
                last_err = e
                continue
            raise
    raise last_err

def estado_indicador(valor_str, rango_min, rango_max):
    try:
        v = float(re.sub(r"[^\d.\-]", "", str(valor_str)))
        if rango_min is not None and v < rango_min: return "bajo"
        if rango_max is not None and v > rango_max: return "alto"
        return "normal"
    except: return "normal"

SYSTEM_DOCTOR = """Eres VitalIA, un asistente médico digital altamente capacitado y empático.
Tu misión es interpretar exámenes médicos, explicar indicadores de salud y orientar al paciente.

PERSONALIDAD:
- Empático, claro y profesional como un médico de confianza
- Explica en términos simples, sin jerga médica innecesaria
- Siempre validas y reconoces las preocupaciones del paciente
- Eres proactivo en recomendar consulta presencial cuando es necesario

CAPACIDADES:
- Interpretar exámenes de sangre, orina, imágenes y otros
- Explicar qué significan los valores fuera de rango
- Sugerir cambios de estilo de vida, alimentación y ejercicio
- Orientar sobre medicamentos (uso general, no prescribes)
- Reconocer señales de alerta que requieren atención urgente

LIMITACIONES IMPORTANTES:
- Siempre aclaras que NO reemplazas la consulta médica presencial
- No prescribes medicamentos específicos ni dosis
- Ante síntomas graves o valores críticos, derivas urgentemente al médico
- Tus respuestas son orientativas, no diagnósticos definitivos

FORMATO DE RESPUESTAS:
- Usa markdown para estructurar la información
- Emplea emojis médicos para mejor legibilidad (🩺 💊 🏃 🥗 ⚠️ ✅)
- Sé conciso pero completo
- Termina siempre con una nota de aliento o próximo paso sugerido

IDIOMA: Responde siempre en español."""

# ── Health check ─────────────────────────────────────────────────────────────
@app.route("/health")
def health():
    key = os.environ.get("GOOGLE_API_KEY","") or os.environ.get("GEMINI_API_KEY","")
    return jsonify({
        "status": "ok",
        "python": sys.version,
        "data_dir": str(DATA_DIR),
        "db_exists": DB_PATH.exists(),
        "upload_dir": str(UPLOAD_DIR),
        "upload_writable": os.access(str(UPLOAD_DIR), os.W_OK),
        "api_key_set": bool(key),
        "gemini_model": GEMINI_MODEL,
        "stripe_configured": bool(STRIPE_SECRET_KEY),
        "resend_configured": bool(os.environ.get("RESEND_API_KEY")),
    })

# ── Email ────────────────────────────────────────────────────────────────────
def _enviar_email_examen(destinatario: str, nombre_paciente: str,
                          examen: dict, indicadores: list,
                          recomendaciones: list, preguntas: list) -> None:
    resend_key = os.environ.get("RESEND_API_KEY", "")
    if not resend_key:
        raise ValueError("RESEND_API_KEY no configurada.")

    estado_cfg = {
        "normal":  {"color":"#10b981","bg":"rgba(16,185,129,0.15)","icon":"✅","label":"Normal"},
        "alto":    {"color":"#ef4444","bg":"rgba(239,68,68,0.15)","icon":"🔴","label":"Alto"},
        "bajo":    {"color":"#f59e0b","bg":"rgba(245,158,11,0.15)","icon":"🟡","label":"Bajo"},
        "critico": {"color":"#ef4444","bg":"rgba(239,68,68,0.2)","icon":"🚨","label":"Crítico"},
    }
    rows_ind = ""
    for ind in indicadores:
        cfg = estado_cfg.get(ind.get("estado","normal"), estado_cfg["normal"])
        barra = ""
        try:
            v = float(re.sub(r"[^\d.\-]","", str(ind.get("valor",""))))
            mn = ind.get("rango_min"); mx = ind.get("rango_max")
            if mn is not None and mx is not None and float(mx) > float(mn):
                pct = max(0, min(100, (v - float(mn)) / (float(mx) - float(mn)) * 100))
                barra = f'<div style="margin-top:5px;height:5px;background:rgba(255,255,255,0.08);border-radius:3px;"><div style="height:5px;width:{pct:.0f}%;background:{cfg["color"]};border-radius:3px;"></div></div>'
        except Exception:
            pass
        rows_ind += f"""<tr>
          <td style="padding:11px 14px;border-bottom:1px solid rgba(255,255,255,0.06);color:#f1f5f9;font-weight:600;font-size:13px;">{ind.get('nombre','')} {barra}</td>
          <td style="padding:11px 14px;border-bottom:1px solid rgba(255,255,255,0.06);font-family:monospace;color:#38bdf8;font-weight:700;">{ind.get('valor','')} <span style="color:#64748b;font-size:11px;">{ind.get('unidad','')}</span></td>
          <td style="padding:11px 14px;border-bottom:1px solid rgba(255,255,255,0.06);color:#64748b;font-size:12px;">{ind.get('rango_ref','—')}</td>
          <td style="padding:11px 14px;border-bottom:1px solid rgba(255,255,255,0.06);"><span style="background:{cfg['bg']};color:{cfg['color']};padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700;">{cfg['icon']} {cfg['label']}</span></td>
          <td style="padding:11px 14px;border-bottom:1px solid rgba(255,255,255,0.06);color:#64748b;font-size:12px;">{ind.get('descripcion','')}</td>
        </tr>"""

    rec_icons = {"dieta":"🥗","ejercicio":"🏃","consulta":"👨‍⚕️","medicamento":"💊","estilo_vida":"🌟","general":"📋"}
    prio_cfg  = {"alta":("#ef4444","rgba(239,68,68,0.12)"),"media":("#f59e0b","rgba(245,158,11,0.12)"),"baja":("#10b981","rgba(16,185,129,0.12)")}
    recs_html = ""
    for rec in recomendaciones:
        icon = rec_icons.get(rec.get("tipo","general"),"📋")
        pc, pb = prio_cfg.get(rec.get("prioridad","media"), prio_cfg["media"])
        recs_html += f"""<div style="display:flex;gap:12px;padding:14px;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:12px;margin-bottom:10px;">
          <div style="width:40px;height:40px;border-radius:10px;background:{pb};display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0;">{icon}</div>
          <div><div style="font-weight:700;font-size:13px;color:#f1f5f9;margin-bottom:4px;">{rec.get('titulo','')} <span style="background:{pb};color:{pc};padding:2px 8px;border-radius:10px;font-size:10px;font-weight:700;margin-left:4px;">{rec.get('prioridad','').upper()}</span></div>
          <div style="font-size:13px;color:#94a3b8;line-height:1.6;">{rec.get('descripcion','')}</div></div></div>"""

    preguntas_html = ""
    if preguntas:
        items = "".join([f'<li style="padding:11px 14px;background:rgba(108,99,255,0.08);border:1px solid rgba(108,99,255,0.2);border-radius:10px;margin-bottom:8px;font-size:13px;color:#e2e8f0;line-height:1.6;">🔹 {p}</li>' for p in preguntas])
        preguntas_html = f"""<div style="margin-bottom:32px;">
          <h2 style="font-size:15px;font-weight:700;color:#a78bfa;margin:0 0 8px;">🩺 Preguntas para tu médico tratante</h2>
          <p style="font-size:13px;color:#64748b;margin:0 0 14px;">Lleva esta lista a tu próxima consulta:</p>
          <ol style="list-style:none;padding:0;margin:0;">{items}</ol></div>"""

    riesgo = examen.get("riesgo","normal")
    riesgo_cfg = {
        "normal":  ("#10b981","rgba(16,185,129,0.1)","rgba(16,185,129,0.3)","✅ Todo en orden"),
        "medio":   ("#f59e0b","rgba(245,158,11,0.1)", "rgba(245,158,11,0.3)", "⚠️ Atención recomendada"),
        "alto":    ("#ef4444","rgba(239,68,68,0.1)",  "rgba(239,68,68,0.3)",  "🔴 Consulta médica sugerida"),
        "critico": ("#ef4444","rgba(239,68,68,0.15)", "rgba(239,68,68,0.5)",  "🚨 Consulta urgente"),
    }
    rc, rb, rborder, rtxt = riesgo_cfg.get(riesgo, riesgo_cfg["normal"])
    fecha_reporte = datetime.now().strftime("%d/%m/%Y %H:%M")

    html = f"""<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8">
<title>Reporte VitalIA</title></head>
<body style="margin:0;padding:0;background:#0a0f1e;font-family:'Segoe UI',Arial,sans-serif;">
<div style="max-width:660px;margin:0 auto;padding:24px 16px;">

  <!-- Header -->
  <div style="background:linear-gradient(135deg,#1e293b,#0f172a);border:1px solid rgba(255,255,255,0.08);border-radius:20px;padding:28px;text-align:center;margin-bottom:18px;">
    <div style="width:60px;height:60px;background:linear-gradient(135deg,#6c63ff,#a78bfa);border-radius:16px;display:inline-flex;align-items:center;justify-content:center;font-size:30px;margin-bottom:14px;box-shadow:0 0 30px rgba(108,99,255,0.4);">🩺</div>
    <h1 style="font-size:26px;font-weight:800;color:#a78bfa;margin:0 0 6px;">VitalIA</h1>
    <p style="color:#94a3b8;font-size:14px;margin:0;">Reporte de análisis médico con IA</p>
    <p style="color:#475569;font-size:12px;margin-top:6px;">{fecha_reporte}</p>
  </div>

  <!-- Saludo -->
  <div style="background:rgba(108,99,255,0.08);border:1px solid rgba(108,99,255,0.2);border-radius:14px;padding:18px 22px;margin-bottom:18px;">
    <p style="color:#e2e8f0;font-size:15px;margin:0;line-height:1.7;">Hola <strong style="color:#a78bfa;">{nombre_paciente}</strong>, aquí está el análisis completo de tu examen <strong style="color:#f1f5f9;">{examen.get('titulo','')}</strong>.</p>
  </div>

  <!-- Banner riesgo -->
  <div style="background:{rb};border:1px solid {rborder};border-radius:14px;padding:16px 20px;margin-bottom:18px;">
    <div style="font-weight:700;font-size:15px;color:{rc};">{rtxt}</div>
    <div style="font-size:13px;color:#94a3b8;margin-top:3px;">Nivel de riesgo general: <strong style="color:{rc};">{riesgo.upper()}</strong></div>
  </div>

  <!-- Info examen -->
  <div style="background:#0f172a;border:1px solid rgba(255,255,255,0.07);border-radius:14px;padding:18px 22px;margin-bottom:18px;display:flex;gap:24px;flex-wrap:wrap;">
    {"<div><div style='font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:.8px;margin-bottom:3px;'>Tipo</div><div style='font-weight:600;color:#f1f5f9;font-size:14px;'>" + examen.get('tipo','') + "</div></div>" if examen.get('tipo') else ''}
    {"<div><div style='font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:.8px;margin-bottom:3px;'>Fecha examen</div><div style='font-weight:600;color:#f1f5f9;font-size:14px;'>" + examen.get('fecha_examen','') + "</div></div>" if examen.get('fecha_examen') else ''}
    {"<div><div style='font-size:10px;color:#475569;text-transform:uppercase;letter-spacing:.8px;margin-bottom:3px;'>Laboratorio</div><div style='font-weight:600;color:#f1f5f9;font-size:14px;'>" + examen.get('laboratorio','') + "</div></div>" if examen.get('laboratorio') else ''}
  </div>

  {"<div style='background:rgba(14,165,233,0.08);border:1px solid rgba(14,165,233,0.2);border-radius:14px;padding:18px 22px;margin-bottom:18px;'><h2 style='font-size:14px;font-weight:700;color:#38bdf8;margin:0 0 10px;'>📋 Resumen del Dr. VitalIA</h2><p style='font-size:14px;color:#e2e8f0;line-height:1.7;margin:0;'>" + examen.get('resumen','') + "</p></div>" if examen.get('resumen') else ''}

  <!-- Tabla indicadores -->
  <div style="margin-bottom:28px;">
    <h2 style="font-size:15px;font-weight:700;color:#f1f5f9;margin:0 0 14px;">📊 Indicadores detectados ({len(indicadores)})</h2>
    <div style="overflow-x:auto;border-radius:14px;border:1px solid rgba(255,255,255,0.07);">
      <table style="width:100%;border-collapse:collapse;background:#0f172a;">
        <thead><tr style="background:rgba(255,255,255,0.04);">
          <th style="padding:10px 14px;text-align:left;font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:.8px;border-bottom:1px solid rgba(255,255,255,0.07);">Indicador</th>
          <th style="padding:10px 14px;text-align:left;font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:.8px;border-bottom:1px solid rgba(255,255,255,0.07);">Valor</th>
          <th style="padding:10px 14px;text-align:left;font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:.8px;border-bottom:1px solid rgba(255,255,255,0.07);">Referencia</th>
          <th style="padding:10px 14px;text-align:left;font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:.8px;border-bottom:1px solid rgba(255,255,255,0.07);">Estado</th>
          <th style="padding:10px 14px;text-align:left;font-size:11px;color:#475569;text-transform:uppercase;letter-spacing:.8px;border-bottom:1px solid rgba(255,255,255,0.07);">¿Qué mide?</th>
        </tr></thead>
        <tbody>{rows_ind}</tbody>
      </table>
    </div>
  </div>

  {"<div style='margin-bottom:28px;'><h2 style='font-size:15px;font-weight:700;color:#f1f5f9;margin:0 0 14px;'>💡 Recomendaciones</h2>" + recs_html + "</div>" if recs_html else ''}

  {preguntas_html}

  <!-- Footer -->
  <div style="background:#0f172a;border:1px solid rgba(255,255,255,0.06);border-radius:14px;padding:18px 22px;text-align:center;">
    <p style="font-size:12px;color:#475569;line-height:1.8;margin:0;">
      ⚠️ Este reporte es <strong style="color:#64748b;">orientativo</strong> y no reemplaza la consulta médica presencial.<br>
      Ante cualquier síntoma grave, consulta a un profesional de salud.<br><br>
      <span style="color:#334155;">Generado por <strong style="color:#6c63ff;">VitalIA — Dr. Digital</strong> · {fecha_reporte}</span>
    </p>
  </div>

</div></body></html>"""

    import urllib.request, urllib.error
    resend_from = os.environ.get("RESEND_FROM", "onboarding@resend.dev")
    payload = json.dumps({
        "from": f"VitalIA Dr. Digital <{resend_from}>",
        "to": [destinatario],
        "subject": f"VitalIA — Reporte: {examen.get('titulo','Examen médico')}",
        "html": html,
    }).encode("utf-8")
    req = urllib.request.Request(
        "https://api.resend.com/emails",
        data=payload,
        headers={"Authorization": f"Bearer {resend_key}", "Content-Type": "application/json"},
        method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = json.loads(resp.read())
            print(f"[VitalIA] Email enviado via Resend: {result}")
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        raise Exception(f"Resend {e.code}: {body}")


# ── Rutas: Autenticación ──────────────────────────────────────────────────────
@app.route("/register", methods=["GET","POST"])
def register():
    if "usuario_id" in session:
        return redirect(url_for("app_main"))

    if request.method == "POST":
        d = request.json or request.form
        nombre = (d.get("nombre","")).strip()
        email  = (d.get("email","")).strip().lower()
        pwd    = d.get("password","")
        pwd2   = d.get("password2","")

        # Validaciones
        if not nombre or not email or not pwd:
            return jsonify({"error": "Todos los campos son obligatorios"}), 400
        if len(pwd) < 8:
            return jsonify({"error": "La contraseña debe tener al menos 8 caracteres"}), 400
        if pwd != pwd2:
            return jsonify({"error": "Las contraseñas no coinciden"}), 400
        if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
            return jsonify({"error": "Email inválido"}), 400

        try:
            with get_db() as db:
                # Crear usuario
                cur = db.execute(
                    "INSERT INTO usuarios (email, password_hash, nombre) VALUES (?,?,?)",
                    (email, generate_password_hash(pwd), nombre)
                )
                uid = cur.lastrowid
                # Crear perfil de paciente asociado
                db.execute(
                    "INSERT INTO pacientes (usuario_id, nombre, email) VALUES (?,?,?)",
                    (uid, nombre, email)
                )
                db.commit()

            session["usuario_id"]     = uid
            session["usuario_nombre"] = nombre
            session["usuario_email"]  = email
            session["es_admin"]       = False
            return jsonify({"ok": True, "redirect": "/app"})
        except sqlite3.IntegrityError:
            return jsonify({"error": "Ya existe una cuenta con ese email"}), 409

    return render_template("register.html")


@app.route("/login", methods=["GET","POST"])
def login():
    if "usuario_id" in session:
        return redirect(url_for("app_main"))

    if request.method == "POST":
        d = request.json or request.form
        email = (d.get("email","")).strip().lower()
        pwd   = d.get("password","")

        if not email or not pwd:
            return jsonify({"error": "Email y contraseña son obligatorios"}), 400

        with get_db() as db:
            u = db.execute("SELECT * FROM usuarios WHERE email=? AND activo=1", (email,)).fetchone()

        if not u or not check_password_hash(u["password_hash"], pwd):
            return jsonify({"error": "Email o contraseña incorrectos"}), 401

        session["usuario_id"]     = u["id"]
        session["usuario_nombre"] = u["nombre"]
        session["usuario_email"]  = u["email"]
        session["es_admin"]       = bool(u["es_admin"])
        return jsonify({"ok": True, "redirect": "/app"})

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("landing"))


# ── Ruta principal ────────────────────────────────────────────────────────────
@app.route("/")
def landing():
    return render_template("landing.html")


@app.route("/app")
@login_required
def app_main():
    paciente = get_current_paciente()
    if not paciente:
        session.clear()
        return redirect(url_for("login"))
    with get_db() as db:
        examenes = db.execute("""
            SELECT e.*, COUNT(i.id) as num_indicadores,
                   SUM(CASE WHEN i.estado != 'normal' THEN 1 ELSE 0 END) as alertas
            FROM examenes e
            LEFT JOIN indicadores i ON i.examen_id = e.id
            WHERE e.paciente_id = ?
            GROUP BY e.id ORDER BY e.creado_en DESC LIMIT 20
        """, (paciente["id"],)).fetchall()
    return render_template("app.html",
                           examenes=[dict(e) for e in examenes],
                           paciente=dict(paciente) if paciente else {},
                           api_key_ok=bool(os.environ.get("GOOGLE_API_KEY","") or os.environ.get("GEMINI_API_KEY","")),
                           stripe_pk=STRIPE_PUBLISHABLE_KEY,
                           stripe_configured=bool(STRIPE_SECRET_KEY),
                           usuario_nombre=session.get("usuario_nombre",""),
                           exam_price=EXAM_PRICE_CENTS,
                           es_admin=is_admin())


# ── API: Paciente ─────────────────────────────────────────────────────────────
@app.route("/api/paciente", methods=["GET","POST"])
@login_required
def paciente_endpoint():
    uid = session["usuario_id"]
    with get_db() as db:
        if request.method == "POST":
            d = request.json or {}
            db.execute("""UPDATE pacientes SET nombre=?,fecha_nac=?,genero=?,notas=?
                          WHERE usuario_id=?""",
                       (d.get("nombre",""), d.get("fecha_nac",""),
                        d.get("genero",""), d.get("notas",""), uid))
            db.commit()
            session["usuario_nombre"] = d.get("nombre", session.get("usuario_nombre",""))
            return jsonify({"ok": True})
        p = db.execute("SELECT * FROM pacientes WHERE usuario_id=?", (uid,)).fetchone()
        return jsonify(dict(p) if p else {})


# ── API: Exámenes ─────────────────────────────────────────────────────────────
@app.route("/api/examenes", methods=["GET"])
@login_required
def listar_examenes():
    paciente = get_current_paciente()
    if not paciente:
        return jsonify({"examenes": []})
    with get_db() as db:
        rows = db.execute("""
            SELECT e.*, COUNT(i.id) as num_indicadores,
                   SUM(CASE WHEN i.estado != 'normal' THEN 1 ELSE 0 END) as alertas
            FROM examenes e
            LEFT JOIN indicadores i ON i.examen_id = e.id
            WHERE e.paciente_id = ?
            GROUP BY e.id ORDER BY e.creado_en DESC
        """, (paciente["id"],)).fetchall()
    return jsonify({"examenes": [dict(r) for r in rows]})


@app.route("/api/examenes/<int:eid>", methods=["GET"])
@login_required
def get_examen(eid):
    paciente = get_current_paciente()
    if not paciente:
        return jsonify({"error": "No encontrado"}), 404
    with get_db() as db:
        e = db.execute("SELECT * FROM examenes WHERE id=? AND paciente_id=?",
                       (eid, paciente["id"])).fetchone()
        if not e: return jsonify({"error":"No encontrado"}), 404
        indicadores    = db.execute("SELECT * FROM indicadores WHERE examen_id=? ORDER BY id", (eid,)).fetchall()
        recomendaciones = db.execute("SELECT * FROM recomendaciones WHERE examen_id=? ORDER BY prioridad DESC", (eid,)).fetchall()
        conversaciones  = db.execute("SELECT * FROM conversaciones WHERE examen_id=? ORDER BY creado_en DESC", (eid,)).fetchall()
    examen_dict = dict(e)
    try:
        examen_dict["preguntas_doctor"] = json.loads(examen_dict.get("preguntas_doctor") or "[]")
    except Exception:
        examen_dict["preguntas_doctor"] = []
    return jsonify({
        "examen": examen_dict,
        "indicadores": [dict(i) for i in indicadores],
        "recomendaciones": [dict(r) for r in recomendaciones],
        "conversaciones": [dict(c) for c in conversaciones],
    })


@app.route("/api/examenes/<int:eid>/enviar-email", methods=["POST"])
@login_required
def enviar_email_examen(eid):
    paciente = get_current_paciente()
    if not paciente:
        return jsonify({"error": "Perfil no encontrado"}), 404
    with get_db() as db:
        e = db.execute("SELECT * FROM examenes WHERE id=? AND paciente_id=?",
                       (eid, paciente["id"])).fetchone()
        if not e:
            return jsonify({"error": "Examen no encontrado"}), 404
        indicadores     = db.execute("SELECT * FROM indicadores WHERE examen_id=? ORDER BY id", (eid,)).fetchall()
        recomendaciones = db.execute("SELECT * FROM recomendaciones WHERE examen_id=? ORDER BY prioridad DESC", (eid,)).fetchall()

    d = request.json or {}
    email_dest = (d.get("email","").strip()
                  or paciente.get("email","").strip()
                  or session.get("usuario_email","").strip())
    if not email_dest or not re.match(r"[^@]+@[^@]+\.[^@]+", email_dest):
        return jsonify({"error": "Email no válido o no configurado en tu perfil."}), 400

    if not os.environ.get("RESEND_API_KEY"):
        return jsonify({"error": "Email no configurado en el servidor.", "smtp_not_configured": True}), 503

    try:
        preguntas = json.loads(dict(e).get("preguntas_doctor") or "[]")
    except Exception:
        preguntas = []

    try:
        _enviar_email_examen(
            destinatario=email_dest,
            nombre_paciente=paciente.get("nombre","Paciente"),
            examen=dict(e),
            indicadores=[dict(i) for i in indicadores],
            recomendaciones=[dict(r) for r in recomendaciones],
            preguntas=preguntas,
        )
    except ValueError as ve:
        return jsonify({"error": str(ve), "smtp_not_configured": True}), 503
    except Exception as ex:
        import traceback
        print(f"[VitalIA] Error email: {traceback.format_exc()}")
        return jsonify({"error": str(ex)}), 500

    return jsonify({"ok": True, "enviado_a": email_dest})


@app.route("/api/examenes/<int:eid>", methods=["DELETE"])
@login_required
def eliminar_examen(eid):
    paciente = get_current_paciente()
    if not paciente:
        return jsonify({"error": "No encontrado"}), 404
    with get_db() as db:
        e = db.execute("SELECT id FROM examenes WHERE id=? AND paciente_id=?",
                       (eid, paciente["id"])).fetchone()
        if not e:
            return jsonify({"error": "No encontrado"}), 404
        db.execute("DELETE FROM examenes WHERE id=?", (eid,))
        db.commit()
    return jsonify({"ok": True})


# ── API: Pagos con Stripe ──────────────────────────────────────────────────────
@app.route("/api/crear-pago", methods=["POST"])
@login_required
def crear_pago():
    """Crea un PaymentIntent de Stripe por $3 USD y retorna el client_secret."""
    if not STRIPE_SECRET_KEY:
        # Modo desarrollo: simular pago exitoso
        dev_pi_id = f"pi_dev_{uuid.uuid4().hex}"
        uid = session["usuario_id"]
        with get_db() as db:
            db.execute("""INSERT INTO pagos (usuario_id, stripe_payment_intent_id, monto, estado)
                          VALUES (?,?,?,'completado')""",
                       (uid, dev_pi_id, EXAM_PRICE_CENTS))
            db.commit()
        return jsonify({
            "dev_mode": True,
            "payment_intent_id": dev_pi_id,
            "message": "Modo desarrollo: Stripe no configurado. Pago simulado."
        })

    try:
        uid = session["usuario_id"]
        email = session.get("usuario_email","")
        idempotency_key = f"vitalia-{uid}-{uuid.uuid4().hex}"

        intent = stripe.PaymentIntent.create(
            amount=EXAM_PRICE_CENTS,
            currency="usd",
            receipt_email=email or None,
            metadata={
                "usuario_id": str(uid),
                "app": "vitalia",
                "descripcion": "Análisis de examen médico VitalIA"
            },
            idempotency_key=idempotency_key,
        )

        with get_db() as db:
            db.execute("""INSERT INTO pagos (usuario_id, stripe_payment_intent_id, monto, estado)
                          VALUES (?,?,?,'pendiente')""",
                       (uid, intent.id, EXAM_PRICE_CENTS))
            db.commit()

        return jsonify({
            "client_secret": intent.client_secret,
            "payment_intent_id": intent.id,
            "amount": EXAM_PRICE_CENTS,
        })
    except stripe.error.StripeError as e:
        return jsonify({"error": str(e.user_message or e)}), 400


@app.route("/api/verificar-pago/<pi_id>", methods=["GET"])
@login_required
def verificar_pago(pi_id):
    """Verifica el estado de un PaymentIntent antes de procesar el examen."""
    uid = session["usuario_id"]

    with get_db() as db:
        pago = db.execute(
            "SELECT * FROM pagos WHERE stripe_payment_intent_id=? AND usuario_id=?",
            (pi_id, uid)
        ).fetchone()

    if not pago:
        return jsonify({"error": "Pago no encontrado"}), 404

    # En modo dev, ya está marcado como completado
    if pi_id.startswith("pi_dev_"):
        return jsonify({"estado": "completado", "ok": True})

    # Verificar con Stripe
    if not STRIPE_SECRET_KEY:
        return jsonify({"error": "Stripe no configurado"}), 400

    try:
        intent = stripe.PaymentIntent.retrieve(pi_id)
        estado_stripe = "completado" if intent.status == "succeeded" else intent.status

        if intent.status == "succeeded":
            with get_db() as db:
                db.execute(
                    "UPDATE pagos SET estado='completado' WHERE stripe_payment_intent_id=?",
                    (pi_id,)
                )
                db.commit()
            return jsonify({"estado": "completado", "ok": True})

        return jsonify({"estado": estado_stripe, "ok": False})
    except stripe.error.StripeError as e:
        return jsonify({"error": str(e)}), 400


# ── Webhook Stripe ────────────────────────────────────────────────────────────
@app.route("/stripe/webhook", methods=["POST"])
def stripe_webhook():
    """Webhook de Stripe para confirmar pagos de forma asíncrona."""
    payload = request.get_data()
    sig_header = request.headers.get("Stripe-Signature","")

    try:
        if STRIPE_WEBHOOK_SECRET:
            event = stripe.Webhook.construct_event(payload, sig_header, STRIPE_WEBHOOK_SECRET)
        else:
            event = json.loads(payload)
    except (ValueError, stripe.error.SignatureVerificationError) as e:
        return jsonify({"error": str(e)}), 400

    if event["type"] == "payment_intent.succeeded":
        pi = event["data"]["object"]
        with get_db() as db:
            db.execute(
                "UPDATE pagos SET estado='completado' WHERE stripe_payment_intent_id=?",
                (pi["id"],)
            )
            db.commit()
        print(f"[VitalIA] Pago confirmado: {pi['id']}")

    elif event["type"] == "payment_intent.payment_failed":
        pi = event["data"]["object"]
        with get_db() as db:
            db.execute(
                "UPDATE pagos SET estado='fallido' WHERE stripe_payment_intent_id=?",
                (pi["id"],)
            )
            db.commit()
        print(f"[VitalIA] Pago fallido: {pi['id']}")

    return jsonify({"received": True})


# ── API: OCR + Análisis ───────────────────────────────────────────────────────
def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _run_analisis(job_id, ruta, ext, titulo, tipo, api_key, paciente_id):
    """Corre el análisis completo en background y guarda resultado en _jobs."""
    def update(status, **kw):
        with _jobs_lock:
            _jobs[job_id] = {"status": status, **kw}

    try:
        update("pending", msg="Leyendo archivo...")
        gc = _gemini_client(api_key)

        # ── Preparar contenido según tipo de archivo ──────────────────────────
        if ext == ".pdf":
            update("pending", msg="Leyendo PDF...")
            with open(str(ruta), "rb") as pdf_f:
                pdf_bytes = pdf_f.read()
            file_part = genai_types.Part(
                inline_data=genai_types.Blob(
                    mime_type="application/pdf",
                    data=pdf_bytes
                )
            )
        else:
            file_part = PIL.Image.open(str(ruta))

        update("pending", msg="Extrayendo indicadores con vision IA...")
        prompt_ocr = """Analiza este examen medico/laboratorio y extrae todos los indicadores.

Responde UNICAMENTE con este JSON compacto (sin markdown, sin texto adicional, sin campos extra):
{"tipo_examen":"","laboratorio":"","fecha":"","paciente_info":"","observaciones_laboratorio":"","indicadores":[{"nombre":"","valor":"","unidad":"","rango_ref":"","rango_min":null,"rango_max":null,"estado":"normal","descripcion":""}]}

Reglas:
- Extrae TODOS los indicadores visibles, uno por uno
- rango_min y rango_max: numeros (ej: 70 y 100 para "70-100"), null si no aplica
- estado: solo "normal", "alto", "bajo" o "critico"
- descripcion: maximo 8 palabras explicando que mide
- Valores aproximados: agrega "~" al inicio (ej: "~12.5")
- NO incluyas el campo texto_completo"""

        resp_ocr = _generate(gc, [prompt_ocr, file_part])

        raw = re.sub(r"```json\s*|\s*```", "", resp_ocr.text.strip()).strip()
        try:
            ocr_data = json.loads(raw)
        except json.JSONDecodeError:
            idx = raw.rfind("},")
            if idx == -1: idx = raw.rfind("}")
            ocr_data = json.loads(raw[:idx+1] + "]}") if idx != -1 else \
                       {"tipo_examen":tipo,"laboratorio":"","fecha":"","paciente_info":"","indicadores":[],"observaciones_laboratorio":""}

        indicadores = [i for i in ocr_data.get("indicadores", []) if i and isinstance(i, dict)]
        alertas  = [i for i in indicadores if i.get("estado") in ("alto","bajo","critico")]
        criticos = [i for i in indicadores if i.get("estado") == "critico"]
        riesgo = "critico" if criticos else ("alto" if len(alertas)>=3 else ("medio" if alertas else "normal"))

        update("pending", msg=f"Interpretando {len(indicadores)} indicadores...")
        prompt_interp = f"""Como medico experto, interpreta estos resultados de examen medico.

TIPO DE EXAMEN: {ocr_data.get('tipo_examen', tipo)}
INDICADORES:
{json.dumps(indicadores, ensure_ascii=False, indent=2)}

OBSERVACIONES DEL LABORATORIO: {ocr_data.get('observaciones_laboratorio','')}

Genera una interpretacion medica completa en JSON:
{{
  "resumen": "Resumen ejecutivo del estado de salud en 2-3 oraciones",
  "interpretacion": "Interpretacion detallada de los hallazgos mas relevantes (markdown)",
  "recomendaciones": [
    {{"tipo": "dieta|ejercicio|medicamento|consulta|estilo_vida", "titulo": "...", "descripcion": "...", "prioridad": "alta|media|baja"}}
  ],
  "alertas_principales": ["lista de los indicadores mas preocupantes con explicacion"],
  "puntos_positivos": ["aspectos de salud que estan bien"],
  "seguimiento": "cuando y con que especialista debe consultar"
}}"""

        try:
            resp2 = _generate(gc, prompt_interp)
            raw2 = re.sub(r"```json\s*|\s*```", "", resp2.text.strip()).strip()
            interp_data = json.loads(raw2)
        except Exception:
            interp_data = {
                "resumen": "Analisis completado.",
                "interpretacion": "Revise los indicadores detallados y consulte con su medico.",
                "recomendaciones": [],
                "alertas_principales": [f"{i['nombre']}: {i['valor']}" for i in alertas],
                "puntos_positivos": [],
                "seguimiento": "Consulte con su medico tratante."
            }

        # ── Generar preguntas para el médico tratante ────────────────────────
        preguntas_doctor = []
        if alertas:
            update("pending", msg="Generando preguntas para tu médico...")
            lineas_alertas = "\n".join([
                f"- {i['nombre']}: {i['valor']} {i.get('unidad','')} [{i.get('estado','').upper()}] (ref: {i.get('rango_ref','')})"
                for i in alertas
            ])
            prompt_pq = f"""Eres un asistente médico experto. Un paciente tiene los siguientes indicadores alterados en su examen de {ocr_data.get('tipo_examen', tipo)}:

{lineas_alertas}

Interpretación: {interp_data.get('resumen', '')}

Genera entre 5 y 8 preguntas concretas que el paciente debería hacerle a su médico tratante. Las preguntas deben:
- Ser específicas a los indicadores alterados
- Estar en primera persona ("¿Debo...", "¿Qué significa...", "¿Es necesario...")
- Cubrir tratamiento, seguimiento y cambios de estilo de vida
- NO incluir diagnósticos ni nombres de medicamentos específicos

Responde ÚNICAMENTE con un JSON array de strings, sin markdown ni texto extra:
["pregunta 1", "pregunta 2", ...]"""
            try:
                resp_pq = _generate(gc, prompt_pq)
                raw_pq = re.sub(r"```json\s*|\s*```", "", resp_pq.text.strip()).strip()
                preguntas_doctor = json.loads(raw_pq)
                if not isinstance(preguntas_doctor, list):
                    preguntas_doctor = []
                preguntas_doctor = [str(p) for p in preguntas_doctor if p][:8]
            except Exception:
                preguntas_doctor = []

        update("pending", msg="Guardando resultados...")
        with get_db() as db:
            cur = db.execute("""INSERT INTO examenes
                (paciente_id,titulo,tipo,fecha_examen,laboratorio,imagen_path,
                 texto_extraido,interpretacion,resumen,estado,riesgo,preguntas_doctor)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (paciente_id, titulo, ocr_data.get("tipo_examen",tipo), ocr_data.get("fecha",""),
                 ocr_data.get("laboratorio",""), ruta.name, "",
                 interp_data.get("interpretacion",""), interp_data.get("resumen",""), "analizado", riesgo,
                 json.dumps(preguntas_doctor, ensure_ascii=False)))
            eid = cur.lastrowid

            for ind in indicadores:
                db.execute("""INSERT INTO indicadores
                    (examen_id,nombre,valor,unidad,rango_ref,rango_min,rango_max,estado,descripcion)
                    VALUES (?,?,?,?,?,?,?,?,?)""",
                    (eid, ind.get("nombre",""), str(ind.get("valor","")),
                     ind.get("unidad",""), ind.get("rango_ref",""),
                     ind.get("rango_min"), ind.get("rango_max"),
                     ind.get("estado","normal"), ind.get("descripcion","")))

            for rec in interp_data.get("recomendaciones",[]):
                db.execute("""INSERT INTO recomendaciones
                    (examen_id,tipo,titulo,descripcion,prioridad) VALUES (?,?,?,?,?)""",
                    (eid, rec.get("tipo","general"), rec.get("titulo",""),
                     rec.get("descripcion",""), rec.get("prioridad","media")))

            conv_cur = db.execute(
                "INSERT INTO conversaciones (examen_id,titulo) VALUES (?,?)",
                (eid, f"Dr. VitalIA — {titulo}"))
            conv_id = conv_cur.lastrowid

            saludo = f"""🩺 **Hola, soy el Dr. VitalIA**

He analizado tu **{ocr_data.get('tipo_examen', titulo)}** y tengo los resultados listos.

**Resumen:** {interp_data.get('resumen','')}

{interp_data.get('interpretacion','')}

---
**Seguimiento recomendado:** {interp_data.get('seguimiento','')}

⚠️ *Recuerda que este análisis es orientativo. Siempre consulta con tu médico tratante para un diagnóstico definitivo.*

¿Tienes alguna pregunta sobre tus resultados? Estoy aquí para ayudarte. 💙"""
            db.execute("INSERT INTO mensajes (conversacion_id,rol,contenido) VALUES (?,?,?)",
                       (conv_id, "assistant", saludo))
            db.commit()

        update("done", ok=True, examen_id=eid, conversacion_id=conv_id,
               resumen=interp_data.get("resumen",""), riesgo=riesgo,
               num_indicadores=len(indicadores), alertas=len(alertas), criticos=len(criticos))

    except Exception as e:
        import traceback; print("[VitalIA ERROR]", traceback.format_exc())
        update("error", error=str(e))


@app.route("/api/analizar", methods=["GET", "POST"])
@login_required
def analizar_examen():
    if request.method == "GET":
        return jsonify({"error": "Usa POST para enviar un examen"}), 405

    uid = session["usuario_id"]
    admin_mode = is_admin()

    # Admin: sin cobro, pago ficticio interno
    if admin_mode:
        payment_intent_id = f"pi_admin_{uuid.uuid4().hex}"
    else:
        # Verificar pago
        payment_intent_id = request.form.get("payment_intent_id","").strip()
        if not payment_intent_id:
            return jsonify({"error": "Se requiere pago previo para analizar un examen"}), 402

        with get_db() as db:
            pago = db.execute(
                "SELECT * FROM pagos WHERE stripe_payment_intent_id=? AND usuario_id=? AND estado='completado'",
                (payment_intent_id, uid)
            ).fetchone()

        if not pago:
            if payment_intent_id.startswith("pi_dev_"):
                with get_db() as db:
                    pago = db.execute(
                        "SELECT * FROM pagos WHERE stripe_payment_intent_id=? AND usuario_id=?",
                        (payment_intent_id, uid)
                    ).fetchone()
                if not pago:
                    return jsonify({"error": "Pago no encontrado"}), 402
            else:
                return jsonify({"error": "Pago no verificado o ya utilizado"}), 402

    try:
        api_key = (os.environ.get("GOOGLE_API_KEY","") or
                   os.environ.get("GEMINI_API_KEY","") or
                   request.headers.get("X-API-Key",""))
        if not api_key: raise ValueError("GOOGLE_API_KEY no configurada")
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    if "imagen" not in request.files:
        return jsonify({"error": "No se recibió imagen"}), 400

    f = request.files["imagen"]
    titulo = request.form.get("titulo", "Examen médico").strip() or "Examen médico"
    tipo   = request.form.get("tipo", "sangre")
    ext = Path(f.filename).suffix.lower() if f.filename else ".jpg"
    if not ext: ext = ".jpg"
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Formato no soportado: {ext}"}), 400

    # Guardar en carpeta del paciente
    paciente = get_current_paciente()
    if not paciente:
        return jsonify({"error": "Perfil de paciente no encontrado. Por favor cierra sesión y vuelve a ingresar."}), 400
    user_upload_dir = UPLOAD_DIR / str(uid)
    user_upload_dir.mkdir(parents=True, exist_ok=True)

    nombre_archivo = f"{uuid.uuid4().hex}{ext}"
    ruta = user_upload_dir / nombre_archivo
    f.save(str(ruta))

    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {"status": "pending", "msg": "Iniciando análisis..."}

    # Marcar pago como en proceso usando campo estado (evita FK constraint)
    if not admin_mode:
        with get_db() as db:
            db.execute(
                "UPDATE pagos SET estado='en_proceso' WHERE stripe_payment_intent_id=?",
                (payment_intent_id,)
            )
            db.commit()

    t = threading.Thread(
        target=_run_analisis_with_payment,
        args=(job_id, ruta, ext, titulo, tipo, api_key, paciente["id"], payment_intent_id, admin_mode),
        daemon=True
    )
    t.start()

    return jsonify({"job_id": job_id})


def _run_analisis_with_payment(job_id, ruta, ext, titulo, tipo, api_key, paciente_id, payment_intent_id, admin_mode=False):
    """Wrapper que actualiza el pago al terminar el análisis."""
    _run_analisis(job_id, ruta, ext, titulo, tipo, api_key, paciente_id)
    if admin_mode:
        return  # Admin no tiene registro en pagos
    # Vincular pago al examen creado
    with _jobs_lock:
        job = _jobs.get(job_id, {})
    if job.get("status") == "done" and job.get("examen_id"):
        with get_db() as db:
            db.execute(
                "UPDATE pagos SET examen_id=? WHERE stripe_payment_intent_id=?",
                (job["examen_id"], payment_intent_id)
            )
            db.commit()
    elif job.get("status") == "error":
        # Si el análisis falló, restaurar pago a completado para poder reintentar
        with get_db() as db:
            db.execute(
                "UPDATE pagos SET estado='completado' WHERE stripe_payment_intent_id=?",
                (payment_intent_id,)
            )
            db.commit()


@app.route("/api/job/<job_id>", methods=["GET"])
@login_required
def get_job(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job no encontrado"}), 404
    return jsonify(job)


# ── API: Chat con Dr. VitalIA ─────────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
@login_required
def chat():
    d = request.json or {}
    conv_id  = d.get("conversacion_id")
    mensaje  = d.get("mensaje","").strip()
    if not conv_id or not mensaje:
        return jsonify({"error":"Faltan datos"}), 400

    try:
        cliente = get_client()
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    paciente = get_current_paciente()
    if not paciente:
        return jsonify({"error": "Sesión expirada. Por favor vuelve a iniciar sesión."}), 401
    with get_db() as db:
        # Verificar que la conversación pertenece al usuario
        conv = db.execute("""
            SELECT c.* FROM conversaciones c
            JOIN examenes e ON e.id = c.examen_id
            WHERE c.id=? AND e.paciente_id=?
        """, (conv_id, paciente["id"])).fetchone()
        if not conv:
            return jsonify({"error":"Conversación no encontrada"}), 404

        contexto_examen = ""
        if conv["examen_id"]:
            e = db.execute("SELECT * FROM examenes WHERE id=?", (conv["examen_id"],)).fetchone()
            inds = db.execute("SELECT * FROM indicadores WHERE examen_id=?", (conv["examen_id"],)).fetchall()
            if e:
                alertas_str = "\n".join([f"- {i['nombre']}: {i['valor']} {i['unidad']} [{i['estado'].upper()}]"
                                          for i in inds if i["estado"] != "normal"])
                normales_str = "\n".join([f"- {i['nombre']}: {i['valor']} {i['unidad']}"
                                           for i in inds if i["estado"] == "normal"])
                contexto_examen = f"""
CONTEXTO DEL EXAMEN DEL PACIENTE:
Tipo: {e['tipo']}
Fecha: {e['fecha_examen']}
Laboratorio: {e['laboratorio']}
Riesgo general: {e['riesgo']}

INDICADORES ALTERADOS:
{alertas_str or '(ninguno)'}

INDICADORES NORMALES:
{normales_str or '(ninguno)'}

INTERPRETACIÓN PREVIA:
{e['interpretacion'][:500] if e['interpretacion'] else ''}
"""

        historial = db.execute(
            "SELECT rol, contenido FROM mensajes WHERE conversacion_id=? ORDER BY id DESC LIMIT 20",
            (conv_id,)
        ).fetchall()
        historial = list(reversed(historial))

        db.execute("INSERT INTO mensajes (conversacion_id, rol, contenido) VALUES (?,?,?)",
                   (conv_id, "user", mensaje))
        db.commit()

    system = SYSTEM_DOCTOR
    if contexto_examen:
        system += f"\n\n{contexto_examen}"

    # Convertir historial al formato Gemini (user / model, sin repetir mismo rol)
    gemini_history = []
    for h in historial:
        role = "user" if h["rol"] == "user" else "model"
        if gemini_history and gemini_history[-1]["role"] == role:
            gemini_history[-1]["parts"][0] += "\n" + h["contenido"]
        else:
            gemini_history.append({"role": role, "parts": [h["contenido"]]})

    # El último mensaje del historial es el del usuario actual — va en send_message
    if gemini_history and gemini_history[-1]["role"] == "user":
        gemini_history.pop()

    def generate():
        respuesta_completa = ""
        try:
            gc = _gemini_client(cliente)
            contents = []
            for h in gemini_history:
                contents.append(genai_types.Content(
                    role=h["role"],
                    parts=[genai_types.Part(text=h["parts"][0])]
                ))
            contents.append(genai_types.Content(
                role="user",
                parts=[genai_types.Part(text=mensaje)]
            ))
            # Inyectar system prompt como primer par user/model (v1 no soporta systemInstruction)
            contents.insert(0, genai_types.Content(
                role="model", parts=[genai_types.Part(text="Entendido. Actuaré como el Dr. VitalIA.")]
            ))
            contents.insert(0, genai_types.Content(
                role="user", parts=[genai_types.Part(text=system)]
            ))

            # Usar modelo detectado al inicio; solo hacer fallback en 503 transitorio
            fallback_models = [m for m in _GEMINI_CANDIDATES if m != GEMINI_MODEL]
            models_to_try = [GEMINI_MODEL] + fallback_models
            last_err = None
            for model_try in models_to_try:
                try:
                    for chunk in gc.models.generate_content_stream(
                        model=model_try, contents=contents
                    ):
                        if chunk.text:
                            respuesta_completa += chunk.text
                            yield f"data: {json.dumps({'text': chunk.text})}\n\n"
                    last_err = None
                    break
                except Exception as e:
                    msg = str(e)
                    if "UNAVAILABLE" in msg or "503" in msg or "high demand" in msg:
                        last_err = e
                        continue
                    raise

            if last_err:
                raise last_err

            with get_db() as db:
                db.execute("INSERT INTO mensajes (conversacion_id, rol, contenido) VALUES (?,?,?)",
                           (conv_id, "assistant", respuesta_completa))
                db.commit()

            yield f"data: {json.dumps({'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_with_context(generate()),
                    mimetype="text/event-stream",
                    headers={"X-Accel-Buffering": "no"})


# ── API: Conversaciones ───────────────────────────────────────────────────────
@app.route("/api/conversaciones/<int:cid>/mensajes")
@login_required
def get_mensajes(cid):
    paciente = get_current_paciente()
    with get_db() as db:
        # Verificar pertenencia
        conv = db.execute("""
            SELECT c.id FROM conversaciones c
            JOIN examenes e ON e.id = c.examen_id
            WHERE c.id=? AND e.paciente_id=?
        """, (cid, paciente["id"])).fetchone()
        if not conv:
            return jsonify({"error": "No encontrado"}), 404
        msgs = db.execute(
            "SELECT * FROM mensajes WHERE conversacion_id=? ORDER BY id",
            (cid,)
        ).fetchall()
    return jsonify({"mensajes": [dict(m) for m in msgs]})


@app.route("/api/conversaciones/nueva", methods=["POST"])
@login_required
def nueva_conversacion():
    d = request.json or {}
    paciente = get_current_paciente()
    with get_db() as db:
        # Verificar que el examen pertenece al usuario
        if d.get("examen_id"):
            e = db.execute("SELECT id FROM examenes WHERE id=? AND paciente_id=?",
                           (d["examen_id"], paciente["id"])).fetchone()
            if not e:
                return jsonify({"error": "Examen no encontrado"}), 404
        cur = db.execute(
            "INSERT INTO conversaciones (examen_id, titulo) VALUES (?,?)",
            (d.get("examen_id"), d.get("titulo", "Nueva consulta"))
        )
        db.commit()
        return jsonify({"ok": True, "id": cur.lastrowid})


# ── API: Stats ────────────────────────────────────────────────────────────────
@app.route("/api/stats")
@login_required
def get_stats():
    paciente = get_current_paciente()
    with get_db() as db:
        pid = paciente["id"]
        total_examenes = db.execute("SELECT COUNT(*) FROM examenes WHERE paciente_id=?", (pid,)).fetchone()[0]
        total_alertas  = db.execute("""
            SELECT COUNT(*) FROM indicadores i
            JOIN examenes e ON e.id=i.examen_id
            WHERE e.paciente_id=? AND i.estado != 'normal'
        """, (pid,)).fetchone()[0]
        total_normales = db.execute("""
            SELECT COUNT(*) FROM indicadores i
            JOIN examenes e ON e.id=i.examen_id
            WHERE e.paciente_id=? AND i.estado = 'normal'
        """, (pid,)).fetchone()[0]
        ultimo_examen = db.execute(
            "SELECT creado_en FROM examenes WHERE paciente_id=? ORDER BY id DESC LIMIT 1", (pid,)
        ).fetchone()
        riesgos = db.execute(
            "SELECT riesgo, COUNT(*) as c FROM examenes WHERE paciente_id=? GROUP BY riesgo", (pid,)
        ).fetchall()
    return jsonify({
        "total_examenes": total_examenes,
        "total_alertas": total_alertas,
        "total_normales": total_normales,
        "ultimo_examen": ultimo_examen[0] if ultimo_examen else None,
        "riesgos": {r["riesgo"]: r["c"] for r in riesgos}
    })


# ── API: Config ───────────────────────────────────────────────────────────────
@app.route("/api/config", methods=["POST"])
@login_required
def config():
    d = request.json or {}
    if d.get("api_key"):
        os.environ["GOOGLE_API_KEY"] = d["api_key"].strip()
    key = os.environ.get("GOOGLE_API_KEY","") or os.environ.get("GEMINI_API_KEY","")
    return jsonify({"ok": True, "api_key_ok": bool(key)})


# ── Imagen upload viewer ──────────────────────────────────────────────────────
@app.route("/uploads/<path:filename>")
@login_required
def serve_upload(filename):
    from flask import send_from_directory
    # Solo servir archivos del usuario actual
    uid = session["usuario_id"]
    user_dir = UPLOAD_DIR / str(uid)
    # Compatibilidad: si el archivo viene de la ruta plana (datos legacy)
    safe = Path(filename)
    if len(safe.parts) > 1 and safe.parts[0] == str(uid):
        return send_from_directory(UPLOAD_DIR, filename)
    return send_from_directory(user_dir, filename)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import threading, webbrowser, sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sep = "=" * 55
    print(sep)
    print("  VitalIA -- Dr. Digital")
    print(f"  URL: http://localhost:{PORT}")
    print(sep)
    print("  ACCESO ADMINISTRADOR")
    print(f"  Email   : {ADMIN_EMAIL}")
    print(f"  Clave   : {ADMIN_PASSWORD}")
    print("  (sin cobros · acceso demo completo)")
    print(sep)
    if not STRIPE_SECRET_KEY:
        print("  ⚠️  STRIPE_SECRET_KEY no configurada — pagos en modo demo")
        print(sep)
    print("  Ctrl+C para detener")
    print(sep)
    threading.Timer(1.2, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
