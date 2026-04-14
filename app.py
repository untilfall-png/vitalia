"""
VitalIA — Dr. Digital
Análisis inteligente de exámenes médicos con IA
"""
import os, sys, json, sqlite3, base64, re, uuid, threading, secrets, smtplib
from pathlib import Path
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
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
    "gemini-2.5-flash",
    "gemini-2.5-pro",
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

# ── Mail (recuperación de contraseña) ─────────────────────────────────────────
MAIL_SERVER   = os.environ.get("MAIL_SERVER",   "smtp.gmail.com")
MAIL_PORT     = int(os.environ.get("MAIL_PORT", "587"))
MAIL_USERNAME = os.environ.get("MAIL_USERNAME", "")
MAIL_PASSWORD = os.environ.get("MAIL_PASSWORD", "")
MAIL_FROM     = os.environ.get("MAIL_FROM",     MAIL_USERNAME)
APP_URL       = os.environ.get("APP_URL",        "https://vitalia.work")

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
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario_id   INTEGER NOT NULL,
            nombre       TEXT NOT NULL DEFAULT 'Paciente',
            fecha_nac    TEXT DEFAULT '',
            genero       TEXT DEFAULT '',
            email        TEXT DEFAULT '',
            notas        TEXT DEFAULT '',
            edad         INTEGER DEFAULT NULL,
            peso         REAL DEFAULT NULL,
            medicamentos TEXT DEFAULT '',
            condiciones  TEXT DEFAULT '',
            sintomas     TEXT DEFAULT '',
            creado_en    TEXT DEFAULT (datetime('now','localtime')),
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

        -- ── Consentimientos ───────────────────────────────────────────────────
        CREATE TABLE IF NOT EXISTS consentimientos (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            usuario_id  INTEGER NOT NULL,
            version     TEXT NOT NULL DEFAULT '2.0',
            ip          TEXT DEFAULT '',
            user_agent  TEXT DEFAULT '',
            aceptado_en TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (usuario_id) REFERENCES usuarios(id)
        );

        -- ── Plan de acción personalizado ──────────────────────────────────────
        CREATE TABLE IF NOT EXISTS planes_accion (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            examen_id   INTEGER UNIQUE NOT NULL,
            nutricion   TEXT DEFAULT '{}',
            movimiento  TEXT DEFAULT '{}',
            habitos     TEXT DEFAULT '{}',
            predicciones TEXT DEFAULT '[]',
            creado_en   TEXT DEFAULT (datetime('now','localtime')),
            FOREIGN KEY (examen_id) REFERENCES examenes(id) ON DELETE CASCADE
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
        # pacientes — campos clínicos extendidos
        pcols = [r[1] for r in db.execute("PRAGMA table_info(pacientes)").fetchall()]
        for col, ddl in [
            ("edad",         "ALTER TABLE pacientes ADD COLUMN edad INTEGER DEFAULT NULL"),
            ("peso",         "ALTER TABLE pacientes ADD COLUMN peso REAL DEFAULT NULL"),
            ("medicamentos", "ALTER TABLE pacientes ADD COLUMN medicamentos TEXT DEFAULT ''"),
            ("condiciones",  "ALTER TABLE pacientes ADD COLUMN condiciones TEXT DEFAULT ''"),
            ("sintomas",     "ALTER TABLE pacientes ADD COLUMN sintomas TEXT DEFAULT ''"),
        ]:
            if col not in pcols:
                db.execute(ddl)
        # examenes.descargas — contador de reportes descargados
        ecols2 = [r[1] for r in db.execute("PRAGMA table_info(examenes)").fetchall()]
        if "descargas" not in ecols2:
            db.execute("ALTER TABLE examenes ADD COLUMN descargas INTEGER DEFAULT 0")
        # usuarios — recuperación de contraseña y control de intentos
        ucols2 = [r[1] for r in db.execute("PRAGMA table_info(usuarios)").fetchall()]
        for col, ddl in [
            ("intentos_fallidos",  "ALTER TABLE usuarios ADD COLUMN intentos_fallidos INTEGER DEFAULT 0"),
            ("reset_token",        "ALTER TABLE usuarios ADD COLUMN reset_token TEXT DEFAULT NULL"),
            ("reset_token_expiry", "ALTER TABLE usuarios ADD COLUMN reset_token_expiry TEXT DEFAULT NULL"),
        ]:
            if col not in ucols2:
                db.execute(ddl)
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

# ── Contexto clínico del paciente para prompts IA ────────────────────────────
def _build_patient_ctx(paciente: dict) -> str:
    """Genera un bloque de texto con el perfil clínico del paciente para inyectar en prompts."""
    parts = []
    if paciente.get("nombre"):
        parts.append(f"- Nombre: {paciente['nombre']}")
    if paciente.get("edad"):
        parts.append(f"- Edad: {paciente['edad']} años")
    if paciente.get("genero"):
        parts.append(f"- Género: {paciente['genero']}")
    if paciente.get("peso"):
        parts.append(f"- Peso: {paciente['peso']} kg")
    if paciente.get("condiciones","").strip():
        parts.append(f"- Condiciones médicas conocidas: {paciente['condiciones']}")
    if paciente.get("medicamentos","").strip():
        parts.append(f"- Medicamentos actuales: {paciente['medicamentos']}")
    if paciente.get("sintomas","").strip():
        parts.append(f"- Síntomas reportados por el paciente: {paciente['sintomas']}")
    if paciente.get("notas","").strip():
        parts.append(f"- Notas adicionales: {paciente['notas']}")
    return "\n".join(parts) if parts else "Sin información de perfil clínico adicional."


# ── Reporte HTML ──────────────────────────────────────────────────────────────
def _build_plan_html_reporte(plan: dict) -> str:
    """Construye la sección HTML del plan de acción para el reporte imprimible."""
    if not plan:
        return ""
    nut = plan.get("nutricion") or {}
    mov = plan.get("movimiento") or {}
    hab = plan.get("habitos") or {}
    preds = plan.get("predicciones") or []
    if not (nut or mov or hab or preds):
        return ""

    H = lambda s: str(s).replace("&","&amp;").replace("<","&lt;").replace(">","&gt;").replace('"',"&quot;")

    # ── Nutrición ─────────────────────────────────────────────────────────────
    com_rows = "".join([
        f'<tr><td style="padding:8px 10px;border-bottom:1px solid #f1f5f9;font-size:13px;">{H(a.get("emoji",""))} {H(a.get("alimento",""))}</td>'
        f'<td style="padding:8px 10px;border-bottom:1px solid #f1f5f9;font-size:12px;color:#475569;">{H(a.get("razon",""))}</td>'
        f'<td style="padding:8px 10px;border-bottom:1px solid #f1f5f9;font-size:12px;color:#4f46e5;white-space:nowrap;">{H(a.get("frecuencia",""))}</td></tr>'
        for a in (nut.get("que_comer") or [])
    ])
    ev_rows = "".join([
        f'<tr><td style="padding:8px 10px;border-bottom:1px solid #f1f5f9;font-size:13px;">{H(a.get("emoji",""))} {H(a.get("alimento",""))}</td>'
        f'<td style="padding:8px 10px;border-bottom:1px solid #f1f5f9;font-size:12px;color:#475569;">{H(a.get("razon",""))}</td>'
        f'<td style="padding:8px 10px;border-bottom:1px solid #f1f5f9;font-size:11px;font-weight:700;color:#991b1b;white-space:nowrap;">{H((a.get("impacto","") or "").upper())} impacto</td></tr>'
        for a in (nut.get("que_evitar") or [])
    ])
    dia_ej = f'<div style="margin-top:10px;padding:12px 14px;background:#f8fafc;border-left:3px solid #4f46e5;border-radius:0 6px 6px 0;font-size:12px;color:#475569;line-height:1.8;white-space:pre-line;">{H(nut.get("dia_ejemplo",""))}</div>' if nut.get("dia_ejemplo") else ""
    nut_html = f"""<div style="margin-bottom:24px;">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
        <span style="font-size:18px;">🥗</span>
        <div style="font-weight:700;font-size:14px;color:#1e293b;">{H(nut.get("titulo","Nutrición"))}</div>
        {f'<span style="background:#ede9fe;color:#5b21b6;padding:2px 10px;border-radius:20px;font-size:10px;font-weight:700;margin-left:auto;">{H(nut.get("patron_dieta",""))}</span>' if nut.get("patron_dieta") else ""}
      </div>
      <div style="font-size:12px;color:#64748b;margin-bottom:12px;">{H(nut.get("objetivo",""))}</div>
      {f'<div style="font-size:11px;font-weight:700;color:#166534;margin-bottom:6px;text-transform:uppercase;letter-spacing:.5px;">✅ Qué comer</div><table style="width:100%;border-collapse:collapse;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;background:#fff;margin-bottom:12px;"><tbody>{com_rows}</tbody></table>' if com_rows else ""}
      {f'<div style="font-size:11px;font-weight:700;color:#991b1b;margin-bottom:6px;text-transform:uppercase;letter-spacing:.5px;">🚫 Qué evitar</div><table style="width:100%;border-collapse:collapse;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;background:#fff;margin-bottom:12px;"><tbody>{ev_rows}</tbody></table>' if ev_rows else ""}
      {dia_ej}
    </div>"""

    # ── Movimiento ────────────────────────────────────────────────────────────
    act_items = "".join([
        f'<div style="padding:12px 14px;background:#f8fafc;border:1px solid #e2e8f0;border-left:4px solid #4f46e5;border-radius:0 8px 8px 0;margin-bottom:8px;">'
        f'<div style="font-weight:700;font-size:13px;color:#1e293b;margin-bottom:6px;">{H(a.get("emoji",""))} {H(a.get("nombre",""))}'
        f'  <span style="background:#ede9fe;color:#5b21b6;padding:1px 8px;border-radius:8px;font-size:10px;font-weight:600;margin-left:6px;">{H(a.get("tipo",""))}</span></div>'
        f'<div style="display:flex;gap:16px;flex-wrap:wrap;font-size:12px;color:#475569;margin-bottom:6px;">'
        f'  <span><strong>Frecuencia:</strong> {H(a.get("frecuencia",""))}</span>'
        f'  <span><strong>Duración:</strong> {H(a.get("duracion",""))}</span>'
        f'  <span><strong>Intensidad:</strong> {H(a.get("intensidad",""))}</span></div>'
        f'<div style="font-size:12px;color:#1d4ed8;">{H(a.get("beneficio",""))}</div>'
        f'</div>'
        for a in (mov.get("actividades") or [])
    ])
    mov_html = f"""<div style="margin-bottom:24px;">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
        <span style="font-size:18px;">🏃</span>
        <div style="font-weight:700;font-size:14px;color:#1e293b;">{H(mov.get("titulo","Movimiento"))}</div>
      </div>
      <div style="font-size:12px;color:#64748b;margin-bottom:12px;">{H(mov.get("objetivo",""))}</div>
      {act_items}
      {f'<div style="padding:10px 14px;background:#f0fdf4;border:1px solid #86efac;border-radius:8px;font-size:12px;color:#166534;margin-top:8px;"><strong>Progresión:</strong> {H(mov.get("progresion",""))}</div>' if mov.get("progresion") else ""}
      {f'<div style="padding:10px 14px;background:#fefce8;border:1px solid #fcd34d;border-radius:8px;font-size:12px;color:#92400e;margin-top:8px;"><strong>⚠️ Precaución:</strong> {H(mov.get("contraindicaciones",""))}</div>' if mov.get("contraindicaciones") else ""}
    </div>"""

    # ── Hábitos ───────────────────────────────────────────────────────────────
    sueno  = hab.get("sueno") or {}
    estres = hab.get("estres") or {}
    alc    = hab.get("alcohol") or {}
    otros  = hab.get("otros","")
    def habito_card(emoji, titulo, objetivo, conexion, tips_or_tecnicas, key_list):
        tips_html = "".join(['<li style="font-size:12px;color:#475569;margin-bottom:4px;">' + H(t) + '</li>' for t in (tips_or_tecnicas or [])])
        obj_html  = ('<div style="font-weight:600;font-size:12px;color:#4f46e5;margin-bottom:6px;">' + H(objetivo) + '</div>') if objetivo else ""
        con_html  = ('<div style="font-size:12px;color:#64748b;margin-bottom:8px;line-height:1.5;">' + H(conexion) + '</div>') if conexion else ""
        ul_html   = ('<ul style="margin:0;padding-left:16px;">' + tips_html + '</ul>') if tips_html else ""
        return ('<div style="flex:1;min-width:200px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:14px;">'
                '<div style="font-weight:700;font-size:13px;color:#1e293b;margin-bottom:4px;">' + emoji + ' ' + titulo + '</div>'
                + obj_html + con_html + ul_html + '</div>')
    hab_html = f"""<div style="margin-bottom:24px;">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:12px;">
        <span style="font-size:18px;">😴</span>
        <div style="font-weight:700;font-size:14px;color:#1e293b;">Hábitos críticos</div>
      </div>
      <div style="display:flex;gap:12px;flex-wrap:wrap;">
        {habito_card("😴","Sueño", sueno.get("objetivo",""), f"{sueno.get('por_que','')} {sueno.get('conexion','')}", sueno.get("tips",[]), "tips")}
        {habito_card("🧘","Estrés", "", estres.get("impacto",""), estres.get("tecnicas",[]), "tecnicas")}
        {habito_card("🍷","Alcohol", alc.get("recomendacion",""), f"{alc.get('por_que','')} {alc.get('conexion','')}", [], "")}
      </div>
      {f'<div style="margin-top:10px;padding:10px 14px;background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;font-size:12px;color:#475569;"><strong>💡 Otros hábitos:</strong> {H(otros)}</div>' if otros else ""}
    </div>"""

    # ── Predicciones ──────────────────────────────────────────────────────────
    pred_rows = "".join([
        f'<tr>'
        f'<td style="padding:10px 12px;border-bottom:1px solid #f1f5f9;font-weight:600;font-size:13px;color:#1e293b;">{H(p.get("indicador",""))}</td>'
        f'<td style="padding:10px 12px;border-bottom:1px solid #f1f5f9;font-family:monospace;font-weight:700;color:#991b1b;font-size:13px;">{H(str(p.get("valor_actual","")))} {H(p.get("unidad",""))}</td>'
        f'<td style="padding:10px 12px;border-bottom:1px solid #f1f5f9;font-family:monospace;font-weight:700;color:#166534;font-size:13px;">{H(p.get("valor_estimado",""))}</td>'
        f'<td style="padding:10px 12px;border-bottom:1px solid #f1f5f9;font-weight:700;color:#166534;font-size:12px;">{H(p.get("mejora_esperada",""))} en {H(p.get("plazo",""))}</td>'
        f'<td style="padding:10px 12px;border-bottom:1px solid #f1f5f9;font-size:11px;color:#64748b;">{H(p.get("condicion",""))}</td>'
        f'</tr>'
        for p in preds
    ])
    pred_html = f"""<div style="margin-bottom:24px;">
      <div style="font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.5px;margin-bottom:10px;">📈 Si sigues el plan: predicciones estimadas</div>
      <table style="width:100%;border-collapse:collapse;border:1px solid #e2e8f0;border-radius:8px;overflow:hidden;background:#fff;">
        <thead><tr style="background:#f1f5f9;">
          <th style="padding:8px 12px;font-size:11px;color:#64748b;text-align:left;border-bottom:1px solid #e2e8f0;">Indicador</th>
          <th style="padding:8px 12px;font-size:11px;color:#64748b;text-align:left;border-bottom:1px solid #e2e8f0;">Actual</th>
          <th style="padding:8px 12px;font-size:11px;color:#64748b;text-align:left;border-bottom:1px solid #e2e8f0;">Estimado</th>
          <th style="padding:8px 12px;font-size:11px;color:#64748b;text-align:left;border-bottom:1px solid #e2e8f0;">Mejora / Plazo</th>
          <th style="padding:8px 12px;font-size:11px;color:#64748b;text-align:left;border-bottom:1px solid #e2e8f0;">Condición</th>
        </tr></thead>
        <tbody>{pred_rows}</tbody>
      </table>
      <p style="font-size:11px;color:#94a3b8;margin-top:8px;font-style:italic;">Estimaciones orientativas basadas en evidencia. Los resultados individuales pueden variar. Siempre consultar con el médico tratante.</p>
    </div>""" if pred_rows else ""

    return f"""<div style="margin-bottom:32px;padding:22px;background:#fff;border:1px solid #c7d2fe;border-radius:12px;border-top:4px solid #4f46e5;">
      <div style="font-size:13px;font-weight:700;color:#4f46e5;text-transform:uppercase;letter-spacing:.8px;margin-bottom:20px;">🚀 Plan de acción personalizado</div>
      {nut_html}
      {mov_html}
      {hab_html}
      {pred_html}
    </div>"""


def _build_reporte_html(nombre_paciente: str, examen: dict,
                         indicadores: list, recomendaciones: list,
                         preguntas: list, plan: dict = None) -> str:

    # Paleta clara, legible en papel y pantalla
    estado_cfg = {
        "normal":  {"color":"#166534","bg":"#dcfce7","border":"#86efac","icon":"✓","label":"Normal"},
        "alto":    {"color":"#991b1b","bg":"#fee2e2","border":"#fca5a5","icon":"↑","label":"Alto"},
        "bajo":    {"color":"#92400e","bg":"#fef3c7","border":"#fcd34d","icon":"↓","label":"Bajo"},
        "critico": {"color":"#7f1d1d","bg":"#ffe4e6","border":"#fda4af","icon":"!","label":"Crítico"},
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
                barra = f'<div style="margin-top:5px;height:4px;background:#e2e8f0;border-radius:2px;"><div style="height:4px;width:{pct:.0f}%;background:{cfg["color"]};border-radius:2px;"></div></div>'
        except Exception:
            pass
        rows_ind += f"""<tr>
          <td style="padding:10px 14px;border-bottom:1px solid #f1f5f9;color:#1e293b;font-weight:600;font-size:13px;">{ind.get('nombre','')} {barra}</td>
          <td style="padding:10px 14px;border-bottom:1px solid #f1f5f9;font-family:monospace;color:#1d4ed8;font-weight:700;font-size:13px;">{ind.get('valor','')} <span style="color:#64748b;font-size:11px;font-family:'Segoe UI',Arial,sans-serif;">{ind.get('unidad','')}</span></td>
          <td style="padding:10px 14px;border-bottom:1px solid #f1f5f9;color:#64748b;font-size:12px;">{ind.get('rango_ref','—')}</td>
          <td style="padding:10px 14px;border-bottom:1px solid #f1f5f9;"><span style="background:{cfg['bg']};color:{cfg['color']};border:1px solid {cfg['border']};padding:3px 10px;border-radius:20px;font-size:11px;font-weight:700;">{cfg['icon']} {cfg['label']}</span></td>
          <td style="padding:10px 14px;border-bottom:1px solid #f1f5f9;color:#64748b;font-size:12px;">{ind.get('descripcion','')}</td>
        </tr>"""

    # ── Detectar si todos los indicadores son normales ──────────────────────────
    todo_normal = bool(indicadores) and all(
        i.get("estado", "normal") == "normal" for i in indicadores
    )

    # ── Sección positiva: mejores indicadores ────────────────────────────────────
    positive_section_html = ""
    if todo_normal:
        def _center_score(ind):
            try:
                v = float(re.sub(r"[^\d.\-]", "", str(ind.get("valor", ""))))
                m = re.search(r"([\d.,]+)\s*[-–]\s*([\d.,]+)", ind.get("rango_ref", ""))
                if not m:
                    return 0.0
                lo = float(m.group(1).replace(",", "."))
                hi = float(m.group(2).replace(",", "."))
                if hi <= lo:
                    return 0.0
                return 1.0 - abs(2 * (v - lo) / (hi - lo) - 1)
            except Exception:
                return 0.0

        scored = sorted(indicadores, key=_center_score, reverse=True)
        top_inds = scored[:6]
        stars = ["🥇", "🥈", "🥉", "⭐", "⭐", "⭐"]

        top_cards = ""
        for idx, ind in enumerate(top_inds):
            score = _center_score(ind)
            zona_label = '<div style="font-size:10px;color:#166534;font-weight:700;margin-top:3px;">Zona óptima</div>' if score > 0.7 else ""
            top_cards += (
                '<div style="background:#f0fdf4;border:1px solid #86efac;border-radius:10px;padding:14px 12px;text-align:center;">'
                + '<div style="font-size:20px;margin-bottom:5px;">' + stars[idx] + '</div>'
                + '<div style="font-size:11px;color:#374151;margin-bottom:4px;line-height:1.3;">' + str(ind.get("nombre", "")) + '</div>'
                + '<div style="font-size:20px;font-weight:700;color:#15803d;font-family:monospace;line-height:1.1;">' + str(ind.get("valor", "")) + '</div>'
                + '<div style="font-size:10px;color:#6b7280;margin-top:2px;">' + str(ind.get("unidad", "")) + '</div>'
                + '<div style="font-size:10px;color:#9ca3af;margin-top:3px;">Ref: ' + str(ind.get("rango_ref", "")) + '</div>'
                + zona_label
                + '</div>'
            )

        interp_text = examen.get("interpretacion", "")
        bienhacer_block = ""
        if interp_text:
            bienhacer_block = (
                '<div style="margin-top:20px;padding:16px 20px;background:#fff;border:1px solid #86efac;border-left:4px solid #16a34a;border-radius:10px;">'
                + '<div style="font-size:12px;font-weight:700;color:#166534;text-transform:uppercase;letter-spacing:.7px;margin-bottom:10px;">✨ Lo que estás haciendo bien</div>'
                + '<div style="font-size:13px;color:#374151;line-height:1.8;">' + interp_text.replace("\n", "<br>") + '</div>'
                + '</div>'
            )

        positive_section_html = (
            '<div style="margin-bottom:28px;">'
            + '<div style="background:#f0fdf4;border:1px solid #86efac;border-left:5px solid #16a34a;border-radius:10px;padding:18px 22px;margin-bottom:18px;display:flex;align-items:flex-start;gap:14px;">'
            + '<div style="font-size:36px;line-height:1;flex-shrink:0;">🎉</div>'
            + '<div><div style="font-size:16px;font-weight:800;color:#166534;margin-bottom:5px;">¡Resultados excelentes!</div>'
            + '<div style="font-size:13px;color:#166534;line-height:1.65;">Todos tus indicadores están dentro del rango normal. El Dr. VitalIA confirma que tu salud en este examen es óptima.</div></div>'
            + '</div>'
            + '<div style="font-size:12px;font-weight:700;color:#166534;text-transform:uppercase;letter-spacing:.8px;margin-bottom:12px;">Tus mejores indicadores</div>'
            + '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;">' + top_cards + '</div>'
            + bienhacer_block
            + '</div>'
        )

    # ── Recomendaciones (solo si hay alertas) ────────────────────────────────────
    rec_icons = {"dieta":"🥗","ejercicio":"🏃","consulta":"👨‍⚕️","medicamento":"💊","estilo_vida":"🌟","general":"📋"}
    prio_cfg  = {
        "alta":  {"tc":"#991b1b","bg":"#fee2e2","border":"#fca5a5","label":"Alta prioridad"},
        "media": {"tc":"#92400e","bg":"#fef3c7","border":"#fcd34d","label":"Prioridad media"},
        "baja":  {"tc":"#166534","bg":"#dcfce7","border":"#86efac","label":"Informativa"},
    }
    recs_html = ""
    if not todo_normal:
        for rec in recomendaciones:
            icon = rec_icons.get(rec.get("tipo","general"),"📋")
            pc = prio_cfg.get(rec.get("prioridad","media"), prio_cfg["media"])
            recs_html += f"""<div style="display:flex;gap:14px;padding:14px 16px;background:#f8fafc;border:1px solid #e2e8f0;border-left:4px solid {pc['tc']};border-radius:8px;margin-bottom:10px;">
          <div style="font-size:22px;flex-shrink:0;line-height:1;">{icon}</div>
          <div style="flex:1;">
            <div style="font-weight:700;font-size:13px;color:#1e293b;margin-bottom:4px;">{rec.get('titulo','')}
              <span style="background:{pc['bg']};color:{pc['tc']};border:1px solid {pc['border']};padding:1px 8px;border-radius:10px;font-size:10px;font-weight:600;margin-left:6px;">{pc['label']}</span>
            </div>
            <div style="font-size:13px;color:#475569;line-height:1.65;">{rec.get('descripcion','')}</div>
          </div>
        </div>"""

    preguntas_html = ""
    if preguntas and not todo_normal:
        items = "".join([f'<li style="padding:10px 14px;background:#f0f4ff;border:1px solid #c7d2fe;border-radius:8px;margin-bottom:8px;font-size:13px;color:#1e293b;line-height:1.6;list-style:none;"><span style="color:#4338ca;font-weight:700;margin-right:8px;">{i+1}.</span>{p}</li>' for i, p in enumerate(preguntas)])
        preguntas_html = f"""<div style="margin-bottom:32px;padding:20px 22px;background:#f0f4ff;border:1px solid #c7d2fe;border-radius:12px;">
          <h2 style="font-size:15px;font-weight:700;color:#3730a3;margin:0 0 4px;">Preguntas para tu médico tratante</h2>
          <p style="font-size:13px;color:#6366f1;margin:0 0 16px;">Lleva esta lista a tu próxima consulta:</p>
          <ul style="padding:0;margin:0;">{items}</ul></div>"""

    riesgo = examen.get("riesgo","normal")
    riesgo_cfg = {
        "normal":  {"tc":"#166534","bg":"#dcfce7","border":"#86efac","txt":"Todo en orden — valores dentro de rangos normales"},
        "medio":   {"tc":"#92400e","bg":"#fef3c7","border":"#fcd34d","txt":"Atención recomendada — algunos valores requieren seguimiento"},
        "alto":    {"tc":"#991b1b","bg":"#fee2e2","border":"#fca5a5","txt":"Se recomienda consultar a su médico próximamente"},
        "critico": {"tc":"#7f1d1d","bg":"#ffe4e6","border":"#fda4af","txt":"Consulte a un médico a la brevedad posible"},
    }
    rc = riesgo_cfg.get(riesgo, riesgo_cfg["normal"])
    fecha_reporte = datetime.now().strftime("%d/%m/%Y %H:%M")

    html = f"""<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reporte VitalIA — {examen.get('titulo','Examen')}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ margin:0; padding:0; background:#f8fafc; font-family:'Segoe UI',Arial,sans-serif; color:#1e293b; }}
  @media print {{
    .no-print {{ display: none !important; }}
    body {{ background: #fff !important; }}
    * {{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }}
    .page-wrap {{ padding: 0 !important; }}
  }}
  .btn-print {{
    position: fixed; top: 20px; right: 20px; z-index: 999;
    background: #4f46e5; color: #fff; border: none; border-radius: 10px;
    padding: 11px 22px; font-size: 14px; font-weight: 600;
    cursor: pointer; box-shadow: 0 4px 14px rgba(79,70,229,0.4);
    font-family: 'Segoe UI', Arial, sans-serif; letter-spacing: .2px;
  }}
  .btn-print:hover {{ background: #4338ca; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th {{ text-align: left; }}
</style>
</head>
<body>
<button class="btn-print no-print" onclick="window.print()">⬇ Descargar PDF</button>

<div class="page-wrap" style="max-width:720px;margin:0 auto;padding:32px 24px 48px;">

  <!-- Header -->
  <div style="display:flex;align-items:center;justify-content:space-between;border-bottom:2px solid #4f46e5;padding-bottom:20px;margin-bottom:28px;">
    <div>
      <div style="font-size:24px;font-weight:800;color:#4f46e5;letter-spacing:-.5px;">VitalIA</div>
      <div style="font-size:13px;color:#64748b;margin-top:2px;">Análisis de exámenes médicos con inteligencia artificial</div>
    </div>
    <div style="text-align:right;">
      <div style="font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:.8px;">Reporte generado</div>
      <div style="font-size:13px;color:#475569;font-weight:600;margin-top:2px;">{fecha_reporte}</div>
    </div>
  </div>

  <!-- Paciente + Examen -->
  <div style="display:flex;gap:16px;flex-wrap:wrap;margin-bottom:24px;">
    <div style="flex:1;min-width:200px;background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:16px 20px;">
      <div style="font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:.8px;margin-bottom:6px;">Paciente</div>
      <div style="font-size:17px;font-weight:700;color:#1e293b;">{nombre_paciente}</div>
    </div>
    <div style="flex:2;min-width:260px;background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:16px 20px;display:flex;gap:24px;flex-wrap:wrap;">
      {"<div><div style='font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:.8px;margin-bottom:4px;'>Examen</div><div style='font-size:14px;font-weight:600;color:#1e293b;'>" + examen.get('titulo','') + "</div></div>" if examen.get('titulo') else ''}
      {"<div><div style='font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:.8px;margin-bottom:4px;'>Tipo</div><div style='font-size:14px;font-weight:600;color:#1e293b;'>" + examen.get('tipo','') + "</div></div>" if examen.get('tipo') else ''}
      {"<div><div style='font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:.8px;margin-bottom:4px;'>Fecha examen</div><div style='font-size:14px;font-weight:600;color:#1e293b;'>" + examen.get('fecha_examen','') + "</div></div>" if examen.get('fecha_examen') else ''}
      {"<div><div style='font-size:10px;color:#94a3b8;text-transform:uppercase;letter-spacing:.8px;margin-bottom:4px;'>Laboratorio</div><div style='font-size:14px;font-weight:600;color:#1e293b;'>" + examen.get('laboratorio','') + "</div></div>" if examen.get('laboratorio') else ''}
    </div>
  </div>

  <!-- Banner riesgo -->
  <div style="background:{rc['bg']};border:1px solid {rc['border']};border-left:5px solid {rc['tc']};border-radius:10px;padding:15px 20px;margin-bottom:24px;display:flex;align-items:center;gap:14px;">
    <div style="flex:1;">
      <div style="font-weight:700;font-size:15px;color:{rc['tc']};">Nivel de riesgo: {riesgo.capitalize()}</div>
      <div style="font-size:13px;color:{rc['tc']};opacity:.85;margin-top:3px;">{rc['txt']}</div>
    </div>
  </div>

  {"<div style='background:#fff;border:1px solid #e2e8f0;border-radius:10px;padding:18px 22px;margin-bottom:24px;'><div style='font-size:12px;font-weight:700;color:#4f46e5;text-transform:uppercase;letter-spacing:.8px;margin-bottom:10px;'>Resumen del análisis</div><p style='font-size:14px;color:#334155;line-height:1.75;margin:0;'>" + examen.get('resumen','') + "</p></div>" if examen.get('resumen') else ''}

  <!-- Tabla indicadores -->
  <div style="margin-bottom:30px;">
    <div style="font-size:13px;font-weight:700;color:#4f46e5;text-transform:uppercase;letter-spacing:.8px;margin-bottom:12px;">Indicadores detectados ({len(indicadores)})</div>
    <div style="border:1px solid #e2e8f0;border-radius:10px;overflow:hidden;">
      <table>
        <thead>
          <tr style="background:#f1f5f9;">
            <th style="padding:10px 14px;font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.6px;border-bottom:1px solid #e2e8f0;font-weight:600;">Indicador</th>
            <th style="padding:10px 14px;font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.6px;border-bottom:1px solid #e2e8f0;font-weight:600;">Valor</th>
            <th style="padding:10px 14px;font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.6px;border-bottom:1px solid #e2e8f0;font-weight:600;">Referencia</th>
            <th style="padding:10px 14px;font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.6px;border-bottom:1px solid #e2e8f0;font-weight:600;">Estado</th>
            <th style="padding:10px 14px;font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.6px;border-bottom:1px solid #e2e8f0;font-weight:600;">¿Qué mide?</th>
          </tr>
        </thead>
        <tbody style="background:#fff;">{rows_ind}</tbody>
      </table>
    </div>
  </div>

  {positive_section_html}

  {"<div style='margin-bottom:30px;'><div style='font-size:13px;font-weight:700;color:#4f46e5;text-transform:uppercase;letter-spacing:.8px;margin-bottom:12px;'>Recomendaciones</div>" + recs_html + "</div>" if recs_html else ''}

  {preguntas_html}

  {_build_plan_html_reporte(plan)}

  <!-- Footer -->
  <div style="border-top:1px solid #e2e8f0;padding-top:20px;text-align:center;">
    <p style="font-size:12px;color:#94a3b8;line-height:1.9;margin:0;">
      Este reporte es <strong style="color:#64748b;">orientativo</strong> y no reemplaza la consulta médica presencial.<br>
      Ante cualquier síntoma o duda, consulte a un profesional de salud calificado.<br>
      <span style="color:#cbd5e1;">Generado por <strong style="color:#4f46e5;">VitalIA — Dr. Digital</strong> · {fecha_reporte}</span>
    </p>
  </div>

</div></body></html>"""

    return html


# ── Email helper ─────────────────────────────────────────────────────────────
def _send_reset_email(to_email: str, nombre: str, token: str) -> bool:
    """Envía email con enlace de recuperación. Retorna True si OK."""
    reset_url = f"{APP_URL}/reset-password/{token}"
    expiry_min = 60

    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"></head><body
      style="margin:0;padding:0;background:#f8fafc;font-family:'Segoe UI',Arial,sans-serif;">
      <div style="max-width:520px;margin:40px auto;background:#fff;border-radius:16px;
                  box-shadow:0 4px 24px rgba(0,0,0,0.08);overflow:hidden;">
        <div style="background:linear-gradient(135deg,#0ea5e9,#8b5cf6);padding:32px 36px;">
          <div style="font-size:26px;font-weight:800;color:#fff;letter-spacing:-.5px;">🩺 VitalIA</div>
          <div style="color:rgba(255,255,255,0.85);font-size:14px;margin-top:4px;">Recuperación de contraseña</div>
        </div>
        <div style="padding:36px;">
          <p style="font-size:16px;font-weight:600;color:#1e293b;margin:0 0 12px;">Hola, {nombre} 👋</p>
          <p style="font-size:14px;color:#475569;line-height:1.7;margin:0 0 24px;">
            Recibimos una solicitud para restablecer la contraseña de tu cuenta VitalIA.
            Haz clic en el botón para crear una nueva contraseña:
          </p>
          <div style="text-align:center;margin-bottom:28px;">
            <a href="{reset_url}"
               style="display:inline-block;background:linear-gradient(135deg,#0ea5e9,#0284c7);
                      color:#fff;text-decoration:none;padding:14px 36px;border-radius:12px;
                      font-size:15px;font-weight:700;letter-spacing:.2px;">
              Restablecer contraseña
            </a>
          </div>
          <div style="background:#f1f5f9;border-radius:10px;padding:14px 16px;
                      font-size:12px;color:#64748b;line-height:1.7;">
            <strong style="color:#334155;">⚠ Importante:</strong><br>
            • Este enlace expira en <strong>{expiry_min} minutos</strong>.<br>
            • Si no solicitaste este cambio, ignora este correo; tu contraseña no cambiará.<br>
            • Por seguridad, el enlace solo puede usarse una vez.
          </div>
          <p style="font-size:11px;color:#94a3b8;margin-top:24px;line-height:1.6;">
            Si el botón no funciona, copia y pega este enlace en tu navegador:<br>
            <span style="color:#0ea5e9;word-break:break-all;">{reset_url}</span>
          </p>
        </div>
        <div style="background:#f8fafc;border-top:1px solid #e2e8f0;padding:16px 36px;
                    text-align:center;font-size:11px;color:#94a3b8;">
          VitalIA — Análisis médico con inteligencia artificial &nbsp;·&nbsp;
          Este es un correo automático, por favor no respondas.
        </div>
      </div>
    </body></html>"""

    if not MAIL_USERNAME or not MAIL_PASSWORD:
        # Modo dev: imprimir enlace en consola
        print(f"\n[VitalIA] RESET LINK (dev mode — configura MAIL_USERNAME/MAIL_PASSWORD):\n{reset_url}\n")
        return True

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "Restablecer contraseña — VitalIA"
        msg["From"]    = f"VitalIA <{MAIL_FROM}>"
        msg["To"]      = to_email
        msg.attach(MIMEText(
            f"Hola {nombre},\n\nRestablece tu contraseña aquí: {reset_url}\n\n"
            f"Este enlace expira en {expiry_min} minutos.\n\nVitalIA",
            "plain", "utf-8"
        ))
        msg.attach(MIMEText(html, "html", "utf-8"))

        with smtplib.SMTP(MAIL_SERVER, MAIL_PORT, timeout=10) as s:
            s.ehlo()
            s.starttls()
            s.login(MAIL_USERNAME, MAIL_PASSWORD)
            s.sendmail(MAIL_FROM, [to_email], msg.as_string())
        return True
    except Exception as e:
        print(f"[VitalIA] Error enviando email: {e}")
        return False

# ── Rutas: Admin ─────────────────────────────────────────────────────────────
@app.route("/api/admin/stats")
@login_required
def admin_stats():
    if not is_admin():
        return jsonify({"error": "Acceso denegado"}), 403
    with get_db() as db:
        # Cuentas
        total_usuarios  = db.execute("SELECT COUNT(*) FROM usuarios WHERE es_admin=0").fetchone()[0]
        nuevos_mes      = db.execute("SELECT COUNT(*) FROM usuarios WHERE es_admin=0 AND creado_en >= date('now','start of month')").fetchone()[0]
        # Exámenes
        total_examenes  = db.execute("SELECT COUNT(*) FROM examenes").fetchone()[0]
        examenes_mes    = db.execute("SELECT COUNT(*) FROM examenes WHERE creado_en >= date('now','start of month')").fetchone()[0]
        alto_riesgo     = db.execute("SELECT COUNT(*) FROM examenes WHERE riesgo IN ('alto','critico')").fetchone()[0]
        # Ingresos (monto en centavos → USD)
        rec_total       = db.execute("SELECT COALESCE(SUM(monto),0) FROM pagos WHERE estado='completado'").fetchone()[0]
        rec_mes         = db.execute("SELECT COALESCE(SUM(monto),0) FROM pagos WHERE estado='completado' AND creado_en >= date('now','start of month')").fetchone()[0]
        transacciones   = db.execute("SELECT COUNT(*) FROM pagos WHERE estado='completado'").fetchone()[0]
        # Chatbot
        total_msgs      = db.execute("SELECT COUNT(*) FROM mensajes WHERE rol='user'").fetchone()[0]
        msgs_mes        = db.execute("SELECT COUNT(*) FROM mensajes WHERE rol='user' AND creado_en >= date('now','start of month')").fetchone()[0]
        # Descargas
        total_descargas = db.execute("SELECT COALESCE(SUM(descargas),0) FROM examenes").fetchone()[0]
        # Usuarios recurrentes (≥2 exámenes)
        recurrentes     = db.execute("""
            SELECT COUNT(*) FROM (
                SELECT p.usuario_id FROM examenes e
                JOIN pacientes p ON e.paciente_id=p.id
                JOIN usuarios u ON p.usuario_id=u.id
                WHERE u.es_admin=0
                GROUP BY p.usuario_id HAVING COUNT(*)>=2
            )""").fetchone()[0]
        # Planes de acción generados
        planes          = db.execute("SELECT COUNT(*) FROM planes_accion").fetchone()[0]
        # Actividad últimos 30 días (exámenes por día)
        actividad = [dict(r) for r in db.execute("""
            SELECT date(creado_en) dia, COUNT(*) cantidad
            FROM examenes WHERE creado_en >= date('now','-29 days')
            GROUP BY dia ORDER BY dia""").fetchall()]
        # Exámenes por tipo
        por_tipo = [dict(r) for r in db.execute("""
            SELECT tipo, COUNT(*) cantidad FROM examenes
            GROUP BY tipo ORDER BY cantidad DESC""").fetchall()]
        # Exámenes por riesgo
        por_riesgo = [dict(r) for r in db.execute("""
            SELECT riesgo, COUNT(*) cantidad FROM examenes
            GROUP BY riesgo ORDER BY cantidad DESC""").fetchall()]
        # Últimos usuarios registrados
        ultimos_usuarios = [dict(r) for r in db.execute("""
            SELECT id, nombre, email, creado_en FROM usuarios
            WHERE es_admin=0 ORDER BY creado_en DESC LIMIT 10""").fetchall()]
        # Usuarios más activos
        top_usuarios = [dict(r) for r in db.execute("""
            SELECT u.nombre, u.email, COUNT(e.id) examenes,
                   COALESCE(SUM(e.descargas),0) descargas,
                   MAX(e.creado_en) ultimo_examen
            FROM usuarios u
            JOIN pacientes p ON p.usuario_id=u.id
            JOIN examenes e ON e.paciente_id=p.id
            WHERE u.es_admin=0
            GROUP BY u.id ORDER BY examenes DESC LIMIT 10""").fetchall()]
        # Exámenes de alto riesgo recientes
        examenes_riesgo = [dict(r) for r in db.execute("""
            SELECT e.id, e.titulo, e.tipo, e.riesgo, e.creado_en, u.nombre usuario, u.email
            FROM examenes e
            JOIN pacientes p ON e.paciente_id=p.id
            JOIN usuarios u ON p.usuario_id=u.id
            WHERE e.riesgo IN ('alto','critico')
            ORDER BY e.creado_en DESC LIMIT 8""").fetchall()]
        # Ingresos por día (últimos 30 días)
        ingresos_dia = [dict(r) for r in db.execute("""
            SELECT date(creado_en) dia, COALESCE(SUM(monto),0) monto
            FROM pagos WHERE estado='completado' AND creado_en >= date('now','-29 days')
            GROUP BY dia ORDER BY dia""").fetchall()]

    return jsonify({
        "cuentas":         {"total": total_usuarios, "mes": nuevos_mes},
        "examenes":        {"total": total_examenes, "mes": examenes_mes, "alto_riesgo": alto_riesgo},
        "ingresos":        {"total_cents": rec_total, "mes_cents": rec_mes, "transacciones": transacciones},
        "chatbot":         {"total": total_msgs, "mes": msgs_mes},
        "descargas":       int(total_descargas),
        "recurrentes":     recurrentes,
        "planes":          planes,
        "actividad":       actividad,
        "por_tipo":        por_tipo,
        "por_riesgo":      por_riesgo,
        "ultimos_usuarios": ultimos_usuarios,
        "top_usuarios":    top_usuarios,
        "examenes_riesgo": examenes_riesgo,
        "ingresos_dia":    ingresos_dia,
    })

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

        # Campos clínicos opcionales
        edad        = d.get("edad", "") or None
        peso        = d.get("peso", "") or None
        genero      = (d.get("genero","") or "").strip()
        condiciones = (d.get("condiciones","") or "").strip()
        medicamentos= (d.get("medicamentos","") or "").strip()
        sintomas    = (d.get("sintomas","") or "").strip()
        try:
            edad = int(edad) if edad else None
        except (ValueError, TypeError):
            edad = None
        try:
            peso = float(str(peso).replace(",",".")) if peso else None
        except (ValueError, TypeError):
            peso = None

        try:
            ip_addr    = request.headers.get("X-Forwarded-For", request.remote_addr or "").split(",")[0].strip()
            user_agent = request.headers.get("User-Agent", "")[:512]
            with get_db() as db:
                # Crear usuario
                cur = db.execute(
                    "INSERT INTO usuarios (email, password_hash, nombre) VALUES (?,?,?)",
                    (email, generate_password_hash(pwd), nombre)
                )
                uid = cur.lastrowid
                # Crear perfil de paciente con datos clínicos
                db.execute(
                    """INSERT INTO pacientes
                       (usuario_id, nombre, email, genero, edad, peso, condiciones, medicamentos, sintomas)
                       VALUES (?,?,?,?,?,?,?,?,?)""",
                    (uid, nombre, email, genero, edad, peso, condiciones, medicamentos, sintomas)
                )
                # Registrar consentimiento informado (cláusula 9 — evidencia de aceptación)
                db.execute(
                    "INSERT INTO consentimientos (usuario_id, version, ip, user_agent) VALUES (?,?,?,?)",
                    (uid, "2.0", ip_addr, user_agent)
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
            # Incrementar contador de intentos fallidos
            intentos = 0
            if u:
                with get_db() as db:
                    db.execute(
                        "UPDATE usuarios SET intentos_fallidos = COALESCE(intentos_fallidos,0)+1 WHERE id=?",
                        (u["id"],)
                    )
                    db.commit()
                    intentos = (u["intentos_fallidos"] or 0) + 1
            msg = "Email o contraseña incorrectos"
            if intentos >= 2:
                msg = f"Contraseña incorrecta ({intentos+1}° intento). ¿Olvidaste tu contraseña?"
            return jsonify({"error": msg, "intentos": intentos + 1}), 401

        # Login correcto → resetear contador
        with get_db() as db:
            db.execute("UPDATE usuarios SET intentos_fallidos=0 WHERE id=?", (u["id"],))
            db.commit()

        session["usuario_id"]     = u["id"]
        session["usuario_nombre"] = u["nombre"]
        session["usuario_email"]  = u["email"]
        session["es_admin"]       = bool(u["es_admin"])
        return jsonify({"ok": True, "redirect": "/app"})

    return render_template("login.html")


@app.route("/forgot-password", methods=["GET","POST"])
def forgot_password():
    if request.method == "POST":
        d = request.json or request.form
        email = (d.get("email","")).strip().lower()
        if not email:
            return jsonify({"error": "Ingresa tu email"}), 400

        with get_db() as db:
            u = db.execute("SELECT * FROM usuarios WHERE email=? AND activo=1", (email,)).fetchone()

        # Siempre responder OK para no revelar si el email existe (seguridad)
        if u:
            token  = secrets.token_urlsafe(48)
            expiry = (datetime.now() + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
            with get_db() as db:
                db.execute(
                    "UPDATE usuarios SET reset_token=?, reset_token_expiry=? WHERE id=?",
                    (token, expiry, u["id"])
                )
                db.commit()
            _send_reset_email(u["email"], u["nombre"], token)

        return jsonify({"ok": True})

    return render_template("forgot_password.html")


@app.route("/reset-password/<token>", methods=["GET","POST"])
def reset_password(token):
    with get_db() as db:
        u = db.execute(
            "SELECT * FROM usuarios WHERE reset_token=? AND activo=1", (token,)
        ).fetchone()

    # Validar token y expiración
    if not u:
        return render_template("reset_password.html", error="El enlace no es válido o ya fue utilizado.", token=None)
    expiry = u["reset_token_expiry"]
    if not expiry or datetime.now() > datetime.strptime(expiry, "%Y-%m-%d %H:%M:%S"):
        return render_template("reset_password.html", error="El enlace ha expirado. Solicita uno nuevo.", token=None)

    if request.method == "POST":
        d = request.json or request.form
        nueva  = d.get("password","")
        nueva2 = d.get("password2","")
        if len(nueva) < 8:
            return jsonify({"error": "La contraseña debe tener al menos 8 caracteres"}), 400
        if nueva != nueva2:
            return jsonify({"error": "Las contraseñas no coinciden"}), 400

        with get_db() as db:
            db.execute(
                "UPDATE usuarios SET password_hash=?, reset_token=NULL, "
                "reset_token_expiry=NULL, intentos_fallidos=0 WHERE id=?",
                (generate_password_hash(nueva), u["id"])
            )
            db.commit()
        return jsonify({"ok": True})

    return render_template("reset_password.html", error=None, token=token)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("landing"))


# ── Ruta principal ────────────────────────────────────────────────────────────
@app.route("/")
def landing():
    return redirect("/brochure")

@app.route("/terminos")
def terminos():
    return render_template("terminos.html")


@app.route("/brochure")
def brochure():
    return render_template("brochure.html")


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
            try:
                edad = int(d["edad"]) if d.get("edad") not in (None, "", "null") else None
            except (ValueError, TypeError):
                edad = None
            try:
                peso = float(str(d["peso"]).replace(",",".")) if d.get("peso") not in (None, "", "null") else None
            except (ValueError, TypeError):
                peso = None
            db.execute("""UPDATE pacientes
                          SET nombre=?,fecha_nac=?,genero=?,notas=?,
                              edad=?,peso=?,condiciones=?,medicamentos=?,sintomas=?
                          WHERE usuario_id=?""",
                       (d.get("nombre",""), d.get("fecha_nac",""),
                        d.get("genero",""), d.get("notas",""),
                        edad, peso,
                        d.get("condiciones",""), d.get("medicamentos",""), d.get("sintomas",""),
                        uid))
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
        plan_row       = db.execute("SELECT * FROM planes_accion WHERE examen_id=?", (eid,)).fetchone()
    examen_dict = dict(e)
    try:
        examen_dict["preguntas_doctor"] = json.loads(examen_dict.get("preguntas_doctor") or "[]")
    except Exception:
        examen_dict["preguntas_doctor"] = []
    plan = None
    if plan_row:
        plan = {}
        for key in ("nutricion", "movimiento", "habitos", "predicciones"):
            try:
                plan[key] = json.loads(plan_row[key] or ("[]" if key == "predicciones" else "{}"))
            except Exception:
                plan[key] = [] if key == "predicciones" else {}
    return jsonify({
        "examen": examen_dict,
        "indicadores": [dict(i) for i in indicadores],
        "recomendaciones": [dict(r) for r in recomendaciones],
        "conversaciones": [dict(c) for c in conversaciones],
        "plan": plan,
    })


@app.route("/api/examenes/<int:eid>/reporte")
@login_required
def reporte_examen(eid):
    paciente = get_current_paciente()
    if not paciente:
        return "No autorizado", 401
    with get_db() as db:
        e = db.execute("SELECT * FROM examenes WHERE id=? AND paciente_id=?",
                       (eid, paciente["id"])).fetchone()
        if not e:
            return "Examen no encontrado", 404
        indicadores     = db.execute("SELECT * FROM indicadores WHERE examen_id=? ORDER BY id", (eid,)).fetchall()
        recomendaciones = db.execute("SELECT * FROM recomendaciones WHERE examen_id=? ORDER BY prioridad DESC", (eid,)).fetchall()
        plan_row        = db.execute("SELECT * FROM planes_accion WHERE examen_id=?", (eid,)).fetchone()
    try:
        preguntas = json.loads(dict(e).get("preguntas_doctor") or "[]")
    except Exception:
        preguntas = []
    plan = None
    if plan_row:
        plan = {}
        for key in ("nutricion", "movimiento", "habitos", "predicciones"):
            try:
                plan[key] = json.loads(plan_row[key] or ("[]" if key == "predicciones" else "{}"))
            except Exception:
                plan[key] = [] if key == "predicciones" else {}
    html = _build_reporte_html(
        nombre_paciente=paciente.get("nombre","Paciente"),
        examen=dict(e),
        indicadores=[dict(i) for i in indicadores],
        recomendaciones=[dict(r) for r in recomendaciones],
        preguntas=preguntas,
        plan=plan,
    )
    with get_db() as db:
        db.execute("UPDATE examenes SET descargas = COALESCE(descargas,0)+1 WHERE id=?", (eid,))
        db.commit()
    from flask import Response as FlaskResponse
    return FlaskResponse(html, mimetype="text/html")


def _delete_examenes_by_ids(db, ids: list):
    """Elimina exámenes y todos sus datos relacionados dado una lista de IDs."""
    if not ids:
        return
    ph = ",".join("?" * len(ids))
    # Mensajes → conversaciones (sin CASCADE en la FK de conversaciones)
    convs = db.execute(f"SELECT id FROM conversaciones WHERE examen_id IN ({ph})", ids).fetchall()
    if convs:
        conv_ids = [c["id"] for c in convs]
        cph = ",".join("?" * len(conv_ids))
        db.execute(f"DELETE FROM mensajes WHERE conversacion_id IN ({cph})", conv_ids)
    db.execute(f"DELETE FROM conversaciones WHERE examen_id IN ({ph})", ids)
    # pagos no tiene CASCADE
    db.execute(f"DELETE FROM pagos WHERE examen_id IN ({ph})", ids)
    # Estos tienen CASCADE pero los borramos explícitamente por seguridad
    db.execute(f"DELETE FROM indicadores WHERE examen_id IN ({ph})", ids)
    db.execute(f"DELETE FROM recomendaciones WHERE examen_id IN ({ph})", ids)
    db.execute(f"DELETE FROM planes_accion WHERE examen_id IN ({ph})", ids)
    db.execute(f"DELETE FROM examenes WHERE id IN ({ph})", ids)
    db.commit()


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
        _delete_examenes_by_ids(db, [eid])
    return jsonify({"ok": True})


@app.route("/api/examenes/bulk-delete", methods=["POST"])
@login_required
def bulk_delete_examenes():
    paciente = get_current_paciente()
    if not paciente:
        return jsonify({"error": "No autorizado"}), 403
    data = request.json or {}
    raw_ids = data.get("ids") or []
    try:
        ids = [int(i) for i in raw_ids]
    except (ValueError, TypeError):
        return jsonify({"error": "IDs inválidos"}), 400
    if not ids:
        return jsonify({"error": "Sin IDs"}), 400
    pid = paciente["id"]
    with get_db() as db:
        ph = ",".join("?" * len(ids))
        rows = db.execute(
            f"SELECT id FROM examenes WHERE id IN ({ph}) AND paciente_id=?",
            ids + [pid]
        ).fetchall()
        valid_ids = [r["id"] for r in rows]
        if not valid_ids:
            return jsonify({"ok": True, "deleted": 0})
        _delete_examenes_by_ids(db, valid_ids)
    return jsonify({"ok": True, "deleted": len(valid_ids)})


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
        update("pending", msg="Leyendo archivo...", progress=5)
        gc = _gemini_client(api_key)

        # ── Cargar perfil clínico del paciente ───────────────────────────────
        with get_db() as db:
            pac_row = db.execute("SELECT * FROM pacientes WHERE id=?", (paciente_id,)).fetchone()
        paciente_dict = dict(pac_row) if pac_row else {}
        patient_ctx = _build_patient_ctx(paciente_dict)

        # ── Preparar contenido según tipo de archivo ──────────────────────────
        if ext == ".pdf":
            update("pending", msg="Leyendo PDF...", progress=10)
            with open(str(ruta), "rb") as pdf_f:
                pdf_bytes = pdf_f.read()
            file_part = genai_types.Part(
                inline_data=genai_types.Blob(
                    mime_type="application/pdf",
                    data=pdf_bytes
                )
            )
        else:
            update("pending", msg="Preparando imagen...", progress=10)
            file_part = PIL.Image.open(str(ruta))

        update("pending", msg="Extrayendo indicadores con visión IA...", progress=20)
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

        # ── Interpretación + preguntas en una sola llamada ────────────────────
        update("pending", msg=f"Analizando {len(indicadores)} indicadores...", progress=60)
        tiene_alertas = len(alertas) > 0
        lineas_alertas = "\n".join([
            f"- {i['nombre']}: {i['valor']} {i.get('unidad','')} [{i.get('estado','').upper()}] (ref: {i.get('rango_ref','')})"
            for i in alertas
        ]) if tiene_alertas else "Ninguno"

        prompt_combined = f"""Como medico experto, interpreta estos resultados y genera preguntas personalizadas para el paciente.

PERFIL DEL PACIENTE:
{patient_ctx}

TIPO DE EXAMEN: {ocr_data.get('tipo_examen', tipo)}
INDICADORES:
{json.dumps(indicadores, ensure_ascii=False, indent=2)}

OBSERVACIONES DEL LABORATORIO: {ocr_data.get('observaciones_laboratorio','')}
INDICADORES ALTERADOS: {lineas_alertas}

INSTRUCCIONES:
- Personaliza la interpretacion segun la edad, genero, peso y condiciones medicas del paciente
- Si tiene condiciones cronicas, relaciona los indicadores con esas condiciones
- Si toma medicamentos, considera su efecto en los valores
- Adapta las recomendaciones al perfil especifico (no recomendaciones genericas)
- preguntas_doctor: genera 6-9 preguntas en primera persona si hay indicadores alterados, array vacio si todos son normales
- Las preguntas deben ser especificas ("¿Debo ajustar mi dosis de metformina?" no "¿Debo cambiar medicamentos?")

Genera UNICAMENTE este JSON (sin markdown):
{{
  "resumen": "Resumen ejecutivo personalizado en 2-3 oraciones",
  "interpretacion": "Interpretacion detallada considerando el perfil del paciente (markdown)",
  "recomendaciones": [
    {{"tipo": "dieta|ejercicio|medicamento|consulta|estilo_vida", "titulo": "...", "descripcion": "...", "prioridad": "alta|media|baja"}}
  ],
  "alertas_principales": ["indicadores preocupantes con explicacion personalizada"],
  "puntos_positivos": ["aspectos de salud que estan bien"],
  "seguimiento": "cuando y con que especialista consultar segun el perfil",
  "preguntas_doctor": ["¿pregunta 1?", "¿pregunta 2?"]
}}"""

        try:
            resp_combined = _generate(gc, prompt_combined)
            raw_combined = re.sub(r"```json\s*|\s*```", "", resp_combined.text.strip()).strip()
            combined_data = json.loads(raw_combined)
            interp_data = {k: combined_data[k] for k in ("resumen","interpretacion","recomendaciones","alertas_principales","puntos_positivos","seguimiento") if k in combined_data}
            preguntas_doctor = combined_data.get("preguntas_doctor", [])
            if not isinstance(preguntas_doctor, list):
                preguntas_doctor = []
            preguntas_doctor = [str(p) for p in preguntas_doctor if p][:8]
        except Exception:
            interp_data = {
                "resumen": "Analisis completado.",
                "interpretacion": "Revise los indicadores detallados y consulte con su medico.",
                "recomendaciones": [],
                "alertas_principales": [f"{i['nombre']}: {i['valor']}" for i in alertas],
                "puntos_positivos": [],
                "seguimiento": "Consulte con su medico tratante."
            }
            preguntas_doctor = []

        update("pending", msg="Guardando resultados...", progress=85)
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

        # ── Generar plan de acción personalizado (solo si hay anomalías) ─────
        tiene_plan = False
        if alertas:
            update("pending", msg="Creando tu plan de acción personalizado...", progress=92)
            plan_result = _generate_and_save_plan(
                gc, eid, patient_ctx, indicadores, alertas, interp_data, ocr_data, tipo)
            tiene_plan = bool(plan_result)

        update("done", ok=True, examen_id=eid, conversacion_id=conv_id,
               resumen=interp_data.get("resumen",""), riesgo=riesgo,
               num_indicadores=len(indicadores), alertas=len(alertas), criticos=len(criticos),
               tiene_plan=tiene_plan, progress=100)

    except Exception as e:
        import traceback; print("[VitalIA ERROR]", traceback.format_exc())
        update("error", error=str(e))


def _generate_and_save_plan(gc, examen_id, patient_ctx, indicadores, alertas, interp_data, ocr_data, tipo):
    """Genera el plan de acción de 3 capas con IA y lo guarda en planes_accion."""
    lineas_alertas = "\n".join([
        f"- {i['nombre']}: {i['valor']} {i.get('unidad','')} [{i.get('estado','').upper()}] (ref: {i.get('rango_ref','')})"
        for i in alertas
    ])
    prompt = f"""Eres un coach de salud integrativo especializado en medicina preventiva basada en evidencia.
Crea un plan de acción personalizado de 3 capas para este paciente, conectado directamente a sus biomarcadores.

PERFIL DEL PACIENTE:
{patient_ctx}

TIPO DE EXAMEN: {ocr_data.get('tipo_examen', tipo)}
INDICADORES FUERA DE RANGO:
{lineas_alertas}

DIAGNÓSTICO RESUMIDO:
{interp_data.get('resumen', '')}

REGLAS:
- Usa lenguaje probabilístico: "podría", "puede ayudar a", "tiende a", "la evidencia sugiere"
- Conecta SIEMPRE cada recomendación con los biomarcadores específicos del paciente
- Explica el "por qué" de forma educativa pero simple
- Predicciones realistas basadas en evidencia (no exagerar; rangos conservadores)
- Mencionar siempre la importancia del médico tratante
- 3-5 items en que_comer, 3-4 items en que_evitar, 2-3 actividades físicas, 2-3 predicciones

Genera ÚNICAMENTE este JSON (sin markdown, sin texto adicional):
{{
  "nutricion": {{
    "titulo": "Título motivador del plan nutricional (máx 8 palabras)",
    "objetivo": "Objetivo específico conectado a los biomarcadores alterados",
    "patron_dieta": "Nombre del patrón dietético base (ej: Mediterránea, DASH, anti-inflamatoria)",
    "que_comer": [
      {{"emoji": "🌾", "alimento": "nombre", "razon": "Podría ayudar a [indicador] porque contiene [componente]", "frecuencia": "X veces/semana o diario"}}
    ],
    "que_evitar": [
      {{"emoji": "⚠️", "alimento": "nombre", "razon": "Tiende a elevar/empeorar [indicador] porque...", "impacto": "alto|medio|bajo"}}
    ],
    "dia_ejemplo": "Desayuno: ...\\nAlmuerzo: ...\\nCena: ...\\nSnack: ...",
    "nota": "Siempre consultar con médico o nutricionista antes de cambios significativos"
  }},
  "movimiento": {{
    "titulo": "Título motivador del plan de movimiento",
    "objetivo": "Objetivo conectado a los biomarcadores",
    "actividades": [
      {{
        "tipo": "cardio|fuerza|hiit|flexibilidad",
        "emoji": "🚶",
        "nombre": "Nombre de la actividad",
        "frecuencia": "X días/semana",
        "duracion": "X-Y minutos/sesión",
        "intensidad": "suave|moderada|alta",
        "beneficio": "Podría mejorar [indicador] aproximadamente [X%] en [plazo] si se practica regularmente",
        "cuando": "Mejor momento del día"
      }}
    ],
    "progresion": "Cómo ir aumentando gradualmente la intensidad en 8-12 semanas",
    "contraindicaciones": "Lo que debe evitar dado el perfil específico",
    "nota": "Consultar con médico antes de iniciar programa de ejercicio intenso"
  }},
  "habitos": {{
    "sueno": {{
      "objetivo": "X-Y horas por noche",
      "por_que": "El sueño insuficiente podría elevar [hormona/indicador] lo que...",
      "conexion": "Tu [indicador específico] podría verse afectado por la falta de sueño porque...",
      "tips": ["tip concreto 1", "tip concreto 2", "tip concreto 3"]
    }},
    "estres": {{
      "impacto": "El estrés crónico tiende a elevar el cortisol, lo que podría...",
      "conexion": "Esto podría afectar directamente tus niveles de [indicador]...",
      "tecnicas": ["Técnica 1 con instrucciones breves", "Técnica 2 con instrucciones"]
    }},
    "alcohol": {{
      "recomendacion": "Recomendación específica (cantidad máxima o abstención si aplica)",
      "por_que": "El alcohol tiende a [efecto metabólico específico]...",
      "conexion": "Tu [indicador] podría mejorar notablemente al reducir el consumo porque..."
    }},
    "otros": "Otras 1-2 recomendaciones de hábitos específicas para este caso (hidratación, tabaco si aplica, etc.)"
  }},
  "predicciones": [
    {{
      "indicador": "Nombre exacto del indicador (igual que en el examen)",
      "valor_actual": 0.0,
      "unidad": "unidad de medida",
      "rango_normal": "<X o X-Y",
      "mejora_esperada": "X-Y%",
      "valor_estimado": "rango estimado tras seguir el plan",
      "plazo": "X meses",
      "condicion": "Si sigues [acción específica del plan]...",
      "evidencia": "La evidencia científica muestra que [patrón/ejercicio] puede reducir [indicador] en [rango]"
    }}
  ]
}}"""
    try:
        resp = _generate(gc, prompt)
        raw = re.sub(r"```json\s*|\s*```", "", resp.text.strip()).strip()
        plan_data = json.loads(raw)
    except Exception as e:
        print(f"[VitalIA] Plan generation failed: {e}")
        return None
    try:
        with get_db() as db:
            db.execute("""INSERT OR REPLACE INTO planes_accion
                (examen_id, nutricion, movimiento, habitos, predicciones)
                VALUES (?, ?, ?, ?, ?)""",
                (examen_id,
                 json.dumps(plan_data.get("nutricion", {}), ensure_ascii=False),
                 json.dumps(plan_data.get("movimiento", {}), ensure_ascii=False),
                 json.dumps(plan_data.get("habitos", {}), ensure_ascii=False),
                 json.dumps(plan_data.get("predicciones", []), ensure_ascii=False)))
            db.commit()
        return examen_id
    except Exception as e:
        print(f"[VitalIA] Plan save failed: {e}")
        return None


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


# ── API: Evolución médica ─────────────────────────────────────────────────────
@app.route("/api/paciente/evolucion")
@login_required
def get_evolucion():
    paciente = get_current_paciente()
    if not paciente:
        return jsonify({"error": "No encontrado"}), 404
    pid = paciente["id"]
    with get_db() as db:
        exams = db.execute(
            "SELECT id, titulo, tipo, fecha_examen, creado_en, riesgo FROM examenes "
            "WHERE paciente_id=? ORDER BY id ASC", (pid,)
        ).fetchall()
        if not exams:
            return jsonify({"tiene_datos": False, "series": [], "examenes": []})
        exam_ids = [e["id"] for e in exams]
        placeholders = ",".join("?" * len(exam_ids))
        all_inds = db.execute(
            f"SELECT examen_id, nombre, valor, unidad, rango_min, rango_max, estado "
            f"FROM indicadores WHERE examen_id IN ({placeholders}) ORDER BY examen_id, id",
            exam_ids
        ).fetchall()

    exams_list = []
    for e in exams:
        fecha = (e["fecha_examen"] or e["creado_en"] or "")[:10]
        exams_list.append({"id": e["id"], "titulo": e["titulo"], "fecha": fecha, "riesgo": e["riesgo"]})
    exam_date = {e["id"]: e["fecha"] for e in exams_list}

    # Group by exam
    inds_by_exam = {}
    for ind in all_inds:
        eid = ind["examen_id"]
        inds_by_exam.setdefault(eid, []).append(ind)

    # Count appearances per indicator name (normalized)
    name_count = {}
    name_meta = {}
    for eid, inds in inds_by_exam.items():
        seen = set()
        for ind in inds:
            key = ind["nombre"].strip().lower()
            if key in seen:
                continue
            seen.add(key)
            name_count[key] = name_count.get(key, 0) + 1
            if key not in name_meta:
                name_meta[key] = {
                    "nombre": ind["nombre"],
                    "unidad": ind["unidad"],
                    "rango_min": ind["rango_min"],
                    "rango_max": ind["rango_max"],
                }

    common = {k for k, v in name_count.items() if v >= 2}
    if not common:
        return jsonify({
            "tiene_datos": True,
            "necesita_mas_examenes": len(exams) < 2,
            "sin_indicadores_comunes": True,
            "examenes": exams_list,
            "series": [],
            "resumen": {"mejorados": [], "empeorados": [], "estables": []}
        })

    # Build time series per common indicator
    series = []
    for name_key in sorted(common):
        meta = name_meta[name_key]
        datos = []
        for exam in exams_list:
            eid = exam["id"]
            for ind in inds_by_exam.get(eid, []):
                if ind["nombre"].strip().lower() == name_key:
                    val_str = str(ind["valor"]).replace("~", "").replace(",", ".").strip()
                    try:
                        val = float(val_str)
                        datos.append({"examen_id": eid, "fecha": exam_date[eid],
                                      "valor": val, "estado": ind["estado"]})
                    except Exception:
                        pass
                    break
        if len(datos) >= 2:
            series.append({
                "nombre": meta["nombre"], "unidad": meta["unidad"],
                "rango_min": meta["rango_min"], "rango_max": meta["rango_max"],
                "datos": datos
            })

    # Compute summary
    mejorados, empeorados, estables = [], [], []
    for s in series:
        d = s["datos"]
        primero, ultimo = d[0]["estado"], d[-1]["estado"]
        if primero != "normal" and ultimo == "normal":
            mejorados.append(s["nombre"])
        elif primero == "normal" and ultimo != "normal":
            empeorados.append(s["nombre"])
        else:
            estables.append(s["nombre"])

    return jsonify({
        "tiene_datos": True,
        "necesita_mas_examenes": False,
        "sin_indicadores_comunes": False,
        "examenes": exams_list,
        "series": series,
        "resumen": {"mejorados": mejorados, "empeorados": empeorados, "estables": estables}
    })


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
