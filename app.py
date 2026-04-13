"""
VitalIA — Dr. Digital
Análisis inteligente de exámenes médicos con IA
"""
import os, sys, json, sqlite3, base64, re, uuid, threading
from pathlib import Path
from datetime import datetime
from functools import wraps

# ── In-memory job store ───────────────────────────────────────────────────────
_jobs: dict = {}   # job_id → {"status": "pending|done|error", "result": {...}}
_jobs_lock = threading.Lock()

from flask import (Flask, render_template, request, jsonify,
                   Response, stream_with_context, redirect, url_for)
from werkzeug.utils import secure_filename
import anthropic

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
            # Verificar que es escribible
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
PORT       = int(os.environ.get("PORT", 5002))
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".pdf"}
MAX_MB = 20

app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = os.environ.get("SECRET_KEY", "vitalia-secret-2025")
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
        CREATE TABLE IF NOT EXISTS pacientes (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            nombre      TEXT NOT NULL DEFAULT 'Paciente',
            fecha_nac   TEXT DEFAULT '',
            genero      TEXT DEFAULT '',
            email       TEXT DEFAULT '',
            notas       TEXT DEFAULT '',
            creado_en   TEXT DEFAULT (datetime('now','localtime'))
        );

        CREATE TABLE IF NOT EXISTS examenes (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            paciente_id     INTEGER DEFAULT 1,
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

        CREATE TABLE IF NOT EXISTS conversaciones (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            examen_id   INTEGER,
            titulo      TEXT DEFAULT 'Consulta médica',
            creado_en   TEXT DEFAULT (datetime('now','localtime')),
            FOREIGN KEY (examen_id) REFERENCES examenes(id)
        );

        CREATE TABLE IF NOT EXISTS mensajes (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            conversacion_id     INTEGER NOT NULL,
            rol                 TEXT NOT NULL,
            contenido           TEXT NOT NULL,
            creado_en           TEXT DEFAULT (datetime('now','localtime')),
            FOREIGN KEY (conversacion_id) REFERENCES conversaciones(id) ON DELETE CASCADE
        );

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

        -- Paciente por defecto
        INSERT OR IGNORE INTO pacientes (id, nombre) VALUES (1, 'Mi Perfil');
        """)
        db.commit()

init_db()

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
def get_client():
    # Siempre re-leer desde el entorno para capturar cambios sin reiniciar
    key = os.environ.get("ANTHROPIC_API_KEY", "") or request.headers.get("X-API-Key","")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY no configurada. Configúrala en Environment Variables de Render.")
    return anthropic.Anthropic(api_key=key)

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
    import sys
    key = os.environ.get("ANTHROPIC_API_KEY","")
    return jsonify({
        "status": "ok",
        "python": sys.version,
        "data_dir": str(DATA_DIR),
        "db_exists": DB_PATH.exists(),
        "upload_dir": str(UPLOAD_DIR),
        "upload_writable": os.access(str(UPLOAD_DIR), os.W_OK),
        "api_key_set": bool(key),
        "api_key_prefix": key[:10] + "..." if key else None
    })

# ── Rutas principales ─────────────────────────────────────────────────────────
@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/app")
def app_main():
    with get_db() as db:
        examenes = db.execute("""
            SELECT e.*, COUNT(i.id) as num_indicadores,
                   SUM(CASE WHEN i.estado != 'normal' THEN 1 ELSE 0 END) as alertas
            FROM examenes e
            LEFT JOIN indicadores i ON i.examen_id = e.id
            GROUP BY e.id ORDER BY e.creado_en DESC LIMIT 20
        """).fetchall()
        paciente = db.execute("SELECT * FROM pacientes WHERE id=1").fetchone()
    return render_template("app.html",
                           examenes=[dict(e) for e in examenes],
                           paciente=dict(paciente) if paciente else {},
                           api_key_ok=bool(os.environ.get("ANTHROPIC_API_KEY","")))

# ── API: Paciente ─────────────────────────────────────────────────────────────
@app.route("/api/paciente", methods=["GET","POST"])
def paciente_endpoint():
    with get_db() as db:
        if request.method == "POST":
            d = request.json or {}
            db.execute("""UPDATE pacientes SET nombre=?,fecha_nac=?,genero=?,email=?,notas=?
                          WHERE id=1""",
                       (d.get("nombre","Mi Perfil"), d.get("fecha_nac",""),
                        d.get("genero",""), d.get("email",""), d.get("notas","")))
            db.commit()
            return jsonify({"ok": True})
        p = db.execute("SELECT * FROM pacientes WHERE id=1").fetchone()
        return jsonify(dict(p) if p else {})

# ── API: Exámenes ─────────────────────────────────────────────────────────────
@app.route("/api/examenes", methods=["GET"])
def listar_examenes():
    with get_db() as db:
        rows = db.execute("""
            SELECT e.*, COUNT(i.id) as num_indicadores,
                   SUM(CASE WHEN i.estado != 'normal' THEN 1 ELSE 0 END) as alertas
            FROM examenes e
            LEFT JOIN indicadores i ON i.examen_id = e.id
            GROUP BY e.id ORDER BY e.creado_en DESC
        """).fetchall()
    return jsonify({"examenes": [dict(r) for r in rows]})

@app.route("/api/examenes/<int:eid>", methods=["GET"])
def get_examen(eid):
    with get_db() as db:
        e = db.execute("SELECT * FROM examenes WHERE id=?", (eid,)).fetchone()
        if not e: return jsonify({"error":"No encontrado"}), 404
        indicadores = db.execute("SELECT * FROM indicadores WHERE examen_id=? ORDER BY id", (eid,)).fetchall()
        recomendaciones = db.execute("SELECT * FROM recomendaciones WHERE examen_id=? ORDER BY prioridad DESC", (eid,)).fetchall()
        conversaciones = db.execute("SELECT * FROM conversaciones WHERE examen_id=? ORDER BY creado_en DESC", (eid,)).fetchall()
    return jsonify({
        "examen": dict(e),
        "indicadores": [dict(i) for i in indicadores],
        "recomendaciones": [dict(r) for r in recomendaciones],
        "conversaciones": [dict(c) for c in conversaciones],
    })

@app.route("/api/examenes/<int:eid>", methods=["DELETE"])
def eliminar_examen(eid):
    with get_db() as db:
        db.execute("DELETE FROM examenes WHERE id=?", (eid,))
        db.commit()
    return jsonify({"ok": True})

# ── API: OCR + Análisis ───────────────────────────────────────────────────────
def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"

def _run_analisis(job_id, ruta, ext, titulo, tipo, api_key):
    """Corre el análisis completo en background y guarda resultado en _jobs."""
    def update(status, **kw):
        with _jobs_lock:
            _jobs[job_id] = {"status": status, **kw}

    try:
        update("pending", msg="Leyendo archivo...")
        cliente = anthropic.Anthropic(api_key=api_key)

        with open(ruta, "rb") as fh:
            img_b64 = base64.standard_b64encode(fh.read()).decode()

        if ext == ".pdf":
            file_block = {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": img_b64}}
        else:
            mt = {".jpg":"image/jpeg",".jpeg":"image/jpeg",".png":"image/png",".webp":"image/webp"}
            file_block = {"type": "image", "source": {"type": "base64", "media_type": mt.get(ext,"image/jpeg"), "data": img_b64}}

        update("pending", msg="Extrayendo indicadores con visión IA...")
        prompt_ocr = """Analiza este examen médico/laboratorio y extrae todos los indicadores.

Responde ÚNICAMENTE con este JSON compacto (sin markdown, sin texto adicional, sin campos extra):
{"tipo_examen":"","laboratorio":"","fecha":"","paciente_info":"","observaciones_laboratorio":"","indicadores":[{"nombre":"","valor":"","unidad":"","rango_ref":"","rango_min":null,"rango_max":null,"estado":"normal","descripcion":""}]}

Reglas:
- Extrae TODOS los indicadores visibles, uno por uno
- rango_min y rango_max: números (ej: 70 y 100 para "70-100"), null si no aplica
- estado: solo "normal", "alto", "bajo" o "critico"
- descripcion: máximo 8 palabras explicando qué mide
- Valores aproximados: agrega "~" al inicio (ej: "~12.5")
- NO incluyas el campo texto_completo"""

        resp_ocr = cliente.messages.create(
            model="claude-opus-4-6", max_tokens=16000,
            messages=[{"role":"user","content":[file_block,{"type":"text","text":prompt_ocr}]}]
        )
        raw = re.sub(r"```json\s*|\s*```", "", resp_ocr.content[0].text.strip()).strip()
        try:
            ocr_data = json.loads(raw)
        except json.JSONDecodeError:
            idx = raw.rfind("},")
            if idx == -1: idx = raw.rfind("}")
            ocr_data = json.loads(raw[:idx+1] + "]}") if idx != -1 else \
                       {"tipo_examen":tipo,"laboratorio":"","fecha":"","paciente_info":"","indicadores":[],"observaciones_laboratorio":""}

        indicadores = ocr_data.get("indicadores", [])
        alertas  = [i for i in indicadores if i.get("estado") in ("alto","bajo","critico")]
        criticos = [i for i in indicadores if i.get("estado") == "critico"]
        riesgo = "critico" if criticos else ("alto" if len(alertas)>=3 else ("medio" if alertas else "normal"))

        update("pending", msg=f"Interpretando {len(indicadores)} indicadores...")
        prompt_interp = f"""Como médico experto, interpreta estos resultados de examen médico.

TIPO DE EXAMEN: {ocr_data.get('tipo_examen', tipo)}
INDICADORES:
{json.dumps(indicadores, ensure_ascii=False, indent=2)}

OBSERVACIONES DEL LABORATORIO: {ocr_data.get('observaciones_laboratorio','')}

Genera una interpretación médica completa en JSON:
{{
  "resumen": "Resumen ejecutivo del estado de salud en 2-3 oraciones",
  "interpretacion": "Interpretación detallada de los hallazgos más relevantes (markdown)",
  "recomendaciones": [
    {{"tipo": "dieta|ejercicio|medicamento|consulta|estilo_vida", "titulo": "...", "descripcion": "...", "prioridad": "alta|media|baja"}}
  ],
  "alertas_principales": ["lista de los indicadores más preocupantes con explicación"],
  "puntos_positivos": ["aspectos de salud que están bien"],
  "seguimiento": "cuándo y con qué especialista debe consultar"
}}"""

        try:
            resp2 = cliente.messages.create(
                model="claude-opus-4-6", max_tokens=4000,
                messages=[{"role":"user","content":prompt_interp}]
            )
            raw2 = re.sub(r"```json\s*|\s*```", "", resp2.content[0].text.strip()).strip()
            interp_data = json.loads(raw2)
        except Exception:
            interp_data = {
                "resumen": "Análisis completado.",
                "interpretacion": "Revise los indicadores detallados y consulte con su médico.",
                "recomendaciones": [],
                "alertas_principales": [f"{i['nombre']}: {i['valor']}" for i in alertas],
                "puntos_positivos": [],
                "seguimiento": "Consulte con su médico tratante."
            }

        update("pending", msg="Guardando resultados...")
        with get_db() as db:
            cur = db.execute("""INSERT INTO examenes
                (titulo,tipo,fecha_examen,laboratorio,imagen_path,texto_extraido,interpretacion,resumen,estado,riesgo)
                VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (titulo, ocr_data.get("tipo_examen",tipo), ocr_data.get("fecha",""),
                 ocr_data.get("laboratorio",""), ruta.name, "",
                 interp_data.get("interpretacion",""), interp_data.get("resumen",""), "analizado", riesgo))
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
def analizar_examen():
    if request.method == "GET":
        return jsonify({"error": "Usa POST para enviar un examen"}), 405

    try:
        api_key = os.environ.get("ANTHROPIC_API_KEY","") or request.headers.get("X-API-Key","")
        if not api_key: raise ValueError("ANTHROPIC_API_KEY no configurada")
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

    nombre_archivo = f"{uuid.uuid4().hex}{ext}"
    ruta = UPLOAD_DIR / nombre_archivo
    f.save(str(ruta))

    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {"status": "pending", "msg": "Iniciando análisis..."}

    t = threading.Thread(target=_run_analisis, args=(job_id, ruta, ext, titulo, tipo, api_key), daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/api/job/<job_id>", methods=["GET"])
def get_job(job_id):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job no encontrado"}), 404
    return jsonify(job)


# ── API: Chat con Dr. VitalIA ─────────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
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

    with get_db() as db:
        conv = db.execute("SELECT * FROM conversaciones WHERE id=?", (conv_id,)).fetchone()
        if not conv: return jsonify({"error":"Conversación no encontrada"}), 404

        # Contexto del examen
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

        # Historial de mensajes (últimos 20)
        historial = db.execute(
            "SELECT rol, contenido FROM mensajes WHERE conversacion_id=? ORDER BY id DESC LIMIT 20",
            (conv_id,)
        ).fetchall()
        historial = list(reversed(historial))

        # Guardar mensaje del usuario
        db.execute("INSERT INTO mensajes (conversacion_id, rol, contenido) VALUES (?,?,?)",
                   (conv_id, "user", mensaje))
        db.commit()

    # Construir mensajes para Claude
    messages = []
    for h in historial:
        messages.append({"role": h["rol"], "content": h["contenido"]})
    messages.append({"role": "user", "content": mensaje})

    system = SYSTEM_DOCTOR
    if contexto_examen:
        system += f"\n\n{contexto_examen}"

    def generate():
        respuesta_completa = ""
        try:
            with cliente.messages.stream(
                model="claude-opus-4-6",
                max_tokens=2000,
                system=system,
                messages=messages
            ) as stream:
                for text in stream.text_stream:
                    respuesta_completa += text
                    yield f"data: {json.dumps({'text': text})}\n\n"

            # Guardar respuesta completa
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
def get_mensajes(cid):
    with get_db() as db:
        msgs = db.execute(
            "SELECT * FROM mensajes WHERE conversacion_id=? ORDER BY id",
            (cid,)
        ).fetchall()
    return jsonify({"mensajes": [dict(m) for m in msgs]})

@app.route("/api/conversaciones/nueva", methods=["POST"])
def nueva_conversacion():
    d = request.json or {}
    with get_db() as db:
        cur = db.execute(
            "INSERT INTO conversaciones (examen_id, titulo) VALUES (?,?)",
            (d.get("examen_id"), d.get("titulo", "Nueva consulta"))
        )
        db.commit()
        return jsonify({"ok": True, "id": cur.lastrowid})

# ── API: Stats ────────────────────────────────────────────────────────────────
@app.route("/api/stats")
def get_stats():
    with get_db() as db:
        total_examenes = db.execute("SELECT COUNT(*) FROM examenes").fetchone()[0]
        total_alertas  = db.execute("SELECT COUNT(*) FROM indicadores WHERE estado != 'normal'").fetchone()[0]
        total_normales = db.execute("SELECT COUNT(*) FROM indicadores WHERE estado = 'normal'").fetchone()[0]
        ultimo_examen  = db.execute("SELECT creado_en FROM examenes ORDER BY id DESC LIMIT 1").fetchone()
        riesgos = db.execute("SELECT riesgo, COUNT(*) as c FROM examenes GROUP BY riesgo").fetchall()
    return jsonify({
        "total_examenes": total_examenes,
        "total_alertas": total_alertas,
        "total_normales": total_normales,
        "ultimo_examen": ultimo_examen[0] if ultimo_examen else None,
        "riesgos": {r["riesgo"]: r["c"] for r in riesgos}
    })

# ── API: Config ───────────────────────────────────────────────────────────────
@app.route("/api/config", methods=["POST"])
def config():
    global ANTHROPIC_API_KEY
    d = request.json or {}
    if d.get("api_key"):
        ANTHROPIC_API_KEY = d["api_key"].strip()
    return jsonify({"ok": True, "api_key_ok": bool(ANTHROPIC_API_KEY)})

# ── Imagen upload viewer ──────────────────────────────────────────────────────
@app.route("/uploads/<path:filename>")
def serve_upload(filename):
    from flask import send_from_directory
    return send_from_directory(UPLOAD_DIR, filename)

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import threading, webbrowser
    print("="*55)
    print("  🩺 VitalIA — Dr. Digital")
    print(f"  URL: http://localhost:{PORT}")
    print("  Ctrl+C para detener")
    print("="*55)
    threading.Timer(1.2, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
