"""
AI-TIP 010 - SQL Database Agent App (V2 Ultra)

Mejoras en esta versión:
- Selección dinámica de proveedor LLM: OpenAI o Gemini (Google Generative AI).
- Adaptador (wrapper) para usar Gemini con parámetros optimizados
- Soporte para modelos avanzados de Gemini (Pro/Ultra)
- Manejo de credenciales desde la sidebar de Streamlit.
- Mensajes y fallbacks claros si faltan dependencias.

NOTAS:
- Instalar dependencias necesarias:
    pip install streamlit sqlalchemy pandas langchain_openai google-generative-ai

Uso:
    streamlit run 010_sql_database_agent_app/app_V2_fixed_ultra.py
"""

from __future__ import annotations

import streamlit as st
import sqlalchemy as sql
import pandas as pd
import asyncio
import typing as t
import json
from datetime import datetime
import time

# Inicializar el estado de Streamlit
def init_session_state():
    """Inicializa todas las variables de estado necesarias."""
    if "gemini_log" not in st.session_state:
        st.session_state.gemini_log = []
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "stored_dataframes" not in st.session_state:
        st.session_state.stored_dataframes = {}
    
    if "last_query" not in st.session_state:
        st.session_state.last_query = None
    
    if "error_log" not in st.session_state:
        st.session_state.error_log = []
    
    if "selected_provider" not in st.session_state:
        st.session_state.selected_provider = "Gemini"

# Llamar a la inicialización al inicio
init_session_state()

# Intentamos importar ChatOpenAI (OpenAI) + Streamlit chat history
try:
    from langchain_community.chat_message_histories import StreamlitChatMessageHistory
except Exception:
    StreamlitChatMessageHistory = None

try:
    from langchain_openai import ChatOpenAI
except Exception:
    ChatOpenAI = None

# Agente externo
try:
    from ai_data_science_team.agents import SQLDatabaseAgent
except Exception:
    SQLDatabaseAgent = None

# Funciones de utilidad para UI
def show_debug_info():
    """Muestra información de depuración en la interfaz de usuario."""
    if st.session_state.get("show_debug", False):
        with st.expander("Debug Info"):
            # Mostrar logs de Gemini
            if st.session_state.gemini_log:
                st.subheader("Gemini Logs")
                for entry in st.session_state.gemini_log:
                    st.write(f"[{entry['timestamp']}] {entry['type']}: {entry['message']}")
                    if entry['error']:
                        st.error(f"Error: {entry['error']}")
            
            # Mostrar historial de chat
            if st.session_state.chat_history:
                st.subheader("Chat History")
                for msg in st.session_state.chat_history:
                    st.write(f"{msg['role']}: {msg['content'][:100]}...")
            
            # Mostrar última consulta
            if st.session_state.last_query:
                st.subheader("Last Query")
                st.code(st.session_state.last_query)

# Intentamos importar el cliente oficial de Google Generative AI
try:
    import google.generativeai as genai
    from google.generativeai.types import Model
    from google.ai import generativelanguage as glm
    
    # Verificar que tenemos la versión correcta del SDK
    import pkg_resources
    genai_version = pkg_resources.get_distribution('google-generativeai').version
    print(f"Versión de google-generativeai: {genai_version}")
    
except ImportError as e:
    print(f"Error importando Google Generative AI: {e}")
    print("Instalando dependencias necesarias...")
    import subprocess
    subprocess.run(["pip", "install", "--upgrade", "google-generativeai"], check=True)
    
    # Reintentar importación
    import google.generativeai as genai
    from google.generativeai.types import Model
    from google.ai import generativelanguage as glm
    genai = None
    Model = None
    try:
        import requests

        class _GenAIFallbackModule:
            _api_key: t.Optional[str] = None

            @staticmethod
            def configure(api_key: str):
                """Guarda la API key y valida que funcione llamando al endpoint de modelos.

                Nota: usamos Authorization: Bearer <key> en las llamadas REST. Si la validación
                falla, lanzamos un RuntimeError con detalle (sin exponer la clave).
                """
                _GenAIFallbackModule._api_key = api_key
                # Validar la clave consultando la lista de modelos
                try:
                    test_url = "https://generativelanguage.googleapis.com/v1/models"
                    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
                    r = requests.get(test_url, headers=headers, timeout=10)
                    r.raise_for_status()
                except requests.exceptions.RequestException as e:
                    # Proveer mensaje legible para depuración sin exponer la API key
                    resp_text = None
                    if getattr(e, 'response', None) is not None:
                        try:
                            resp_text = e.response.json()
                        except Exception:
                            resp_text = e.response.text
                    raise RuntimeError(f"Error validando API key de Gemini: {resp_text or str(e)}")

            class chat:
                @staticmethod
                def create(model: str, messages: t.List[dict]):
                    """Llamada REST optimizada al endpoint de Generative Language.
                    
                    Usa los últimos modelos y parámetros optimizados para SQL.
                    """
                    api_key = _GenAIFallbackModule._api_key
                    if not api_key:
                        raise RuntimeError("No API key configured for Generative AI (fallback)")

                    # Normalizar mensajes a texto
                    parts = []
                    for m in messages:
                        if isinstance(m, dict):
                            parts.append(m.get("content", ""))
                        else:
                            parts.append(getattr(m, "content", None) or getattr(m, "text", None) or str(m))

                    prompt_text = "\n".join([p for p in parts if p]) or ""

                    # Endpoint v1beta con parámetros optimizados para SQL
                    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
                    # Enviar API key por Authorization header en lugar de query string
                    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
                    
                    # Configuración optimizada para SQL:
                    # - temperature baja para respuestas más deterministas
                    # - top_k y top_p ajustados para mejor precisión
                    # - safety settings reducidos para permitir SQL complejo
                    payload = {
                        "contents": [{"parts": [{"text": prompt_text}]}],
                        "generationConfig": {
                            "temperature": 0.1,        # Muy determinista para SQL
                            "topP": 0.95,             # Mantener alta precisión
                            "topK": 60,               # Aumentado para vocabulario SQL
                            "maxOutputTokens": 2048,   # Permitir queries complejas
                            "candidateCount": 1,       # Un solo resultado determinista
                        },
                        "safetySettings": [
                            {
                                "category": "HARM_CATEGORY_DANGEROUS",
                                "threshold": "BLOCK_NONE"
                            }
                        ]
                    }

                    resp = requests.post(url, headers=headers, json=payload, timeout=30)
                    try:
                        resp.raise_for_status()
                    except requests.exceptions.HTTPError as e:
                        # Adjuntar body JSON si está disponible para diagnóstico
                        body = None
                        try:
                            body = resp.json()
                        except Exception:
                            body = resp.text
                        raise RuntimeError(f"API Gemini devolvió {resp.status_code}: {body}")
                    data = resp.json()

                    # Intentar extraer texto de forma robusta
                    text = None
                    try:
                        if "candidates" in data:
                            text = data["candidates"][0].get("content", "")
                        elif "content" in data:
                            text = data["content"].get("text", "")
                    except Exception:
                        text = None
                    if not text:
                        for key in ("output", "result", "response"):
                            val = data.get(key)
                            if isinstance(val, dict):
                                text = val.get("content") or val.get("text")
                                if text:
                                    break

                    if not text:
                        text = str(data)

                    # Construir objeto de respuesta
                    class _C:
                        def __init__(self, content):
                            self.content = content

                    class _R:
                        def __init__(self, content):
                            self.candidates = [_C(content)]

                    return _R(text)

        genai = _GenAIFallbackModule
    except Exception:
        genai = None

TITLE = "Your SQL Database Agent — V2 Ultra"

DB_OPTIONS = {
    "Northwind Database": "sqlite:///data/northwind.db",
}

MODEL_LIST = ["gpt-4-1106-preview", "gpt-4"]

st.set_page_config(page_title=TITLE, page_icon="📊")
st.title(TITLE)

st.markdown(
    """
    Versión ultra optimizada con los últimos modelos de Gemini y OpenAI.
    - OpenAI: GPT-4 Turbo
    - Gemini: Pro optimizado para SQL
    """
)

with st.expander("Example Questions", expanded=False):
    st.write(
        """
        - What tables exist in the database?
        - What are the first 10 rows in the territory table?
        - Aggregate sales for each territory.
        - Aggregate sales by month for each territory.
        - Show me a complex analysis of sales trends by region.
        """
    )

# Sidebar: database & provider selection
db_option = st.sidebar.selectbox("Select a Database", list(DB_OPTIONS.keys()))
st.session_state["PATH_DB"] = DB_OPTIONS.get(db_option)

provider = st.sidebar.selectbox("Proveedor LLM", ["OpenAI", "Gemini (Google)"])

sql_engine = sql.create_engine(st.session_state["PATH_DB"])
conn = sql_engine.connect()

# Credentials y modelo
if provider == "OpenAI":
    st.sidebar.header("OpenAI API Key")
    # Intentar leer desde streamlit secrets si el usuario no provee la key en la UI
    secret_openai = None
    try:
        secret_openai = st.secrets.get("OPENAI_API_KEY") or st.secrets.get("openai_api_key") or st.secrets.get("openai")
    except Exception:
        secret_openai = None

    st.session_state["OPENAI_API_KEY"] = st.sidebar.text_input("OpenAI API Key", type="password", value=secret_openai or "")
    
    # Selector de modelo OpenAI
    model_name = st.sidebar.selectbox(
        "Modelo OpenAI",
        ["gpt-4-1106-preview", "gpt-4"],
        help="gpt-4-1106-preview es GPT-4 Turbo, más rápido y actualizado"
    )
else:
    st.sidebar.header("Google (Gemini) API Key")
    # Intentar leer la API key desde streamlit secrets si existe
    secret_google = None
    try:
        secret_google = st.secrets.get("GOOGLE_API_KEY") or st.secrets.get("google_api_key") or st.secrets.get("google")
    except Exception:
        secret_google = None

    st.session_state["GOOGLE_API_KEY"] = st.sidebar.text_input(
        "Google API Key",
        type="password",
        value=secret_google or "",
        help="Si usas una cuenta de servicio, define GOOGLE_APPLICATION_CREDENTIALS"
    )
    
    # Selector de modelo Gemini si está disponible el SDK oficial
    if genai and hasattr(genai, 'list_models'):
        try:
            genai.configure(api_key=st.session_state["GOOGLE_API_KEY"])
            available_models = [m for m in genai.list_models() if 'gemini' in m.name.lower()]
            if available_models:
                model_name = st.sidebar.selectbox(
                    "Modelo Gemini",
                    [m.name for m in available_models],
                    help="gemini-pro es el recomendado para SQL"
                )
            else:
                model_name = "gemini-pro"  # fallback
        except Exception:
            model_name = "gemini-pro"  # fallback si hay error
    else:
        model_name = "gemini-pro"  # fallback si no está el SDK


def get_db_objects(conn, obj_type=None):
    """Consulta objetos de la base de datos (tablas/vistas) de forma robusta."""
    try:
        query = """
        SELECT 
            name,
            type,
            sql
        FROM sqlite_master 
        WHERE type IN ('table', 'view')
        """
        if obj_type:
            query += f" AND type = '{obj_type}'"
        query += " ORDER BY type, name;"
        
        try:
            # Intentar primero con SQLAlchemy text
            result = conn.execute(sql.text(query))
            rows = result.fetchall()
        except Exception:
            # Fallback a pandas
            df = pd.read_sql_query(query, conn)
            rows = df.to_records(index=False)
        
        objects = []
        for row in rows:
            try:
                # Intentar acceso por índice
                name, type_, definition = row[0], row[1], row[2]
            except Exception:
                # Fallback a acceso por nombre
                name = row.get('name', str(row))
                type_ = row.get('type', '?')
                definition = row.get('sql', '')
            
            objects.append({
                'name': name,
                'type': type_,
                'definition': definition
            })
        
        return objects
    
    except Exception as e:
        raise RuntimeError(f"Error consultando objetos de la base de datos: {e}")


# --- Gemini adapter with optimizations ---
class GeminiAdapter:
    """Adaptador avanzado para Gemini con optimizaciones para SQL."""

    def __init__(self, model: str = "models/gemini-2.5-pro"):
        """
        Inicializa el adaptador con el modelo especificado.
        Por defecto usa gemini-2.5-pro que es óptimo para generación de SQL.
        """
        self.model = model
        self.model_instance = None
        self.genai_client = None
        
        # Registrar la inicialización
        st.session_state.setdefault("gemini_log", [])
        self._log_event("init", f"Inicializando GeminiAdapter con modelo {model}")
        
        # Configuración optimizada para SQL
        self.generation_config = {
            "temperature": 0.1,        # Muy determinista para SQL
            "top_p": 0.95,            # Alta precisión
            "top_k": 60,              # Vocabulario SQL amplio
            "max_output_tokens": 2048  # Queries complejas
        }
        
        # Safety settings optimizados
        self.safety_settings = {
            "HARM_CATEGORY_DANGEROUS": "BLOCK_NONE",
            "HARM_CATEGORY_HARASSMENT": "BLOCK_NONE",
            "HARM_CATEGORY_HATE_SPEECH": "BLOCK_NONE",
            "HARM_CATEGORY_SEXUALLY_EXPLICIT": "BLOCK_NONE",
            "HARM_CATEGORY_DANGEROUS_CONTENT": "BLOCK_NONE"
        }

    
    def _messages_to_genai(self, messages: t.List[t.Any]) -> str:
        """Conversión mejorada de mensajes con mejor contexto SQL.

        Devuelve un único string que se incorporará en el campo `parts[0]['text']`
        del payload esperado por la API de GenerativeModels.
        """
        system_prompt = (
            "You are an expert SQL analyst. When asked about database operations:\n"
            "1. Generate clean, optimized SQL queries\n"
            "2. Support complex analytics and aggregations\n"
            "3. Consider performance implications\n"
            "4. Use modern SQL features when beneficial\n"
            "5. Do not include explanations unless asked\n"
            "6. Never use markdown - return raw SQL only"
        )

        parts = [f"[System] {system_prompt}"]

        for m in messages:
            if isinstance(m, dict):
                role = m.get("role", "user")
                content = m.get("content", "")
            else:
                content = getattr(m, "content", None) or getattr(m, "text", None) or str(m)
                role = getattr(m, "role", None) or getattr(m, "type", None) or "user"

            parts.append(f"[{role.capitalize()}] {content}")

        # Devolver un único texto que se colocará en parts[0]['text']
        return "\n\n".join(parts)
    def _log_event(self, event_type: str, message: str, error: Exception = None):
        """Registra eventos de forma segura sin exponer información sensible."""
        try:
            timestamp = datetime.now().isoformat()
            log_entry = {
                "timestamp": timestamp,
                "type": event_type,
                "message": message,
                "error": str(error) if error else None
            }
            
            # Asegurar que gemini_log existe
            if "gemini_log" not in st.session_state:
                st.session_state.gemini_log = []
                
            st.session_state.gemini_log.append(log_entry)
            
            # Si es un error, también lo agregamos al log de errores
            if event_type == "error":
                if "error_log" not in st.session_state:
                    st.session_state.error_log = []
                st.session_state.error_log.append(log_entry)
        except Exception as e:
            # Si falla el logging, al menos mostrar el error en la consola
            print(f"Error logging event: {e}\nOriginal message: {message}")
        
    def _handle_gemini_error(self, e: Exception) -> str:
        """Maneja errores de Gemini y proporciona mensajes claros."""
        error_msg = str(e)
        if "400" in error_msg:
            msg = "Error en la solicitud. Verifica la longitud del prompt y el formato."
        elif "401" in error_msg:
            msg = "Error de autenticación. Verifica tu API key."
        elif "403" in error_msg:
            msg = "No tienes permiso para usar este modelo."
        elif "404" in error_msg:
            msg = f"Modelo {self.model} no encontrado. Intenta con models/gemini-2.5-pro"
        else:
            msg = f"Error al procesar la solicitud: {error_msg}"
            
        self._log_event("error", msg, e)
        return msg

    def _ensure_model_initialized(self):
        """Asegura que el modelo está inicializado correctamente."""
        if not self.model_instance:
            try:
                # Obtener API key de secrets
                api_key = st.secrets.get("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("API key no encontrada en secrets")
                
                # Configurar cliente y validar modelos disponibles
                genai.configure(api_key=api_key)
                models = genai.list_models()
                model_names = [m.name for m in models]
                
                if self.model not in model_names:
                    self._log_event("warning", f"Modelo {self.model} no encontrado, usando alternativa")
                    # Intentar encontrar un modelo alternativo adecuado
                    for alt_model in [
                        "models/gemini-2.5-pro",
                        "models/gemini-pro-latest",
                        "models/gemini-2.0-pro-latest"
                    ]:
                        if alt_model in model_names:
                            self.model = alt_model
                            break
                    else:
                        raise ValueError("No se encontró ningún modelo compatible")
                
                # Inicializar el modelo
                self.model_instance = genai.GenerativeModel(model_name=self.model)
                self._log_event("init", f"Modelo {self.model} inicializado correctamente")
            except Exception as e:
                self._log_event("error", "Error al inicializar el modelo", e)
                raise

    def _clean_sql(self, text: str) -> str:
        """Limpieza avanzada de SQL con soporte para características modernas."""
        if not text:
            return ""
        
        import re
        
        # Eliminar bloques de código
        text = re.sub(r'```(?:sql)?(.*?)```', r'\1', text, flags=re.DOTALL)
        
        # Procesamiento línea por línea mejorado
        lines = []
        in_sql = False
        comment_block = False
        
        # Palabras clave SQL comunes y modernas
        sql_keywords = {
            'SELECT', 'FROM', 'WHERE', 'GROUP', 'ORDER', 'JOIN', 'HAVING',
            'WITH', 'WINDOW', 'PARTITION', 'OVER', 'RANK', 'LAG', 'LEAD',
            'FIRST_VALUE', 'LAST_VALUE', 'DENSE_RANK', 'ROW_NUMBER',
            'CUBE', 'ROLLUP', 'GROUPING SETS'
        }
        
        for line in text.split('\n'):
            line = line.strip()
            
            # Saltar líneas vacías y comentarios
            if not line or line.startswith('#') or line.startswith('--'):
                continue
                
            # Detectar bloques de comentarios
            if '/*' in line:
                comment_block = True
                continue
            if '*/' in line:
                comment_block = False
                continue
            if comment_block:
                continue
            
            # Detectar SQL por keywords
            if any(kw in line.upper() for kw in sql_keywords):
                in_sql = True
            
            if in_sql:
                lines.append(line)
        
        sql = '\n'.join(lines)
        
        # Limpieza final
        sql = re.sub(r'/\*.*?\*/', '', sql, flags=re.DOTALL)  # eliminar comentarios /* */
        sql = re.sub(r'--.*$', '', sql, flags=re.MULTILINE)   # eliminar comentarios --
        
        return sql.strip()

    def generate(self, messages: t.List[t.Any], stop: t.Optional[t.List[str]] = None) -> _SimpleGenerateResult:
        """Genera una respuesta usando el modelo Gemini."""
        try:
            self._ensure_model_initialized()
            
            # Convertir mensajes al formato de Gemini
            formatted_messages = self._messages_to_genai(messages)
            
            # Generar respuesta
            generation_config = {
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 60,
                "max_output_tokens": 2048,
                "stop_sequences": stop if stop else None
            }
            
            # Lógica de reintentos para manejar quotas (429) y errores transitorios
            max_retries = 4
            delay = 1.0
            backoff = 2.0
            response = None
            last_exc = None
            for attempt in range(1, max_retries + 1):
                try:
                    self._log_event("attempt", f"Generación intento {attempt} usando modelo {self.model}")
                    response = self.model_instance.generate_content(
                        formatted_messages,
                        generation_config=generation_config,
                        safety_settings=[
                            {"category": "HARM_CATEGORY_DANGEROUS", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                        ]
                    )
                    # Si llegamos aquí, la llamada fue exitosa
                    break
                except Exception as e:
                    last_exc = e
                    err_str = str(e)
                    # Intentar extraer un Retry-After si está disponible
                    retry_after = None
                    try:
                        resp = getattr(e, 'response', None)
                        if resp is not None:
                            retry_after = resp.headers.get('retry-after') if hasattr(resp, 'headers') else None
                    except Exception:
                        retry_after = None

                    # Si es un error de cuota/rate limit, intentar reintentar con backoff
                    if ('quota' in err_str.lower()) or ('429' in err_str) or ('rate limit' in err_str.lower()):
                        wait = float(retry_after) if retry_after and retry_after.isdigit() else delay
                        self._log_event("rate_limit", f"Quota excedida, esperando {wait} segundos antes de reintentar", e)
                        if attempt < max_retries:
                            time.sleep(wait)
                            delay *= backoff
                            continue
                        else:
                            # Excedimos reintentos
                            self._log_event("error", "Excedidos reintentos por cuota", e)
                            raise
                    else:
                        # Error no relacionado con cuota: no reintentamos muchas veces
                        self._log_event("error", "Error en generate_content", e)
                        raise
            # Si no obtuvimos respuesta válida, lanzar el último error
            if response is None and last_exc is not None:
                raise last_exc
            
            if not response.text:
                raise ValueError("Respuesta vacía del modelo")
                
            # Procesar y limpiar la respuesta
            text = self._clean_sql(response.text)
            if not text:
                raise ValueError("No se pudo extraer SQL válido de la respuesta")
                
            self._log_event("generate", "Generación exitosa", None)
            return _SimpleGenerateResult(text)

        except Exception as e:
            error_msg = self._handle_gemini_error(e)
            raise RuntimeError(error_msg)

    async def agenerate(self, messages: t.List[t.Any], stop: t.Optional[t.List[str]] = None):
        """Versión asíncrona de generate."""
        return self.generate(messages, stop=stop)

    def __call__(self, prompt: t.Union[str, t.List[t.Any], dict], **kwargs) -> str:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, dict) and "messages" in prompt:
            messages = prompt["messages"]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            messages = [{"role": "user", "content": str(prompt)}]

        result = self.generate(messages)
        try:
            return result.generations[0][0].text
        except Exception:
            return str(result)

    async def __acall__(self, prompt: t.Union[str, t.List[t.Any], dict], **kwargs) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.__call__(prompt, **kwargs))

    def predict(self, prompt: str) -> str:
        return self.__call__(prompt)

    def predict_messages(self, messages: t.List[t.Any]) -> str:
        return self.__call__(messages)


class _SimpleGeneration:
    def __init__(self, text: str):
        self.text = text


class _SimpleGenerateResult:
    def __init__(self, text: str):
        self.generations = [[_SimpleGeneration(text)]]


def _fail_if_missing(name: str, obj: t.Any) -> bool:
    """Valida que una dependencia requerida esté disponible."""
    if obj is None:
        st.error(f"Falta dependencia: {name}. Instala la librería correspondiente y reinicia la app.")
        return True
    return False


# Valida dependencias básicas
if provider == "OpenAI" and _fail_if_missing("langchain_openai (ChatOpenAI)", ChatOpenAI):
    st.stop()

if provider == "Gemini (Google)" and _fail_if_missing("google-generative-ai (genai) o requests", not (genai or 'requests' in globals())):
    st.stop()

if SQLDatabaseAgent is None:
    st.error("Falta `ai_data_science_team.agents.SQLDatabaseAgent`. Instala el paquete `ai_data_science_team`.")
    st.stop()

# Inicializar LLM según proveedor
llm = None
if provider == "OpenAI":
    if not st.session_state.get("OPENAI_API_KEY"):
        st.info("Introduce tu OpenAI API Key en la barra lateral para continuar.")
        st.stop()
    
    llm = ChatOpenAI(
        model=model_name,
        api_key=st.session_state["OPENAI_API_KEY"],
        temperature=0.1,  # Optimizado para SQL
        model_kwargs={
            "top_p": 0.95,
            "frequency_penalty": 0.1,
            "presence_penalty": 0.1
        }
    )
else:
    if not st.session_state.get("GOOGLE_API_KEY"):
        st.info("Introduce tu Google API Key en la barra lateral para continuar.")
        st.stop()
    
    try:
        # Intentar usar la nueva librería `google-genai` si está disponible
        try:
            from google import genai as genai_v2
            genai_client = genai_v2.Client(api_key=st.session_state["GOOGLE_API_KEY"])
            llm = GeminiAdapter(model=model_name)
            # Pasar el cliente moderno al adaptador
            llm.genai_client = genai_client
        except Exception:
            # Fallback a la librería legacy/configuración previa
            genai.configure(api_key=st.session_state["GOOGLE_API_KEY"])
            llm = GeminiAdapter(model=model_name)
    except Exception as e:
        st.error(f"Error al inicializar Gemini: {e}")
        st.stop()

# Set up memory
if StreamlitChatMessageHistory is None:
    st.warning("Falta StreamlitChatMessageHistory; la interfaz de chat será básica.")
    class _DummyHistory:
        def __init__(self, key):
            self.key = key
            self.messages = []
        def add_ai_message(self, m):
            self.messages.append(type("M", (), {"type":"ai","content":m}))
        def add_user_message(self, m):
            self.messages.append(type("M", (), {"type":"human","content":m}))
    msgs = _DummyHistory(key="langchain_messages")
else:
    msgs = StreamlitChatMessageHistory(key="langchain_messages")

if len(msgs.messages) == 0:
    msgs.add_ai_message("¿Cómo puedo ayudarte con tus consultas SQL?")

if "dataframes" not in st.session_state:
    st.session_state.dataframes = []


def display_chat_history():
    for i, msg in enumerate(msgs.messages):
        with st.chat_message(msg.type):
            if "DATAFRAME_INDEX:" in msg.content:
                df_index = int(msg.content.split("DATAFRAME_INDEX:")[1])
                st.dataframe(st.session_state.dataframes[df_index])
            else:
                st.write(msg.content)
 
def safe_show_dataframe(df: pd.DataFrame, key: str, max_rows_per_page: int = 500):
    """Muestra un DataFrame de forma segura en Streamlit con paginado y descarga.

    - Guarda el DataFrame completo en `st.session_state.stored_dataframes[key]`.
    - Muestra una sola página con `max_rows_per_page` filas.
    - Ofrece descarga de la página y del dataset completo.
    """
    if key not in st.session_state.get("stored_dataframes", {}):
        st.session_state.stored_dataframes[key] = df.copy()

    stored = st.session_state.stored_dataframes[key]
    n_rows = len(stored)

    st.write(f"Dataset '{key}': {n_rows:,} filas × {len(stored.columns):,} columnas")

    with st.expander("Resumen rápido"):
        st.write("Columnas:", list(stored.columns))
        st.write("Tipos de datos:")
        st.write(stored.dtypes)
        st.write("Primeras filas (head):")
        st.dataframe(stored.head(5))

    total_pages = max(1, (n_rows + max_rows_per_page - 1) // max_rows_per_page)
    page = st.number_input(f"Página (1..{total_pages})", min_value=1, max_value=total_pages, value=1, key=f"{key}_page")
    start = (page - 1) * max_rows_per_page
    end = start + max_rows_per_page
    page_df = stored.iloc[start:end]

    st.write(f"Mostrando filas {start + 1} a {min(end, n_rows)} (página {page} de {total_pages})")
    st.dataframe(page_df)

    # Botón de descarga de la página actual
    try:
        import io
        buf = io.StringIO()
        page_df.to_csv(buf, index=False)
        st.download_button(
            label=f"Descargar página {page} (CSV, {len(page_df):,} filas)",
            data=buf.getvalue().encode("utf-8"),
            file_name=f"{key}_page_{page}.csv",
            mime="text/csv"
        )
    except Exception as e:
        st.warning(f"No se pudo preparar la descarga de la página: {e}")

    # Botón para descargar todo el dataset (esto crea el CSV en memoria)
    if st.button(f"Descargar dataset completo ({key})"):
        try:
            buf = io.StringIO()
            stored.to_csv(buf, index=False)
            st.download_button(label="Click para descargar dataset completo", data=buf.getvalue().encode("utf-8"), file_name=f"{key}_full.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Error preparando descarga completa: {e}")


display_chat_history()

# Create SQL Database Agent
sql_db_agent = SQLDatabaseAgent(
    model=llm,
    connection=conn,
    n_samples=1,
    log=False,
    bypass_recommended_steps=True,
)


async def handle_question(question: str):
    if hasattr(sql_db_agent, "ainvoke_agent"):
        await sql_db_agent.ainvoke_agent(user_instructions=question)
    elif hasattr(sql_db_agent, "invoke_agent"):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: sql_db_agent.invoke_agent(user_instructions=question))
    else:
        raise RuntimeError("SQLDatabaseAgent no tiene método de invocación conocido")
    return sql_db_agent


if st.session_state["PATH_DB"] and (question := st.chat_input("Escribe tu consulta aquí:", key="query_input")):
    # Fast path para preguntas sobre tablas o vistas (evita llamar al LLM)
    q_lower = question.strip().lower()
    is_table_question = any(x in q_lower for x in ["what tables", "qué tablas", "que tablas", "list the tables", "tablas"]) 
    is_view_question = any(x in q_lower for x in ["what views", "qué vistas", "que vistas", "list the views", "vistas", "cuales son las vistas", "cuáles son las vistas"]) 

    if is_table_question or is_view_question:
        try:
            if is_view_question:
                objects = get_db_objects(conn, obj_type='view')
                if objects:
                    st.markdown("### Vistas en la base de datos:")
                    for obj in objects:
                        name = obj.get('name')
                        definition = obj.get('definition', '')
                        with st.expander(f"Vista: {name}", expanded=False):
                            st.code(definition, language='sql')
                else:
                    st.info("No hay vistas definidas en la base de datos.")
            else:
                objects = get_db_objects(conn)
                st.markdown("### Tablas y vistas en la base de datos:")
                for obj in objects:
                    st.write(f"- {obj.get('name')} ({obj.get('type')})")
        except Exception as e:
            st.error(f"Error consultando la base de datos: {e}")
        st.stop()

    with st.spinner("Pensando..."):
        st.chat_message("human").write(question)
        msgs.add_user_message(question)

        error_occured = False
        try:
            result = asyncio.run(handle_question(question))
        except Exception as e:
            error_occured = True
            debug_info = {}
            try:
                debug_info['agent_repr'] = repr(sql_db_agent)
            except Exception:
                debug_info['agent_repr'] = '<no disponible>'
            response_text = f"Lo siento, tuve dificultades para responder esa pregunta.\n\nError: {e}\n\nDebug: {debug_info}"
            msgs.add_ai_message(response_text)
            st.chat_message("ai").write(response_text)
            st.error(f"Error: {e}")

        if not error_occured:
            try:
                sql_query = result.get_sql_query_code()
            except Exception as e:
                sql_query = None
                st.warning(f"No hay query SQL disponible (error leyendo query): {e}")

            try:
                response_df = result.get_data_sql()
            except Exception as e:
                response_df = None
                st.warning(f"No hay dataframe disponible (error leyendo datos): {e}")

            if sql_query:
                response_1 = f"### Resultados SQL:\n\nConsulta SQL:\n\n```sql\n{sql_query}\n```\n\nResultado:"

                df_index = len(st.session_state.dataframes)
                st.session_state.dataframes.append(response_df)
                # Guardar copia segura para paginado/descarga
                st.session_state.stored_dataframes[f"df_{df_index}"] = response_df.copy()

                msgs.add_ai_message(response_1)
                msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")

                st.chat_message("ai").write(response_1)
                # Mostrar de forma segura con paginado (500 filas por página)
                safe_show_dataframe(response_df, f"df_{df_index}", max_rows_per_page=500)

            else:
                msgs.add_ai_message("El agente no generó ninguna consulta SQL.")
                st.chat_message("ai").write("El agente no generó ninguna consulta SQL.")