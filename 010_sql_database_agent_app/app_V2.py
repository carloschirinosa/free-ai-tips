"""
AI-TIP 010 - SQL Database Agent App (V2)

Mejoras en esta versión:
- Selección dinámica de proveedor LLM: OpenAI o Gemini (Google Generative AI).
- Adaptador (wrapper) para usar Gemini cuando no exista un wrapper LangChain disponible.
- Manejo de credenciales desde la sidebar de Streamlit.
- Mensajes y fallbacks claros si faltan dependencias.

NOTAS:
- Instalar dependencias necesarias (ejemplo):
    pip install streamlit sqlalchemy pandas langchain_openai google-generative-ai
  (ajusta según tu entorno y versiones)

Uso:
    streamlit run 010_sql_database_agent_app/app_V2.py

Crear variables de entorno en Windows (opcional):
    setx OPENAI_API_KEY "sk-..."
    setx GOOGLE_API_KEY "AIza..."

"""

from __future__ import annotations

import streamlit as st
import sqlalchemy as sql
import pandas as pd
import asyncio
import typing as t

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

# Intentamos importar el cliente oficial de Google Generative AI.
# Si no está disponible, definimos un fallback ligero que hace llamadas REST
# a la API de Generative Language vía requests. Esto evita que la app falle
# cuando el paquete oficial no esté instalado.
try:
    import google.generativeai as genai
except Exception:
    genai = None
    try:
        import requests

        class _GenAIFallbackModule:
            _api_key: t.Optional[str] = None

            @staticmethod
            def configure(api_key: str):
                _GenAIFallbackModule._api_key = api_key

            class chat:
                @staticmethod
                def create(model: str, messages: t.List[dict]):
                    """Llamada REST mínima al endpoint de Generative Language.

                    Construye un prompt concatenando los contenidos de `messages`.
                    Nota: este fallback usa el endpoint público y la API Key como
                    parámetro `key=` en la URL. Si usas cuentas de servicio (OAuth2)
                    deberías usar el cliente oficial.
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
                            # Si es un objeto, intentar atributos comunes
                            parts.append(getattr(m, "content", None) or getattr(m, "text", None) or str(m))

                    prompt_text = "\n".join([p for p in parts if p]) or ""

                    # Endpoint (v1beta2). Model names pueden variar; si falla, prueba con text-bison-001
                    url = f"https://generativelanguage.googleapis.com/v1beta2/models/{model}:generateText?key={api_key}"
                    payload = {"prompt": {"text": prompt_text}}

                    resp = requests.post(url, json=payload, timeout=30)
                    resp.raise_for_status()
                    data = resp.json()

                    # Intentar extraer texto de forma robusta
                    text = None
                    try:
                        if isinstance(data, dict):
                            if "candidates" in data and isinstance(data["candidates"], list) and len(data["candidates"]) > 0:
                                cand = data["candidates"][0]
                                # varias formas posibles según versión
                                text = cand.get("content") or cand.get("output") or cand.get("text")
                            if not text:
                                # fallback a campos comunes
                                for key in ("output", "result", "response"):
                                    val = data.get(key)
                                    if isinstance(val, dict):
                                        text = val.get("content") or val.get("text")
                                        if text:
                                            break
                    except Exception:
                        text = None

                    if not text:
                        # último recurso: stringify whole response
                        text = str(data)

                    # Construir objeto con atributo .candidates[0].content similar al cliente
                    class _C:
                        def __init__(self, content):
                            self.content = content

                    class _R:
                        def __init__(self, content):
                            self.candidates = [ _C(content) ]

                    return _R(text)

        genai = _GenAIFallbackModule
    except Exception:
        # si tampoco hay requests, dejamos genai = None y la app mostrará el error
        genai = None

TITLE = "Your SQL Database Agent — V2"

DB_OPTIONS = {
    "Northwind Database": "sqlite:///data/northwind.db",
}

MODEL_LIST = ["gpt-4o-mini", "gpt-4o"]

st.set_page_config(page_title=TITLE, page_icon="📊")
st.title(TITLE)

st.markdown(
    """
    Versión mejorada: puedes seleccionar OpenAI o Gemini (Google) como proveedor del LLM.
    Si seleccionas Gemini, la app intentará usar la librería `google-generative-ai` y un adaptador
    para integrar el modelo con el `SQLDatabaseAgent`.
    """
)

with st.expander("Example Questions", expanded=False):
    st.write(
        """
        - What tables exist in the database?
        - What are the first 10 rows in the territory table?
        - Aggregate sales for each territory.
        - Aggregate sales by month for each territory.
        """
    )

# Sidebar: database & provider selection
db_option = st.sidebar.selectbox("Select a Database", list(DB_OPTIONS.keys()))
st.session_state["PATH_DB"] = DB_OPTIONS.get(db_option)

provider = st.sidebar.selectbox("Proveedor LLM", ["OpenAI", "Gemini (Google)"])

sql_engine = sql.create_engine(st.session_state["PATH_DB"])
conn = sql_engine.connect()

# Credentials
if provider == "OpenAI":
    st.sidebar.header("OpenAI API Key")
    st.session_state["OPENAI_API_KEY"] = st.sidebar.text_input("OpenAI API Key", type="password")
else:
    st.sidebar.header("Google (Gemini) API Key / Service Account")
    st.session_state["GOOGLE_API_KEY"] = st.sidebar.text_input("Google API Key", type="password", help="Si usas cuenta de servicio, define GOOGLE_APPLICATION_CREDENTIALS en el sistema.")


def _fail_if_missing(name: str, obj: t.Any) -> bool:
    if obj is None:
        st.error(f"Falta dependencia: {name}. Instala la librería correspondiente y reinicia la app.")
        return True
    return False


# --- Gemini adapter ---
class _SimpleGeneration:
    def __init__(self, text: str):
        self.text = text


class _SimpleGenerateResult:
    def __init__(self, text: str):
        # Estructura similar a la que usan algunas partes de LangChain: generations -> list[list[Generation]]
        self.generations = [[_SimpleGeneration(text)]]


class GeminiAdapter:
    """Adaptador mínimo para usar google.generativeai con una interfaz compatible
    con usuarios que esperan un objeto tipo LangChain ChatModel con `generate`/`agenerate`.
    """

    def __init__(self, model: str = "gemini-1.5-mini"):
        self.model = model
        if genai is None:
            raise RuntimeError("google.generativeai no está disponible")

    def _messages_to_genai(self, messages: t.List[t.Any]) -> t.List[dict]:
        # Acepta mensajes de LangChain o strings. Normaliza a la forma {"author","content"}
        out = []
        for m in messages:
            if isinstance(m, dict):
                # ya es dict
                out.append({"author": m.get("role", "user"), "content": m.get("content", "")})
            else:
                # intentamos extraer atributos comunes
                txt = getattr(m, "content", None) or getattr(m, "text", None) or str(m)
                role = getattr(m, "role", None) or getattr(m, "type", None) or "user"
                out.append({"author": role, "content": txt})
        return out

    def generate(self, messages: t.List[t.Any], stop: t.Optional[t.List[str]] = None) -> _SimpleGenerateResult:
        # messages: lista de objetos/strings. Usaremos el último mensaje como prompt de usuario si es necesario.
        gen_messages = self._messages_to_genai(messages)
        try:
            resp = genai.chat.create(model=self.model, messages=gen_messages)
            # Extracción robusta del texto según distintas versiones del cliente
            text = None
            try:
                # estructura moderna: resp.candidates[0].content
                if hasattr(resp, "candidates") and len(resp.candidates) > 0:
                    text = resp.candidates[0].content
            except Exception:
                text = None
            if not text:
                # fallback: intentar key 'output' o 'content' en dict
                try:
                    d = dict(resp)
                    # buscar en d recursivamente podría ser excesivo; usar str fallback
                    text = d.get("output", None) or d.get("content", None) or str(resp)
                except Exception:
                    text = str(resp)
        except Exception as e:
            text = f"[Gemini error] {e}"

        return _SimpleGenerateResult(text)

    async def agenerate(self, messages: t.List[t.Any], stop: t.Optional[t.List[str]] = None):
        # Para compatibilidad con agentes asíncronos
        return self.generate(messages, stop=stop)

    # Compatibilidad: hacer la instancia callable (síncrona)
    def __call__(self, prompt: t.Union[str, t.List[t.Any], dict], **kwargs) -> str:
        """Permite que el adaptador sea pasado como callable.

        Acepta:
        - prompt: str -> lo convierte a messages con role user
        - prompt: list -> se asume lista de mensajes
        - prompt: dict -> si contiene 'messages' lo usa
        Devuelve el texto generado.
        """
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        elif isinstance(prompt, dict) and "messages" in prompt:
            messages = prompt["messages"]
        elif isinstance(prompt, list):
            messages = prompt
        else:
            # fallback
            messages = [{"role": "user", "content": str(prompt)}]

        result = self.generate(messages)
        # extraer texto
        try:
            return result.generations[0][0].text
        except Exception:
            # fallback a str
            return str(result)

    # Compatibilidad asíncrona: permite await adapter(...) en sitios que lo requieran
    async def __acall__(self, prompt: t.Union[str, t.List[t.Any], dict], **kwargs) -> str:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.__call__(prompt, **kwargs))

    # Interfaz adicional usada por algunos wrappers
    def predict(self, prompt: str) -> str:
        return self.__call__(prompt)

    def predict_messages(self, messages: t.List[t.Any]) -> str:
        return self.__call__(messages)


# Valida dependencias básicas
if provider == "OpenAI" and _fail_if_missing("langchain_openai (ChatOpenAI)", ChatOpenAI):
    st.stop()

if provider == "Gemini (Google)" and _fail_if_missing("google-generative-ai (genai)", genai):
    st.stop()

if SQLDatabaseAgent is None:
    st.error("Falta `ai_data_science_team.agents.SQLDatabaseAgent`. Instala el paquete `ai_data_science_team` o revisa las importaciones.")
    st.stop()


# Inicializar el LLM según proveedor
llm = None
if provider == "OpenAI":
    if not st.session_state.get("OPENAI_API_KEY"):
        st.info("Introduce tu OpenAI API Key en la barra lateral para continuar.")
        st.stop()
    OPENAI_LLM = ChatOpenAI(model=MODEL_LIST[0], api_key=st.session_state["OPENAI_API_KEY"])
    llm = OPENAI_LLM
else:
    if not st.session_state.get("GOOGLE_API_KEY"):
        st.info("Introduce tu Google API Key en la barra lateral para continuar.")
        st.stop()
    # Configurar cliente genai
    try:
        genai.configure(api_key=st.session_state["GOOGLE_API_KEY"])
    except Exception:
        # en algunas versiones: genai.configure(api_key=...)
        try:
            genai.configure(api_key=st.session_state["GOOGLE_API_KEY"])
        except Exception as e:
            st.error(f"No se pudo configurar genai: {e}")
            st.stop()

    try:
        llm = GeminiAdapter(model="gemini-1.5-mini")
    except Exception as e:
        st.error(f"No se pudo inicializar el adaptador de Gemini: {e}")
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
    msgs.add_ai_message("How can I help you?")

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


display_chat_history()


# Create the SQL Database Agent
sql_db_agent = SQLDatabaseAgent(
    model=llm,
    connection=conn,
    n_samples=1,
    log=False,
    bypass_recommended_steps=True,
)


async def handle_question(question: str):
    # Ejecuta la versión asíncrona del agente (si la tiene)
    if hasattr(sql_db_agent, "ainvoke_agent"):
        await sql_db_agent.ainvoke_agent(user_instructions=question)
    elif hasattr(sql_db_agent, "invoke_agent"):
        # versión sincrónica envolviendo en hilo
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: sql_db_agent.invoke_agent(user_instructions=question))
    else:
        raise RuntimeError("SQLDatabaseAgent no tiene método de invocación conocido")
    return sql_db_agent


if st.session_state["PATH_DB"] and (question := st.chat_input("Enter your question here:", key="query_input")):

    # Manejo rápido para preguntas simples sobre las tablas de la DB (evita errores del agente)
    q_lower = question.strip().lower()
    if ("what tables" in q_lower) or ("qué tablas" in q_lower) or ("que tablas" in q_lower) or ("list the tables" in q_lower) or ("tablas" in q_lower and "exi" in q_lower):
        try:
            # Usar sqlalchemy.text para compatibilidad con SQLAlchemy 1.x/2.x
            query = sql.text("SELECT name, type FROM sqlite_master WHERE type IN ('table','view') ORDER BY name;")
            result = conn.execute(query)
            rows = result.fetchall()
            st.markdown("### Tables and views in the database:")
            for r in rows:
                # r puede ser Row; acceder por índice o por nombre
                try:
                    name = r[0]
                    typ = r[1]
                except Exception:
                    name = r['name'] if 'name' in r else str(r)
                    typ = r['type'] if 'type' in r else ''
                st.write(f"- {name} ({typ})")
        except Exception as e:
            # fallback: usar pandas.read_sql_query si la conexión es compatible con DBAPI
            try:
                df = pd.read_sql_query("SELECT name, type FROM sqlite_master WHERE type IN ('table','view') ORDER BY name;", conn)
                st.markdown("### Tables and views in the database (via pandas):")
                for _, row in df.iterrows():
                    st.write(f"- {row['name']} ({row['type']})")
            except Exception as e2:
                st.error(f"Error querying database directly: {e}; fallback failed: {e2}")
        st.stop()

    with st.spinner("Thinking..."):
        st.chat_message("human").write(question)
        msgs.add_user_message(question)

        error_occured = False
        try:
            result = asyncio.run(handle_question(question))
        except Exception as e:
            error_occured = True
            # Intentar extraer estado interno del agente para depuración
            debug_info = {}
            try:
                debug_info['agent_repr'] = repr(sql_db_agent)
            except Exception:
                debug_info['agent_repr'] = '<unavailable>'
            response_text = f"I'm sorry. I am having difficulty answering that question.\n\nError: {e}\n\nDebug: {debug_info}"
            msgs.add_ai_message(response_text)
            st.chat_message("ai").write(response_text)
            st.error(f"Error: {e}")

        if not error_occured:
            try:
                sql_query = result.get_sql_query_code()
            except Exception as e:
                sql_query = None
                st.warning(f"No SQL query available (error reading query): {e}")

            try:
                response_df = result.get_data_sql()
            except Exception as e:
                response_df = None
                st.warning(f"No dataframe available (error reading data): {e}")

            if sql_query:
                response_1 = f"### SQL Results:\n\nSQL Query:\n\n```sql\n{sql_query}\n```\n\nResult:"

                df_index = len(st.session_state.dataframes)
                st.session_state.dataframes.append(response_df)

                msgs.add_ai_message(response_1)
                msgs.add_ai_message(f"DATAFRAME_INDEX:{df_index}")

                st.chat_message("ai").write(response_1)
                st.dataframe(response_df)

            else:
                msgs.add_ai_message("No SQL query was generated by the agent.")
                st.chat_message("ai").write("No SQL query was generated by the agent.")
