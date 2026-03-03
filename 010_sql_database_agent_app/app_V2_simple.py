"""
AI-TIP 010 - SQL Database Agent App (V2 Ultra Fixed)
"""

from __future__ import annotations

import streamlit as st
import sqlalchemy as sql
import pandas as pd
import asyncio
import typing as t
import json
from datetime import datetime

# Configuración básica
st.set_page_config(page_title="SQL Database Agent", page_icon="📊")

# Función para obtener información de la base de datos
def get_db_info(conn, info_type="all"):
    """Obtiene información sobre objetos de la base de datos.
    
    Args:
        conn: Conexión SQLAlchemy
        info_type: "all", "tables", o "views"
    """
    try:
        query = """
        SELECT 
            name,
            type,
            sql as definition
        FROM sqlite_master 
        WHERE type IN ('table', 'view')
        """
        
        if info_type == "tables":
            query += " AND type = 'table'"
        elif info_type == "views":
            query += " AND type = 'view'"
            
        query += " ORDER BY type, name;"
        
        try:
            result = conn.execute(sql.text(query))
            rows = result.fetchall()
        except Exception:
            # Fallback a pandas
            df = pd.read_sql_query(query, conn)
            rows = df.to_records(index=False)
            
        objects = []
        for row in rows:
            try:
                name = row[0]
                type_ = row[1]
                definition = row[2]
            except Exception:
                name = row.get('name', str(row))
                type_ = row.get('type', '?')
                definition = row.get('definition', '')
                
            objects.append({
                "name": name,
                "type": type_,
                "definition": definition
            })
            
        return objects
        
    except Exception as e:
        raise RuntimeError(f"Error consultando la base de datos: {e}")

def display_db_objects(objects, show_type="all"):
    """Muestra los objetos de la base de datos de forma amigable."""
    if not objects:
        st.info("No hay objetos para mostrar.")
        return
        
    if show_type == "views":
        st.markdown("### 📊 Vistas en la base de datos:")
        views = [obj for obj in objects if obj["type"].lower() == "view"]
        if views:
            for view in views:
                with st.expander(f"Vista: {view['name']}", expanded=False):
                    st.code(view['definition'], language='sql')
        else:
            st.info("No hay vistas definidas en la base de datos.")
            
    elif show_type == "tables":
        st.markdown("### 📋 Tablas en la base de datos:")
        tables = [obj for obj in objects if obj["type"].lower() == "table"]
        for table in tables:
            st.write(f"- {table['name']}")
            
    else:
        st.markdown("### 📊 Objetos en la base de datos")
        
        tables = [obj for obj in objects if obj["type"].lower() == "table"]
        if tables:
            st.markdown("#### Tablas:")
            for table in tables:
                st.write(f"- {table['name']}")
                
        views = [obj for obj in objects if obj["type"].lower() == "view"]
        if views:
            st.markdown("#### Vistas:")
            for view in views:
                with st.expander(f"Vista: {view['name']}", expanded=False):
                    st.code(view['definition'], language='sql')

def main():
    st.title("SQL Database Agent — V2 Ultra Fixed")
    
    # Configuración inicial
    DB_PATH = "data/northwind.db"
    DB_URL = f"sqlite:///{DB_PATH}"
    
    try:
        engine = sql.create_engine(DB_URL)
        conn = engine.connect()
    except Exception as e:
        st.error(f"Error conectando a la base de datos: {e}")
        st.stop()
        
    # Input del usuario
    question = st.chat_input("Escribe tu consulta aquí:", key="query_input")
    if not question:
        return
        
    st.chat_message("human").write(question)
    
    # Detectar tipo de pregunta
    q_lower = question.strip().lower()
    is_view_question = any(x in q_lower for x in [
        "what views", "qué vistas", "que vistas",
        "list the views", "vistas", "cuales son las vistas"
    ])
    is_table_question = any(x in q_lower for x in [
        "what tables", "qué tablas", "que tablas",
        "list the tables", "tablas"
    ])
    
    # Manejar consultas sobre estructura de la DB
    if is_view_question or is_table_question:
        try:
            if is_view_question:
                objects = get_db_info(conn, "views")
                display_db_objects(objects, "views")
            elif is_table_question:
                objects = get_db_info(conn, "tables")
                display_db_objects(objects, "tables")
                
            st.chat_message("ai").write(
                "¿Necesitas más información sobre algún objeto específico?"
            )
            
        except Exception as e:
            st.error(f"Error al consultar la base de datos: {e}")
            
    else:
        st.error("""
        Lo siento, esta versión simplificada solo puede mostrar la estructura de la base de datos.
        
        Prueba preguntas como:
        - ¿Qué vistas hay?
        - ¿Qué tablas existen?
        """)
        
if __name__ == "__main__":
    main()