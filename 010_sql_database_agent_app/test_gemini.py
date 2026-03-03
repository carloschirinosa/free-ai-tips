import streamlit as st
import google.generativeai as genai
import json

def test_gemini_connection():
    try:
        # Configurar el cliente con la API key
        api_key = st.secrets["GOOGLE_API_KEY"]
        genai.configure(api_key=api_key)
        
        # Listar modelos disponibles
        models = genai.list_models()
        print("Modelos disponibles:")
        for model in models:
            print(f"- {model.name}")
            print(f"  Métodos soportados: {model.supported_generation_methods}")
        
        MODEL_NAME = 'models/gemini-2.5-pro'
        
        # Prueba 1: Generación simple
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content("Di hola en español")
        print("\nPrueba 1 - Generación simple:")
        print(f"Respuesta: {response.text}")
        
        # Prueba 2: Consulta SQL
        sql_prompt = """
        Dado este esquema:
        CREATE TABLE usuarios (
            id INT PRIMARY KEY,
            nombre VARCHAR(100),
            edad INT
        );
        
        Escribe una consulta SQL para obtener todos los usuarios mayores de 25 años.
        """
        response = model.generate_content(sql_prompt)
        print("\nPrueba 2 - Generación SQL:")
        print(f"Respuesta: {response.text}")
        
        # Prueba 3: Contar tokens
        token_count = model.count_tokens(sql_prompt)
        print("\nPrueba 3 - Conteo de tokens:")
        print(f"Tokens en el prompt SQL: {token_count.total_tokens}")
        
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_gemini_connection()
    print(f"\nPrueba completada {'exitosamente' if success else 'con errores'}")