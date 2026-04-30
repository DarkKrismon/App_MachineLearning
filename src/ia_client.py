import streamlit as st
from groq import Groq

''' 
Este script gestiona la conexión en la nube con la Inteligencia Artificial a través de Groq.
Su única misión es coger un texto, enviarlo al modelo y devolverte la respuesta limpia.
'''


try:
    cliente_ia = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception:
    cliente_ia = None

# Usamos el modelo Llama 3 de 8 billones de parámetros optimizado por Groq
MODELO_DEFAULT = "llama-3.1-8b-instant" 


def consultar_diagnostico(resumen_total, modelo=MODELO_DEFAULT):
    """Envía el resumen técnico a la IA para generar un reporte amigable."""
    if not cliente_ia:
        return "🚨 Error: No se ha configurado la API Key de Groq en los secretos de Streamlit."

    prompt_sistema = """
    Eres un asistente experto en Data Science. Tu objetivo es leer el resumen técnico 
    de los archivos que ha subido el usuario y explicarle de forma muy clara, breve y amigable 
    el estado de sus datos. Si hay nulos o múltiples archivos, adviértele que necesitaremos 
    limpiar y unificar antes de hacer Machine Learning. NO escribas código, solo da el diagnóstico y en Español.
    """
    
    prompt_usuario = f"Aquí tienes el resumen técnico de los datos:\n{resumen_total}"
    
    try:
        respuesta = cliente_ia.chat.completions.create(
            messages=[
                {'role': 'system', 'content': prompt_sistema},
                {'role': 'user', 'content': prompt_usuario}
            ],
            model=modelo,
            temperature=0.3 # Baja temperatura para análisis precisos
        )
        return respuesta.choices[0].message.content
    except Exception as e:
        return f"🚨 Error en la conexión con la IA (Diagnóstico): {e}"
    

def generar_reporte_ejecutivo(resultados_ml, modelo=MODELO_DEFAULT):
    """Convierte métricas matemáticas crudas en un reporte de negocio."""
    if not cliente_ia:
        return "🚨 Error: No se ha configurado la API Key de Groq."

    prompt_sistema = """
    Eres un Consultor Estratégico de Negocio de alto nivel.
    Te voy a pasar los resultados de un modelo de Machine Learning.
    Tu objetivo es traducir estos datos técnicos a un lenguaje ejecutivo claro y en Español.
    No hables de algoritmos, matrices o hiperparámetros. 
    Habla de fiabilidad, variables clave, impacto en el negocio.
    """
    
    datos_crudos = f"""
    - Problema a resolver: {resultados_ml['tipo_problema']}
    - Precisión / Fiabilidad del modelo: {resultados_ml['ganador_score_cv'] * 100:.2f}%
    - Algoritmo ganador: {resultados_ml['ganador_nombre']}
    """
    
    try:
        respuesta = cliente_ia.chat.completions.create(
            messages=[
                {'role': 'system', 'content': prompt_sistema},
                {'role': 'user', 'content': f"Genera el reporte ejecutivo basándote en esto:\n{datos_crudos}"}
            ],
            model=modelo,
            temperature=0.5 # Un poco más de temperatura para mejorar la redacción
        )
        return respuesta.choices[0].message.content
    except Exception as e:
        return f"🚨 Error al generar el reporte ejecutivo: {e}"


def interpretar_grupos(df_centroides, modelo=MODELO_DEFAULT):
    """
    Toma el DataFrame de los centroides, lo convierte a texto y le pide a la IA
    que le ponga nombre y apellidos a cada grupo.
    """
    if not cliente_ia:
        return "🚨 Error: No se ha configurado la API Key de Groq."

    datos_crudos = df_centroides.to_markdown(index=False)
    
    prompt_sistema = """
    Eres un analista de datos experto en sociología y estrategia de negocio.
    Acabo de ejecutar un modelo de Clustering (K-Means) sobre una base de datos.
    Tu objetivo es traducir los números fríos de los centroides en perfiles humanos que el equipo directivo pueda entender y en Español.
    Haz exactamente lo siguiente para cada Tribu:
    1. Ponle un TÍTULO llamativo (Ej: "Tribu 0: La Élite", "Tribu 1: Familias Numerosas"...).
    2. Lista en un punto llamado 'Los Datos' las 2 o 3 métricas que más destacan matemáticamente.
    3. Escribe un párrafo llamado 'El Perfil' describiendo quiénes son sociológicamente.

    Responde directamente con el análisis en formato Markdown. Sé analítico, crudo y realista. No inventes datos que no estén en la tabla.
    """

    prompt_usuario = f"Aquí tienes la tabla con los 'Centroides' (el valor medio de cada variable para cada Tribu/Grupo):\n\n{datos_crudos}"
    
    try:
        respuesta = cliente_ia.chat.completions.create(
            messages=[
                {'role': 'system', 'content': prompt_sistema},
                {'role': 'user', 'content': prompt_usuario}
            ],
            model=modelo,
            temperature=0.4
        )
        return respuesta.choices[0].message.content
    except Exception as e:
        return f"🚨 Error al interpretar las tribus: {e}"