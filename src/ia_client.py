"""
Módulo de conexión con la API de Groq (LLMs).
Gestiona la comunicación en la nube con los modelos de Inteligencia Artificial
para generar diagnósticos y reportes de negocio a partir de datos técnicos.
"""
import streamlit as st
import pandas as pd
from groq import Groq


try:
    cliente_ia = Groq(api_key=st.secrets["GROQ_API_KEY"])
except Exception:
    cliente_ia = None

# Usamos el modelo Llama 3 de 8 billones de parámetros optimizado por Groq
MODELO_DEFAULT = "llama-3.1-8b-instant" 


def consultar_diagnostico(resumen_total: str, modelo: str = MODELO_DEFAULT) -> str:
    """
    Envía el resumen técnico inicial de los datos a la IA para generar un 
    diagnóstico fácil de entender para el usuario.
    
    Parámetros:
    ----------
    Resumen_total : str
        El texto crudo con el conteo de filas, columnas y valores nulos.
    Modelo : str
        El modelo de Groq a utilizar (por defecto llama-3.1-8b-instant).
    """

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
            temperature=0.3
        )
        return respuesta.choices[0].message.content
    except Exception as e:
        return f"🚨 Error en la conexión con la IA (Diagnóstico): {e}"
    

def generar_reporte_ejecutivo(resultados_ml: dict, df_test: pd.DataFrame, columna_target: str, modelo: str = MODELO_DEFAULT) -> str:
    """
    Convierte las métricas matemáticas crudas del modelo supervisado 
    (precisión, algoritmo ganador) en un reporte estratégico de negocio.
    
    Parámetros:
    ----------
    resultados_ml : dict
        Diccionario con las métricas del modelo (score, ganador, etc.).
    df_test : pd.DataFrame
        Los datos de prueba para extraer el nombre de las variables usadas.
    columna_target : str
        El nombre de la columna que el modelo intentó predecir.
    """
    
    if not cliente_ia:
        return "🚨 Error: No se ha configurado la API Key de Groq."

    prompt_sistema = """
    Eres un Consultor Estratégico de Negocio de alto nivel.
    Te voy a pasar los resultados de un modelo de Machine Learning.
    Tu objetivo es traducir estos datos técnicos a un lenguaje ejecutivo claro y en Español.
    No hables de algoritmos, matrices o hiperparámetros. 
    Habla de fiabilidad, variables clave, impacto en el negocio.
    """
    
    columnas_usadas = ", ".join(df_test.columns.tolist())
    
    datos_crudos = f"""
    - Problema a resolver: {resultados_ml['tipo_problema']}
    - Objetivo de negocio (Target): Predecir la variable '{columna_target}'
    - Precisión / Fiabilidad del modelo: {resultados_ml['ganador_score_cv'] * 100:.2f}%
    - Algoritmo ganador: {resultados_ml['ganador_nombre']}
    - Variables/Datos utilizados para la predicción: {columnas_usadas}
    """
    
    try:
        respuesta = cliente_ia.chat.completions.create(
            messages=[
                {'role': 'system', 'content': prompt_sistema},
                {'role': 'user', 'content': f"Genera el reporte ejecutivo basándote en esto:\n{datos_crudos}"}
            ],
            model=modelo,
            temperature=0.5
        )
        return respuesta.choices[0].message.content
    except Exception as e:
        return f"🚨 Error al generar el reporte ejecutivo: {e}"


def interpretar_grupos(df_centroides: pd.DataFrame, modelo: str = MODELO_DEFAULT) -> str:
    """
    Traduce los valores medios (centroides) de cada cluster en perfiles 
    humanos o tribus de negocio para la fase No Supervisada.
    
    Parámetros:
    ----------
    df_centroides : pd.DataFrame
        Tabla con los promedios matemáticos de cada grupo.
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



def generar_reporte_clustering(df_perfiles: pd.DataFrame, modelo: str = MODELO_DEFAULT) -> str:
    """
    Redacta un informe estratégico bautizando a cada cluster basándose 
    en sus características medias y propone estrategias de acción.
    
    Parámetros:
    ----------
    df_perfiles : pd.DataFrame
        Tabla con los perfiles matemáticos de cada tribu/cluster.
    """

    if not cliente_ia:
        return "🚨 Error: No se ha configurado la API Key de Groq en los secretos de Streamlit."

    # Convertimos el DataFrame a texto estructurado para que la IA lo entienda
    perfiles_texto = df_perfiles.to_string(index=False)
    
    prompt_sistema = """
    Eres un Consultor Estratégico de Datos de alto nivel. He aplicado un modelo de Machine Learning No Supervisado 
    y he encontrado distintos grupos (Clusters) en mis datos.
    Tu misión es traducir los números fríos en perfiles sociológicos y estrategias de negocio.
    """
    
    prompt_usuario = f"""
    Aquí tienes los valores promedios (o la moda) de cada variable para cada grupo descubierto:
    
    {perfiles_texto}
    
    Por favor, redacta un informe ejecutivo estructurado con lo siguiente:
    1. Bautiza a cada Cluster con un nombre comercial y llamativo (Ej: "Los Ahorradores Cautelosos", "Los Fieles VIP", etc).
    2. Describe en una frase qué hace único a ese grupo basándote en sus números.
    3. Da 1 recomendación de negocio accionable o estrategia para ese grupo en concreto.
    
    Usa formato Markdown (negritas, listas). Sé directo, profesional e inspirador. No expliques qué es el clustering.
    """
    
    try:
        respuesta = cliente_ia.chat.completions.create(
            messages=[
                {'role': 'system', 'content': prompt_sistema},
                {'role': 'user', 'content': prompt_usuario}
            ],
            model=modelo,
            temperature=0.7
        )
        return respuesta.choices[0].message.content
    except Exception as e:
        return f"❌ Error al generar el reporte con Groq: {e}"