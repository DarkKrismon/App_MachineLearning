import ollama

''' 
Este script gestiona la conexión local con tu Inteligencia Artificial.
Su única misión es coger un texto, enviarlo al modelo y devolverte la respuesta limpia.
'''



MODELO_DEFAULT = "llama3" 

def consultar_diagnostico(resumen_total, modelo=MODELO_DEFAULT):
    """Envía el resumen técnico a Ollama para generar un reporte amigable."""
    prompt_sistema = """
    Eres un asistente experto en Data Science. Tu objetivo es leer el resumen técnico 
    de los archivos que ha subido el usuario y explicarle de forma muy clara, breve y amigable 
    el estado de sus datos. Si hay nulos o múltiples archivos, adviértele que necesitaremos 
    limpiar y unificar antes de hacer Machine Learning. NO escribas código, solo da el diagnóstico y en Español.
    """
    
    prompt_usuario = f"Aquí tienes el resumen técnico de los datos:\n{resumen_total}"
    
    try:
        respuesta = ollama.chat(model=modelo, messages=[
            {'role': 'system', 'content': prompt_sistema},
            {'role': 'user', 'content': prompt_usuario}
        ])
        return respuesta['message']['content']
    except Exception as e:
        return f"Error Conexión con Ollama. Asegúrate de que está activo localmente."
    

def generar_reporte_ejecutivo(resultados_ml, modelo=MODELO_DEFAULT):
    """Convierte métricas matemáticas crudas en un reporte de negocio."""

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
    
    respuesta = ollama.chat(model=modelo, messages=[
        {'role': 'system', 'content': prompt_sistema},
        {'role': 'user', 'content': f"Genera el reporte ejecutivo basándote en esto:\n{datos_crudos}"}
    ])

    return respuesta['message']['content']


def interpretar_grupos(df_centroides):
    """
    Toma el DataFrame de los centroides, lo convierte a texto y le pide a Ollama
    que le ponga nombre y apellidos a cada grupo.
    """

    datos_crudos = df_centroides.to_markdown(index=False)
    
    prompt = f"""
    Eres un analista de datos experto en sociología y estrategia de negocio.
    Acabo de ejecutar un modelo de Clustering (K-Means) sobre una base de datos.
    Aquí tienes la tabla con los 'Centroides' (el valor medio de cada variable para cada Tribu/Grupo):

    {datos_crudos}

    Tu objetivo es traducir estos números fríos en perfiles humanos que el equipo directivo pueda entender y en Español.
    Haz exactamente lo siguiente para cada Tribu:
    1. Ponle un TÍTULO llamativo (Ej: "Tribu 0: La Élite", "Tribu 1: Familias Numerosas"...).
    2. Lista en un punto llamado 'Los Datos' las 2 o 3 métricas que más destacan matemáticamente.
    3. Escribe un párrafo llamado 'El Perfil' describiendo quiénes son sociológicamente.

    Responde directamente con el análisis en formato Markdown. Sé analítico, crudo y realista. No inventes datos que no estén en la tabla.
    """
    try:
        respuesta = ollama.chat(model=MODELO_DEFAULT, messages=[{'role': 'user', 'content': prompt}])
        return respuesta['message']['content']
    except Exception as e:
        return f"Error al conectar con Ollama: {e}"