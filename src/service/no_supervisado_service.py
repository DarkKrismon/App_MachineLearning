import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

def preprocesar_datos_clustering(df):
    """
    Convierte todo el texto en números de forma segura y escala matemáticamente.
    K-Means es ciego si no escalamos los datos (un salario de 50.000 aplastaría a una edad de 40).
    """
    df_procesado = df.copy()
    traductores = {}
    
    # 1. Label Encoding para todo lo que no sea numérico
    for col in df_procesado.columns:
        if df_procesado[col].dtype in ['object', 'string', 'category', 'bool']:
            le = LabelEncoder()
            df_procesado[col] = le.fit_transform(df_procesado[col].astype(str))
            traductores[col] = le
            
    # 2. Escalado Estándar (Vital para distancias euclidianas)
    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(df_procesado)
    
    return datos_escalados, scaler, traductores


def escanear_mejores_k(datos_escalados, max_k=10):
    """
    Prueba múltiples agrupaciones y devuelve las métricas matemáticas 
    para que el usuario no tenga que adivinar cuántos clusters crear.
    """
    max_k = min(max_k, len(datos_escalados) - 1)
    k_values = range(2, max_k + 1)
    
    inercia = []
    silueta = []
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        etiquetas = kmeans.fit_predict(datos_escalados)
        inercia.append(kmeans.inertia_)
        silueta.append(silhouette_score(datos_escalados, etiquetas))
        
    # El K ideal suele tener el mayor silhouette_score
    mejor_k = k_values[np.argmax(silueta)]
    
    return {
        "k_values": list(k_values),
        "inercia": inercia,
        "silueta": silueta,
        "mejor_k_recomendado": mejor_k
    }


def entrenar_clustering_3d(df, num_clusters, algoritmo):
    """
    Entrena el modelo definitivo (K-Means o Jerárquico) y comprime la realidad a 3 dimensiones 
    para poder dibujarla en un gráfico 3D interactivo.
    """
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.decomposition import PCA
    from sklearn.metrics import silhouette_score
    
    datos_escalados, scaler, traductores = preprocesar_datos_clustering(df)
    
    # Elección del Algoritmo
    if algoritmo == "K-Means":
        modelo = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        clusters = modelo.fit_predict(datos_escalados)
    elif algoritmo == "Clustering Jerárquico":
        modelo = AgglomerativeClustering(n_clusters=num_clusters)
        clusters = modelo.fit_predict(datos_escalados)
    else:
        raise ValueError("Algoritmo no soportado")
    
    # Evaluar calidad
    calidad = silhouette_score(datos_escalados, clusters)
    
    # Reducir a 3 dimensiones para la visualización
    pca = PCA(n_components=3)
    datos_3d = pca.fit_transform(datos_escalados)
    varianza_explicada = pca.explained_variance_ratio_.sum() * 100
    
    df_resultados = df.copy()
    df_resultados['Cluster'] = clusters
    df_resultados['PCA_X'] = datos_3d[:, 0]
    df_resultados['PCA_Y'] = datos_3d[:, 1]
    df_resultados['PCA_Z'] = datos_3d[:, 2]
    
    return {
        "df_con_clusters": df_resultados,
        "modelo_usado": modelo,
        "nombre_algoritmo": algoritmo,
        "calidad_silueta": calidad,
        "varianza_pca": varianza_explicada,
        "scaler": scaler,
        "traductores": traductores
    }


def obtener_perfil_clusters(df_con_clusters):
    """
    Calcula la media (o la moda para texto) de cada variable dentro de cada cluster.
    Esto permite a la IA de Negocio explicar qué hace único a cada grupo.
    """
    columnas_analisis = [c for c in df_con_clusters.columns if c not in ['Cluster', 'PCA_X', 'PCA_Y', 'PCA_Z']]
    
    def calcular_perfil(x):
        # Usamos la API oficial de Pandas para saber si es puramente matemático
        if pd.api.types.is_numeric_dtype(x):
            return x.mean()
        # Para todo lo demás (textos, categorías, fechas), cogemos el más repetido
        else:
            return x.mode()[0] if not x.mode().empty else "Desconocido"

    resumen = df_con_clusters.groupby('Cluster')[columnas_analisis].agg(calcular_perfil).reset_index()
    
    return resumen


def predecir_nuevos_datos(df_nuevo, resultados):
    """
    Procesa clientes nuevos y los asigna a las tribus existentes.
    Maneja la limitación matemática del Clustering Jerárquico usando un clasificador KNN.
    """
    import pandas as pd
    from sklearn.neighbors import KNeighborsClassifier
    
    df_proc = df_nuevo.copy()
    traductores = resultados['traductores']
    scaler = resultados['scaler']
    modelo = resultados['modelo_usado']
    algoritmo = resultados['nombre_algoritmo']

    # 1. Aplicar la misma traducción y limpieza al NUEVO usuario
    for col in df_proc.columns:
        if col in traductores:
            le = traductores[col]
            clase_defecto = le.classes_[0]
            # Rellenar nulos
            df_proc[col] = df_proc[col].fillna(clase_defecto).astype(str)
            # Manejar datos no vistos mapeándolos a -1
            clases_conocidas = set(le.classes_)
            df_proc[col] = df_proc[col].map(lambda x: le.transform([x])[0] if x in clases_conocidas else -1).fillna(-1)
        else:
            df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce').fillna(0)

    # 2. Escalar matemáticamente al nuevo usuario
    X_nuevo = scaler.transform(df_proc)

    # 3. La Predicción (El Hack)
    if algoritmo == "K-Means":
        predicciones = modelo.predict(X_nuevo)
    elif algoritmo == "Clustering Jerárquico":
        # Hack: Usamos los datos ya clasificados para enseñar a un KNN a asignar nuevos
        df_entrenamiento = resultados['df_con_clusters'].copy()
        columnas_base = [c for c in df_entrenamiento.columns if c not in ['Cluster', 'PCA_X', 'PCA_Y', 'PCA_Z']]
        
        df_train_limpio = df_entrenamiento[columnas_base].copy()

        # 💉 LA CURA: Traducir los textos del entrenamiento a números antes de dárselos al KNN
        for col in df_train_limpio.columns:
            if col in traductores:
                # Usamos el traductor para volver a convertir 'male' en el número original
                df_train_limpio[col] = traductores[col].transform(df_train_limpio[col].astype(str))
            else:
                df_train_limpio[col] = pd.to_numeric(df_train_limpio[col], errors='coerce').fillna(0)

        # Ahora sí, escalamos sin que explote
        datos_originales_escalados = scaler.transform(df_train_limpio)
        etiquetas_originales = df_entrenamiento['Cluster']

        # Entrenamos al portero y predecimos
        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(datos_originales_escalados, etiquetas_originales)
        predicciones = knn.predict(X_nuevo)
    else:
        raise ValueError("Algoritmo no soportado en producción.")

    return predicciones