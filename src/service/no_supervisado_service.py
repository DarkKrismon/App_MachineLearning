"""
Módulo de Servicio para Clustering (Machine Learning No Supervisado).
Contiene la lógica matemática para preprocesar datos, escanear el K óptimo, 
entrenar modelos (K-Means, Jerárquico), reducir dimensiones (PCA 3D) 
y simular predicciones sobre algoritmos no predictivos usando K-NN.
"""
import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier


def preprocesar_datos_clustering(df: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler, dict]:
    """
    Convierte el texto en números mediante LabelEncoding y aplica un StandardScaler.
    Paso crítico para distancias euclidianas: evita que variables con números 
    muy grandes dominen sobre variables más pequeñas.
    
    Retorna:
    -------
    Tuple
        Matriz numpy con datos escalados, el objeto StandardScaler y el diccionario de traductores.
    """

    df_procesado = df.copy()
    traductores = {}
    
    for col in df_procesado.columns:
        if df_procesado[col].dtype in ['object', 'string', 'category', 'bool']:
            le = LabelEncoder()
            df_procesado[col] = le.fit_transform(df_procesado[col].astype(str))
            traductores[col] = le
            
    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(df_procesado)
    
    return datos_escalados, scaler, traductores


def escanear_mejores_k(datos_escalados: np.ndarray, max_k: int = 10) -> Dict[str, Any]:
    """
    Entrena múltiples modelos K-Means en segundo plano para escanear el 'Silhouette Score'.
    Calcula matemáticamente cuál es el número óptimo de agrupaciones sin requerir
    adivinación por parte del usuario.
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


def entrenar_clustering_3d(df: pd.DataFrame, num_clusters: int, algoritmo: str) -> Dict[str, Any]:
    """
    Pipeline completo: preprocesa, entrena el algoritmo seleccionado, 
    calcula la métrica de silueta y aplica PCA (Principal Component Analysis) 
    para reducir N dimensiones a 3 componentes principales renderizables.
    """
    
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


def obtener_perfil_clusters(df_con_clusters: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula el 'Centroide' conceptual de cada tribu.
    Aplica la media para columnas numéricas y la moda (valor más frecuente) 
    para variables categóricas, generando un perfil limpio para la IA o negocio.
    """

    columnas_analisis = [c for c in df_con_clusters.columns if c not in ['Cluster', 'PCA_X', 'PCA_Y', 'PCA_Z']]
    
    def calcular_perfil(x):
        if pd.api.types.is_numeric_dtype(x):
            return x.mean()
        else:
            return x.mode()[0] if not x.mode().empty else "Desconocido"

    resumen = df_con_clusters.groupby('Cluster')[columnas_analisis].agg(calcular_perfil).reset_index()
    
    return resumen


def predecir_nuevos_datos(df_nuevo: pd.DataFrame, resultados: dict) -> np.ndarray:
    """
    Clasifica nuevos individuos en las tribus ya existentes.
    Si el algoritmo base fue Clustering Jerárquico (que carece de función .predict()),
    inyecta un modelo K-Nearest Neighbors (KNN) intermedio entrenado al vuelo con 
    los centroides actuales para resolver la limitación matemática.
    """
    
    df_proc = df_nuevo.copy()
    traductores = resultados['traductores']
    scaler = resultados['scaler']
    modelo = resultados['modelo_usado']
    algoritmo = resultados['nombre_algoritmo']


    for col in df_proc.columns:
        if col in traductores:
            le = traductores[col]
            clase_defecto = le.classes_[0]
            df_proc[col] = df_proc[col].fillna(clase_defecto).astype(str)
            clases_conocidas = set(le.classes_)
            df_proc[col] = df_proc[col].map(lambda x: le.transform([x])[0] if x in clases_conocidas else -1).fillna(-1)
        else:
            df_proc[col] = pd.to_numeric(df_proc[col], errors='coerce').fillna(0)

    
    X_nuevo = scaler.transform(df_proc)

    if algoritmo == "K-Means":
        predicciones = modelo.predict(X_nuevo)
    elif algoritmo == "Clustering Jerárquico":
        df_entrenamiento = resultados['df_con_clusters'].copy()
        columnas_base = [c for c in df_entrenamiento.columns if c not in ['Cluster', 'PCA_X', 'PCA_Y', 'PCA_Z']]
        
        df_train_limpio = df_entrenamiento[columnas_base].copy()

        for col in df_train_limpio.columns:
            if col in traductores:
                df_train_limpio[col] = traductores[col].transform(df_train_limpio[col].astype(str))
            else:
                df_train_limpio[col] = pd.to_numeric(df_train_limpio[col], errors='coerce').fillna(0)

        datos_originales_escalados = scaler.transform(df_train_limpio)
        etiquetas_originales = df_entrenamiento['Cluster']

        knn = KNeighborsClassifier(n_neighbors=3)
        knn.fit(datos_originales_escalados, etiquetas_originales)
        predicciones = knn.predict(X_nuevo)

    return predicciones