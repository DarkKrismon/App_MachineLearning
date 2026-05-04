import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

''' 
Encargado de las funciones para la Fase 4
del modelo No Supervisado.
Funciones que reciben y devuelven valores listos.
'''

def preparar_datos_clustering(df, columnas):
    """Escala los datos porque K-Means es súper sensible a las magnitudes."""
    try:
        df_filtrado = df[columnas].dropna().copy()
        scaler = StandardScaler()
        datos_escalados = scaler.fit_transform(df_filtrado)
        return df_filtrado, datos_escalados, scaler
    except ValueError as e:
        st.error("🚨 Error matemático: Has seleccionado una columna que contiene texto o formatos no numéricos. K-Means solo acepta números.")
        st.stop()

def calcular_codo(datos_escalados, max_clusters=10):
    """Ejecuta K-Means múltiples veces para encontrar el número ideal de grupos."""
    inercia = []
    rango = range(2, max_clusters + 1)
    for k in rango:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
        kmeans.fit(datos_escalados)
        inercia.append(kmeans.inertia_)
    return list(rango), inercia

def ejecutar_clustering(df_original, datos_escalados, n_clusters):
    """Aplica K-Means definitivo y devuelve el dataset con la nueva columna."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    tribus = kmeans.fit_predict(datos_escalados)
    
    df_resultado = df_original.copy()
    df_resultado['Tribu'] = tribus
    df_resultado['Tribu'] = df_resultado['Tribu'].astype(str) # Lo pasamos a texto para los colores de los gráficos
    return df_resultado, kmeans

def aplicar_pca_2d(datos_escalados, tribus):
    """Aplasta las dimensiones a 2D (X e Y) para poder dibujarlas en la pantalla. PCA"""
    pca = PCA(n_components=2)
    componentes = pca.fit_transform(datos_escalados)
    
    df_pca = pd.DataFrame(data=componentes, columns=['Componente_X', 'Componente_Y'])
    df_pca['Tribu'] = tribus.astype(str)
    
    varianza_explicada = sum(pca.explained_variance_ratio_) * 100
    return df_pca, varianza_explicada

def obtener_centroides_radar(df_resultado, columnas):
    """Calcula la media de cada variable por Tribu para dibujar el Radar."""
    perfiles = df_resultado.groupby('Tribu')[columnas].mean().reset_index()
    return perfiles