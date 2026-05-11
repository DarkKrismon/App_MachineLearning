"""
Módulo de Servicio de Datos (Data Service).
Contiene toda la lógica matemática y de manipulación de DataFrames de Pandas 
para la Fase 1 (Ingesta) y Fase 2 (Limpieza e Ingeniería de Características).
Este archivo no tiene interfaz, solo recibe datos, los opera y los devuelve.
"""
import pandas as pd
import streamlit as st
from typing import Any


#------------------------------------------------------------------------------
# FASE 1
#------------------------------------------------------------------------------
def leer_dataset_universal(archivo: Any) -> pd.DataFrame:
    """
    Lee archivos subidos en formato CSV, Excel o Parquet.
    
    Parámetros:
    ----------
    archivo : Any
        El objeto de archivo subido a través de st.file_uploader.
        
    Retorna:
    -------
    pd.DataFrame o None
        El DataFrame cargado en memoria, o None si el formato no es compatible.
    """
    
    if archivo is None:
        return None
        
    nombre_archivo = archivo.name.lower()
    
    try:
        # Si es un CSV
        if nombre_archivo.endswith('.csv'):
            df = pd.read_csv(archivo, sep=None, engine='python')
                
        # Si es un Excel
        elif nombre_archivo.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(archivo)
            
        # Si es Parquet
        elif nombre_archivo.endswith('.parquet'):
            df = pd.read_parquet(archivo)
            
        else:
            st.error("❌ Formato de archivo no soportado.")
            return None
            
        return df

    except Exception as e:
        st.error(f"🚨 Error crítico al leer el archivo: {e}")
        return None
    

def generar_resumen_df(df: pd.DataFrame, nombre_archivo: str) -> str:
    """
    Analiza un DataFrame para extraer un diagnóstico inicial (filas, columnas, nulos).
    Este texto es el que consume posteriormente el LLM de Groq.
    """

    filas, columnas = df.shape
    nulos = df.isnull().sum().sum()
    columnas_con_nulos = df.columns[df.isnull().any()].tolist()
    
    resumen = f"Archivo: {nombre_archivo}\n"
    resumen += f"- Filas: {filas}, Columnas: {columnas}\n"
    resumen += f"- Total de valores nulos: {nulos}\n"
    if nulos > 0:
        resumen += f"- Columnas con nulos: {', '.join(columnas_con_nulos)}\n"
    return resumen


def obtener_columnas_comunes(df1: pd.DataFrame, df2: pd.DataFrame) -> list:
    """
    Compara las cabeceras de dos tablas y extrae los nombres coincidentes 
    para sugerir claves automáticas durante un JOIN.
    """

    return list(set(df1.columns).intersection(set(df2.columns)))


def unificar_tablas(df1: pd.DataFrame, df2: pd.DataFrame, col_left: str, col_right: str, tipo_join: str) -> pd.DataFrame:
    """
    Cruza dos tablas horizontalmente (Merge) utilizando claves primarias.
    Soporta cruces donde las columnas clave se llaman de forma distinta en cada tabla.
    """

    if col_left == col_right:
        # Si se llaman igual, uso on
        return pd.merge(df1, df2, on=col_left, how=tipo_join)
    else:
        # Si se llaman distinto, uso left_on y right_on
        return pd.merge(df1, df2, left_on = col_left, right_on=col_right, how = tipo_join)
    

def preparar_descarga_csv(df: pd.DataFrame) -> bytes:
    """
    Serializa un DataFrame a un formato de bytes codificado en UTF-8,
    preparado para ser consumido por el botón de descarga nativo de Streamlit.
    """

    return df.to_csv(index=False, sep=';').encode('utf-8')


def apilar_tablas(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Fusiona dos tablas verticalmente (Concat). Útil para bases de datos
    con la misma estructura pero diferentes registros.
    """

    return pd.concat([df1, df2], ignore_index=True)


#------------------------------------------------------------------------------
# FASE 2
#------------------------------------------------------------------------------
def escanear_dataset(df: pd.DataFrame) -> dict:
    """
    Auditoría profunda de la tabla. 
    Analiza cada columna buscando anomalías matemáticas (varianza cero, alta cardinalidad)
    para recomendar la limpieza idónea al usuario antes de aplicar ML.
    """

    reporte = {
        'total_filas': len(df),
        'duplicados_exactos': int(df.duplicated().sum()),
        'columnas': {}
    }
    
    for col in df.columns:
        nulos = int(df[col].isnull().sum())
        percent_nulos = (nulos / len(df)) * 100
        tipo = str(df[col].dtype)
        unicos = int(df[col].nunique())
        
        alertas = []
        accion_recomendada = "Mantener intacta"
        
        if nulos > 0:
            alertas.append(f"{nulos} nulos ({percent_nulos:.1f}%)")
            accion_recomendada = "Imputar nulos"
            
        if unicos == 1:
            alertas.append("Valor constante (Varianza 0)")
            accion_recomendada = "Borrar columna"
            
        elif unicos == len(df) and tipo in ['int64', 'object', 'string']:
            alertas.append("Posible ID o Clave Primaria")
            accion_recomendada = "Borrar columna"
            
        reporte['columnas'][col] = {
            'tipo': tipo,
            'nulos': nulos,
            'unicos': unicos,
            'alertas': alertas,
            'recomendacion': accion_recomendada
        }
        
    return reporte


def aplicar_limpieza(df: pd.DataFrame, acciones: dict) -> pd.DataFrame:
    """
    Ejecuta las mutaciones de limpieza (imputación, borrado) sobre un único dataset.
    Se utiliza en la ruta No Supervisada donde no existe el concepto de Data Leakage.
    """

    df_limpio = df.copy() # Trabajar sobre una copia
    
    for col, accion in acciones.items():
        if accion == "Ignorar (No hacer nada)":
            continue
            
        elif accion == "Borrar columna entera":
            if col in df_limpio.columns:
                df_limpio = df_limpio.drop(columns=[col])
                
        elif "Rellenar nulos con Media" in accion:
            df_limpio[col] = df_limpio[col].fillna(df_limpio[col].mean())
            
        elif "Rellenar nulos con Mediana" in accion:
            df_limpio[col] = df_limpio[col].fillna(df_limpio[col].median())
            
        elif "Rellenar nulos con 0" in accion:
            df_limpio[col] = df_limpio[col].fillna(0)
            
        elif "Rellenar nulos con 'Desconocido'" in accion:
            df_limpio[col] = df_limpio[col].fillna("Desconocido")
            
        elif "Rellenar nulos con Moda" in accion:
            df_limpio[col] = df_limpio[col].fillna(df_limpio[col].mode()[0])
            
        elif "Codificar a números" in accion:
            df_limpio[col] = pd.factorize(df_limpio[col])[0]
            
    return df_limpio

def aplicar_limpieza_dual(df_train: pd.DataFrame, df_test: pd.DataFrame, acciones: dict) -> tuple:
    """
    Limpieza anti Data Leakage para la ruta Supervisada.
    Calcula las estadísticas (medias, modas) ESTRICTAMENTE sobre el conjunto de Train,
    y aplica esa misma transformación matemática al Test para no contaminar el experimento.
    """

    train_limpio = df_train.copy()
    test_limpio = df_test.copy()
    
    for col, accion in acciones.items():
        if accion == "Ignorar (No hacer nada)":
            continue
            
        elif accion == "Borrar columna entera":
            if col in train_limpio.columns:
                train_limpio = train_limpio.drop(columns=[col])
            if col in test_limpio.columns:
                test_limpio = test_limpio.drop(columns=[col])
                
        elif "Rellenar nulos con Media" in accion:
            valor_imputacion = train_limpio[col].mean()
            train_limpio[col] = train_limpio[col].fillna(valor_imputacion)
            test_limpio[col] = test_limpio[col].fillna(valor_imputacion)
            
        elif "Rellenar nulos con Mediana" in accion:
            valor_imputacion = train_limpio[col].median()
            train_limpio[col] = train_limpio[col].fillna(valor_imputacion)
            test_limpio[col] = test_limpio[col].fillna(valor_imputacion)
            
        elif "Rellenar nulos con Moda" in accion:
            valor_imputacion = train_limpio[col].mode()[0]
            train_limpio[col] = train_limpio[col].fillna(valor_imputacion)
            test_limpio[col] = test_limpio[col].fillna(valor_imputacion)
            
        elif "Rellenar nulos con 0" in accion:
            train_limpio[col] = train_limpio[col].fillna(0)
            test_limpio[col] = test_limpio[col].fillna(0)
            
        elif "Rellenar nulos con 'Desconocido'" in accion:
            train_limpio[col] = train_limpio[col].fillna("Desconocido")
            test_limpio[col] = test_limpio[col].fillna("Desconocido")
            
        elif "Codificar a números" in accion:
            categorias = train_limpio[col].dropna().unique()
            mapeo = {val: i for i, val in enumerate(categorias)}
    
            train_limpio[col] = train_limpio[col].map(mapeo)
            test_limpio[col] = test_limpio[col].map(mapeo)
            test_limpio[col] = test_limpio[col].fillna(-1)
            
    return train_limpio, test_limpio

def detectar_outliers_iqr(df: pd.DataFrame, col: str) -> list:
    """
    Escáner estadístico de Rango Intercuartílico (IQR).
    Encuentra valores atípicos que están demasiado lejos del núcleo principal
    de la distribución matemática de la columna.
    """

    if df[col].dtype not in ['int64', 'float64']:
        return []
    
    # Metodo estandar para detectar valores anómalos
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    
    # Buscamos filas que se salgan de los límites
    indices = df[(df[col] < limite_inferior) | (df[col] > limite_superior)].index
    return indices.tolist()

def crear_nueva_columna(df: pd.DataFrame, col1: str, col2: str, operacion: str, nombre_nuevo: str) -> pd.DataFrame:
    """
    Motor de ingeniería de características. 
    Permite combinar dos columnas numéricas usando matemáticas básicas
    para crear variables que el modelo pueda aprovechar mejor.
    """
    
    try:
        if operacion == "Suma (+)":
            df[nombre_nuevo] = df[col1] + df[col2]
        elif operacion == "Resta (-)":
            df[nombre_nuevo] = df[col1] - df[col2]
        elif operacion == "Multiplicación (*)":
            df[nombre_nuevo] = df[col1] * df[col2]
        elif operacion == "División (/)":
            # Evitamos división por cero
            df[nombre_nuevo] = df[col1] / df[col2].replace(0, 0.001)
        return df
    except Exception as e:
        print(f"Error al crear variable: {e}")
        return df

