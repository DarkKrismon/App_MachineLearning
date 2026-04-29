import pandas as pd
import streamlit as st

''' 
Aquí metemos todo lo relacionado con la Fase 1 y 2.
Este archivo solo recibe tablas de datos, 
opera con ellas usando matemáticas y devuelve resultados.
'''

#------------------------------------------------------------------------------
# FASE 1
#------------------------------------------------------------------------------
def leer_dataset_universal(archivo):
    """
    Lee archivos CSV, Excel o Parquet y devuelve un DataFrame.
    Incluye manejo de errores para que la app no colapse.
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
    

def generar_resumen_df(df, nombre_archivo):
    """Extrae datos matemáticos del DataFrame."""
    filas, columnas = df.shape
    nulos = df.isnull().sum().sum()
    columnas_con_nulos = df.columns[df.isnull().any()].tolist()
    
    resumen = f"Archivo: {nombre_archivo}\n"
    resumen += f"- Filas: {filas}, Columnas: {columnas}\n"
    resumen += f"- Total de valores nulos: {nulos}\n"
    if nulos > 0:
        resumen += f"- Columnas con nulos: {', '.join(columnas_con_nulos)}\n"
    return resumen


def obtener_columnas_comunes(df1, df2):
    """Devuelve una lista con las columnas que interseccionan en ambos DataFrames."""
    return list(set(df1.columns).intersection(set(df2.columns)))


def unificar_tablas(df1, df2, col_left, col_right, tipo_join):
    """Realiza el Merge de dos tablas, soportando nombres de columnas distintos."""
    if col_left == col_right:
        # Si se llaman igual, uso on
        return pd.merge(df1, df2, on=col_left, how=tipo_join)
    else:
        # Si se llaman distinto, uso left_on y right_on
        return pd.merge(df1, df2, left_on = col_left, right_on=col_right, how = tipo_join)
    

def preparar_descarga_csv(df):
    """Convierte el DataFrame completo a CSV."""
    return df.to_csv(index=False, sep=';').encode('utf-8')


def apilar_tablas(df1, df2):
    """Apila dos tablas verticalmente (Concat), alineando las columnas comunes."""
    return pd.concat([df1, df2], ignore_index=True)


#------------------------------------------------------------------------------
# FASE 2
#------------------------------------------------------------------------------
def escanear_dataset(df):
    """
    Escanea el DataFrame.
    Alerta de nulos, valores constantes y posibles IDs.
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


def aplicar_limpieza(df, acciones):
    """
    Recibe el DataFrame y un diccionario con las acciones decididas por el usuario.
    Aplica las transformaciones y devuelve el DataFrame limpio.
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

def aplicar_limpieza_dual(df_train, df_test, acciones):
    """
    Recibe Train y Test. Calcula las estadísticas (Media, Mediana, Moda) 
    sobre el Train solamente, y usa esos valores para rellenar los nulos 
    Tanto en el Train como en el Test.
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

def detectar_outliers_iqr(df, col):
    """Detecta anomalías usando el método IQR y devuelve los índices."""
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

def crear_nueva_columna(df, col1, col2, operacion, nombre_nuevo):
    """Crea una variable calculada a partir de dos columnas existentes."""
    try:
        if operacion == "Suma (+)":
            df[nombre_nuevo] = df[col1] + df[col2]
        elif operacion == "Resta (-)":
            df[nombre_nuevo] = df[col1] - df[col2]
        elif operacion == "Multiplicación (*)":
            df[nombre_nuevo] = df[col1] * df[col2]
        elif operacion == "División (/)":
            # Evitamos división por cero con un pequeño truco
            df[nombre_nuevo] = df[col1] / df[col2].replace(0, 0.001)
        return df
    except Exception as e:
        print(f"Error al crear variable: {e}")
        return df

