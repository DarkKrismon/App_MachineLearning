from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
import pandas as pd
from typing import Dict, Any

"""
Módulo de Servicio para Machine Learning Supervisado.
Contiene el motor de entrenamiento de la Fase 3 (AutoML y GridSearchCV), 
incluyendo codificación, escalado, cross-validation estricto en Train y 
generación de reportes de rendimiento y matrices de confusión.
"""

def entrenar_automl(df_train: pd.DataFrame, df_test: pd.DataFrame, columna_target: str, hiperparametros: dict) -> Dict[str, Any]:
    """
    Entrena cinco algoritmos en paralelo utilizando Cross-Validation sobre el conjunto de Train.
    Selecciona automáticamente el ganador, lo entrena de forma final y evalúa su rendimiento 
    contra el conjunto de Test para evitar métricas sesgadas por Data Leakage.
    
    Parámetros:
    ----------
    df_train : pd.DataFrame
        Los datos de entrenamiento limpios.
    df_test : pd.DataFrame
        Los datos de examen (invisibles para el modelo durante el entrenamiento).
    columna_target : str
        La variable que se intenta predecir.
    hiperparametros : dict
        Los ajustes seleccionados por el usuario desde la UI.
        
    Retorna:
    -------
    dict
        Un diccionario empaquetando el modelo ganador, los traductores, el escalador y las métricas.
    """

    X_train = df_train.drop(columns=[columna_target]).copy()
    y_train = df_train[columna_target].copy()
    
    X_test = df_test.drop(columns=[columna_target]).copy()
    y_test = df_test[columna_target].copy()
    
    diccionario_traductores = {}

    for col in X_train.columns:
        if X_train[col].dtype in ['object', 'string', 'category', 'bool']:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            diccionario_traductores[col] = le
            

            clases_conocidas = set(le.classes_)
            X_test[col] = X_test[col].astype(str).map(lambda x: le.transform([x])[0] if x in clases_conocidas else -1).fillna(-1)
            

    clases_reales = []
    es_clasificacion = y_train.nunique() < 15 or str(y_train.dtype) in ['object', 'string', 'category']
    tipo_problema = "Clasificación" if es_clasificacion else "Regresión"
    traductor_target = None 
    
    if es_clasificacion:
        if y_train.dtype in ['object', 'string', 'category']:
            le_y = LabelEncoder()
            y_train = le_y.fit_transform(y_train.astype(str))
            
            # Mapeo seguro para el Test
            clases_y = set(le_y.classes_)
            y_test = y_test.astype(str).map(lambda x: le_y.transform([x])[0] if x in clases_y else -1).fillna(-1)
            
            clases_reales = le_y.classes_.tolist()
            traductor_target = le_y 
        else:
            clases_reales = sorted(y_train.unique().tolist())
            clases_reales = [str(c) for c in clases_reales]
    

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    #Extraemos hiperparámetros
    rf_n = hiperparametros['rf_arboles']
    rf_depth = hiperparametros['rf_profundidad'] if hiperparametros['rf_profundidad'] > 0 else None
    rf_min_split = hiperparametros['rf_min_split']
    rf_min_leaf = hiperparametros['rf_min_leaf']
    
    gb_n = hiperparametros['gb_arboles']
    gb_lr = hiperparametros['gb_learning_rate']
    gb_depth = hiperparametros['gb_profundidad']
    gb_subsample = hiperparametros['gb_subsample']

    knn_k = hiperparametros.get('knn_neighbors', 5)
    svm_c = hiperparametros.get('svm_c', 1.0)

    # Inyectamos los parámetros en los algoritmos
    if es_clasificacion:
        modelos = {
            "Regresión Logística": LogisticRegression(max_iter=1000, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=rf_n, max_depth=rf_depth, min_samples_split=rf_min_split, min_samples_leaf=rf_min_leaf, random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(n_estimators=gb_n, learning_rate=gb_lr, max_depth=gb_depth, subsample=gb_subsample, random_state=42),
            "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=knn_k),
            "Support Vector Machine": SVC(C=svm_c, probability=True, random_state=42)
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    else:
        modelos = {
            "Regresión Lineal": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=rf_n, max_depth=rf_depth, min_samples_split=rf_min_split, min_samples_leaf=rf_min_leaf, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=gb_n, learning_rate=gb_lr, max_depth=gb_depth, subsample=gb_subsample, random_state=42),
            "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=knn_k),
            "Support Vector Machine": SVR(C=svm_c)
        }
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        
    resultados = {}
    mejor_score_cv = -float('inf')
    mejor_nombre = ""
    mejor_modelo = None
    
    for nombre, modelo in modelos.items():
        # Validamos estrictamente sobre TRAIN
        if es_clasificacion:
            scores_cv = cross_val_score(modelo, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        else:
            scores_cv = cross_val_score(modelo, X_train_scaled, y_train, cv=cv, scoring='r2')
            
        score_realista = scores_cv.mean()
        resultados[nombre] = score_realista
        
        # Entrenamiento final de este modelo en todo el Train
        modelo.fit(X_train_scaled, y_train)
        
        if score_realista > mejor_score_cv:
            mejor_score_cv = score_realista
            mejor_nombre = nombre
            mejor_modelo = modelo


    reporte_detallado = ""
    reporte_dict = {}
    matriz_conf = None

    if es_clasificacion:
        y_pred_ganador = mejor_modelo.predict(X_test_scaled)
        
        reporte_detallado = classification_report(y_test, y_pred_ganador, target_names=clases_reales, zero_division=0)
        reporte_dict = classification_report(y_test, y_pred_ganador, target_names=clases_reales, zero_division=0, output_dict=True)
        matriz_conf = confusion_matrix(y_test, y_pred_ganador).tolist()

    return {
        "tipo_problema": tipo_problema,
        "resultados_todos": resultados, 
        "ganador_nombre": mejor_nombre,
        "ganador_score_cv": mejor_score_cv, 
        "ganador_modelo": mejor_modelo,
        "reporte_clasificacion": reporte_detallado,
        "reporte_dict": reporte_dict,
        "matriz_confusion": matriz_conf,
        "clases": clases_reales,
        "traductores_X": diccionario_traductores,
        "traductor_y": traductor_target,
        "escalador": scaler
    }


def entrenar_automl_gridsearch(df_train: pd.DataFrame, df_test: pd.DataFrame, columna_target: str) -> Dict[str, Any]:
    """
    Ejecuta una búsqueda en cuadrícula (GridSearchCV) sobre un Random Forest
    para encontrar matemáticamente la mejor combinación posible de hiperparámetros.
    Es más lento pero potencialmente más preciso que los ajustes manuales.
    
    Parámetros:
    ----------
    df_train, df_test : pd.DataFrame
        Los conjuntos de datos.
    columna_target : str
        La variable a predecir.
    """
    
    X_train = df_train.drop(columns=[columna_target]).copy()
    y_train = df_train[columna_target].copy()
    
    X_test = df_test.drop(columns=[columna_target]).copy()
    y_test = df_test[columna_target].copy()
    
    diccionario_traductores = {}

    for col in X_train.columns:
        if X_train[col].dtype in ['object', 'string', 'category', 'bool']:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            diccionario_traductores[col] = le
            
            clases_conocidas = set(le.classes_)
            X_test[col] = X_test[col].astype(str).map(lambda x: le.transform([x])[0] if x in clases_conocidas else -1).fillna(-1)
            
    clases_reales = []
    es_clasificacion = y_train.nunique() < 15 or str(y_train.dtype) in ['object', 'string', 'category']
    tipo_problema = "Clasificación" if es_clasificacion else "Regresión"
    traductor_target = None 
    
    if es_clasificacion:
        if y_train.dtype in ['object', 'string', 'category']:
            le_y = LabelEncoder()
            y_train = le_y.fit_transform(y_train.astype(str))
            
            clases_y = set(le_y.classes_)
            y_test = y_test.astype(str).map(lambda x: le_y.transform([x])[0] if x in clases_y else -1).fillna(-1)
            
            clases_reales = le_y.classes_.tolist()
            traductor_target = le_y 
        else:
            clases_reales = sorted(y_train.unique().tolist())
            clases_reales = [str(c) for c in clases_reales]
            
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Cuadrícula de búsqueda
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'min_samples_split': [5, 10]
    }
    
    if es_clasificacion:
        modelo_base = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=modelo_base, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    else:
        modelo_base = RandomForestRegressor(random_state=42)
        grid_search = GridSearchCV(estimator=modelo_base, param_grid=param_grid, cv=5, scoring='r2', n_jobs=-1)

    # Entrenamos solo con X_train_scaled
    grid_search.fit(X_train_scaled, y_train)
    
    mejor_modelo = grid_search.best_estimator_
    mejor_score_cv = grid_search.best_score_
    
    params_ganadores = grid_search.best_params_
    mejor_nombre = "GridSearchCV"
    
    resultados = {mejor_nombre: mejor_score_cv} 
    
    reporte_detallado = ""
    reporte_dict = {}
    matriz_conf = None

    if es_clasificacion:
        y_pred_ganador = mejor_modelo.predict(X_test_scaled)
        reporte_detallado = classification_report(y_test, y_pred_ganador, target_names=clases_reales, zero_division=0)
        reporte_dict = classification_report(y_test, y_pred_ganador, target_names=clases_reales, zero_division=0, output_dict=True)
        matriz_conf = confusion_matrix(y_test, y_pred_ganador).tolist()

    return {
        "tipo_problema": tipo_problema,
        "resultados_todos": resultados, 
        "ganador_nombre": mejor_nombre,
        "ganador_score_cv": mejor_score_cv, 
        "ganador_modelo": mejor_modelo,
        "reporte_clasificacion": reporte_detallado,
        "reporte_dict": reporte_dict,
        "matriz_confusion": matriz_conf,
        "clases": clases_reales,
        "traductores_X": diccionario_traductores,
        "traductor_y": traductor_target,
        "escalador": scaler
    }

#------------------------------------------------------------------------------
# FASE 4
#------------------------------------------------------------------------------

def obtener_importancia_variables(modelo_entrenado: Any, nombres_columnas: list) -> pd.DataFrame:
    """
    Extrae los coeficientes o la importancia de características (feature_importances_)
    del modelo entrenado para entender el peso de cada variable en la decisión final.
    Devuelve un DataFrame listo para ser graficado con Plotly, o None si el modelo no lo soporta.
    """

    # Los modelos basados en árboles
    if hasattr(modelo_entrenado, 'feature_importances_'):
        importancias = modelo_entrenado.feature_importances_

    # Los modelos lineales
    elif hasattr(modelo_entrenado, 'coef_'):
        importancias = modelo_entrenado.coef_[0] if len(modelo_entrenado.coef_.shape) > 1 else modelo_entrenado.coef_
    else:
        return None
 
    df_imp = pd.DataFrame({
        'Variable': nombres_columnas,
        'Importancia': importancias
    }).sort_values(by='Importancia', ascending=True)
    
    df_imp['Importancia'] = df_imp['Importancia'].abs() 
    return df_imp