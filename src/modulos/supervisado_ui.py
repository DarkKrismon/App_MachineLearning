import streamlit as st
import pandas as pd
import pickle
import ia_client
from service import supervisado_service
from service import data_service


''' 
Encargado visual de la fase 3, 4 y 5 de modelo supervisado 
llamando a la función "renderizar_fase_supervisada""
'''

def renderizar_fase_supervisada():
    fase = st.session_state.get('fase_actual', 3)

    df_train = st.session_state.get('df_train')
    df_test = st.session_state.get('df_test')
    columna_target = st.session_state.get('target_elegido')
    
    if df_train is None or df_test is None or columna_target is None:
        st.error("🚨 No se han encontrado los datos particionados. Por favor, vuelve a la Fase 1 y vuelve a elegir tu Target.")
        return

    # ==========================================
    #                 FASE 3
    # ==========================================
    if fase == 3:
        st.title("⚙️ Fase 3: Ruta Supervisada (AutoML)")
        
        st.markdown("### 1. Objetivo y División de Datos")
        st.success(f"🎯 **Variable a predecir (Target):** `{columna_target}`")
        
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.info(f"📚 **Datos de Entrenamiento (Train):** {len(df_train)} pasajeros.\n\nAquí extraeremos los patrones.")
        with col_info2:
            st.warning(f"📝 **Datos de Examen (Test):** {len(df_test)} pasajeros.\n\nInvisibles para el modelo hasta el examen final.")

        st.divider()
        
        st.markdown("### 2. Configuración de Hiperparámetros")
        st.info("Hemos preseleccionado los valores estándar. Despliega el menú para personalizar a fondo la arquitectura de los algoritmos.")
        
        params_usuario = {
            'rf_arboles': 100,
            'rf_profundidad': 0,
            'rf_min_split': 2,
            'rf_min_leaf': 1,
            'gb_arboles': 100,
            'gb_learning_rate': 0.1,
            'gb_profundidad': 3,
            'gb_subsample': 1.0,
            'knn_neighbors': 5,
            'svm_c': 1.0
        }
        

        with st.expander("🛠️ Ajustes Avanzados de los Algoritmos"):
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 🌲 Random Forest", help="Construye cientos de Árboles de Decisión al azar y los pone a votar, " \
                "casi imposible que memorice los datos (overfitting) porque el ruido se diluye entre las votaciones.")

                params_usuario['rf_arboles'] = st.slider(
                    "Árboles (n_estimators)", 10, 500, 100, 10,
                    help="Recomendación: 100. Más árboles = más precisión pero mayor tiempo de carga."
                )
                params_usuario['rf_profundidad'] = st.slider(
                    "Profundidad Máxima (max_depth)", 0, 50, 0, 1,
                    help="Recomendación: 0 (Sin límite). Limítalo si tu modelo memoriza (Overfitting)."
                )
                params_usuario['rf_min_split'] = st.slider(
                    "Mínimo para dividir (min_samples_split)", 2, 20, 2, 1,
                    help="Datos mínimos necesarios en una rama antes de que pueda volver a bifurcarse."
                )
                params_usuario['rf_min_leaf'] = st.slider(
                    "Mínimo en hoja (min_samples_leaf)", 1, 20, 1, 1,
                    help="Datos mínimos que deben quedar en el resultado final de la rama."
                )
                
            with col2:
                st.markdown("#### 🚀 Gradient Boosting", help="Usa árboles, pero en lugar de ponerlos a votar todos a la vez, " \
                "los construye en serie. El primer árbol hace una predicción; el segundo árbol se enfoca única y exclusivamente en corregir los errores del primer árbol; " \
                "el tercero corrige al segundo... y así sucesivamente.")

                params_usuario['gb_arboles'] = st.slider(
                    "Etapas (n_estimators)", 10, 500, 100, 10,
                    help="Recomendación: 100. Cuántos mini-árboles construirá para corregir sus propios errores."
                )
                params_usuario['gb_learning_rate'] = st.select_slider(
                    "Tasa de Aprendizaje (learning_rate)", options=[0.01, 0.05, 0.1, 0.2, 0.5], value=0.1,
                    help="Recomendación: 0.1. Tasa baja requiere más árboles pero encuentra patrones más finos."
                )
                params_usuario['gb_profundidad'] = st.slider(
                    "Profundidad de mini-árboles (max_depth)", 1, 10, 3, 1,
                    help="Recomendación: 3. Mantenerlo bajo evita que se memorice el dataset."
                )
                params_usuario['gb_subsample'] = st.select_slider(
                    "Porcentaje de datos por árbol (subsample)", options=[0.5, 0.7, 0.8, 0.9, 1.0], value=1.0,
                    help="Recomendación: 1.0. Bajarlo a 0.8 introduce aleatoriedad y reduce el Overfitting."
                )

            col3, col4 = st.columns(2)
            with col3:
                st.markdown("#### 📍 K-Nearest Neighbors (KNN)", help="Es excelente para encontrar anomalías y patrones basados en grupos locales muy específicos que las ecuaciones matemáticas pasan por alto.")
                params_usuario['knn_neighbors'] = st.slider(
                    "Número de Vecinos (n_neighbors)", 1, 30, 5, 1,
                    help="Recomendación: 5. Cuántos puntos cercanos mira para tomar una decisión. Valores muy bajos causan que memorice (Overfitting)."
                )
            with col4:
                st.markdown("#### 🧲 Support Vector Machine (SVM)", help="Es brutalmente eficaz cuando los grupos están muy entremezclados y no se pueden separar con una línea recta, gracias a 'Truco de Kernel'")
                params_usuario['svm_c'] = st.select_slider(
                    "Regularización (C)", options=[0.01, 0.1, 1.0, 10.0, 100.0], value=1.0,
                    help="Recomendación: 1.0. Un valor alto fuerza al modelo a clasificar todo perfecto, un valor bajo permite más margen de error general."
                )

        st.divider()


        # --- BOTÓN DE ENTRENAMIENTO ---
        st.markdown("### 3. El Entrenamiento de Modelos")
        st.info("Los entrenamientos se quedarán guardados en un historial de entrenamiento, podrás elegir entre todos los que hagas para la última fase. " \
        "Puedes modificar los hiperparámetros entre entrenamientos.")

        st.write("Se podrá a entrenar 5 modelos en la versión de Entrenamiento Rápido donde competirán por un mayor resultado, " \
        "el que obtenga mayor resultado saldrá como principal. Por otro lado, tenemos GridSearch " \
        "que ajusta automáticamente los hiperparámetros de nuestro modelo RandomForest.")
        
        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            btn_rapido = st.button("Entrenamiento Rápido (5 Modelos)", use_container_width=True,  help="Entrena: Regresión Logística, Random Forest, Gradient Boosting, KNN y SVM")
        with col_btn2:
            btn_grid = st.button("GridSearchCV", use_container_width=True, help="Busca automáticamente la mejor combinación matemática. Tarda un poco más.")

        if btn_rapido or btn_grid:
            mensaje_carga = "Ejecutando GridSearch..." if btn_grid else "Entrenando los 5 modelos..."
            
            with st.spinner(mensaje_carga):
                try:
                    # Elegimos la ruta según el botón pulsado
                    if btn_rapido:
                        resultados = supervisado_service.entrenar_automl(df_train, df_test, columna_target, params_usuario)
                    else:
                        resultados = supervisado_service.entrenar_automl_gridsearch(df_train, df_test, columna_target)


                    if 'historial_entrenamientos' not in st.session_state:
                        st.session_state['historial_entrenamientos'] = []

                    porcentaje_test = len(df_test) / (len(df_train) + len(df_test)) * 100

                    # Guardamos en el Historial
                    registro = {
                        "ID": len(st.session_state['historial_entrenamientos']) + 1,
                        "Algoritmo": resultados['ganador_nombre'],
                        "Target": columna_target,
                        "Precisión": f"{resultados['ganador_score_cv'] * 100:.2f}%",
                        "Test Size": f"{porcentaje_test:.0f}%",
                        "resultados_completos": resultados, 
                        "columnas_train": df_train.drop(columns=[columna_target]).columns.tolist()
                    }
                    st.session_state['historial_entrenamientos'].append(registro)
                    st.success("¡Entrenamiento Finalizado!")

                except Exception as e:
                    st.error(f"Error durante el entrenamiento: {e}")

        if st.session_state.get('historial_entrenamientos'):
            st.divider()
            st.markdown("### 📜 Historial de Modelos Entrenados")
            df_historial = pd.DataFrame(st.session_state['historial_entrenamientos'])
            
            columnas_visibles = ["ID", "Algoritmo", "Target", "Precisión", "Test Size"]
            st.dataframe(df_historial[columnas_visibles].set_index("ID"), use_container_width=True)
            
            if st.button("🗑️ Limpiar Historial al completo"):
                st.session_state['historial_entrenamientos'] = []
                st.rerun()

            st.divider()
            
            st.markdown("### 🏆 Selección del Modelo para la fase 4, Análisis de Datos")
            st.info("Elige que intento del historial quieres llevar a producción.")
            
            opciones_exp = [f"Exp {r['ID']} - {r['Algoritmo']} ({r['Precisión']})" for r in st.session_state['historial_entrenamientos']]
            
            experimento_elegido = st.selectbox("Selecciona el intento:", opciones_exp, index=len(opciones_exp)-1)
            
            if st.button("📈 Fase 4: Interpretación de Datos", type="primary"):
                idx_elegido = opciones_exp.index(experimento_elegido)
                registro_elegido = st.session_state['historial_entrenamientos'][idx_elegido]
                
                st.session_state['ml_results'] = registro_elegido['resultados_completos']
                st.session_state['columnas_entrenamiento'] = registro_elegido['columnas_train']
                
                st.session_state['fase_actual'] = 4
                st.rerun()
        

    # ==========================================
    #                 FASE 4
    # ==========================================
    elif fase == 4:

        st.title("⚖️ Fase 4: Dashboard de Interpretación")
        st.write("Un modelo no sirve de nada si no se puede explicar. Analiza los resultados desde dos perspectivas.")
        
        resultados = st.session_state['ml_results']
        columnas_X = st.session_state['columnas_entrenamiento']
        modelo_ganador = resultados['ganador_modelo']

        tab_tec, tab_neg = st.tabs([
            "⚙️ Panel Técnico", 
            "👔 Panel Ejecutivo"
        ])

        # --- PESTAÑA 1: TÉCNICA (Dashboard Visual) ---
        with tab_tec:
            # 1. KPIs Principales en la cabecera
            col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
            col_kpi1.metric(label="🏆 Algoritmo Ganador", value=resultados['ganador_nombre'])
            col_kpi2.metric(label="🎯 Fiabilidad Real (CV)", value=f"{resultados['ganador_score_cv']*100:.2f}%")
            col_kpi3.metric(label="🧩 Tipo de Problema", value=resultados['tipo_problema'])

            st.write("")
            
            modelo_bytes = pickle.dumps(modelo_ganador)
            st.download_button(
                label="💾 Descargar Modelo para Producción (.pkl)", 
                data=modelo_bytes, 
                file_name=f"modelo_{resultados['ganador_nombre'].replace(' ', '_')}.pkl", 
                mime="application/octet-stream",
                type="primary"
            )

            st.divider()

            # 2. Gráficos interactivos (Fila de 2 columnas)
            col_g1, col_g2 = st.columns(2)
            
            with col_g1:
                st.subheader("📍 Matriz de Confusión")
                st.info("💡 Muestra los Falsos Positivos y Falsos Negativos. Idealmente, la diagonal principal debe tener los números más altos.")
                if resultados['tipo_problema'] == "Clasificación" and resultados.get('matriz_confusion'):
                    import plotly.express as px
                    fig_cm = px.imshow(
                        resultados['matriz_confusion'], 
                        text_auto=True, 
                        labels=dict(x="Predicción del Modelo", y="Realidad (Target)", color="Casos"),
                        x=resultados['clases'], 
                        y=resultados['clases'], 
                        color_continuous_scale="Blues",
                        aspect="auto"
                    )
                    st.plotly_chart(fig_cm, use_container_width=True)
                else:
                    st.warning("La matriz de confusión solo está disponible para problemas de Clasificación.")

            with col_g2:
                st.subheader("⚖️ Importancia de Variables")
                st.info("💡 Muestra qué columnas de tu CSV tuvieron más peso matemático para tomar la decisión final.")
                df_importancia = supervisado_service.obtener_importancia_variables(modelo_ganador, columnas_X)
                if df_importancia is not None:
                    fig_imp = px.bar(
                        df_importancia, x='Importancia', y='Variable', orientation='h',
                        color='Importancia', color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig_imp, use_container_width=True)
                else:
                    st.warning("Este algoritmo no permite extraer la importancia directa de las variables.")

            st.divider()

            # 3. La Cuadrícula Matemática (Dataframe)
            st.subheader("📊 Reporte Matemático Detallado")
            if resultados['tipo_problema'] == "Clasificación" and resultados.get('reporte_dict'):
                df_reporte = pd.DataFrame(resultados['reporte_dict']).transpose()
                
                st.dataframe(
                    df_reporte.style.format("{:.3f}").background_gradient(cmap='Greens', subset=['f1-score', 'precision', 'recall']),
                    use_container_width=True
                )
            else:
                st.write("Métricas de regresión en desarrollo...")


        # --- PESTAÑA 2: EJECUTIVA (IA de Negocio) ---
        with tab_neg:
            st.header("Reporte de Datos (Traducción a Negocio)")
            st.info("💡 La IA leerá los gráficos técnicos y redactará un resumen con recomendaciones estratégicas para tus directivos.")
            
            if st.button("🧠 Generar Reporte Estratégico", type="primary"):
                with st.spinner("La IA está redactando el informe ejecutivo..."):
                    reporte_ia = ia_client.generar_reporte_ejecutivo(resultados, df_test, columna_target)
                    st.success("Reporte generado con éxito.")
                    st.markdown(reporte_ia)

        st.divider()
        st.markdown("### 🚀 Siguiente Paso: Producción y Predicción")
        st.info("¿Estás satisfecho con las métricas del modelo? Es hora de ponerlo a trabajar. Avanza a la Fase 5 para subir datos nuevos (clientes, pacientes, casas...) y usar esta IA para predecir su futuro.")
        
        if st.button("🔮 Ir a Fase 5 (Simulador y Predicciones)", type="primary", use_container_width=True):
            st.session_state['fase_actual'] = 5
            st.rerun()



    # ==============================================
    #                 FASE 5: PRODUCCIÓN
    # ==============================================
    elif fase == 5:
        st.title("🔮 Fase 5: Producción y Simulación")
        st.markdown("Pon tu modelo a trabajar. Puedes hacer predicciones masivas subiendo un CSV, o simular un caso único en tiempo real.")

        resultados = st.session_state.get('ml_results')
        df_train = st.session_state.get('df_train')
        columna_target = st.session_state.get('target_elegido')

        if resultados is None or df_train is None:
            st.error("🚨 El modelo no está en memoria. Vuelve a la Fase 3 y entrénalo.")

            if st.button("⬅️ Volver a los Resultados (Fase 3)", use_container_width=True):
                st.session_state['fase_actual'] = 3
                st.rerun()
        else:
            columnas_entrenamiento = [col for col in df_train.columns if col != columna_target]
            tab_lotes, tab_vivo = st.tabs(["📁 Predicción Masiva (Subir Archivos)", "🎛️ Simulador en Vivo"])

            # ---------------------------------------------------------
            # PESTAÑA 1: PREDICCIÓN POR LOTES
            # ---------------------------------------------------------
            with tab_lotes:
                st.markdown("### 📥 Predicciones en Bloque")
                archivo_nuevo = st.file_uploader("Sube los nuevos clientes/datos:", type=["csv", "xlsx", "parquet"], key="uploader_simulador")
                
                if archivo_nuevo:

                    df_nuevo = data_service.leer_dataset_universal(archivo_nuevo)
                    
                    if df_nuevo is not None:

                        if columna_target in df_nuevo.columns:
                            df_nuevo = df_nuevo.drop(columns=[columna_target])
                            st.toast("🛡️ Se ha eliminado la columna objetivo del archivo para evitar errores.", icon="✅")

                        st.write("**Vista previa de los datos limpios:**")
                        st.dataframe(df_nuevo.head(5))

                        st.markdown("#### ⚙️ Configuración Opcional")
                        columna_id = st.selectbox("¿Hay alguna columna de Identidad (ID/Nombre)?", ["Ninguna"] + df_nuevo.columns.tolist(), 
                                                  help="Seleccione la columna que le permita diferenciar las diferentes líneas.")
                        
                        if st.button("🚀 Ejecutar Predicción Masiva", type="primary"):
                            try:
                                df_procesar = df_nuevo.copy()
                                
                                caja_fuerte_ids = None
                                if columna_id != "Ninguna":
                                    caja_fuerte_ids = df_procesar[columna_id].copy()
                                
                                # Filtramos solo las columnas que el modelo conoce
                                df_limpio = df_procesar.reindex(columns=columnas_entrenamiento)
                                
                                # Traducción y Nulos
                                traductores = resultados['traductores_X']
                                for col in df_limpio.columns:
                                    if col in traductores:
                                        clase_por_defecto = traductores[col].classes_[0]
                                        df_limpio[col] = df_limpio[col].fillna(clase_por_defecto).astype(str)
                                        df_limpio[col] = traductores[col].transform(df_limpio[col])
                                    else:
                                        df_limpio[col] = pd.to_numeric(df_limpio[col], errors='coerce').fillna(0)

                                # Escalado Matemático
                                escalador = resultados.get('escalador')
                                if escalador:
                                    X_final = escalador.transform(df_limpio)
                                else:
                                    X_final = df_limpio

                                # Predicción
                                modelo = resultados['ganador_modelo']
                                predicciones_crudas = modelo.predict(X_final)
                                
                                traductor_y = resultados.get('traductor_y')
                                if traductor_y is not None:
                                    predicciones_legibles = traductor_y.inverse_transform(predicciones_crudas)
                                else:
                                    predicciones_legibles = predicciones_crudas
                                    
                                # Empaquetado
                                df_final = pd.DataFrame()
                                if caja_fuerte_ids is not None:
                                    df_final[columna_id] = caja_fuerte_ids
                                df_final['Predicción_IA'] = predicciones_legibles
                                
                                st.success("✅ ¡Predicciones masivas completadas con éxito!")
                                st.dataframe(df_final, use_container_width=True)
                                
                                # Descarga
                                csv_final = df_final.to_csv(index=False).encode('utf-8')
                                st.download_button(label="💾 Descargar Resultados (.csv)", data=csv_final, file_name="predicciones_lote.csv", mime="text/csv")

                            except Exception as e:
                                st.error(f"⚠️ Error procesando los datos. Asegúrate de que las columnas coinciden. Detalle: {e}")

            # ---------------------------------------------------------
            # PESTAÑA 2: SIMULADOR EN VIVO
            # ---------------------------------------------------------
            with tab_vivo:
                st.markdown("### 🎛️ Simulador")
                st.info("Ajusta los valores para predecir un único caso en tiempo real. Los límites se generan dinámicamente según lo que aprendió la IA.")
                
                with st.form("form_simulador"):
                    columnas_ui = st.columns(3)
                    input_usuario = {}
                    
                    for idx, col in enumerate(columnas_entrenamiento):
                        with columnas_ui[idx % 3]:
                            if df_train[col].dtype in ['int64', 'float64']:
                                v_min = float(df_train[col].min())
                                v_max = float(df_train[col].max())
                                v_mean = float(df_train[col].mean())

                                # Evitar error si el min y max son iguales
                                if v_min == v_max:
                                    input_usuario[col] = st.number_input(f"{col}", value=v_min)
                                else:
                                    input_usuario[col] = st.slider(f"{col}", min_value=v_min, max_value=v_max, value=v_mean)

                            else:
                                opciones = df_train[col].dropna().unique().tolist()
                                input_usuario[col] = st.selectbox(f"{col}", options=opciones)
                                
                    submit_simulacion = st.form_submit_button("🔮 Predecir este caso", type="primary", use_container_width=True)
                    
                if submit_simulacion:
                    try:
                        # Convertimos el dict del usuario a DataFrame de 1 fila
                        df_sim = pd.DataFrame([input_usuario])
                        
                        # Aplicamos traductores
                        traductores = resultados['traductores_X']
                        for col in df_sim.columns:
                            if col in traductores:
                                df_sim[col] = traductores[col].transform(df_sim[col].astype(str))
                                
                        # Aplicamos escalador
                        escalador = resultados.get('escalador')
                        X_sim = escalador.transform(df_sim) if escalador else df_sim
                        
                        # Predecimos
                        modelo = resultados['ganador_modelo']
                        pred_cruda = modelo.predict(X_sim)
                        
                        traductor_y = resultados.get('traductor_y')
                        pred_legible = traductor_y.inverse_transform(pred_cruda)[0] if traductor_y else pred_cruda[0]
                        
                        st.divider()
                        st.markdown(f'<div class="resultado-prediccion">Resultado Predicho: {pred_legible}</div>', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error en la simulación: {e}")

        # --- NAVEGACIÓN ---
        st.divider()
        st.markdown("### 🧭 Navegación")
        col_back, col_reset = st.columns(2)
        
        with col_back:
            if st.button("⬅️ Volver a los Resultados (Fase 4)", use_container_width=True):
                st.session_state['fase_actual'] = 4
                st.rerun()
                
        with col_reset:
            if st.button("🔄 Empezar un Proyecto Nuevo (Ir a Fase 1)", type="secondary", use_container_width=True):
                for key in list(st.session_state.keys()):
                    if key != 'fase_actual':
                        del st.session_state[key]
                st.session_state['fase_actual'] = 1
                st.rerun()