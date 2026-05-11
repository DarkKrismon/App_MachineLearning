import streamlit as st
import pandas as pd
import plotly.express as px
from service import no_supervisado_service
import ia_client

def renderizar_fase_no_supervisada():
    fase = st.session_state.get('fase_actual', 3)
    
    # Asegurarnos de que tenemos datos limpios (se asume que df_entero ya pasó por limpieza)
    df_actual = st.session_state.get('df_entero')
    
    if df_actual is None:
        st.error("🚨 No se ha encontrado el dataset. Vuelve a la Fase 1.")
        return

    # ==========================================
    #                 FASE 3
    # ==========================================
    if fase == 3:
        st.title("⚙️ Fase 3: Ruta No Supervisada (Clustering)")
        st.markdown("### 1. Diagnóstico Algorítmico")
        st.info("Antes de que elijas a ciegas, la IA ha escaneado tus datos buscando la agrupación matemática perfecta usando el *Silhouette Score*.")

        # Calculamos los datos escalados y escaneamos la mejor K en segundo plano
        if 'datos_escalados' not in st.session_state:
            with st.spinner("Escaneando el tejido matemático de tus datos..."):
                datos_escalados, scaler, traductores = no_supervisado_service.preprocesar_datos_clustering(df_actual)
                analisis = no_supervisado_service.escanear_mejores_k(datos_escalados)
                
                # Guardamos para no recalcular
                st.session_state['datos_escalados'] = datos_escalados
                st.session_state['analisis_k'] = analisis
                st.session_state['ns_scaler'] = scaler
                st.session_state['ns_traductores'] = traductores

        analisis = st.session_state['analisis_k']
        mejor_k = analisis['mejor_k_recomendado']

        # Mostramos los resultados del escáner visualmente
        col_res1, col_res2 = st.columns([1, 2])
        with col_res1:
            st.metric("Grupos (Clusters) Recomendados", value=mejor_k, delta="Matemáticamente Óptimo", delta_color="normal")
            st.write("La IA sugiere crear esta cantidad de tribus porque es donde los datos están más cohesionados sin mezclarse entre sí.")
            
        with col_res2:
            # Gráfico de codo/silueta para que los técnicos vean por qué
            df_metricas = pd.DataFrame({
                'Número de Clusters (K)': analisis['k_values'],
                'Calidad (Silhouette)': analisis['silueta']
            })
            fig_scan = px.line(df_metricas, x='Número de Clusters (K)', y='Calidad (Silhouette)', markers=True, 
                               title="Curva de Calidad de Agrupación")
            # Marcamos el punto óptimo
            fig_scan.add_scatter(x=[mejor_k], y=[df_metricas.loc[df_metricas['Número de Clusters (K)'] == mejor_k, 'Calidad (Silhouette)'].values[0]], 
                                 mode='markers', marker=dict(color='red', size=15), name="Recomendado")
            st.plotly_chart(fig_scan, use_container_width=True)

        st.divider()

        st.markdown("### 2. Configuración Final y Entrenamiento")
        st.write("Puedes aceptar la sugerencia de la IA o forzar un número distinto si el negocio te lo exige (ej: necesitas 3 tarifas para clientes).")
        
        num_clusters = st.slider("Selecciona el número de Tribus (Clusters) a generar:", 
                                 min_value=2, max_value=10, value=mejor_k)
        
        # --- NUEVO SELECTOR DE ALGORITMO ---
        algoritmo_elegido = st.radio("Selecciona el motor matemático:", 
                                     ["K-Means", "Clustering Jerárquico"], 
                                     horizontal=True,
                                     help="K-Means es rapidísimo y busca centros geométricos. Jerárquico va fusionando vecinos desde abajo hacia arriba (mejor para formas complejas).")

        if st.button("🚀 Iniciar Entrenamiento", type="primary", use_container_width=True):
            with st.spinner(f"Entrenando {algoritmo_elegido} con {num_clusters} clusters..."):
                try:
                    # Fíjate que llamamos a la nueva función y le pasamos el algoritmo
                    resultados = no_supervisado_service.entrenar_clustering_3d(df_actual, num_clusters, algoritmo_elegido)
                    st.session_state['resultados_ns'] = resultados
                    st.session_state['fase_actual'] = 4
                    st.success("¡Entrenamiento completado!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Error durante el entrenamiento: {e}")

    # ==========================================
    #                 FASE 4
    # ==========================================
    elif fase == 4:
        st.title("🌌 Fase 4: Observatorio 3D y Perfiles")
        
        resultados = st.session_state.get('resultados_ns')
        if not resultados:
            st.error("No hay resultados de entrenamiento. Vuelve a la Fase 3.")
            return

        df_clusters = resultados['df_con_clusters']
        calidad = resultados['calidad_silueta']
        var_pca = resultados['varianza_pca']
        nombre_algoritmo = resultados['nombre_algoritmo']

        # 1. KPIs
        c1, c2, c3 = st.columns(3)
        c1.metric("Modelo Usado", nombre_algoritmo)
        c2.metric("Calidad de Agrupación", f"{calidad:.2f}", help="De -1 a 1. Valores cercanos a 1 significan grupos muy bien definidos.")
        c3.metric("Datos retenidos en el 3D", f"{var_pca:.1f}%", help="Al aplastar tus datos a 3 dimensiones para dibujarlos, este es el porcentaje de realidad que sobrevive.")

        st.divider()

        tab_galaxia, tab_perfiles = st.tabs(["🔭 La Galaxia de Datos (3D)", "👔 Perfiles de Tribus (Negocio)"])

        # --- Pestaña 1: El Universo 3D ---
        with tab_galaxia:
            st.subheader("Mapa Tridimensional de Agrupaciones (PCA)")
            st.info("💡 Rota el gráfico con el ratón. Cada punto es una fila de tu Excel. Los colores muestran las tribus que la IA ha descubierto.")
            
            fig_3d = px.scatter_3d(
                df_clusters, x='PCA_X', y='PCA_Y', z='PCA_Z',
                color='Cluster', opacity=0.7, color_continuous_scale='Turbo'
            )
            fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=0))
            st.plotly_chart(fig_3d, use_container_width=True)

        # --- Pestaña 2: Perfiles para Negocio ---
        with tab_perfiles:
            st.subheader("¿Qué define a cada Tribu?")
            st.info("💡 Estos son los promedios matemáticos de cada grupo. Usa la IA para traducirlos a estrategias de negocio.")
            
            perfiles = no_supervisado_service.obtener_perfil_clusters(df_clusters)
            st.dataframe(perfiles.style.background_gradient(cmap='Blues'), use_container_width=True)
            
            st.divider()
            if st.button("🧠 Pedir a la IA que Bautice a las Tribus", type="primary"):
                with st.spinner("La IA está analizando los perfiles y redactando la estrategia..."):
                    reporte_tribus = ia_client.generar_reporte_clustering(perfiles)
                    st.success("¡Análisis completado!")
                    st.markdown(reporte_tribus)

        st.divider()

        # --- NUEVO BOTÓN DE SALTO A FASE 5 ---
        st.markdown("### 🚀 Siguiente Paso: Clasificador de Tribus")
        st.info("Ahora que tienes tus tribus definidas, pasa a la Fase 5 para simular a qué grupo pertenece un nuevo usuario o subir una base de datos nueva para clasificarlos masivamente.")
        
        if st.button("🔮 Ir a Fase 5 (Simulador en Producción)", type="primary", use_container_width=True):
            st.session_state['fase_actual'] = 5
            st.rerun()
        
        st.divider()
                    
        # Opciones de salida
        st.markdown("### 📥 Navegar")
        
        # Preparar CSV para descarga
        csv_final = df_clusters.drop(columns=['PCA_X', 'PCA_Y', 'PCA_Z']).to_csv(index=False).encode('utf-8')
        
        col_down, col_reset = st.columns(2)
        with col_down:
            st.download_button(
                label="💾 Descargar Dataset con Clusters Asignados",
                data=csv_final,
                file_name="dataset_agrupado.csv",
                mime="text/csv",
                use_container_width=True,
                type="primary"
            )
        
        with col_reset:
            if st.button("🔄 Volver a intentar (Ir a Fase 3)", use_container_width=True):
                st.session_state['fase_actual'] = 3
                st.rerun()


    # ==========================================
    #                 FASE 5: PRODUCCIÓN
    # ==========================================
    elif fase == 5:
        st.title("🔮 Fase 5: Clasificador de Tribus")
        st.markdown("El modelo ya sabe cómo dividir a la población. Usa estas herramientas para clasificar nuevos individuos.")

        resultados = st.session_state.get('resultados_ns')
        
        if resultados is None or df_actual is None:
            st.error("🚨 El modelo no está en memoria. Vuelve a la Fase 3 y entrénalo.")
        else:
            tab_lotes, tab_vivo = st.tabs(["📁 Clasificación Masiva (CSV)", "🎛️ Simulador de Individuos"])

            # ---------------------------------------------------------
            # PESTAÑA 1: PREDICCIÓN POR LOTES
            # ---------------------------------------------------------
            with tab_lotes:
                st.markdown("### 📥 Asignación de Tribus en Bloque")
                archivo_nuevo = st.file_uploader("Sube el CSV con los nuevos datos:", type=["csv", "xlsx", "parquet"], key="uploader_ns")
                
                if archivo_nuevo:
                    from service import data_service
                    df_nuevo = data_service.leer_dataset_universal(archivo_nuevo)
                    
                    if df_nuevo is not None:
                        st.write("**Vista previa de los datos limpios:**")
                        st.dataframe(df_nuevo.head(3))

                        columna_id = st.selectbox("¿Hay alguna columna de Identidad (ID/Nombre)?", ["Ninguna"] + df_nuevo.columns.tolist())
                        
                        if st.button("🚀 Ejecutar Clasificación", type="primary"):
                            try:
                                df_procesar = df_nuevo.copy()
                                
                                caja_fuerte_ids = None
                                if columna_id != "Ninguna":
                                    caja_fuerte_ids = df_procesar[columna_id].copy()
                                    df_procesar = df_procesar.drop(columns=[columna_id], errors='ignore')
                                
                                # Reindexar para asegurar el mismo orden de columnas
                                columnas_entrenamiento = [c for c in df_actual.columns if c != columna_id]
                                df_limpio = df_procesar.reindex(columns=columnas_entrenamiento)
                                
                                # Llamamos a nuestro motor con el hack antibloqueos
                                predicciones = no_supervisado_service.predecir_nuevos_datos(df_limpio, resultados)
                                
                                df_final = pd.DataFrame()
                                if caja_fuerte_ids is not None:
                                    df_final[columna_id] = caja_fuerte_ids
                                df_final['Tribu_Asignada'] = predicciones
                                
                                st.success("✅ ¡Clasificación masiva completada!")
                                st.dataframe(df_final, use_container_width=True)
                                
                                csv_final = df_final.to_csv(index=False).encode('utf-8')
                                st.download_button(label="💾 Descargar Clasificación (.csv)", data=csv_final, file_name="clasificacion_tribus.csv", mime="text/csv")

                            except Exception as e:
                                st.error(f"⚠️ Error procesando los datos: {e}")

            # ---------------------------------------------------------
            # PESTAÑA 2: SIMULADOR EN VIVO
            # ---------------------------------------------------------
            with tab_vivo:
                st.markdown("### 🎛️ Simulador de Tribu")
                
                with st.form("form_simulador_ns"):
                    columnas_ui = st.columns(3)
                    input_usuario = {}
                    
                    columnas_entrenamiento = df_actual.columns.tolist()
                    
                    for idx, col in enumerate(columnas_entrenamiento):
                        with columnas_ui[idx % 3]:
                            if pd.api.types.is_numeric_dtype(df_actual[col]):
                                v_min = float(df_actual[col].min())
                                v_max = float(df_actual[col].max())
                                v_mean = float(df_actual[col].mean())
                                if v_min == v_max:
                                    input_usuario[col] = st.number_input(f"{col}", value=v_min)
                                else:
                                    input_usuario[col] = st.slider(f"{col}", min_value=v_min, max_value=v_max, value=v_mean)
                            else:
                                opciones = df_actual[col].dropna().unique().tolist()
                                input_usuario[col] = st.selectbox(f"{col}", options=opciones)
                                
                    submit_simulacion = st.form_submit_button("🔮 ¿A qué Tribu pertenece?", type="primary", use_container_width=True)
                    
                if submit_simulacion:
                    try:
                        df_sim = pd.DataFrame([input_usuario])
                        pred_cruda = no_supervisado_service.predecir_nuevos_datos(df_sim, resultados)
                        tribu_final = pred_cruda[0]
                        
                        st.divider()
                        st.markdown(f'<div class="resultado-prediccion">Pertenece a la Tribu: {tribu_final}</div>', unsafe_allow_html=True)
                        
                    except Exception as e:
                        st.error(f"Error en la simulación: {e}")

        # --- NAVEGACIÓN INFERIOR ---
        st.divider()
        st.markdown("### 🧭 Navegación")
        col_back, col_reset = st.columns(2)
        
        with col_back:
            if st.button("⬅️ Volver al Observatorio (Fase 4)", use_container_width=True):
                st.session_state['fase_actual'] = 4
                st.rerun()
                
        with col_reset:
            if st.button("🔄 Proyecto Nuevo (Fase 1)", type="secondary", use_container_width=True):
                for key in list(st.session_state.keys()):
                    if key != 'fase_actual':
                        del st.session_state[key]
                st.session_state['fase_actual'] = 1
                st.rerun()