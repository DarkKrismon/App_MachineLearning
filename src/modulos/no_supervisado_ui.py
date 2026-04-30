import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from service import no_supervisado_service
import ia_client as ia_client


''' 
Encargado visual de la fase 3 y 4 de modelo NO Sºupervisado 
llamando a la función "renderizar_fase_no_supervisada""
'''


def renderizar_fase_no_supervisada():
    st.title("🌌 Fase 3: Descubrimiento de Grupos (Clustering)")
    st.info("Bienvenido al entorno No Supervisado. Aquí no hay respuestas correctas, solo agrupaciones matemáticas.")

    if 'analisis_ia_tribus' not in st.session_state:
        st.session_state['analisis_ia_tribus'] = None

    df_actual = st.session_state['dataframes']["dataset_unificado.csv"]
    

    cols_num = df_actual.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if len(cols_num) < 2:
        st.error("❌ Necesitas al menos 2 variables numéricas en el dataset para buscar patrones.")
        return

    st.markdown("### 1. Selección de Variables")
    st.write("¿Qué características quieres que mire la IA para crear las agrupaciones?")
    cols_elegidas = st.multiselect("Variables clave:", cols_num, default=cols_num[:3])

    if len(cols_elegidas) >= 2:
        df_base, datos_escalados, scaler = no_supervisado_service.preparar_datos_clustering(df_actual, cols_elegidas)
        
        st.divider()
        st.markdown("### 2. El Dilema del Codo (Buscando el número ideal)")
        col_grafico, col_texto = st.columns([2, 1])
        
        with col_texto:
            st.write("La gráfica de la izquierda muestra la 'Inercia' (el caos dentro de los grupos).")
            st.write("👉 **Busca el 'Codo':** El punto donde la caída de la línea empieza a suavizarse. Ese es el número matemático ideal de tribus.")
            n_clusters = st.slider("Selecciona el número de Tribus:", min_value=2, max_value=10, value=3)

        with col_grafico:
            rango_k, inercias = no_supervisado_service.calcular_codo(datos_escalados)
            fig_codo = px.line(x=rango_k, y=inercias, markers=True, title="Gráfico del Codo", labels={'x':'Número de Tribus (k)', 'y':'Inercia / Caos'})
            st.plotly_chart(fig_codo, use_container_width=True)


        btn_entrenar = st.button("🚀 Ejecutar Agrupación K-Means", type="primary")
        if btn_entrenar:
            st.session_state['clustering_calculado'] = True
            st.session_state['analisis_ia_tribus'] = None
            
            df_tribus, modelo_kmeans = no_supervisado_service.ejecutar_clustering(df_base, datos_escalados, n_clusters)
            st.session_state['df_tribus'] = df_tribus
            st.session_state['perfiles_tribus'] = no_supervisado_service.obtener_centroides_radar(df_tribus, cols_elegidas)
            st.session_state['n_clusters_guardado'] = n_clusters

        # Si el baúl tiene datos: Pintamos la Fase 3
        if st.session_state.get('clustering_calculado', False):
            st.divider()
            
            # Rescatamos los datos del baúl para usarlos
            df_tribus = st.session_state['df_tribus']
            perfiles = st.session_state['perfiles_tribus']
            n_clusters_guardado = st.session_state['n_clusters_guardado']

            st.markdown(f"### 3. Resultados")
            
            tab_mapa, tab_perfiles = st.tabs(["🗺️ Mapa de Territorios (PCA)", "🕸️ Radiografía de Tribus (Radar)"])
            
            with tab_mapa:
                st.write("Hemos aplastado tus variables a 2 dimensiones (X e Y) usando PCA para que puedas ver la separación entre grupos.")
                df_pca, varianza = no_supervisado_service.aplicar_pca_2d(datos_escalados, df_tribus['Tribu'])
                fig_pca = px.scatter(df_pca, x='Componente_X', y='Componente_Y', color='Tribu', title=f"Mapa PCA (Varianza conservada: {varianza:.1f}%)")
                st.plotly_chart(fig_pca, use_container_width=True)
                
            with tab_perfiles:
                st.write("Este gráfico muestra el valor promedio de cada variable para cada grupo.")
                
                # Gráfico de Radar chulo con Plotly
                fig_radar = go.Figure()
                for i, fila in perfiles.iterrows():
                    fig_radar.add_trace(go.Scatterpolar(
                        r=fila[cols_elegidas].values,
                        theta=cols_elegidas,
                        fill='toself',
                        name=f"Tribu {fila['Tribu']}"
                    ))
                fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True)), showlegend=True)
                st.plotly_chart(fig_radar, use_container_width=True)

                st.write("**Datos Crudos (Media de cada Tribu):**")
                st.dataframe(perfiles.set_index('Tribu'), use_container_width=True)
                
            # --- SECCIÓN DE INTELIGENCIA ARTIFICIAL ---
            st.divider()
            st.markdown("### 4. Análisis Sociológico con Inteligencia Artificial")
            st.info("¿Los números crudos no te dicen nada? Deja que nuestra IA lea la tabla y le ponga cara y ojos a cada Tribu.")
                
            # Este es el segundo botón. Al pulsarlo, recargará, pero como 'clustering_calculado' es True, la sección 3 sobrevivirá
            if st.button("✨ Generar Lectura de los Grupos", type="primary", use_container_width=True):
                with st.spinner("Ollama está leyendo los datos y escribiendo el reporte..."):
                    st.session_state['analisis_ia_tribus'] = ia_client.interpretar_grupos(perfiles)
                    st.rerun() # Forzamos recarga

            if st.session_state['analisis_ia_tribus']:
                st.success("¡Análisis completado!")
                st.markdown(st.session_state['analisis_ia_tribus'])
                    
                if st.button("🗑️ Borrar Análisis"):
                    st.session_state['analisis_ia_tribus'] = None
                    st.rerun()