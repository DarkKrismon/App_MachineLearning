import streamlit as st
import service.data_service as data_service
import ia_client
from utils import generador_titanic

"""
Módulo de la Interfaz de Usuario para la Fase 1 (Ingesta de Datos).
Permite al usuario cargar archivos locales (CSV, Excel, Parquet) o inyectar 
datasets de prueba (Titanic/California) para arrancar el pipeline de Machine Learning.
"""

def renderizar_fase_ingesta() -> None:
    """
    Renderiza visualmente toda la página de la Fase 1 en Streamlit.
    Gestiona los botones de datos de demostración, el área para subir archivos 
    y la llamada inicial a Groq para generar el diagnóstico base.
    """
    
    st.title("📊 Fase 1: Ingesta y Diagnóstico")
    st.markdown("En esta página encontrarás un proceso completo dividido en fases de un proyecto de Machine Learning guiado. "\
    "Este proyecto consta tanto de un modelo **Supervisado** como de un modelo **No Supervisado**.")
    st.markdown("**Ingesta de Datos ➡️ Procesado ➡️ Entrenamiento de la IA ➡️ Resultados**")
    st.divider()

    # ==========================================
    # GENERACIÓN DATOS SINTÉTICOS
    # ==========================================
    st.markdown("### 🚀 Modo Demo (1 Clic)")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("🚢 **Supervivencia en el Titanic**\n\nPrueba el motor de *Smart Join* uniendo datos de pasajeros con sus billetes. Ideal para **Clasificación**.")
        if st.button("Cargar Demo Titanic", use_container_width=True, help="Inyecta 2 DataFrames en memoria automáticamente para probar cruces de datos."):
            df_personal, df_viaje = generador_titanic.generar_datos()
            st.session_state['dataframes'] = {
                "titanic_pasajeros.csv": df_personal,
                "titanic_viaje.csv": df_viaje
            }
            resumen = data_service.generar_resumen_df(df_personal, "titanic_pasajeros.csv") + "\n" + data_service.generar_resumen_df(df_viaje, "titanic_viaje.csv")
            st.session_state['resumen_tecnico'] = resumen
            st.rerun()

    with col2:
        st.success("🏡 **Precios de Inmuebles (California)**\n\nUn dataset masivo listo para descubrir patrones matemáticos. Ideal para **Regresión**.")
        if st.button("Cargar Demo Casas", use_container_width=True, help="Carga datos públicos de Scikit-Learn directamente."):
            from sklearn.datasets import fetch_california_housing
            data = fetch_california_housing(as_frame=True)
            st.session_state['dataframes'] = {"california_housing.csv": data.frame}
            st.session_state['resumen_tecnico'] = data_service.generar_resumen_df(data.frame, "california_housing.csv")
            st.rerun()

    st.divider()

    # ==========================================
    # CARGA DE ARCHIVOS
    # ==========================================
    st.markdown("### 📁 O sube tus propios datos")
    archivos_subidos = st.file_uploader(
        "Soporta CSV, Excel y Parquet", 
        type=["csv", "xlsx", "xls", "parquet"], 
        accept_multiple_files=True,
        help="Sube uno o varios archivos. Si subes varios, la aplicación te ayudará a cruzarlos en la siguiente fase."
    )

    if archivos_subidos:
        if 'dataframes' not in st.session_state or len(st.session_state['dataframes']) != len(archivos_subidos):
            st.session_state['dataframes'] = {}
            resumen_total = ""
            
            for archivo in archivos_subidos:
                df = data_service.leer_dataset_universal(archivo)
                if df is not None:
                    st.session_state['dataframes'][archivo.name] = df
                    resumen_total += data_service.generar_resumen_df(df, archivo.name) + "\n"
            
            st.session_state['resumen_tecnico'] = resumen_total

    # Mostrar DataFrames cargados
    if 'dataframes' in st.session_state and st.session_state['dataframes']:
        st.subheader("👀 Vista previa de los datos")
        for nombre, df in st.session_state['dataframes'].items():
            with st.expander(f"📄 {nombre} ({df.shape[0]} filas, {df.shape[1]} columnas)", expanded=False):
                st.dataframe(df.head(), use_container_width=True)

        st.subheader("Diagnóstico de la IA")
        st.info("Antes de avanzar, deja que la IA analice la salud de tus datos y te sugiera los próximos pasos.")
        
        if st.button("✨ Solicitar Diagnóstico a la IA", type="primary", help="Envía los metadatos a Groq (Llama 3.1) para identificar nulos y cruces necesarios."):
            with st.spinner("La IA está analizando la estructura de los datos..."):
                diagnostico = ia_client.consultar_diagnostico(st.session_state['resumen_tecnico'])
                st.session_state['diagnostico_ia'] = diagnostico
                
        if st.session_state.get('diagnostico_ia'):
            st.success("Diagnóstico completado:")
            st.markdown(st.session_state['diagnostico_ia'])

        st.divider()
        if st.button("Avanzar a Fase 2 (Cruze y Limpieza) ➡️", use_container_width=True):
            st.session_state.fase_actual = 2
            st.rerun()