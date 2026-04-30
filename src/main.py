import streamlit as st
import service.data_service as data_service
import ia_client
from sklearn.model_selection import train_test_split


st.set_page_config(page_title="Machine-Learning", page_icon="🤖", layout="wide")

fase = st.session_state.get('fase_actual', 1)


# =========================================================
# FASE 1
# =========================================================
if fase == 1:
    st.title("📊 Fase 1: Ingesta y Diagnóstico")
    st.markdown("En esta página encontrarás un proceso completo dividido en fases de un proyecto de Machine Learning guiado. "\
    "Este proyecto consta tanto de un modelo **Supervisado** como de un modelo **No Supervisado**.")
    st.markdown("Ingesta de Datos ➡️ Procesado ➡️ Entrenamiento de la IA ➡️ Resultados")
    st.markdown("---")

    archivos_subidos = st.file_uploader("Sube uno o varios archivos", type=['csv', 'xlsx', 'xls', 'parquet'], accept_multiple_files=True)
    

    if archivos_subidos:
        nombres_subidos = [f.name for f in archivos_subidos]
        archivos_en_memoria = st.session_state.get('archivos_procesados', [])
            
        # Solo procesar si hay archivos nuevos que no estén en memoria
        if set(nombres_subidos) != set(archivos_en_memoria):
            st.session_state['dataframes'] = {}
            st.session_state['archivos_procesados'] = nombres_subidos

            resumen_completo = ""
            lista_dfs_para_unir = []

            for archivo in archivos_subidos:
                nombre = archivo.name
                df = data_service.leer_dataset_universal(archivo)

                if df is not None:
                    st.session_state['dataframes'][nombre] = df
                    lista_dfs_para_unir.append(df)
                    resumen_completo += data_service.generar_resumen_df(df, nombre) + "\n"
                
            if resumen_completo:
                with st.spinner("La IA está analizando la estructura de tus datos..."):
                    st.session_state['diagnostico_ia'] = ia_client.consultar_diagnostico(resumen_completo)


    diagnostico = st.session_state.get('diagnostico_ia')
    if diagnostico:
        st.markdown("### 📰 Diagnóstico:")
        st.info(diagnostico)
        
        st.markdown("### 📋 Vista Previa: ")
        for nombre, df in st.session_state['dataframes'].items():
            with st.expander(f"Ver {nombre}"):
                st.dataframe(df.head(10))
        st.divider()

        # --- MOTOR DE UNIFICACIÓN ---
        num_archivos = len(st.session_state['dataframes'])

        if num_archivos > 1:
            st.warning("⚠️ Tienes más de un archivo. El siguiente paso obligatorio es unificarlos.")
            st.markdown("### 🔗 Unificador de Tablas:")

            nombres_archivos = list(st.session_state['dataframes'].keys())
            col1, col2 = st.columns(2)
            with col1: df1_name = st.selectbox("Tabla Principal", nombres_archivos, index=0)
            with col2: df2_name = st.selectbox("Tabla a Unir", nombres_archivos, index=1)

            if df1_name != df2_name:
                df1 = st.session_state['dataframes'][df1_name]
                df2 = st.session_state['dataframes'][df2_name]
                metodo_union = st.radio("Método:", ["🔽 Apilar las filas verticalmente", "▶️ Cruzar columnas horizontalmente"])

                if "Apilar" in metodo_union:
                    if st.button("🟢 Ejecutar Apilado"):
                        df_merged = data_service.apilar_tablas(df1, df2)
                        st.session_state['dataframes'] = {"dataset_unificado.csv": df_merged}
                        st.rerun() 
                else:
                    col_comunes = data_service.obtener_columnas_comunes(df1, df2)
                    col3, col4 = st.columns(2)
                    with col3:
                        if col_comunes:
                            col_left = st.selectbox("Clave:", col_comunes)
                            col_right = col_left 
                        else:
                            col_left = st.selectbox(f"Columna clave en {df1_name}:", df1.columns)
                            col_right = st.selectbox(f"Columna clave en {df2_name}:", df2.columns)
                    with col4: tipo_join = st.radio("Tipo:", ["inner", "left", "outer"], horizontal=True, 
                                                    help="INNER JOIN: Selecciona el cruce exacto en ambas (intersección)\n"
                                                         "LEFT JOIN: Mantiene todos los regiatros de la principal\n"
                                                         "OUTER JOIN: Mantiene todos los registros de ambas")

                    if st.button("🟢 Ejecutar Cruce"):
                        df_merged = data_service.unificar_tablas(df1, df2, col_left, col_right, tipo_join)
                        st.session_state['dataframes'] = {"dataset_unificado.csv": df_merged}
                        st.rerun() 

        elif num_archivos == 1:
            nombre_actual = list(st.session_state['dataframes'].keys())[0]

            if nombre_actual != "dataset_unificado.csv":
                st.session_state['dataframes']["dataset_unificado.csv"] = st.session_state['dataframes'].pop(nombre_actual)

            st.success("✅ Dataset unificado y listo para trabajar.")
            
            df_actual = st.session_state['dataframes']["dataset_unificado.csv"]
            csv_ready = data_service.preparar_descarga_csv(df_actual)
            st.download_button("📥 Descargar Dataset", data=csv_ready, file_name="dataset_unificado.csv", mime="text/csv")
            

            st.divider()
            st.markdown("### ¿Qué quieres hacer con estos datos?")
            
            
            ruta = st.radio(
                "Selecciona tu objetivo analítico:",
                ["🎯 Predecir un objetivo específico (Machine Learning Supervisado)", 
                "🌌 Descubrir patrones ocultos y grupos (Clustering No Supervisado)"]
            )
            
            # --- RUTA SUPERVISADA ---
            if "Predecir" in ruta:
                st.info("Has elegido la ruta predictiva. Para evitar Data Leakage, debemos separar los datos de train y test" \
                " antes de limpiar los datos.")
                
                columna_target = st.selectbox("Selecciona la Variable Objetivo (La que quieres predecir):", df_actual.columns)
                
                if st.button("✂️ Cortar Datos y avanzar a Fase 2", type="secondary"):
                    try:
                        test_size = 0.20 if len(df_actual) >= 1000 else 0.25
                        
                        df_train, df_test = train_test_split(
                            df_actual, 
                            test_size=test_size, 
                            random_state=42, 
                            stratify=df_actual[columna_target]
                        )
                        
                        # Guardamos todo en el baúl fuerte
                        st.session_state['ruta_elegida'] = 'supervisado'
                        st.session_state['target_elegido'] = columna_target
                        st.session_state['df_train'] = df_train
                        st.session_state['df_test'] = df_test
                        
                        st.session_state['fase_actual'] = 2
                        st.rerun()
                        
                    except ValueError as e:
                        st.error(f"Error al cortar los datos. Prueba con otra variable.")

                # --- RUTA NO SUPERVISADA ---
            else:
                st.info("Has elegido descubrir patrones. Como no hay un examen final, la IA analizará el 100% de los datos.")
                
                if st.button("🚀 Avanzar a Fase 2 (Limpieza)", type="primary"):
                    st.session_state['ruta_elegida'] = 'no_supervisado'
                    st.session_state['df_entero'] = df_actual.copy()
                    st.session_state['fase_actual'] = 2
                    st.rerun()


# =========================================================
# FASE 2: LIMPIEZA DE DATOS
# =========================================================

elif fase == 2:
    st.title("🧹 Fase 2: Limpieza del Dataset")
    
    # 1. IDENTIFICAR DE DÓNDE VENIMOS Y QUÉ DATOS TOCAR
    ruta = st.session_state.get('ruta_elegida', 'no_supervisado')

    if ruta == 'supervisado':
        df_actual = st.session_state.get('df_train')
    else:
        df_actual = st.session_state.get('df_entero')

    # 3 Pestañas para organización
    tab_limpieza, tab_outliers, tab_variables = st.tabs([
        "📰 1. Limpieza Datos", 
        "🚨 2. Radar de Anomalías", 
        "🧬 3. Mutación de Variables"
    ])

    # --- PESTAÑA 1: LIMPIEZA ESTÁNDAR ---
    with tab_limpieza:
        st.markdown("### Paso 1: Diagnóstico Algorítmico")
        reporte = data_service.escanear_dataset(df_actual)
        columnas_enfermas = {col: d for col, d in reporte['columnas'].items() if d['alertas']}

        if not columnas_enfermas:
            st.success("¡Datos base sanos! Revisa las otras pestañas o avanza a la Fase 3.")
        else:
            st.warning(f"⚠️ Se requieren {len(columnas_enfermas)} intervenciones por nulos o datos innecesarios.")

            with st.form("formulario_limpieza"):
                acciones_usuario = {}
                for col, datos in columnas_enfermas.items():
                    st.markdown(f"**🔹 `{col}`** | Tipo: `{datos['tipo']}`")
                    st.error(f"⛔ Problema: {', '.join(datos['alertas'])}")
                    
                    opciones = ["Ignorar (No hacer nada)", "Borrar columna entera"]
                    if datos['nulos'] > 0:
                        if datos['tipo'] in ['float64', 'int64']:
                            opciones.extend(["Rellenar nulos con Media", "Rellenar nulos con Mediana", "Rellenar nulos con 0"])
                        else:
                            opciones.extend(["Rellenar nulos con 'Desconocido'", "Rellenar nulos con Moda"])
                    
                    idx_defecto = 1 if "Borrar" in datos['recomendacion'] else len(opciones)-1
                    acciones_usuario[col] = st.selectbox(f"Acción para '{col}':", opciones, index=idx_defecto, key=f"sel_{col}")
                    st.write("---")

                if st.form_submit_button("Aplicar Cambios del Bloque"):
                    # EL GRAN CAMBIO DE LÓGICA AQUÍ
                    if ruta == 'supervisado':
                        # NECESITAS CREAR ESTA FUNCIÓN EN TU SERVICE. 
                        # Debe calcular medias en Train y aplicarlas a Train Y Test.
                        df_train_limpio, df_test_limpio = data_service.aplicar_limpieza_dual(
                            st.session_state['df_train'], 
                            st.session_state['df_test'], 
                            acciones_usuario
                        )
                        st.session_state['df_train'] = df_train_limpio
                        st.session_state['df_test'] = df_test_limpio
                    else:
                        df_curado = data_service.aplicar_limpieza(df_actual, acciones_usuario)
                        st.session_state['df_entero'] = df_curado
                    
                    st.rerun()

        st.markdown("### Paso 2: Intuición de Negocio")
        col_guillotina, col_btn = st.columns([3, 1])
        with col_guillotina:
            columnas_a_borrar = st.multiselect("Selecciona columnas para borrar:", df_actual.columns)
        with col_btn:
            st.write("") 
            st.write("")
            if st.button("🗑️ Borrar Columnas"):
                if columnas_a_borrar:
                    if ruta == 'supervisado':
                        st.session_state['df_train'] = st.session_state['df_train'].drop(columns=columnas_a_borrar)
                        st.session_state['df_test'] = st.session_state['df_test'].drop(columns=columnas_a_borrar)
                    else:
                        st.session_state['df_entero'] = st.session_state['df_entero'].drop(columns=columnas_a_borrar)
                    st.rerun()

    # --- PESTAÑA 2: RADAR DE OUTLIERS ---
    with tab_outliers:
        st.markdown("### Radar de Anomalías (Outliers)")
        
        cols_numericas = df_actual.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if cols_numericas:
            col_analisis = st.selectbox("Selecciona una columna numérica para escanear:", cols_numericas)
            
            col_btn1, col_btn2 = st.columns(2)
            with col_btn1:
                if st.button("Escanear Anomalías"):
                    indices_outliers = data_service.detectar_outliers_iqr(df_actual, col_analisis)
                    if indices_outliers:
                        st.session_state['outliers_temporales'] = indices_outliers
                    else:
                        st.success(f"✅ La columna '{col_analisis}' no tiene valores extremos preocupantes.")
                        st.session_state['outliers_temporales'] = []
            
            if 'outliers_temporales' in st.session_state and len(st.session_state['outliers_temporales']) > 0:
                indices = st.session_state['outliers_temporales']
                st.error(f"⚠️ ¡Se han encontrado {len(indices)} filas con valores anómalos en '{col_analisis}'!")
                st.dataframe(df_actual.loc[indices].head(5))
                
                with col_btn2:
                    if st.button("☢️ Eliminar estas filas", type="primary"):
                        if ruta == 'supervisado':
                            # ¡OJO A ESTA REGLA DE NEGOCIO! 
                            st.session_state['df_train'] = st.session_state['df_train'].drop(index=indices)
                            # NO TOCAMOS EL TEST AQUÍ (Lee la explicación abajo)
                        else:
                            st.session_state['df_entero'] = st.session_state['df_entero'].drop(index=indices)
                        
                        st.session_state['outliers_temporales'] = []
                        st.rerun()
        else:
            st.write("No hay columnas numéricas para analizar.")

    # --- PESTAÑA 3: MUTACIÓN DE VARIABLES ---
    with tab_variables:
        st.markdown("### Creación de nuevas Variables")
        
        cols_num = df_actual.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if len(cols_num) >= 2:
            c1, c2, c3 = st.columns(3)
            with c1: col_math_1 = st.selectbox("Columna A:", cols_num)
            with c2: operacion = st.selectbox("Operación:", ["Suma (+)", "Resta (-)", "Multiplicación (*)", "División (/)"])
            with c3: col_math_2 = st.selectbox("Columna B:", cols_num)
            
            nombre_nueva_col = st.text_input("Nombre de la nueva columna:", "Nueva_Columna")
            
            if st.button("Calcular Columna"):
                if nombre_nueva_col in df_actual.columns:
                    st.error("⚠️ Ya existe una columna con ese nombre.")
                else:
                    if ruta == 'supervisado':
                        st.session_state['df_train'] = data_service.crear_nueva_columna(st.session_state['df_train'], col_math_1, col_math_2, operacion, nombre_nueva_col)
                        st.session_state['df_test'] = data_service.crear_nueva_columna(st.session_state['df_test'], col_math_1, col_math_2, operacion, nombre_nueva_col)
                    else:
                        st.session_state['df_entero'] = data_service.crear_nueva_columna(st.session_state['df_entero'], col_math_1, col_math_2, operacion, nombre_nueva_col)
                    
                    st.success(f"✅ Variable '{nombre_nueva_col}' creada con éxito.")
                    st.rerun()
        else:
            st.warning("Necesitas al menos 2 columnas numéricas para realizar operaciones matemáticas.")

    st.divider()

    with st.expander("Ver Dataset Actualizado", expanded=True):
        st.dataframe(df_actual.head(10))
    
    st.divider()

    if st.button("🚀 Avanzar a Fase 3 (Entrenamiento)", type="primary", use_container_width=True):
            st.session_state['fase_actual'] = 3
            st.session_state.pop('outliers_temporales', None)
            st.rerun()

elif fase == 3 or fase == 4:
        ruta = st.session_state.get('ruta_elegida', 'supervisado')
        
        if ruta == 'supervisado':
            from modulos import supervisado_ui
            supervisado_ui.renderizar_fase_supervisada()
            
        elif ruta == 'no_supervisado':
            from modulos import no_supervisado_ui
            no_supervisado_ui.renderizar_fase_no_supervisada()   
        