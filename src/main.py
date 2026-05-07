import streamlit as st
from modulos import ingesta_ui, procesado_ui

'''
Función Main encargada de administrar el cambio de fases dentro de la aplicación.
Ejecución por consola:

streamlit run main.py
'''

def main():
    st.set_page_config(page_title="Machine-Learning", page_icon="🤖", layout="wide")

    if 'fase_actual' not in st.session_state:
        st.session_state.fase_actual = 0

    fase = st.session_state.get('fase_actual', 0)

    # ========================================================
    # FASE 0: PORTADA
    # ========================================================
    if fase == 0:
        # Usamos columnas vacías a los lados para "empujar" el contenido al centro
        col_izq, col_centro, col_der = st.columns([1, 2, 1])
        
        with col_centro:
            st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?q=80&w=2070&auto=format&fit=crop", use_container_width=True)
            
            st.title("🧠 AutoML")
            st.markdown("### Tu copiloto de Machine Learning")
            st.markdown("Sube tus datos, descubre patrones ocultos y entrena modelos predictivos de forma guiada y sin escribir código.")

            st.write("Por: Juan Ramón Torres Martínez")
            

            if st.button("Comenzar Análisis", use_container_width=True):
                st.session_state.fase_actual = 1
                st.rerun()

    # =========================================================
    # FASE 1
    # =========================================================
    elif fase == 1:
        ingesta_ui.renderizar_fase_ingesta()


    # =========================================================
    # FASE 2: LIMPIEZA DE DATOS
    # =========================================================

    elif fase == 2:
        procesado_ui.renderizar_fase_procesado()

    # =========================================================
    # FASE 3 y 4: ENTRENAMIENTO Y RESULTADOS
    # =========================================================

    elif fase in [3, 4, 5]:
            ruta = st.session_state.get('ruta_elegida', 'supervisado')
            
            if ruta == 'supervisado':
                from modulos import supervisado_ui
                supervisado_ui.renderizar_fase_supervisada()
                
            elif ruta == 'no_supervisado':
                from modulos import no_supervisado_ui
                no_supervisado_ui.renderizar_fase_no_supervisada()   


if __name__ == "__main__":
    main()