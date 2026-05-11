import os
import streamlit as st
from modulos import ingesta_ui, procesado_ui


def cargar_css():
    """Lee el archivo CSS externo y lo inyecta en Streamlit"""
    # Ajusta la ruta dependiendo de dónde hayas creado la carpeta assets
    ruta_css = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
    
    try:
        with open(ruta_css, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("⚠️ No se encontró el archivo styles.css")

def renderizar_gps(fase_actual):
    """Genera el HTML del GPS utilizando las clases del archivo styles.css"""
    pasos = ["📥 Ingesta", "🛠️ Procesado", "⚙️ Entrenamiento", "📊 Análisis", "🔮 Producción"]
    
    st.markdown('<div class="gps-container">', unsafe_allow_html=True)
    cols = st.columns(len(pasos))
    
    for i, nombre in enumerate(pasos):
        paso_num = i + 1
        with cols[i]:
            if paso_num < fase_actual:
                estado_clase = "step-done"
                texto_estado = "✅ Listo"
            elif paso_num == fase_actual:
                estado_clase = "step-active"
                texto_estado = "📍 Estás aquí"
            else:
                estado_clase = "step-future"
                texto_estado = "🔒 Esperando"
                
            st.markdown(f"""
                <div class="step-box {estado_clase}">
                    <div class="step-title">{nombre}</div>
                    <div class="step-status">{texto_estado}</div>
                </div>
            """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    st.write("")


def main():
    # 1. Cargar el motor de estilos
    cargar_css()
    
    # 2. Recuperar la fase
    fase = st.session_state.get('fase_actual', 0)
    
    # 3. Dibujar el GPS
    if fase > 0:
        renderizar_gps(fase)
    
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