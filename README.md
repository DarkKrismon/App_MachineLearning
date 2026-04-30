# 🧠 AutoML Suite & Data Copilot: Plataforma Segura de Machine Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]([text](https://appmachinelearning-darkkrismon4.streamlit.app/))

Una aplicación web *End-to-End* desarrollada en **Python** y **Streamlit** que sigue el ciclo de vida de los datos. Actúa como un copiloto interactivo que guía al usuario desde la ingesta de datos hasta la interpretación de modelos predictivos y de clustering.

**⚡ Motor de IA Cloud-Native:** La interpretación de datos y sugerencias estratégicas están impulsadas por **Llama 3.1** a través de la API ultrarrápida de **Groq**. La IA actúa como un agente de extracción de conclusiones, garantizando respuestas instantáneas en producción sin comprometer la fluidez de la aplicación web.

---

## 🚀 Arquitectura y Flujo de Trabajo (Pipeline)

El sistema está diseñado bajo un paradigma de **Divulgación Progresiva**: ofrece rutas automáticas para usuarios de negocio, pero mantiene paneles de "Ajustes Avanzados" para perfiles técnicos.

### 📊 Fase 1: Ingesta y Cruce Inteligente
*   **Cargador Universal:** Soporte robusto para `.csv` (con auto-detección de separadores europeos `;` o americanos `,`), `.xlsx` y `.parquet`.
*   **Smart Join:** Si se detectan múltiples fuentes, el motor unificador busca claves comunes y asiste mediante IA en la selección del método de cruce (Inner, Left, Outer Join) o apilado vertical.
*   **Prevención de Data Leakage (Fuga de Datos):** En la ruta supervisada, el *Train/Test Split* (con estratificación) se realiza **antes** de cualquier visualización o transformación, garantizando la pureza matemática del modelo.

### 🧹 Fase 2: Data Wrangling (Panel de Control)
*   Detección masiva de valores nulos y anomalías.
*   Imputación paramétrica (Media, Mediana, Moda) en bloque con confirmación interactiva.

### 🤖 Fase 3: AutoML Supervisado y No Supervisado
*   **Ruta Supervisada:** Detección automática del tipo de problema (Clasificación vs Regresión). Competición en la sombra de modelos robustos (Random Forest, Gradient Boosting, SVM, KNN) y optimización de hiperparámetros mediante `GridSearchCV` validado de forma cruzada (Cross-Validation) para evitar el *Overfitting*.
*   **Ruta No Supervisada:** Búsqueda de tribus/segmentos mediante K-Means, utilizando el "Gráfico del Codo" interactivo para la selección del valor *K* óptimo.

### 📈 Fase 4: Interpretación Dual
*   **Técnica:** Gráficos de Importancia de Variables, PCA en 2D, Matrices de Confusión y Gráficos de Radar desarrollados con **Plotly**.
*   **Ejecutiva:** Reportes sociológicos y de negocio generados por el LLM en la nube, que traducen métricas matemáticas crudas a lenguaje de toma de decisiones.

---

## 🛠️ Retos de Ingeniería Resueltos

Durante el desarrollo, se priorizó la solidez del *pipeline* frente a implementaciones ingenuas:

1.  **Alineación de Encoding (Train vs Test):** Entrenamiento con datos categóricos gestionado mediante diccionarios en el `LabelEncoder` personalizado, gestionando valores ocultos/no vistos en la fase de test mediante asignación de etiquetas neutras (`-1`) para evitar caídas en producción.
2.  **Gestión de Memoria en Streamlit:** Arquitectura guiada por estados (`st.session_state`) para encapsular las recargas de la interfaz, evitando la pérdida de los DataFrames procesados entre fases.
3.  **Persistencia del Escalado:** El ajuste topológico de los datos (`StandardScaler`) se instancia en la Fase 3 y se inyecta en la Fase 4 para garantizar que las inferencias visuales se correspondan con el espacio matemático de entrenamiento.
4.  **Despliegue y Seguridad de Credenciales:** Migración de un entorno local a Streamlit Community Cloud, implementando un puente seguro (`secrets.toml`) para las API Keys, separando estrictamente el código fuente de las credenciales de producción.

---
## 🗂️ Estructura del Proyecto

El código fuente está modularizado aplicando separación de responsabilidades (Clean Code):
```text
📁 MACHINE_LEARNING/
│
├── 📁 data/                  # Datasets de prueba (ignorados en git)
├── 📁 src/
│   ├── 📁 modulos/           # UI y lógica de renderizado por fases
│   │   ├── no_supervisado_ui.py
│   │   └── supervisado_ui.py
│   ├── 📁 service/           # Lógica de negocio, matemáticas y transformaciones
│   │   ├── data_service.py
│   │   ├── no_supervisado_service.py
│   │   └── supervisado_service.py
│   ├── __init__.py
│   ├── main.py               # Orquestador principal de Streamlit
│   └── ia_client.py      # Cliente de comunicación con LLM local
│
├── 📁 utils/                 # Scripts auxiliares
│   ├── generador_datos.py    # Generador de datos sintéticos con correlaciones
│   └── generador_titanic.py  
│
├── .gitignore
├── Estructura.ipynb          # Notas de arquitectura y bitácora de desarrollo
└── README.md