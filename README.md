# 🧠 AutoML Suite & Data Copilot: Plataforma End-to-End de Machine Learning

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://appmachinelearning-darkkrismon4.streamlit.app/)

Una aplicación web *SaaS-like* desarrollada en **Python** y **Streamlit** que abarca el ciclo de vida completo de los datos. Actúa como un copiloto interactivo que guía al usuario desde la ingesta de datos brutos hasta el despliegue de modelos predictivos en producción.

**⚡ Motor de IA Cloud-Native:** La interpretación de datos, el bautizo de clusters y las sugerencias estratégicas están impulsadas por **Llama 3.1** a través de la API ultrarrápida de **Groq**. La IA actúa como un analista de negocio integrado, traduciendo matrices matemáticas a lenguaje ejecutivo en milisegundos.

---

## 🚀 Arquitectura y Flujo de Trabajo (Pipeline)

El sistema está diseñado bajo un paradigma de **Divulgación Progresiva**: ofrece flujos automatizados para perfiles de negocio, manteniendo paneles de "Ajustes Avanzados" para perfiles técnicos. La navegación está orquestada por una máquina de estados (State Machine) con un GPS.

### 📥 Fase 1: Ingesta y Cruce Inteligente
* **Cargador Universal:** Soporte para `.csv`, `.xlsx` y `.parquet`.
* **Smart Join:** Motor unificador que detecta claves comunes entre múltiples fuentes y ejecuta cruces relacionales (Inner, Left, Outer Join) o apilados verticales (Concat).

### 🧹 Fase 2: Data Wrangling y Prevención de Data Leakage
* **Aislamiento Train/Test:** En la ruta predictiva, la separación de datos estratificada se realiza **antes** de cualquier visualización, garantizando la pureza del experimento.
* **Limpieza Paramétrica:** Imputación masiva de nulos y gestión de *Outliers* (vía IQR) calculada estrictamente sobre el Train y proyectada al Test.

### ⚙️ Fase 3: Motores de Machine Learning
* **Ruta Supervisada (AutoML):** Competición en paralelo de algoritmos robustos (Random Forest, Gradient Boosting, SVM, KNN) y optimización de hiperparámetros mediante `GridSearchCV` validado de forma cruzada (Cross-Validation).
* **Ruta No Supervisada (Clustering):** Búsqueda de tribus usando K-Means y Agglomerative Clustering. Incluye un escáner automatizado basado en el *Silhouette Score* para determinar la *K* óptima matemáticamente.

### 📊 Fase 4: Observatorio e Interpretación Dual
* **Panel Técnico:** Importancia de Variables, PCA (Reducción a 3 Dimensiones interactivas), Matrices de Confusión y reporte de métricas generados con **Plotly**.
* **Panel Ejecutivo:** Reportes generados por LLM que traducen el rendimiento del modelo a estrategias de negocio accionables.

### 🔮 Fase 5: Producción y Simulación
* **Predicción por Lotes:** Capacidad para ingestar nuevos CSV ciegos y devolver la asignación de predicciones o clusters listos para descargar.
* **Simulador en Vivo:** Interfaz dinámica generada a partir de los metadatos de entrenamiento para inferir resultados sobre un único individuo en tiempo real.

---

## 🔒 Privacidad y Tratamiento de Datos (Compliance)

Dado que la plataforma maneja la ingesta de datasets personalizados, la arquitectura ha sido diseñada con un enfoque estricto en la privacidad:

* **Procesamiento Core Aislado:** Toda la limpieza de datos (Data Wrangling), divisiones (Train/Test) y entrenamiento de modelos matemáticos (Scikit-Learn) se ejecuta **estrictamente en la memoria del servidor local o del contenedor**. Ninguna fila de datos brutos se envía a APIs de terceros para su entrenamiento.
* **Interacciones con LLM (Groq API):** La conexión con la inteligencia artificial generativa se realiza mediante un puente seguro que **solo transmite metadatos o datos agregados**.
    * *Fase 1:* Solo se envía un resumen de estructura (nombres de columnas, conteo de filas y nulos).
    * *Fases 4:* Solo se envían métricas de rendimiento (precisión, recall) o los *centroides* matemáticos (promedios) de los clusters descubiertos.
* **Zero PII (Personal Identifiable Information):** El sistema invita al usuario a descartar columnas de Identidad (IDs, Nombres) antes del entrenamiento y la predicción, garantizando que ninguna información personal identificable viaje a través de la red durante la generación de reportes ejecutivos.
---

## 🗂️ Estructura del Proyecto

```text
📁 MACHINE_LEARNING/
├── 📁 src/
│   ├── 📁 assets/            # CSS para White-labeling y UI personalizada
│   │   └── styles.css
│   ├── 📁 modulos/           # Interfaces de usuario modulares (Fase 1 a 5)
│   │   ├── ingesta_ui.py
│   │   ├── procesado_ui.py
│   │   ├── supervisado_ui.py
│   │   └── no_supervisado_ui.py
│   ├── 📁 service/           # Lógica matemática, Data Wrangling y Modelado (Type Hinted)
│   │   ├── data_service.py
│   │   ├── supervisado_service.py
│   │   └── no_supervisado_service.py
│   ├── 📁 utils/             # Generadores de datos sintéticos
│   │   ├── generador_datos.py
│   │   └── generador_titanic.py
│   │
│   ├── main.py               # Orquestador y enrutador de estados
│   └── ia_client.py          # Conexión Groq/Llama 3.1
│
├── .gitignore
├── requirements.txt
└── README.md