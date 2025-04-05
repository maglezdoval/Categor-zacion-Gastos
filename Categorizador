import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import io # Necesario para leer el archivo subido

# --- Funciones Auxiliares (con caché para rendimiento) ---

# Cache para la carga y procesamiento de datos
@st.cache_data
def load_and_process_data(uploaded_file):
    """Carga y preprocesa los datos desde un archivo subido."""
    try:
        # Leer el archivo subido directamente
        bytes_data = uploaded_file.getvalue()
        # Intentar leer con utf-8, si falla, probar con latin1
        try:
            df = pd.read_csv(io.BytesIO(bytes_data), encoding='utf-8', sep=';')
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(bytes_data), encoding='latin1', sep=';')

        # Filtrar filas donde la categoría o subcategoría no esté presente (importante para el aprendizaje)
        df = df.dropna(subset=['CATEGORÍA', 'SUBCATEGORIA'])
        df = df[df['CATEGORÍA'].str.strip() != ''] # Asegurar que no haya categorías vacías

        # Limpieza y normalización
        # Manejar posibles errores si 'IMPORTE' no es convertible directamente
        try:
            df['IMPORTE'] = df['IMPORTE'].str.replace(',', '.', regex=False).astype(float)
        except AttributeError:
            st.warning("La columna 'IMPORTE' no parece ser de tipo texto. Intentando conversión directa.")
            try:
                df['IMPORTE'] = df['IMPORTE'].astype(float)
            except ValueError:
                st.error("No se pudo convertir la columna 'IMPORTE' a número. Revise el formato.")
                return None
        except ValueError:
             st.error("Error al convertir 'IMPORTE' a número. Verifique que no haya valores no numéricos (después de reemplazar ',').")
             return None


        text_cols = ['CONCEPTO', 'COMERCIO', 'CATEGORÍA', 'SUBCATEGORIA']
        for col in text_cols:
             if col in df.columns:
                # Rellenar NaNs con string vacío antes de aplicar métodos de string
                df[col] = df[col].fillna('').astype(str).str.lower().str.strip()


        # Combinar CONCEPTO y COMERCIO para el aprendizaje
        df['TEXTO'] = df['CONCEPTO'] + ' ' + df['COMERCIO']

        return df

    except Exception as e:
        st.error(f"Error al cargar o procesar el archivo CSV: {e}")
        return None

# Cache para la extracción de la base de conocimiento
@st.cache_data
def extract_knowledge(df):
    """Extrae categorías, subcategorías y comercios del DataFrame."""
    if df is None or 'CATEGORÍA' not in df.columns:
        return [], {}, {}

    categorias = df['CATEGORÍA'].unique().tolist()
    # Asegurarse de que '' no esté en categorías si surge de alguna fila
    categorias = [cat for cat in categorias if cat]


    subcategorias_por_categoria = {}
    comercios_por_categoria = {}

    if 'SUBCATEGORIA' in df.columns:
        for categoria in categorias:
             # Filtrar NaNs o vacíos en subcategoría y comercio antes de unique()
             subcats_series = df.loc[df['CATEGORÍA'] == categoria, 'SUBCATEGORIA'].dropna()
             subcats_series = subcats_series[subcats_series.str.strip() != '']
             subcategorias = subcats_series.unique().tolist()
             subcategorias_por_categoria[categoria] = subcategorias

    if 'COMERCIO' in df.columns:
         for categoria in categorias:
            comers_series = df.loc[df['CATEGORÍA'] == categoria, 'COMERCIO'].dropna()
            comers_series = comers_series[comers_series.str.strip() != '']
            comercios = comers_series.unique().tolist()
            # Quitar 'n/a' si existe como comercio aprendido
            comercios = [com for com in comercios if com != 'n/a']
            comercios_por_categoria[categoria] = comercios

    return categorias, subcategorias_por_categoria, comercios_por_categoria

# Cache para el entrenamiento del modelo (más pesado)
@st.cache_resource # Usar cache_resource para objetos complejos como modelos y vectorizadores
def train_model(df):
    """Entrena un modelo de clasificación."""
    if df is None or df.empty or 'TEXTO' not in df.columns or 'CATEGORÍA' not in df.columns:
        st.warning("No hay suficientes datos válidos para entrenar el modelo.")
        return None, None, "No se pudo entrenar el modelo."

    # Asegurarse de que no haya NaNs en las columnas clave para el modelo
    df_train = df.dropna(subset=['TEXTO', 'CATEGORÍA'])
    df_train = df_train[df_train['CATEGORÍA'].str.strip() != '']

    if df_train.empty:
         st.warning("No hay datos válidos para entrenar después de filtrar.")
         return None, None, "No se pudo entrenar el modelo."

    if len(df_train['CATEGORÍA'].unique()) < 2:
        st.warning("Se necesita al menos 2 categorías diferentes para entrenar el modelo de clasificación.")
        return None, None, "Datos insuficientes (menos de 2 categorías)."


    try:
        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(df_train['TEXTO'])
        y = df_train['CATEGORÍA']

        # Manejar el caso de pocas muestras para el split
        test_size = 0.2
        if X.shape[0] * test_size < 1:
            test_size = 0 # No hacer split si no hay suficientes datos para el test set
            st.warning("Pocos datos, usando todo para entrenamiento.")
            X_train, X_test, y_train, y_test = X, X, y, y # Simplificación para evitar error
        elif len(y.unique()) == 1:
             st.warning("Solo se detectó una categoría. El modelo predecirá siempre esa categoría.")
             X_train, X_test, y_train, y_test = X, X, y, y # No se puede estratificar
        else:
             # Intentar estratificar si hay más de una clase y suficientes muestras
             try:
                 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
             except ValueError:
                 st.warning("No se pudo estratificar el split. Usando split normal.")
                 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)


        model = MultinomialNB()
        model.fit(X_train, y_train)

        # Generar reporte solo si hay datos de test
        report_string = "Entrenamiento completado."
        if test_size > 0 and X_test.shape[0] > 0:
            try:
                y_pred = model.predict(X_test)
                # Obtener las etiquetas presentes en y_test y y_pred para el reporte
                labels = sorted(list(set(y_test) | set(y_pred)))
                report_string = classification_report(y_test, y_pred, labels=labels, zero_division=0)
            except Exception as report_error:
                report_string = f"Modelo entrenado, pero error al generar reporte: {report_error}"


        return model, vectorizer, report_string

    except Exception as e:
        st.error(f"Error durante el entrenamiento del modelo: {e}")
        return None, None, f"Error en entrenamiento: {e}"

# --- Interfaz de Streamlit ---
st.set_page_config(layout="wide") # Aprovechar mejor el espacio
st.title("📊 Categorizador de Transacciones Bancarias")
st.write("""
Sube tu archivo CSV de gastos (con columnas como CONCEPTO, COMERCIO, CATEGORÍA, SUBCATEGORIA)
para extraer información y entrenar un modelo básico de categorización.
""")

uploaded_file = st.file_uploader("Carga tu archivo Gastos.csv", type="csv")

if uploaded_file is not None:
    df = load_and_process_data(uploaded_file)

    if df is not None:
        st.success("Archivo cargado y procesado correctamente!")
        st.dataframe(df.head()) # Mostrar una vista previa

        # Extraer y mostrar base de conocimiento
        categorias, subcategorias_por_categoria, comercios_por_categoria = extract_knowledge(df)

        col1, col2 = st.columns(2) # Crear columnas para mejor distribución

        with col1:
            st.header("📚 Base de Conocimiento Extraída")
            st.subheader("Categorías Detectadas:")
            if categorias:
                st.write(categorias)
            else:
                st.warning("No se encontraron categorías válidas.")

            st.subheader("Comercios por Categoría:")
            with st.expander("Ver Comercios Aprendidos"):
                if comercios_por_categoria:
                    st.json(comercios_por_categoria, expanded=False)
                else:
                     st.warning("No se encontraron comercios válidos asociados a categorías.")


        with col2:
            st.subheader("Subcategorías por Categoría:")
            if subcategorias_por_categoria:
                # Usar expanders para que no ocupe tanto espacio si hay muchas categorías
                for cat, subcats in subcategorias_por_categoria.items():
                    with st.expander(f"{cat.capitalize()}"):
                        if subcats:
                           st.write(subcats)
                        else:
                           st.write("_Sin subcategorías específicas encontradas_")
            else:
                st.warning("No se encontraron subcategorías válidas.")


        st.divider() # Separador visual

        # Entrenar y mostrar resultados del modelo
        st.header("🤖 Aprendizaje Automático")
        st.write("Entrenando un modelo simple (Naive Bayes) para predecir 'CATEGORÍA' basado en 'CONCEPTO' y 'COMERCIO'.")

        model, vectorizer, report_string = train_model(df.copy()) # Pasar copia por si acaso

        if model and vectorizer:
            st.subheader("Informe de Clasificación (Evaluación del Modelo)")
            st.text(report_string)
            st.info("Este informe muestra qué tan bien el modelo predice las categorías en una porción de los datos que no usó para aprender (conjunto de prueba).")

            st.divider()

            # Sección para probar la categorización
            st.header("🧪 Probar Categorización")
            st.write("Introduce los detalles de una nueva transacción para obtener una predicción de categoría.")

            new_concepto = st.text_input("Nuevo Concepto:")
            new_comercio = st.text_input("Nuevo Comercio (opcional):")

            if st.button("Categorizar Nueva Transacción"):
                if not new_concepto:
                    st.warning("Por favor, introduce al menos el concepto.")
                else:
                    texto_nuevo = new_concepto.lower().strip() + ' ' + new_comercio.lower().strip()
                    try:
                        vector_nuevo = vectorizer.transform([texto_nuevo])
                        prediccion = model.predict(vector_nuevo)
                        probabilidades = model.predict_proba(vector_nuevo)
                        clases_ordenadas = model.classes_

                        st.subheader("Predicción:")
                        st.success(f"**Categoría Predicha:** {prediccion[0].capitalize()}")

                        # Mostrar probabilidades (opcional, pero útil)
                        st.subheader("Probabilidades por Categoría:")
                        probs_df = pd.DataFrame(probabilidades, columns=clases_ordenadas).T
                        probs_df.columns = ['Probabilidad']
                        probs_df = probs_df.sort_values(by='Probabilidad', ascending=False)
                        st.dataframe(probs_df.style.format("{:.2%}"))

                    except Exception as e:
                        st.error(f"Error durante la predicción: {e}")
        else:
            st.error("No se pudo entrenar el modelo. Revisa los datos o los mensajes de advertencia anteriores.")

    else:
        st.warning("Esperando la carga del archivo CSV...")

else:
     st.info("Por favor, carga un archivo CSV para comenzar.")

st.sidebar.header("Acerca de")
st.sidebar.info(
    "Esta aplicación analiza un archivo CSV de gastos, extrae categorías, "
    "subcategorías y comercios, y entrena un modelo básico para "
    "predecir categorías de nuevas transacciones."
)
