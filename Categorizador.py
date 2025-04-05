import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import io

# --- Session State Initialization ---
# To store the trained model and related objects between runs
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'knowledge' not in st.session_state:
    st.session_state.knowledge = {'categorias': [], 'subcategorias': {}, 'comercios': {}}
if 'report_string' not in st.session_state:
    st.session_state.report_string = ""

# --- Functions (Cached) ---

@st.cache_data # Cache the result of loading and processing
def load_and_process_data(uploaded_file, is_training_data=True):
    """Carga y preprocesa los datos desde un archivo subido."""
    if uploaded_file is None:
        return None
    try:
        bytes_data = uploaded_file.getvalue()
        try:
            df = pd.read_csv(io.BytesIO(bytes_data), encoding='utf-8', sep=';')
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(bytes_data), encoding='latin1', sep=';')

        # --- Basic Cleaning ---
        # Convert column names to uppercase for consistency
        df.columns = [col.upper().strip() for col in df.columns]

        # Check for essential columns
        required_cols = ['CONCEPTO']
        if is_training_data:
            required_cols.extend(['CATEGORÍA', 'SUBCATEGORIA', 'IMPORTE'])
        else: # For new data, 'COMERCIO' is needed if model used it
             if 'COMERCIO' in st.session_state.get('required_features',[]): # Check if COMERCIO was used
                  required_cols.append('COMERCIO')


        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Faltan columnas esenciales en el archivo {'de entrenamiento' if is_training_data else 'nuevo'}: {', '.join(missing_cols)}")
            return None

        # Handle 'IMPORTE' only if it's training data (or present in new)
        if 'IMPORTE' in df.columns:
            try:
                df['IMPORTE'] = df['IMPORTE'].astype(str).str.replace(',', '.', regex=False).astype(float)
            except Exception as e:
                st.warning(f"No se pudo convertir 'IMPORTE': {e}. Se continuará sin ella si no es necesaria.")
                # Decide si eliminarla o dejarla tal cual si no es crucial
                # df = df.drop(columns=['IMPORTE']) # Opcional: eliminarla si causa problemas

        # Handle Text Columns (including optional 'COMERCIO')
        text_cols_base = ['CONCEPTO']
        if 'COMERCIO' in df.columns:
            text_cols_base.append('COMERCIO')
        if is_training_data:
            text_cols_base.extend(['CATEGORÍA', 'SUBCATEGORIA'])

        for col in text_cols_base:
            if col in df.columns:
                df[col] = df[col].fillna('').astype(str).str.lower().str.strip()

        # --- Processing Specific to Data Type ---
        if is_training_data:
            df = df.dropna(subset=['CATEGORÍA']) # Crucial for training
            df = df[df['CATEGORÍA'] != '']
            if df.empty:
                st.error("El archivo de entrenamiento no contiene filas válidas con categorías después de la limpieza.")
                return None
            # Combine features for training text
            df['TEXTO_ENTRENAMIENTO'] = df['CONCEPTO'] + ' ' + df.get('COMERCIO', '') # Use .get for optional COMERCIO


        else: # Processing for new data
            # Combine features for prediction using the same logic as training
            df['TEXTO_A_CATEGORIZAR'] = df['CONCEPTO'] + ' ' + df.get('COMERCIO', '') # Use .get

        return df

    except Exception as e:
        st.error(f"Error al cargar o procesar el archivo CSV: {e}")
        return None

@st.cache_data
def extract_knowledge(df):
    """Extrae categorías, subcategorías y comercios del DataFrame de entrenamiento."""
    knowledge = {'categorias': [], 'subcategorias': {}, 'comercios': {}}
    if df is None or 'CATEGORÍA' not in df.columns:
        return knowledge

    # Use .dropna() before unique() on the relevant columns
    knowledge['categorias'] = sorted([cat for cat in df['CATEGORÍA'].dropna().unique() if cat])

    for cat in knowledge['categorias']:
        # Subcategorías
        subcat_col = 'SUBCATEGORIA'
        if subcat_col in df.columns:
             subcats_series = df.loc[df['CATEGORÍA'] == cat, subcat_col].dropna()
             subcats_series = subcats_series[subcats_series != '']
             knowledge['subcategorias'][cat] = sorted(subcats_series.unique().tolist())
        else:
             knowledge['subcategorias'][cat] = []


        # Comercios
        comercio_col = 'COMERCIO'
        if comercio_col in df.columns:
             comers_series = df.loc[df['CATEGORÍA'] == cat, comercio_col].dropna()
             comers_series = comers_series[comers_series.str.strip() != '']
             comers_list = sorted([com for com in comers_series.unique() if com != 'n/a'])
             knowledge['comercios'][cat] = comers_list
        else:
            knowledge['comercios'][cat] = []


    return knowledge

@st.cache_resource # Cache model and vectorizer objects
def train_classifier(df):
    """Entrena el modelo de clasificación."""
    report = "Modelo no entrenado."
    model = None
    vectorizer = None

    if df is None or df.empty or 'TEXTO_ENTRENAMIENTO' not in df.columns or 'CATEGORÍA' not in df.columns:
        st.warning("No hay datos válidos para entrenar.")
        return model, vectorizer, report

    df_train = df.dropna(subset=['TEXTO_ENTRENAMIENTO', 'CATEGORÍA'])
    df_train = df_train[df_train['CATEGORÍA'] != ''] # Asegurar categorías no vacías

    if df_train.empty or len(df_train['CATEGORÍA'].unique()) < 2:
        st.warning("Datos insuficientes o pocas categorías (<2) para entrenar.")
        return model, vectorizer, report

    try:
        X = df_train['TEXTO_ENTRENAMIENTO']
        y = df_train['CATEGORÍA']

        # Split data if enough samples and classes exist
        unique_classes = y.unique()
        if len(unique_classes) > 1 and len(y) > 5: # Heurística simple para split
            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            except ValueError: # Si no se puede estratificar
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            test_available = True
        else:
            X_train, y_train = X, y # Usar todo para entrenar
            X_test, y_test = pd.Series(dtype='str'), pd.Series(dtype='str') # Vacío para evitar error en reporte
            test_available = False
            st.info("Pocos datos o categorías, usando todo para entrenar (sin evaluación detallada).")

        vectorizer = TfidfVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)

        model = MultinomialNB()
        model.fit(X_train_vec, y_train)

        if test_available and not X_test.empty:
            X_test_vec = vectorizer.transform(X_test)
            y_pred = model.predict(X_test_vec)
            labels = sorted(list(set(y_test) | set(y_pred)))
            report = classification_report(y_test, y_pred, labels=labels, zero_division=0)
        elif test_available and X_test.empty:
             report = "Modelo entrenado, pero no hay datos suficientes para un conjunto de prueba."
        else:
            report = "Modelo entrenado con todos los datos disponibles."

        st.success("¡Modelo entrenado exitosamente!")
        # Guardar qué columnas se usaron implícitamente para el texto
        st.session_state.required_features = ['CONCEPTO']
        if 'COMERCIO' in df.columns:
            st.session_state.required_features.append('COMERCIO')


    except Exception as e:
        st.error(f"Error durante el entrenamiento: {e}")
        report = f"Error en entrenamiento: {e}"

    return model, vectorizer, report

# --- Interfaz de Streamlit ---
st.set_page_config(layout="wide")
st.title("📊 Categorizador de Transacciones Bancarias v2")

st.write("""
**Paso 1:** Sube tu archivo CSV **categorizado** (`Gastos.csv`) para aprender las categorías y entrenar el modelo.
""")

uploaded_training_file = st.file_uploader("Carga tu archivo CSV de ENTRENAMIENTO", type="csv", key="trainer")

if uploaded_training_file:
    df_train = load_and_process_data(uploaded_training_file, is_training_data=True)

    if df_train is not None:
        st.sidebar.success("Archivo de entrenamiento cargado.")
        st.session_state.knowledge = extract_knowledge(df_train)

        st.sidebar.subheader("Conocimiento Aprendido")
        with st.sidebar.expander("Ver Categorías"):
            st.write(st.session_state.knowledge['categorias'])
        with st.sidebar.expander("Ver Comercios"):
             st.json(st.session_state.knowledge['comercios'], expanded=False)
        # No mostrar subcategorías aquí para no saturar

        # Entrenar el modelo y guardar en session_state
        st.session_state.model, st.session_state.vectorizer, st.session_state.report_string = train_classifier(df_train.copy())

        if st.session_state.model and st.session_state.vectorizer:
            st.session_state.model_trained = True
            st.sidebar.subheader("Resultado Entrenamiento")
            st.sidebar.text(st.session_state.report_string)
        else:
            st.session_state.model_trained = False
            st.sidebar.error("Fallo en el entrenamiento del modelo.")

st.divider()

st.write("""
**Paso 2:** Una vez entrenado el modelo (ver barra lateral), sube tu **nuevo** archivo CSV (sin categorizar) para aplicar la categorización.
""")

# Solo mostrar el segundo uploader si el modelo está listo
if st.session_state.model_trained:
    uploaded_new_file = st.file_uploader("Carga tu archivo CSV NUEVO a categorizar", type="csv", key="categorizer")

    if uploaded_new_file:
        df_new = load_and_process_data(uploaded_new_file, is_training_data=False)

        if df_new is not None and 'TEXTO_A_CATEGORIZAR' in df_new.columns:
            st.success("Archivo nuevo cargado.")
            st.subheader("Vista Previa del Archivo Nuevo (Antes)")
            st.dataframe(df_new.head())

            try:
                # Aplicar el vectorizador y el modelo entrenados
                X_new_vec = st.session_state.vectorizer.transform(df_new['TEXTO_A_CATEGORIZAR'])
                predictions = st.session_state.model.predict(X_new_vec)

                # Añadir la columna de predicciones
                df_new['CATEGORIA_PREDICHA'] = predictions
                 # Opcional: Capitalizar para mejor visualización
                df_new['CATEGORIA_PREDICHA'] = df_new['CATEGORIA_PREDICHA'].str.capitalize()


                # --- (Opcional/Simple) Lógica básica para Subcategoría ---
                # Aquí podrías añadir una lógica más simple si quieres un placeholder
                # Por ejemplo, buscar la primera subcategoría conocida para la categoría predicha
                def get_first_subcategory(predicted_cat):
                    known_subcats = st.session_state.knowledge['subcategorias'].get(predicted_cat.lower(), [])
                    return known_subcats[0].capitalize() if known_subcats else "Desconocida"

                # Aplicar la función (esto es muy básico, ¡cuidado con la precisión!)
                # df_new['SUBCATEGORIA_PREDICHA'] = df_new['CATEGORIA_PREDICHA'].apply(get_first_subcategory)
                # Por ahora, nos centramos en la categoría principal

                st.subheader("Resultado de la Categorización")
                st.dataframe(df_new)

                # Botón de descarga
                st.download_button(
                    label="📥 Descargar CSV Categorizado",
                    data=df_new.to_csv(index=False, sep=';').encode('utf-8'), # UTF-8 es más estándar
                    file_name='gastos_categorizados.csv',
                    mime='text/csv',
                )

            except Exception as e:
                st.error(f"Error al aplicar la categorización: {e}")
                st.error("Asegúrate de que el archivo nuevo tenga las columnas 'CONCEPTO' y 'COMERCIO' (si se usó en el entrenamiento) con un formato similar al de entrenamiento.")

        elif df_new is not None:
             st.error("No se encontró la columna combinada 'TEXTO_A_CATEGORIZAR' en el archivo nuevo después del procesamiento.")


else:
    st.warning("Primero carga y procesa el archivo CSV de entrenamiento en el Paso 1.")
