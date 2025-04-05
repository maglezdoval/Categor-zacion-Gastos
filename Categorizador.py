import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import io
import numpy as np # Import numpy for handling potential NaN issues after conversion

# --- Session State Initialization ---
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
if 'training_features' not in st.session_state:
     st.session_state.training_features = [] # Track features used in training text

# --- Functions (Cached) ---

@st.cache_data
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

        # --- Basic Cleaning & Column Handling ---
        df.columns = [col.upper().strip() for col in df.columns]

        # --- Define Required Columns based on data type ---
        if is_training_data:
            # Training data needs these for learning structure and model
            required_cols = ['CONCEPTO', 'CATEGORÍA', 'SUBCATEGORIA', 'IMPORTE', 'AÑO', 'MES', 'DIA']
            # COMERCIO is highly recommended for better training but check if exists
            if 'COMERCIO' not in df.columns:
                st.warning("Columna 'COMERCIO' no encontrada en datos de entrenamiento. El modelo se entrenará solo con 'CONCEPTO'.")
        else:
            # New data MUST have these for processing and basic context
            required_cols = ['CONCEPTO', 'IMPORTE', 'AÑO', 'MES', 'DIA']
            # COMERCIO is optional for new data

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Faltan columnas esenciales en el archivo {'de entrenamiento' if is_training_data else 'nuevo'}: {', '.join(missing_cols)}")
            return None

        # --- Type Conversion & Cleaning ---
        # IMPORTE (mandatory for both now)
        if 'IMPORTE' in df.columns:
             try:
                 # Ensure it's string first, replace comma, then convert to float
                 df['IMPORTE'] = df['IMPORTE'].astype(str).str.replace(',', '.', regex=False)
                 # Attempt conversion to numeric, coercing errors to NaN
                 df['IMPORTE'] = pd.to_numeric(df['IMPORTE'], errors='coerce')
                 # Check if any NaNs were produced (indicates conversion failure)
                 if df['IMPORTE'].isnull().any():
                      st.warning("Algunos valores en 'IMPORTE' no pudieron ser convertidos a número y serán tratados como NaN.")
             except Exception as e:
                 st.error(f"Error crítico al procesar la columna 'IMPORTE': {e}")
                 return None # Stop processing if 'IMPORTE' fails critically
        else:
            st.error("¡Falta la columna IMPORTE!") # Should have been caught earlier, but double-check
            return None


        # FECHA columns (mandatory for both)
        date_cols = ['AÑO', 'MES', 'DIA']
        for col in date_cols:
             if col in df.columns:
                try:
                    # Convert to numeric, coerce errors, then fill potential NaNs with 0/1 and convert to int
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # Fill NaN with a default value (e.g., 0 for year/month/day is problematic, maybe 1 for day/month?)
                    # A better approach might be to drop rows with invalid dates if needed.
                    # For now, let's warn and fill with 0, then convert to Int64 to handle NaN during conversion phase.
                    if df[col].isnull().any():
                         st.warning(f"Valores no numéricos encontrados en '{col}'. Se intentará rellenar con 0.")
                         df[col] = df[col].fillna(0)
                    df[col] = df[col].astype(int) # Convert to standard integer after handling NaN
                except Exception as e:
                    st.error(f"Error al procesar la columna de fecha '{col}': {e}")
                    return None
             else:
                  st.error(f"¡Falta la columna de fecha {col}!") # Should be caught by required_cols check
                  return None


        # TEXT columns (CONCEPTO mandatory, others optional/training-specific)
        text_cols_to_process = ['CONCEPTO']
        if 'COMERCIO' in df.columns:
            text_cols_to_process.append('COMERCIO')
        if is_training_data:
            if 'CATEGORÍA' in df.columns: text_cols_to_process.append('CATEGORÍA')
            if 'SUBCATEGORIA' in df.columns: text_cols_to_process.append('SUBCATEGORIA')

        for col in text_cols_to_process:
             if col in df.columns:
                 df[col] = df[col].fillna('').astype(str).str.lower().str.strip()


        # --- Feature Engineering for Model ---
        if is_training_data:
            # Filter rows for training (must have category)
            df = df.dropna(subset=['CATEGORÍA'])
            df = df[df['CATEGORÍA'] != '']
            if df.empty:
                st.error("El archivo de entrenamiento no tiene filas válidas con categorías después de la limpieza.")
                return None

            # Create training text: CONCEPTO + COMERCIO (if available)
            if 'COMERCIO' in df.columns:
                df['TEXTO_ENTRENAMIENTO'] = df['CONCEPTO'] + ' ' + df['COMERCIO']
                st.session_state.training_features = ['CONCEPTO', 'COMERCIO'] # Record that COMERCIO was used
            else:
                df['TEXTO_ENTRENAMIENTO'] = df['CONCEPTO'] # Only CONCEPTO used
                st.session_state.training_features = ['CONCEPTO']
            df['TEXTO_ENTRENAMIENTO'] = df['TEXTO_ENTRENAMIENTO'].str.strip()

        else: # Processing for new data
            # Create prediction text: Use COMERCIO only if it exists and was used in training
            if 'COMERCIO' in df.columns and 'COMERCIO' in st.session_state.get('training_features', []):
                 df['TEXTO_A_CATEGORIZAR'] = df['CONCEPTO'] + ' ' + df['COMERCIO']
            else:
                 df['TEXTO_A_CATEGORIZAR'] = df['CONCEPTO'] # Fallback to only CONCEPTO
            df['TEXTO_A_CATEGORIZAR'] = df['TEXTO_A_CATEGORIZAR'].str.strip()


        return df

    except Exception as e:
        st.error(f"Error general al cargar o procesar el archivo CSV: {e}")
        return None

# --- extract_knowledge and train_classifier functions remain largely the same ---
# (Make sure train_classifier uses 'TEXTO_ENTRENAMIENTO')
@st.cache_data
def extract_knowledge(df):
    """Extrae categorías, subcategorías y comercios del DataFrame de entrenamiento."""
    knowledge = {'categorias': [], 'subcategorias': {}, 'comercios': {}}
    if df is None or 'CATEGORÍA' not in df.columns:
        return knowledge

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
    """Entrena el modelo de clasificación usando TEXTO_ENTRENAMIENTO."""
    report = "Modelo no entrenado."
    model = None
    vectorizer = None

    if df is None or df.empty or 'TEXTO_ENTRENAMIENTO' not in df.columns or 'CATEGORÍA' not in df.columns:
        st.warning("No hay datos válidos (TEXTO_ENTRENAMIENTO, CATEGORÍA) para entrenar.")
        return model, vectorizer, report

    df_train = df.dropna(subset=['TEXTO_ENTRENAMIENTO', 'CATEGORÍA'])
    df_train = df_train[df_train['CATEGORÍA'] != '']

    if df_train.empty or len(df_train['CATEGORÍA'].unique()) < 2:
        st.warning("Datos insuficientes o pocas categorías (<2) para entrenar.")
        return model, vectorizer, report

    try:
        X = df_train['TEXTO_ENTRENAMIENTO'] # Use the combined training text
        y = df_train['CATEGORÍA']

        unique_classes = y.unique()
        test_available = False
        if len(unique_classes) > 1 and len(y) > 5:
            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                test_available = True
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                test_available = True # Still split, just not stratified
        else:
            X_train, y_train = X, y
            X_test, y_test = pd.Series(dtype='str'), pd.Series(dtype='str')
            st.info("Usando todos los datos para entrenar (sin evaluación detallada).")

        vectorizer = TfidfVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)

        model = MultinomialNB()
        model.fit(X_train_vec, y_train)

        if test_available and not X_test.empty:
            try:
                X_test_vec = vectorizer.transform(X_test)
                y_pred = model.predict(X_test_vec)
                labels = sorted(list(set(y_test) | set(y_pred)))
                report = classification_report(y_test, y_pred, labels=labels, zero_division=0)
            except Exception as report_err:
                st.warning(f"Error generando reporte de clasificación: {report_err}")
                report = "Modelo entrenado, pero no se pudo generar el reporte."
        else:
            report = "Modelo entrenado con todos los datos disponibles."

        st.success("¡Modelo entrenado exitosamente!")

    except Exception as e:
        st.error(f"Error durante el entrenamiento: {e}")
        report = f"Error en entrenamiento: {e}"

    return model, vectorizer, report


# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("📊 Categorizador de Transacciones Bancarias v3")

st.info("💡 **Instrucciones:** Primero carga tu archivo CSV histórico **ya categorizado** (como `Gastos.csv`) para entrenar el sistema. Luego, podrás cargar un **nuevo** archivo CSV (sin categorizar) para que la aplicación le asigne categorías.")

st.header("Paso 1: Entrenar con Datos Históricos")
st.write("Sube tu archivo CSV **con categorías ya asignadas** (columnas requeridas: `CONCEPTO`, `CATEGORÍA`, `SUBCATEGORIA`, `IMPORTE`, `AÑO`, `MES`, `DIA`). Se recomienda incluir `COMERCIO` para mejor precisión.")

uploaded_training_file = st.file_uploader("Carga tu archivo CSV de ENTRENAMIENTO", type="csv", key="trainer_v3")

if uploaded_training_file:
    df_train = load_and_process_data(uploaded_training_file, is_training_data=True)

    if df_train is not None:
        st.sidebar.success("Archivo de entrenamiento cargado.")
        # Extract knowledge and store it
        st.session_state.knowledge = extract_knowledge(df_train)

        # Display knowledge in sidebar
        st.sidebar.subheader("Conocimiento Aprendido")
        with st.sidebar.expander("Ver Categorías"):
            st.write(st.session_state.knowledge['categorias'])
        with st.sidebar.expander("Ver Comercios"):
             st.json(st.session_state.knowledge['comercios'], expanded=False)

        # Train model and store results
        st.session_state.model, st.session_state.vectorizer, st.session_state.report_string = train_classifier(df_train.copy())

        if st.session_state.model and st.session_state.vectorizer:
            st.session_state.model_trained = True
            st.sidebar.subheader("Resultado Entrenamiento")
            with st.sidebar.expander("Ver Informe de Clasificación", expanded=False):
                 st.text(st.session_state.report_string)
        else:
            st.session_state.model_trained = False
            st.sidebar.error("Fallo en el entrenamiento del modelo.")
            st.session_state.report_string = "Entrenamiento fallido." # Update status

# Separator
st.divider()

st.header("Paso 2: Categorizar Nuevo Archivo")

# Check if model is ready before allowing upload of new file
if not st.session_state.model_trained:
    st.warning("⚠️ Primero debes cargar y procesar un archivo de entrenamiento válido en el Paso 1.")
else:
    st.write("Sube tu **nuevo** archivo CSV (columnas requeridas: `CONCEPTO`, `IMPORTE`, `AÑO`, `MES`, `DIA`. La columna `COMERCIO` es opcional pero ayuda si existe).")
    uploaded_new_file = st.file_uploader("Carga tu archivo CSV NUEVO a categorizar", type="csv", key="categorizer_v3")

    if uploaded_new_file:
        df_new = load_and_process_data(uploaded_new_file, is_training_data=False)

        if df_new is not None and 'TEXTO_A_CATEGORIZAR' in df_new.columns:
            st.success("Archivo nuevo cargado.")
            st.subheader("Vista Previa del Archivo Nuevo (Antes)")
            st.dataframe(df_new.head())

            try:
                # Apply the trained vectorizer and model from session state
                X_new_vec = st.session_state.vectorizer.transform(df_new['TEXTO_A_CATEGORIZAR'])
                predictions = st.session_state.model.predict(X_new_vec)

                # Add the prediction column
                df_new['CATEGORIA_PREDICHA'] = predictions
                # Capitalize for display
                df_new['CATEGORIA_PREDICHA'] = df_new['CATEGORIA_PREDICHA'].str.capitalize()

                # --- (Optional) Predict Subcategory (Simple Placeholder) ---
                # This needs a separate, potentially more complex model or rule-based system
                # For now, we focus on the main category prediction.
                # df_new['SUBCATEGORIA_PREDICHA'] = "Pendiente"

                st.subheader("Resultado de la Categorización")
                # Select columns to display, putting predicted category prominently
                display_cols = ['CATEGORIA_PREDICHA'] + [col for col in df_new.columns if col != 'CATEGORIA_PREDICHA' and col != 'TEXTO_A_CATEGORIZAR']
                st.dataframe(df_new[display_cols])

                # Download Button
                csv_output = df_new.to_csv(index=False, sep=';', decimal=',').encode('utf-8') # Use comma decimal for Excel Spain
                st.download_button(
                    label="📥 Descargar CSV Categorizado",
                    data=csv_output,
                    file_name='gastos_categorizados.csv',
                    mime='text/csv',
                )

            except Exception as e:
                st.error(f"Error al aplicar la categorización: {e}")
                st.error("Verifica que el archivo nuevo tenga las columnas requeridas (`CONCEPTO`, `IMPORTE`, `AÑO`, `MES`, `DIA`) y un formato consistente.")

        elif df_new is not None:
             st.error("No se encontró la columna combinada 'TEXTO_A_CATEGORIZAR' necesaria para la predicción después del procesamiento.")

# Sidebar Info
st.sidebar.header("Acerca de")
st.sidebar.info(
    "Esta aplicación aprende de tus gastos históricos categorizados y luego aplica ese aprendizaje para sugerir categorías a nuevas transacciones."
)
