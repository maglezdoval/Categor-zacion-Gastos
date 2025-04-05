import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import io
import numpy as np
from datetime import datetime # Para manejo de fechas más complejo si fuera necesario

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
if 'processed_training_data' not in st.session_state:
     st.session_state.processed_training_data = pd.DataFrame() # Para acumular datos estandarizados
if 'categorized_data_list' not in st.session_state:
     st.session_state.categorized_data_list = [] # Para guardar resultados de categorización


# --- CONSTANTES ---
# Nombres estándar de columnas internas
CONCEPTO_STD = 'CONCEPTO_STD'
COMERCIO_STD = 'COMERCIO_STD'
IMPORTE_STD = 'IMPORTE_STD'
AÑO_STD = 'AÑO'
MES_STD = 'MES'
DIA_STD = 'DIA'
CATEGORIA_STD = 'CATEGORIA_STD'
SUBCATEGORIA_STD = 'SUBCATEGORIA_STD'
TEXTO_MODELO = 'TEXTO_MODELO' # Columna combinada para el modelo
CATEGORIA_PREDICHA = 'CATEGORIA_PREDICHA'

# --- Funciones de Parseo Específicas por Banco ---

def parse_standard_format(df, is_training_data=True):
    """
    Parsea el formato encontrado en el archivo Gastos.csv (Santander, EVO, Wizink, Amex según el ejemplo).
    Devuelve un DataFrame estandarizado o None si hay error.
    """
    try:
        df.columns = [col.upper().strip() for col in df.columns]
        required_base = ['CONCEPTO', 'IMPORTE', 'AÑO', 'MES', 'DIA']
        required_training = ['CATEGORÍA', 'SUBCATEGORIA'] if is_training_data else []
        required_cols = required_base + required_training

        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"Formato Estándar: Faltan columnas requeridas: {', '.join(missing_cols)}")
            return None

        df_std = pd.DataFrame()

        # --- Extracción y Estandarización ---
        df_std[CONCEPTO_STD] = df['CONCEPTO'].fillna('').astype(str).str.lower().str.strip()
        # Usar .get para COMERCIO ya que puede no estar siempre (aunque en el formato std sí está)
        df_std[COMERCIO_STD] = df.get('COMERCIO', pd.Series(dtype=str)).fillna('').astype(str).str.lower().str.strip()

        # Importe
        try:
            df_std[IMPORTE_STD] = pd.to_numeric(df['IMPORTE'].astype(str).str.replace(',', '.', regex=False), errors='coerce')
        except Exception as e:
            st.error(f"Error convirtiendo IMPORTE en formato estándar: {e}")
            return None

        # Fecha
        for col in ['AÑO', 'MES', 'DIA']:
            try:
                df_std[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            except Exception as e:
                st.error(f"Error convirtiendo columna de fecha '{col}' en formato estándar: {e}")
                return None

        # Categorías (solo para entrenamiento)
        if is_training_data:
            df_std[CATEGORIA_STD] = df['CATEGORÍA'].fillna('').astype(str).str.lower().str.strip()
            df_std[SUBCATEGORIA_STD] = df['SUBCATEGORIA'].fillna('').astype(str).str.lower().str.strip()
            # Filtrar filas sin categoría válida para entrenamiento
            df_std = df_std[df_std[CATEGORIA_STD] != '']
            if df_std.empty:
                 st.warning("Formato Estándar: No se encontraron filas con categorías válidas para entrenamiento.")
                 # Return None o un DF vacío dependiendo de cómo quieras manejarlo arriba
                 return pd.DataFrame() # Devolver DF vacío para concatenar sin error

        # Combinar texto para el modelo
        df_std[TEXTO_MODELO] = df_std[CONCEPTO_STD] + ' ' + df_std[COMERCIO_STD]
        df_std[TEXTO_MODELO] = df_std[TEXTO_MODELO].str.strip()


        # Mantener otras columnas originales si existen (opcional)
        original_cols_to_keep = [c for c in df.columns if c.upper() not in [
            'CONCEPTO','COMERCIO','IMPORTE','AÑO','MES','DIA',
            'CATEGORÍA','SUBCATEGORIA','CATEGORIA','SUBCATEGORIA']] # Añadir alias si es necesario
        for col in original_cols_to_keep:
             df_std[f"ORIG_{col}"] = df[col] # Prefijo para evitar colisiones


        return df_std

    except Exception as e:
        st.error(f"Error inesperado parseando formato estándar: {e}")
        return None

# --- Aquí añadirías las funciones parse_evo, parse_wizink, parse_amex ---
# --- si tuvieran formatos *REALMENTE* diferentes. Por ahora, asumimos ---
# --- que todos los ejemplos en Gastos.csv usan el mismo formato. ---

# Ejemplo placeholder (si EVO fuera diferente):
# def parse_evo(df, is_training_data=True):
#     st.info("Parseando formato específico de EVO...")
#     # ... lógica específica para columnas de EVO ...
#     df_std = pd.DataFrame()
#     # ... mapear columnas de EVO a df_std[CONCEPTO_STD], df_std[IMPORTE_STD], etc. ...
#     # ... manejar fechas, importes, etc. como en parse_standard_format ...
#     # ... crear df_std[TEXTO_MODELO] ...
#     return df_std

# --- Función Principal de Carga y Estandarización ---

# @st.cache_data # Cachear puede ser complejo con múltiples archivos y bancos
def load_and_standardize(uploaded_file_info, bank_type, is_training_data=True):
    """
    Lee un archivo subido, llama al parser correcto y devuelve el DF estandarizado.
    uploaded_file_info es un objeto File Uploader de Streamlit.
    """
    file_name = uploaded_file_info.name
    st.write(f"Procesando archivo: {file_name} (Banco: {bank_type})")

    try:
        bytes_data = uploaded_file_info.getvalue()
        try:
            # Intentar leer con diferentes encodings y separadores si es necesario
            df_raw = pd.read_csv(io.BytesIO(bytes_data), encoding='utf-8', sep=';')
        except UnicodeDecodeError:
            df_raw = pd.read_csv(io.BytesIO(bytes_data), encoding='latin1', sep=';')
        except Exception as read_err: # Captura otros errores de lectura
             st.error(f"No se pudo leer el CSV '{file_name}'. ¿Está bien formado y usa ';' como separador? Error: {read_err}")
             return None

        # Llamar a la función de parseo específica
        if bank_type == "SANTANDER" or bank_type == "EVO" or bank_type == "WIZINK" or bank_type == "AMEX":
            # En este caso, todos usan el mismo parser basado en Gastos.csv
            df_standard = parse_standard_format(df_raw.copy(), is_training_data)
        # elif bank_type == "OTRO_BANCO":
        #     df_standard = parse_otro_banco(df_raw.copy(), is_training_data)
        else:
            st.error(f"Tipo de banco '{bank_type}' no reconocido o sin parser definido.")
            return None

        if df_standard is None:
            st.error(f"Fallo al parsear el archivo: {file_name}")
            return None
        elif df_standard.empty and is_training_data:
             st.warning(f"El archivo de entrenamiento {file_name} no produjo datos válidos tras el parseo.")
             return df_standard # Devolver vacío para que no rompa la concatenación
        elif df_standard.empty and not is_training_data:
             st.error(f"El archivo nuevo {file_name} no produjo datos válidos tras el parseo.")
             return None

        st.success(f"Archivo '{file_name}' parseado y estandarizado.")
        return df_standard

    except Exception as e:
        st.error(f"Error procesando '{file_name}': {e}")
        return None


# --- Funciones de ML (extract_knowledge, train_classifier) ---
#     (Estas funciones ahora operan sobre el DataFrame ESTANDARIZADO
#      usando las columnas _STD y TEXTO_MODELO)

@st.cache_data
def extract_knowledge_std(df_std):
    knowledge = {'categorias': [], 'subcategorias': {}, 'comercios': {}}
    if df_std is None or CATEGORIA_STD not in df_std.columns or df_std.empty:
        return knowledge

    knowledge['categorias'] = sorted([cat for cat in df_std[CATEGORIA_STD].dropna().unique() if cat])

    for cat in knowledge['categorias']:
        subcat_col = SUBCATEGORIA_STD
        if subcat_col in df_std.columns:
             subcats_series = df_std.loc[df_std[CATEGORIA_STD] == cat, subcat_col].dropna()
             knowledge['subcategorias'][cat] = sorted([s for s in subcats_series.unique() if s])
        else:
             knowledge['subcategorias'][cat] = []

        comercio_col = COMERCIO_STD
        if comercio_col in df_std.columns:
             comers_series = df_std.loc[df_std[CATEGORIA_STD] == cat, comercio_col].dropna()
             comers_list = sorted([com for com in comers_series.unique() if com and com != 'n/a'])
             knowledge['comercios'][cat] = comers_list
        else:
            knowledge['comercios'][cat] = []
    return knowledge

@st.cache_resource
def train_classifier_std(df_std):
    report = "Modelo no entrenado."
    model = None
    vectorizer = None

    required_cols = [TEXTO_MODELO, CATEGORIA_STD]
    if df_std is None or df_std.empty or not all(col in df_std.columns for col in required_cols):
        st.warning("No hay datos estandarizados válidos (TEXTO_MODELO, CATEGORIA_STD) para entrenar.")
        return model, vectorizer, report

    df_train = df_std.dropna(subset=required_cols)
    df_train = df_train[df_train[CATEGORIA_STD] != '']

    if df_train.empty or len(df_train[CATEGORIA_STD].unique()) < 2:
        st.warning("Datos insuficientes o pocas categorías (<2) en datos estandarizados para entrenar.")
        return model, vectorizer, report

    try:
        X = df_train[TEXTO_MODELO]
        y = df_train[CATEGORIA_STD] # Usar la categoría estandarizada

        # (Split logic - unchanged conceptually)
        unique_classes = y.unique()
        test_available = False
        if len(unique_classes) > 1 and len(y) > 5:
            try:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                test_available = True
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                test_available = True
        else:
            X_train, y_train = X, y
            X_test, y_test = pd.Series(dtype='str'), pd.Series(dtype='str')
            st.info("Usando todos los datos estandarizados para entrenar.")

        # Train
        vectorizer = TfidfVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        model = MultinomialNB()
        model.fit(X_train_vec, y_train)

        # Report (if possible)
        if test_available and not X_test.empty:
            try:
                X_test_vec = vectorizer.transform(X_test)
                y_pred = model.predict(X_test_vec)
                labels = sorted(list(set(y_test) | set(y_pred)))
                report = classification_report(y_test, y_pred, labels=labels, zero_division=0)
            except Exception as report_err:
                 report = f"Modelo entrenado, error en reporte: {report_err}"
        else:
            report = "Modelo entrenado con todos los datos estandarizados."

        st.success("¡Modelo entrenado con datos estandarizados!")

    except Exception as e:
        st.error(f"Error durante el entrenamiento con datos estandarizados: {e}")
        report = f"Error en entrenamiento: {e}"

    return model, vectorizer, report

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🏦 Categorizador Bancario Multi-Formato")

st.info("""
**Instrucciones:**
1.  **Entrenar:** Selecciona el banco, sube tus archivos CSV **categorizados** de ese banco. Repite si tienes de varios bancos. Luego presiona "Entrenar Modelo".
2.  **Categorizar:** Una vez entrenado, selecciona el banco y sube tus archivos CSV **nuevos (sin categorizar)**. Los resultados aparecerán abajo.
""")

# --- Paso 1: Entrenamiento ---
st.header("Paso 1: Entrenar Modelo con Datos Históricos")

bank_options = ["SANTANDER", "EVO", "WIZINK", "AMEX"] # Añade más bancos aquí si creas parsers
selected_bank_train = st.selectbox("Selecciona el Banco del Archivo de Entrenamiento:", bank_options, key="bank_train")

uploaded_training_files = st.file_uploader(
    f"Carga archivo(s) CSV CATEGORIZADOS de {selected_bank_train}",
    type="csv",
    accept_multiple_files=True,
    key="trainer_multi"
)

# Botón para añadir archivos al conjunto de entrenamiento acumulado
if uploaded_training_files:
    if st.button(f"Añadir Archivos de {selected_bank_train} al Entrenamiento", key="add_train"):
        if 'processed_training_data' not in st.session_state:
            st.session_state.processed_training_data = pd.DataFrame()

        processed_list = []
        for file in uploaded_training_files:
            df_std = load_and_standardize(file, selected_bank_train, is_training_data=True)
            if df_std is not None and not df_std.empty:
                processed_list.append(df_std)

        if processed_list:
            current_data = st.session_state.processed_training_data
            new_data = pd.concat(processed_list, ignore_index=True)
            st.session_state.processed_training_data = pd.concat([current_data, new_data], ignore_index=True)
            st.success(f"{len(processed_list)} archivo(s) de {selected_bank_train} añadidos. Total filas para entrenar: {len(st.session_state.processed_training_data)}")
            # Limpiar uploader para evitar re-procesar al presionar de nuevo accidentalmente
            # st.experimental_rerun() # O una forma más moderna de resetear el uploader si existe

        else:
            st.warning("No se procesaron archivos válidos en esta carga.")

# Mostrar resumen de datos acumulados para entrenar
if not st.session_state.processed_training_data.empty:
    st.write("Datos Acumulados para Entrenamiento (primeras filas):")
    st.dataframe(st.session_state.processed_training_data.head())

    # Botón para entrenar el modelo una vez que hay datos
    if st.button("🧠 Entrenar Modelo con Datos Acumulados", key="train_button"):
        with st.spinner("Entrenando modelo..."):
            df_train_final = st.session_state.processed_training_data.copy()
            # Extraer conocimiento ANTES de entrenar
            st.session_state.knowledge = extract_knowledge_std(df_train_final)

            # Entrenar
            st.session_state.model, st.session_state.vectorizer, st.session_state.report_string = train_classifier_std(df_train_final)

            if st.session_state.model and st.session_state.vectorizer:
                st.session_state.model_trained = True
                st.success("Modelo entrenado y listo para usar.")
                # Mostrar conocimiento y reporte en sidebar
                st.sidebar.success("Modelo Entrenado")
                st.sidebar.subheader("Conocimiento")
                with st.sidebar.expander("Categorías"): st.write(st.session_state.knowledge['categorias'])
                with st.sidebar.expander("Comercios"): st.json(st.session_state.knowledge['comercios'])
                st.sidebar.subheader("Evaluación")
                with st.sidebar.expander("Informe Detallado"): st.text(st.session_state.report_string)
            else:
                st.session_state.model_trained = False
                st.error("Fallo en el entrenamiento.")
                st.sidebar.error("Entrenamiento Fallido")
                st.sidebar.text(st.session_state.report_string) # Mostrar el error
else:
    st.info("Sube archivos de entrenamiento y presiona 'Añadir...' para empezar.")


# --- Paso 2: Categorización ---
st.divider()
st.header("Paso 2: Categorizar Nuevos Archivos")

if not st.session_state.model_trained:
    st.warning("⚠️ El modelo aún no ha sido entrenado (Ver Paso 1).")
else:
    selected_bank_predict = st.selectbox("Selecciona el Banco del Nuevo Archivo:", bank_options, key="bank_predict")

    uploaded_new_files = st.file_uploader(
        f"Carga archivo(s) CSV NUEVOS de {selected_bank_predict} (sin categorizar)",
        type="csv",
        accept_multiple_files=True,
        key="categorizer_multi"
    )

    if uploaded_new_files:
         # Reset previous results when new files are uploaded
        st.session_state.categorized_data_list = []

        all_new_std_dfs = []
        processing_success = True
        for file in uploaded_new_files:
             df_new_std = load_and_standardize(file, selected_bank_predict, is_training_data=False)
             if df_new_std is not None and not df_new_std.empty and TEXTO_MODELO in df_new_std.columns:
                  all_new_std_dfs.append((file.name, df_new_std)) # Guardar nombre y df
             elif df_new_std is not None and (df_new_std.empty or TEXTO_MODELO not in df_new_std.columns):
                  st.warning(f"Archivo '{file.name}' no contiene datos válidos o la columna {TEXTO_MODELO} tras estandarizar. Se omitirá.")
             else:
                  st.error(f"No se pudo procesar el archivo '{file.name}'.")
                  processing_success = False # Marcar que hubo un error

        if all_new_std_dfs and processing_success:
            st.subheader("Resultados de la Categorización:")
            try:
                with st.spinner("Aplicando categorización..."):
                     for file_name, df_to_categorize in all_new_std_dfs:
                          # Asegurarse de que TEXTO_MODELO existe y no tiene NaNs
                          df_to_categorize = df_to_categorize.dropna(subset=[TEXTO_MODELO])
                          if not df_to_categorize.empty:
                            X_new_vec = st.session_state.vectorizer.transform(df_to_categorize[TEXTO_MODELO])
                            predictions = st.session_state.model.predict(X_new_vec)
                            df_to_categorize[CATEGORIA_PREDICHA] = predictions.astype(str).str.capitalize()

                            # Guardar el resultado con su nombre original
                            st.session_state.categorized_data_list.append({'name': file_name, 'data': df_to_categorize})
                          else:
                               st.warning(f"No quedaron filas válidas en '{file_name}' después de limpiar NAs en {TEXTO_MODELO}.")


            except Exception as e:
                 st.error(f"Error al aplicar la categorización: {e}")
                 st.error("Asegúrate de que los nuevos archivos tengan un formato consistente con los datos de entrenamiento (especialmente CONCEPTO y COMERCIO si se usó).")


# --- Mostrar y Descargar Resultados ---
if st.session_state.categorized_data_list:
     st.success("¡Categorización completada!")
     for result in st.session_state.categorized_data_list:
          file_name = result['name']
          df_result = result['data']

          st.write(f"**Archivo:** `{file_name}`")
          # Mostrar columnas relevantes
          display_cols = [CATEGORIA_PREDICHA, CONCEPTO_STD, IMPORTE_STD, AÑO_STD, MES_STD, DIA_STD]
          if COMERCIO_STD in df_result.columns: display_cols.insert(2, COMERCIO_STD)
          # Añadir columnas originales si existen
          orig_cols = [c for c in df_result.columns if c.startswith('ORIG_')]
          display_cols.extend(orig_cols)

          st.dataframe(df_result[display_cols])

          # Botón de descarga para CADA archivo
          csv_output = df_result.to_csv(index=False, sep=';', decimal=',').encode('utf-8')
          st.download_button(
              label=f"📥 Descargar '{file_name}' Categorizado",
              data=csv_output,
              file_name=f"categorizado_{file_name}",
              mime='text/csv',
              key=f"download_{file_name}" # Clave única para cada botón
          )
     st.divider()


# Sidebar Info
st.sidebar.header("Acerca de")
st.sidebar.info(
    "Sube archivos CSV de diferentes bancos. Entrena el modelo con datos históricos categorizados. Luego, categoriza nuevos archivos no etiquetados."
)
