import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import io
import numpy as np
from datetime import datetime
import json # Para guardar/cargar mapeos
import traceback # Para imprimir errores detallados
import os # Para manejo de archivos (aunque no guardaremos en servidor)

# --- Constantes de Columnas Estándar Internas ---
CONCEPTO_STD = 'CONCEPTO_STD'
COMERCIO_STD = 'COMERCIO_STD'
IMPORTE_STD = 'IMPORTE_STD'
AÑO_STD = 'AÑO'
MES_STD = 'MES'
DIA_STD = 'DIA'
FECHA_STD = 'FECHA_STD'
CATEGORIA_STD = 'CATEGORIA_STD'
SUBCATEGORIA_STD = 'SUBCATEGORIA_STD'
TEXTO_MODELO = 'TEXTO_MODELO'
CATEGORIA_PREDICHA = 'CATEGORIA_PREDICHA'

MANDATORY_STD_COLS = [CONCEPTO_STD, IMPORTE_STD, FECHA_STD] # Fecha se maneja AÑO/MES/DIA o FECHA_STD
OPTIONAL_STD_COLS = [COMERCIO_STD]
CONFIG_FILENAME = "Configuracion_Mapeo_Bancos.json" # Nombre estándar para el archivo de config

# --- Session State Initialization ---
# ... (sin cambios respecto a la v anterior) ...
if 'model_trained' not in st.session_state: st.session_state.model_trained = False
if 'model' not in st.session_state: st.session_state.model = None
if 'vectorizer' not in st.session_state: st.session_state.vectorizer = None
if 'knowledge' not in st.session_state: st.session_state.knowledge = {'categorias': [], 'subcategorias': {}, 'comercios': {}}
if 'bank_mappings' not in st.session_state: st.session_state.bank_mappings = {}
if 'training_report' not in st.session_state: st.session_state.training_report = "Modelo no entrenado."

# --- Funciones (Parseo, ML, Estandarización - sin cambios significativos respecto a la última corrección) ---

# Nota: Se asume que las funciones parse_historic_categorized, read_sample_csv,
# extract_knowledge_std, train_classifier_std, y standardize_data_with_mapping
# de la respuesta anterior están aquí y funcionan correctamente después de las
# correcciones previas. Pégalas aquí si es necesario.
# Asegúrate que parse_historic_categorized y standardize_data_with_mapping
# usan .str correctamente y manejan errores.

# ----- PEGA AQUÍ LAS FUNCIONES CORREGIDAS DE LA RESPUESTA ANTERIOR -----
# parse_historic_categorized, read_sample_csv, extract_knowledge_std,
# train_classifier_std, standardize_data_with_mapping
# -----------------------------------------------------------------------
# (Asegúrate de que la función parse_historic_categorized ya no da el error 'strip')
def parse_historic_categorized(df_raw):
    """Parsea el Gastos.csv inicial para entrenamiento."""
    try:
        #st.write("Debug (parse_historic): Iniciando parseo...")
        if not isinstance(df_raw, pd.DataFrame):
            st.error("Error Interno: parse_historic_categorized no recibió un DataFrame.")
            return None
        df = df_raw.copy()
        try:
            df.columns = [str(col).upper().strip() for col in df.columns]
            #st.write(f"Debug (parse_historic): Columnas limpiadas: {df.columns.tolist()}")
        except Exception as e_col:
            st.error(f"Error limpiando nombres de columna: {e_col}")
            return None

        required = ['CONCEPTO', 'CATEGORÍA', 'SUBCATEGORIA', 'IMPORTE', 'AÑO', 'MES', 'DIA']
        if 'COMERCIO' not in df.columns:
            #st.warning("Archivo histórico: Columna 'COMERCIO' no encontrada. Se creará vacía.")
            df['COMERCIO'] = ''
        missing = [col for col in required if col not in df.columns]
        if missing:
            st.error(f"Archivo histórico: Faltan columnas esenciales: {', '.join(missing)}")
            return None

        df_std = pd.DataFrame()
        text_cols_mapping = { CONCEPTO_STD: 'CONCEPTO', COMERCIO_STD: 'COMERCIO', CATEGORIA_STD: 'CATEGORÍA', SUBCATEGORIA_STD: 'SUBCATEGORIA' }

        for std_col, raw_col in text_cols_mapping.items():
            #st.write(f"Debug (parse_historic): Procesando '{raw_col}' -> '{std_col}'")
            if raw_col not in df.columns:
                 if std_col == COMERCIO_STD: df_std[COMERCIO_STD] = ''; continue # Crear vacía si es comercio
                 st.error(f"Error Interno: Columna '{raw_col}' desapareció."); return None
            try:
                series = df[raw_col].fillna('').astype(str)
                # Verificar si realmente es necesario usar .str (si ya es string)
                if pd.api.types.is_string_dtype(series.dtype): # Chequeo redundante pero seguro
                     df_std[std_col] = series.str.lower().str.strip()
                else: # Si astype(str) falló o produjo algo raro (poco probable aquí)
                     df_std[std_col] = series.apply(lambda x: str(x).lower().strip()) # Fallback más lento
                #st.write(f"Debug (parse_historic): Columna '{std_col}' procesada OK.")
            except AttributeError as ae:
                st.error(f"!!! Error de Atributo procesando '{raw_col}' -> '{std_col}'.")
                try:
                    problematic_types = df[raw_col].apply(type).value_counts()
                    st.error(f"Tipos de datos encontrados en '{raw_col}':\n{problematic_types}")
                    non_string_indices = df[raw_col].apply(lambda x: not isinstance(x, (str, type(None), float, int))).index
                    if not non_string_indices.empty:
                         st.error(f"Primeros valores no textuales en '{raw_col}':\n{df.loc[non_string_indices, raw_col].head()}")
                except Exception as e_diag: st.error(f"No se pudo diagnosticar: {e_diag}")
                return None
            except Exception as e:
                st.error(f"Error inesperado procesando texto en '{raw_col}': {e}"); st.error(traceback.format_exc()); return None

        #st.write("Debug (parse_historic): Procesando IMPORTE")
        try:
            importe_str = df['IMPORTE'].astype(str).str.replace(',', '.', regex=False)
            df_std[IMPORTE_STD] = pd.to_numeric(importe_str, errors='coerce')
            if df_std[IMPORTE_STD].isnull().any(): st.warning("Histórico: Algunos importes no son números.")
        except Exception as e: st.error(f"Error procesando IMPORTE histórico: {e}"); return None

        #st.write("Debug (parse_historic): Procesando Fechas")
        try:
            for col in ['AÑO', 'MES', 'DIA']:
                df_std[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                # Validaciones opcionales
                #if col == 'MES' and not df_std[col].between(1, 12).all(): st.warning(f"Valores MES fuera de rango.")
                #if col == 'DIA' and not df_std[col].between(1, 31).all(): st.warning(f"Valores DIA fuera de rango.")
        except Exception as e: st.error(f"Error procesando Fechas históricas: {e}"); return None

        #st.write("Debug (parse_historic): Creando TEXTO_MODELO")
        try:
            if CONCEPTO_STD not in df_std: df_std[CONCEPTO_STD] = ''
            if COMERCIO_STD not in df_std: df_std[COMERCIO_STD] = ''
            df_std[TEXTO_MODELO] = df_std[CONCEPTO_STD] + ' ' + df_std[COMERCIO_STD]
            df_std[TEXTO_MODELO] = df_std[TEXTO_MODELO].str.strip()
        except Exception as e: st.error(f"Error creando TEXTO_MODELO: {e}"); return None

        #st.write("Debug (parse_historic): Filtrando filas finales...")
        initial_rows = len(df_std)
        if CATEGORIA_STD not in df_std.columns: st.error("Falta CATEGORIA_STD antes del filtrado."); return None
        df_std = df_std.dropna(subset=[IMPORTE_STD, CATEGORIA_STD])
        df_std = df_std[df_std[CATEGORIA_STD] != '']
        rows_after_filter = len(df_std)
        #st.write(f"Debug (parse_historic): Filas antes: {initial_rows}, después: {rows_after_filter}")
        if df_std.empty: st.warning("No quedaron filas válidas tras filtrar."); return pd.DataFrame()

        #st.write("Debug (parse_historic): Finalizado OK.")
        return df_std

    except Exception as e:
        st.error(f"Error general crítico parseando histórico: {e}")
        st.error(traceback.format_exc()); return None

@st.cache_data # Cachear la lectura puede ser útil
def read_sample_csv(uploaded_file):
    """Lee un CSV de muestra y devuelve el DataFrame y sus columnas."""
    if uploaded_file is None: return None, []
    try:
        bytes_data = uploaded_file.getvalue()
        sniffer_content = bytes_data.decode('utf-8', errors='replace')
        sniffer = io.StringIO(sniffer_content)
        sep = ';'
        try:
            sample_data = sniffer.read(min(1024 * 20, len(sniffer_content)))
            if sample_data:
                 dialect = pd.io.parsers.readers.csv.Sniffer().sniff(sample_data)
                 if dialect.delimiter in [',', ';', '\t', '|']: sep = dialect.delimiter; #st.info(f"Separador detectado: '{sep}'")
                 #else: st.warning(f"Separador detectado ('{dialect.delimiter}') inusual, usando ';'.")
            else: st.error("Archivo vacío."); return None, []
        except Exception: pass #st.warning(f"No se detectó separador, asumiendo ';'.")

        try:
            df = pd.read_csv(io.BytesIO(bytes_data), encoding='utf-8', sep=sep, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(bytes_data), encoding='latin1', sep=sep, low_memory=False)
        except Exception as read_err: st.error(f"Error leyendo CSV con separador '{sep}': {read_err}"); return None, []

        original_columns = df.columns.tolist()
        df.columns = [str(col).strip() for col in original_columns]
        detected_columns = df.columns.tolist()
        return df, detected_columns
    except Exception as e: st.error(f"Error leyendo muestra: {e}"); return None, []

@st.cache_data
def extract_knowledge_std(df_std):
    knowledge = {'categorias': [], 'subcategorias': {}, 'comercios': {}}
    if df_std is None or CATEGORIA_STD not in df_std.columns or df_std.empty: return knowledge
    try:
        knowledge['categorias'] = sorted([c for c in df_std[CATEGORIA_STD].dropna().unique() if c])
        for cat in knowledge['categorias']:
            subcat_col = SUBCATEGORIA_STD
            if subcat_col in df_std.columns:
                 subcats = df_std.loc[df_std[CATEGORIA_STD] == cat, subcat_col].dropna().unique()
                 knowledge['subcategorias'][cat] = sorted([s for s in subcats if s])
            else: knowledge['subcategorias'][cat] = []
            comercio_col = COMERCIO_STD
            if comercio_col in df_std.columns:
                 comers = df_std.loc[df_std[CATEGORIA_STD] == cat, comercio_col].dropna().unique()
                 knowledge['comercios'][cat] = sorted([c for c in comers if c and c != 'n/a'])
            else: knowledge['comercios'][cat] = []
    except Exception as e_kg: st.error(f"Error extrayendo conocimiento: {e_kg}")
    return knowledge

@st.cache_resource
def train_classifier_std(df_std):
    report = "Modelo no entrenado."
    model = None; vectorizer = None
    required = [TEXTO_MODELO, CATEGORIA_STD]
    if df_std is None or df_std.empty or not all(c in df_std.columns for c in required): return model, vectorizer, report
    df_train = df_std.dropna(subset=required); df_train = df_train[df_train[CATEGORIA_STD] != '']
    if df_train.empty or len(df_train[CATEGORIA_STD].unique()) < 2: return model, vectorizer, report
    try:
        X = df_train[TEXTO_MODELO]; y = df_train[CATEGORIA_STD]
        test_available = False
        if len(y.unique()) > 1 and len(y) > 5:
             try:
                 valid_idx = y.dropna().index; X_clean = X[valid_idx]; y_clean = y[valid_idx]
                 if len(y_clean.unique()) > 1 and len(y_clean) > 5 :
                    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean); test_available = True
                 else: X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42); test_available = True
             except ValueError: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42); test_available = True
        else: X_train, y_train = X, y; X_test, y_test = pd.Series(dtype='str'), pd.Series(dtype='str')
        vectorizer = TfidfVectorizer(); X_train_vec = vectorizer.fit_transform(X_train)
        model = MultinomialNB(); model.fit(X_train_vec, y_train)
        if test_available and not X_test.empty:
            try:
                X_test_vec = vectorizer.transform(X_test); y_pred = model.predict(X_test_vec)
                present_labels = sorted(list(set(y_test.unique()) | set(y_pred)))
                report = classification_report(y_test, y_pred, labels=present_labels, zero_division=0)
            except Exception as report_err: report = f"Modelo OK, error reporte: {report_err}"
        else: report = "Modelo entrenado (sin test detallado)."
    except Exception as e: report = f"Error entrenamiento: {e}"; model, vectorizer = None, None
    return model, vectorizer, report

def standardize_data_with_mapping(df_raw, mapping):
    """Aplica el mapeo guardado para estandarizar un DataFrame nuevo."""
    try:
        df_std = pd.DataFrame()
        df = df_raw.copy() # Trabajar con copia
        df.columns = [str(col).strip() for col in df.columns]
        original_columns = df.columns.tolist()

        temp_std_data = {}
        source_cols_used = []
        # Verificar si las columnas mapeadas existen en el archivo raw
        for std_col, source_col in mapping['columns'].items():
             if source_col in original_columns:
                  temp_std_data[std_col] = df[source_col]
                  source_cols_used.append(source_col)
             else: # Mapeada pero no existe
                  is_essential = std_col in [CONCEPTO_STD, IMPORTE_STD] or (std_col==FECHA_STD and not (AÑO_STD in mapping['columns'] and MES_STD in mapping['columns'] and DIA_STD in mapping['columns'])) or (std_col in [AÑO_STD, MES_STD, DIA_STD] and not FECHA_STD in mapping['columns'])
                  if is_essential:
                       st.error(f"¡Error Crítico! Columna esencial mapeada '{source_col}' para '{std_col}' no encontrada en archivo.")
                       return None
                  else: # Opcional (ej: COMERCIO)
                       st.info(f"Columna opcional mapeada '{source_col}' ('{std_col}') no encontrada. Se omitirá.")

        df_std = pd.DataFrame(temp_std_data) # Crear DF con columnas mapeadas encontradas

        # --- Procesar Fecha ---
        fecha_col_std = mapping['columns'].get(FECHA_STD)
        año_col_std = mapping['columns'].get(AÑO_STD)
        mes_col_std = mapping['columns'].get(MES_STD)
        dia_col_std = mapping['columns'].get(DIA_STD)
        date_processed_ok = False

        if FECHA_STD in df_std.columns: # Se mapeó columna única
            date_format = mapping.get('date_format')
            if not date_format: st.error("Falta formato de fecha."); return None
            try:
                date_series = df_std[FECHA_STD].astype(str).str.strip()
                valid_dates = pd.to_datetime(date_series, format=date_format, errors='coerce')
                if valid_dates.isnull().all(): st.error(f"Ninguna fecha coincide con formato '{date_format}'."); return None
                if valid_dates.isnull().any(): st.warning("Algunas fechas no coinciden con formato.")
                df_std[AÑO_STD] = valid_dates.dt.year.fillna(0).astype(int)
                df_std[MES_STD] = valid_dates.dt.month.fillna(0).astype(int)
                df_std[DIA_STD] = valid_dates.dt.day.fillna(0).astype(int)
                df_std = df_std.drop(columns=[FECHA_STD])
                date_processed_ok = True
            except Exception as e_date: st.error(f"Error procesando fecha única: {e_date}"); return None

        elif all(col in df_std.columns for col in [AÑO_STD, MES_STD, DIA_STD]): # Se mapearon A/M/D
            try:
                for col_std in [AÑO_STD, MES_STD, DIA_STD]:
                    df_std[col_std] = pd.to_numeric(df_std[col_std], errors='coerce').fillna(0).astype(int)
                date_processed_ok = True
            except Exception as e_num: st.error(f"Error convirtiendo A/M/D a número: {e_num}"); return None
        else: st.error("Mapeo de fecha incompleto."); return None

        if not date_processed_ok: return None # Salir si la fecha falló

        # --- Procesar Importe ---
        if IMPORTE_STD in df_std.columns:
            try:
                importe_str = df_std[IMPORTE_STD].fillna('0').astype(str)
                thousands_sep = mapping.get('thousands_sep')
                if thousands_sep: importe_str = importe_str.str.replace(thousands_sep, '', regex=False)
                decimal_sep = mapping.get('decimal_sep', ',')
                importe_str = importe_str.str.replace(decimal_sep, '.', regex=False)
                df_std[IMPORTE_STD] = pd.to_numeric(importe_str, errors='coerce')
                if df_std[IMPORTE_STD].isnull().any(): st.warning("Algunos importes no pudieron convertirse.")
            except Exception as e_imp: st.error(f"Error procesando importe: {e_imp}"); return None
        else: st.error("Falta columna IMPORTE_STD mapeada."); return None

        # --- Limpiar Concepto y Comercio ---
        for col_std in [CONCEPTO_STD, COMERCIO_STD]:
            if col_std in df_std.columns:
                df_std[col_std] = df_std[col_std].fillna('').astype(str).str.lower().str.strip()
            elif col_std == COMERCIO_STD: df_std[COMERCIO_STD] = ''

        # --- Crear Texto Modelo ---
        if CONCEPTO_STD not in df_std: df_std[CONCEPTO_STD] = ''
        if COMERCIO_STD not in df_std: df_std[COMERCIO_STD] = ''
        df_std[TEXTO_MODELO] = df_std[CONCEPTO_STD] + ' ' + df_std[COMERCIO_STD]
        df_std[TEXTO_MODELO] = df_std[TEXTO_MODELO].str.strip()

        # --- Mantener Originales ---
        original_cols_to_keep = [c for c in original_columns if c not in source_cols_used]
        for col in original_cols_to_keep:
            target_col_name = f"ORIG_{col}"
            suffix = 1
            while target_col_name in df_std.columns: target_col_name = f"ORIG_{col}_{suffix}"; suffix += 1
            df_std[target_col_name] = df[col] # Usar df (copia original)

        # --- Filtrado Final ---
        df_std = df_std.dropna(subset=[IMPORTE_STD, TEXTO_MODELO])
        df_std = df_std[df_std[TEXTO_MODELO] != '']

        return df_std

    except Exception as e:
        st.error(f"Error inesperado aplicando mapeo '{mapping.get('bank_name', 'Desconocido')}': {e}")
        st.error(traceback.format_exc()); return None

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🏦 Categorizador Bancario Multi-Formato v3")

# --- Fase 1: Entrenamiento Inicial ---
with st.expander("Fase 1: Entrenar Modelo con Datos Históricos Categorizados", expanded=True):
    # ... (UI de Fase 1 sin cambios respecto a la versión anterior) ...
    st.write("Sube tu archivo CSV histórico (ej: `Gastos.csv`) que ya contiene las categorías y subcategorías asignadas. Este archivo entrena el modelo base.")
    uploaded_historic_file = st.file_uploader("Cargar Archivo Histórico Categorizado (.csv)", type="csv", key="historic_uploader_f1")
    if uploaded_historic_file:
        if st.button("🧠 Entrenar Modelo y Aprender Conocimiento Inicial", key="train_historic_f1"):
            with st.spinner("Procesando archivo histórico y entrenando..."):
                df_raw_hist, _ = read_sample_csv(uploaded_historic_file)
                if df_raw_hist is not None:
                    df_std_hist = parse_historic_categorized(df_raw_hist.copy())
                    if df_std_hist is not None and not df_std_hist.empty:
                        st.success("Archivo histórico parseado.")
                        st.session_state.knowledge = extract_knowledge_std(df_std_hist)
                        st.sidebar.success("Conocimiento Inicial Extraído")
                        with st.sidebar.expander("Categorías Aprendidas"): st.write(st.session_state.knowledge['categorias'])
                        model, vectorizer, report = train_classifier_std(df_std_hist)
                        if model and vectorizer:
                            st.session_state.model = model; st.session_state.vectorizer = vectorizer
                            st.session_state.model_trained = True; st.session_state.training_report = report
                            st.success("¡Modelo entrenado exitosamente!")
                            st.sidebar.subheader("Evaluación Modelo Base")
                            with st.sidebar.expander("Ver Informe"): st.text(st.session_state.training_report)
                        else:
                            st.error("Fallo en entrenamiento."); st.session_state.model_trained = False
                            st.session_state.training_report = report; st.sidebar.error("Entrenamiento Fallido")
                            st.sidebar.text(st.session_state.training_report)
                    else: st.error("No se pudo parsear el histórico o no contenía datos válidos."); st.session_state.model_trained = False
                else: st.error("No se pudo leer el archivo histórico."); st.session_state.model_trained = False


# --- Fase 2: Aprendizaje de Formatos Bancarios ---
with st.expander("Fase 2: Aprender Formatos y Cargar/Guardar Configuración"):
    st.write("Aquí puedes enseñar a la aplicación cómo leer archivos de diferentes bancos o cargar una configuración guardada.")

    # **Cargar Configuración**
    st.subheader("Cargar Configuración Guardada")
    uploaded_config_file = st.file_uploader(f"Cargar Archivo '{CONFIG_FILENAME}'", type="json", key="config_loader")
    if uploaded_config_file:
        try:
            config_data = json.load(uploaded_config_file)
            if isinstance(config_data, dict):
                # Podrías añadir validaciones más estrictas aquí si quieres
                st.session_state.bank_mappings = config_data
                st.success(f"Configuración cargada desde '{uploaded_config_file.name}'!")
                st.sidebar.success("Config. Cargada") # Feedback en sidebar
                # Limpiar uploader para evitar recarga accidental (si Streamlit lo permite)
                # uploaded_config_file = None # Esto no funciona directamente en Streamlit
            else:
                st.error("El archivo de configuración no tiene el formato esperado (debe ser un diccionario JSON).")
        except json.JSONDecodeError:
            st.error("Error al leer el archivo JSON. Asegúrate de que el archivo es válido.")
        except Exception as e_load:
            st.error(f"Error inesperado al cargar la configuración: {e_load}")

    st.divider()

    # **Aprender Nuevo Formato**
    st.subheader("Aprender/Editar Formato de Banco")
    bank_options = ["SANTANDER", "EVO", "WIZINK", "AMEX"] # Añade más bancos aquí
    selected_bank_learn = st.selectbox("Selecciona Banco:", bank_options, key="bank_learn_f2")

    uploaded_sample_file = st.file_uploader(f"Cargar archivo CSV de ejemplo de {selected_bank_learn}", type="csv", key="sample_uploader_f2")

    if uploaded_sample_file:
        df_sample, detected_columns = read_sample_csv(uploaded_sample_file)
        if df_sample is not None:
            # ... (UI de Mapeo sin cambios respecto a la v anterior) ...
            # (Incluye los selectbox, checkbox de fecha, inputs de formato/separadores)
            st.write(f"Columnas detectadas en el archivo de {selected_bank_learn}:")
            st.code(f"{detected_columns}")
            st.dataframe(df_sample.head(3))
            st.subheader("Mapeo de Columnas")
            saved_mapping = st.session_state.bank_mappings.get(selected_bank_learn, {'columns': {}})
            current_mapping_ui = {'columns': {}}
            cols_with_none = [None] + detected_columns
            st.markdown("**Campos Esenciales:**")
            current_mapping_ui['columns'][CONCEPTO_STD] = st.selectbox(f"`{CONCEPTO_STD}` (Descripción)", cols_with_none, index=cols_with_none.index(saved_mapping['columns'].get(CONCEPTO_STD)) if saved_mapping['columns'].get(CONCEPTO_STD) in cols_with_none else 0, key=f"map_{CONCEPTO_STD}_{selected_bank_learn}")
            current_mapping_ui['columns'][IMPORTE_STD] = st.selectbox(f"`{IMPORTE_STD}` (Valor)", cols_with_none, index=cols_with_none.index(saved_mapping['columns'].get(IMPORTE_STD)) if saved_mapping['columns'].get(IMPORTE_STD) in cols_with_none else 0, key=f"map_{IMPORTE_STD}_{selected_bank_learn}")
            st.markdown("**Campo de Fecha (elige una opción):**")
            is_single_date_saved = FECHA_STD in saved_mapping['columns']
            map_single_date = st.checkbox("La fecha está en una sola columna", value=is_single_date_saved, key=f"map_single_date_{selected_bank_learn}")
            if map_single_date:
                current_mapping_ui['columns'][FECHA_STD] = st.selectbox(f"`{FECHA_STD}` (Columna Única)", cols_with_none, index=cols_with_none.index(saved_mapping['columns'].get(FECHA_STD)) if saved_mapping['columns'].get(FECHA_STD) in cols_with_none else 0, key=f"map_{FECHA_STD}_{selected_bank_learn}")
                date_format_guess = st.text_input("Formato fecha (ej: %d/%m/%Y, %Y-%m-%d)", value=saved_mapping.get('date_format', ''), help="Usa códigos `strftime` de Python", key=f"map_date_format_{selected_bank_learn}")
                if date_format_guess: current_mapping_ui['date_format'] = date_format_guess.strip()
                current_mapping_ui['columns'].pop(AÑO_STD, None); current_mapping_ui['columns'].pop(MES_STD, None); current_mapping_ui['columns'].pop(DIA_STD, None)
            else:
                current_mapping_ui['columns'][AÑO_STD] = st.selectbox(f"`{AÑO_STD}`", cols_with_none, index=cols_with_none.index(saved_mapping['columns'].get(AÑO_STD)) if saved_mapping['columns'].get(AÑO_STD) in cols_with_none else 0, key=f"map_{AÑO_STD}_{selected_bank_learn}")
                current_mapping_ui['columns'][MES_STD] = st.selectbox(f"`{MES_STD}`", cols_with_none, index=cols_with_none.index(saved_mapping['columns'].get(MES_STD)) if saved_mapping['columns'].get(MES_STD) in cols_with_none else 0, key=f"map_{MES_STD}_{selected_bank_learn}")
                current_mapping_ui['columns'][DIA_STD] = st.selectbox(f"`{DIA_STD}`", cols_with_none, index=cols_with_none.index(saved_mapping['columns'].get(DIA_STD)) if saved_mapping['columns'].get(DIA_STD) in cols_with_none else 0, key=f"map_{DIA_STD}_{selected_bank_learn}")
                current_mapping_ui['columns'].pop(FECHA_STD, None); current_mapping_ui.pop('date_format', None)
            st.markdown("**Campos Opcionales:**")
            current_mapping_ui['columns'][COMERCIO_STD] = st.selectbox(f"`{COMERCIO_STD}` (Comercio)", cols_with_none, index=cols_with_none.index(saved_mapping['columns'].get(COMERCIO_STD)) if saved_mapping['columns'].get(COMERCIO_STD) in cols_with_none else 0, key=f"map_{COMERCIO_STD}_{selected_bank_learn}")
            st.markdown("**Configuración de Importe:**")
            current_mapping_ui['decimal_sep'] = st.text_input("Separador Decimal", value=saved_mapping.get('decimal_sep', ','), key=f"map_decimal_{selected_bank_learn}")
            current_mapping_ui['thousands_sep'] = st.text_input("Separador de Miles (si aplica)", value=saved_mapping.get('thousands_sep', ''), key=f"map_thousands_{selected_bank_learn}")

            # --- Validación y Guardado ---
            final_mapping_cols = {std: src for std, src in current_mapping_ui['columns'].items() if src is not None}
            valid_mapping = True
            if not final_mapping_cols.get(CONCEPTO_STD): st.error("Falta mapear CONCEPTO_STD."); valid_mapping = False
            if not final_mapping_cols.get(IMPORTE_STD): st.error("Falta mapear IMPORTE_STD."); valid_mapping = False
            if map_single_date:
                 if not final_mapping_cols.get(FECHA_STD): st.error("Falta mapear FECHA_STD."); valid_mapping = False
                 elif not current_mapping_ui.get('date_format'): st.error("Falta formato de fecha."); valid_mapping = False
            else:
                 if not all(final_mapping_cols.get(d) for d in [AÑO_STD, MES_STD, DIA_STD]): st.error("Faltan mapeos para AÑO, MES o DIA."); valid_mapping = False

            if valid_mapping:
                 mapping_to_save = {'bank_name': selected_bank_learn, 'columns': final_mapping_cols, 'decimal_sep': current_mapping_ui.get('decimal_sep', ',').strip(), 'thousands_sep': current_mapping_ui.get('thousands_sep', '').strip() or None}
                 if map_single_date and current_mapping_ui.get('date_format'): mapping_to_save['date_format'] = current_mapping_ui['date_format']
                 if st.button(f"💾 Guardar Mapeo para {selected_bank_learn}", key="save_mapping_f2"):
                      st.session_state.bank_mappings[selected_bank_learn] = mapping_to_save
                      st.success(f"¡Mapeo para {selected_bank_learn} guardado/actualizado!")
            else: st.warning("Revisa los errores antes de guardar.")

    # **Descargar Configuración**
    st.divider()
    st.subheader("Descargar Configuración Completa")
    if st.session_state.bank_mappings:
        try:
            # Convertir el diccionario de mapeos a una cadena JSON formateada
            config_json_str = json.dumps(st.session_state.bank_mappings, indent=4, ensure_ascii=False)
            st.download_button(
                label=f"💾 Descargar '{CONFIG_FILENAME}'",
                data=config_json_str.encode('utf-8'), # Codificar como bytes UTF-8
                file_name=CONFIG_FILENAME,
                mime='application/json',
                key='download_config'
            )
        except Exception as e_dump:
            st.error(f"Error al preparar la configuración para descarga: {e_dump}")
    else:
        st.info("No hay mapeos guardados para descargar.")


# --- Fase 3: Categorización ---
with st.expander("Fase 3: Categorizar Nuevos Archivos", expanded=True):
    if not st.session_state.model_trained:
        st.warning("⚠️ Modelo no entrenado (Ver Fase 1).")
    elif not st.session_state.bank_mappings:
        st.warning("⚠️ No se han aprendido/cargado formatos bancarios (Ver Fase 2).")
    else:
        st.write("Selecciona el banco y sube el archivo CSV **sin categorizar** que deseas procesar.")
        available_banks_for_pred = list(st.session_state.bank_mappings.keys())
        if not available_banks_for_pred:
             st.warning("No hay mapeos disponibles.")
        else:
            selected_bank_predict = st.selectbox("Banco del Nuevo Archivo:", available_banks_for_pred, key="bank_predict_f3")

            uploaded_final_file = st.file_uploader(f"Cargar archivo CSV NUEVO de {selected_bank_predict}", type="csv", key="final_uploader_f3")

            if uploaded_final_file and selected_bank_predict:
                mapping_to_use = st.session_state.bank_mappings.get(selected_bank_predict)
                if not mapping_to_use:
                     st.error(f"Error interno: No se encontró el mapeo para {selected_bank_predict}.")
                else:
                     st.write(f"Procesando '{uploaded_final_file.name}'...")
                     with st.spinner(f"Estandarizando datos..."):
                          df_raw_new, _ = read_sample_csv(uploaded_final_file)
                          df_std_new = None
                          if df_raw_new is not None:
                              df_std_new = standardize_data_with_mapping(df_raw_new.copy(), mapping_to_use)
                          else: st.error(f"No se pudo leer: {uploaded_final_file.name}")

                     if df_std_new is not None and not df_std_new.empty:
                          st.success("Datos estandarizados.")
                          with st.spinner("Aplicando modelo..."):
                              try:
                                   if TEXTO_MODELO not in df_std_new.columns:
                                       st.error(f"Error: Falta {TEXTO_MODELO} tras estandarizar.")
                                   else:
                                        df_pred = df_std_new.dropna(subset=[TEXTO_MODELO]).copy()
                                        if not df_pred.empty:
                                             X_new_vec = st.session_state.vectorizer.transform(df_pred[TEXTO_MODELO])
                                             predictions = st.session_state.model.predict(X_new_vec)

                                             # *** CORRECCIÓN AQUÍ ***
                                             # Convertir cada elemento del array a string y capitalizar
                                             capitalized_predictions = [str(p).capitalize() for p in predictions]
                                             df_pred[CATEGORIA_PREDICHA] = capitalized_predictions
                                             # *** FIN CORRECCIÓN ***

                                             st.subheader("📊 Resultados")
                                             display_cols = [CATEGORIA_PREDICHA, CONCEPTO_STD, IMPORTE_STD, AÑO_STD, MES_STD, DIA_STD]
                                             if COMERCIO_STD in df_pred.columns: display_cols.insert(2, COMERCIO_STD)
                                             orig_cols = [c for c in df_pred.columns if c.startswith('ORIG_')]
                                             display_cols.extend(orig_cols)
                                             final_display_cols = [col for col in display_cols if col in df_pred.columns]
                                             st.dataframe(df_pred[final_display_cols])

                                             csv_output = df_pred.to_csv(index=False, sep=';', decimal=',').encode('utf-8')
                                             st.download_button(
                                                  label=f"📥 Descargar '{uploaded_final_file.name}' Categorizado",
                                                  data=csv_output, file_name=f"categorizado_{uploaded_final_file.name}",
                                                  mime='text/csv', key=f"download_final_{uploaded_final_file.name}"
                                             )
                                        else: st.warning("No quedaron filas válidas para categorizar.")
                              except Exception as e:
                                   st.error(f"Error durante la predicción: {e}")
                                   st.error(f"Vectorizador: {st.session_state.vectorizer}")
                                   st.error(f"Texto (head): {df_pred[TEXTO_MODELO].head().tolist() if TEXTO_MODELO in df_pred else 'N/A'}")

                     elif df_std_new is not None and df_std_new.empty:
                         st.warning("Archivo vacío o sin datos válidos tras estandarizar.")
                     else:
                         st.error("Fallo en la estandarización usando el mapeo.")

# Sidebar Info
st.sidebar.divider()
st.sidebar.header("Acerca de")
st.sidebar.info(
    "1. Entrena con tu CSV histórico. "
    "2. Enseña/Carga los formatos de tus bancos. Guarda la configuración. "
    "3. Sube nuevos archivos para categorizarlos."
)
