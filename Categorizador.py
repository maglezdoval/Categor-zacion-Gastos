import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import io
import numpy as np
from datetime import datetime
import json
import traceback
import os
from collections import Counter
from fuzzywuzzy import process # Requires: pip install fuzzywuzzy python-Levenshtein
import re

# --- Constantes ---
CONCEPTO_STD = 'CONCEPTO_STD'; COMERCIO_STD = 'COMERCIO_STD'; IMPORTE_STD = 'IMPORTE_STD'
AÑO_STD = 'AÑO'; MES_STD = 'MES'; DIA_STD = 'DIA'; FECHA_STD = 'FECHA_STD'
CATEGORIA_STD = 'CATEGORIA_STD'; SUBCATEGORIA_STD = 'SUBCATEGORIA_STD'
TEXTO_MODELO = 'TEXTO_MODELO'; CATEGORIA_PREDICHA = 'CATEGORIA_PREDICHA'
SUBCATEGORIA_PREDICHA = 'SUBCATEGORIA_PREDICHA'; COMERCIO_PREDICHO = 'COMERCIO_PREDICHO'
# Adjusted DB_FINAL_COLS to use _PREDICHA columns as the primary ones for editing/export
DB_FINAL_COLS = [CATEGORIA_PREDICHA, SUBCATEGORIA_PREDICHA, COMERCIO_PREDICHO, CONCEPTO_STD, IMPORTE_STD, AÑO_STD, MES_STD, DIA_STD]
MANDATORY_STD_COLS = [CONCEPTO_STD, IMPORTE_STD, FECHA_STD] # FECHA_STD is placeholder, actual need is FECHA_STD OR (AÑO,MES,DIA)
OPTIONAL_STD_COLS = [COMERCIO_STD]
CONFIG_FILENAME = "Configuracion_Categorizador.json"
DB_FILENAME = "Database_Gastos_Acumulados.csv"
FUZZY_MATCH_THRESHOLD = 80
CUENTA_COL_ORIG = 'CUENTA'; CUENTA_COL_STD = 'ORIG_CUENTA' # Standard name for original account column if kept

# --- Session State Initialization ---
# Basic flags and model persistence
if 'model_trained' not in st.session_state: st.session_state.model_trained = False
if 'knowledge_loaded' not in st.session_state: st.session_state.knowledge_loaded = False
if 'model' not in st.session_state: st.session_state.model = None
if 'vectorizer' not in st.session_state: st.session_state.vectorizer = None
if 'bank_mappings' not in st.session_state: st.session_state.bank_mappings = {}
if 'training_report' not in st.session_state: st.session_state.training_report = "Modelo no entrenado."
# File processing tracking
if 'config_loader_processed_id_f1' not in st.session_state: st.session_state.config_loader_processed_id_f1 = None
if 'db_loader_processed_id' not in st.session_state: st.session_state.db_loader_processed_id = None
# Data persistence
if 'accumulated_data' not in st.session_state: st.session_state.accumulated_data = pd.DataFrame()
# Learned knowledge structure
if 'learned_knowledge' not in st.session_state:
    st.session_state.learned_knowledge = {
        'categorias': [],
        'subcategorias_por_cat': {},
        'comercios_por_cat': {},
        'subcat_unica_por_comercio_y_cat': {},
        'subcat_mas_frecuente_por_comercio_y_cat': {},
        'all_subcategories': [],
        'all_comercios': []
    }
# Debugging
if 'debug_predictions' not in st.session_state: st.session_state.debug_predictions = []

# --- Funciones ---
@st.cache_data # Cache file reading based on content
def read_uploaded_file(uploaded_file):
    """Reads CSV, XLSX, or XLS file into a DataFrame."""
    if uploaded_file is None: return None, []
    try:
        file_name = uploaded_file.name
        bytes_data = uploaded_file.getvalue()
        df = None

        if file_name.lower().endswith('.csv'):
            # Attempt to sniff separator
            sniffer_content = bytes_data.decode('utf-8', errors='replace') # Decode for sniffing
            sniffer = io.StringIO(sniffer_content)
            sep = ';' # Default separator
            try:
                # Read a larger sample for better sniffing
                sample_data = sniffer.read(min(1024 * 20, len(sniffer_content)))
                if sample_data:
                     dialect = pd.io.parsers.readers.csv.Sniffer().sniff(sample_data)
                     # Common delimiters check
                     if dialect.delimiter in [',', ';', '\t', '|']:
                         sep = dialect.delimiter
                else:
                     st.error(f"El archivo '{file_name}' parece estar vacío.")
                     return None, []
            except Exception as sniff_err:
                # st.warning(f"Fallo al detectar separador CSV para '{file_name}', usando '{sep}'. Error: {sniff_err}")
                pass # Proceed with default ';'

            # Try reading with UTF-8, then Latin-1
            try:
                df = pd.read_csv(io.BytesIO(bytes_data), encoding='utf-8', sep=sep, low_memory=False)
            except UnicodeDecodeError:
                st.warning(f"Fallo lectura UTF-8 de '{file_name}', intentando Latin-1...")
                try:
                    df = pd.read_csv(io.BytesIO(bytes_data), encoding='latin1', sep=sep, low_memory=False)
                except Exception as read_err_latin1:
                    st.error(f"Error al leer CSV '{file_name}' con sep '{sep}' (Latin-1): {read_err_latin1}")
                    return None, []
            except Exception as read_err_utf8:
                st.error(f"Error al leer CSV '{file_name}' con sep '{sep}' (UTF-8): {read_err_utf8}")
                return None, []

        elif file_name.lower().endswith('.xlsx'):
            try:
                df = pd.read_excel(io.BytesIO(bytes_data), engine='openpyxl')
            except ImportError:
                st.error("Para leer archivos XLSX, necesitas instalar 'openpyxl'. Ejecuta: pip install openpyxl")
                return None, []
            except Exception as read_excel_err:
                st.error(f"Error al leer XLSX '{file_name}': {read_excel_err}")
                return None, []

        elif file_name.lower().endswith('.xls'):
            try:
                df = pd.read_excel(io.BytesIO(bytes_data), engine='xlrd') # Note: xlrd might have limitations with newer .xls files
            except ImportError:
                st.error("Para leer archivos XLS antiguos, necesitas instalar 'xlrd'. Ejecuta: pip install xlrd")
                return None, []
            except Exception as read_excel_err:
                st.error(f"Error al leer XLS '{file_name}': {read_excel_err}")
                return None, []

        else:
            st.error(f"Formato de archivo no soportado: '{file_name}'. Usa CSV, XLSX o XLS.")
            return None, []

        # Post-read processing
        if df is not None:
            if df.empty:
                st.warning(f"El archivo '{file_name}' se leyó pero no contiene datos.")
                detected_columns = [str(col).strip() for col in df.columns] if hasattr(df, 'columns') else []
                return df, detected_columns # Return empty df and cols if possible
            # Clean column names immediately after reading
            original_columns = df.columns.tolist()
            df.columns = [str(col).strip() for col in original_columns]
            detected_columns = df.columns.tolist()
            return df, detected_columns
        else:
            # Error messages handled within specific read blocks
            # st.error(f"No se pudo leer el archivo '{file_name}'.") # Redundant if specific error shown
            return None, []

    except Exception as e:
        st.error(f"Error general al procesar el archivo '{uploaded_file.name if uploaded_file else 'N/A'}': {e}")
        st.error(traceback.format_exc()) # Show full traceback for debugging
        return None, []

# NOTE: This function is DEPRECATED / Not directly used for training anymore.
# Training now uses `parse_accumulated_db_for_training`. Kept for reference or potential future use.
def parse_historic_categorized(df_raw):
    """Parses a DF assumed to have 'CONCEPTO', 'CATEGORÍA', 'SUBCATEGORIA', etc."""
    st.warning("Función 'parse_historic_categorized' es obsoleta, usar 'parse_accumulated_db_for_training'.")
    try:
        if not isinstance(df_raw, pd.DataFrame):
            st.error("Error Interno (parse_historic): La entrada no es un DataFrame.")
            return None
        df = df_raw.copy()
        # Standardize column names for processing
        df.columns = [str(col).upper().strip() for col in df.columns]

        # Define required and optional columns in the RAW historical format
        required_raw_cols = ['CONCEPTO', 'CATEGORÍA', 'SUBCATEGORIA', 'IMPORTE', 'AÑO', 'MES', 'DIA']
        optional_raw_cols = ['COMERCIO', CUENTA_COL_ORIG.upper()] # Add CUENTA as optional raw

        # Add missing optional columns with empty strings
        if 'COMERCIO' not in df.columns: df['COMERCIO'] = ''
        if CUENTA_COL_ORIG.upper() not in df.columns: df[CUENTA_COL_ORIG.upper()] = ''

        # Check for missing required columns
        missing_req = [col for col in required_raw_cols if col not in df.columns]
        if missing_req:
            st.error(f"Error en archivo histórico: Faltan columnas requeridas: {', '.join(missing_req)}")
            return None

        # Create standardized DataFrame
        df_std = pd.DataFrame()

        # Map RAW -> STD text columns and standardize them
        text_map = {
            CONCEPTO_STD: 'CONCEPTO',
            COMERCIO_STD: 'COMERCIO',
            CATEGORIA_STD: 'CATEGORÍA',
            SUBCATEGORIA_STD: 'SUBCATEGORIA',
            CUENTA_COL_STD: CUENTA_COL_ORIG.upper() # Map raw cuenta to std cuenta
        }
        for std_col, raw_col in text_map.items():
            # Should always exist because we added missing optional ones
            if raw_col not in df.columns:
                 st.error(f"Error Interno Crítico: Columna '{raw_col}' debería existir pero no se encuentra.")
                 return None
            try:
                # Ensure conversion to string, fill NaNs, lowercase, strip whitespace
                series = df[raw_col].fillna('').astype(str)
                df_std[std_col] = series.str.lower().str.strip()
            except Exception as e:
                st.error(f"Error procesando columna de texto '{raw_col}' a '{std_col}': {e}")
                return None

        # Process IMPORTE
        try:
            # Convert to string, replace comma decimal separator, convert to numeric
            imp_str = df['IMPORTE'].astype(str).str.replace(',', '.', regex=False)
            df_std[IMPORTE_STD] = pd.to_numeric(imp_str, errors='coerce')
            # Warn if any import amounts couldn't be converted
            if df_std[IMPORTE_STD].isnull().any():
                st.warning("Aviso en histórico: Algunos valores en 'IMPORTE' no pudieron ser convertidos a número.")
        except Exception as e:
            st.error(f"Error procesando columna 'IMPORTE' del histórico: {e}")
            return None

        # Process date components
        try:
            for col in ['AÑO', 'MES', 'DIA']:
                # Convert to numeric, coerce errors to NaN, fill NaN with 0, convert to integer
                df_std[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        except Exception as e:
            st.error(f"Error procesando columnas de fecha ('AÑO', 'MES', 'DIA') del histórico: {e}")
            return None

        # Create TEXTO_MODELO from CONCEPTO and COMERCIO
        try:
            # Ensure columns exist even if they were originally empty
            if CONCEPTO_STD not in df_std: df_std[CONCEPTO_STD] = ''
            if COMERCIO_STD not in df_std: df_std[COMERCIO_STD] = ''
            df_std[TEXTO_MODELO] = (df_std[CONCEPTO_STD] + ' ' + df_std[COMERCIO_STD]).str.strip()
        except Exception as e:
            st.error(f"Error creando la columna '{TEXTO_MODELO}' desde el histórico: {e}")
            return None

        # Validate essential columns after processing
        if CATEGORIA_STD not in df_std.columns:
            st.error(f"Error Interno Crítico: Falta '{CATEGORIA_STD}' después del procesado inicial.")
            return None

        # Drop rows with missing essential numeric or category data
        df_std = df_std.dropna(subset=[IMPORTE_STD, CATEGORIA_STD])
        # Drop rows where category is empty string after cleaning
        df_std = df_std[df_std[CATEGORIA_STD] != '']

        if df_std.empty:
            st.warning("Aviso en histórico: No quedaron filas válidas después de la limpieza inicial (NaNs, categorías vacías).")
            return pd.DataFrame() # Return empty DF with correct structure

        return df_std

    except Exception as e:
        st.error(f"Error general parseando datos históricos categorizados: {e}")
        st.error(traceback.format_exc())
        return None

# Important: Function now relies on CATEGORIA_STD being present. If using _PREDICHA, rename before calling.
@st.cache_data # Cache knowledge extraction based on DataFrame hash
def extract_knowledge_std(df_std):
    """Extracts categories, subcategories, merchants, and relationships from standardized data."""
    knowledge = {
        'categorias': [],
        'subcategorias_por_cat': {},
        'comercios_por_cat': {},
        'subcat_unica_por_comercio_y_cat': {},
        'subcat_mas_frecuente_por_comercio_y_cat': {},
        'all_subcategories': [],
        'all_comercios': []
    }
    if df_std is None or df_std.empty:
        st.warning("Extracción conocimiento: DataFrame de entrada vacío.")
        return knowledge
    # Ensure required columns are present
    # Check for CATEGORIA_STD (primary) or CATEGORIA_PREDICHA (fallback for this function)
    cat_col_to_use = None
    if CATEGORIA_STD in df_std.columns:
        cat_col_to_use = CATEGORIA_STD
    elif CATEGORIA_PREDICHA in df_std.columns:
        cat_col_to_use = CATEGORIA_PREDICHA
        st.info(f"Extracción conocimiento: Usando '{CATEGORIA_PREDICHA}' como fuente de categorías.")
    else:
        st.warning(f"Extracción conocimiento: Faltan columnas '{CATEGORIA_STD}' o '{CATEGORIA_PREDICHA}'. No se puede extraer conocimiento.")
        return knowledge


    try:
        df_clean = df_std.copy()

        # --- Data Cleaning and Preparation ---
        # Standardize text columns: fill NaNs, ensure string type, lowercase, strip whitespace
        df_clean[cat_col_to_use] = df_clean[cat_col_to_use].fillna('').astype(str).str.lower().str.strip()

        # Check and standardize optional columns if they exist (using both _STD and _PREDICHA possibilities)
        subcat_col_to_use = None
        if SUBCATEGORIA_STD in df_clean.columns: subcat_col_to_use = SUBCATEGORIA_STD
        elif SUBCATEGORIA_PREDICHA in df_clean.columns: subcat_col_to_use = SUBCATEGORIA_PREDICHA

        if subcat_col_to_use:
            df_clean[subcat_col_to_use] = df_clean[subcat_col_to_use].fillna('').astype(str).str.lower().str.strip()
        else:
             # If neither subcat column exists, we can't extract subcat knowledge
             # Create an empty one for consistency in loops later if needed? Or just handle absence.
             # Let's handle absence:
             subcat_col_to_use = None # Explicitly mark as absent

        comercio_col_to_use = None
        if COMERCIO_STD in df_clean.columns: comercio_col_to_use = COMERCIO_STD
        elif COMERCIO_PREDICHO in df_clean.columns: comercio_col_to_use = COMERCIO_PREDICHO

        if comercio_col_to_use:
            df_clean[comercio_col_to_use] = df_clean[comercio_col_to_use].fillna('').astype(str).str.lower().str.strip()
        else:
             comercio_col_to_use = None # Explicitly mark as absent


        # Filter out rows with empty category after cleaning
        df_clean = df_clean[df_clean[cat_col_to_use] != '']
        if df_clean.empty:
            st.warning("Extracción conocimiento: No quedaron filas después de filtrar categorías vacías.")
            return knowledge

        # --- Knowledge Extraction ---
        knowledge['categorias'] = sorted(list(df_clean[cat_col_to_use].unique()))

        all_subcats_set = set()
        all_comers_set = set()

        for cat in knowledge['categorias']:
            df_cat = df_clean[df_clean[cat_col_to_use] == cat]

            # Initialize dictionaries for the current category
            knowledge['subcategorias_por_cat'][cat] = []
            knowledge['comercios_por_cat'][cat] = []
            knowledge['subcat_unica_por_comercio_y_cat'][cat] = {}
            knowledge['subcat_mas_frecuente_por_comercio_y_cat'][cat] = {}

            # Extract Subcategories per Category (if subcat column exists)
            if subcat_col_to_use:
                 subcats_in_cat = df_cat[subcat_col_to_use].unique()
                 # Filter out empty strings and sort
                 current_subcats = sorted([s for s in subcats_in_cat if s])
                 knowledge['subcategorias_por_cat'][cat] = current_subcats
                 all_subcats_set.update(current_subcats) # Add to global set

            # Extract Merchants per Category and Subcategory relationships (if comercio column exists)
            if comercio_col_to_use:
                 # Consider only rows with a non-empty merchant for merchant-specific analysis
                 df_cat_comers = df_cat[df_cat[comercio_col_to_use] != '']
                 comers_in_cat = df_cat_comers[comercio_col_to_use].unique()
                 # Filter out empty strings and 'n/a', sort
                 current_comers = sorted([c for c in comers_in_cat if c and c != 'n/a'])
                 knowledge['comercios_por_cat'][cat] = current_comers
                 all_comers_set.update(current_comers) # Add to global set

                 # Analyze subcategories associated with each merchant within this category (if subcat column also exists)
                 if subcat_col_to_use:
                     for comercio in knowledge['comercios_por_cat'][cat]:
                         # Filter for the specific merchant and non-empty subcategory
                         df_comercio_cat = df_cat_comers[
                             (df_cat_comers[comercio_col_to_use] == comercio) & (df_cat_comers[subcat_col_to_use].fillna('') != '')
                         ]

                         if not df_comercio_cat.empty:
                             subcats_for_this_comercio = df_comercio_cat[subcat_col_to_use]
                             unique_subcats_for_comercio = subcats_for_this_comercio.unique()
                             comercio_key = comercio # Use cleaned comercio name
                             cat_key = cat # Use cleaned category name

                             # 1. Check for unique subcategory rule
                             if len(unique_subcats_for_comercio) == 1:
                                 knowledge['subcat_unica_por_comercio_y_cat'][cat_key][comercio_key] = unique_subcats_for_comercio[0]

                             # 2. Find the most frequent subcategory rule (even if only one exists)
                             if len(subcats_for_this_comercio) > 0:
                                 try:
                                     # Calculate value counts and find the index (subcategory name) of the maximum count
                                     most_frequent_subcat = subcats_for_this_comercio.value_counts().idxmax()
                                     knowledge['subcat_mas_frecuente_por_comercio_y_cat'][cat_key][comercio_key] = most_frequent_subcat
                                 except Exception as e_freq:
                                     # Fallback if idxmax fails (e.g., unexpected data)
                                     st.warning(f"Cálculo subcat frecuente falló para '{comercio}' en '{cat}': {e_freq}. Usando la primera encontrada.")
                                     if len(unique_subcats_for_comercio) > 0:
                                         knowledge['subcat_mas_frecuente_por_comercio_y_cat'][cat_key][comercio_key] = unique_subcats_for_comercio[0]


        # Finalize global lists
        knowledge['all_subcategories'] = sorted(list(all_subcats_set))
        knowledge['all_comercios'] = sorted(list(all_comers_set))

    except Exception as e_kg:
        st.error(f"Error crítico extrayendo conocimiento: {e_kg}")
        st.error(traceback.format_exc())
        # Return the partially filled (or empty) knowledge dict in case of error
    return knowledge

# Resource cache is better for ML models and vectorizers
@st.cache_resource
def train_classifier_std(df_std):
    """Trains a Multinomial Naive Bayes classifier on standardized data."""
    report = "Modelo no entrenado."
    model = None
    vectorizer = None

    # Use CATEGORIA_STD as the target variable for training
    required_cols = [TEXTO_MODELO, CATEGORIA_STD]
    if df_std is None or df_std.empty:
        report = "Error Entrenamiento: DataFrame vacío."
        return model, vectorizer, report
    if not all(c in df_std.columns for c in required_cols):
        missing = [c for c in required_cols if c not in df_std.columns]
        report = f"Error Entrenamiento: Faltan columnas requeridas para entrenamiento ({', '.join(missing)})."
        return model, vectorizer, report

    # Prepare data: drop rows with missing text or category, filter empty categories
    df_train = df_std.dropna(subset=required_cols).copy()
    df_train = df_train[df_train[CATEGORIA_STD] != '']
    df_train[TEXTO_MODELO] = df_train[TEXTO_MODELO].fillna('') # Ensure no NaNs in text

    if df_train.empty:
        report = "Error Entrenamiento: No hay datos válidos para entrenar después de limpiar."
        return model, vectorizer, report
    if df_train[CATEGORIA_STD].nunique() < 2:
        report = "Error Entrenamiento: Se necesita al menos 2 categorías distintas para entrenar."
        # Optionally, could train a dummy classifier here, but better to report error.
        return model, vectorizer, report

    try:
        X = df_train[TEXTO_MODELO]
        y = df_train[CATEGORIA_STD] # Target is CATEGORIA_STD

        # --- Train/Test Split ---
        test_available = False
        # Only split if we have enough data and multiple classes for stratification
        if len(y.unique()) > 1 and len(y) > 10: # Increased threshold for meaningful split
             try:
                 # Attempt stratified split first
                 X_train, X_test, y_train, y_test = train_test_split(
                     X, y, test_size=0.2, random_state=42, stratify=y
                 )
                 test_available = True
             except ValueError:
                 # Fallback to non-stratified if stratification fails (e.g., very small classes)
                 st.warning("Entrenamiento: No se pudo hacer split estratificado, usando split normal.")
                 X_train, X_test, y_train, y_test = train_test_split(
                     X, y, test_size=0.2, random_state=42
                 )
                 test_available = True
        else:
            # Not enough data or classes for a meaningful split, train on everything
            st.warning("Entrenamiento: Datos insuficientes para split, entrenando con todos los datos.")
            X_train, y_train = X, y
            # Create empty Series for consistency if needed later, though report handles it
            X_test, y_test = pd.Series(dtype='str'), pd.Series(dtype='str')

        # --- Vectorization and Training ---
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000) # Added some common params
        X_train_vec = vectorizer.fit_transform(X_train)

        model = MultinomialNB()
        model.fit(X_train_vec, y_train)

        # --- Evaluation (if test set exists) ---
        if test_available and not X_test.empty:
            try:
                X_test_vec = vectorizer.transform(X_test)
                y_pred = model.predict(X_test_vec)

                # Get all unique labels present in either test or prediction for the report
                present_labels = sorted(list(set(y_test.unique()) | set(y_pred)))

                report = classification_report(
                    y_test,
                    y_pred,
                    labels=present_labels, # Ensure all relevant labels are included
                    zero_division=0 # Avoid warnings for metrics that are zero due to no support
                )
            except Exception as report_err:
                report = f"Modelo entrenado OK, pero falló la generación del informe de clasificación: {report_err}"
        else:
            report = "Modelo entrenado exitosamente (no se generó informe de test por datos insuficientes)."

    except Exception as e:
        report = f"Error crítico durante el entrenamiento: {e}\n{traceback.format_exc()}"
        model, vectorizer = None, None # Ensure model/vectorizer are None on failure

    return model, vectorizer, report

def standardize_data_with_mapping(df_raw, mapping):
    """Applies a bank mapping configuration to standardize a raw DataFrame."""
    if not isinstance(df_raw, pd.DataFrame):
        st.error("Error (Estandarizar): La entrada no es un DataFrame.")
        return None
    if not isinstance(mapping, dict) or 'columns' not in mapping:
        st.error(f"Error (Estandarizar): Mapeo inválido o incompleto para '{mapping.get('bank_name', 'desconocido')}'.")
        return None

    try:
        df_std = pd.DataFrame()
        df = df_raw.copy()
        df.columns = [str(col).strip() for col in df.columns] # Clean raw column names
        original_columns = df.columns.tolist()
        temp_std_data = {}
        source_cols_used = set() # Track used source columns to keep unused originals

        # --- 1. Map columns based on 'columns' dictionary ---
        mapping_cols = mapping.get('columns', {})
        for std_col, source_col in mapping_cols.items():
             if source_col is not None and source_col in original_columns:
                  # Only copy if source_col is valid and exists
                  temp_std_data[std_col] = df[source_col]
                  source_cols_used.add(source_col)
             else:
                  # Check if a *required* standard column's source is missing
                  is_required = False
                  if std_col in [CONCEPTO_STD, IMPORTE_STD]: is_required = True
                  # Date is required either as single FECHA_STD or as AÑO, MES, DIA triplet
                  mapped_date_cols = mapping_cols.get(AÑO_STD), mapping_cols.get(MES_STD), mapping_cols.get(DIA_STD)
                  has_amd = all(c is not None and c in original_columns for c in mapped_date_cols)
                  mapped_fecha_std = mapping_cols.get(FECHA_STD)
                  has_fecha_std = mapped_fecha_std is not None and mapped_fecha_std in original_columns

                  if std_col == FECHA_STD and not has_amd: is_required = True
                  if std_col in [AÑO_STD, MES_STD, DIA_STD] and not has_fecha_std: is_required = True


                  if is_required and source_col and source_col not in original_columns:
                      st.error(f"Error Mapeo '{mapping.get('bank_name', '?')}': Columna fuente '{source_col}' (para '{std_col}') no encontrada en el archivo.")
                      return None
                  elif is_required and not source_col:
                       # Don't error if one date path is missing but the other is potentially available
                       # Error will be caught later if neither path is complete
                       if not (std_col==FECHA_STD and has_amd) and not (std_col in [AÑO_STD,MES_STD,DIA_STD] and has_fecha_std):
                            st.error(f"Error Mapeo '{mapping.get('bank_name', '?')}': Columna esencial '{std_col}' no tiene mapeo definido o la columna mapeada no existe.")
                            # return None # Delay return to check date processing fully
                       pass # Pass for now, date check later is more robust

                  # If optional like COMERCIO_STD is missing source, it will be handled later

        # Create initial DataFrame from mapped columns
        df_std = pd.DataFrame(temp_std_data)

        # --- 2. Validate essential columns are present after initial mapping ---
        # Check Concepto and Importe
        if CONCEPTO_STD not in df_std.columns: st.error(f"Mapeo Inválido: Falta '{CONCEPTO_STD}' mapeado desde una columna existente."); return None
        if IMPORTE_STD not in df_std.columns: st.error(f"Mapeo Inválido: Falta '{IMPORTE_STD}' mapeado desde una columna existente."); return None

        # --- 3. Process Date ---
        date_ok = False
        # Check if FECHA_STD was mapped *and* exists in the temp df
        if FECHA_STD in df_std.columns:
            date_fmt = mapping.get('date_format')
            source_col_fecha = mapping_cols.get(FECHA_STD) # Get the original source column name for error messages
            if not date_fmt:
                st.error(f"Error Mapeo '{mapping.get('bank_name', '?')}': Se mapeó '{source_col_fecha}' a '{FECHA_STD}' pero falta 'date_format' en la configuración.")
                return None
            try:
                # Convert to string, strip, then parse
                date_series = df_std[FECHA_STD].astype(str).str.strip()
                # Handle potential empty strings before conversion
                date_series = date_series.replace('', pd.NaT)
                dates = pd.to_datetime(date_series, format=date_fmt, errors='coerce')

                if dates.isnull().all():
                    st.error(f"Error Mapeo '{mapping.get('bank_name', '?')}': Ninguna fecha en '{source_col_fecha}' coincide con el formato '{date_fmt}'.")
                    # Display some problematic values
                    problem_samples = df_std.loc[dates.isnull() & df_std[FECHA_STD].notna(), FECHA_STD].unique()
                    st.info(f"Algunos valores no parseados: {list(problem_samples[:5])}")
                    return None
                if dates.isnull().any():
                    st.warning(f"Aviso Mapeo '{mapping.get('bank_name', '?')}': Algunas fechas en '{source_col_fecha}' no coinciden con formato '{date_fmt}' o estaban vacías.")

                # Extract components, fill NaNs from parsing errors with 0
                df_std[AÑO_STD] = dates.dt.year.fillna(0).astype(int)
                df_std[MES_STD] = dates.dt.month.fillna(0).astype(int)
                df_std[DIA_STD] = dates.dt.day.fillna(0).astype(int)
                # Drop the original source date column (now named FECHA_STD)
                df_std = df_std.drop(columns=[FECHA_STD])
                date_ok = True
            except ValueError as e_dt_val:
                 st.error(f"Error Mapeo '{mapping.get('bank_name', '?')}': Formato de fecha '{date_fmt}' inválido o no aplicable a columna '{source_col_fecha}'. Error: {e_dt_val}")
                 return None
            except Exception as e_dt:
                st.error(f"Error procesando columna de fecha única '{source_col_fecha}': {e_dt}")
                st.error(traceback.format_exc())
                return None
        # Check if AÑO, MES, DIA were mapped *and* exist in the temp df
        elif all(c in df_std.columns for c in [AÑO_STD, MES_STD, DIA_STD]):
            source_cols_amd = mapping_cols.get(AÑO_STD), mapping_cols.get(MES_STD), mapping_cols.get(DIA_STD)
            try:
                for c in [AÑO_STD, MES_STD, DIA_STD]:
                    # Convert to numeric, coerce errors, fill NaNs with 0, convert to int
                    df_std[c] = pd.to_numeric(df_std[c], errors='coerce').fillna(0).astype(int)
                # Basic validation for date parts
                if (df_std[MES_STD] < 1) | (df_std[MES_STD] > 12) | (df_std[DIA_STD] < 1) | (df_std[DIA_STD] > 31):
                     st.warning(f"Aviso Mapeo '{mapping.get('bank_name', '?')}': Columnas A/M/D ({source_cols_amd}) contienen valores inválidos (ej. Mes>12, Dia>31).")
                date_ok = True
            except Exception as e_num:
                st.error(f"Error convirtiendo columnas AÑO/MES/DIA ({source_cols_amd}) a números: {e_num}")
                return None
        else: # Neither single date nor A/M/D triplet is fully mapped and present
            st.error(f"Error Mapeo '{mapping.get('bank_name', '?')}': Mapeo de fecha incompleto. Se necesita '{FECHA_STD}' (con formato) O las tres columnas '{AÑO_STD}', '{MES_STD}', '{DIA_STD}' mapeadas desde columnas existentes.")
            return None

        if not date_ok: return None # Should not happen if logic above is correct, but safety check

        # --- 4. Process Importe ---
        if IMPORTE_STD in df_std.columns: # Should always be true based on earlier check
            source_col_importe = mapping_cols.get(IMPORTE_STD)
            try:
                imp_str = df_std[IMPORTE_STD].fillna('0').astype(str).str.strip()
                # Get separators from mapping, providing defaults
                ts = mapping.get('thousands_sep') # Can be None or empty string
                ds = mapping.get('decimal_sep', ',') # Default to comma if not specified

                # Remove thousands separator if specified
                if ts:
                    imp_str = imp_str.str.replace(ts, '', regex=False)
                # Replace decimal separator with period
                imp_str = imp_str.str.replace(ds, '.', regex=False)

                # Convert to numeric
                df_std[IMPORTE_STD] = pd.to_numeric(imp_str, errors='coerce')

                if df_std[IMPORTE_STD].isnull().any():
                    st.warning(f"Aviso Mapeo '{mapping.get('bank_name', '?')}': Algunos importes en '{source_col_importe}' no pudieron ser convertidos a número después de aplicar separadores (Miles: '{ts}', Decimal: '{ds}').")
                    # Display some problematic values from the original data
                    original_importe_col = df[source_col_importe] if source_col_importe in df.columns else pd.Series()
                    problem_indices = df_std[IMPORTE_STD].isnull().index
                    problem_samples_orig = original_importe_col.loc[problem_indices].unique()
                    st.info(f"Valores originales problemáticos: {list(problem_samples_orig[:5])}")


            except Exception as e_imp:
                st.error(f"Error procesando columna de importe '{source_col_importe}': {e_imp}")
                st.error(traceback.format_exc())
                return None
        else:
             st.error("Error Interno: IMPORTE_STD falta inesperadamente.") # Should have been caught earlier
             return None


        # --- 5. Process Text Columns (Concepto, Comercio) ---
        for c in [CONCEPTO_STD, COMERCIO_STD]:
            if c in df_std.columns:
                # Fill NaNs, convert to string, lowercase, strip whitespace
                df_std[c] = df_std[c].fillna('').astype(str).str.lower().str.strip()
            elif c == COMERCIO_STD:
                # If COMERCIO_STD was not mapped, create it as an empty column
                df_std[COMERCIO_STD] = ''

        # Ensure Concepto exists (should be guaranteed by now)
        if CONCEPTO_STD not in df_std: df_std[CONCEPTO_STD] = ''
        # Ensure Comercio exists
        if COMERCIO_STD not in df_std: df_std[COMERCIO_STD] = ''

        # --- 6. Create TEXTO_MODELO ---
        df_std[TEXTO_MODELO] = (df_std[CONCEPTO_STD] + ' ' + df_std[COMERCIO_STD]).str.strip()

        # --- 7. Keep Original Unused Columns ---
        original_cols_to_keep = [c for c in original_columns if c not in source_cols_used]
        # Specifically check for the original 'CUENTA' column if it wasn't used in mapping
        cuenta_col_orig_upper = CUENTA_COL_ORIG.upper()
        if cuenta_col_orig_upper in original_columns and cuenta_col_orig_upper not in source_cols_used:
             if cuenta_col_orig_upper not in original_cols_to_keep: # Avoid adding twice if already there
                original_cols_to_keep.append(cuenta_col_orig_upper)

        for col in original_cols_to_keep:
            target_col_name = f"ORIG_{col}" # Prefix original columns
            sfx = 1
            # Handle potential name collisions (unlikely but possible)
            while target_col_name in df_std.columns:
                target_col_name = f"ORIG_{col}_{sfx}"
                sfx += 1

            # Special handling for the account column: name it CUENTA_COL_STD
            if col == cuenta_col_orig_upper:
                 target_col_name = CUENTA_COL_STD

            # Add the original column with the new name if it doesn't exist
            if target_col_name not in df_std.columns:
                 # Ensure added original columns are also cleaned (fillna, str type)
                 df_std[target_col_name] = df[col].fillna('').astype(str)


        # --- 8. Final Cleanup ---
        # Drop rows where essential numeric or text model data is missing after processing
        essential_cols_final = [IMPORTE_STD, TEXTO_MODELO, AÑO_STD, MES_STD, DIA_STD]
        initial_rows = len(df_std)
        # Check for NaNs in essential numeric/date columns
        df_std = df_std.dropna(subset=[IMPORTE_STD, AÑO_STD, MES_STD, DIA_STD])
        # Also drop rows where the text used for the model is empty
        df_std = df_std[df_std[TEXTO_MODELO].fillna('').str.strip() != '']
        rows_after_na = len(df_std)
        if initial_rows > rows_after_na:
             st.info(f"Estandarización '{mapping.get('bank_name', '?')}': Se eliminaron {initial_rows - rows_after_na} filas por datos faltantes/inválidos en columnas esenciales (Importe, Fecha, Concepto).")

        if df_std.empty:
             st.warning(f"Aviso Mapeo '{mapping.get('bank_name', '?')}': No quedaron filas válidas después de la estandarización completa.")
             # Return empty DF but with correct columns if possible
             # Construct expected cols based on mapping + ORIG_ cols + derived cols
             expected_cols = list(mapping_cols.keys()) + [AÑO_STD, MES_STD, DIA_STD, TEXTO_MODELO, CONCEPTO_STD, IMPORTE_STD, COMERCIO_STD] + [c for c in df_std.columns if c.startswith("ORIG_")]
             expected_cols = list(dict.fromkeys(expected_cols)) # unique order-preserving
             # Filter out FECHA_STD if it was replaced by A/M/D
             if AÑO_STD in expected_cols and FECHA_STD in expected_cols: expected_cols.remove(FECHA_STD)
             return pd.DataFrame(columns=[col for col in expected_cols if col is not None])


        return df_std

    except Exception as e:
        st.error(f"Error general aplicando mapeo '{mapping.get('bank_name', '?')}': {e}")
        st.error(traceback.format_exc())
        return None


def parse_accumulated_db_for_training(df_db):
    """Prepares the accumulated DB DataFrame for training or knowledge extraction."""
    if not isinstance(df_db, pd.DataFrame) or df_db.empty:
        st.error("Error Preparación BD: La base de datos acumulada está vacía o no es válida.")
        return None

    df = df_db.copy()
    # Standardize column names from DB file (assuming they might vary slightly)
    df.columns = [str(col).upper().strip() for col in df.columns]

    # --- Identify the Category Column ---
    # Prioritize *_PREDICHA if available (assuming it holds reviewed categories), then _STD, then others.
    cat_col_options = [CATEGORIA_PREDICHA, CATEGORIA_STD, 'CATEGORIA', 'CATEGORÍA', 'CATEGORIA_X']
    cat_col_found = next((c for c in cat_col_options if c in df.columns), None)

    if not cat_col_found:
        st.error(f"Error Preparación BD: No se encontró una columna de categoría reconocida en la BD ({', '.join(cat_col_options)}).")
        return None

    # Rename the found category column to the standard name used for training/knowledge (CATEGORIA_STD)
    if cat_col_found != CATEGORIA_STD:
        # Check if CATEGORIA_STD already exists; if so, warn and prefer it.
        if CATEGORIA_STD in df.columns:
             st.warning(f"Preparación BD: Se encontraron '{cat_col_found}' y '{CATEGORIA_STD}'. Se usará '{CATEGORIA_STD}' para entrenar.")
        else:
             df = df.rename(columns={cat_col_found: CATEGORIA_STD})
             st.info(f"Preparación BD: Usando columna '{cat_col_found}' como '{CATEGORIA_STD}' para entrenamiento.")
    # Now CATEGORIA_STD is guaranteed to be the name of the category column to use, if one was found


    # --- Check for other required columns for training logic ---
    required_std_cols_train = [CONCEPTO_STD, IMPORTE_STD, AÑO_STD, MES_STD, DIA_STD, CATEGORIA_STD] # Now CATEGORIA_STD must exist

    # --- Ensure optional standard columns exist or map from _PREDICHA if needed ---
    # For training/knowledge, we prefer _STD versions if they exist, otherwise map from _PREDICHA
    map_optional = {
        SUBCATEGORIA_STD: SUBCATEGORIA_PREDICHA,
        COMERCIO_STD: COMERCIO_PREDICHO
    }
    for std_col, pred_col in map_optional.items():
         if std_col not in df.columns:
             if pred_col in df.columns:
                 df = df.rename(columns={pred_col: std_col})
                 st.info(f"Preparación BD: Usando '{pred_col}' como '{std_col}'.")
             else:
                 df[std_col] = '' # Create empty std column if neither exists

    # Check again for required cols after potential renaming
    missing_req = [col for col in required_std_cols_train if col not in df.columns]
    if missing_req:
        st.error(f"Error Preparación BD: Faltan columnas esenciales en la BD después de procesar nombres: {', '.join(missing_req)}")
        return None

    # --- Data Cleaning and Type Conversion ---
    df_train = df.copy()
    try:
        # Clean text columns: fillna, string type, lower, strip
        text_cols_to_clean = [CONCEPTO_STD, COMERCIO_STD, CATEGORIA_STD, SUBCATEGORIA_STD]
        for col in text_cols_to_clean:
            if col in df_train.columns: # Ensure column exists before cleaning
                 df_train[col] = df_train[col].fillna('').astype(str).str.lower().str.strip()

        # Clean numeric columns: coerce to numeric, fillna with appropriate value
        df_train[IMPORTE_STD] = pd.to_numeric(df_train[IMPORTE_STD], errors='coerce') # NaNs for failed conversions
        for col_f in [AÑO_STD, MES_STD, DIA_STD]:
            # Ensure conversion to Int64 to handle potential NaNs after coerce before fillna
            df_train[col_f] = pd.to_numeric(df_train[col_f], errors='coerce').astype('Int64').fillna(0).astype(int) # 0 for failed date parts

    except Exception as e_clean:
        st.error(f"Error limpiando datos de la BD acumulada: {e_clean}")
        st.error(traceback.format_exc())
        return None

    # --- Create TEXTO_MODELO ---
    # Ensure components exist (even if empty after cleaning) before concatenation
    if CONCEPTO_STD not in df_train: df_train[CONCEPTO_STD] = ''
    if COMERCIO_STD not in df_train: df_train[COMERCIO_STD] = ''
    df_train[TEXTO_MODELO] = (df_train[CONCEPTO_STD] + ' ' + df_train[COMERCIO_STD]).str.strip()


    # --- Final Filtering ---
    # Drop rows missing essential data for training AFTER cleaning/conversion
    essential_final = [IMPORTE_STD, CATEGORIA_STD, TEXTO_MODELO, AÑO_STD, MES_STD, DIA_STD]
    initial_rows = len(df_train)
    df_train = df_train.dropna(subset=essential_final)

    # Drop rows with empty category or empty text model input
    df_train = df_train[df_train[CATEGORIA_STD] != '']
    df_train = df_train[df_train[TEXTO_MODELO] != '']
    rows_after_filter = len(df_train)

    if initial_rows > rows_after_filter:
        st.info(f"Preparación BD: Se eliminaron {initial_rows - rows_after_filter} filas de la BD por datos faltantes/inválidos para entrenamiento.")

    if df_train.empty:
        st.warning("Aviso Preparación BD: No quedaron filas válidas en la base de datos para entrenar/extraer conocimiento después de la limpieza.")
        return None # Return None instead of empty DF for clarity

    # --- Select Columns for Output ---
    # Include columns needed for training (TEXTO_MODELO, CATEGORIA_STD)
    # and knowledge extraction (SUBCATEGORIA_STD, COMERCIO_STD)
    # and potentially for reference or debugging (IMPORTE, DATE parts, CONCEPTO)
    # Also keep ORIG_CUENTA if it exists.
    cols_to_keep_for_train = [
        TEXTO_MODELO, CATEGORIA_STD, SUBCATEGORIA_STD, COMERCIO_STD,
        IMPORTE_STD, AÑO_STD, MES_STD, DIA_STD, CONCEPTO_STD
    ]
    # Add account column if present, ensuring the standard name
    cuenta_col_db_std = next((c for c in [CUENTA_COL_STD, CUENTA_COL_ORIG.upper(), 'CUENTA'] if c in df_train.columns), None)
    if cuenta_col_db_std:
        if cuenta_col_db_std not in cols_to_keep_for_train:
             cols_to_keep_for_train.append(cuenta_col_db_std)
        # Rename to CUENTA_COL_STD if found under a different name and std name doesn't exist
        if cuenta_col_db_std != CUENTA_COL_STD and CUENTA_COL_STD not in df_train.columns:
             df_train = df_train.rename(columns={cuenta_col_db_std: CUENTA_COL_STD})
             # Update the name in cols_to_keep if it was added with the old name
             if cuenta_col_db_std in cols_to_keep_for_train:
                  cols_to_keep_for_train.remove(cuenta_col_db_std)
                  if CUENTA_COL_STD not in cols_to_keep_for_train:
                      cols_to_keep_for_train.append(CUENTA_COL_STD)
        elif cuenta_col_db_std != CUENTA_COL_STD and CUENTA_COL_STD in df_train.columns:
             st.warning(f"BD tiene '{cuenta_col_db_std}' y '{CUENTA_COL_STD}'. Usando '{CUENTA_COL_STD}'.")
             # Remove the non-standard one from list if present
             if cuenta_col_db_std in cols_to_keep_for_train: cols_to_keep_for_train.remove(cuenta_col_db_std)


    # Ensure all desired columns exist before selecting
    final_cols = [col for col in cols_to_keep_for_train if col in df_train.columns]

    # Return only the necessary, cleaned columns
    return df_train[final_cols]


def get_last_transaction_dates(df_db):
    """Calculates the last transaction date for each account in the DB."""
    if df_db is None or df_db.empty:
        # Return empty DF with expected columns for consistency downstream
        return pd.DataFrame(columns=['Cuenta', 'Ultima Fecha'])

    df = df_db.copy()
    # Standardize column names just in case
    df.columns = [str(col).upper().strip() for col in df.columns]

    # --- Find the account column ---
    # Prioritize the standardized ORIG_CUENTA, then common variations
    cuenta_col = next((c for c in [CUENTA_COL_STD, CUENTA_COL_ORIG.upper(), 'CUENTA'] if c in df.columns), None)

    # --- Check required columns ---
    date_cols = [AÑO_STD, MES_STD, DIA_STD]
    if not cuenta_col:
        # Don't display error here, return specific message DF
        return pd.DataFrame({'Mensaje': ["No se encontró columna de cuenta (e.g., ORIG_CUENTA, CUENTA)"]}, index=[0])
    if not all(c in df.columns for c in date_cols):
        missing_dates = [c for c in date_cols if c not in df.columns]
        return pd.DataFrame({'Mensaje': [f"Faltan columnas de fecha ({', '.join(missing_dates)})"]}, index=[0])

    try:
        # --- Prepare Date Column ---
        # Ensure date parts are numeric, fill errors with 0 (will lead to invalid date)
        for col in date_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64').fillna(0)

        # Build date string, padding month and day ('YYYY-MM-DD')
        # Handle potential leading zeros in year if < 1000, though unlikely
        year_str = df[AÑO_STD].astype(int).astype(str).str.zfill(4)
        month_str = df[MES_STD].astype(int).astype(str).str.zfill(2)
        day_str = df[DIA_STD].astype(int).astype(str).str.zfill(2)
        date_str_series = year_str + '-' + month_str + '-' + day_str

        # Convert to datetime, coercing errors (like '0000-00-00' or invalid dates) to NaT
        df['FECHA_COMPLETA_TEMP'] = pd.to_datetime(date_str_series, format='%Y-%m-%d', errors='coerce')

        # Drop rows where the date could not be parsed (NaT)
        df_valid_dates = df.dropna(subset=['FECHA_COMPLETA_TEMP', cuenta_col]) # Also drop if account is missing

        if df_valid_dates.empty:
             return pd.DataFrame({'Mensaje': ["No hay fechas o cuentas válidas en la BD para calcular resumen"]}, index=[0])

        # --- Find Last Transaction per Account ---
        # Ensure account column is string for grouping
        df_valid_dates[cuenta_col] = df_valid_dates[cuenta_col].astype(str)
        # Use idxmax() on the valid datetime column to get the index of the max date per group
        # Group by the account column found earlier
        last_indices = df_valid_dates.loc[df_valid_dates.groupby(cuenta_col)['FECHA_COMPLETA_TEMP'].idxmax()]

        # Format the date for display
        last_indices['Ultima Fecha'] = last_indices['FECHA_COMPLETA_TEMP'].dt.strftime('%Y-%m-%d')

        # Select and rename columns for the final summary DataFrame
        summary_df = last_indices[[cuenta_col, 'Ultima Fecha']].rename(columns={cuenta_col: 'Cuenta'})

        # Sort by account name for consistent display
        return summary_df.sort_values(by='Cuenta').reset_index(drop=True)

    except KeyError as ke:
         # This error indicates a column expected (like cuenta_col or date parts) was not found during processing
         st.error(f"Error de Clave generando resumen: Falta la columna {ke}. Columnas disponibles: {df.columns.tolist()}")
         return pd.DataFrame({'Error': [f"Falta columna requerida: {ke}"]})
    except Exception as e:
         st.error(f"Error inesperado generando resumen de últimas fechas: {e}")
         st.error(traceback.format_exc())
         return pd.DataFrame({'Error': [f"Error interno: {e}"]})


# ------------------------------------------------------------------------------------
# --- Streamlit UI ---
st.set_page_config(layout="wide", page_title="Categorizador Bancario")
st.title("🏦 Categorizador Bancario Inteligente v5.4")
st.caption(f"Usando Archivo de Configuración: `{CONFIG_FILENAME}` | Base de Datos: `{DB_FILENAME}`")

# --- Carga BD (Sidebar) ---
st.sidebar.header("Base de Datos Global")
uploaded_db_file = st.sidebar.file_uploader(f"Cargar BD ({DB_FILENAME})", type=["csv", "xlsx", "xls"], key="db_loader", help=f"Carga la base de datos acumulada existente '{DB_FILENAME}'. Debe contener columnas como {CONCEPTO_STD}, {IMPORTE_STD}, {AÑO_STD}, {MES_STD}, {DIA_STD} y una columna de categoría ({CATEGORIA_PREDICHA} o {CATEGORIA_STD}).")

if uploaded_db_file:
    db_uploader_key = "db_loader_processed_id" # Use this key to track processing
    # Check if this is a new file upload instance
    if uploaded_db_file.file_id != st.session_state.get(db_uploader_key, None):
        st.sidebar.info(f"Cargando '{uploaded_db_file.name}'...")
        df_db_loaded, _ = read_uploaded_file(uploaded_db_file) # Use the robust reader

        if df_db_loaded is not None and not df_db_loaded.empty:
            df_db_loaded_cleaned = df_db_loaded.copy()
            df_db_loaded_cleaned.columns = [str(col).upper().strip() for col in df_db_loaded_cleaned.columns] # Standardize cols

            # --- Validate essential DB columns ---
            # Find category column (_PREDICHA preferred for consistency)
            cat_col_loaded = next((c for c in [CATEGORIA_PREDICHA, CATEGORIA_STD, 'CATEGORIA', 'CATEGORÍA', 'CATEGORIA_X'] if c in df_db_loaded_cleaned.columns), None)
            if not cat_col_loaded:
                 st.sidebar.error(f"Error BD: No se encontró columna de categoría en '{uploaded_db_file.name}'.")
                 df_db_loaded_cleaned = None # Invalidate
            elif cat_col_loaded != CATEGORIA_PREDICHA:
                 # If we found a category, but it's not _PREDICHA, rename it to _PREDICHA for consistency in editing/display
                 # If _PREDICHA already exists, this won't overwrite, which is fine.
                 if CATEGORIA_PREDICHA not in df_db_loaded_cleaned.columns:
                     df_db_loaded_cleaned = df_db_loaded_cleaned.rename(columns={cat_col_loaded: CATEGORIA_PREDICHA})
                     st.sidebar.info(f"Usando columna '{cat_col_loaded}' como '{CATEGORIA_PREDICHA}'.")
                 # Else: If _PREDICHA exists and we also found another (e.g. _STD), prefer _PREDICHA
                 # elif CATEGORIA_PREDICHA in df_db_loaded_cleaned.columns:
                 #     st.sidebar.info(f"Se encontró '{cat_col_loaded}' pero se usará la columna '{CATEGORIA_PREDICHA}' existente.")


            # Check other essential columns (using STD names as base requirement)
            essential_db_cols = [CONCEPTO_STD, IMPORTE_STD, AÑO_STD, MES_STD, DIA_STD]
            missing_db_cols = [col for col in essential_db_cols if col not in df_db_loaded_cleaned.columns] if df_db_loaded_cleaned is not None else essential_db_cols

            if df_db_loaded_cleaned is not None and not missing_db_cols:
                 # --- Ensure other standard/predicted columns exist ---
                 # Prefer _PREDICHA for subcat/comercio, map from _STD if needed, else create empty
                 map_to_pred = {
                     SUBCATEGORIA_PREDICHA: SUBCATEGORIA_STD,
                     COMERCIO_PREDICHO: COMERCIO_STD
                 }
                 for pred_col, std_col in map_to_pred.items():
                     if pred_col not in df_db_loaded_cleaned.columns:
                          if std_col in df_db_loaded_cleaned.columns:
                               df_db_loaded_cleaned = df_db_loaded_cleaned.rename(columns={std_col: pred_col})
                               st.sidebar.info(f"Mapeando '{std_col}' a '{pred_col}' en BD cargada.")
                          else:
                               df_db_loaded_cleaned[pred_col] = ''

                 # Find and standardize account column to ORIG_CUENTA
                 cuenta_col_db = next((c for c in [CUENTA_COL_STD, CUENTA_COL_ORIG.upper(), 'CUENTA'] if c in df_db_loaded_cleaned.columns), None)
                 if cuenta_col_db and cuenta_col_db != CUENTA_COL_STD:
                      # Rename to standard only if standard doesn't exist
                      if CUENTA_COL_STD not in df_db_loaded_cleaned.columns:
                         df_db_loaded_cleaned = df_db_loaded_cleaned.rename(columns={cuenta_col_db: CUENTA_COL_STD})
                         st.sidebar.info(f"Mapeando '{cuenta_col_db}' a '{CUENTA_COL_STD}' en BD cargada.")
                      else: # Both exist, prefer standard name
                           st.sidebar.warning(f"BD tiene '{cuenta_col_db}' y '{CUENTA_COL_STD}'. Se usará '{CUENTA_COL_STD}'.")
                           # Optionally drop the non-standard one? df_db_loaded_cleaned = df_db_loaded_cleaned.drop(columns=[cuenta_col_db])
                 elif not cuenta_col_db and CUENTA_COL_STD not in df_db_loaded_cleaned.columns:
                      df_db_loaded_cleaned[CUENTA_COL_STD] = '' # Add empty standard account column if none found

                 # Fill NaNs in key text columns used for display/editing/processing
                 # Include all potential key columns
                 cols_to_fillna = [
                     CATEGORIA_PREDICHA, SUBCATEGORIA_PREDICHA, COMERCIO_PREDICHO,
                     CONCEPTO_STD, CUENTA_COL_STD,
                     CATEGORIA_STD, SUBCATEGORIA_STD, COMERCIO_STD # Also fill std versions if they exist
                 ]
                 for col in cols_to_fillna:
                      if col in df_db_loaded_cleaned.columns:
                          # Use fillna('') for object/string columns
                          if pd.api.types.is_object_dtype(df_db_loaded_cleaned[col].dtype) or pd.api.types.is_string_dtype(df_db_loaded_cleaned[col].dtype):
                             df_db_loaded_cleaned[col] = df_db_loaded_cleaned[col].fillna('')
                          # Consider filling numeric NaNs with 0 if appropriate, but usually handled later
                          # Example: if IMPORTE_STD in df_db_loaded_cleaned.columns: df_db_loaded_cleaned[IMPORTE_STD] = df_db_loaded_cleaned[IMPORTE_STD].fillna(0)

                 # --- Store loaded data and update state ---
                 st.session_state.accumulated_data = df_db_loaded_cleaned
                 st.session_state[db_uploader_key] = uploaded_db_file.file_id # Mark as processed
                 st.sidebar.success(f"BD '{uploaded_db_file.name}' cargada ({len(df_db_loaded_cleaned)} filas).")

                 # --- Extract Knowledge from newly loaded DB ---
                 # Use the prepared DB data
                 st.sidebar.info("Extrayendo conocimiento de la BD cargada...")
                 # Prepare the DB using the dedicated function which ensures CATEGORIA_STD exists
                 df_for_knowledge = parse_accumulated_db_for_training(st.session_state.accumulated_data.copy())

                 if df_for_knowledge is not None:
                     st.session_state.learned_knowledge = extract_knowledge_std(df_for_knowledge)
                     knowledge_ok = bool(st.session_state.learned_knowledge.get('categorias'))
                     st.session_state.knowledge_loaded = knowledge_ok
                     if knowledge_ok:
                         st.sidebar.success("Conocimiento extraído/actualizado desde BD.")
                     else:
                          st.sidebar.warning("Se cargó la BD, pero no se pudo extraer conocimiento útil (¿faltan categorías válidas?).")
                 else:
                      st.sidebar.warning("Se cargó la BD, pero falló la preparación para extraer conocimiento.")

                 # Rerun to update UI reflecting loaded data/knowledge
                 st.rerun()

            elif df_db_loaded_cleaned is not None: # Missing essential columns
                 st.sidebar.error(f"Error BD: Archivo '{uploaded_db_file.name}' inválido. Faltan columnas esenciales: {', '.join(missing_db_cols)}")
                 st.session_state[db_uploader_key] = None # Reset processed ID
            # Else: read_uploaded_file handled the error message if df_db_loaded_cleaned is None

        elif df_db_loaded is not None and df_db_loaded.empty:
             st.sidebar.warning(f"El archivo BD '{uploaded_db_file.name}' está vacío.")
             st.session_state[db_uploader_key] = None # Reset processed ID
        else:
             # Error message already shown by read_uploaded_file
             st.session_state[db_uploader_key] = None # Reset processed ID

# --- Tabs Principales ---
tab1, tab2 = st.tabs(["⚙️ Configuración y Entrenamiento", "📊 Categorización y Gestión BD"])

# ==============================================================
# --- Tab 1: Configuración y Entrenamiento ---
# ==============================================================
with tab1:
    st.header("Configuración y Entrenamiento")
    col1a, col1b = st.columns([1, 2]) # Adjust column widths if needed

    # --- Columna Izquierda: Configuración y Entrenamiento ---
    with col1a:
        st.subheader("Cargar/Descargar Configuración")
        st.write(f"Gestiona el archivo `{CONFIG_FILENAME}` que contiene los mapeos de bancos y el conocimiento aprendido.")

        # --- Carga de Configuración ---
        uploaded_config_file_f1 = st.file_uploader(f"Cargar `{CONFIG_FILENAME}`", type="json", key="config_loader_f1", help="Carga un archivo JSON con mapeos y conocimiento previamente guardado.")
        if uploaded_config_file_f1:
            config_uploader_key = "config_loader_processed_id_f1"
            if uploaded_config_file_f1.file_id != st.session_state.get(config_uploader_key, None):
                try:
                    # Load JSON data
                    config_data = json.load(uploaded_config_file_f1)
                    is_valid_config = True
                    error_msg = ""

                    # Validate structure
                    if not isinstance(config_data, dict):
                        is_valid_config = False; error_msg = "Archivo no es un diccionario JSON."
                    else:
                        if 'bank_mappings' not in config_data or not isinstance(config_data['bank_mappings'], dict):
                            is_valid_config = False; error_msg += " Falta sección 'bank_mappings' o no es un diccionario."
                        if 'learned_knowledge' not in config_data or not isinstance(config_data['learned_knowledge'], dict):
                             is_valid_config = False; error_msg += " Falta sección 'learned_knowledge' o no es un diccionario."
                        elif not all(key in config_data['learned_knowledge'] for key in ['categorias', 'subcategorias_por_cat', 'comercios_por_cat', 'subcat_unica_por_comercio_y_cat', 'subcat_mas_frecuente_por_comercio_y_cat', 'all_subcategories', 'all_comercios']):
                             is_valid_config = False; error_msg += " Sección 'learned_knowledge' incompleta."

                    # If valid, update session state
                    if is_valid_config:
                        st.session_state.bank_mappings = config_data.get('bank_mappings', {})
                        st.session_state.learned_knowledge = config_data.get('learned_knowledge', st.session_state.learned_knowledge) # Keep existing if missing
                        # Update knowledge loaded status based on loaded categories
                        knowledge_ok_config = bool(st.session_state.learned_knowledge.get('categorias'))
                        st.session_state.knowledge_loaded = knowledge_ok_config
                        st.session_state[config_uploader_key] = uploaded_config_file_f1.file_id # Mark as processed
                        st.success(f"Configuración '{uploaded_config_file_f1.name}' cargada.")
                        st.sidebar.success("Configuración Cargada.") # Update sidebar status
                        if not st.session_state.model_trained and knowledge_ok_config:
                             st.info("Conocimiento cargado. Considera entrenar el modelo si vas a categorizar nuevos archivos.")
                        elif not knowledge_ok_config:
                             st.warning("Configuración cargada, pero sección 'learned_knowledge' vacía o inválida.")
                        st.rerun() # Rerun to reflect loaded state in UI
                    else:
                        st.error(f"Error en formato del archivo de configuración: {error_msg.strip()}")
                        st.session_state[config_uploader_key] = None # Reset processed ID

                except json.JSONDecodeError:
                    st.error(f"Error: El archivo '{uploaded_config_file_f1.name}' no es un JSON válido.")
                    st.session_state[config_uploader_key] = None
                except Exception as e:
                    st.error(f"Error inesperado al cargar configuración: {e}")
                    st.error(traceback.format_exc())
                    st.session_state[config_uploader_key] = None

        # --- Descarga de Configuración ---
        # Enable download if there's something to download
        if st.session_state.bank_mappings or st.session_state.knowledge_loaded:
            try:
                # Prepare data to save
                config_to_save = {
                    'bank_mappings': st.session_state.get('bank_mappings', {}),
                    'learned_knowledge': st.session_state.get('learned_knowledge', {}) # Include current knowledge
                }
                # Convert to JSON string
                config_json = json.dumps(config_to_save, indent=4, ensure_ascii=False)
                # Download button
                st.download_button(
                    label=f"💾 Descargar Config Actual",
                    data=config_json.encode('utf-8'), # Encode to bytes
                    file_name=CONFIG_FILENAME,
                    mime='application/json',
                    key='download_config_f1'
                )
            except Exception as e:
                st.error(f"Error preparando config para descarga: {e}")
        else:
            st.info("No hay configuración (mapeos o conocimiento) en memoria para descargar.")

        st.divider()

        # --- Entrenamiento del Modelo ---
        st.subheader("(Re)Entrenar Modelo")
        st.write("Entrena el clasificador de categorías usando la Base de Datos Acumulada cargada.")

        if st.session_state.accumulated_data.empty:
            st.warning("Carga la Base de Datos Acumulada en la barra lateral para poder entrenar.")
        else:
            if st.button("🧠 Entrenar/Reentrenar con BD", key="train_db_f1b", help="Usa la BD cargada para aprender categorías y (re)entrenar el modelo."):
                 with st.spinner("Preparando datos de BD y entrenando..."):
                    # 1. Parse/Prepare the accumulated DB for training
                    # This function now ensures CATEGORIA_STD is the target column name
                    df_to_train = parse_accumulated_db_for_training(st.session_state.accumulated_data.copy())

                    if df_to_train is not None and not df_to_train.empty:
                        st.success(f"Datos de BD preparados para entrenamiento ({len(df_to_train)} filas).")

                        # 2. Re-extract knowledge from the *prepared* training data
                        # This ensures knowledge matches exactly what the model is trained on
                        # extract_knowledge_std uses CATEGORIA_STD (or _PREDICHA if _STD missing)
                        st.session_state.learned_knowledge = extract_knowledge_std(df_to_train)
                        knowledge_ok_train = bool(st.session_state.learned_knowledge.get('categorias'))
                        st.session_state.knowledge_loaded = knowledge_ok_train
                        if knowledge_ok_train:
                             st.sidebar.success("Conocimiento Actualizado (BD)") # Update sidebar
                        else:
                             st.sidebar.warning("Conocimiento no actualizado (¿sin categorías válidas en BD?).")


                        # 3. Train the classifier (uses CATEGORIA_STD)
                        model, vectorizer, report = train_classifier_std(df_to_train)

                        # 4. Update session state based on training outcome
                        if model and vectorizer:
                            st.session_state.model = model
                            st.session_state.vectorizer = vectorizer
                            st.session_state.model_trained = True
                            st.session_state.training_report = report if report else "Modelo entrenado (sin informe detallado)."
                            st.success(f"¡Modelo entrenado exitosamente!")
                            st.sidebar.success("✅ Modelo Entrenado") # Update sidebar status

                            # Display training report in sidebar expander
                            st.sidebar.divider()
                            st.sidebar.subheader("Evaluación Modelo (Entrenamiento)")
                            with st.sidebar.expander("Ver Informe de Clasificación"):
                                st.text(st.session_state.training_report)

                        else:
                            # Training failed, update state accordingly
                            st.session_state.model = None
                            st.session_state.vectorizer = None
                            st.session_state.model_trained = False
                            # Use the report from train_classifier_std which contains the error
                            st.session_state.training_report = report if report else "Fallo entrenamiento (razón desconocida)."
                            st.error(f"Fallo el entrenamiento del modelo.")
                            st.sidebar.error("❌ Entrenamiento Fallido") # Update sidebar status
                            st.sidebar.divider()
                            st.sidebar.subheader("Evaluación Modelo (Entrenamiento)")
                            st.sidebar.text(st.session_state.training_report) # Show error in sidebar too
                    else:
                        # Failed to prepare data from DB
                        st.error("No se pudieron preparar datos válidos desde la BD Acumulada para entrenar.")
                        st.session_state.model_trained = False # Ensure model state is false

        st.divider()

        # --- Mostrar Conocimiento Base Actual ---
        st.subheader("Conocimiento Base Actual")
        knowledge_display = st.session_state.get('learned_knowledge', {})
        if knowledge_display and st.session_state.get('knowledge_loaded', False):
             categorias_list = knowledge_display.get('categorias', [])
             all_subs_list = knowledge_display.get('all_subcategories', [])
             all_coms_list = knowledge_display.get('all_comercios', [])

             # Capitalize for display purposes
             categorias_display = sorted([c.capitalize() for c in categorias_list])
             all_subs_display = sorted([s.capitalize() for s in all_subs_list])
             all_coms_display = sorted([c.capitalize() for c in all_coms_list])


             st.write(f"**Categorías Conocidas ({len(categorias_display)}):**")
             if categorias_display:
                  st.dataframe(pd.DataFrame(categorias_display, columns=['Categoría']), use_container_width=True, height=150, hide_index=True)
             else:
                  st.caption("Ninguna")

             col_k1, col_k2 = st.columns(2)
             with col_k1:
                  with st.expander(f"Subcategorías Conocidas ({len(all_subs_display)})", expanded=False):
                       if all_subs_display:
                            st.dataframe(pd.DataFrame(all_subs_display, columns=['Subcategoría']), use_container_width=True, height=200, hide_index=True)
                       else:
                            st.caption("Ninguna")
             with col_k2:
                   with st.expander(f"Comercios Conocidos ({len(all_coms_display)})", expanded=False):
                       if all_coms_display:
                            st.dataframe(pd.DataFrame(all_coms_display, columns=['Comercio']), use_container_width=True, height=200, hide_index=True)
                       else:
                            st.caption("Ninguno")

             # --- CORRECTED DEBUG EXPANDER ---
             with st.expander("Mostrar detalles avanzados del conocimiento (Debug)", expanded=False):
                 st.write("Subcategorías por Categoría:")
                 st.json(knowledge_display.get('subcategorias_por_cat', {}))
                 st.write("Comercios por Categoría:")
                 st.json(knowledge_display.get('comercios_por_cat', {}))
                 st.write("Regla Subcategoría Única (por Comercio y Cat):")
                 st.json(knowledge_display.get('subcat_unica_por_comercio_y_cat', {}))
                 st.write("Regla Subcategoría Más Frecuente (por Comercio y Cat):")
                 st.json(knowledge_display.get('subcat_mas_frecuente_por_comercio_y_cat', {}))
             # --- END CORRECTION ---

        else:
             st.info("No hay conocimiento base cargado o aprendido todavía. Carga una configuración, carga una BD o entrena el modelo.")


    # --- Columna Derecha: Definir Mapeos Bancarios ---
    with col1b:
        st.subheader("Definir Formatos Bancarios (Mapeo)")
        st.write("Enseña a la aplicación cómo leer archivos de diferentes bancos subiendo un archivo de ejemplo y mapeando sus columnas a los campos estándar.")

        # --- Selección del Banco y Carga de Ejemplo ---
        # Allow adding new bank names
        existing_banks = list(st.session_state.get('bank_mappings', {}).keys())
        # Provide a default list + learned banks, sorted
        bank_options = sorted(list(set(["SANTANDER", "EVO", "WIZINK", "AMEX", "N26", "BBVA", "ING", "CAIXABANK"] + existing_banks)))
        # Use text input for potentially new banks, or select existing
        new_bank_name = st.text_input("Nombre del Banco (o selecciona existente abajo):", key="new_bank_name_f2", placeholder="Ej: MI_BANCO_NUEVO").strip().upper()
        selected_bank_learn = st.selectbox("Banco Existente:", [""] + bank_options, key="bank_learn_f2_select", index=0, format_func=lambda x: x if x else "---Selecciona---") # Add empty option


        # Determine the bank name to use for mapping
        bank_to_map = new_bank_name if new_bank_name else selected_bank_learn
        if not bank_to_map:
             st.info("Escribe un nombre para un banco nuevo o selecciona uno existente para definir/editar su mapeo.")
        else:
            st.markdown(f"**Editando mapeo para: `{bank_to_map}`**")
            # Use a unique key for the file uploader based on the bank name being mapped
            # This prevents issues if the user switches banks after uploading a file
            uploader_key = f"sample_uploader_{bank_to_map}"
            uploaded_sample_file = st.file_uploader(f"Cargar archivo de ejemplo de `{bank_to_map}`", type=["csv", "xlsx", "xls"], key=uploader_key)

            if uploaded_sample_file:
                # Generate a unique ID for the current upload instance
                # file_instance_id = f"{uploaded_sample_file.file_id}_{bank_to_map}"
                # Check if we've already processed this specific file instance for this bank
                # processed_key = f"processed_{uploader_key}" # Key to track processing

                # Rerun logic might re-trigger processing. Let's simplify: Process whenever file is present.
                # If the state needs to be preserved across reruns for *this specific upload*, more complex state needed.
                # For now, assume re-uploading or changing bank clears the previous sample view.

                df_sample, detected_columns = read_uploaded_file(uploaded_sample_file)

                if df_sample is not None and detected_columns:
                    st.write(f"Primeras filas del ejemplo '{uploaded_sample_file.name}':")
                    st.dataframe(df_sample.head(3), use_container_width=True)
                    st.write(f"Columnas detectadas en el archivo:")
                    st.code(f"{detected_columns}", language=None)

                    st.write("**Define el Mapeo:** Asocia las columnas del archivo a los campos estándar.")
                    # Get saved mapping for this bank, or provide default structure
                    # Use .get() defensively for keys that might not exist in older configs
                    saved_mapping = st.session_state.bank_mappings.get(bank_to_map, {})
                    saved_mapping_cols = saved_mapping.get('columns', {}) # Default to empty dict if 'columns' key missing

                    # Options for select boxes: None + detected columns
                    cols_with_none = [None] + detected_columns

                    # --- Widget Definitions ---
                    # Use unique keys involving the bank_to_map to avoid state conflicts if user switches banks
                    sub_c1, sub_c2 = st.columns(2)
                    with sub_c1:
                        st.markdown("**Columnas Esenciales:**")
                        # Function to find index safely
                        def get_index(col_name):
                            val = saved_mapping_cols.get(col_name)
                            try:
                                return cols_with_none.index(val) if val in cols_with_none else 0
                            except ValueError:
                                return 0 # Should not happen if val is None or in detected_columns

                        # Concepto
                        map_concepto = st.selectbox(f"`{CONCEPTO_STD}` (Descripción Gasto/Ingreso)", cols_with_none, index=get_index(CONCEPTO_STD), key=f"map_{CONCEPTO_STD}_{bank_to_map}", format_func=lambda x: x if x else "---Selecciona---")
                        # Importe
                        map_importe = st.selectbox(f"`{IMPORTE_STD}` (Cantidad Numérica)", cols_with_none, index=get_index(IMPORTE_STD), key=f"map_{IMPORTE_STD}_{bank_to_map}", format_func=lambda x: x if x else "---Selecciona---")

                        st.markdown("**Manejo de Fecha:**")
                        # Checkbox to choose between single date column or Año/Mes/Día
                        is_single_date_saved = FECHA_STD in saved_mapping_cols and saved_mapping_cols.get(FECHA_STD) is not None
                        map_single_date = st.checkbox("La fecha está en una sola columna", value=is_single_date_saved, key=f"map_single_date_{bank_to_map}")

                        map_fecha_unica=None; map_formato_fecha=None; map_año=None; map_mes=None; map_dia=None # Initialize
                        if map_single_date:
                            map_fecha_unica = st.selectbox(f"`{FECHA_STD}` (Columna con fecha completa)", cols_with_none, index=get_index(FECHA_STD), key=f"map_{FECHA_STD}_{bank_to_map}", format_func=lambda x: x if x else "---Selecciona---")
                            # Use current saved format, provide examples
                            map_formato_fecha = st.text_input("Formato Fecha (ej: %d/%m/%Y, %Y-%m-%d)", value=saved_mapping.get('date_format', ''), key=f"map_date_format_{bank_to_map}", help="Usa códigos Python: %d día, %m mes, %Y año (4 dig), %y año (2 dig). Ver docs de `strftime`.")
                        else:
                            map_año = st.selectbox(f"`{AÑO_STD}` (Columna Año)", cols_with_none, index=get_index(AÑO_STD), key=f"map_{AÑO_STD}_{bank_to_map}", format_func=lambda x: x if x else "---Selecciona---")
                            map_mes = st.selectbox(f"`{MES_STD}` (Columna Mes)", cols_with_none, index=get_index(MES_STD), key=f"map_{MES_STD}_{bank_to_map}", format_func=lambda x: x if x else "---Selecciona---")
                            map_dia = st.selectbox(f"`{DIA_STD}` (Columna Día)", cols_with_none, index=get_index(DIA_STD), key=f"map_{DIA_STD}_{bank_to_map}", format_func=lambda x: x if x else "---Selecciona---")

                    with sub_c2:
                        st.markdown("**Columnas Opcionales:**")
                        # Comercio
                        map_comercio = st.selectbox(f"`{COMERCIO_STD}` (Nombre Comercio/Tienda, si existe)", cols_with_none, index=get_index(COMERCIO_STD), key=f"map_{COMERCIO_STD}_{bank_to_map}", format_func=lambda x: x if x else "---Ninguna---")

                        st.markdown("**Formato de Importe:**")
                        # Separadores
                        val_map_decimal_sep = st.text_input("Separador Decimal (usualmente ',' o '.')", value=saved_mapping.get('decimal_sep', ','), key=f"map_decimal_{bank_to_map}", max_chars=1)
                        val_map_thousands_sep = st.text_input("Separador Miles (si existe, ej: '.' o ',', dejar vacío si no hay)", value=saved_mapping.get('thousands_sep', '') or '', key=f"map_thousands_{bank_to_map}", max_chars=1)


                    # --- Botón de Guardado y Lógica ---
                    if st.button(f"💾 Guardar Mapeo para `{bank_to_map}`", key=f"save_mapping_{bank_to_map}"):
                        # --- Read current widget values AT THE TIME OF CLICK ---
                        # Use .get() for safety if widgets could be None (though selectbox returns selection or None)
                        current_concepto = map_concepto
                        current_importe = map_importe
                        current_is_single_date = map_single_date
                        current_fecha_unica = map_fecha_unica
                        current_formato_fecha = map_formato_fecha.strip() if map_formato_fecha else ""
                        current_año = map_año
                        current_mes = map_mes
                        current_dia = map_dia
                        current_comercio = map_comercio
                        current_decimal_sep = val_map_decimal_sep.strip() if val_map_decimal_sep else "," # Default to comma if empty
                        current_thousands_sep = val_map_thousands_sep.strip() if val_map_thousands_sep else None # Use None if empty

                        # --- Build the mapping dictionary to save ---
                        final_mapping_cols = {}
                        if current_concepto: final_mapping_cols[CONCEPTO_STD] = current_concepto
                        if current_importe: final_mapping_cols[IMPORTE_STD] = current_importe
                        if current_is_single_date and current_fecha_unica: final_mapping_cols[FECHA_STD] = current_fecha_unica
                        if not current_is_single_date and current_año: final_mapping_cols[AÑO_STD] = current_año
                        if not current_is_single_date and current_mes: final_mapping_cols[MES_STD] = current_mes
                        if not current_is_single_date and current_dia: final_mapping_cols[DIA_STD] = current_dia
                        if current_comercio: final_mapping_cols[COMERCIO_STD] = current_comercio

                        # --- Validation ---
                        is_valid = True
                        errors = []
                        if not final_mapping_cols.get(CONCEPTO_STD):
                            errors.append(f"Debes mapear la columna para `{CONCEPTO_STD}`.")
                            is_valid = False
                        if not final_mapping_cols.get(IMPORTE_STD):
                            errors.append(f"Debes mapear la columna para `{IMPORTE_STD}`.")
                            is_valid = False
                        if not current_decimal_sep:
                            errors.append("Debes especificar el Separador Decimal (normalmente ',' o '.').")
                            is_valid = False

                        if current_is_single_date:
                            if not final_mapping_cols.get(FECHA_STD):
                                errors.append(f"Si la fecha está en una columna, debes mapear `{FECHA_STD}`.")
                                is_valid = False
                            elif not current_formato_fecha:
                                errors.append("Si la fecha está en una columna, debes especificar el Formato Fecha.")
                                is_valid = False
                            # Simple format validation (presence of %d, %m, %Y or %y)
                            elif not ('%d' in current_formato_fecha and '%m' in current_formato_fecha and ('%Y' in current_formato_fecha or '%y' in current_formato_fecha)):
                                 errors.append("Formato Fecha parece inválido. Debe contener al menos %d, %m, y %Y (o %y). Ejemplo: %d/%m/%Y")
                                 is_valid = False
                        else: # Separate A/M/D columns needed
                            if not final_mapping_cols.get(AÑO_STD): errors.append(f"Si la fecha está separada, debes mapear `{AÑO_STD}`."); is_valid = False
                            if not final_mapping_cols.get(MES_STD): errors.append(f"Si la fecha está separada, debes mapear `{MES_STD}`."); is_valid = False
                            if not final_mapping_cols.get(DIA_STD): errors.append(f"Si la fecha está separada, debes mapear `{DIA_STD}`."); is_valid = False

                        # --- Save if Valid ---
                        if is_valid:
                            mapping_to_save = {
                                'bank_name': bank_to_map,
                                'columns': final_mapping_cols,
                                'decimal_sep': current_decimal_sep,
                                'thousands_sep': current_thousands_sep, # Store None if empty/not provided
                            }
                            if current_is_single_date:
                                mapping_to_save['date_format'] = current_formato_fecha

                            st.session_state.bank_mappings[bank_to_map] = mapping_to_save
                            st.success(f"¡Mapeo para `{bank_to_map}` guardado/actualizado!")
                            st.info("Puedes descargar la configuración actualizada desde el panel izquierdo.")
                            # Optionally clear the sample file uploader state after saving?
                            # st.session_state[uploader_key] = None # Might cause rerun issues if not careful
                            # No rerun here, allow user to continue editing or move on. Changes are saved in state.
                        else:
                            # Display all errors found
                            st.error("No se pudo guardar el mapeo. Corrige los siguientes errores:")
                            for error in errors:
                                st.error(f"- {error}")


                elif df_sample is not None and not detected_columns:
                     st.warning(f"El archivo de ejemplo '{uploaded_sample_file.name}' se leyó pero no se detectaron columnas o está vacío.")
                # Else: read_uploaded_file handled the error message if df_sample is None


# ==============================================================
# --- Tab 2: Categorización y Gestión BD ---
# ==============================================================
with tab2:
    st.header("Categorización y Gestión de la Base de Datos")

    # --- Resumen Última Transacción ---
    st.subheader("Resumen: Última Transacción por Cuenta")
    with st.spinner("Calculando resumen..."):
        # Pass a copy to avoid potential mutation issues if get_last_transaction_dates modifies it
        df_summary = get_last_transaction_dates(st.session_state.get('accumulated_data', pd.DataFrame()).copy())
    if not df_summary.empty:
        # Check if the result is an error/message or actual data
        if 'Error' in df_summary.columns or 'Mensaje' in df_summary.columns:
             st.info(df_summary.iloc[0, 0]) # Display the message/error
        else:
             st.dataframe(df_summary, use_container_width=True, hide_index=True)
    # else: # Function returned empty DF likely due to empty input DB or no valid rows
    #     st.info("No hay datos en la Base de Datos Acumulada para generar el resumen.") # Message handled by function now


    st.divider()

    # --- Categorización de Nuevos Archivos ---
    st.subheader("Categorizar Nuevos Archivos y Añadir a BD")

    # Check prerequisites for categorization
    model_ready_for_pred = st.session_state.get('model_trained', False)
    mappings_available = bool(st.session_state.get('bank_mappings', {}))
    knowledge_ready = st.session_state.get('knowledge_loaded', False)
    can_categorize = model_ready_for_pred and mappings_available and knowledge_ready

    # Provide clear guidance if prerequisites are not met
    if not can_categorize:
        st.warning("⚠️ **Acción Requerida para Categorizar:**")
        if not knowledge_ready:
            st.markdown("- Falta **Conocimiento Base**. Ve a 'Configuración y Entrenamiento' -> Carga Config/BD o Entrena.")
        if not mappings_available:
            st.markdown("- Faltan **Formatos Bancarios**. Ve a 'Configuración y Entrenamiento' -> Define al menos un mapeo.")
        if not model_ready_for_pred:
            st.markdown("- Falta **Modelo Entrenado**. Ve a 'Configuración y Entrenamiento' -> Entrena con BD.")
    else:
        # Prerequisites met, proceed with UI for categorization
        st.success("✅ Listo para categorizar nuevos archivos.")
        st.write("Selecciona el banco correspondiente al archivo que quieres categorizar y súbelo.")

        available_banks_for_pred = list(st.session_state.bank_mappings.keys())
        if not available_banks_for_pred: # Should not happen due to check above, but safeguard
             st.error("Error Interno: No hay mapeos disponibles aunque el estado indica que sí.")
        else:
            selected_bank_predict = st.selectbox("Banco del archivo a categorizar:", available_banks_for_pred, key="bank_predict_f3", format_func=lambda x: x if x else "---Selecciona---")
            # Use a unique key for the final uploader
            final_uploader_key = f"final_uploader_{selected_bank_predict}" if selected_bank_predict else "final_uploader_f3"
            uploaded_final_file = st.file_uploader(f"Cargar archivo de '{selected_bank_predict}' (sin categorizar)", type=["csv", "xlsx", "xls"], key=final_uploader_key, disabled=(not selected_bank_predict))

            if uploaded_final_file and selected_bank_predict:
                mapping_to_use = st.session_state.bank_mappings.get(selected_bank_predict)
                # Double-check mapping exists (should always exist if selected from list)
                if not mapping_to_use:
                    st.error(f"Error crítico: No se encontró el mapeo para el banco seleccionado '{selected_bank_predict}'.")
                else:
                     st.info(f"Procesando archivo '{uploaded_final_file.name}' usando mapeo para '{selected_bank_predict}'...")
                     df_std_new = None
                     df_pred_display = None # Use this to hold the df for display and potential addition

                     # --- Step 1: Read and Standardize ---
                     with st.spinner(f"Leyendo y estandarizando datos según mapeo '{selected_bank_predict}'..."):
                          df_raw_new, _ = read_uploaded_file(uploaded_final_file)
                          if df_raw_new is not None and not df_raw_new.empty:
                              df_std_new = standardize_data_with_mapping(df_raw_new.copy(), mapping_to_use)
                          elif df_raw_new is not None and df_raw_new.empty:
                               st.warning(f"El archivo '{uploaded_final_file.name}' está vacío.")
                          # Else: read_uploaded_file or standardize handled errors

                     # --- Step 2: Predict Categories and Apply Knowledge ---
                     if df_std_new is not None and not df_std_new.empty:
                          st.success(f"Estandarización completada ({len(df_std_new)} filas).")
                          with st.spinner("Aplicando modelo de IA y conocimiento para categorizar..."):
                              st.session_state.debug_predictions = [] # Reset debug info
                              try:
                                   # Ensure TEXTO_MODELO exists (should be created by standardize)
                                   if TEXTO_MODELO not in df_std_new.columns:
                                       st.error(f"Error Interno: Falta la columna '{TEXTO_MODELO}' después de estandarizar. No se puede categorizar.")
                                   else:
                                        # Prepare data for prediction (handle potential NaNs in text input)
                                        df_pred = df_std_new.dropna(subset=[TEXTO_MODELO]).copy()
                                        df_pred[TEXTO_MODELO] = df_pred[TEXTO_MODELO].fillna('')

                                        if not df_pred.empty:
                                             # Vectorize text using the *loaded/trained* vectorizer
                                             X_new_vec = st.session_state.vectorizer.transform(df_pred[TEXTO_MODELO])
                                             # Predict categories using the *loaded/trained* model
                                             predictions_cat = st.session_state.model.predict(X_new_vec)

                                             # Add predicted category (capitalize for display)
                                             # Use .astype(str) just in case predict returns non-strings
                                             df_pred[CATEGORIA_PREDICHA] = [str(p).capitalize() for p in predictions_cat]

                                             # --- Apply Knowledge for Subcategory and Merchant ---
                                             pred_comercios_final = []
                                             pred_subcats_final = []
                                             knowledge = st.session_state.learned_knowledge # Get current knowledge
                                             debug_info_list = [] # Store debug messages

                                             # Use .iterrows() for row-by-row processing needed here
                                             for index, row in df_pred.iterrows():
                                                 # Get prediction and inputs safely, convert to lower for matching
                                                 pred_cat_lower = str(row.get(CATEGORIA_PREDICHA, '')).lower()
                                                 input_comercio_lower = str(row.get(COMERCIO_STD, '')).lower().strip()
                                                 input_concepto_lower = str(row.get(CONCEPTO_STD, '')).lower().strip()

                                                 debug_step = f"Fila:{index}|CatPred:'{pred_cat_lower}'|ComercioInput:'{input_comercio_lower}'|ConceptoInput:'{input_concepto_lower[:30]}...'" # Start debug string

                                                 # --- Merchant Prediction/Refinement ---
                                                 comercio_final = input_comercio_lower # Default to input
                                                 best_match_comercio = None # Track if fuzzy match was used
                                                 # Get known merchants for the predicted category (lowercase)
                                                 known_comers_for_cat = knowledge.get('comercios_por_cat', {}).get(pred_cat_lower, [])

                                                 if input_comercio_lower and known_comers_for_cat:
                                                     # Find the best fuzzy match above threshold
                                                     try:
                                                          match_result = process.extractOne(input_comercio_lower, known_comers_for_cat)
                                                          if match_result and match_result[1] >= FUZZY_MATCH_THRESHOLD:
                                                              comercio_final = match_result[0] # Use the known merchant name
                                                              best_match_comercio = match_result[0] # Store the matched name
                                                              debug_step += f" -> ComercioMatch:'{comercio_final}'(Score:{match_result[1]})"
                                                     except Exception as e_fuzzy:
                                                          st.warning(f"Error en fuzzy matching para comercio '{input_comercio_lower}': {e_fuzzy}")
                                                          # Continue with original comercio

                                                 # Capitalize the final merchant name for display
                                                 pred_comercios_final.append(str(comercio_final).capitalize())

                                                 # --- Subcategory Prediction/Refinement ---
                                                 subcat_final = '' # Default empty
                                                 # Use the matched merchant name (lowercase) if available, otherwise original input (lowercase)
                                                 comercio_lookup_key = best_match_comercio if best_match_comercio else input_comercio_lower
                                                 subcat_msg = " -> SubCat:" # Start subcat debug message

                                                 # Rule 1: Unique Subcategory by Merchant+Category
                                                 if comercio_lookup_key: # Only if we have a merchant name
                                                    # Access nested dicts safely using .get()
                                                    cat_rules = knowledge.get('subcat_unica_por_comercio_y_cat', {}).get(pred_cat_lower, {})
                                                    subcat_unica = cat_rules.get(comercio_lookup_key)
                                                    if subcat_unica:
                                                        subcat_final = subcat_unica
                                                        subcat_msg += f"'{subcat_final}'(Regla:Única_por_Comercio)"

                                                 # Rule 2: Most Frequent Subcategory by Merchant+Category (if Rule 1 failed)
                                                 if not subcat_final and comercio_lookup_key:
                                                     cat_rules_freq = knowledge.get('subcat_mas_frecuente_por_comercio_y_cat', {}).get(pred_cat_lower, {})
                                                     subcat_frecuente = cat_rules_freq.get(comercio_lookup_key)
                                                     if subcat_frecuente:
                                                         subcat_final = subcat_frecuente
                                                         subcat_msg += f"'{subcat_final}'(Regla:Frecuente_por_Comercio)"

                                                 # Rule 3: Keyword Match in Concept (if Rules 1 & 2 failed)
                                                 if not subcat_final and input_concepto_lower:
                                                      known_subcats_for_cat = knowledge.get('subcategorias_por_cat', {}).get(pred_cat_lower, [])
                                                      # Find known subcats (exact word match, case-insensitive) within the concept text
                                                      found_keywords = [
                                                          sk for sk in known_subcats_for_cat if sk and
                                                          re.search(r'\b' + re.escape(sk) + r'\b', input_concepto_lower, re.IGNORECASE)
                                                      ]
                                                      # Use this rule only if *exactly one* keyword subcategory is found
                                                      if len(found_keywords) == 1:
                                                          subcat_final = found_keywords[0]
                                                          subcat_msg += f"'{subcat_final}'(Regla:Keyword_en_Concepto)"
                                                      elif len(found_keywords) > 1:
                                                           subcat_msg += f"N/A(MúltiplesKeywords:{found_keywords[:3]})" # Show only first few


                                                 # Rule 4: Only one Subcategory known for the entire Category (Fallback if others failed)
                                                 if not subcat_final:
                                                     known_subcats_for_cat = knowledge.get('subcategorias_por_cat', {}).get(pred_cat_lower, [])
                                                     # Use if exactly one subcategory is associated with the predicted category overall
                                                     if len(known_subcats_for_cat) == 1:
                                                         subcat_final = known_subcats_for_cat[0]
                                                         subcat_msg += f"'{subcat_final}'(Regla:Única_en_Categoría)"

                                                 # If no rule applied, subcat_final remains ''
                                                 if not subcat_final and subcat_msg == " -> SubCat:": subcat_msg += "N/A(SinReglaAplicable)"

                                                 pred_subcats_final.append(str(subcat_final).capitalize()) # Capitalize for display
                                                 debug_info_list.append(debug_step + subcat_msg) # Add full debug line

                                             # Add the final predicted/refined columns to the DataFrame
                                             df_pred[COMERCIO_PREDICHO] = pred_comercios_final
                                             df_pred[SUBCATEGORIA_PREDICHA] = pred_subcats_final

                                             # Store debug info
                                             st.session_state.debug_predictions = debug_info_list

                                             # --- Display Results ---
                                             st.subheader(f"📊 Resultados de Categorización para '{uploaded_final_file.name}'")

                                             # Define column order for display, prioritizing predicted/refined cols
                                             display_cols_order = [
                                                 CATEGORIA_PREDICHA, SUBCATEGORIA_PREDICHA, COMERCIO_PREDICHO,
                                                 CONCEPTO_STD, IMPORTE_STD, AÑO_STD, MES_STD, DIA_STD
                                             ]
                                             # Add account column if it exists (using standard name)
                                             if CUENTA_COL_STD in df_pred.columns:
                                                 if CUENTA_COL_STD not in display_cols_order: display_cols_order.append(CUENTA_COL_STD)
                                             # Add all original columns kept during standardization
                                             orig_cols = sorted([c for c in df_pred.columns if c.startswith('ORIG_') and c != CUENTA_COL_STD])
                                             display_cols_order.extend(orig_cols)
                                             # Add any other remaining columns (that aren't internal like TEXTO_MODELO or already included)
                                             internal_or_duplicate = [TEXTO_MODELO, COMERCIO_STD, SUBCATEGORIA_STD, CATEGORIA_STD] + display_cols_order
                                             other_cols = sorted([c for c in df_pred.columns if c not in internal_or_duplicate])
                                             display_cols_order.extend(other_cols)


                                             # Filter to only include columns that actually exist in df_pred
                                             final_display_cols = [col for col in display_cols_order if col in df_pred.columns]

                                             # Show the results in a DataFrame
                                             st.dataframe(df_pred[final_display_cols], use_container_width=True, hide_index=True)
                                             df_pred_display = df_pred # Store for potential addition to DB

                                        else:
                                             st.warning("No quedaron filas válidas para categorizar después de limpiar datos estandarizados.")
                              except AttributeError as ae:
                                   st.error(f"Error de Atributo durante la predicción. ¿Están cargados el modelo y vectorizador correctamente? Error: {ae}")
                                   st.error(traceback.format_exc())
                              except Exception as e_pred:
                                   st.error(f"Error inesperado durante la categorización: {e_pred}")
                                   st.error(traceback.format_exc())

                          # --- Display Debug Info (if generated) ---
                          if st.session_state.debug_predictions:
                               with st.expander("Ver Detalles del Proceso de Predicción (Debug)", expanded=False):
                                   # Show a limited number of lines for performance
                                   max_debug_lines = 100
                                   st.text("\n".join(st.session_state.debug_predictions[:max_debug_lines]))
                                   if len(st.session_state.debug_predictions) > max_debug_lines:
                                       st.caption(f"... (mostrando {max_debug_lines} de {len(st.session_state.debug_predictions)} líneas)")

                          # --- Button to Add Results to DB ---
                          if df_pred_display is not None and not df_pred_display.empty:
                              # Use file_id for uniqueness if available, else name
                              file_identifier = uploaded_final_file.file_id if uploaded_final_file.file_id else uploaded_final_file.name
                              add_button_key = f"add_db_{file_identifier}" # Unique key per file upload instance
                              if st.button(f"➕ Añadir {len(df_pred_display)} Resultados a la Base de Datos", key=add_button_key, help="Añade las filas categorizadas (excluyendo duplicados) a la BD en memoria."):

                                  current_db = st.session_state.get('accumulated_data', pd.DataFrame())
                                  df_to_append_raw = df_pred_display.copy()

                                  # Define columns to keep when appending (use _PREDICHA as primary)
                                  # DB_FINAL_COLS already uses _PREDICHA versions + CONCEPTO_STD etc.
                                  db_cols_to_keep = list(DB_FINAL_COLS)
                                  # Add standard account column if exists in the data to append
                                  if CUENTA_COL_STD in df_to_append_raw.columns:
                                       if CUENTA_COL_STD not in db_cols_to_keep: db_cols_to_keep.append(CUENTA_COL_STD)
                                  # Add all ORIG_ columns from the data to append
                                  orig_cols_to_add = [c for c in df_to_append_raw.columns if c.startswith('ORIG_')]
                                  for col in orig_cols_to_add:
                                       if col not in db_cols_to_keep: db_cols_to_keep.append(col)

                                  # Ensure only existing columns in df_to_append_raw are selected
                                  final_cols_to_append = [col for col in db_cols_to_keep if col in df_to_append_raw.columns]
                                  df_to_append = df_to_append_raw[final_cols_to_append]

                                  # --- Duplicate Detection ---
                                  num_added = 0
                                  new_transactions_only = pd.DataFrame(columns=df_to_append.columns) # Initialize with correct columns

                                  if current_db.empty:
                                       st.write("BD vacía. Añadiendo todas las filas nuevas.")
                                       new_transactions_only = df_to_append
                                       num_added = len(new_transactions_only)
                                  else:
                                       st.write("Detectando duplicados antes de añadir...")
                                       # Define key columns for identifying duplicates (must exist in both DFs)
                                       key_cols_base = [AÑO_STD, MES_STD, DIA_STD, IMPORTE_STD, CONCEPTO_STD]
                                       key_cols = []

                                       # Check if base keys exist in BOTH dataframes
                                       keys_ok = True
                                       for col in key_cols_base:
                                           if col not in current_db.columns:
                                               st.error(f"Error Duplicados: Falta columna clave '{col}' en la BD actual.")
                                               keys_ok = False
                                           if col not in df_to_append.columns:
                                                st.error(f"Error Duplicados: Falta columna clave '{col}' en los datos nuevos (Error Interno).")
                                                keys_ok = False
                                           if keys_ok: key_cols.append(col)

                                       # Add account column to keys if available in *both* dataframes and base keys are OK
                                       account_key_col_name = None
                                       if keys_ok:
                                           acc_col_db = CUENTA_COL_STD if CUENTA_COL_STD in current_db.columns else None
                                           acc_col_new = CUENTA_COL_STD if CUENTA_COL_STD in df_to_append.columns else None

                                           if acc_col_db and acc_col_new:
                                               key_cols.append(CUENTA_COL_STD)
                                               account_key_col_name = CUENTA_COL_STD
                                               st.info(f"Usando '{CUENTA_COL_STD}' como parte de la clave de duplicados.")
                                           elif acc_col_db or acc_col_new:
                                                st.warning(f"Columna '{CUENTA_COL_STD}' encontrada solo en BD o en datos nuevos, no se usará para detectar duplicados.")


                                       if not keys_ok:
                                            num_added = -1 # Special flag for error due to missing keys
                                       else:
                                            try:
                                                # Prepare copies for comparison
                                                df1 = current_db.copy()
                                                df2 = df_to_append.copy()

                                                # Clean key columns in both dataframes before merge
                                                for df_temp in [df1, df2]:
                                                    # Round importe to 2 decimal places for reliable comparison
                                                    df_temp[IMPORTE_STD] = pd.to_numeric(df_temp[IMPORTE_STD], errors='coerce').round(2)
                                                    # Ensure date parts are integer
                                                    for col in [AÑO_STD, MES_STD, DIA_STD]:
                                                        # Use Int64 for intermediate step to handle potential NaNs from coerce
                                                        df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce').astype('Int64').fillna(0).astype(int)
                                                    # Ensure text keys are string and cleaned (lowercase, stripped)
                                                    df_temp[CONCEPTO_STD] = df_temp[CONCEPTO_STD].fillna('').astype(str).str.lower().str.strip()
                                                    if account_key_col_name:
                                                         df_temp[account_key_col_name] = df_temp[account_key_col_name].fillna('').astype(str).str.lower().str.strip()

                                                # --- Perform Merge for Duplicate Check ---
                                                # Drop rows with NaNs in any key column *after* cleaning/conversion
                                                # This prevents merge errors and ensures comparison is on valid keys
                                                df1_clean_keys = df1.dropna(subset=key_cols)
                                                df2_clean_keys = df2.dropna(subset=key_cols)


                                                if df1_clean_keys.empty: # If DB has no valid keys after cleaning
                                                    new_transactions_only = df2_clean_keys # Add all new rows with valid keys
                                                elif not df2_clean_keys.empty:
                                                    # Use merge with indicator=True to find rows in df2_clean_keys not in df1_clean_keys' unique key combinations
                                                    merged = df2_clean_keys.merge(
                                                        df1_clean_keys[key_cols].drop_duplicates(),
                                                        on=key_cols,
                                                        how='left',
                                                        indicator=True
                                                    )
                                                    # Select rows only present in the left dataframe (df2_clean_keys)
                                                    new_transactions_only = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
                                                # Else: df2_clean_keys is empty, new_transactions_only remains empty DF

                                                num_added = len(new_transactions_only)
                                                num_duplicates = len(df2_clean_keys) - num_added # Duplicates are those from df2 that had a match
                                                num_invalid_keys_new = len(df2) - len(df2_clean_keys) # Rows in new data dropped due to invalid keys

                                                if num_duplicates > 0:
                                                     st.info(f"Se detectaron y omitieron {num_duplicates} filas duplicadas (ya existentes en BD).")
                                                if num_invalid_keys_new > 0:
                                                     st.warning(f"Se omitieron {num_invalid_keys_new} filas de los datos nuevos debido a valores inválidos/faltantes en columnas clave ({', '.join(key_cols)}).")


                                            except KeyError as ke:
                                                 st.error(f"Error de Clave durante detección de duplicados: {ke}. Columnas BD: {df1.columns.tolist()}. Columnas Nuevas: {df2.columns.tolist()}")
                                                 num_added = -1 # Flag error
                                            except Exception as e_dup:
                                                st.error(f"Error inesperado durante la detección de duplicados: {e_dup}")
                                                st.error(traceback.format_exc())
                                                num_added = -1 # Flag error

                                  # --- Append Non-Duplicates ---
                                  if num_added > 0:
                                       st.write(f"Añadiendo {num_added} nuevas transacciones a la Base de Datos...")
                                       # Ensure columns match before concat (handle potentially new ORIG_ cols from either side)
                                       # Use outer join logic for columns
                                       combined_cols = current_db.columns.union(new_transactions_only.columns)
                                       # Reindex both DFs to have the union of columns, filling missing with appropriate nulls/defaults
                                       current_db_reindexed = current_db.reindex(columns=combined_cols, fill_value='') # Fill missing with empty string for simplicity
                                       new_transactions_reindexed = new_transactions_only.reindex(columns=combined_cols, fill_value='')

                                       # Concatenate
                                       st.session_state.accumulated_data = pd.concat(
                                           [current_db_reindexed, new_transactions_reindexed],
                                           ignore_index=True
                                       )
                                       # Optional: Clean up types again after concat if needed, e.g., ensure numeric columns are numeric
                                       # for col in [IMPORTE_STD, AÑO_STD, MES_STD, DIA_STD]:
                                       #     if col in st.session_state.accumulated_data.columns:
                                       #         st.session_state.accumulated_data[col] = pd.to_numeric(st.session_state.accumulated_data[col], errors='coerce')


                                       # --- Re-extract Knowledge AFTER adding data ---
                                       st.info("Actualizando conocimiento base con los nuevos datos...")
                                       df_for_knowledge_update = parse_accumulated_db_for_training(st.session_state.accumulated_data.copy())
                                       if df_for_knowledge_update is not None:
                                             st.session_state.learned_knowledge = extract_knowledge_std(df_for_knowledge_update)
                                             knowledge_ok_add = bool(st.session_state.learned_knowledge.get('categorias'))
                                             st.session_state.knowledge_loaded = knowledge_ok_add # Update loaded flag too
                                             if knowledge_ok_add: st.sidebar.info("Conocimiento actualizado (tras añadir).") # Update sidebar status
                                       else:
                                            st.sidebar.warning("BD actualizada, pero falló la re-extracción de conocimiento.")


                                       st.success(f"¡Éxito! {num_added} transacciones NUEVAS añadidas a la Base de Datos.")
                                       # Rerun to refresh the DB view and summary
                                       st.rerun()
                                  elif num_added == 0:
                                       st.info("No se añadieron filas nuevas (0 filas únicas encontradas o todas eran duplicados/inválidas).")
                                       # Optionally rerun to clear the displayed results if desired
                                       # st.rerun()
                                  else: # num_added == -1 (Error)
                                       st.error("No se añadieron filas debido a un error en la detección de duplicados (ver mensajes anteriores).")


                     elif df_std_new is not None and df_std_new.empty:
                          # Standardize returned an empty DF (e.g., all rows failed mapping/cleaning)
                          st.warning(f"El archivo '{uploaded_final_file.name}' no contenía filas válidas después de la estandarización.")
                     # Else: Failure during reading or standardization, error already shown

    st.divider()

    # ==============================================================
    # --- Sub-Sección: Ver/Gestionar Base de Datos Acumulada (Editable) ---
    # ==============================================================
    st.subheader("Base de Datos Acumulada (Editable)")
    db_state_tab = st.session_state.get('accumulated_data', pd.DataFrame())

    if db_state_tab is not None and not db_state_tab.empty:
        st.write(f"Mostrando {len(db_state_tab)} filas. Haz doble clic en una celda para editar (Categoría, Subcategoría, Comercio).")

        # --- Prepare for Editing ---
        knowledge = st.session_state.get('learned_knowledge', {})
        # Get options for dropdowns from current knowledge, add "" for empty option, Capitalize for display
        categorias_options = [""] + sorted([c.capitalize() for c in knowledge.get('categorias', [])])
        all_subcats_options = [""] + sorted([s.capitalize() for s in knowledge.get('all_subcategories', [])])
        all_comers_options = [""] + sorted([c.capitalize() for c in knowledge.get('all_comercios', [])])


        # Ensure the primary editable columns (_PREDICHA versions) exist in the DataFrame copy used for editor
        db_editable_copy = db_state_tab.copy() # Work on a copy for the editor
        cols_to_ensure = {
            CATEGORIA_PREDICHA: CATEGORIA_STD,
            SUBCATEGORIA_PREDICHA: SUBCATEGORIA_STD,
            COMERCIO_PREDICHO: COMERCIO_STD
        }
        for pred_col, std_col in cols_to_ensure.items():
             if pred_col not in db_editable_copy.columns:
                 if std_col in db_editable_copy.columns:
                      db_editable_copy[pred_col] = db_editable_copy[std_col]
                      # st.info(f"Copiando datos de '{std_col}' a '{pred_col}' para edición.") # Optional info message
                 else:
                      db_editable_copy[pred_col] = '' # Create empty if neither exists
             # Ensure column is string type and NaNs are filled for editor compatibility
             db_editable_copy[pred_col] = db_editable_copy[pred_col].fillna('').astype(str)
             # Capitalize existing values to match options? (Careful: might lose original casing if needed)
             # db_editable_copy[pred_col] = db_editable_copy[pred_col].str.capitalize()


        # --- Configure Columns for st.data_editor ---
        column_config_editor = {
            # Editable columns with dropdowns
            CATEGORIA_PREDICHA: st.column_config.SelectboxColumn(
                "Categoría Editada", width="medium",
                options=categorias_options, required=False, # Allow empty selection
                help="Edita la categoría asignada. Las opciones provienen del conocimiento actual."
            ),
            SUBCATEGORIA_PREDICHA: st.column_config.SelectboxColumn(
                "Subcategoría Editada", width="medium",
                options=all_subcats_options, required=False,
                help="Edita la subcategoría asignada. Las opciones provienen del conocimiento actual."
            ),
            COMERCIO_PREDICHO: st.column_config.SelectboxColumn(
                "Comercio Estandarizado", width="medium",
                options=all_comers_options, required=False,
                help="Edita/Asigna el comercio estandarizado. Las opciones provienen del conocimiento actual."
            ),
            # Non-Editable columns (display only) - Configure as needed
            CONCEPTO_STD: st.column_config.TextColumn("Concepto Original", disabled=True, help="Concepto leído del banco."),
            IMPORTE_STD: st.column_config.NumberColumn("Importe", format="%.2f", disabled=True, help="Importe leído."),
            AÑO_STD: st.column_config.NumberColumn("Año", format="%d", disabled=True),
            MES_STD: st.column_config.NumberColumn("Mes", format="%02d", disabled=True), # Pad month
            DIA_STD: st.column_config.NumberColumn("Día", format="%02d", disabled=True), # Pad day
            CUENTA_COL_STD: st.column_config.TextColumn("Cuenta Origen", disabled=True, help="Cuenta original del archivo (si existe)."),
            # Add configuration for ORIG_ columns dynamically
        }
        # Add config for any ORIG_ columns dynamically, making them disabled
        for col in db_editable_copy.columns:
             if col.startswith("ORIG_") and col not in column_config_editor:
                 # Truncate long original column names for display label if needed
                 label = col.replace("ORIG_", "")[:25] + ('...' if len(col.replace("ORIG_", "")) > 25 else '')
                 column_config_editor[col] = st.column_config.TextColumn(f"Orig: {label}", disabled=True, help=f"Columna original: {col}")


        # --- Define Column Order for Editor ---
        # Prioritize editable and key info columns
        cols_order_edit = [
            CATEGORIA_PREDICHA, SUBCATEGORIA_PREDICHA, COMERCIO_PREDICHO,
            CONCEPTO_STD, IMPORTE_STD, AÑO_STD, MES_STD, DIA_STD
        ]
        # Add account column if it exists
        if CUENTA_COL_STD in db_editable_copy.columns:
             if CUENTA_COL_STD not in cols_order_edit: cols_order_edit.append(CUENTA_COL_STD)
        # Add ORIG columns, sorted alphabetically
        orig_cols_edit = sorted([col for col in db_editable_copy.columns if col.startswith('ORIG_') and col != CUENTA_COL_STD])
        cols_order_edit.extend(orig_cols_edit)
        # Add any remaining standard columns not yet included (e.g. _STD versions if they exist and are different from _PRED)
        other_std_cols = [c for c in [CATEGORIA_STD, SUBCATEGORIA_STD, COMERCIO_STD] if c in db_editable_copy.columns and c not in cols_order_edit]
        cols_order_edit.extend(other_std_cols)
        # Add any other columns (unlikely but possible)
        internal_or_existing = [TEXTO_MODELO] + cols_order_edit
        other_cols_edit = sorted([col for col in db_editable_copy.columns if col not in internal_or_existing])
        cols_order_edit.extend(other_cols_edit)


        # Ensure all columns in the order list actually exist in the dataframe
        final_cols_edit_display = [col for col in cols_order_edit if col in db_editable_copy.columns]

        # --- Display Data Editor ---
        # Use a unique key to allow state preservation for the editor itself
        edited_df = st.data_editor(
            db_editable_copy[final_cols_edit_display], # Pass the copy with the right columns and order
            key="db_editor_main",
            column_config=column_config_editor,
            num_rows="dynamic", # Allow adding/deleting rows (Use with caution!)
            use_container_width=True,
            hide_index=True,
            height=600 # Set a fixed height for better layout
        )

        # --- Save Changes Button ---
        if st.button("💾 Confirmar Cambios en BD", key="save_edited_db", help="Guarda las modificaciones hechas en la tabla de arriba a la BD en memoria."):
            # --- Post-Edit Validation / Cleaning (Optional but Recommended) ---
            edited_df_clean = edited_df.copy()
            # Example: Convert edited dropdown values back to lowercase for internal consistency?
            # for col in [CATEGORIA_PREDICHA, SUBCATEGORIA_PREDICHA, COMERCIO_PREDICHO]:
            #      if col in edited_df_clean.columns:
            #          edited_df_clean[col] = edited_df_clean[col].str.lower()
            # Example: Trim whitespace from edited text fields? (Not directly editable here, but if they were)

            # Check if essential columns were accidentally cleared or removed
            essential_cols_check = [CATEGORIA_PREDICHA, CONCEPTO_STD, IMPORTE_STD, AÑO_STD, MES_STD, DIA_STD]
            valid_edit = True
            missing_essentials = []
            for col in essential_cols_check:
                 if col not in edited_df_clean.columns:
                      missing_essentials.append(col)
                      valid_edit = False
                 # Check for excessive nulls (optional, might be intended)
                 # elif edited_df_clean[col].isnull().sum() > (0.5 * len(edited_df_clean)): # Example threshold
                 #     st.warning(f"Aviso: Más del 50% de valores en '{col}' están vacíos después de la edición.")

            if not valid_edit:
                 st.error(f"Error Crítico: Faltan columnas esenciales ({', '.join(missing_essentials)}) en los datos editados. Cambios NO guardados.")
            else:
                # --- Update Session State ---
                # Replace the entire DataFrame in session state with the edited version.
                # This handles added/deleted rows correctly.
                st.session_state.accumulated_data = edited_df_clean.copy()

                # --- Re-extract Knowledge AFTER saving edits ---
                st.info("Actualizando conocimiento base con los datos editados...")
                # Prepare the newly edited data for knowledge extraction
                df_for_knowledge_update_edit = parse_accumulated_db_for_training(st.session_state.accumulated_data.copy())
                if df_for_knowledge_update_edit is not None:
                      st.session_state.learned_knowledge = extract_knowledge_std(df_for_knowledge_update_edit)
                      knowledge_ok_edit = bool(st.session_state.learned_knowledge.get('categorias'))
                      st.session_state.knowledge_loaded = knowledge_ok_edit # Update flag
                      if knowledge_ok_edit: st.sidebar.info("Conocimiento actualizado (tras edición).")
                else:
                      st.sidebar.warning("BD editada guardada, pero falló la re-extracción de conocimiento.")


                st.success("¡Cambios guardados en la Base de Datos en memoria!")
                # Rerun to refresh the editor (reflecting potential type changes or re-extracted knowledge in dropdowns)
                # and the summary view.
                st.rerun()


    else: # DB is empty
        st.info("La Base de Datos Acumulada está vacía. Cárgala desde la barra lateral o añade datos categorizados.")


# ==============================================================
# --- Sidebar Info and State ---
# ==============================================================
st.sidebar.divider()
st.sidebar.header("Acerca de")
st.sidebar.info("Categorizador Bancario Inteligente v5.4\nDesarrollado con Streamlit & Scikit-learn.")
st.sidebar.divider()
st.sidebar.subheader("Estado Actual")

# Display Model Status
model_ready_sidebar = st.session_state.get('model_trained', False)
knowledge_ready_sidebar = st.session_state.get('knowledge_loaded', False)
if model_ready_sidebar:
    st.sidebar.success("✅ Modelo Entrenado")
elif knowledge_ready_sidebar: # Knowledge loaded but model not trained yet
    st.sidebar.info("ℹ️ Conocimiento Cargado (Modelo no entrenado)")
else: # Neither knowledge nor model ready
    st.sidebar.warning("❌ Sin Modelo ni Conocimiento")

# Display Mappings Status
num_mappings = len(st.session_state.get('bank_mappings', {}))
if num_mappings > 0:
    st.sidebar.success(f"✅ Mapeos Definidos ({num_mappings})")
else:
    st.sidebar.warning("❌ Sin Mapeos Bancarios")

# Display DB Status
db_state_sidebar = st.session_state.get('accumulated_data', pd.DataFrame())
db_rows = len(db_state_sidebar) if db_state_sidebar is not None else 0
if db_rows > 0:
    st.sidebar.success(f"✅ BD Cargada ({db_rows} filas)")
else:
    st.sidebar.info("ℹ️ BD Vacía")


# ==============================================================
# --- Descarga BD (Sidebar) ---
# ==============================================================
st.sidebar.divider()
st.sidebar.subheader("Guardar Base de Datos")
db_state_sidebar_save = st.session_state.get('accumulated_data', pd.DataFrame()) # Get current state again

if db_state_sidebar_save is not None and not db_state_sidebar_save.empty:
    try:
        # Define columns to export - Prioritize _PREDICHA, include standard essentials, account, ORIG_
        # Use DB_FINAL_COLS as the base, which already includes PREDICHA versions + essentials
        cols_to_export_db = list(DB_FINAL_COLS)

        # Ensure account column is included using standard name (ORIG_CUENTA)
        cuenta_col_export = CUENTA_COL_STD if CUENTA_COL_STD in db_state_sidebar_save.columns else None
        if cuenta_col_export and cuenta_col_export not in cols_to_export_db:
             cols_to_export_db.append(cuenta_col_export)

        # Add all ORIG_ columns (except account if already added)
        orig_cols_export = sorted([
            col for col in db_state_sidebar_save.columns
            if col.startswith('ORIG_') and col != cuenta_col_export
        ])
        for col in orig_cols_export:
            if col not in cols_to_export_db: cols_to_export_db.append(col)

        # Add STD versions only if they exist and are NOT the same as PREDICHA versions (less common)
        # This avoids duplicate columns if STD and PREDICHA were the same initially
        std_cols_maybe = {CATEGORIA_STD: CATEGORIA_PREDICHA, SUBCATEGORIA_STD: SUBCATEGORIA_PREDICHA, COMERCIO_STD: COMERCIO_PREDICHO}
        for std_c, pred_c in std_cols_maybe.items():
             if std_c in db_state_sidebar_save.columns and std_c not in cols_to_export_db:
                 # Only add if it's meaningfully different or pred_c wasn't included?
                 # Safest is probably just to ensure the PREDICHA ones are primary and skip adding STD here.
                 pass # Let's keep the export focused on the final state (PREDICHA)


        # Filter list to ensure all columns actually exist in the current DB state
        final_cols_to_export = [col for col in cols_to_export_db if col in db_state_sidebar_save.columns]

        # Create DataFrame for export with selected columns and order
        df_to_export = db_state_sidebar_save[final_cols_to_export].copy()

        # --- CSV Download ---
        try:
             # Use utf-8-sig for better Excel compatibility with accents/special chars
             db_csv_output_sb = df_to_export.to_csv(index=False, sep=';', decimal=',', encoding='utf-8-sig')
             st.sidebar.download_button(
                 label=f"💾 Descargar BD (CSV)",
                 data=db_csv_output_sb,
                 file_name=DB_FILENAME,
                 mime='text/csv',
                 key='download_db_csv_sb'
             )
        except Exception as e_csv_down:
             st.sidebar.error(f"Error al generar CSV: {e_csv_down}")


        # --- Excel Download ---
        try:
            output_excel_sb = io.BytesIO()
            with pd.ExcelWriter(output_excel_sb, engine='openpyxl') as writer:
                df_to_export.to_excel(writer, index=False, sheet_name='Gastos')
            excel_data_sb = output_excel_sb.getvalue()
            # Generate Excel filename based on DB_FILENAME
            db_excel_filename_sb = DB_FILENAME.replace('.csv', '.xlsx') if DB_FILENAME.lower().endswith('.csv') else f"{DB_FILENAME}.xlsx"
            st.sidebar.download_button(
                label=f"💾 Descargar BD (Excel)",
                data=excel_data_sb,
                file_name=db_excel_filename_sb,
                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                key='download_db_excel_sb'
            )
        except ImportError:
             st.sidebar.info("Para descargar en Excel (.xlsx), instala 'openpyxl': pip install openpyxl")
        except Exception as e_xls_down:
             st.sidebar.error(f"Error al generar Excel: {e_xls_down}")


    except Exception as e_db_down_prep:
        st.sidebar.error(f"Error preparando datos de BD para descarga: {e_db_down_prep}")
        st.sidebar.error(traceback.format_exc())

else:
    st.sidebar.info("La Base de Datos está vacía, no hay nada que guardar.")
