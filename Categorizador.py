# -*- coding: utf-8 -*-
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
            return None, []

    except Exception as e:
        st.error(f"Error general al procesar el archivo '{uploaded_file.name if uploaded_file else 'N/A'}': {e}")
        st.error(traceback.format_exc()) # Show full traceback for debugging
        return None, []

# NOTE: parse_historic_categorized is DEPRECATED. Use parse_accumulated_db_for_training
def parse_historic_categorized(df_raw):
    st.warning("Función 'parse_historic_categorized' es obsoleta, usar 'parse_accumulated_db_for_training'.")
    return None


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

    # Determine which subcategory/commerce columns to use (prefer _STD)
    subcat_col_to_use = SUBCATEGORIA_STD if SUBCATEGORIA_STD in df_std.columns else SUBCATEGORIA_PREDICHA if SUBCATEGORIA_PREDICHA in df_std.columns else None
    comercio_col_to_use = COMERCIO_STD if COMERCIO_STD in df_std.columns else COMERCIO_PREDICHO if COMERCIO_PREDICHO in df_std.columns else None

    if not subcat_col_to_use: st.info("Extracción conocimiento: No se encontró columna de subcategoría (_STD o _PREDICHA).")
    if not comercio_col_to_use: st.info("Extracción conocimiento: No se encontró columna de comercio (_STD o _PREDICHO).")


    try:
        df_clean = df_std.copy()

        # --- Data Cleaning and Preparation ---
        # Standardize text columns: fill NaNs, ensure string type, lowercase, strip whitespace
        df_clean[cat_col_to_use] = df_clean[cat_col_to_use].fillna('').astype(str).str.lower().str.strip()
        if subcat_col_to_use:
            df_clean[subcat_col_to_use] = df_clean[subcat_col_to_use].fillna('').astype(str).str.lower().str.strip()
        if comercio_col_to_use:
            df_clean[comercio_col_to_use] = df_clean[comercio_col_to_use].fillna('').astype(str).str.lower().str.strip()

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
                 df_cat_comers = df_cat[df_cat[comercio_col_to_use].fillna('') != '']
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
                             # Clean keys before using them in dicts
                             comercio_key = str(comercio).lower().strip()
                             cat_key = str(cat).lower().strip()


                             # 1. Check for unique subcategory rule
                             if len(unique_subcats_for_comercio) == 1:
                                 knowledge['subcat_unica_por_comercio_y_cat'].setdefault(cat_key, {})[comercio_key] = unique_subcats_for_comercio[0]


                             # 2. Find the most frequent subcategory rule (even if only one exists)
                             if len(subcats_for_this_comercio) > 0:
                                 try:
                                     # Calculate value counts and find the index (subcategory name) of the maximum count
                                     most_frequent_subcat = subcats_for_this_comercio.value_counts().idxmax()
                                     knowledge['subcat_mas_frecuente_por_comercio_y_cat'].setdefault(cat_key, {})[comercio_key] = most_frequent_subcat
                                 except Exception as e_freq:
                                     # Fallback if idxmax fails (e.g., all counts are 1)
                                     # st.warning(f"Cálculo subcat frecuente falló para '{comercio}' en '{cat}': {e_freq}. Usando la primera encontrada.")
                                     if len(unique_subcats_for_comercio) > 0:
                                         # Use setdefault to ensure the keys exist before assignment
                                          knowledge['subcat_mas_frecuente_por_comercio_y_cat'].setdefault(cat_key, {})[comercio_key] = unique_subcats_for_comercio[0]



        # Finalize global lists
        knowledge['all_subcategories'] = sorted(list(all_subcats_set))
        knowledge['all_comercios'] = sorted(list(all_comers_set))

        st.success(f"Conocimiento extraído: {len(knowledge['categorias'])} Cat, {len(knowledge['all_subcategories'])} SubCat, {len(knowledge['all_comercios'])} Comercios.")


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
             # Don't automatically error here if source_col is missing; validation happens later

        # Create initial DataFrame from mapped columns that were found
        df_std = pd.DataFrame(temp_std_data)

        # --- 2. Validate essential columns are present after initial mapping ---
        if CONCEPTO_STD not in df_std.columns:
            st.error(f"Error Mapeo '{mapping.get('bank_name', '?')}': La columna mapeada para '{CONCEPTO_STD}' ('{mapping_cols.get(CONCEPTO_STD)}') no existe en el archivo.")
            return None
        if IMPORTE_STD not in df_std.columns:
            st.error(f"Error Mapeo '{mapping.get('bank_name', '?')}': La columna mapeada para '{IMPORTE_STD}' ('{mapping_cols.get(IMPORTE_STD)}') no existe en el archivo.")
            return None

        # --- 3. Process Date ---
        date_ok = False
        source_col_fecha = mapping_cols.get(FECHA_STD)
        source_cols_amd = mapping_cols.get(AÑO_STD), mapping_cols.get(MES_STD), mapping_cols.get(DIA_STD)

        # Check if FECHA_STD was mapped *and* exists in the temp df
        if FECHA_STD in df_std.columns:
            date_fmt = mapping.get('date_format')
            if not date_fmt:
                st.error(f"Error Mapeo '{mapping.get('bank_name', '?')}': Se mapeó '{source_col_fecha}' a '{FECHA_STD}' pero falta 'date_format' en la configuración.")
                return None
            try:
                date_series = df_std[FECHA_STD].astype(str).str.strip().replace('', pd.NaT)
                dates = pd.to_datetime(date_series, format=date_fmt, errors='coerce')

                if dates.isnull().all():
                    st.error(f"Error Mapeo '{mapping.get('bank_name', '?')}': Ninguna fecha en '{source_col_fecha}' coincide con el formato '{date_fmt}'.")
                    problem_samples = df_std.loc[dates.isnull() & df_std[FECHA_STD].notna(), FECHA_STD].unique()
                    st.info(f"Algunos valores no parseados: {list(problem_samples[:5])}")
                    return None
                if dates.isnull().any():
                    st.warning(f"Aviso Mapeo '{mapping.get('bank_name', '?')}': Algunas fechas en '{source_col_fecha}' no coinciden con formato '{date_fmt}' o estaban vacías.")

                df_std[AÑO_STD] = dates.dt.year.fillna(0).astype(int)
                df_std[MES_STD] = dates.dt.month.fillna(0).astype(int)
                df_std[DIA_STD] = dates.dt.day.fillna(0).astype(int)
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
            try:
                for i, c in enumerate([AÑO_STD, MES_STD, DIA_STD]):
                    df_std[c] = pd.to_numeric(df_std[c], errors='coerce').astype('Int64').fillna(0).astype(int)
                # Basic validation for date parts
                if ((df_std[MES_STD] < 1) | (df_std[MES_STD] > 12) | (df_std[DIA_STD] < 1) | (df_std[DIA_STD] > 31)).any():
                     st.warning(f"Aviso Mapeo '{mapping.get('bank_name', '?')}': Columnas A/M/D ({source_cols_amd}) contienen valores inválidos (ej. Mes>12, Dia>31).")
                date_ok = True
            except Exception as e_num:
                st.error(f"Error convirtiendo columnas AÑO/MES/DIA ({source_cols_amd}) a números: {e_num}")
                return None
        else: # Neither date path is valid
            err_msg = f"Error Mapeo '{mapping.get('bank_name', '?')}': Mapeo de fecha incompleto o columnas no encontradas. "
            err_msg += f"Se necesita '{FECHA_STD}' (mapeado a '{source_col_fecha}') con formato O "
            err_msg += f"las tres columnas '{AÑO_STD}/{MES_STD}/{DIA_STD}' (mapeadas a '{source_cols_amd[0]}/{source_cols_amd[1]}/{source_cols_amd[2]}'). "
            err_msg += f"Columnas encontradas en archivo: {original_columns}"
            st.error(err_msg)
            return None

        if not date_ok: return None # Safety check

        # --- 4. Process Importe ---
        source_col_importe = mapping_cols.get(IMPORTE_STD) # Already checked it exists in df_std
        try:
            imp_str = df_std[IMPORTE_STD].fillna('0').astype(str).str.strip()
            ts = mapping.get('thousands_sep')
            ds = mapping.get('decimal_sep', ',')
            if ts: imp_str = imp_str.str.replace(ts, '', regex=False)
            imp_str = imp_str.str.replace(ds, '.', regex=False)
            df_std[IMPORTE_STD] = pd.to_numeric(imp_str, errors='coerce')

            if df_std[IMPORTE_STD].isnull().any():
                st.warning(f"Aviso Mapeo '{mapping.get('bank_name', '?')}': Algunos importes en '{source_col_importe}' no pudieron ser convertidos a número después de aplicar separadores (Miles: '{ts}', Decimal: '{ds}').")
                original_importe_col = df[source_col_importe] if source_col_importe in df.columns else pd.Series(index=df_std.index)
                problem_indices = df_std[df_std[IMPORTE_STD].isnull()].index
                problem_samples_orig = original_importe_col.loc[problem_indices].unique()
                st.info(f"Valores originales problemáticos: {list(problem_samples_orig[:5])}")
        except Exception as e_imp:
            st.error(f"Error procesando columna de importe '{source_col_importe}': {e_imp}")
            st.error(traceback.format_exc())
            return None

        # --- 5. Process Text Columns (Concepto, Comercio) ---
        for c in [CONCEPTO_STD, COMERCIO_STD]:
            if c in df_std.columns:
                df_std[c] = df_std[c].fillna('').astype(str).str.lower().str.strip()
            elif c == COMERCIO_STD: # If COMERCIO_STD was not mapped or source col didn't exist
                df_std[COMERCIO_STD] = ''

        # Ensure Concepto/Comercio exist for TEXTO_MODELO creation
        if CONCEPTO_STD not in df_std: df_std[CONCEPTO_STD] = ''
        if COMERCIO_STD not in df_std: df_std[COMERCIO_STD] = ''

        # --- 6. Create TEXTO_MODELO ---
        df_std[TEXTO_MODELO] = (df_std[CONCEPTO_STD] + ' ' + df_std[COMERCIO_STD]).str.strip()

        # --- 7. Keep Original Unused Columns ---
        original_cols_to_keep = [c for c in original_columns if c not in source_cols_used]
        cuenta_col_orig_upper = CUENTA_COL_ORIG.upper()
        if cuenta_col_orig_upper in original_columns and cuenta_col_orig_upper not in source_cols_used:
             if cuenta_col_orig_upper not in original_cols_to_keep:
                original_cols_to_keep.append(cuenta_col_orig_upper)

        for col in original_cols_to_keep:
            target_col_name = f"ORIG_{col}"
            sfx = 1
            while target_col_name in df_std.columns:
                target_col_name = f"ORIG_{col}_{sfx}"; sfx += 1
            if col == cuenta_col_orig_upper: target_col_name = CUENTA_COL_STD # Standard name for account
            if target_col_name not in df_std.columns:
                 df_std[target_col_name] = df[col].fillna('').astype(str) # Clean original kept cols too

        # --- 8. Final Cleanup ---
        essential_cols_final = [IMPORTE_STD, TEXTO_MODELO, AÑO_STD, MES_STD, DIA_STD]
        initial_rows = len(df_std)
        df_std = df_std.dropna(subset=[IMPORTE_STD, AÑO_STD, MES_STD, DIA_STD]) # Drop rows missing numeric essentials
        df_std = df_std[df_std[TEXTO_MODELO].fillna('').str.strip() != ''] # Drop rows with empty text for model
        rows_after_na = len(df_std)
        if initial_rows > rows_after_na:
             st.info(f"Estandarización '{mapping.get('bank_name', '?')}': Se eliminaron {initial_rows - rows_after_na} filas por datos faltantes/inválidos en columnas esenciales (Importe, Fecha, Concepto).")

        if df_std.empty:
             st.warning(f"Aviso Mapeo '{mapping.get('bank_name', '?')}': No quedaron filas válidas después de la estandarización completa.")
             # Return empty DF with expected structure
             expected_cols = [CONCEPTO_STD, IMPORTE_STD, AÑO_STD, MES_STD, DIA_STD, COMERCIO_STD, TEXTO_MODELO] + [c for c in df_std.columns if c.startswith("ORIG_")]
             return pd.DataFrame(columns=list(dict.fromkeys(expected_cols)))

        return df_std

    except Exception as e:
        st.error(f"Error general aplicando mapeo '{mapping.get('bank_name', '?')}': {e}")
        st.error(traceback.format_exc())
        return None


# --- CORRECTED FUNCTION ---
def parse_accumulated_db_for_training(df_db):
    """Prepares the accumulated DB DataFrame for training or knowledge extraction."""
    if not isinstance(df_db, pd.DataFrame) or df_db.empty:
        st.error("Error Preparación BD: La base de datos acumulada está vacía o no es válida.")
        return None

    df = df_db.copy()
    # Standardize column names from DB file to UPPERCASE
    original_cols_case_preserved = df.columns.tolist() # Keep original for error messages if needed
    df.columns = [str(col).upper().strip() for col in df.columns]
    st.write(f"Columnas detectadas (mayúsculas): {df.columns.tolist()}") # DEBUG

    # --- Identify the Category Column ---
    # Prioritize *_PREDICHA, then _STD, then RAW variations.
    cat_col_options = [CATEGORIA_PREDICHA, CATEGORIA_STD, 'CATEGORÍA', 'CATEGORIA', 'CATEGORIA_X'] # Added 'CATEGORIA'
    cat_col_found = next((c for c in cat_col_options if c in df.columns), None)

    if not cat_col_found:
        st.error(f"Error Preparación BD: No se encontró una columna de categoría reconocida en la BD ({', '.join(cat_col_options)}).")
        st.error(f"Columnas disponibles: {df.columns.tolist()}")
        return None

    # Rename the found category column to CATEGORIA_STD for training/knowledge
    if cat_col_found != CATEGORIA_STD:
        if CATEGORIA_STD in df.columns:
             st.warning(f"Preparación BD: Se encontraron '{cat_col_found}' y '{CATEGORIA_STD}'. Se usará '{CATEGORIA_STD}' para entrenar/conocimiento.")
        else:
             df = df.rename(columns={cat_col_found: CATEGORIA_STD})
             st.info(f"Preparación BD: Usando columna '{cat_col_found}' como '{CATEGORIA_STD}' para entrenamiento/conocimiento.")
    # Now CATEGORIA_STD should exist and contain the category data

    # --- Ensure standard columns exist by mapping from PREDICHA or RAW ---
    # Order of preference: STD exists -> PREDICHA exists -> RAW exists -> Create Empty (for optional only)
    map_columns = {
        # Standard Col      # Predicted Col           # Raw Col (as uppercased from user's file)
        SUBCATEGORIA_STD: (SUBCATEGORIA_PREDICHA, 'SUBCATEGORIA'),
        COMERCIO_STD:     (COMERCIO_PREDICHO,     'COMERCIO'),
        CONCEPTO_STD:     (None,                  'CONCEPTO'),
        IMPORTE_STD:      (None,                  'IMPORTE'),
        AÑO_STD:          (None,                  'AÑO'),
        MES_STD:          (None,                  'MES'),
        DIA_STD:          (None,                  'DIA'),
        # CUENTA_COL_STD is 'ORIG_CUENTA'. Map 'CUENTA' (raw) to it.
        CUENTA_COL_STD:   (None,                  'CUENTA')
    }

    # st.write(f"Columnas ANTES del mapeo opcional: {df.columns.tolist()}") # DEBUG

    for std_col, (pred_col, raw_col) in map_columns.items():
         if std_col not in df.columns: # If standard name doesn't exist yet...
             renamed_or_created = False
             # 1. Try renaming from Predicted column if it exists
             if pred_col and pred_col in df.columns:
                 # Avoid renaming if it would overwrite an existing column accidentally
                 if std_col not in df.columns:
                     df = df.rename(columns={pred_col: std_col})
                     st.info(f"Preparación BD: Mapeado '{pred_col}' a '{std_col}'.")
                     renamed_or_created = True
                 else:
                     st.warning(f"Preparación BD: '{pred_col}' encontrado, pero '{std_col}' ya existe. No se renombró.")

             # 2. Else, try renaming from Raw (uppercased) column if it exists
             elif raw_col and raw_col in df.columns:
                  # Avoid renaming if it would overwrite an existing column accidentally
                 if std_col not in df.columns:
                     df = df.rename(columns={raw_col: std_col})
                     st.info(f"Preparación BD: Mapeado columna original '{raw_col}' a '{std_col}'.")
                     renamed_or_created = True
                 else:
                     st.warning(f"Preparación BD: Columna original '{raw_col}' encontrada, pero '{std_col}' ya existe. No se renombró.")

             # 3. Else, create empty column *only* if it's truly optional and not essential for core logic
             elif std_col in [SUBCATEGORIA_STD, COMERCIO_STD, CUENTA_COL_STD]: # Only create these if not found
                 if std_col not in df.columns: # Check again it wasn't created above
                    df[std_col] = ''
                    st.info(f"Preparación BD: Columna '{raw_col}' (o predicha) no encontrada para '{std_col}', creada vacía.")
                    renamed_or_created = True
             # Else: Essential columns like IMPORTE, CONCEPTO, DATE parts will trigger error later if not found via RAW mapping

    # st.write(f"Columnas DESPUÉS del mapeo opcional: {df.columns.tolist()}") # DEBUG

    # --- Now, *after* attempting renaming, check for required columns ---
    required_std_cols_train = [CONCEPTO_STD, IMPORTE_STD, AÑO_STD, MES_STD, DIA_STD, CATEGORIA_STD]
    missing_req = [col for col in required_std_cols_train if col not in df.columns]
    if missing_req:
        st.error(f"Error Preparación BD CRÍTICO: Faltan columnas esenciales después de intentar mapear desde originales: {', '.join(missing_req)}")
        st.error(f"Asegúrate de que tu archivo Excel/CSV contenga columnas nombradas (o similares a): {['CONCEPTO', 'IMPORTE', 'AÑO', 'MES', 'DIA', 'CATEGORÍA'] + ['SUBCATEGORIA', 'COMERCIO', 'CUENTA'] }")
        st.error(f"Columnas disponibles encontradas finalmente: {df.columns.tolist()}") # DEBUG
        return None

    # --- Data Cleaning and Type Conversion (using standard names) ---
    df_train = df.copy()
    try:
        # Clean text columns: fillna, string type, lower, strip
        # Include all potentially mapped columns
        text_cols_to_clean = [CONCEPTO_STD, COMERCIO_STD, CATEGORIA_STD, SUBCATEGORIA_STD, CUENTA_COL_STD]
        for col in text_cols_to_clean:
            if col in df_train.columns: # Ensure column exists before cleaning
                 # Convert to string first to handle potential mixed types before fillna/lower/strip
                 df_train[col] = df_train[col].astype(str).fillna('').str.lower().str.strip()

        # Clean numeric columns: coerce to numeric, fillna with appropriate value
        # Handle IMPORTE potentially having commas as decimal sep even in training data
        if IMPORTE_STD in df_train.columns:
             imp_str = df_train[IMPORTE_STD].astype(str).str.replace(',', '.', regex=False)
             df_train[IMPORTE_STD] = pd.to_numeric(imp_str, errors='coerce')
             if df_train[IMPORTE_STD].isnull().any():
                  st.warning(f"Preparación BD: Algunos valores en '{IMPORTE_STD}' no pudieron ser convertidos a número.")

        for col_f in [AÑO_STD, MES_STD, DIA_STD]:
            if col_f in df_train.columns:
                 # Ensure conversion to Int64 to handle potential NaNs after coerce before fillna
                 df_train[col_f] = pd.to_numeric(df_train[col_f], errors='coerce').astype('Int64').fillna(0).astype(int) # 0 for failed date parts

    except Exception as e_clean:
        st.error(f"Error limpiando datos de la BD acumulada: {e_clean}")
        st.error(traceback.format_exc())
        return None

    # --- Create TEXTO_MODELO (using COMERCIO_STD which should now be correct) ---
    if CONCEPTO_STD not in df_train: df_train[CONCEPTO_STD] = '' # Should exist after check, but safety
    if COMERCIO_STD not in df_train: df_train[COMERCIO_STD] = '' # Might be empty if not in original file
    df_train[TEXTO_MODELO] = (df_train[CONCEPTO_STD] + ' ' + df_train[COMERCIO_STD]).str.strip()


    # --- Final Filtering ---
    essential_final = [IMPORTE_STD, CATEGORIA_STD, TEXTO_MODELO, AÑO_STD, MES_STD, DIA_STD]
    initial_rows = len(df_train)
    df_train = df_train.dropna(subset=essential_final) # Drop rows missing core data needed for model/logic

    # Drop rows with empty category or empty text model input after cleaning
    df_train = df_train[df_train[CATEGORIA_STD] != '']
    df_train = df_train[df_train[TEXTO_MODELO] != '']
    rows_after_filter = len(df_train)

    if initial_rows > rows_after_filter:
        st.info(f"Preparación BD: Se eliminaron {initial_rows - rows_after_filter} filas de la BD por datos faltantes/inválidos para entrenamiento/conocimiento.")

    if df_train.empty:
        st.warning("Aviso Preparación BD: No quedaron filas válidas en la base de datos para entrenar/extraer conocimiento después de la limpieza.")
        return None # Return None instead of empty DF for clarity

    # --- Select Columns for Output ---
    # Include columns needed for training, knowledge extraction, and potentially duplicates/reference
    cols_to_keep_for_train = [
        TEXTO_MODELO, CATEGORIA_STD, SUBCATEGORIA_STD, COMERCIO_STD,
        IMPORTE_STD, AÑO_STD, MES_STD, DIA_STD, CONCEPTO_STD, CUENTA_COL_STD # Now includes CUENTA_COL_STD mapped from CUENTA
    ]
    # Keep only columns that actually exist in the final df_train
    final_cols = [col for col in cols_to_keep_for_train if col in df_train.columns]

    st.success(f"Preparación BD OK. Columnas finales para conocimiento/entrenamiento: {final_cols}")
    # st.dataframe(df_train[final_cols].head()) # DEBUG: Show head of prepared data
    return df_train[final_cols]
# --- END CORRECTED FUNCTION ---


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
    # Use CUENTA_COL_STD which should be 'ORIG_CUENTA'
    cuenta_col = CUENTA_COL_STD if CUENTA_COL_STD in df.columns else None
    if not cuenta_col: # Fallback to raw 'CUENTA' if standard name wasn't mapped/found
        cuenta_col = 'CUENTA' if 'CUENTA' in df.columns else None


    # --- Check required columns ---
    date_cols = [AÑO_STD, MES_STD, DIA_STD]
    if not cuenta_col:
        # Don't display error here, return specific message DF
        return pd.DataFrame({'Mensaje': [f"No se encontró columna de cuenta ({CUENTA_COL_STD} o CUENTA)"]}, index=[0])
    if not all(c in df.columns for c in date_cols):
        missing_dates = [c for c in date_cols if c not in df.columns]
        return pd.DataFrame({'Mensaje': [f"Faltan columnas de fecha ({', '.join(missing_dates)})"]}, index=[0])

    try:
        # --- Prepare Date Column ---
        # Ensure date parts are numeric, fill errors with 0 (will lead to invalid date)
        for col in date_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64').fillna(0)

        # Build date string, padding month and day ('YYYY-MM-DD')
        year_str = df[AÑO_STD].astype(int).astype(str).str.zfill(4)
        month_str = df[MES_STD].astype(int).astype(str).str.zfill(2)
        day_str = df[DIA_STD].astype(int).astype(str).str.zfill(2)
        date_str_series = year_str + '-' + month_str + '-' + day_str

        # Convert to datetime, coercing errors (like '0000-00-00' or invalid dates) to NaT
        df['FECHA_COMPLETA_TEMP'] = pd.to_datetime(date_str_series, format='%Y-%m-%d', errors='coerce')

        # Drop rows where the date could not be parsed (NaT) or account is missing/NaN
        df_valid_dates = df.dropna(subset=['FECHA_COMPLETA_TEMP', cuenta_col])

        # Also drop rows where account name is empty string after potential fillna
        df_valid_dates = df_valid_dates[df_valid_dates[cuenta_col].astype(str).str.strip() != '']


        if df_valid_dates.empty:
             return pd.DataFrame({'Mensaje': ["No hay fechas o cuentas válidas en la BD para calcular resumen"]}, index=[0])

        # --- Find Last Transaction per Account ---
        # Ensure account column is string for grouping
        df_valid_dates[cuenta_col] = df_valid_dates[cuenta_col].astype(str)
        # Use idxmax() on the valid datetime column to get the index of the max date per group
        last_indices = df_valid_dates.loc[df_valid_dates.groupby(cuenta_col)['FECHA_COMPLETA_TEMP'].idxmax()]

        # Format the date for display
        last_indices['Ultima Fecha'] = last_indices['FECHA_COMPLETA_TEMP'].dt.strftime('%Y-%m-%d')

        # Select and rename columns for the final summary DataFrame
        summary_df = last_indices[[cuenta_col, 'Ultima Fecha']].rename(columns={cuenta_col: 'Cuenta'})

        # Sort by account name for consistent display
        return summary_df.sort_values(by='Cuenta').reset_index(drop=True)

    except KeyError as ke:
         st.error(f"Error de Clave generando resumen: Falta la columna {ke}. Columnas disponibles: {df.columns.tolist()}")
         return pd.DataFrame({'Error': [f"Falta columna requerida: {ke}"]})
    except Exception as e:
         st.error(f"Error inesperado generando resumen de últimas fechas: {e}")
         st.error(traceback.format_exc())
         return pd.DataFrame({'Error': [f"Error interno: {e}"]})


# ------------------------------------------------------------------------------------
# --- Streamlit UI ---
# ------------------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Categorizador Bancario")
st.title("🏦 Categorizador Bancario Inteligente v5.4")
st.caption(f"Usando Archivo de Configuración: `{CONFIG_FILENAME}` | Base de Datos: `{DB_FILENAME}`")

# --- Carga BD (Sidebar) ---
st.sidebar.header("Base de Datos Global")
uploaded_db_file = st.sidebar.file_uploader(f"Cargar BD ({DB_FILENAME})", type=["csv", "xlsx", "xls"], key="db_loader", help=f"Carga la base de datos acumulada existente '{DB_FILENAME}'. Debe contener columnas como CONCEPTO, IMPORTE, AÑO, MES, DIA y una columna de categoría (CATEGORÍA o similar). Opcionalmente SUBCATEGORIA, COMERCIO, CUENTA.")

if uploaded_db_file:
    db_uploader_key = "db_loader_processed_id" # Use this key to track processing
    # Check if this is a new file upload instance
    if uploaded_db_file.file_id != st.session_state.get(db_uploader_key, None):
        st.sidebar.info(f"Cargando '{uploaded_db_file.name}'...")
        # Read the raw file
        df_db_loaded, _ = read_uploaded_file(uploaded_db_file)

        if df_db_loaded is not None and not df_db_loaded.empty:
             # Store the raw loaded data (or a standardized version if preferred)
             # Let's store it relatively raw, parse_accumulated handles the rest
             st.session_state.accumulated_data_raw_load = df_db_loaded.copy() # Keep a copy
             st.session_state[db_uploader_key] = uploaded_db_file.file_id # Mark as processed

             # --- Prepare and Extract Knowledge from newly loaded DB ---
             st.sidebar.info("Preparando BD cargada y extrayendo conocimiento...")
             # Use the corrected parse function
             df_for_knowledge = parse_accumulated_db_for_training(df_db_loaded.copy())

             if df_for_knowledge is not None:
                 # Store the *parsed* data as the main accumulated data for editing/saving later
                 # This ensures consistency
                 st.session_state.accumulated_data = df_for_knowledge.copy()
                 st.sidebar.success(f"BD '{uploaded_db_file.name}' procesada ({len(st.session_state.accumulated_data)} filas válidas).")

                 # Extract knowledge from the parsed data
                 st.session_state.learned_knowledge = extract_knowledge_std(df_for_knowledge)
                 knowledge_ok = bool(st.session_state.learned_knowledge.get('categorias'))
                 st.session_state.knowledge_loaded = knowledge_ok
                 if knowledge_ok:
                     st.sidebar.success("Conocimiento extraído/actualizado desde BD.")
                 else:
                      st.sidebar.warning("Se procesó la BD, pero no se pudo extraer conocimiento útil (¿faltan categorías válidas?).")

                 # Rerun to update UI reflecting loaded data/knowledge
                 st.rerun()
             else:
                  st.sidebar.error("Falló la preparación de la BD cargada. No se pudo extraer conocimiento.")
                  st.session_state.accumulated_data = pd.DataFrame() # Clear data on failure
                  st.session_state[db_uploader_key] = None # Reset processed ID


        elif df_db_loaded is not None and df_db_loaded.empty:
             st.sidebar.warning(f"El archivo BD '{uploaded_db_file.name}' está vacío.")
             st.session_state.accumulated_data = pd.DataFrame()
             st.session_state[db_uploader_key] = None # Reset processed ID
        else:
             # Error message already shown by read_uploaded_file
             st.session_state.accumulated_data = pd.DataFrame()
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
                        # Allow flexibility in learned_knowledge structure, check only basics
                        elif 'categorias' not in config_data['learned_knowledge']:
                             is_valid_config = False; error_msg += " Sección 'learned_knowledge' debe contener al menos 'categorias'."

                    # If valid, update session state
                    if is_valid_config:
                        st.session_state.bank_mappings = config_data.get('bank_mappings', {})
                        # Update knowledge, ensuring all expected keys exist even if loading partial config
                        loaded_knowledge = config_data.get('learned_knowledge', {})
                        knowledge_keys = ['categorias', 'subcategorias_por_cat', 'comercios_por_cat', 'subcat_unica_por_comercio_y_cat', 'subcat_mas_frecuente_por_comercio_y_cat', 'all_subcategories', 'all_comercios']
                        for k in knowledge_keys:
                            st.session_state.learned_knowledge[k] = loaded_knowledge.get(k, st.session_state.learned_knowledge.get(k)) # Keep old if missing in new

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
        if st.session_state.bank_mappings or st.session_state.knowledge_loaded:
            try:
                config_to_save = {
                    'bank_mappings': st.session_state.get('bank_mappings', {}),
                    'learned_knowledge': st.session_state.get('learned_knowledge', {})
                }
                config_json = json.dumps(config_to_save, indent=4, ensure_ascii=False)
                st.download_button(
                    label=f"💾 Descargar Config Actual", data=config_json.encode('utf-8'),
                    file_name=CONFIG_FILENAME, mime='application/json', key='download_config_f1'
                )
            except Exception as e: st.error(f"Error preparando config para descarga: {e}")
        else: st.info("No hay configuración (mapeos o conocimiento) en memoria para descargar.")

        st.divider()

        # --- Entrenamiento del Modelo ---
        st.subheader("(Re)Entrenar Modelo")
        st.write("Entrena el clasificador de categorías usando la Base de Datos Acumulada cargada.")

        # Use the *parsed* accumulated data for training check
        current_db_parsed = st.session_state.get('accumulated_data', pd.DataFrame())
        if current_db_parsed.empty:
            st.warning("Carga la Base de Datos Acumulada en la barra lateral (y asegúrate de que se procese correctamente) para poder entrenar.")
        else:
            if st.button("🧠 Entrenar/Reentrenar con BD", key="train_db_f1b", help="Usa la BD procesada para aprender categorías y (re)entrenar el modelo."):
                 with st.spinner("Entrenando modelo... (Usando datos ya procesados)"):
                    # We train directly on the already parsed data stored in session state
                    df_to_train = current_db_parsed.copy() # Use the parsed data

                    if df_to_train is not None and not df_to_train.empty:
                        # Knowledge should already be extracted from this data, but re-extract for safety?
                        # Or trust the knowledge loaded/extracted when DB was loaded/parsed. Let's trust.
                        # st.session_state.learned_knowledge = extract_knowledge_std(df_to_train) # Optional re-extract
                        # knowledge_ok_train = bool(st.session_state.learned_knowledge.get('categorias'))
                        # st.session_state.knowledge_loaded = knowledge_ok_train

                        # Train the classifier (uses CATEGORIA_STD)
                        model, vectorizer, report = train_classifier_std(df_to_train)

                        if model and vectorizer:
                            st.session_state.model = model
                            st.session_state.vectorizer = vectorizer
                            st.session_state.model_trained = True
                            st.session_state.training_report = report if report else "Modelo entrenado (sin informe detallado)."
                            st.success(f"¡Modelo entrenado exitosamente!")
                            st.sidebar.success("✅ Modelo Entrenado") # Update sidebar status

                            st.sidebar.divider()
                            st.sidebar.subheader("Evaluación Modelo (Entrenamiento)")
                            with st.sidebar.expander("Ver Informe de Clasificación"):
                                st.text(st.session_state.training_report)
                        else:
                            st.session_state.model = None; st.session_state.vectorizer = None
                            st.session_state.model_trained = False
                            st.session_state.training_report = report if report else "Fallo entrenamiento (razón desconocida)."
                            st.error(f"Fallo el entrenamiento del modelo.")
                            st.sidebar.error("❌ Entrenamiento Fallido") # Update sidebar status
                            st.sidebar.divider()
                            st.sidebar.subheader("Evaluación Modelo (Entrenamiento)")
                            st.sidebar.text(st.session_state.training_report) # Show error in sidebar too
                    else:
                        st.error("No hay datos procesados válidos en memoria para entrenar.")
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
             if categorias_display: st.dataframe(pd.DataFrame(categorias_display, columns=['Categoría']), use_container_width=True, height=150, hide_index=True)
             else: st.caption("Ninguna")

             col_k1, col_k2 = st.columns(2)
             with col_k1:
                  with st.expander(f"Subcategorías Conocidas ({len(all_subs_display)})", expanded=False):
                       if all_subs_display: st.dataframe(pd.DataFrame(all_subs_display, columns=['Subcategoría']), use_container_width=True, height=200, hide_index=True)
                       else: st.caption("Ninguna")
             with col_k2:
                   with st.expander(f"Comercios Conocidos ({len(all_coms_display)})", expanded=False):
                       if all_coms_display: st.dataframe(pd.DataFrame(all_coms_display, columns=['Comercio']), use_container_width=True, height=200, hide_index=True)
                       else: st.caption("Ninguno")

             with st.expander("Mostrar detalles avanzados del conocimiento (Debug)", expanded=False):
                 st.write("Subcategorías por Categoría:")
                 st.json(knowledge_display.get('subcategorias_por_cat', {}))
                 st.write("Comercios por Categoría:")
                 st.json(knowledge_display.get('comercios_por_cat', {}))
                 st.write("Regla Subcategoría Única (por Comercio y Cat):")
                 st.json(knowledge_display.get('subcat_unica_por_comercio_y_cat', {}))
                 st.write("Regla Subcategoría Más Frecuente (por Comercio y Cat):")
                 st.json(knowledge_display.get('subcat_mas_frecuente_por_comercio_y_cat', {}))
        else:
             st.info("No hay conocimiento base cargado o aprendido todavía. Carga una configuración o carga/procesa una BD.")


    # --- Columna Derecha: Definir Mapeos Bancarios ---
    with col1b:
        st.subheader("Definir Formatos Bancarios (Mapeo)")
        st.write("Enseña a la aplicación cómo leer archivos de diferentes bancos subiendo un archivo de ejemplo y mapeando sus columnas a los campos estándar.")

        existing_banks = list(st.session_state.get('bank_mappings', {}).keys())
        bank_options = sorted(list(set(["SANTANDER", "EVO", "WIZINK", "AMEX", "N26", "BBVA", "ING", "CAIXABANK"] + existing_banks)))
        new_bank_name = st.text_input("Nombre del Banco (o selecciona existente abajo):", key="new_bank_name_f2", placeholder="Ej: MI_BANCO_NUEVO").strip().upper()
        selected_bank_learn = st.selectbox("Banco Existente:", [""] + bank_options, key="bank_learn_f2_select", index=0, format_func=lambda x: x if x else "---Selecciona---")

        bank_to_map = new_bank_name if new_bank_name else selected_bank_learn
        if not bank_to_map:
             st.info("Escribe un nombre para un banco nuevo o selecciona uno existente para definir/editar su mapeo.")
        else:
            st.markdown(f"**Editando mapeo para: `{bank_to_map}`**")
            uploader_key = f"sample_uploader_{bank_to_map}"
            uploaded_sample_file = st.file_uploader(f"Cargar archivo de ejemplo de `{bank_to_map}`", type=["csv", "xlsx", "xls"], key=uploader_key)

            if uploaded_sample_file:
                df_sample, detected_columns = read_uploaded_file(uploaded_sample_file)

                if df_sample is not None and detected_columns:
                    st.write(f"Primeras filas del ejemplo '{uploaded_sample_file.name}':")
                    st.dataframe(df_sample.head(3), use_container_width=True)
                    st.write(f"Columnas detectadas en el archivo:")
                    st.code(f"{detected_columns}", language=None)

                    st.write("**Define el Mapeo:** Asocia las columnas del archivo a los campos estándar.")
                    saved_mapping = st.session_state.bank_mappings.get(bank_to_map, {})
                    saved_mapping_cols = saved_mapping.get('columns', {})

                    cols_with_none = [None] + detected_columns

                    sub_c1, sub_c2 = st.columns(2)
                    with sub_c1:
                        st.markdown("**Columnas Esenciales:**")
                        def get_index(col_name):
                            val = saved_mapping_cols.get(col_name)
                            try: return cols_with_none.index(val) if val in cols_with_none else 0
                            except ValueError: return 0

                        map_concepto = st.selectbox(f"`{CONCEPTO_STD}` (Descripción)", cols_with_none, index=get_index(CONCEPTO_STD), key=f"map_{CONCEPTO_STD}_{bank_to_map}", format_func=lambda x: x if x else "---Selecciona---")
                        map_importe = st.selectbox(f"`{IMPORTE_STD}` (Cantidad)", cols_with_none, index=get_index(IMPORTE_STD), key=f"map_{IMPORTE_STD}_{bank_to_map}", format_func=lambda x: x if x else "---Selecciona---")

                        st.markdown("**Manejo de Fecha:**")
                        is_single_date_saved = FECHA_STD in saved_mapping_cols and saved_mapping_cols.get(FECHA_STD) is not None
                        map_single_date = st.checkbox("Fecha en una sola columna", value=is_single_date_saved, key=f"map_single_date_{bank_to_map}")

                        map_fecha_unica=None; map_formato_fecha=None; map_año=None; map_mes=None; map_dia=None
                        if map_single_date:
                            map_fecha_unica = st.selectbox(f"`{FECHA_STD}` (Columna fecha)", cols_with_none, index=get_index(FECHA_STD), key=f"map_{FECHA_STD}_{bank_to_map}", format_func=lambda x: x if x else "---Selecciona---")
                            map_formato_fecha = st.text_input("Formato Fecha (ej: %d/%m/%Y)", value=saved_mapping.get('date_format', ''), key=f"map_date_format_{bank_to_map}", help="Usa códigos Python: %d día, %m mes, %Y año (4 dig), %y año (2 dig).")
                        else:
                            map_año = st.selectbox(f"`{AÑO_STD}` (Columna Año)", cols_with_none, index=get_index(AÑO_STD), key=f"map_{AÑO_STD}_{bank_to_map}", format_func=lambda x: x if x else "---Selecciona---")
                            map_mes = st.selectbox(f"`{MES_STD}` (Columna Mes)", cols_with_none, index=get_index(MES_STD), key=f"map_{MES_STD}_{bank_to_map}", format_func=lambda x: x if x else "---Selecciona---")
                            map_dia = st.selectbox(f"`{DIA_STD}` (Columna Día)", cols_with_none, index=get_index(DIA_STD), key=f"map_{DIA_STD}_{bank_to_map}", format_func=lambda x: x if x else "---Selecciona---")

                    with sub_c2:
                        st.markdown("**Columnas Opcionales:**")
                        map_comercio = st.selectbox(f"`{COMERCIO_STD}` (Comercio/Tienda)", cols_with_none, index=get_index(COMERCIO_STD), key=f"map_{COMERCIO_STD}_{bank_to_map}", format_func=lambda x: x if x else "---Ninguna---")

                        st.markdown("**Formato de Importe:**")
                        val_map_decimal_sep = st.text_input("Separador Decimal", value=saved_mapping.get('decimal_sep', ','), key=f"map_decimal_{bank_to_map}", max_chars=1)
                        val_map_thousands_sep = st.text_input("Separador Miles (si existe)", value=saved_mapping.get('thousands_sep', '') or '', key=f"map_thousands_{bank_to_map}", max_chars=1)

                    if st.button(f"💾 Guardar Mapeo para `{bank_to_map}`", key=f"save_mapping_{bank_to_map}"):
                        current_concepto = map_concepto
                        current_importe = map_importe
                        current_is_single_date = map_single_date
                        current_fecha_unica = map_fecha_unica
                        current_formato_fecha = map_formato_fecha.strip() if map_formato_fecha else ""
                        current_año = map_año
                        current_mes = map_mes
                        current_dia = map_dia
                        current_comercio = map_comercio
                        current_decimal_sep = val_map_decimal_sep.strip() if val_map_decimal_sep else ","
                        current_thousands_sep = val_map_thousands_sep.strip() if val_map_thousands_sep else None

                        final_mapping_cols = {}
                        if current_concepto: final_mapping_cols[CONCEPTO_STD] = current_concepto
                        if current_importe: final_mapping_cols[IMPORTE_STD] = current_importe
                        if current_is_single_date and current_fecha_unica: final_mapping_cols[FECHA_STD] = current_fecha_unica
                        if not current_is_single_date and current_año: final_mapping_cols[AÑO_STD] = current_año
                        if not current_is_single_date and current_mes: final_mapping_cols[MES_STD] = current_mes
                        if not current_is_single_date and current_dia: final_mapping_cols[DIA_STD] = current_dia
                        if current_comercio: final_mapping_cols[COMERCIO_STD] = current_comercio

                        is_valid = True; errors = []
                        if not final_mapping_cols.get(CONCEPTO_STD): errors.append(f"Mapea `{CONCEPTO_STD}`."); is_valid = False
                        if not final_mapping_cols.get(IMPORTE_STD): errors.append(f"Mapea `{IMPORTE_STD}`."); is_valid = False
                        if not current_decimal_sep: errors.append("Especifica Separador Decimal."); is_valid = False

                        if current_is_single_date:
                            if not final_mapping_cols.get(FECHA_STD): errors.append(f"Mapea `{FECHA_STD}`."); is_valid = False
                            elif not current_formato_fecha: errors.append("Especifica Formato Fecha."); is_valid = False
                            elif not ('%d' in current_formato_fecha and '%m' in current_formato_fecha and ('%Y' in current_formato_fecha or '%y' in current_formato_fecha)):
                                 errors.append("Formato Fecha inválido (ej: %d/%m/%Y)."); is_valid = False
                        else:
                            if not final_mapping_cols.get(AÑO_STD): errors.append(f"Mapea `{AÑO_STD}`."); is_valid = False
                            if not final_mapping_cols.get(MES_STD): errors.append(f"Mapea `{MES_STD}`."); is_valid = False
                            if not final_mapping_cols.get(DIA_STD): errors.append(f"Mapea `{DIA_STD}`."); is_valid = False

                        if is_valid:
                            mapping_to_save = {
                                'bank_name': bank_to_map, 'columns': final_mapping_cols,
                                'decimal_sep': current_decimal_sep, 'thousands_sep': current_thousands_sep,
                            }
                            if current_is_single_date: mapping_to_save['date_format'] = current_formato_fecha
                            st.session_state.bank_mappings[bank_to_map] = mapping_to_save
                            st.success(f"¡Mapeo para `{bank_to_map}` guardado/actualizado!")
                            st.info("Descarga la config actualizada desde el panel izquierdo.")
                        else:
                            st.error("No se pudo guardar. Corrige los errores:"); [st.error(f"- {e}") for e in errors]

                elif df_sample is not None and not detected_columns:
                     st.warning(f"Archivo ejemplo '{uploaded_sample_file.name}' vacío o sin columnas.")
                # Else: read_uploaded_file handled error

# ==============================================================
# --- Tab 2: Categorización y Gestión BD ---
# ==============================================================
with tab2:
    st.header("Categorización y Gestión de la Base de Datos")

    # --- Resumen Última Transacción ---
    st.subheader("Resumen: Última Transacción por Cuenta")
    with st.spinner("Calculando resumen..."):
        # Use the *parsed* data for summary
        df_summary = get_last_transaction_dates(st.session_state.get('accumulated_data', pd.DataFrame()).copy())
    if not df_summary.empty:
        if 'Error' in df_summary.columns or 'Mensaje' in df_summary.columns: st.info(df_summary.iloc[0, 0])
        else: st.dataframe(df_summary, use_container_width=True, hide_index=True)
    # else: st.info("BD vacía o sin datos válidos para resumen.") # Message handled by function

    st.divider()

    # --- Categorización de Nuevos Archivos ---
    st.subheader("Categorizar Nuevos Archivos y Añadir a BD")

    model_ready_for_pred = st.session_state.get('model_trained', False)
    mappings_available = bool(st.session_state.get('bank_mappings', {}))
    knowledge_ready = st.session_state.get('knowledge_loaded', False)
    can_categorize = model_ready_for_pred and mappings_available and knowledge_ready

    if not can_categorize:
        st.warning("⚠️ **Acción Requerida para Categorizar:**")
        if not knowledge_ready: st.markdown("- Falta **Conocimiento Base**. Ve a 'Configuración' -> Carga Config/BD o Entrena.")
        if not mappings_available: st.markdown("- Faltan **Formatos Bancarios**. Ve a 'Configuración' -> Define mapeos.")
        if not model_ready_for_pred: st.markdown("- Falta **Modelo Entrenado**. Ve a 'Configuración' -> Entrena con BD.")
    else:
        st.success("✅ Listo para categorizar nuevos archivos.")
        st.write("Selecciona el banco correspondiente al archivo que quieres categorizar y súbelo.")

        available_banks_for_pred = list(st.session_state.bank_mappings.keys())
        if not available_banks_for_pred: st.error("Error Interno: No hay mapeos disponibles.")
        else:
            selected_bank_predict = st.selectbox("Banco del archivo a categorizar:", available_banks_for_pred, key="bank_predict_f3", format_func=lambda x: x if x else "---Selecciona---")
            final_uploader_key = f"final_uploader_{selected_bank_predict}" if selected_bank_predict else "final_uploader_f3"
            uploaded_final_file = st.file_uploader(f"Cargar archivo de '{selected_bank_predict}' (sin categorizar)", type=["csv", "xlsx", "xls"], key=final_uploader_key, disabled=(not selected_bank_predict))

            if uploaded_final_file and selected_bank_predict:
                mapping_to_use = st.session_state.bank_mappings.get(selected_bank_predict)
                if not mapping_to_use: st.error(f"Error crítico: No se encontró mapeo para '{selected_bank_predict}'.")
                else:
                     st.info(f"Procesando archivo '{uploaded_final_file.name}' usando mapeo '{selected_bank_predict}'...")
                     df_std_new = None; df_pred_display = None

                     # --- Step 1: Read and Standardize ---
                     with st.spinner(f"Estandarizando datos..."):
                          df_raw_new, _ = read_uploaded_file(uploaded_final_file)
                          if df_raw_new is not None and not df_raw_new.empty:
                              df_std_new = standardize_data_with_mapping(df_raw_new.copy(), mapping_to_use)
                          elif df_raw_new is not None: st.warning(f"Archivo '{uploaded_final_file.name}' vacío.")
                          # Else: error handled by read_uploaded_file or standardize

                     # --- Step 2: Predict Categories and Apply Knowledge ---
                     if df_std_new is not None and not df_std_new.empty:
                          st.success(f"Estandarización OK ({len(df_std_new)} filas).")
                          with st.spinner("Categorizando..."):
                              st.session_state.debug_predictions = []
                              try:
                                   if TEXTO_MODELO not in df_std_new.columns: st.error(f"Error Interno: Falta '{TEXTO_MODELO}' después de estandarizar.")
                                   else:
                                        df_pred = df_std_new.dropna(subset=[TEXTO_MODELO]).copy()
                                        df_pred[TEXTO_MODELO] = df_pred[TEXTO_MODELO].fillna('')

                                        if not df_pred.empty:
                                             X_new_vec = st.session_state.vectorizer.transform(df_pred[TEXTO_MODELO])
                                             predictions_cat = st.session_state.model.predict(X_new_vec)
                                             df_pred[CATEGORIA_PREDICHA] = [str(p).capitalize() for p in predictions_cat]

                                             pred_comercios_final = []; pred_subcats_final = []; debug_info_list = []
                                             knowledge = st.session_state.learned_knowledge

                                             for index, row in df_pred.iterrows():
                                                 pred_cat_lower = str(row.get(CATEGORIA_PREDICHA, '')).lower()
                                                 input_comercio_lower = str(row.get(COMERCIO_STD, '')).lower().strip()
                                                 input_concepto_lower = str(row.get(CONCEPTO_STD, '')).lower().strip()
                                                 debug_step = f"F:{index}|Cat:'{pred_cat_lower}'|Com:'{input_comercio_lower}'|Con:'{input_concepto_lower[:20]}..'"

                                                 # --- Merchant Prediction ---
                                                 comercio_final = input_comercio_lower; best_match_comercio = None
                                                 known_comers_for_cat = knowledge.get('comercios_por_cat', {}).get(pred_cat_lower, [])
                                                 if input_comercio_lower and known_comers_for_cat:
                                                     try:
                                                          match_result = process.extractOne(input_comercio_lower, known_comers_for_cat)
                                                          if match_result and match_result[1] >= FUZZY_MATCH_THRESHOLD:
                                                              comercio_final = match_result[0]; best_match_comercio = match_result[0]
                                                              debug_step += f"->ComM:'{comercio_final}'({match_result[1]})"
                                                     except Exception as e_fuzzy: pass # Ignore fuzzy errors
                                                 pred_comercios_final.append(str(comercio_final).capitalize())

                                                 # --- Subcategory Prediction ---
                                                 subcat_final = ''; subcat_msg = "->Sub:"
                                                 comercio_lookup_key = best_match_comercio if best_match_comercio else input_comercio_lower

                                                 if comercio_lookup_key:
                                                    subcat_unica = knowledge.get('subcat_unica_por_comercio_y_cat', {}).get(pred_cat_lower, {}).get(comercio_lookup_key)
                                                    if subcat_unica: subcat_final = subcat_unica; subcat_msg += f"U(C:'{subcat_final}')"
                                                 if not subcat_final and comercio_lookup_key:
                                                     subcat_frecuente = knowledge.get('subcat_mas_frecuente_por_comercio_y_cat', {}).get(pred_cat_lower, {}).get(comercio_lookup_key)
                                                     if subcat_frecuente: subcat_final = subcat_frecuente; subcat_msg += f"F(C:'{subcat_final}')"
                                                 if not subcat_final and input_concepto_lower:
                                                      known_subcats_for_cat = knowledge.get('subcategorias_por_cat', {}).get(pred_cat_lower, [])
                                                      found_keywords = [sk for sk in known_subcats_for_cat if sk and re.search(r'\b' + re.escape(sk) + r'\b', input_concepto_lower, re.IGNORECASE)]
                                                      if len(found_keywords) == 1: subcat_final = found_keywords[0]; subcat_msg += f"KW({subcat_final})"
                                                      # elif len(found_keywords) > 1: subcat_msg += f"KW?(Multi:{found_keywords[:2]})" # Optional: note multiple keywords
                                                 if not subcat_final:
                                                     known_subcats_for_cat = knowledge.get('subcategorias_por_cat', {}).get(pred_cat_lower, [])
                                                     if len(known_subcats_for_cat) == 1: subcat_final = known_subcats_for_cat[0]; subcat_msg += f"U(K:'{subcat_final}')"

                                                 if not subcat_final and subcat_msg == "->Sub:": subcat_msg += "N/A"
                                                 pred_subcats_final.append(str(subcat_final).capitalize())
                                                 debug_info_list.append(debug_step + subcat_msg)

                                             df_pred[COMERCIO_PREDICHO] = pred_comercios_final
                                             df_pred[SUBCATEGORIA_PREDICHA] = pred_subcats_final
                                             st.session_state.debug_predictions = debug_info_list

                                             # --- Display Results ---
                                             st.subheader(f"📊 Resultados Categorización para '{uploaded_final_file.name}'")
                                             display_cols_order = [
                                                 CATEGORIA_PREDICHA, SUBCATEGORIA_PREDICHA, COMERCIO_PREDICHO,
                                                 CONCEPTO_STD, IMPORTE_STD, AÑO_STD, MES_STD, DIA_STD
                                             ]
                                             if CUENTA_COL_STD in df_pred.columns: display_cols_order.append(CUENTA_COL_STD)
                                             orig_cols = sorted([c for c in df_pred.columns if c.startswith('ORIG_') and c != CUENTA_COL_STD])
                                             display_cols_order.extend(orig_cols)
                                             internal_or_duplicate = [TEXTO_MODELO, COMERCIO_STD, SUBCATEGORIA_STD, CATEGORIA_STD] + display_cols_order
                                             other_cols = sorted([c for c in df_pred.columns if c not in internal_or_duplicate])
                                             display_cols_order.extend(other_cols)
                                             final_display_cols = [col for col in display_cols_order if col in df_pred.columns]

                                             st.dataframe(df_pred[final_display_cols], use_container_width=True, hide_index=True)
                                             df_pred_display = df_pred

                                        else: st.warning("No quedaron filas válidas para categorizar.")
                              except Exception as e_pred: st.error(f"Error categorización: {e_pred}"); st.error(traceback.format_exc())

                          # --- Display Debug Info ---
                          if st.session_state.debug_predictions:
                               with st.expander("Detalles Predicción (Debug)", expanded=False):
                                   max_debug_lines = 100
                                   st.text("\n".join(st.session_state.debug_predictions[:max_debug_lines]))
                                   if len(st.session_state.debug_predictions) > max_debug_lines: st.caption(f"... ({max_debug_lines} de {len(st.session_state.debug_predictions)} líneas)")

                          # --- Button to Add Results to DB ---
                          if df_pred_display is not None and not df_pred_display.empty:
                              file_identifier = uploaded_final_file.file_id if uploaded_final_file.file_id else uploaded_final_file.name
                              add_button_key = f"add_db_{file_identifier}"
                              if st.button(f"➕ Añadir {len(df_pred_display)} Resultados a BD", key=add_button_key):

                                  # Use the *parsed* accumulated data from session state
                                  current_db = st.session_state.get('accumulated_data', pd.DataFrame())
                                  df_to_append_raw = df_pred_display.copy()

                                  # Define columns to keep when appending
                                  db_cols_to_keep = list(DB_FINAL_COLS)
                                  if CUENTA_COL_STD in df_to_append_raw.columns:
                                       if CUENTA_COL_STD not in db_cols_to_keep: db_cols_to_keep.append(CUENTA_COL_STD)
                                  orig_cols_to_add = [c for c in df_to_append_raw.columns if c.startswith('ORIG_')]
                                  for col in orig_cols_to_add:
                                       if col not in db_cols_to_keep: db_cols_to_keep.append(col)
                                  final_cols_to_append = [col for col in db_cols_to_keep if col in df_to_append_raw.columns]
                                  df_to_append = df_to_append_raw[final_cols_to_append]

                                  # --- Duplicate Detection ---
                                  num_added = 0
                                  new_transactions_only = pd.DataFrame(columns=df_to_append.columns)

                                  if current_db.empty:
                                       st.write("BD vacía. Añadiendo todas las filas nuevas.")
                                       new_transactions_only = df_to_append
                                       num_added = len(new_transactions_only)
                                  else:
                                       st.write("Detectando duplicados...")
                                       key_cols_base = [AÑO_STD, MES_STD, DIA_STD, IMPORTE_STD, CONCEPTO_STD]
                                       key_cols = []
                                       keys_ok = True
                                       for col in key_cols_base:
                                           if col not in current_db.columns: st.error(f"Error Dup: Falta clave '{col}' en BD."); keys_ok = False
                                           if col not in df_to_append.columns: st.error(f"Error Dup: Falta clave '{col}' en datos nuevos."); keys_ok = False
                                           if keys_ok: key_cols.append(col)

                                       account_key_col_name = None
                                       if keys_ok:
                                           acc_col_db = CUENTA_COL_STD if CUENTA_COL_STD in current_db.columns else None
                                           acc_col_new = CUENTA_COL_STD if CUENTA_COL_STD in df_to_append.columns else None
                                           if acc_col_db and acc_col_new: key_cols.append(CUENTA_COL_STD); account_key_col_name = CUENTA_COL_STD
                                           # else: st.warning(f"'{CUENTA_COL_STD}' no usado para duplicados (falta en BD o nuevos).")

                                       if not keys_ok: num_added = -1
                                       else:
                                            try:
                                                df1 = current_db.copy(); df2 = df_to_append.copy()
                                                for df_temp in [df1, df2]:
                                                    df_temp[IMPORTE_STD] = pd.to_numeric(df_temp[IMPORTE_STD], errors='coerce').round(2)
                                                    for col in [AÑO_STD, MES_STD, DIA_STD]: df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce').astype('Int64').fillna(0).astype(int)
                                                    df_temp[CONCEPTO_STD] = df_temp[CONCEPTO_STD].fillna('').astype(str).str.lower().str.strip()
                                                    if account_key_col_name: df_temp[account_key_col_name] = df_temp[account_key_col_name].fillna('').astype(str).str.lower().str.strip()

                                                df1_clean_keys = df1.dropna(subset=key_cols); df2_clean_keys = df2.dropna(subset=key_cols)

                                                if df1_clean_keys.empty: new_transactions_only = df2_clean_keys
                                                elif not df2_clean_keys.empty:
                                                    merged = df2_clean_keys.merge(df1_clean_keys[key_cols].drop_duplicates(), on=key_cols, how='left', indicator=True)
                                                    new_transactions_only = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])

                                                num_added = len(new_transactions_only)
                                                num_duplicates = len(df2_clean_keys) - num_added
                                                num_invalid_keys_new = len(df2) - len(df2_clean_keys)
                                                if num_duplicates > 0: st.info(f"Omitidas {num_duplicates} filas duplicadas.")
                                                if num_invalid_keys_new > 0: st.warning(f"Omitidas {num_invalid_keys_new} filas nuevas por claves inválidas ({', '.join(key_cols)}).")
                                            except Exception as e_dup: st.error(f"Error detectando duplicados: {e_dup}"); num_added = -1

                                  # --- Append Non-Duplicates ---
                                  if num_added > 0:
                                       st.write(f"Añadiendo {num_added} nuevas transacciones...")
                                       combined_cols = current_db.columns.union(new_transactions_only.columns)
                                       current_db_reindexed = current_db.reindex(columns=combined_cols, fill_value='')
                                       new_transactions_reindexed = new_transactions_only.reindex(columns=combined_cols, fill_value='')

                                       st.session_state.accumulated_data = pd.concat(
                                           [current_db_reindexed, new_transactions_reindexed], ignore_index=True
                                       )

                                       # --- Re-extract Knowledge AFTER adding data ---
                                       st.info("Actualizando conocimiento base...")
                                       # Use the main accumulated_data which is already parsed
                                       st.session_state.learned_knowledge = extract_knowledge_std(st.session_state.accumulated_data.copy())
                                       knowledge_ok_add = bool(st.session_state.learned_knowledge.get('categorias'))
                                       st.session_state.knowledge_loaded = knowledge_ok_add
                                       if knowledge_ok_add: st.sidebar.info("Conocimiento actualizado (tras añadir).")
                                       else: st.sidebar.warning("BD actualizada, pero falló re-extracción conocimiento.")

                                       st.success(f"¡Éxito! {num_added} transacciones NUEVAS añadidas.")
                                       st.rerun()
                                  elif num_added == 0: st.info("No se añadieron filas nuevas (0 únicas o todas duplicadas/inválidas).")
                                  else: st.error("No se añadieron filas por error en duplicados.")


                     elif df_std_new is not None and df_std_new.empty: st.warning(f"Archivo '{uploaded_final_file.name}' sin filas válidas tras estandarizar.")
                     # Else: Failure handled earlier

    st.divider()

    # ==============================================================
    # --- Sub-Sección: Ver/Gestionar Base de Datos Acumulada (Editable) ---
    # ==============================================================
    st.subheader("Base de Datos Acumulada (Editable)")
    # Use the *parsed* data for display/editing
    db_state_tab = st.session_state.get('accumulated_data', pd.DataFrame())

    if db_state_tab is not None and not db_state_tab.empty:
        st.write(f"Mostrando {len(db_state_tab)} filas. Doble clic para editar (Categoría, Subcategoría, Comercio).")

        knowledge = st.session_state.get('learned_knowledge', {})
        categorias_options = [""] + sorted([c.capitalize() for c in knowledge.get('categorias', [])])
        all_subcats_options = [""] + sorted([s.capitalize() for s in knowledge.get('all_subcategories', [])])
        all_comers_options = [""] + sorted([c.capitalize() for c in knowledge.get('all_comercios', [])])

        db_editable_copy = db_state_tab.copy()
        cols_to_ensure = { CATEGORIA_PREDICHA: CATEGORIA_STD, SUBCATEGORIA_PREDICHA: SUBCATEGORIA_STD, COMERCIO_PREDICHO: COMERCIO_STD }

        # Ensure PREDICHA columns exist for editing, potentially copying from STD
        # Also ensure they are string type for the editor's selectbox
        for pred_col, std_col in cols_to_ensure.items():
             if pred_col not in db_editable_copy.columns:
                 if std_col in db_editable_copy.columns: db_editable_copy[pred_col] = db_editable_copy[std_col]
                 else: db_editable_copy[pred_col] = ''
             # Ensure string type and fill NaNs
             db_editable_copy[pred_col] = db_editable_copy[pred_col].fillna('').astype(str)


        column_config_editor = {
            CATEGORIA_PREDICHA: st.column_config.SelectboxColumn("Categoría Editada", width="medium", options=categorias_options, required=False),
            SUBCATEGORIA_PREDICHA: st.column_config.SelectboxColumn("Subcategoría Editada", width="medium", options=all_subcats_options, required=False),
            COMERCIO_PREDICHO: st.column_config.SelectboxColumn("Comercio Estandarizado", width="medium", options=all_comers_options, required=False),
            CONCEPTO_STD: st.column_config.TextColumn("Concepto", disabled=True),
            IMPORTE_STD: st.column_config.NumberColumn("Importe", format="%.2f", disabled=True),
            AÑO_STD: st.column_config.NumberColumn("Año", format="%d", disabled=True),
            MES_STD: st.column_config.NumberColumn("Mes", format="%02d", disabled=True),
            DIA_STD: st.column_config.NumberColumn("Día", format="%02d", disabled=True),
            CUENTA_COL_STD: st.column_config.TextColumn("Cuenta Origen", disabled=True),
        }
        for col in db_editable_copy.columns: # Disable ORIG_ cols
             if col.startswith("ORIG_") and col != CUENTA_COL_STD and col not in column_config_editor: # Avoid re-adding cuenta
                 label = col.replace("ORIG_", "")[:25] + ('...' if len(col.replace("ORIG_", "")) > 25 else '')
                 column_config_editor[col] = st.column_config.TextColumn(f"Orig: {label}", disabled=True, help=f"Columna original: {col}")
             # Disable base STD cols if PRED versions exist
             elif col in [CATEGORIA_STD, SUBCATEGORIA_STD, COMERCIO_STD] and cols_to_ensure.get(col.replace('_STD','_PREDICHA')) and col not in column_config_editor:
                   column_config_editor[col] = st.column_config.TextColumn(f"{col} (Base)", disabled=True, help="Valor base antes de predicción/edición.")

        cols_order_edit = [
            CATEGORIA_PREDICHA, SUBCATEGORIA_PREDICHA, COMERCIO_PREDICHO,
            CONCEPTO_STD, IMPORTE_STD, AÑO_STD, MES_STD, DIA_STD
        ]
        if CUENTA_COL_STD in db_editable_copy.columns: cols_order_edit.append(CUENTA_COL_STD)
        orig_cols_edit = sorted([col for col in db_editable_copy.columns if col.startswith('ORIG_') and col != CUENTA_COL_STD])
        cols_order_edit.extend(orig_cols_edit)
        # Add base STD cols if they exist
        base_std_cols = [c for c in [CATEGORIA_STD, SUBCATEGORIA_STD, COMERCIO_STD] if c in db_editable_copy.columns and c not in cols_order_edit]
        cols_order_edit.extend(base_std_cols)
        internal_or_existing = [TEXTO_MODELO] + cols_order_edit
        other_cols_edit = sorted([col for col in db_editable_copy.columns if col not in internal_or_existing])
        cols_order_edit.extend(other_cols_edit)

        final_cols_edit_display = [col for col in cols_order_edit if col in db_editable_copy.columns]

        edited_df = st.data_editor(
            db_editable_copy[final_cols_edit_display],
            key="db_editor_main", column_config=column_config_editor,
            num_rows="dynamic", use_container_width=True, hide_index=True, height=600
        )

        if st.button("💾 Confirmar Cambios en BD", key="save_edited_db"):
            edited_df_clean = edited_df.copy()
            essential_cols_check = [CATEGORIA_PREDICHA, CONCEPTO_STD, IMPORTE_STD, AÑO_STD, MES_STD, DIA_STD]
            valid_edit = True; missing_essentials = []
            for col in essential_cols_check:
                 if col not in edited_df_clean.columns: missing_essentials.append(col); valid_edit = False

            if not valid_edit: st.error(f"Error Crítico: Faltan columnas esenciales ({', '.join(missing_essentials)}). Cambios NO guardados.")
            else:
                # Update main data state with the edited version
                st.session_state.accumulated_data = edited_df_clean.copy()

                # Re-extract Knowledge AFTER saving edits
                st.info("Actualizando conocimiento base...")
                # Parse again to ensure CATEGORIA_STD is present for knowledge extraction if needed
                df_for_knowledge_update_edit = parse_accumulated_db_for_training(st.session_state.accumulated_data.copy())
                if df_for_knowledge_update_edit is not None:
                      st.session_state.learned_knowledge = extract_knowledge_std(df_for_knowledge_update_edit)
                      knowledge_ok_edit = bool(st.session_state.learned_knowledge.get('categorias'))
                      st.session_state.knowledge_loaded = knowledge_ok_edit
                      if knowledge_ok_edit: st.sidebar.info("Conocimiento actualizado (tras edición).")
                      else: st.sidebar.warning("Error actualizando conocimiento tras edición.")
                else: st.sidebar.warning("Error preparando datos editados para actualizar conocimiento.")

                st.success("¡Cambios guardados en Base de Datos en memoria!")
                st.rerun()

    else: st.info("BD acumulada vacía. Cárgala o añade datos categorizados.")


# ==============================================================
# --- Sidebar Info and State ---
# ==============================================================
st.sidebar.divider(); st.sidebar.header("Acerca de")
st.sidebar.info("Categorizador Bancario v5.4"); st.sidebar.divider(); st.sidebar.subheader("Estado Actual")

model_ready_sidebar = st.session_state.get('model_trained', False)
knowledge_ready_sidebar = st.session_state.get('knowledge_loaded', False)
if model_ready_sidebar: st.sidebar.success("✅ Modelo Entrenado")
elif knowledge_ready_sidebar: st.sidebar.info("ℹ️ Conocimiento Cargado (Modelo no entrenado)")
else: st.sidebar.warning("❌ Sin Modelo ni Conocimiento")

num_mappings = len(st.session_state.get('bank_mappings', {}))
if num_mappings > 0: st.sidebar.success(f"✅ Mapeos Definidos ({num_mappings})")
else: st.sidebar.warning("❌ Sin Mapeos Bancarios")

# Check the *parsed* data for DB status
db_state_sidebar = st.session_state.get('accumulated_data', pd.DataFrame())
db_rows = len(db_state_sidebar) if db_state_sidebar is not None else 0
if db_rows > 0: st.sidebar.success(f"✅ BD Procesada ({db_rows} filas)")
else: st.sidebar.info("ℹ️ BD Vacía o Sin Procesar")


# ==============================================================
# --- Descarga BD (Sidebar) ---
# ==============================================================
st.sidebar.divider(); st.sidebar.subheader("Guardar Base de Datos")
# Save the *parsed* data
db_state_sidebar_save = st.session_state.get('accumulated_data', pd.DataFrame())

if db_state_sidebar_save is not None and not db_state_sidebar_save.empty:
    try:
        # Define columns to export - Use the current columns of the parsed data
        # Prioritize the PREDICHA versions if they exist, along with essentials and ORIG_
        cols_to_export_db = []
        # Start with the preferred final columns
        preferred_order = [
            CATEGORIA_PREDICHA, SUBCATEGORIA_PREDICHA, COMERCIO_PREDICHO,
            CONCEPTO_STD, IMPORTE_STD, AÑO_STD, MES_STD, DIA_STD, CUENTA_COL_STD
        ]
        for col in preferred_order:
             if col in db_state_sidebar_save.columns:
                 cols_to_export_db.append(col)

        # Add all ORIG_ columns not already included
        orig_cols_export = sorted([c for c in db_state_sidebar_save.columns if c.startswith('ORIG_') and c not in cols_to_export_db])
        cols_to_export_db.extend(orig_cols_export)

        # Add any other columns from the processed data that weren't included yet
        other_cols_export = sorted([c for c in db_state_sidebar_save.columns if c not in cols_to_export_db and c != TEXTO_MODELO])
        cols_to_export_db.extend(other_cols_export)


        # Filter list to ensure all columns actually exist
        final_cols_to_export = [col for col in cols_to_export_db if col in db_state_sidebar_save.columns]
        df_to_export = db_state_sidebar_save[final_cols_to_export].copy()

        # --- CSV Download ---
        try:
             db_csv_output_sb = df_to_export.to_csv(index=False, sep=';', decimal=',', encoding='utf-8-sig')
             st.sidebar.download_button(label=f"💾 Descargar BD (CSV)", data=db_csv_output_sb, file_name=DB_FILENAME, mime='text/csv', key='download_db_csv_sb')
        except Exception as e_csv_down: st.sidebar.error(f"Error generando CSV: {e_csv_down}")

        # --- Excel Download ---
        try:
            output_excel_sb = io.BytesIO()
            with pd.ExcelWriter(output_excel_sb, engine='openpyxl') as writer: df_to_export.to_excel(writer, index=False, sheet_name='Gastos')
            excel_data_sb = output_excel_sb.getvalue()
            db_excel_filename_sb = DB_FILENAME.replace('.csv', '.xlsx') if DB_FILENAME.lower().endswith('.csv') else f"{DB_FILENAME}.xlsx"
            st.sidebar.download_button(label=f"💾 Descargar BD (Excel)", data=excel_data_sb, file_name=db_excel_filename_sb, mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', key='download_db_excel_sb')
        except ImportError: st.sidebar.info("Instala 'openpyxl' para descargar Excel: pip install openpyxl")
        except Exception as e_xls_down: st.sidebar.error(f"Error generando Excel: {e_xls_down}")

    except Exception as e_db_down_prep: st.sidebar.error(f"Error preparando BD para descarga: {e_db_down_prep}"); st.sidebar.error(traceback.format_exc())

else: st.sidebar.info("BD vacía o sin procesar, nada que guardar.")
