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
from fuzzywuzzy import process
import re

# --- Constantes ---
CONCEPTO_STD = 'CONCEPTO_STD'; COMERCIO_STD = 'COMERCIO_STD'; IMPORTE_STD = 'IMPORTE_STD'
AÑO_STD = 'AÑO'; MES_STD = 'MES'; DIA_STD = 'DIA'; FECHA_STD = 'FECHA_STD'
CATEGORIA_STD = 'CATEGORIA_STD'; SUBCATEGORIA_STD = 'SUBCATEGORIA_STD'
TEXTO_MODELO = 'TEXTO_MODELO'; CATEGORIA_PREDICHA = 'CATEGORIA_PREDICHA'
SUBCATEGORIA_PREDICHA = 'SUBCATEGORIA_PREDICHA'; COMERCIO_PREDICHO = 'COMERCIO_PREDICHO'
DB_FINAL_COLS = [CATEGORIA_PREDICHA, SUBCATEGORIA_PREDICHA, COMERCIO_PREDICHO, CONCEPTO_STD, IMPORTE_STD, AÑO_STD, MES_STD, DIA_STD]
MANDATORY_STD_COLS = [CONCEPTO_STD, IMPORTE_STD, FECHA_STD]
OPTIONAL_STD_COLS = [COMERCIO_STD]
CONFIG_FILENAME = "Configuracion_Categorizador.json"
DB_FILENAME = "Database_Gastos_Acumulados.csv"
FUZZY_MATCH_THRESHOLD = 80 # Ajustable
CUENTA_COL_ORIG = 'CUENTA'; CUENTA_COL_STD = 'ORIG_CUENTA'

# --- Session State Initialization ---
if 'model_trained' not in st.session_state: st.session_state.model_trained = False
if 'knowledge_loaded' not in st.session_state: st.session_state.knowledge_loaded = False
if 'model' not in st.session_state: st.session_state.model = None
if 'vectorizer' not in st.session_state: st.session_state.vectorizer = None
if 'bank_mappings' not in st.session_state: st.session_state.bank_mappings = {}
if 'training_report' not in st.session_state: st.session_state.training_report = "Modelo no entrenado."
if 'config_loader_processed_id' not in st.session_state: st.session_state.config_loader_processed_id = None
if 'accumulated_data' not in st.session_state: st.session_state.accumulated_data = pd.DataFrame()
if 'db_loader_processed_id' not in st.session_state: st.session_state.db_loader_processed_id = None
if 'learned_knowledge' not in st.session_state:
    st.session_state.learned_knowledge = {'categorias': [], 'subcategorias_por_cat': {}, 'comercios_por_cat': {}, 'subcat_unica_por_comercio_y_cat': {}, 'subcat_mas_frecuente_por_comercio_y_cat': {}, 'all_subcategories': [], 'all_comercios': []}
if 'debug_predictions' not in st.session_state: st.session_state.debug_predictions = []

# --- Funciones ---
@st.cache_data
def read_uploaded_file(uploaded_file):
    # ... (sin cambios) ...
    if uploaded_file is None: return None, []
    try:
        file_name = uploaded_file.name; bytes_data = uploaded_file.getvalue(); df = None
        if file_name.lower().endswith('.csv'):
            sniffer_content = bytes_data.decode('utf-8', errors='replace'); sniffer = io.StringIO(sniffer_content); sep = ';'
            try:
                sample_data = sniffer.read(min(1024 * 20, len(sniffer_content)))
                if sample_data:
                     dialect = pd.io.parsers.readers.csv.Sniffer().sniff(sample_data)
                     if dialect.delimiter in [',', ';', '\t', '|']: sep = dialect.delimiter
                else: st.error(f"'{file_name}' vacío."); return None, []
            except Exception: pass
            try: df = pd.read_csv(io.BytesIO(bytes_data), encoding='utf-8', sep=sep, low_memory=False)
            except UnicodeDecodeError: df = pd.read_csv(io.BytesIO(bytes_data), encoding='latin1', sep=sep, low_memory=False)
            except Exception as read_err: st.error(f"Error CSV '{file_name}' sep '{sep}': {read_err}"); return None, []
        elif file_name.lower().endswith('.xlsx'):
            try: df = pd.read_excel(io.BytesIO(bytes_data), engine='openpyxl')
            except ImportError: st.error("Instala 'openpyxl'."); return None, []
            except Exception as read_excel_err: st.error(f"Error XLSX '{file_name}': {read_excel_err}"); return None, []
        elif file_name.lower().endswith('.xls'):
            try: df = pd.read_excel(io.BytesIO(bytes_data), engine='xlrd')
            except ImportError: st.error("Instala 'xlrd'."); return None, []
            except Exception as read_excel_err: st.error(f"Error XLS '{file_name}': {read_excel_err}"); return None, []
        else: st.error(f"Formato no soportado: '{file_name}'."); return None, []
        if df is not None:
            if df.empty: st.warning(f"'{file_name}' sin datos."); detected_columns = [str(col).strip() for col in df.columns] if hasattr(df, 'columns') else []; return df, detected_columns
            original_columns = df.columns.tolist(); df.columns = [str(col).strip() for col in original_columns]; detected_columns = df.columns.tolist()
            return df, detected_columns
        else: st.error(f"Fallo lectura '{file_name}'."); return None, []
    except Exception as e: st.error(f"Error Gral leyendo '{uploaded_file.name if uploaded_file else ''}': {e}"); st.error(traceback.format_exc()); return None, []

def parse_historic_categorized(df_raw):
    # ... (sin cambios) ...
    try:
        if not isinstance(df_raw, pd.DataFrame): st.error("Parse Histórico: No es DF."); return None
        df = df_raw.copy(); df.columns = [str(col).upper().strip() for col in df.columns]
        required = ['CONCEPTO', 'CATEGORÍA', 'SUBCATEGORIA', 'IMPORTE', 'AÑO', 'MES', 'DIA']
        if 'COMERCIO' not in df.columns: df['COMERCIO'] = ''
        if CUENTA_COL_ORIG not in df.columns: df[CUENTA_COL_ORIG] = ''
        missing = [col for col in required if col not in df.columns]
        if missing: st.error(f"Histórico: Faltan cols: {missing}"); return None
        df_std = pd.DataFrame()
        text_map = { CONCEPTO_STD: 'CONCEPTO', COMERCIO_STD: 'COMERCIO', CATEGORIA_STD: 'CATEGORÍA', SUBCATEGORIA_STD: 'SUBCATEGORIA', CUENTA_COL_STD: CUENTA_COL_ORIG}
        for std_col, raw_col in text_map.items():
            if raw_col not in df.columns:
                 if std_col in [COMERCIO_STD, CUENTA_COL_STD]: df_std[std_col] = ''; continue
                 st.error(f"Error Interno: Falta '{raw_col}'."); return None
            try:
                series = df[raw_col].fillna('').astype(str)
                df_std[std_col] = series.str.lower().str.strip() if pd.api.types.is_string_dtype(series.dtype) else series.apply(lambda x: str(x).lower().strip())
            except AttributeError as ae: st.error(f"!!! Error Atributo '{raw_col}' -> '{std_col}'."); return None
            except Exception as e: st.error(f"Error proc. texto '{raw_col}': {e}"); return None
        try:
            imp_str = df['IMPORTE'].astype(str).str.replace(',', '.', regex=False)
            df_std[IMPORTE_STD] = pd.to_numeric(imp_str, errors='coerce')
            if df_std[IMPORTE_STD].isnull().any(): st.warning("Histórico: Importes no numéricos.")
        except Exception as e: st.error(f"Error proc. IMPORTE histórico: {e}"); return None
        try:
            for col in ['AÑO', 'MES', 'DIA']: df_std[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        except Exception as e: st.error(f"Error proc. Fechas históricas: {e}"); return None
        try:
            if CONCEPTO_STD not in df_std: df_std[CONCEPTO_STD] = ''
            if COMERCIO_STD not in df_std: df_std[COMERCIO_STD] = ''
            df_std[TEXTO_MODELO] = (df_std[CONCEPTO_STD] + ' ' + df_std[COMERCIO_STD]).str.strip()
        except Exception as e: st.error(f"Error creando TEXTO_MODELO: {e}"); return None
        if CATEGORIA_STD not in df_std.columns: st.error("Falta CATEGORIA_STD."); return None
        df_std = df_std.dropna(subset=[IMPORTE_STD, CATEGORIA_STD])
        df_std = df_std[df_std[CATEGORIA_STD] != '']
        if df_std.empty: st.warning("Histórico: No quedaron filas válidas."); return pd.DataFrame()
        return df_std
    except Exception as e: st.error(f"Error Gral parseando histórico: {e}"); st.error(traceback.format_exc()); return None

# **** MODIFICADO: extract_knowledge_std con más verificaciones ****
@st.cache_data
def extract_knowledge_std(df_std):
    """Extrae conocimiento y relaciones (incluyendo subcategoría más frecuente)."""
    knowledge = {
        'categorias': [], 'subcategorias_por_cat': {}, 'comercios_por_cat': {},
        'subcat_unica_por_comercio_y_cat': {}, 'subcat_mas_frecuente_por_comercio_y_cat': {},
        'all_subcategories': [], 'all_comercios': []
    }
    if df_std is None or df_std.empty:
        st.warning("extract_knowledge: DataFrame de entrada vacío.")
        return knowledge
    if CATEGORIA_STD not in df_std.columns:
         st.warning(f"extract_knowledge: Falta columna '{CATEGORIA_STD}'.")
         return knowledge

    try:
        # Asegurar que las columnas existan y limpiarlas ANTES de usarlas
        df_clean = df_std.copy()
        if CATEGORIA_STD in df_clean: df_clean[CATEGORIA_STD] = df_clean[CATEGORIA_STD].fillna('').astype(str).str.lower().str.strip()
        has_subcat = SUBCATEGORIA_STD in df_clean.columns
        if has_subcat: df_clean[SUBCATEGORIA_STD] = df_clean[SUBCATEGORIA_STD].fillna('').astype(str).str.lower().str.strip()
        has_comercio = COMERCIO_STD in df_clean.columns
        if has_comercio: df_clean[COMERCIO_STD] = df_clean[COMERCIO_STD].fillna('').astype(str).str.lower().str.strip()

        # Filtrar filas sin categoría válida
        df_clean = df_clean[df_clean[CATEGORIA_STD] != '']
        if df_clean.empty: st.warning("extract_knowledge: Sin filas tras filtrar categorías vacías."); return knowledge

        knowledge['categorias'] = sorted(df_clean[CATEGORIA_STD].unique())
        all_subcats = set(); all_comers = set()

        for cat in knowledge['categorias']:
            df_cat = df_clean[df_clean[CATEGORIA_STD] == cat]
            knowledge['subcategorias_por_cat'][cat] = []
            knowledge['comercios_por_cat'][cat] = []
            knowledge['subcat_unica_por_comercio_y_cat'][cat] = {}
            knowledge['subcat_mas_frecuente_por_comercio_y_cat'][cat] = {}

            if has_subcat:
                 subcats = df_cat[SUBCATEGORIA_STD].unique()
                 current_subcats = sorted([s for s in subcats if s]) # Filtrar vacíos
                 knowledge['subcategorias_por_cat'][cat] = current_subcats
                 all_subcats.update(current_subcats)

            if has_comercio:
                 df_cat_comers = df_cat[df_cat[COMERCIO_STD] != ''] # Filtrar comercios vacíos
                 comers = df_cat_comers[COMERCIO_STD].unique()
                 current_comers = sorted([c for c in comers if c and c != 'n/a']) # Filtrar N/A también
                 knowledge['comercios_por_cat'][cat] = current_comers
                 all_comers.update(current_comers)

                 if has_subcat:
                     for comercio in knowledge['comercios_por_cat'][cat]:
                         df_comercio_cat = df_cat_comers[
                             (df_cat_comers[COMERCIO_STD] == comercio) &
                             (df_cat_comers[SUBCATEGORIA_STD] != '') # Solo contar si hay subcategoría
                         ]
                         if not df_comercio_cat.empty:
                             subcats_comercio = df_comercio_cat[SUBCATEGORIA_STD]
                             unique_subcats = subcats_comercio.unique(); comercio_key = comercio; cat_key = cat
                             if len(unique_subcats) == 1: knowledge['subcat_unica_por_comercio_y_cat'][cat_key][comercio_key] = unique_subcats[0]
                             if len(subcats_comercio) > 0:
                                 try:
                                     most_frequent = subcats_comercio.value_counts().idxmax()
                                     knowledge['subcat_mas_frecuente_por_comercio_y_cat'][cat_key][comercio_key] = most_frequent
                                 except Exception:
                                     if len(unique_subcats)>0: knowledge['subcat_mas_frecuente_por_comercio_y_cat'][cat_key][comercio_key] = unique_subcats[0]

        knowledge['all_subcategories'] = sorted(list(all_subcats))
        knowledge['all_comercios'] = sorted(list(all_comers))
        # Debug print
        # st.write("Debug Knowledge Extraction Results:")
        # st.json(knowledge, expanded=False)

    except Exception as e_kg: st.error(f"Error extrayendo conocimiento: {e_kg}"); st.error(traceback.format_exc())
    return knowledge

@st.cache_resource
def train_classifier_std(df_std):
    # ... (sin cambios) ...
    report = "Modelo no entrenado."; model = None; vectorizer = None
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
    # ... (sin cambios) ...
    try:
        df_std = pd.DataFrame(); df = df_raw.copy()
        df.columns = [str(col).strip() for col in df.columns]
        original_columns = df.columns.tolist(); temp_std_data = {}; source_cols_used = []
        found_essential = {sc: False for sc in MANDATORY_STD_COLS if sc != FECHA_STD}
        for std_col, source_col in mapping['columns'].items():
             if source_col in original_columns:
                  temp_std_data[std_col] = df[source_col]; source_cols_used.append(source_col)
                  if std_col in found_essential: found_essential[std_col] = True
             else:
                  is_ess = std_col in [CONCEPTO_STD, IMPORTE_STD] or \
                           (std_col==FECHA_STD and not all(c in mapping['columns'] for c in [AÑO_STD, MES_STD, DIA_STD])) or \
                           (std_col in [AÑO_STD, MES_STD, DIA_STD] and FECHA_STD not in mapping['columns'])
                  if is_ess: st.error(f"Col. esencial mapeada '{source_col}' ('{std_col}') no encontrada."); return None
        df_std = pd.DataFrame(temp_std_data)
        missing_ess = [k for k, v in found_essential.items() if not v]
        if missing_ess: st.error(f"Faltan mapeos esenciales: {missing_ess}"); return None
        fecha_cs = mapping['columns'].get(FECHA_STD); año_cs = mapping['columns'].get(AÑO_STD)
        mes_cs = mapping['columns'].get(MES_STD); dia_cs = mapping['columns'].get(DIA_STD)
        date_ok = False
        if FECHA_STD in df_std.columns:
            date_fmt = mapping.get('date_format');
            if not date_fmt: st.error("Falta formato fecha."); return None
            try:
                dates = pd.to_datetime(df_std[FECHA_STD].astype(str).str.strip(), format=date_fmt, errors='coerce')
                if dates.isnull().all(): st.error(f"Ninguna fecha coincide con formato '{date_fmt}'."); return None
                if dates.isnull().any(): st.warning("Algunas fechas no coinciden con formato.")
                df_std[AÑO_STD] = dates.dt.year.fillna(0).astype(int); df_std[MES_STD] = dates.dt.month.fillna(0).astype(int); df_std[DIA_STD] = dates.dt.day.fillna(0).astype(int)
                df_std = df_std.drop(columns=[FECHA_STD]); date_ok = True
            except Exception as e_dt: st.error(f"Error proc. fecha única: {e_dt}"); return None
        elif all(c in df_std.columns for c in [AÑO_STD, MES_STD, DIA_STD]):
            try:
                for c in [AÑO_STD, MES_STD, DIA_STD]: df_std[c] = pd.to_numeric(df_std[c], errors='coerce').fillna(0).astype(int)
                date_ok = True
            except Exception as e_num: st.error(f"Error convirtiendo A/M/D: {e_num}"); return None
        else: st.error("Mapeo fecha incompleto."); return None
        if not date_ok: return None
        if IMPORTE_STD in df_std.columns:
            try:
                imp_str = df_std[IMPORTE_STD].fillna('0').astype(str)
                ts = mapping.get('thousands_sep'); ds = mapping.get('decimal_sep', ',')
                if ts: imp_str = imp_str.str.replace(ts, '', regex=False)
                imp_str = imp_str.str.replace(ds, '.', regex=False)
                df_std[IMPORTE_STD] = pd.to_numeric(imp_str, errors='coerce')
                if df_std[IMPORTE_STD].isnull().any(): st.warning("Algunos importes no convertidos.")
            except Exception as e_imp: st.error(f"Error proc. importe: {e_imp}"); return None
        else: st.error("Falta IMPORTE_STD mapeado."); return None
        for c in [CONCEPTO_STD, COMERCIO_STD]:
            if c in df_std.columns: df_std[c] = df_std[c].fillna('').astype(str).str.lower().str.strip()
            elif c == COMERCIO_STD: df_std[COMERCIO_STD] = ''
        if CONCEPTO_STD not in df_std: df_std[CONCEPTO_STD] = ''
        if COMERCIO_STD not in df_std: df_std[COMERCIO_STD] = ''
        df_std[TEXTO_MODELO] = (df_std[CONCEPTO_STD] + ' ' + df_std[COMERCIO_STD]).str.strip()
        original_cols_to_keep = [c for c in original_columns if c not in source_cols_used]
        cuenta_col_orig_upper = CUENTA_COL_ORIG.upper()
        if cuenta_col_orig_upper in original_columns and cuenta_col_orig_upper not in source_cols_used: original_cols_to_keep.append(cuenta_col_orig_upper)
        for col in original_cols_to_keep:
            target_col_name = f"ORIG_{col}"; sfx = 1
            while target_col_name in df_std.columns: target_col_name = f"ORIG_{col}_{sfx}"; sfx += 1
            if col == cuenta_col_orig_upper: target_col_name = CUENTA_COL_STD
            if target_col_name not in df_std.columns: df_std[target_col_name] = df[col]
        df_std = df_std.dropna(subset=[IMPORTE_STD, TEXTO_MODELO])
        df_std = df_std[df_std[TEXTO_MODELO] != '']
        return df_std
    except Exception as e: st.error(f"Error Gral aplicando mapeo '{mapping.get('bank_name', '?')}': {e}"); st.error(traceback.format_exc()); return None

def parse_accumulated_db_for_training(df_db):
    # ... (código sin cambios) ...
    if not isinstance(df_db, pd.DataFrame) or df_db.empty: st.error("BD Acumulada vacía."); return None
    df = df_db.copy(); df.columns = [str(col).upper().strip() for col in df.columns]
    cat_col_found = next((c for c in [CATEGORIA_STD, CATEGORIA_PREDICHA, 'CATEGORIA', 'CATEGORÍA', 'CATEGORIA_X'] if c in df.columns), None)
    if not cat_col_found: st.error(f"BD Acumulada sin col. categoría."); return None
    if cat_col_found != CATEGORIA_STD: df = df.rename(columns={cat_col_found: CATEGORIA_STD})
    required = [CONCEPTO_STD, IMPORTE_STD, AÑO_STD, MES_STD, DIA_STD, CATEGORIA_STD]
    if COMERCIO_STD not in df.columns: df[COMERCIO_STD] = ''
    missing = [col for col in required if col not in df.columns]
    if missing: st.error(f"BD Incompleta. Faltan: {missing}"); return None
    df_train = df.copy()
    try:
        for col in [CONCEPTO_STD, COMERCIO_STD, CATEGORIA_STD]: df_train[col] = df_train[col].fillna('').astype(str).str.lower().str.strip()
        df_train[IMPORTE_STD] = pd.to_numeric(df_train[IMPORTE_STD], errors='coerce')
        for col_f in [AÑO_STD, MES_STD, DIA_STD]: df_train[col_f] = pd.to_numeric(df_train[col_f], errors='coerce').fillna(0).astype(int)
    except Exception as e_clean: st.error(f"Error limpiando BD: {e_clean}"); return None
    if TEXTO_MODELO not in df_train.columns: df_train[TEXTO_MODELO] = (df_train[CONCEPTO_STD] + ' ' + df_train[COMERCIO_STD]).str.strip()
    if SUBCATEGORIA_STD not in df_train.columns: df_train[SUBCATEGORIA_STD] = ''
    else: df_train[SUBCATEGORIA_STD] = df_train[SUBCATEGORIA_STD].fillna('').astype(str).str.lower().str.strip()
    df_train = df_train.dropna(subset=[IMPORTE_STD, CATEGORIA_STD, TEXTO_MODELO])
    df_train = df_train[df_train[CATEGORIA_STD] != ''][df_train[TEXTO_MODELO] != '']
    if df_train.empty: st.warning("BD Acumulada sin filas válidas para entrenar."); return None
    return df_train

def get_last_transaction_dates(df_accumulated):
    # ... (código sin cambios) ...
    if df_accumulated is None or df_accumulated.empty: return pd.DataFrame(columns=['Cuenta', 'Última Transacción'])
    cuenta_col = next((c for c in [CUENTA_COL_STD, CUENTA_COL_ORIG.upper(), 'CUENTA'] if c in df_accumulated.columns), None)
    if not cuenta_col: return pd.DataFrame(columns=['Cuenta', 'Última Transacción'])
    df_temp = df_accumulated.copy(); date_cols = [AÑO_STD, MES_STD, DIA_STD]
    if not all(col in df_temp.columns for col in date_cols): return pd.DataFrame(columns=['Cuenta', 'Última Transacción'])
    for col in date_cols: df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce').fillna(0)
    try: df_temp['FECHA_COMPLETA'] = pd.to_datetime(df_temp[date_cols].rename(columns={AÑO_STD: 'year', MES_STD: 'month', DIA_STD: 'day'}), errors='coerce')
    except ValueError as e_dt_conv: st.error(f"Error construyendo fechas: {e_dt_conv}"); return pd.DataFrame(columns=['Cuenta', 'Última Transacción'])
    df_temp = df_temp.dropna(subset=[cuenta_col, 'FECHA_COMPLETA'])
    if df_temp.empty: return pd.DataFrame(columns=['Cuenta', 'Última Transacción'])
    last_dates = df_temp.groupby(cuenta_col)['FECHA_COMPLETA'].max()
    if last_dates.empty: return pd.DataFrame(columns=['Cuenta', 'Última Transacción'])
    summary_df = last_dates.reset_index(); summary_df.columns = ['Cuenta', 'Última Transacción']
    summary_df['Última Transacción'] = summary_df['Última Transacción'].dt.strftime('%d/%m/%Y')
    summary_df['Cuenta'] = summary_df['Cuenta'].astype(str).str.capitalize()
    return summary_df.sort_values(by='Cuenta')
# ------------------------------------------------------------------------------------

# --- Streamlit UI ---
st.set_page_config(layout="wide"); st.title("🏦 Categorizador Bancario Inteligente v5.4")
st.caption(f"Config: `{CONFIG_FILENAME}`, BD: `{DB_FILENAME}`")

# --- Carga BD (Sidebar) ---
st.sidebar.header("Base de Datos Global")
uploaded_db_file = st.sidebar.file_uploader(f"Cargar BD ({DB_FILENAME})", type=["csv", "xlsx", "xls"], key="db_loader", help="Carga BD acumulada.")
if uploaded_db_file:
    db_uploader_key = "db_loader_processed_id"
    if uploaded_db_file.file_id != st.session_state.get(db_uploader_key, None):
        st.sidebar.info("Cargando BD...")
        df_db_loaded, _ = read_uploaded_file(uploaded_db_file)
        if df_db_loaded is not None:
            df_db_loaded.columns = [str(col).upper().strip() for col in df_db_loaded.columns]
            cat_col_loaded = next((c for c in [CATEGORIA_STD, CATEGORIA_PREDICHA, 'CATEGORIA', 'CATEGORÍA', 'CATEGORIA_X'] if c in df_db_loaded.columns), None)
            if cat_col_loaded and cat_col_loaded != CATEGORIA_STD: df_db_loaded = df_db_loaded.rename(columns={cat_col_loaded: CATEGORIA_STD})
            elif not cat_col_loaded: st.sidebar.error("BD sin col. categoría."); df_db_loaded = None
            if df_db_loaded is not None:
                expected_db_cols = [CONCEPTO_STD, IMPORTE_STD, AÑO_STD, MES_STD, DIA_STD, CATEGORIA_STD]
                missing_db_cols = [col for col in expected_db_cols if col not in df_db_loaded.columns]
                if not missing_db_cols:
                    if SUBCATEGORIA_STD not in df_db_loaded.columns: df_db_loaded[SUBCATEGORIA_STD] = ''
                    if COMERCIO_STD not in df_db_loaded.columns: df_db_loaded[COMERCIO_STD] = ''
                    cuenta_col_db = next((c for c in [CUENTA_COL_STD, CUENTA_COL_ORIG.upper(), 'CUENTA'] if c in df_db_loaded.columns), None)
                    if cuenta_col_db and cuenta_col_db != CUENTA_COL_STD: df_db_loaded = df_db_loaded.rename(columns={cuenta_col_db: CUENTA_COL_STD})
                    elif not cuenta_col_db: df_db_loaded[CUENTA_COL_STD] = ''
                    cols_fill = [CONCEPTO_STD, COMERCIO_STD, CATEGORIA_STD, SUBCATEGORIA_STD, CUENTA_COL_STD]
                    for col in cols_fill:
                         if col in df_db_loaded.columns: df_db_loaded[col] = df_db_loaded[col].fillna('')
                    st.session_state.accumulated_data = df_db_loaded; st.session_state[db_uploader_key] = uploaded_db_file.file_id
                    st.sidebar.success(f"BD cargada ({len(df_db_loaded)}).")
                    # **** CORRECCIÓN: Extraer conocimiento SIEMPRE después de cargar BD ****
                    if CATEGORIA_STD in st.session_state.accumulated_data.columns:
                         st.session_state.learned_knowledge = extract_knowledge_std(st.session_state.accumulated_data)
                         st.session_state.knowledge_loaded = bool(st.session_state.learned_knowledge.get('categorias'))
                         if st.session_state.knowledge_loaded: st.sidebar.info("Conocimiento extraído/actualizado de BD.")
                         else: st.sidebar.warning("No se pudo extraer conocimiento de BD.")
                    else: st.sidebar.warning("BD cargada sin CATEGORIA_STD para extraer conocimiento.")
                    st.rerun()
                else: st.sidebar.error(f"DB inválida. Faltan: {missing_db_cols}"); st.session_state[db_uploader_key] = None
        else: st.sidebar.error("No se pudo leer BD."); st.session_state[db_uploader_key] = None

# --- Tabs Principales ---
tab1, tab2 = st.tabs(["⚙️ Configuración y Entrenamiento", "📊 Categorización y Gestión BD"])

# --- Tab 1: Configuración y Entrenamiento ---
with tab1:
    st.header("Configuración y Entrenamiento")
    col1a, col1b = st.columns([1, 2])
    with col1a:
        st.subheader("Cargar/Descargar Configuración")
        st.write(f"Gestiona `{CONFIG_FILENAME}`."); uploaded_config_file_f1 = st.file_uploader(f"Cargar Config", type="json", key="config_loader_f1")
        if uploaded_config_file_f1:
            k = "config_loader_processed_id_f1"
            if uploaded_config_file_f1.file_id != st.session_state.get(k, None):
                try:
                    cfg = json.load(uploaded_config_file_f1); ok = True; msg = ""
                    if not isinstance(cfg,dict): ok=False; msg="No dict."
                    elif 'bank_mappings' not in cfg or not isinstance(cfg['bank_mappings'], dict): ok=False; msg+=" Falta/Inv 'bank_mappings'."
                    elif 'learned_knowledge' not in cfg or not isinstance(cfg['learned_knowledge'], dict): ok=False; msg+=" Falta/Inv 'learned_knowledge'."
                    elif not all(key in cfg['learned_knowledge'] for key in ['categorias', 'subcategorias_por_cat', 'comercios_por_cat', 'subcat_unica_por_comercio_y_cat', 'subcat_mas_frecuente_por_comercio_y_cat', 'all_subcategories', 'all_comercios']): ok=False; msg+=" Faltan claves knowledge."
                    if ok:
                        st.session_state.bank_mappings = cfg['bank_mappings']; st.session_state.learned_knowledge = cfg['learned_knowledge']
                        st.session_state.knowledge_loaded = bool(st.session_state.learned_knowledge.get('categorias'))
                        st.success(f"Config cargada."); st.sidebar.success("Config. Cargada"); st.session_state[k] = uploaded_config_file_f1.file_id
                        if not st.session_state.model_trained: st.info("Conocimiento OK. Entrena si vas a categorizar.")
                        st.rerun()
                    else: st.error(f"Error formato config: {msg.strip()}"); st.session_state[k] = None
                except Exception as e: st.error(f"Error carga config: {e}"); st.error(traceback.format_exc()); st.session_state[k] = None
        if st.session_state.bank_mappings or st.session_state.learned_knowledge.get('categorias'):
            try:
                cfg_save = {'bank_mappings': st.session_state.get('bank_mappings', {}), 'learned_knowledge': st.session_state.get('learned_knowledge', {})}
                cfg_json = json.dumps(cfg_save, indent=4, ensure_ascii=False)
                st.download_button(label=f"💾 Descargar Config Actual", data=cfg_json.encode('utf-8'), file_name=CONFIG_FILENAME, mime='application/json', key='download_config_f1')
            except Exception as e: st.error(f"Error descarga config: {e}")
        else: st.info("No hay config en memoria.")

        st.divider()
        st.subheader("(Re)Entrenar Modelo")
        st.write("Usa la BD Acumulada (cargada en sidebar).")
        if st.session_state.accumulated_data.empty: st.warning("Carga la BD.")
        elif st.button("🧠 Entrenar/Reentrenar con BD", key="train_db_f1b"):
             with st.spinner("Preparando y entrenando..."):
                df_to_train = parse_accumulated_db_for_training(st.session_state.accumulated_data.copy())
                if df_to_train is not None and not df_to_train.empty:
                    st.success("Datos BD OK.")
                    # **** CORRECCIÓN: Actualizar conocimiento y marcar cargado ****
                    st.session_state.learned_knowledge = extract_knowledge_std(df_to_train)
                    st.session_state.knowledge_loaded = True
                    st.sidebar.success("Conocimiento Actualizado (BD)")

                    model, vectorizer, report = train_classifier_std(df_to_train)
                    if model and vectorizer:
                        st.session_state.model = model; st.session_state.vectorizer = vectorizer
                        st.session_state.model_trained = True; st.session_state.training_report = report
                        st.success(f"¡Modelo entrenado!")
                        st.sidebar.subheader("Evaluación Modelo"); with st.sidebar.expander("Ver Informe"): st.text(st.session_state.training_report)
                    else: st.error(f"Fallo entrenamiento."); st.session_state.model_trained = False; st.session_state.training_report = report; st.sidebar.error("Entrenamiento Fallido"); st.sidebar.text(st.session_state.training_report)
                else: st.error("No se prepararon datos de BD."); st.session_state.model_trained = False

        # **** CORRECCIÓN: Mostrar Conocimiento Base Actual ****
        st.divider()
        st.subheader("Conocimiento Base Actual")
        knowledge_display = st.session_state.get('learned_knowledge', {})
        if knowledge_display and knowledge_display.get('categorias'):
             st.write(f"**Categorías ({len(knowledge_display['categorias'])}):**")
             st.dataframe(pd.DataFrame(knowledge_display['categorias'], columns=['Categoría']), use_container_width=True, hide_index=True)
             all_subs = knowledge_display.get('all_subcategories', [])
             all_coms = knowledge_display.get('all_comercios', [])
             col_k1, col_k2 = st.columns(2)
             with col_k1:
                  with st.expander(f"Subcategorías Conocidas ({len(all_subs)})"):
                       st.dataframe(pd.DataFrame(all_subs, columns=['Subcategoría']), use_container_width=True, hide_index=True)
             with col_k2:
                   with st.expander(f"Comercios Conocidos ({len(all_coms)})"):
                       st.dataframe(pd.DataFrame(all_coms, columns=['Comercio']), use_container_width=True, hide_index=True)
             # Detalles opcionales
             # with st.expander("Subcategorías por Categoría (Detalle)"): st.json(knowledge_display.get('subcategorias_por_cat', {}))
             # with st.expander("Comercios por Categoría (Detalle)"): st.json(knowledge_display.get('comercios_por_cat', {}))
        else:
             st.info("No hay conocimiento base cargado o aprendido.")


    with col1b:
        st.subheader("Definir Formatos Bancarios (Mapeo)")
        # ... (UI Mapeo sin cambios) ...
        st.write("Enseña cómo leer archivos de bancos subiendo un ejemplo.")
        bank_options = ["SANTANDER", "EVO", "WIZINK", "AMEX"]
        selected_bank_learn = st.selectbox("Banco:", bank_options, key="bank_learn_f2_select")
        uploaded_sample_file = st.file_uploader(f"Cargar ejemplo {selected_bank_learn}", type=["csv", "xlsx", "xls"], key="sample_uploader_f2")
        if uploaded_sample_file:
            df_sample, detected_columns = read_uploaded_file(uploaded_sample_file)
            if df_sample is not None:
                st.write(f"Columnas:"); st.code(f"{detected_columns}"); st.dataframe(df_sample.head(3))
                st.write("**Mapeo:**"); saved_mapping = st.session_state.bank_mappings.get(selected_bank_learn, {'columns': {}})
                cols_with_none = [None] + detected_columns
                sub_c1, sub_c2 = st.columns(2)
                with sub_c1:
                    st.markdown("**Esenciales:**"); map_concepto = st.selectbox(f"`{CONCEPTO_STD}`", cols_with_none, index=cols_with_none.index(saved_mapping['columns'].get(CONCEPTO_STD)) if saved_mapping['columns'].get(CONCEPTO_STD) in cols_with_none else 0, key=f"map_{CONCEPTO_STD}_{selected_bank_learn}")
                    map_importe = st.selectbox(f"`{IMPORTE_STD}`", cols_with_none, index=cols_with_none.index(saved_mapping['columns'].get(IMPORTE_STD)) if saved_mapping['columns'].get(IMPORTE_STD) in cols_with_none else 0, key=f"map_{IMPORTE_STD}_{selected_bank_learn}")
                    st.markdown("**Fecha:**"); is_single_date_saved = FECHA_STD in saved_mapping['columns']
                    map_single_date = st.checkbox("Fecha en 1 col", value=is_single_date_saved, key=f"map_single_date_{selected_bank_learn}")
                    map_fecha_unica=None; map_formato_fecha=None; map_año=None; map_mes=None; map_dia=None
                    if map_single_date:
                        map_fecha_unica = st.selectbox(f"`{FECHA_STD}`", cols_with_none, index=cols_with_none.index(saved_mapping['columns'].get(FECHA_STD)) if saved_mapping['columns'].get(FECHA_STD) in cols_with_none else 0, key=f"map_{FECHA_STD}_{selected_bank_learn}")
                        map_formato_fecha = st.text_input("Formato", value=saved_mapping.get('date_format', ''), key=f"map_date_format_{selected_bank_learn}")
                    else:
                        map_año = st.selectbox(f"`{AÑO_STD}`", cols_with_none, index=cols_with_none.index(saved_mapping['columns'].get(AÑO_STD)) if saved_mapping['columns'].get(AÑO_STD) in cols_with_none else 0, key=f"map_{AÑO_STD}_{selected_bank_learn}")
                        map_mes = st.selectbox(f"`{MES_STD}`", cols_with_none, index=cols_with_none.index(saved_mapping['columns'].get(MES_STD)) if saved_mapping['columns'].get(MES_STD) in cols_with_none else 0, key=f"map_{MES_STD}_{selected_bank_learn}")
                        map_dia = st.selectbox(f"`{DIA_STD}`", cols_with_none, index=cols_with_none.index(saved_mapping['columns'].get(DIA_STD)) if saved_mapping['columns'].get(DIA_STD) in cols_with_none else 0, key=f"map_{DIA_STD}_{selected_bank_learn}")
                with sub_c2:
                    st.markdown("**Opcionales:**"); map_comercio = st.selectbox(f"`{COMERCIO_STD}`", cols_with_none, index=cols_with_none.index(saved_mapping['columns'].get(COMERCIO_STD)) if saved_mapping['columns'].get(COMERCIO_STD) in cols_with_none else 0, key=f"map_{COMERCIO_STD}_{selected_bank_learn}")
                    st.markdown("**Importe:**"); val_map_decimal_sep = st.text_input("Separador Decimal", value=saved_mapping.get('decimal_sep', ','), key=f"map_decimal_{selected_bank_learn}")
                    val_map_thousands_sep = st.text_input("Separador Miles", value=saved_mapping.get('thousands_sep', ''), key=f"map_thousands_{selected_bank_learn}")

                if st.button(f"💾 Guardar Mapeo {selected_bank_learn}", key="save_mapping_f2"):
                    final_mapping_cols = {}
                    if map_concepto: final_mapping_cols[CONCEPTO_STD] = map_concepto
                    if map_importe: final_mapping_cols[IMPORTE_STD] = map_importe
                    if map_single_date and map_fecha_unica: final_mapping_cols[FECHA_STD] = map_fecha_unica
                    if not map_single_date and map_año: final_mapping_cols[AÑO_STD] = map_año
                    if not map_single_date and map_mes: final_mapping_cols[MES_STD] = map_mes
                    if not map_single_date and map_dia: final_mapping_cols[DIA_STD] = map_dia
                    if map_comercio: final_mapping_cols[COMERCIO_STD] = map_comercio
                    # Validación Corregida
                    valid = True; current_fmt = map_formato_fecha
                    if not final_mapping_cols.get(CONCEPTO_STD): st.error("Mapea CONCEPTO."); valid=False
                    if not final_mapping_cols.get(IMPORTE_STD): st.error("Mapea IMPORTE."); valid=False
                    if map_single_date:
                        if not final_mapping_cols.get(FECHA_STD): st.error("Mapea FECHA."); valid=False
                        elif not current_fmt: st.error("Especifica formato."); valid=False
                    else:
                        if not final_mapping_cols.get(AÑO_STD): st.error("Mapea AÑO."); valid=False
                        if not final_mapping_cols.get(MES_STD): st.error("Mapea MES."); valid=False
                        if not final_mapping_cols.get(DIA_STD): st.error("Mapea DIA."); valid=False
                    # Fin Validación
                    if valid:
                        current_decimal_sep = val_map_decimal_sep; current_thousands_sep = val_map_thousands_sep
                        mapping_to_save = {'bank_name': selected_bank_learn, 'columns': final_mapping_cols, 'decimal_sep': current_decimal_sep.strip(), 'thousands_sep': current_thousands_sep.strip() or None}
                        if map_single_date and current_fmt: mapping_to_save['date_format'] = current_fmt.strip()
                        st.session_state.bank_mappings[selected_bank_learn] = mapping_to_save
                        st.success(f"¡Mapeo {selected_bank_learn} guardado!"); st.rerun()
                    else: st.warning("Revisa errores.")


# --- Tab 2: Categorización y Gestión BD ---
with tab2:
    st.header("Categorización y Gestión BD")
    st.subheader("Resumen: Última Transacción por Cuenta")
    df_summary = get_last_transaction_dates(st.session_state.accumulated_data)
    if not df_summary.empty: st.dataframe(df_summary, use_container_width=True, hide_index=True)
    else: st.info("No hay datos en BD o falta col. cuenta.")
    st.divider()
    st.subheader("Categorizar Nuevos Archivos y Añadir a BD")
    model_ready_for_pred = st.session_state.get('model_trained', False)
    mappings_available = bool(st.session_state.get('bank_mappings', {}))
    knowledge_ready = st.session_state.get('knowledge_loaded', False)
    if not knowledge_ready: st.warning("⚠️ Conocimiento no cargado (Pestaña Config).")
    elif not mappings_available: st.warning("⚠️ Formatos bancarios no definidos (Pestaña Config).")
    elif not model_ready_for_pred: st.warning("⚠️ Modelo no entrenado (Pestaña Config).")
    else:
        st.write("Selecciona banco y sube archivo **sin categorizar**."); available_banks_for_pred = list(st.session_state.bank_mappings.keys())
        selected_bank_predict = st.selectbox("Banco:", available_banks_for_pred, key="bank_predict_f3")
        uploaded_final_file = st.file_uploader(f"Cargar archivo {selected_bank_predict}", type=["csv", "xlsx", "xls"], key="final_uploader_f3")
        if uploaded_final_file and selected_bank_predict:
            mapping_to_use = st.session_state.bank_mappings.get(selected_bank_predict)
            if not mapping_to_use: st.error(f"Error: No mapeo para {selected_bank_predict}.")
            else:
                 st.write(f"Procesando '{uploaded_final_file.name}'..."); df_std_new = None; df_pred = None
                 with st.spinner(f"Estandarizando..."):
                      df_raw_new, _ = read_uploaded_file(uploaded_final_file)
                      if df_raw_new is not None: df_std_new = standardize_data_with_mapping(df_raw_new.copy(), mapping_to_use)
                      else: st.error(f"No se pudo leer: {uploaded_final_file.name}")
                 if df_std_new is not None and not df_std_new.empty:
                      st.success("Datos estandarizados.")
                      with st.spinner("Categorizando..."):
                          st.session_state.debug_predictions = []
                          try:
                               if TEXTO_MODELO not in df_std_new.columns: st.error(f"Error: Falta {TEXTO_MODELO}.")
                               else:
                                    df_pred = df_std_new.dropna(subset=[TEXTO_MODELO]).copy()
                                    if not df_pred.empty:
                                         X_new_vec = st.session_state.vectorizer.transform(df_pred[TEXTO_MODELO])
                                         predictions_cat = st.session_state.model.predict(X_new_vec)
                                         df_pred[CATEGORIA_PREDICHA] = [str(p).capitalize() for p in predictions_cat]
                                         pred_comercios_final = []; pred_subcats_final = []
                                         knowledge = st.session_state.learned_knowledge; debug_info_list = []
                                         for index, row in df_pred.iterrows():
                                             pred_cat_lower = row[CATEGORIA_PREDICHA].lower(); input_comercio_lower = row.get(COMERCIO_STD, ''); input_concepto_lower = row.get(CONCEPTO_STD, '')
                                             debug_step = f"F:{index}|Cat:{pred_cat_lower}|Com:'{input_comercio_lower}'"
                                             comercio_final = input_comercio_lower; best_match_comercio = None
                                             known_comers_for_cat = knowledge['comercios_por_cat'].get(pred_cat_lower, [])
                                             if input_comercio_lower and known_comers_for_cat:
                                                 match_result = process.extractOne(input_comercio_lower, known_comers_for_cat)
                                                 if match_result and match_result[1] >= FUZZY_MATCH_THRESHOLD: comercio_final = match_result[0]; best_match_comercio = match_result[0]; debug_step += f"->ComM:'{comercio_final}'({match_result[1]})"
                                             pred_comercios_final.append(comercio_final.capitalize())
                                             subcat_final = ''; comercio_lookup_key = best_match_comercio if best_match_comercio else input_comercio_lower
                                             subcat_msg = "->Sub:"
                                             if comercio_lookup_key:
                                                subcat_unica = knowledge['subcat_unica_por_comercio_y_cat'].get(pred_cat_lower, {}).get(comercio_lookup_key)
                                                if subcat_unica: subcat_final = subcat_unica; subcat_msg += "U(C)"
                                             if not subcat_final and comercio_lookup_key:
                                                 subcat_frecuente = knowledge['subcat_mas_frecuente_por_comercio_y_cat'].get(pred_cat_lower, {}).get(comercio_lookup_key)
                                                 if subcat_frecuente: subcat_final = subcat_frecuente; subcat_msg += "F(C)"
                                             if not subcat_final and input_concepto_lower:
                                                  known_subcats_for_cat = knowledge['subcategorias_por_cat'].get(pred_cat_lower, [])
                                                  found_kw = [sk for sk in known_subcats_for_cat if sk and re.search(r'\b' + re.escape(sk) + r'\b', input_concepto_lower, re.IGNORECASE)]
                                                  if len(found_kw) == 1: subcat_final = found_kw[0]; subcat_msg += f"KW({found_kw[0]})"
                                             if not subcat_final:
                                                 known_subcats_for_cat = knowledge['subcategorias_por_cat'].get(pred_cat_lower, [])
                                                 if len(known_subcats_for_cat) == 1: subcat_final = known_subcats_for_cat[0]; subcat_msg += "U(K)"
                                             if not subcat_final: subcat_msg += "N/A"
                                             pred_subcats_final.append(subcat_final.capitalize()); debug_info_list.append(debug_step + subcat_msg)
                                         df_pred[COMERCIO_PREDICHO] = pred_comercios_final; df_pred[SUBCATEGORIA_PREDICHA] = pred_subcats_final
                                         st.session_state.debug_predictions = debug_info_list
                                         st.subheader(f"📊 Resultados para '{uploaded_final_file.name}'")
                                         display_cols_order = [CATEGORIA_PREDICHA, SUBCATEGORIA_PREDICHA, COMERCIO_PREDICHO] + \
                                                              [c for c in DB_FINAL_COLS if c not in [CATEGORIA_PREDICHA, SUBCATEGORIA_PREDICHA, COMERCIO_PREDICHO]] + \
                                                              [c for c in df_pred.columns if c.startswith('ORIG_')]
                                         cuenta_col_pred = next((c for c in [CUENTA_COL_STD, CUENTA_COL_ORIG.upper()] if c in df_pred.columns), None)
                                         if cuenta_col_pred and cuenta_col_pred not in display_cols_order: display_cols_order.append(cuenta_col_pred)
                                         final_display_cols = [col for col in display_cols_order if col in df_pred.columns]
                                         st.dataframe(df_pred[final_display_cols])

                                         # --- Botón Añadir a BD y Lógica Duplicados ---
                                         if st.button(f"➕ Añadir a BD", key=f"add_db_{uploaded_final_file.name}"):
                                             current_db = st.session_state.get('accumulated_data', pd.DataFrame())
                                             db_cols_to_keep = DB_FINAL_COLS + [c for c in df_pred.columns if c.startswith('ORIG_')]
                                             cuenta_col_append = next((c for c in [CUENTA_COL_STD, CUENTA_COL_ORIG.upper()] if c in df_pred.columns), None)
                                             if cuenta_col_append and cuenta_col_append not in db_cols_to_keep: db_cols_to_keep.append(cuenta_col_append)
                                             final_db_cols_append = [col for col in db_cols_to_keep if col in df_pred.columns]
                                             df_to_append = df_pred[final_db_cols_append].copy()
                                             if current_db.empty: new_transactions_only = df_to_append; num_added = len(new_transactions_only)
                                             else:
                                                 st.write("Detectando duplicados..."); key_cols = [AÑO_STD, MES_STD, DIA_STD, IMPORTE_STD, CONCEPTO_STD]
                                                 acc_col_db = next((c for c in [CUENTA_COL_STD, CUENTA_COL_ORIG.upper(), 'CUENTA'] if c in current_db.columns), None)
                                                 acc_col_new = next((c for c in [CUENTA_COL_STD, CUENTA_COL_ORIG.upper(), 'CUENTA'] if c in df_to_append.columns), None)
                                                 account_key_col = None
                                                 if acc_col_db and acc_col_new: key_cols.append(acc_col_db); account_key_col = acc_col_db
                                                 df1 = current_db.copy(); df2 = df_to_append.copy()
                                                 if acc_col_db and acc_col_new and acc_col_db != acc_col_new: df2 = df2.rename(columns={acc_col_new: acc_col_db})
                                                 for df_temp in [df1, df2]:
                                                     if IMPORTE_STD in df_temp.columns: df_temp[IMPORTE_STD] = pd.to_numeric(df_temp[IMPORTE_STD], errors='coerce').round(2)
                                                     for col in key_cols:
                                                          if col not in df_temp.columns: df_temp[col] = np.nan
                                                          if col in [AÑO_STD, MES_STD, DIA_STD]: df_temp[col] = pd.to_numeric(df_temp[col], errors='coerce').astype('Int64')
                                                          elif col in [CONCEPTO_STD] or (account_key_col and col==account_key_col) : df_temp[col] = df_temp[col].fillna('').astype(str)
                                                 df1 = df1.dropna(subset=key_cols); df2 = df2.dropna(subset=key_cols)
                                                 if df1.empty and not df2.empty: new_transactions_only = df2
                                                 elif not df1.empty and not df2.empty:
                                                      merged = df2.merge(df1[key_cols].drop_duplicates(), on=key_cols, how='left', indicator=True)
                                                      new_transactions_only = merged[merged['_merge'] == 'left_only'].drop(columns=['_merge'])
                                                 else: new_transactions_only = pd.DataFrame(columns=df2.columns)
                                                 num_added = len(new_transactions_only)
                                             if num_added > 0:
                                                 st.write(f"Añadiendo {num_added} nuevas.")
                                                 combined_cols = current_db.columns.union(new_transactions_only.columns)
                                                 current_db_reindexed = current_db.reindex(columns=combined_cols); new_transactions_reindexed = new_transactions_only.reindex(columns=combined_cols)
                                                 st.session_state.accumulated_data = pd.concat([current_db_reindexed, new_transactions_reindexed], ignore_index=True).fillna('')
                                                 st.success(f"{num_added} trans. NUEVAS añadidas."); st.rerun()
                                             else: st.info("No se añadieron filas (0 nuevas o duplicadas)."); st.rerun()
                                    else: st.warning("No quedaron filas válidas para categorizar.")
                          except Exception as e_pred: st.error(f"Error predicción: {e_pred}"); st.error(traceback.format_exc())
                      if st.session_state.debug_predictions:
                           with st.expander("Detalles Predicción (Debug)"):
                               st.text("\n".join(st.session_state.debug_predictions[:100]))
                               if len(st.session_state.debug_predictions) > 100: st.text("...")
                 elif df_std_new is not None and df_std_new.empty: st.warning("Archivo vacío o sin datos válidos.")
                 else: st.error("Fallo en estandarización.")

    st.divider()
    # --- Sub-Sección: Ver/Gestionar Base de Datos Acumulada ---
    st.subheader("Base de Datos Acumulada (Editable)")
    db_state_tab = st.session_state.get('accumulated_data', pd.DataFrame())
    if db_state_tab is not None and not db_state_tab.empty:
        st.write(f"({len(db_state_tab)} filas). Doble clic para editar Cat/Subcat/Comercio.")
        knowledge = st.session_state.get('learned_knowledge', {})
        categorias_options = knowledge.get('categorias', []) # Categorías principales
        all_subcats_options = knowledge.get('all_subcategories', []) # Lista plana de todas las subcats
        all_comers_options = knowledge.get('all_comercios', []) # Lista plana de todos los comercios

        # **** Asegurar que las columnas editables existen ANTES de configurar ****
        for col in [CATEGORIA_PREDICHA, SUBCATEGORIA_PREDICHA, COMERCIO_PREDICHO]:
            if col not in db_state_tab.columns: db_state_tab[col] = ''

        column_config_editor = {
            CATEGORIA_PREDICHA: st.column_config.SelectboxColumn("Categoría", width="medium", options=categorias_options, required=False),
            SUBCATEGORIA_PREDICHA: st.column_config.SelectboxColumn("Subcategoría", width="medium", options=all_subcats_options, required=False),
            COMERCIO_PREDICHO: st.column_config.SelectboxColumn("Comercio Estd.", width="medium", options=all_comers_options, required=False)
        }
        cuenta_col_display = next((c for c in [CUENTA_COL_STD, CUENTA_COL_ORIG.upper(), 'CUENTA'] if c in db_state_tab.columns), None)
        cols_to_edit_base = [CATEGORIA_PREDICHA, SUBCATEGORIA_PREDICHA, COMERCIO_PREDICHO, CONCEPTO_STD, IMPORTE_STD, AÑO_STD, MES_STD, DIA_STD]
        cols_to_show_edit = [col for col in cols_to_edit_base if col in db_state_tab.columns]
        if cuenta_col_display and cuenta_col_display not in cols_to_show_edit: cols_to_show_edit.append(cuenta_col_display)
        cols_to_show_edit += sorted([col for col in db_state_tab.columns if col.startswith('ORIG_') and col != cuenta_col_display])
        cols_to_show_edit += sorted([col for col in db_state_tab.columns if col not in cols_to_show_edit and col != TEXTO_MODELO])
        cols_to_show_edit = [col for col in cols_to_show_edit if col in db_state_tab.columns]

        # **** Usar st.data_editor ****
        edited_df = st.data_editor(
            db_state_tab[cols_to_show_edit].copy(), # Pasar copia para editar
            key="db_editor_main", column_config=column_config_editor,
            num_rows="dynamic", use_container_width=True,
        )

        # Botón para guardar cambios del editor
        if st.button("💾 Confirmar Cambios en BD", key="save_edited_db"):
            # **** CORRECCIÓN: Actualizar BD y re-extraer conocimiento ****
            st.session_state.accumulated_data = edited_df.copy()
            # Re-extraer conocimiento DESPUÉS de editar manualmente, ya que las categorías/subs pueden haber cambiado
            if CATEGORIA_STD in st.session_state.accumulated_data.columns: # Asegurar que existe la columna base
                 st.session_state.learned_knowledge = extract_knowledge_std(st.session_state.accumulated_data)
                 st.session_state.knowledge_loaded = True
                 st.sidebar.success("Conocimiento re-extraído tras edición.")
            else:
                 st.sidebar.warning("No se pudo re-extraer conocimiento (falta CATEGORIA_STD en BD editada).")

            st.success("Cambios guardados en memoria. ¡Descarga la BD actualizada (Sidebar)!")
            st.rerun()
    else: st.info("BD acumulada vacía.")


# --- Sidebar Info y Estado ---
st.sidebar.divider(); st.sidebar.header("Acerca de")
st.sidebar.info("Categorizador v5.3"); st.sidebar.divider(); st.sidebar.subheader("Estado Actual")
model_ready_sidebar = st.session_state.get('model_trained', False); knowledge_ready_sidebar = st.session_state.get('knowledge_loaded', False)
if model_ready_sidebar: st.sidebar.success("✅ Modelo Entrenado")
elif knowledge_ready_sidebar: st.sidebar.info("ℹ️ Conocimiento Cargado")
else: st.sidebar.warning("❌ Sin Modelo/Conocimiento")
if st.session_state.get('bank_mappings', {}): st.sidebar.success(f"✅ Mapeos ({len(st.session_state.bank_mappings)})")
else: st.sidebar.warning("❌ Sin Mapeos Bancarios")
db_state_sidebar = st.session_state.get('accumulated_data', pd.DataFrame())
if db_state_sidebar is not None and not db_state_sidebar.empty: st.sidebar.success(f"✅ BD ({len(db_state_sidebar)} filas)")
else: st.sidebar.info("ℹ️ BD Vacía")

# --- Descarga BD (Sidebar) ---
st.sidebar.divider(); st.sidebar.subheader("Guardar Base de Datos")
if db_state_sidebar is not None and not db_state_sidebar.empty:
    try:
        cuenta_col_export = next((c for c in [CUENTA_COL_STD, CUENTA_COL_ORIG.upper(), 'CUENTA'] if c in db_state_sidebar.columns), None)
        cols_to_export_db = [col for col in DB_FINAL_COLS if col in db_state_sidebar.columns]
        if cuenta_col_export and cuenta_col_export not in cols_to_export_db: cols_to_export_db.append(cuenta_col_export)
        cols_to_export_db += [col for col in db_state_sidebar.columns if col.startswith('ORIG_') and col != cuenta_col_export]
        cols_to_export_db = [col for col in cols_to_export_db if col in db_state_sidebar.columns]
        df_to_export = db_state_sidebar[cols_to_export_db].copy()
        db_csv_output_sb = df_to_export.to_csv(index=False, sep=';', decimal=',').encode('utf-8')
        st.sidebar.download_button(label=f"💾 Descargar BD (CSV)", data=db_csv_output_sb, file_name=DB_FILENAME, mime='text/csv', key='download_db_csv_sb')
        output_excel_sb = io.BytesIO()
        with pd.ExcelWriter(output_excel_sb, engine='openpyxl') as writer: df_to_export.to_excel(writer, index=False, sheet_name='Gastos')
        excel_data_sb = output_excel_sb.getvalue(); db_excel_filename_sb = DB_FILENAME.replace('.csv', '.xlsx')
        st.sidebar.download_button(label=f"💾 Descargar BD (Excel)", data=excel_data_sb, file_name=db_excel_filename_sb, mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', key='download_db_excel_sb')
    except Exception as e_db_down: st.sidebar.error(f"Error descarga BD: {e_db_down}")
else: st.sidebar.info("BD vacía.")
