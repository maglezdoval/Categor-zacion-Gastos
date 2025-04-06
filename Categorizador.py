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
from collections import Counter # Para encontrar la subcategoría más común por comercio

# --- Constantes ---
CONCEPTO_STD = 'CONCEPTO_STD'; COMERCIO_STD = 'COMERCIO_STD'; IMPORTE_STD = 'IMPORTE_STD'
AÑO_STD = 'AÑO'; MES_STD = 'MES'; DIA_STD = 'DIA'; FECHA_STD = 'FECHA_STD'
CATEGORIA_STD = 'CATEGORIA_STD'; SUBCATEGORIA_STD = 'SUBCATEGORIA_STD'
TEXTO_MODELO = 'TEXTO_MODELO'; CATEGORIA_PREDICHA = 'CATEGORIA_PREDICHA'
# **** NUEVAS COLUMNAS PREDICHAS ****
SUBCATEGORIA_PREDICHA = 'SUBCATEGORIA_PREDICHA'
COMERCIO_PREDICHO = 'COMERCIO_PREDICHO' # Versión estandarizada/predicha

# Columnas estándar ESPERADAS en la base de datos acumulada (para entrenamiento)
DB_TRAIN_COLS = [CONCEPTO_STD, COMERCIO_STD, IMPORTE_STD, AÑO_STD, MES_STD, DIA_STD, CATEGORIA_STD, SUBCATEGORIA_STD]
# **** ACTUALIZADO: Columnas a MOSTRAR/GUARDAR en la base de datos final ****
DB_FINAL_COLS = [CATEGORIA_PREDICHA, SUBCATEGORIA_PREDICHA, COMERCIO_PREDICHO, CONCEPTO_STD, IMPORTE_STD, AÑO_STD, MES_STD, DIA_STD]

MANDATORY_STD_COLS = [CONCEPTO_STD, IMPORTE_STD, FECHA_STD]
OPTIONAL_STD_COLS = [COMERCIO_STD]
CONFIG_FILENAME = "Configuracion_Categorizador.json"
DB_FILENAME = "Database_Gastos_Acumulados.csv"

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
# **** Modificado: learned_knowledge ahora contendrá más detalle ****
if 'learned_knowledge' not in st.session_state:
    st.session_state.learned_knowledge = {
        'categorias': [],
        'subcategorias_por_cat': {}, # Lista de subcats por categoría
        'comercios_por_cat': {},     # Lista de comercios por categoría
        'subcat_por_comercio_y_cat': {} # Dict[cat][comercio] -> subcat (solo si es única)
    }


# --- Funciones ---
@st.cache_data
def read_uploaded_file(uploaded_file):
    # ... (Sin cambios respecto a la versión anterior) ...
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
    # ... (Sin cambios respecto a la versión anterior) ...
    try:
        if not isinstance(df_raw, pd.DataFrame): st.error("Parse Histórico: No es DF."); return None
        df = df_raw.copy(); df.columns = [str(col).upper().strip() for col in df.columns]
        required = ['CONCEPTO', 'CATEGORÍA', 'SUBCATEGORIA', 'IMPORTE', 'AÑO', 'MES', 'DIA']
        if 'COMERCIO' not in df.columns: df['COMERCIO'] = ''
        missing = [col for col in required if col not in df.columns]
        if missing: st.error(f"Histórico: Faltan cols: {missing}"); return None
        df_std = pd.DataFrame()
        text_map = { CONCEPTO_STD: 'CONCEPTO', COMERCIO_STD: 'COMERCIO', CATEGORIA_STD: 'CATEGORÍA', SUBCATEGORIA_STD: 'SUBCATEGORIA' }
        for std_col, raw_col in text_map.items():
            if raw_col not in df.columns:
                 if std_col == COMERCIO_STD: df_std[COMERCIO_STD] = ''; continue
                 st.error(f"Error Interno: Falta '{raw_col}'."); return None
            try:
                series = df[raw_col].fillna('').astype(str)
                df_std[std_col] = series.str.lower().str.strip() if pd.api.types.is_string_dtype(series.dtype) else series.apply(lambda x: str(x).lower().strip())
            except AttributeError as ae:
                st.error(f"!!! Error de Atributo procesando '{raw_col}' -> '{std_col}'.")
                try:
                    problematic_types = df[raw_col].apply(type).value_counts(); st.error(f"Tipos encontrados: {problematic_types}")
                    non_string_indices = df[raw_col].apply(lambda x: not isinstance(x, (str, type(None), float, int))).index
                    if not non_string_indices.empty: st.error(f"Valores no textuales: {df.loc[non_string_indices, raw_col].head()}")
                except Exception as e_diag: st.error(f"No se pudo diagnosticar: {e_diag}")
                return None
            except Exception as e: st.error(f"Error proc. texto '{raw_col}': {e}"); st.error(traceback.format_exc()); return None
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

# **** MODIFICADO: extract_knowledge_std ****
@st.cache_data
def extract_knowledge_std(df_std):
    """Extrae conocimiento y relaciones del DF estandarizado."""
    knowledge = {
        'categorias': [],
        'subcategorias_por_cat': {},
        'comercios_por_cat': {},
        'subcat_por_comercio_y_cat': {} # Nuevo: Dict[cat][comercio] -> subcat (solo si única)
    }
    if df_std is None or CATEGORIA_STD not in df_std.columns or df_std.empty:
        st.warning("No hay datos válidos para extraer conocimiento.")
        return knowledge
    try:
        # Asegurar que las columnas necesarias para las relaciones existen
        has_subcat = SUBCATEGORIA_STD in df_std.columns
        has_comercio = COMERCIO_STD in df_std.columns

        knowledge['categorias'] = sorted([c for c in df_std[CATEGORIA_STD].dropna().unique() if c])

        for cat in knowledge['categorias']:
            df_cat = df_std[df_std[CATEGORIA_STD] == cat]

            # Subcategorías generales por categoría
            knowledge['subcategorias_por_cat'][cat] = []
            if has_subcat:
                 subcats = df_cat[SUBCATEGORIA_STD].dropna().unique()
                 knowledge['subcategorias_por_cat'][cat] = sorted([s for s in subcats if s])

            # Comercios generales por categoría
            knowledge['comercios_por_cat'][cat] = []
            knowledge['subcat_por_comercio_y_cat'][cat] = {} # Inicializar dict para esta categoría
            if has_comercio:
                 comers = df_cat[COMERCIO_STD].dropna().unique()
                 knowledge['comercios_por_cat'][cat] = sorted([c for c in comers if c and c != 'n/a'])

                 # Nuevo: Calcular subcategoría única o más común por comercio DENTRO de esta categoría
                 if has_subcat:
                     for comercio in knowledge['comercios_por_cat'][cat]:
                         # Filtrar filas para este comercio y categoría, con subcategoría válida
                         df_comercio_cat = df_cat[
                             (df_cat[COMERCIO_STD] == comercio) &
                             (df_cat[SUBCATEGORIA_STD].notna()) &
                             (df_cat[SUBCATEGORIA_STD] != '')
                         ]
                         if not df_comercio_cat.empty:
                             unique_subcats = df_comercio_cat[SUBCATEGORIA_STD].unique()
                             if len(unique_subcats) == 1:
                                 # Si solo hay una subcategoría para este comercio en esta categoría, la guardamos
                                 knowledge['subcat_por_comercio_y_cat'][cat][comercio] = unique_subcats[0]
                             # Opcional: Podrías guardar la más frecuente si no es única
                             # else:
                             #     most_common_subcat = Counter(df_comercio_cat[SUBCATEGORIA_STD]).most_common(1)[0][0]
                             #     knowledge['subcat_por_comercio_y_cat'][cat][comercio] = most_common_subcat + " (más común)"

    except Exception as e_kg: st.error(f"Error extrayendo conocimiento detallado: {e_kg}")
    return knowledge

# --- train_classifier_std (sin cambios) ---
@st.cache_resource
def train_classifier_std(df_std):
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

# --- standardize_data_with_mapping (sin cambios) ---
def standardize_data_with_mapping(df_raw, mapping):
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
        orig_keep = [c for c in original_columns if c not in source_cols_used]
        for col in orig_keep:
            t_col = f"ORIG_{col}"; sfx = 1
            while t_col in df_std.columns: t_col = f"ORIG_{col}_{sfx}"; sfx += 1
            df_std[t_col] = df[col]
        df_std = df_std.dropna(subset=[IMPORTE_STD, TEXTO_MODELO])
        df_std = df_std[df_std[TEXTO_MODELO] != '']
        return df_std
    except Exception as e: st.error(f"Error Gral aplicando mapeo '{mapping.get('bank_name', '?')}': {e}"); st.error(traceback.format_exc()); return None

# **** NUEVA Función para parsear BD Acumulada ****
def parse_accumulated_db_for_training(df_db):
    if not isinstance(df_db, pd.DataFrame) or df_db.empty: st.error("BD Acumulada vacía."); return None
    df_db.columns = [str(col).upper().strip() for col in df_db.columns] # Limpiar columnas al cargar
    category_col_to_use = None
    if CATEGORIA_STD in df_db.columns: category_col_to_use = CATEGORIA_STD
    elif CATEGORIA_PREDICHA in df_db.columns: category_col_to_use = CATEGORIA_PREDICHA
    else: st.error("BD Acumulada no tiene 'CATEGORIA_STD' ni 'CATEGORIA_PREDICHA'."); return None
    required_for_train = [CONCEPTO_STD, IMPORTE_STD, AÑO_STD, MES_STD, DIA_STD, category_col_to_use]
    if COMERCIO_STD not in df_db.columns: df_db[COMERCIO_STD] = ''
    missing_cols = [col for col in required_for_train if col not in df_db.columns]
    if missing_cols: st.error(f"BD Acumulada incompleta. Faltan: {missing_cols}"); return None
    df_train_ready = df_db.copy()
    try:
        df_train_ready[CONCEPTO_STD] = df_train_ready[CONCEPTO_STD].fillna('').astype(str).str.lower().str.strip()
        df_train_ready[COMERCIO_STD] = df_train_ready[COMERCIO_STD].fillna('').astype(str).str.lower().str.strip()
        df_train_ready[category_col_to_use] = df_train_ready[category_col_to_use].fillna('').astype(str).str.lower().str.strip()
        df_train_ready[IMPORTE_STD] = pd.to_numeric(df_train_ready[IMPORTE_STD], errors='coerce')
    except Exception as e_clean: st.error(f"Error limpiando BD: {e_clean}"); return None
    if TEXTO_MODELO not in df_train_ready.columns:
        df_train_ready[TEXTO_MODELO] = (df_train_ready[CONCEPTO_STD] + ' ' + df_train_ready[COMERCIO_STD]).str.strip()
    if category_col_to_use == CATEGORIA_PREDICHA:
        df_train_ready = df_train_ready.rename(columns={CATEGORIA_PREDICHA: CATEGORIA_STD})
        if SUBCATEGORIA_STD not in df_train_ready.columns: df_train_ready[SUBCATEGORIA_STD] = ''
    df_train_ready = df_train_ready.dropna(subset=[IMPORTE_STD, CATEGORIA_STD, TEXTO_MODELO])
    df_train_ready = df_train_ready[df_train_ready[CATEGORIA_STD] != ''][df_train_ready[TEXTO_MODELO] != '']
    if df_train_ready.empty: st.warning("BD Acumulada sin filas válidas para entrenar."); return None
    return df_train_ready
# ------------------------------------------------------------------------------------

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🏦 Categorizador Bancario Multi-Formato v4.1")
st.caption(f"Archivo Config: `{CONFIG_FILENAME}`, Archivo BD: `{DB_FILENAME}`")

# --- Carga Base de Datos Acumulada (Sidebar) ---
st.sidebar.header("Base de Datos Acumulada")
uploaded_db_file = st.sidebar.file_uploader(
    f"Cargar Base de Datos ({DB_FILENAME})",
    type=["csv", "xlsx", "xls"],
    key="db_loader"
)
if uploaded_db_file:
    db_uploader_key = "db_loader_processed_id"
    if uploaded_db_file.file_id != st.session_state.get(db_uploader_key, None):
        st.sidebar.info("Cargando base de datos...")
        df_db_loaded, _ = read_uploaded_file(uploaded_db_file)
        if df_db_loaded is not None:
            # Limpiar columnas y renombrar CATEGORIA si es necesario
            df_db_loaded.columns = [str(col).upper().strip() for col in df_db_loaded.columns]
            if 'CATEGORIA' in df_db_loaded.columns and CATEGORIA_PREDICHA not in df_db_loaded.columns:
                 df_db_loaded = df_db_loaded.rename(columns={'CATEGORIA': CATEGORIA_PREDICHA})
            elif 'CATEGORÍA' in df_db_loaded.columns and CATEGORIA_PREDICHA not in df_db_loaded.columns:
                 df_db_loaded = df_db_loaded.rename(columns={'CATEGORÍA': CATEGORIA_PREDICHA})

            # Verificar columnas mínimas para que la BD sea útil
            expected_db_cols = [CONCEPTO_STD, IMPORTE_STD, AÑO_STD, MES_STD, DIA_STD]
            if CATEGORIA_PREDICHA in df_db_loaded.columns: expected_db_cols.append(CATEGORIA_PREDICHA)
            elif CATEGORIA_STD in df_db_loaded.columns: expected_db_cols.append(CATEGORIA_STD)
            else: expected_db_cols.append("CATEGORIA_X") # Añadir una que seguramente faltará

            missing_db_cols = [col for col in expected_db_cols if col not in df_db_loaded.columns]

            if not missing_db_cols:
                st.session_state.accumulated_data = df_db_loaded
                st.session_state[db_uploader_key] = uploaded_db_file.file_id
                st.sidebar.success(f"BD cargada ({len(df_db_loaded)} filas).")
                # Intentar extraer conocimiento si no se cargó config
                if not st.session_state.knowledge_loaded:
                     # Necesitamos asegurar que la columna de categoría es CATEGORIA_STD para extract_knowledge
                     df_for_knowledge = df_db_loaded.copy()
                     if CATEGORIA_PREDICHA in df_for_knowledge.columns and CATEGORIA_STD not in df_for_knowledge.columns:
                         df_for_knowledge = df_for_knowledge.rename(columns={CATEGORIA_PREDICHA: CATEGORIA_STD})
                     # Añadir SUBCATEGORIA_STD si falta
                     if SUBCATEGORIA_STD not in df_for_knowledge.columns: df_for_knowledge[SUBCATEGORIA_STD] = ''

                     if CATEGORIA_STD in df_for_knowledge.columns:
                          st.session_state.learned_knowledge = extract_knowledge_std(df_for_knowledge)
                          st.session_state.knowledge_loaded = bool(st.session_state.learned_knowledge.get('categorias'))
                          if st.session_state.knowledge_loaded: st.sidebar.info("Conocimiento extraído de BD cargada.")
                          else: st.sidebar.warning("No se pudo extraer conocimiento de la BD cargada.")
                     else:
                          st.sidebar.warning("BD cargada no tiene columna de categoría estándar para extraer conocimiento.")

                st.rerun()
            else:
                st.sidebar.error(f"Archivo DB inválido. Faltan: {missing_db_cols}")
                st.session_state[db_uploader_key] = None
        else:
            st.sidebar.error("No se pudo leer archivo de BD.")
            st.session_state[db_uploader_key] = None

# --- Fase 1: Cargar Configuración / Entrenar Modelo Base ---
with st.expander("Paso 1: Cargar Configuración o Entrenar Modelo", expanded=True):
    col1a, col1b = st.columns(2)
    with col1a:
        st.subheader("Opción A: Cargar Configuración Completa")
        st.write(f"Carga `{CONFIG_FILENAME}` (mapeos + conocimiento).")
        uploaded_config_file_f1 = st.file_uploader(f"Cargar '{CONFIG_FILENAME}'", type="json", key="config_loader_f1")
        if uploaded_config_file_f1:
            config_uploader_key_f1 = "config_loader_processed_id_f1"
            if uploaded_config_file_f1.file_id != st.session_state.get(config_uploader_key_f1, None):
                try:
                    config_data = json.load(uploaded_config_file_f1)
                    is_valid = True; error_msg = ""
                    if not isinstance(config_data, dict): is_valid = False; error_msg = "No es dict."
                    elif 'bank_mappings' not in config_data or not isinstance(config_data['bank_mappings'], dict): is_valid = False; error_msg += " Falta/Inválido 'bank_mappings'."
                    elif 'learned_knowledge' not in config_data or not isinstance(config_data['learned_knowledge'], dict): is_valid = False; error_msg += " Falta/Inválido 'learned_knowledge'."
                    elif not all(k in config_data['learned_knowledge'] for k in ['categorias', 'subcategorias_por_cat', 'comercios_por_cat', 'subcat_por_comercio_y_cat']): is_valid = False; error_msg += " Faltan claves en 'learned_knowledge'."
                    if is_valid:
                        st.session_state.bank_mappings = config_data['bank_mappings']
                        st.session_state.learned_knowledge = config_data['learned_knowledge']
                        st.session_state.knowledge_loaded = bool(st.session_state.learned_knowledge.get('categorias'))
                        st.success(f"Configuración completa cargada.")
                        st.sidebar.success("Config. Cargada")
                        st.session_state[config_uploader_key_f1] = uploaded_config_file_f1.file_id
                        if not st.session_state.model_trained: st.info("Conocimiento cargado. Entrena el modelo si quieres categorizar.")
                        st.rerun()
                    else: st.error(f"Error formato config: {error_msg.strip()}"); st.session_state[config_uploader_key_f1] = None
                except Exception as e_load: st.error(f"Error cargando config: {e_load}"); st.error(traceback.format_exc()); st.session_state[config_uploader_key_f1] = None

    with col1b:
        st.subheader("Opción B: (Re)Entrenar Modelo desde BD")
        st.write("Entrena el modelo usando los datos de la BD Acumulada (cargada en sidebar).")
        if st.session_state.accumulated_data.empty:
            st.warning("BD Acumulada está vacía. Cárgala o añade datos categorizando.")
        elif st.button("🧠 Entrenar/Reentrenar Modelo con BD Actual", key="train_db_f1b"):
             with st.spinner("Preparando BD y entrenando..."):
                df_train_ready = parse_accumulated_db_for_training(st.session_state.accumulated_data.copy())
                if df_train_ready is not None and not df_train_ready.empty:
                    st.success("Datos de BD preparados.")
                    # Actualizar conocimiento desde la BD
                    st.session_state.learned_knowledge = extract_knowledge_std(df_train_ready)
                    st.session_state.knowledge_loaded = True
                    st.sidebar.success("Conocimiento Actualizado (BD)")
                    with st.sidebar.expander("Categorías (BD)"): st.write(st.session_state.learned_knowledge['categorias'])
                    # Entrenar
                    model, vectorizer, report = train_classifier_std(df_train_ready)
                    if model and vectorizer:
                        st.session_state.model = model; st.session_state.vectorizer = vectorizer
                        st.session_state.model_trained = True; st.session_state.training_report = report
                        st.success("¡Modelo (re)entrenado con BD!"); st.sidebar.subheader("Evaluación Modelo");
                        with st.sidebar.expander("Ver Informe"): st.text(st.session_state.training_report)
                    else:
                        st.error("Fallo entrenamiento con BD."); st.session_state.model_trained = False
                        st.session_state.training_report = report; st.sidebar.error("Entrenamiento Fallido")
                        st.sidebar.text(st.session_state.training_report)
                else: st.error("No se pudieron preparar datos de BD para entrenar."); st.session_state.model_trained = False

# --- Fase 2: Formatos Bancarios y Guardar Configuración ---
with st.expander("Paso 2: Definir Formatos Bancarios y Guardar Configuración"):
    # ... (UI sin cambios, pero ahora descarga mapeos + learned_knowledge) ...
    st.write("Aquí puedes enseñar a la aplicación cómo leer archivos de diferentes bancos (CSV/Excel) o ver/editar mapeos existentes.")
    st.info("Los cambios se guardan en memoria. Usa el botón al final para descargar la configuración completa (mapeos + conocimiento base).")
    st.subheader("Aprender/Editar Formato de Banco")
    bank_options = ["SANTANDER", "EVO", "WIZINK", "AMEX"]
    selected_bank_learn = st.selectbox("Selecciona Banco:", bank_options, key="bank_learn_f2_select")
    uploaded_sample_file = st.file_uploader(
        f"Cargar archivo de ejemplo de {selected_bank_learn} (.csv, .xlsx, .xls)",
        type=["csv", "xlsx", "xls"],
        key="sample_uploader_f2"
    )
    if uploaded_sample_file:
        df_sample, detected_columns = read_uploaded_file(uploaded_sample_file)
        if df_sample is not None:
            st.write(f"Columnas detectadas en {selected_bank_learn}:"); st.code(f"{detected_columns}")
            st.dataframe(df_sample.head(3))
            st.subheader("Mapeo de Columnas")
            saved_mapping = st.session_state.bank_mappings.get(selected_bank_learn, {'columns': {}})
            cols_with_none = [None] + detected_columns
            st.markdown("**Esenciales:**")
            map_concepto = st.selectbox(f"`{CONCEPTO_STD}`", cols_with_none, index=cols_with_none.index(saved_mapping['columns'].get(CONCEPTO_STD)) if saved_mapping['columns'].get(CONCEPTO_STD) in cols_with_none else 0, key=f"map_{CONCEPTO_STD}_{selected_bank_learn}")
            map_importe = st.selectbox(f"`{IMPORTE_STD}`", cols_with_none, index=cols_with_none.index(saved_mapping['columns'].get(IMPORTE_STD)) if saved_mapping['columns'].get(IMPORTE_STD) in cols_with_none else 0, key=f"map_{IMPORTE_STD}_{selected_bank_learn}")
            st.markdown("**Fecha:**"); is_single_date_saved = FECHA_STD in saved_mapping['columns']
            map_single_date = st.checkbox("Fecha en 1 columna", value=is_single_date_saved, key=f"map_single_date_{selected_bank_learn}")
            map_fecha_unica=None; map_formato_fecha=None; map_año=None; map_mes=None; map_dia=None
            if map_single_date:
                map_fecha_unica = st.selectbox(f"`{FECHA_STD}`", cols_with_none, index=cols_with_none.index(saved_mapping['columns'].get(FECHA_STD)) if saved_mapping['columns'].get(FECHA_STD) in cols_with_none else 0, key=f"map_{FECHA_STD}_{selected_bank_learn}")
                map_formato_fecha = st.text_input("Formato", value=saved_mapping.get('date_format', ''), key=f"map_date_format_{selected_bank_learn}")
            else:
                map_año = st.selectbox(f"`{AÑO_STD}`", cols_with_none, index=cols_with_none.index(saved_mapping['columns'].get(AÑO_STD)) if saved_mapping['columns'].get(AÑO_STD) in cols_with_none else 0, key=f"map_{AÑO_STD}_{selected_bank_learn}")
                map_mes = st.selectbox(f"`{MES_STD}`", cols_with_none, index=cols_with_none.index(saved_mapping['columns'].get(MES_STD)) if saved_mapping['columns'].get(MES_STD) in cols_with_none else 0, key=f"map_{MES_STD}_{selected_bank_learn}")
                map_dia = st.selectbox(f"`{DIA_STD}`", cols_with_none, index=cols_with_none.index(saved_mapping['columns'].get(DIA_STD)) if saved_mapping['columns'].get(DIA_STD) in cols_with_none else 0, key=f"map_{DIA_STD}_{selected_bank_learn}")
            st.markdown("**Opcionales:**")
            map_comercio = st.selectbox(f"`{COMERCIO_STD}`", cols_with_none, index=cols_with_none.index(saved_mapping['columns'].get(COMERCIO_STD)) if saved_mapping['columns'].get(COMERCIO_STD) in cols_with_none else 0, key=f"map_{COMERCIO_STD}_{selected_bank_learn}")
            st.markdown("**Importe:**")
            val_map_decimal_sep = st.text_input("Separador Decimal", value=saved_mapping.get('decimal_sep', ','), key=f"map_decimal_{selected_bank_learn}")
            val_map_thousands_sep = st.text_input("Separador Miles", value=saved_mapping.get('thousands_sep', ''), key=f"map_thousands_{selected_bank_learn}")
            if st.button(f"💾 Guardar Mapeo para {selected_bank_learn}", key="save_mapping_f2"):
                final_mapping_cols = {}
                if map_concepto: final_mapping_cols[CONCEPTO_STD] = map_concepto
                if map_importe: final_mapping_cols[IMPORTE_STD] = map_importe
                if map_single_date and map_fecha_unica: final_mapping_cols[FECHA_STD] = map_fecha_unica
                if not map_single_date and map_año: final_mapping_cols[AÑO_STD] = map_año
                if not map_single_date and map_mes: final_mapping_cols[MES_STD] = map_mes
                if not map_single_date and map_dia: final_mapping_cols[DIA_STD] = map_dia
                if map_comercio: final_mapping_cols[COMERCIO_STD] = map_comercio
                valid = True; current_fmt = map_formato_fecha
                if not final_mapping_cols.get(CONCEPTO_STD): st.error("Mapea CONCEPTO."); valid=False
                if not final_mapping_cols.get(IMPORTE_STD): st.error("Mapea IMPORTE."); valid=False
                if map_single_date:
                    if not final_mapping_cols.get(FECHA_STD): st.error("Mapea FECHA."); valid=False
                    elif not current_fmt: st.error("Especifica formato."); valid=False
                else:
                    if not all(final_mapping_cols.get(d) for d in [AÑO_STD,MES_STD,DIA_STD]): st.error("Mapea AÑO, MES y DIA."); valid=False
                if valid:
                    mapping_to_save = {'bank_name': selected_bank_learn, 'columns': final_mapping_cols, 'decimal_sep': val_map_decimal_sep.strip(), 'thousands_sep': val_map_thousands_sep.strip() or None}
                    if map_single_date and current_fmt: mapping_to_save['date_format'] = current_fmt.strip()
                    st.session_state.bank_mappings[selected_bank_learn] = mapping_to_save
                    st.success(f"¡Mapeo {selected_bank_learn} guardado!"); st.rerun()
                else: st.warning("Revisa errores.")
    st.divider()
    st.subheader("Descargar Configuración Completa (Mapeos + Conocimiento)")
    if st.session_state.bank_mappings or st.session_state.learned_knowledge.get('categorias'):
        try:
            config_to_save = {
                'bank_mappings': st.session_state.get('bank_mappings', {}),
                'learned_knowledge': st.session_state.get('learned_knowledge', {'categorias': [], 'subcategorias_por_cat': {}, 'comercios_por_cat': {}, 'subcat_por_comercio_y_cat': {}})
            }
            config_json_str = json.dumps(config_to_save, indent=4, ensure_ascii=False)
            st.download_button(label=f"💾 Descargar '{CONFIG_FILENAME}'", data=config_json_str.encode('utf-8'), file_name=CONFIG_FILENAME, mime='application/json', key='download_config_f2')
        except Exception as e_dump: st.error(f"Error preparando descarga: {e_dump}")
    else: st.info("No hay mapeos ni conocimiento base para guardar.")

# --- Fase 3: Categorización ---
with st.expander("Paso 3: Categorizar Nuevos Archivos y Añadir a BD", expanded=True):
    model_ready_for_pred = st.session_state.get('model_trained', False)
    mappings_available = bool(st.session_state.get('bank_mappings', {}))
    knowledge_ready = st.session_state.get('knowledge_loaded', False)

    if not knowledge_ready: st.warning("⚠️ Conocimiento base no aprendido o cargado (Ver Paso 1).")
    elif not mappings_available: st.warning("⚠️ No se han aprendido o cargado formatos bancarios (Ver Paso 2).")
    elif not model_ready_for_pred:
        st.warning("⚠️ Modelo no entrenado en esta sesión.")
        st.info("Ve al Paso 1 (Opción B) y entrena el modelo usando la BD Acumulada antes de categorizar.")
    else: # Modelo entrenado Y Mapeos disponibles
        st.write("Selecciona el banco y sube el archivo **sin categorizar** (CSV o Excel).")
        available_banks_for_pred = list(st.session_state.bank_mappings.keys())
        selected_bank_predict = st.selectbox("Banco del Nuevo Archivo:", available_banks_for_pred, key="bank_predict_f3")
        uploaded_final_file = st.file_uploader(
            f"Cargar archivo NUEVO de {selected_bank_predict} (.csv, .xlsx, .xls)",
            type=["csv", "xlsx", "xls"],
            key="final_uploader_f3"
        )
        if uploaded_final_file and selected_bank_predict:
            mapping_to_use = st.session_state.bank_mappings.get(selected_bank_predict)
            if not mapping_to_use: st.error(f"Error interno: No mapeo para {selected_bank_predict}.")
            else:
                 st.write(f"Procesando '{uploaded_final_file.name}'...")
                 df_std_new = None
                 with st.spinner(f"Estandarizando..."):
                      df_raw_new, _ = read_uploaded_file(uploaded_final_file)
                      if df_raw_new is not None: df_std_new = standardize_data_with_mapping(df_raw_new.copy(), mapping_to_use)
                      else: st.error(f"No se pudo leer: {uploaded_final_file.name}")
                 if df_std_new is not None and not df_std_new.empty:
                      st.success("Datos estandarizados.")
                      with st.spinner("Categorizando..."):
                          try:
                               if TEXTO_MODELO not in df_std_new.columns: st.error(f"Error: Falta {TEXTO_MODELO}.")
                               else:
                                    df_pred = df_std_new.dropna(subset=[TEXTO_MODELO]).copy()
                                    if not df_pred.empty:
                                         # --- Predicción Categoría Principal ---
                                         X_new_vec = st.session_state.vectorizer.transform(df_pred[TEXTO_MODELO])
                                         predictions_cat = st.session_state.model.predict(X_new_vec)
                                         df_pred[CATEGORIA_PREDICHA] = [str(p).capitalize() for p in predictions_cat]

                                         # --- Predicción/Asignación Comercio y Subcategoría ---
                                         pred_comercios = []
                                         pred_subcats = []
                                         knowledge = st.session_state.learned_knowledge # Acceder al conocimiento
                                         for index, row in df_pred.iterrows():
                                             pred_cat = row[CATEGORIA_PREDICHA].lower() # Usar la categoría predicha (en minúsculas)
                                             input_comercio = row.get(COMERCIO_STD, '') # Comercio estandarizado del input

                                             # 1. Comercio Predicho/Estandarizado
                                             known_comers_for_cat = knowledge['comercios_por_cat'].get(pred_cat, [])
                                             comercio_final = input_comercio # Default al input
                                             if input_comercio and input_comercio in known_comers_for_cat:
                                                 comercio_final = input_comercio # Usar el nombre conocido si hay match exacto
                                             pred_comercios.append(comercio_final.capitalize()) # Capitalizar para mostrar

                                             # 2. Subcategoría Predicha (Heurística)
                                             subcat_final = '' # Default vacía
                                             # Regla 1: Subcategoría única por comercio+categoría
                                             if comercio_final: # Solo si tenemos un comercio final
                                                subcat_unica = knowledge['subcat_por_comercio_y_cat'].get(pred_cat, {}).get(comercio_final.lower())
                                                if subcat_unica:
                                                     subcat_final = subcat_unica
                                             # Regla 2: Única subcategoría para la categoría general
                                             if not subcat_final: # Si la regla 1 no aplicó
                                                 known_subcats_for_cat = knowledge['subcategorias_por_cat'].get(pred_cat, [])
                                                 if len(known_subcats_for_cat) == 1:
                                                     subcat_final = known_subcats_for_cat[0]
                                             # Regla 3: Default (ya está vacía)

                                             pred_subcats.append(subcat_final.capitalize()) # Capitalizar para mostrar

                                         df_pred[COMERCIO_PREDICHO] = pred_comercios
                                         df_pred[SUBCATEGORIA_PREDICHA] = pred_subcats
                                         # --- Fin Predicción Comercio/Subcategoría ---

                                         # --- ACUMULACIÓN BD ---
                                         st.write("Añadiendo a base de datos...")
                                         # Asegurar que las columnas estándar existen antes de seleccionar
                                         for col in DB_FINAL_COLS:
                                              if col not in df_pred.columns: df_pred[col] = '' # Crear vacías si faltan
                                         db_cols_to_keep = DB_FINAL_COLS + [c for c in df_pred.columns if c.startswith('ORIG_')]
                                         final_db_cols = [col for col in db_cols_to_keep if col in df_pred.columns]
                                         df_to_append = df_pred[final_db_cols].copy()
                                         if 'accumulated_data' not in st.session_state or st.session_state.accumulated_data.empty:
                                             st.session_state.accumulated_data = df_to_append
                                         else:
                                             current_db = st.session_state.accumulated_data
                                             combined_cols = current_db.columns.union(df_to_append.columns)
                                             current_db = current_db.reindex(columns=combined_cols); df_to_append = df_to_append.reindex(columns=combined_cols)
                                             st.session_state.accumulated_data = pd.concat([current_db, df_to_append], ignore_index=True).fillna('') # Rellenar NaNs por si acaso
                                         st.success(f"{len(df_to_append)} transacciones añadidas a BD.")
                                         # --- FIN ACUMULACIÓN ---

                                         st.subheader("📊 Resultados (este archivo)")
                                         # Reordenar para mostrar las nuevas columnas al principio
                                         display_cols_order = [CATEGORIA_PREDICHA, SUBCATEGORIA_PREDICHA, COMERCIO_PREDICHO] + \
                                                              [c for c in final_display_cols if c not in [CATEGORIA_PREDICHA, SUBCATEGORIA_PREDICHA, COMERCIO_PREDICHO]]
                                         st.dataframe(df_pred[display_cols_order])

                                         csv_output = df_pred.to_csv(index=False, sep=';', decimal=',').encode('utf-8')
                                         st.download_button(label=f"📥 Descargar '{uploaded_final_file.name}' Categorizado", data=csv_output, file_name=f"categorizado_{uploaded_final_file.name}", mime='text/csv', key=f"download_final_{uploaded_final_file.name}")
                                    else: st.warning("No quedaron filas válidas para categorizar.")
                          except AttributeError as ae_inner: st.error(f"Error Atributo (interno): {ae_inner}"); st.error(traceback.format_exc())
                          except Exception as e_pred: st.error(f"Error durante la predicción: {e_pred}"); st.error(traceback.format_exc())
                 elif df_std_new is not None and df_std_new.empty: st.warning("Archivo vacío o sin datos válidos tras estandarizar.")
                 else: st.error("Fallo en la estandarización usando el mapeo.")

# --- Fase 4: Base de Datos Acumulada ---
with st.expander("Paso 4: Ver y Descargar Base de Datos Acumulada", expanded=False):
    db_state_f4 = st.session_state.get('accumulated_data', pd.DataFrame())
    if db_state_f4 is not None and not db_state_f4.empty:
        st.write(f"Base de datos actual en memoria ({len(db_state_f4)} filas):")
        # Mostrar columnas relevantes de la BD final
        cols_to_show_db = [col for col in DB_FINAL_COLS if col in db_state_f4.columns]
        st.dataframe(db_state_f4[cols_to_show_db]) # Mostrar todas las filas con columnas seleccionadas

        st.subheader(f"Descargar Base de Datos Completa")
        col_db1, col_db2 = st.columns(2)
        with col_db1:
            try:
                db_csv_output = db_state_f4.to_csv(index=False, sep=';', decimal=',').encode('utf-8')
                st.download_button(label=f"💾 Descargar '{DB_FILENAME}' (CSV)", data=db_csv_output, file_name=DB_FILENAME, mime='text/csv', key='download_db_csv_f4')
            except Exception as e_db_csv: st.error(f"Error CSV BD: {e_db_csv}")
        with col_db2:
            try:
                output_excel = io.BytesIO()
                with pd.ExcelWriter(output_excel, engine='openpyxl') as writer: db_state_f4.to_excel(writer, index=False, sheet_name='Gastos')
                excel_data = output_excel.getvalue(); db_excel_filename = DB_FILENAME.replace('.csv', '.xlsx')
                st.download_button(label=f"💾 Descargar '{db_excel_filename}' (Excel)", data=excel_data, file_name=db_excel_filename, mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet', key='download_db_excel_f4')
            except Exception as e_db_xlsx: st.error(f"Error Excel BD: {e_db_xlsx}")
    else:
        st.info("La base de datos acumulada está vacía. Carga una existente o añade datos categorizando.")

# --- Sidebar Info y Estado (Final del script) ---
st.sidebar.divider()
st.sidebar.header("Acerca de")
st.sidebar.info( "1. Carga Config/Entrena. 2. Define Formatos/Guarda Config. 3. Categoriza y Acumula. 4. Gestiona BD.")
st.sidebar.divider()
st.sidebar.subheader("Estado Actual")
model_ready_sidebar = st.session_state.get('model_trained', False)
knowledge_ready_sidebar = st.session_state.get('knowledge_loaded', False)
if model_ready_sidebar: st.sidebar.success("✅ Modelo Entrenado")
elif knowledge_ready_sidebar: st.sidebar.info("ℹ️ Conocimiento Cargado (Entrenar modelo si es necesario)")
else: st.sidebar.warning("❌ Sin Modelo/Conocimiento")

if st.session_state.get('bank_mappings', {}): st.sidebar.success(f"✅ Mapeos Cargados ({len(st.session_state.bank_mappings)} bancos)")
else: st.sidebar.warning("❌ Sin Mapeos Bancarios")

db_state_sidebar = st.session_state.get('accumulated_data', pd.DataFrame())
if db_state_sidebar is not None and not db_state_sidebar.empty:
    st.sidebar.success(f"✅ BD en Memoria ({len(db_state_sidebar)} filas)")
else:
    st.sidebar.info("ℹ️ BD en Memoria Vacía")
