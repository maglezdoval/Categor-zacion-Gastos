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
FECHA_STD = 'FECHA_STD' # Usaremos una columna de fecha estándar internamente
CATEGORIA_STD = 'CATEGORIA_STD'
SUBCATEGORIA_STD = 'SUBCATEGORIA_STD'
TEXTO_MODELO = 'TEXTO_MODELO' # Columna combinada para el modelo
CATEGORIA_PREDICHA = 'CATEGORIA_PREDICHA'

# Columnas estándar que NECESITAMOS mapear desde los archivos bancarios
MANDATORY_STD_COLS = [CONCEPTO_STD, IMPORTE_STD, FECHA_STD] # Fecha se maneja AÑO/MES/DIA o FECHA_STD
OPTIONAL_STD_COLS = [COMERCIO_STD] # Comercio es opcional en el archivo de origen
CONFIG_FILENAME = "Configuracion_Mapeo_Bancos.json" # Nombre estándar para el archivo de config

# --- Session State Initialization ---
if 'model_trained' not in st.session_state: st.session_state.model_trained = False
if 'model' not in st.session_state: st.session_state.model = None
if 'vectorizer' not in st.session_state: st.session_state.vectorizer = None
if 'knowledge' not in st.session_state: st.session_state.knowledge = {'categorias': [], 'subcategorias': {}, 'comercios': {}}
if 'bank_mappings' not in st.session_state: st.session_state.bank_mappings = {}
if 'training_report' not in st.session_state: st.session_state.training_report = "Modelo no entrenado."

# --- Funciones (Parseo, ML, Estandarización) ---
# (Se asume que las funciones parse_historic_categorized, read_sample_csv,
# extract_knowledge_std, train_classifier_std, y standardize_data_with_mapping
# de la respuesta *anterior* (la que corregía el error 'str') están aquí)

# ----- COPIA AQUÍ LAS FUNCIONES COMPLETAS Y CORREGIDAS DE LA RESPUESTA ANTERIOR -----
def parse_historic_categorized(df_raw):
    """Parsea el Gastos.csv inicial para entrenamiento."""
    try:
        #st.write("Debug (parse_historic): Iniciando parseo...")
        if not isinstance(df_raw, pd.DataFrame): st.error("Error Interno: parse_historic_categorized no recibió un DataFrame."); return None
        df = df_raw.copy()
        try:
            df.columns = [str(col).upper().strip() for col in df.columns]
        except Exception as e_col: st.error(f"Error limpiando nombres de columna: {e_col}"); return None
        required = ['CONCEPTO', 'CATEGORÍA', 'SUBCATEGORIA', 'IMPORTE', 'AÑO', 'MES', 'DIA']
        if 'COMERCIO' not in df.columns: df['COMERCIO'] = ''
        missing = [col for col in required if col not in df.columns]
        if missing: st.error(f"Archivo histórico: Faltan columnas esenciales: {', '.join(missing)}"); return None
        df_std = pd.DataFrame()
        text_cols_mapping = { CONCEPTO_STD: 'CONCEPTO', COMERCIO_STD: 'COMERCIO', CATEGORIA_STD: 'CATEGORÍA', SUBCATEGORIA_STD: 'SUBCATEGORIA' }
        for std_col, raw_col in text_cols_mapping.items():
            if raw_col not in df.columns:
                 if std_col == COMERCIO_STD: df_std[COMERCIO_STD] = ''; continue
                 st.error(f"Error Interno: Columna '{raw_col}' desapareció."); return None
            try:
                series = df[raw_col].fillna('').astype(str)
                if pd.api.types.is_string_dtype(series.dtype): df_std[std_col] = series.str.lower().str.strip()
                else: df_std[std_col] = series.apply(lambda x: str(x).lower().strip())
            except AttributeError as ae:
                st.error(f"!!! Error de Atributo procesando '{raw_col}' -> '{std_col}'.")
                try:
                    problematic_types = df[raw_col].apply(type).value_counts()
                    st.error(f"Tipos encontrados: {problematic_types}")
                    non_string_indices = df[raw_col].apply(lambda x: not isinstance(x, (str, type(None), float, int))).index
                    if not non_string_indices.empty: st.error(f"Valores no textuales: {df.loc[non_string_indices, raw_col].head()}")
                except Exception as e_diag: st.error(f"No se pudo diagnosticar: {e_diag}")
                return None
            except Exception as e: st.error(f"Error proc. texto '{raw_col}': {e}"); st.error(traceback.format_exc()); return None
        try:
            importe_str = df['IMPORTE'].astype(str).str.replace(',', '.', regex=False)
            df_std[IMPORTE_STD] = pd.to_numeric(importe_str, errors='coerce')
            if df_std[IMPORTE_STD].isnull().any(): st.warning("Histórico: Algunos importes no son números.")
        except Exception as e: st.error(f"Error proc. IMPORTE histórico: {e}"); return None
        try:
            for col in ['AÑO', 'MES', 'DIA']: df_std[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
        except Exception as e: st.error(f"Error proc. Fechas históricas: {e}"); return None
        try:
            if CONCEPTO_STD not in df_std: df_std[CONCEPTO_STD] = ''
            if COMERCIO_STD not in df_std: df_std[COMERCIO_STD] = ''
            df_std[TEXTO_MODELO] = df_std[CONCEPTO_STD] + ' ' + df_std[COMERCIO_STD]
            df_std[TEXTO_MODELO] = df_std[TEXTO_MODELO].str.strip()
        except Exception as e: st.error(f"Error creando TEXTO_MODELO: {e}"); return None
        if CATEGORIA_STD not in df_std.columns: st.error("Falta CATEGORIA_STD."); return None
        df_std = df_std.dropna(subset=[IMPORTE_STD, CATEGORIA_STD])
        df_std = df_std[df_std[CATEGORIA_STD] != '']
        if df_std.empty: st.warning("No quedaron filas válidas tras filtrar."); return pd.DataFrame()
        return df_std
    except Exception as e: st.error(f"Error Gral parseando histórico: {e}"); st.error(traceback.format_exc()); return None

@st.cache_data
def read_sample_csv(uploaded_file):
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
                 if dialect.delimiter in [',', ';', '\t', '|']: sep = dialect.delimiter;
            else: st.error("Archivo vacío."); return None, []
        except Exception: pass
        try:
            df = pd.read_csv(io.BytesIO(bytes_data), encoding='utf-8', sep=sep, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(bytes_data), encoding='latin1', sep=sep, low_memory=False)
        except Exception as read_err: st.error(f"Error leyendo CSV con sep '{sep}': {read_err}"); return None, []
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
        df = df_raw.copy()
        df.columns = [str(col).strip() for col in df.columns]
        original_columns = df.columns.tolist()
        temp_std_data = {}
        source_cols_used = []
        found_essential = {std_col: False for std_col in MANDATORY_STD_COLS if std_col != FECHA_STD}
        found_optional = {std_col: False for std_col in OPTIONAL_STD_COLS}
        for std_col, source_col in mapping['columns'].items():
             if source_col in original_columns:
                  temp_std_data[std_col] = df[source_col]
                  source_cols_used.append(source_col)
                  if std_col in found_essential: found_essential[std_col] = True
                  if std_col in found_optional: found_optional[std_col] = True
             else:
                  is_essential = std_col in [CONCEPTO_STD, IMPORTE_STD] or \
                                (std_col==FECHA_STD and not (AÑO_STD in mapping['columns'] and MES_STD in mapping['columns'] and DIA_STD in mapping['columns'])) or \
                                (std_col in [AÑO_STD, MES_STD, DIA_STD] and not FECHA_STD in mapping['columns'])
                  if is_essential: st.error(f"Columna esencial mapeada '{source_col}' ('{std_col}') no encontrada."); return None
        df_std = pd.DataFrame(temp_std_data)
        missing_essential_mapped = [k for k, v in found_essential.items() if not v]
        if missing_essential_mapped: st.error(f"Faltan mapeos esenciales: {missing_essential_mapped}"); return None
        fecha_col_std = mapping['columns'].get(FECHA_STD)
        año_col_std = mapping['columns'].get(AÑO_STD)
        mes_col_std = mapping['columns'].get(MES_STD)
        dia_col_std = mapping['columns'].get(DIA_STD)
        date_processed_ok = False
        if FECHA_STD in df_std.columns:
            date_format = mapping.get('date_format')
            if not date_format: st.error("Falta formato fecha."); return None
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
        elif all(col in df_std.columns for col in [AÑO_STD, MES_STD, DIA_STD]):
            try:
                for col_std in [AÑO_STD, MES_STD, DIA_STD]: df_std[col_std] = pd.to_numeric(df_std[col_std], errors='coerce').fillna(0).astype(int)
                date_processed_ok = True
            except Exception as e_num: st.error(f"Error convirtiendo A/M/D a número: {e_num}"); return None
        else: st.error("Mapeo de fecha incompleto."); return None
        if not date_processed_ok: return None
        if IMPORTE_STD in df_std.columns:
            try:
                importe_str = df_std[IMPORTE_STD].fillna('0').astype(str)
                thousands_sep = mapping.get('thousands_sep'); decimal_sep = mapping.get('decimal_sep', ',')
                if thousands_sep: importe_str = importe_str.str.replace(thousands_sep, '', regex=False)
                importe_str = importe_str.str.replace(decimal_sep, '.', regex=False)
                df_std[IMPORTE_STD] = pd.to_numeric(importe_str, errors='coerce')
                if df_std[IMPORTE_STD].isnull().any(): st.warning("Algunos importes no pudieron convertirse.")
            except Exception as e_imp: st.error(f"Error procesando importe: {e_imp}"); return None
        else: st.error("Falta columna IMPORTE_STD mapeada."); return None
        for col_std in [CONCEPTO_STD, COMERCIO_STD]:
            if col_std in df_std.columns: df_std[col_std] = df_std[col_std].fillna('').astype(str).str.lower().str.strip()
            elif col_std == COMERCIO_STD: df_std[COMERCIO_STD] = ''
        if CONCEPTO_STD not in df_std: df_std[CONCEPTO_STD] = ''
        if COMERCIO_STD not in df_std: df_std[COMERCIO_STD] = ''
        df_std[TEXTO_MODELO] = df_std[CONCEPTO_STD] + ' ' + df_std[COMERCIO_STD]
        df_std[TEXTO_MODELO] = df_std[TEXTO_MODELO].str.strip()
        original_cols_to_keep = [c for c in original_columns if c not in source_cols_used]
        for col in original_cols_to_keep:
            target_col_name = f"ORIG_{col}"; suffix = 1
            while target_col_name in df_std.columns: target_col_name = f"ORIG_{col}_{suffix}"; suffix += 1
            df_std[target_col_name] = df[col]
        df_std = df_std.dropna(subset=[IMPORTE_STD, TEXTO_MODELO])
        df_std = df_std[df_std[TEXTO_MODELO] != '']
        return df_std
    except Exception as e: st.error(f"Error Gral aplicando mapeo '{mapping.get('bank_name', 'Desconocido')}': {e}"); st.error(traceback.format_exc()); return None
# ------------------------------------------------------------------------------------

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🏦 Categorizador Bancario Multi-Formato v3")

# --- Fase 1: Entrenamiento Inicial ---
with st.expander("Fase 1: Entrenar Modelo con Datos Históricos Categorizados", expanded=True):
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

# --- Fase 2: Aprendizaje de Formatos Bancarios y Configuración ---
with st.expander("Fase 2: Aprender Formatos y Cargar/Guardar Configuración"):
    st.write("Aquí puedes enseñar a la aplicación cómo leer archivos de diferentes bancos o cargar una configuración guardada.")

    # **Cargar Configuración**
    st.subheader("Cargar Configuración Guardada")
    uploaded_config_file = st.file_uploader(f"Cargar Archivo '{CONFIG_FILENAME}'", type="json", key="config_loader")
    if uploaded_config_file:
        try:
            config_data = json.load(uploaded_config_file)
            if isinstance(config_data, dict):
                st.session_state.bank_mappings = config_data
                st.success(f"Configuración cargada desde '{uploaded_config_file.name}'!")
                st.sidebar.success("Config. Cargada")
                st.rerun() # Recargar para que la UI refleje los mapeos cargados
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
            st.write(f"Columnas detectadas en {selected_bank_learn}:"); st.code(f"{detected_columns}")
            st.dataframe(df_sample.head(3))
            st.subheader("Mapeo de Columnas")
            saved_mapping = st.session_state.bank_mappings.get(selected_bank_learn, {'columns': {}})
            # Usaremos directamente los widgets para obtener los valores al presionar el botón
            # No necesitamos construir current_mapping_ui aquí

            cols_with_none = [None] + detected_columns

            st.markdown("**Campos Esenciales:**")
            map_concepto = st.selectbox(f"`{CONCEPTO_STD}`", cols_with_none, index=cols_with_none.index(saved_mapping['columns'].get(CONCEPTO_STD)) if saved_mapping['columns'].get(CONCEPTO_STD) in cols_with_none else 0, key=f"map_{CONCEPTO_STD}_{selected_bank_learn}")
            map_importe = st.selectbox(f"`{IMPORTE_STD}`", cols_with_none, index=cols_with_none.index(saved_mapping['columns'].get(IMPORTE_STD)) if saved_mapping['columns'].get(IMPORTE_STD) in cols_with_none else 0, key=f"map_{IMPORTE_STD}_{selected_bank_learn}")

            st.markdown("**Campo de Fecha:**")
            is_single_date_saved = FECHA_STD in saved_mapping['columns']
            map_single_date = st.checkbox("Fecha en una sola columna", value=is_single_date_saved, key=f"map_single_date_{selected_bank_learn}")
            map_fecha_unica = None
            map_formato_fecha = None
            map_año = None
            map_mes = None
            map_dia = None
            if map_single_date:
                map_fecha_unica = st.selectbox(f"`{FECHA_STD}`", cols_with_none, index=cols_with_none.index(saved_mapping['columns'].get(FECHA_STD)) if saved_mapping['columns'].get(FECHA_STD) in cols_with_none else 0, key=f"map_{FECHA_STD}_{selected_bank_learn}")
                map_formato_fecha = st.text_input("Formato fecha (ej: %d/%m/%Y)", value=saved_mapping.get('date_format', ''), help="Códigos `strftime` Python", key=f"map_date_format_{selected_bank_learn}")
            else:
                map_año = st.selectbox(f"`{AÑO_STD}`", cols_with_none, index=cols_with_none.index(saved_mapping['columns'].get(AÑO_STD)) if saved_mapping['columns'].get(AÑO_STD) in cols_with_none else 0, key=f"map_{AÑO_STD}_{selected_bank_learn}")
                map_mes = st.selectbox(f"`{MES_STD}`", cols_with_none, index=cols_with_none.index(saved_mapping['columns'].get(MES_STD)) if saved_mapping['columns'].get(MES_STD) in cols_with_none else 0, key=f"map_{MES_STD}_{selected_bank_learn}")
                map_dia = st.selectbox(f"`{DIA_STD}`", cols_with_none, index=cols_with_none.index(saved_mapping['columns'].get(DIA_STD)) if saved_mapping['columns'].get(DIA_STD) in cols_with_none else 0, key=f"map_{DIA_STD}_{selected_bank_learn}")

            st.markdown("**Campos Opcionales:**")
            map_comercio = st.selectbox(f"`{COMERCIO_STD}`", cols_with_none, index=cols_with_none.index(saved_mapping['columns'].get(COMERCIO_STD)) if saved_mapping['columns'].get(COMERCIO_STD) in cols_with_none else 0, key=f"map_{COMERCIO_STD}_{selected_bank_learn}")

            st.markdown("**Configuración Importe:**")
            map_decimal_sep = st.text_input("Separador Decimal", value=saved_mapping.get('decimal_sep', ','), key=f"map_decimal_{selected_bank_learn}")
            map_thousands_sep = st.text_input("Separador Miles", value=saved_mapping.get('thousands_sep', ''), key=f"map_thousands_{selected_bank_learn}")

            # --- Botón de Guardado y Lógica ---
            if st.button(f"💾 Guardar Mapeo para {selected_bank_learn}", key="save_mapping_f2"):
                # Construir el mapeo AHORA, usando los valores actuales de los widgets
                final_mapping_cols = {}
                if map_concepto: final_mapping_cols[CONCEPTO_STD] = map_concepto
                if map_importe: final_mapping_cols[IMPORTE_STD] = map_importe
                if map_single_date and map_fecha_unica: final_mapping_cols[FECHA_STD] = map_fecha_unica
                if not map_single_date and map_año: final_mapping_cols[AÑO_STD] = map_año
                if not map_single_date and map_mes: final_mapping_cols[MES_STD] = map_mes
                if not map_single_date and map_dia: final_mapping_cols[DIA_STD] = map_dia
                if map_comercio: final_mapping_cols[COMERCIO_STD] = map_comercio # Opcional

                # Validación
                valid_mapping = True
                if not final_mapping_cols.get(CONCEPTO_STD): st.error("Mapea CONCEPTO_STD."); valid_mapping = False
                if not final_mapping_cols.get(IMPORTE_STD): st.error("Mapea IMPORTE_STD."); valid_mapping = False
                if map_single_date:
                    if not final_mapping_cols.get(FECHA_STD): st.error("Mapea FECHA_STD."); valid_mapping = False
                    elif not map_formato_fecha: st.error("Especifica formato fecha."); valid_mapping = False
                else:
                    if not all(final_mapping_cols.get(d) for d in [AÑO_STD, MES_STD, DIA_STD]): st.error("Mapea AÑO, MES y DIA."); valid_mapping = False

                if valid_mapping:
                    mapping_to_save = {
                        'bank_name': selected_bank_learn,
                        'columns': final_mapping_cols,
                        'decimal_sep': map_decimal_sep.strip(),
                        'thousands_sep': map_thousands_sep.strip() or None,
                    }
                    if map_single_date and map_formato_fecha:
                        mapping_to_save['date_format'] = map_formato_fecha.strip()

                    st.session_state.bank_mappings[selected_bank_learn] = mapping_to_save
                    st.success(f"¡Mapeo para {selected_bank_learn} guardado/actualizado!")
                    st.rerun() # Recargar para reflejar cambio en sidebar
                else:
                    st.warning("Revisa los errores antes de guardar.")

    # **Descargar Configuración**
    st.divider()
    st.subheader("Descargar Configuración Completa")
    if st.session_state.bank_mappings:
        try:
            config_json_str = json.dumps(st.session_state.bank_mappings, indent=4, ensure_ascii=False)
            st.download_button(
                label=f"💾 Descargar '{CONFIG_FILENAME}'", data=config_json_str.encode('utf-8'),
                file_name=CONFIG_FILENAME, mime='application/json', key='download_config'
            )
        except Exception as e_dump: st.error(f"Error preparando descarga: {e_dump}")
    else: st.info("No hay mapeos guardados.")


# --- Fase 3: Categorización ---
with st.expander("Fase 3: Categorizar Nuevos Archivos", expanded=True):
    # Condiciones para mostrar la fase de categorización:
    # 1. El modelo debe estar entrenado
    # 2. Debe haber mapeos disponibles (ya sea aprendidos o cargados)
    model_ready = st.session_state.get('model_trained', False)
    mappings_available = bool(st.session_state.get('bank_mappings', {})) # True si el dict no está vacío

    if not model_ready:
        st.warning("⚠️ Modelo no entrenado (Ver Fase 1).")
    elif not mappings_available:
        # Esta advertencia SÓLO se muestra si NO hay mapeos en memoria
        st.warning("⚠️ No se han aprendido o cargado formatos bancarios (Ver Fase 2).")
    else: # Modelo entrenado Y mapeos disponibles
        st.write("Selecciona el banco y sube el archivo CSV **sin categorizar** que deseas procesar.")
        available_banks_for_pred = list(st.session_state.bank_mappings.keys())
        # No necesitamos el if not available_banks_for_pred aquí porque ya lo comprobamos arriba

        selected_bank_predict = st.selectbox("Banco del Nuevo Archivo:", available_banks_for_pred, key="bank_predict_f3")

        uploaded_final_file = st.file_uploader(f"Cargar archivo CSV NUEVO de {selected_bank_predict}", type="csv", key="final_uploader_f3")

        if uploaded_final_file and selected_bank_predict:
            mapping_to_use = st.session_state.bank_mappings.get(selected_bank_predict)
            if not mapping_to_use:
                 st.error(f"Error interno: No se encontró el mapeo para {selected_bank_predict}.")
            else:
                 st.write(f"Procesando '{uploaded_final_file.name}'...")
                 df_std_new = None
                 with st.spinner(f"Estandarizando datos..."):
                      df_raw_new, _ = read_sample_csv(uploaded_final_file)
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
                                         capitalized_predictions = [str(p).capitalize() for p in predictions]
                                         df_pred[CATEGORIA_PREDICHA] = capitalized_predictions

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
                          except AttributeError as ae_inner:
                               st.error(f"Error de Atributo (interno): {ae_inner}"); st.error(traceback.format_exc())
                          except Exception as e_pred:
                               st.error(f"Error durante la predicción: {e_pred}"); st.error(traceback.format_exc())

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
# Mostrar estado actual en Sidebar
st.sidebar.divider()
st.sidebar.subheader("Estado Actual")
if st.session_state.get('model_trained', False):
    st.sidebar.success("✅ Modelo Entrenado")
else:
    st.sidebar.warning("❌ Modelo NO Entrenado")

if st.session_state.get('bank_mappings', {}):
    st.sidebar.success(f"✅ Mapeos Cargados/Guardados ({len(st.session_state.bank_mappings)} bancos)")
else:
    st.sidebar.warning("❌ Sin Mapeos Bancarios")
