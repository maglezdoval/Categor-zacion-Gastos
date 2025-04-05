import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import io
import numpy as np
from datetime import datetime
import json # Para guardar/cargar mapeos fácilmente (opcional)
import traceback # Para imprimir errores detallados

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
MANDATORY_STD_COLS = [CONCEPTO_STD, IMPORTE_STD, FECHA_STD]
OPTIONAL_STD_COLS = [COMERCIO_STD] # Comercio es opcional en el archivo de origen

# --- Session State Initialization ---
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'vectorizer' not in st.session_state:
    st.session_state.vectorizer = None
if 'knowledge' not in st.session_state:
    st.session_state.knowledge = {'categorias': [], 'subcategorias': {}, 'comercios': {}}
# Guardará los mapeos: {'SANTANDER': {'CONCEPTO_STD': 'Concepto Banco', ...}, 'EVO': {...}}
if 'bank_mappings' not in st.session_state:
    st.session_state.bank_mappings = {}
if 'training_report' not in st.session_state:
    st.session_state.training_report = "Modelo no entrenado."

# --- Funciones ---

def parse_historic_categorized(df_raw):
    """Parsea el Gastos.csv inicial para entrenamiento."""
    try:
        st.write("Debug (parse_historic): Iniciando parseo...")
        if not isinstance(df_raw, pd.DataFrame):
            st.error("Error Interno: parse_historic_categorized no recibió un DataFrame.")
            return None

        # Copiar para evitar modificar el original fuera de la función
        df = df_raw.copy()

        # --- Limpieza de Nombres de Columna ---
        try:
            df.columns = [str(col).upper().strip() for col in df.columns]
            st.write(f"Debug (parse_historic): Columnas limpiadas: {df.columns.tolist()}")
        except Exception as e_col:
            st.error(f"Error limpiando nombres de columna: {e_col}")
            return None

        # --- Verificación de Columnas Requeridas ---
        required = ['CONCEPTO', 'CATEGORÍA', 'SUBCATEGORIA', 'IMPORTE', 'AÑO', 'MES', 'DIA']
        # Añadir COMERCIO si no existe para evitar errores posteriores
        if 'COMERCIO' not in df.columns:
            st.warning("Archivo histórico: Columna 'COMERCIO' no encontrada. Se creará vacía.")
            df['COMERCIO'] = ''

        missing = [col for col in required if col not in df.columns]
        if missing:
            st.error(f"Archivo histórico: Faltan columnas esenciales: {', '.join(missing)}")
            return None

        # --- Procesamiento de Columnas ---
        df_std = pd.DataFrame() # DataFrame estandarizado final

        # ** Texto (CONCEPTO, COMERCIO, CATEGORÍA, SUBCATEGORIA) **
        text_cols_mapping = {
            CONCEPTO_STD: 'CONCEPTO',
            COMERCIO_STD: 'COMERCIO', # Ya nos aseguramos que existe
            CATEGORIA_STD: 'CATEGORÍA',
            SUBCATEGORIA_STD: 'SUBCATEGORIA'
        }
        for std_col, raw_col in text_cols_mapping.items():
            st.write(f"Debug (parse_historic): Procesando '{raw_col}' -> '{std_col}'")
            if raw_col not in df.columns: # Doble chequeo por si acaso
                 st.error(f"Error Interno: Columna '{raw_col}' desapareció.")
                 return None
            try:
                # 1. Seleccionar Serie y asegurar que es string ANTES de .str
                series = df[raw_col].fillna('').astype(str)
                # 2. Aplicar métodos de string usando .str
                df_std[std_col] = series.str.lower().str.strip()
                st.write(f"Debug (parse_historic): Columna '{std_col}' procesada OK.")
            except AttributeError as ae:
                # Este error es el que reportas. Intentemos dar más info.
                st.error(f"!!! Error de Atributo procesando '{raw_col}' -> '{std_col}'. Esto usualmente significa que la conversión a string falló o la columna contiene datos inesperados.")
                # Intentar mostrar tipos de datos problemáticos
                try:
                    problematic_types = df[raw_col].apply(type).value_counts()
                    st.error(f"Tipos de datos encontrados en '{raw_col}':\n{problematic_types}")
                    # Mostrar algunos valores problemáticos si es posible
                    non_string_indices = df[raw_col].apply(lambda x: not isinstance(x, (str, type(None), float, int))).index # Buscar no strings/None/numéricos
                    if not non_string_indices.empty:
                         st.error(f"Primeros valores no textuales en '{raw_col}':\n{df.loc[non_string_indices, raw_col].head()}")
                except Exception as e_diag:
                    st.error(f"No se pudo diagnosticar el error de atributo: {e_diag}")
                return None # Detener el proceso si hay error aquí
            except Exception as e:
                st.error(f"Error inesperado procesando texto en '{raw_col}': {e}")
                st.error(traceback.format_exc())
                return None

        # ** Importe **
        st.write("Debug (parse_historic): Procesando IMPORTE")
        try:
            importe_str = df['IMPORTE'].astype(str).str.replace(',', '.', regex=False)
            df_std[IMPORTE_STD] = pd.to_numeric(importe_str, errors='coerce')
            if df_std[IMPORTE_STD].isnull().any():
                st.warning("Archivo histórico: Algunos importes no pudieron convertirse a número (ahora son NaN).")
        except Exception as e:
            st.error(f"Error procesando IMPORTE en archivo histórico: {e}")
            return None

        # ** Fechas **
        st.write("Debug (parse_historic): Procesando Fechas")
        try:
            for col in ['AÑO', 'MES', 'DIA']:
                # Convertir a numérico, errores a NaN, llenar NaN con 0, luego a int
                df_std[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
                # Validar rangos básicos (opcional pero útil)
                if col == 'MES' and not df_std[col].between(1, 12, inclusive='both').all():
                     st.warning(f"Valores de MES fuera del rango 1-12 detectados en columna '{col}'.")
                if col == 'DIA' and not df_std[col].between(1, 31, inclusive='both').all():
                     st.warning(f"Valores de DIA fuera del rango 1-31 detectados en columna '{col}'.")
        except Exception as e:
            st.error(f"Error procesando Fechas en archivo histórico: {e}")
            return None

        # ** Crear Texto para Modelo **
        st.write("Debug (parse_historic): Creando TEXTO_MODELO")
        try:
            # Asegurarse que las columnas existen antes de sumar
            if CONCEPTO_STD not in df_std: df_std[CONCEPTO_STD] = ''
            if COMERCIO_STD not in df_std: df_std[COMERCIO_STD] = ''
            df_std[TEXTO_MODELO] = df_std[CONCEPTO_STD] + ' ' + df_std[COMERCIO_STD]
            df_std[TEXTO_MODELO] = df_std[TEXTO_MODELO].str.strip()
        except Exception as e:
             st.error(f"Error creando TEXTO_MODELO: {e}")
             return None

        # ** Filtrar Filas Inválidas **
        st.write("Debug (parse_historic): Filtrando filas finales...")
        initial_rows = len(df_std)
        # Asegurar que CATEGORIA_STD existe antes de filtrar
        if CATEGORIA_STD not in df_std.columns:
             st.error("Error Interno: Falta la columna CATEGORIA_STD antes del filtrado final.")
             return None

        df_std = df_std.dropna(subset=[IMPORTE_STD, CATEGORIA_STD])
        df_std = df_std[df_std[CATEGORIA_STD] != '']
        rows_after_filter = len(df_std)
        st.write(f"Debug (parse_historic): Filas antes: {initial_rows}, después de filtrar NaN importe/categoría y categoría vacía: {rows_after_filter}")

        if df_std.empty:
            st.warning("No se encontraron filas válidas con categorías e importes en el archivo histórico después del filtrado final.")
            return pd.DataFrame() # Devolver vacío para permitir continuar, aunque entrenamiento fallará

        st.write("Debug (parse_historic): Finalizado con éxito.")
        return df_std

    except Exception as e:
        st.error(f"Error general y crítico parseando archivo histórico: {e}")
        st.error(traceback.format_exc()) # Imprimir traceback completo
        return None

# --- read_sample_csv (sin cambios respecto a la versión anterior) ---
def read_sample_csv(uploaded_file):
    """Lee un CSV de muestra y devuelve el DataFrame y sus columnas."""
    if uploaded_file is None:
        return None, []
    try:
        bytes_data = uploaded_file.getvalue()
        # Intentar detectar separador común (puede fallar)
        sniffer_content = bytes_data.decode('utf-8', errors='replace')
        sniffer = io.StringIO(sniffer_content)
        sep = ';' # Asumir punto y coma por defecto para archivos bancarios españoles
        try:
            sample_data = sniffer.read(min(1024 * 20, len(sniffer_content)))
            if sample_data:
                 dialect = pd.io.parsers.readers.csv.Sniffer().sniff(sample_data)
                 # Solo sobrescribir si el separador detectado es diferente y común (evitar caracteres raros)
                 if dialect.delimiter in [',', ';', '\t', '|']:
                      sep = dialect.delimiter
                      st.info(f"Separador detectado: '{sep}'")
                 else:
                      st.warning(f"Separador detectado ('{dialect.delimiter}') es inusual, se usará ';'.")
            else:
                 st.error("El archivo parece estar vacío o no se pudo leer muestra.")
                 return None, []
        except Exception as sniff_err:
             st.warning(f"No se pudo detectar separador (Error: {sniff_err}), asumiendo ';'.")

        # Leer con encoding flexible y separador detectado/asumido
        try:
            df = pd.read_csv(io.BytesIO(bytes_data), encoding='utf-8', sep=sep, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(bytes_data), encoding='latin1', sep=sep, low_memory=False)
        except Exception as read_err:
             st.error(f"Error leyendo CSV completo con separador '{sep}': {read_err}")
             return None, []

        # Limpiar nombres de columna
        original_columns = df.columns.tolist()
        df.columns = [str(col).strip() for col in original_columns] # Ensure names are strings
        detected_columns = df.columns.tolist()

        return df, detected_columns
    except Exception as e:
        st.error(f"Error leyendo archivo de muestra: {e}")
        return None, []


# --- extract_knowledge_std (sin cambios) ---
@st.cache_data
def extract_knowledge_std(df_std):
    knowledge = {'categorias': [], 'subcategorias': {}, 'comercios': {}}
    if df_std is None or CATEGORIA_STD not in df_std.columns or df_std.empty:
        st.warning("No hay datos estandarizados válidos para extraer conocimiento.")
        return knowledge
    knowledge['categorias'] = sorted([c for c in df_std[CATEGORIA_STD].dropna().unique() if c])
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
             comers_list = sorted([c for c in comers_series.unique() if c and c != 'n/a'])
             knowledge['comercios'][cat] = comers_list
        else:
            knowledge['comercios'][cat] = []
    return knowledge

# --- train_classifier_std (sin cambios) ---
@st.cache_resource # Cachear el modelo y vectorizador
def train_classifier_std(df_std):
    report = "Modelo no entrenado."
    model = None; vectorizer = None
    required = [TEXTO_MODELO, CATEGORIA_STD]
    if df_std is None or df_std.empty or not all(c in df_std.columns for c in required):
        st.warning("Entrenamiento: Datos estandarizados inválidos o faltan columnas.")
        return model, vectorizer, report

    df_train = df_std.dropna(subset=required)
    df_train = df_train[df_train[CATEGORIA_STD] != '']
    if df_train.empty or len(df_train[CATEGORIA_STD].unique()) < 2:
        st.warning("Entrenamiento: Datos insuficientes o menos de 2 categorías únicas.")
        return model, vectorizer, report

    try:
        X = df_train[TEXTO_MODELO]; y = df_train[CATEGORIA_STD]

        test_available = False
        if len(y.unique()) > 1 and len(y) > 5:
             try:
                 valid_idx = y.dropna().index
                 X_clean = X[valid_idx]
                 y_clean = y[valid_idx]
                 if len(y_clean.unique()) > 1 and len(y_clean) > 5 :
                    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean)
                    test_available = True
                 else:
                      X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
                      test_available = True
             except ValueError:
                 st.warning("No se pudo realizar el split estratificado, usando split simple.")
                 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                 test_available = True
        else:
             X_train, y_train = X, y
             X_test, y_test = pd.Series(dtype='str'), pd.Series(dtype='str')

        vectorizer = TfidfVectorizer()
        X_train_vec = vectorizer.fit_transform(X_train)
        model = MultinomialNB()
        model.fit(X_train_vec, y_train)

        if test_available and not X_test.empty:
            try:
                X_test_vec = vectorizer.transform(X_test)
                y_pred = model.predict(X_test_vec)
                present_labels = sorted(list(set(y_test.unique()) | set(y_pred)))
                report = classification_report(y_test, y_pred, labels=present_labels, zero_division=0)
            except Exception as report_err:
                 report = f"Modelo entrenado, error en reporte: {report_err}"
        else:
            report = "Modelo entrenado (sin evaluación detallada por falta de datos de test)."

    except Exception as e:
         report = f"Error entrenamiento: {e}"
         model, vectorizer = None, None

    return model, vectorizer, report

# --- standardize_data_with_mapping (sin cambios) ---
def standardize_data_with_mapping(df_raw, mapping):
    """Aplica el mapeo guardado para estandarizar un DataFrame nuevo."""
    try:
        df_std = pd.DataFrame()
        df_raw.columns = [str(col).strip() for col in df_raw.columns]
        original_columns = df_raw.columns.tolist()

        temp_std_data = {}
        found_essential = {std_col: False for std_col in MANDATORY_STD_COLS if std_col != FECHA_STD}
        found_optional = {std_col: False for std_col in OPTIONAL_STD_COLS}
        source_cols_used = []

        for std_col, source_col in mapping['columns'].items():
             if source_col in original_columns:
                  temp_std_data[std_col] = df_raw[source_col]
                  source_cols_used.append(source_col)
                  if std_col in found_essential: found_essential[std_col] = True
                  if std_col in found_optional: found_optional[std_col] = True
             elif std_col in MANDATORY_STD_COLS and std_col != FECHA_STD:
                  st.error(f"¡Error Crítico! La columna esencial mapeada '{source_col}' para '{std_col}' no existe en este archivo.")
                  return None
             elif std_col in OPTIONAL_STD_COLS or std_col == FECHA_STD:
                  st.info(f"Columna opcional/fecha mapeada '{source_col}' para '{std_col}' no encontrada. Se omitirá.")

        df_std = pd.DataFrame(temp_std_data)
        missing_essential_mapped = [k for k, v in found_essential.items() if not v]
        if missing_essential_mapped:
             st.error(f"Faltan columnas esenciales en el archivo que estaban mapeadas: {missing_essential_mapped}")
             return None

        fecha_col_source = mapping['columns'].get(FECHA_STD)
        año_col_source = mapping['columns'].get(AÑO_STD)
        mes_col_source = mapping['columns'].get(MES_STD)
        dia_col_source = mapping['columns'].get(DIA_STD)
        date_processed_ok = False

        if fecha_col_source and fecha_col_source in source_cols_used:
             date_format = mapping.get('date_format')
             if not date_format:
                  st.error(f"Error: Se mapeó la columna de fecha '{fecha_col_source}' pero falta el formato de fecha en el mapeo.")
                  return None
             try:
                date_series = df_std[FECHA_STD].astype(str).str.strip()
                valid_dates = pd.to_datetime(date_series, format=date_format, errors='coerce')
                if valid_dates.isnull().all():
                      st.error(f"Ninguna fecha en '{fecha_col_source}' coincide con el formato '{date_format}'. Verifique el formato y los datos.")
                      return None
                elif valid_dates.isnull().any():
                      st.warning(f"Algunas fechas en '{fecha_col_source}' no coinciden con formato '{date_format}' o son inválidas.")

                df_std[AÑO_STD] = valid_dates.dt.year.fillna(0).astype(int)
                df_std[MES_STD] = valid_dates.dt.month.fillna(0).astype(int)
                df_std[DIA_STD] = valid_dates.dt.day.fillna(0).astype(int)
                df_std = df_std.drop(columns=[FECHA_STD])
                date_processed_ok = True
             except ValueError as ve:
                 st.error(f"Error de formato de fecha para '{fecha_col_source}' con formato '{date_format}'. ¿Es correcto el formato? Error: {ve}")
                 return None
             except Exception as e_date:
                 st.error(f"Error inesperado procesando fecha única '{fecha_col_source}': {e_date}")
                 return None

        elif año_col_source and mes_col_source and dia_col_source:
            all_date_cols_found = True
            for col_std, col_source in zip([AÑO_STD, MES_STD, DIA_STD], [año_col_source, mes_col_source, dia_col_source]):
                 if col_source in source_cols_used:
                     try:
                         df_std[col_std] = pd.to_numeric(df_std[col_std], errors='coerce').fillna(0).astype(int)
                     except Exception as e_num:
                          st.error(f"Error convirtiendo columna de fecha '{col_source}' a número: {e_num}")
                          all_date_cols_found = False; break
                 else:
                      st.error(f"Error: La columna de fecha mapeada '{col_source}' para '{col_std}' no se encontró en los datos procesados.")
                      all_date_cols_found = False; break
            if all_date_cols_found: date_processed_ok = True

        if not date_processed_ok:
             st.error("Error crítico: No se pudo procesar la información de fecha según el mapeo.")
             return None

        importe_col_source = mapping['columns'].get(IMPORTE_STD)
        if IMPORTE_STD in df_std.columns:
            try:
                importe_str = df_std[IMPORTE_STD].fillna('0').astype(str)
                thousands_sep = mapping.get('thousands_sep')
                if thousands_sep: importe_str = importe_str.str.replace(thousands_sep, '', regex=False)
                decimal_sep = mapping.get('decimal_sep', ',')
                importe_str = importe_str.str.replace(decimal_sep, '.', regex=False)
                df_std[IMPORTE_STD] = pd.to_numeric(importe_str, errors='coerce')
                if df_std[IMPORTE_STD].isnull().any():
                     st.warning(f"Algunos valores de importe en '{importe_col_source}' no pudieron convertirse tras aplicar separadores.")
            except Exception as e_imp:
                 st.error(f"Error procesando importe desde columna '{importe_col_source}': {e_imp}")
                 return None
        elif importe_col_source:
            st.error(f"Error interno: Columna de importe '{importe_col_source}' mapeada pero no encontrada.")
            return None
        else:
             st.error("Error: La columna IMPORTE_STD no fue mapeada.")
             return None

        for col_std in [CONCEPTO_STD, COMERCIO_STD]:
            if col_std in df_std.columns:
                df_std[col_std] = df_std[col_std].fillna('').astype(str).str.lower().str.strip()
            elif col_std == COMERCIO_STD:
                 df_std[COMERCIO_STD] = ''

        if CONCEPTO_STD not in df_std.columns: df_std[CONCEPTO_STD] = ''
        if COMERCIO_STD not in df_std.columns: df_std[COMERCIO_STD] = ''
        df_std[TEXTO_MODELO] = df_std[CONCEPTO_STD] + ' ' + df_std[COMERCIO_STD]
        df_std[TEXTO_MODELO] = df_std[TEXTO_MODELO].str.strip()

        mapped_source_cols = list(mapping['columns'].values())
        original_cols_to_keep = [c for c in original_columns if c not in mapped_source_cols]
        for col in original_cols_to_keep:
              target_col_name = f"ORIG_{col}"
              if target_col_name not in df_std.columns:
                   df_std[target_col_name] = df_raw[col]
              else:
                   suffix = 1
                   while f"ORIG_{col}_{suffix}" in df_std.columns: suffix += 1
                   df_std[f"ORIG_{col}_{suffix}"] = df_raw[col]

        df_std = df_std.dropna(subset=[IMPORTE_STD, TEXTO_MODELO])
        df_std = df_std[df_std[TEXTO_MODELO]!='']

        return df_std

    except Exception as e:
        st.error(f"Error inesperado aplicando mapeo '{mapping.get('bank_name', 'Desconocido')}': {e}")
        st.error(traceback.format_exc())
        return None

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🏦 Categorizador Bancario Multi-Formato v2")

# --- Fase 1: Entrenamiento Inicial ---
with st.expander("Fase 1: Entrenar Modelo con Datos Históricos Categorizados", expanded=True):
    st.write("Sube tu archivo CSV histórico (ej: `Gastos.csv`) que ya contiene las categorías y subcategorías asignadas. Este archivo entrena el modelo base.")

    uploaded_historic_file = st.file_uploader("Cargar Archivo Histórico Categorizado (.csv)", type="csv", key="historic_uploader_f1")

    if uploaded_historic_file:
        # Mostrar botón solo si hay archivo
        if st.button("🧠 Entrenar Modelo y Aprender Conocimiento Inicial", key="train_historic_f1"):
            with st.spinner("Procesando archivo histórico y entrenando..."):
                # Leer el archivo raw ANTES de pasarlo a la función de parseo
                df_raw_hist, _ = read_sample_csv(uploaded_historic_file)
                if df_raw_hist is not None:
                    df_std_hist = parse_historic_categorized(df_raw_hist.copy()) # Pasar copia
                    # Continuar solo si el parseo fue exitoso y devolvió datos
                    if df_std_hist is not None and not df_std_hist.empty:
                        st.success("Archivo histórico parseado.")
                        # Extraer conocimiento
                        st.session_state.knowledge = extract_knowledge_std(df_std_hist)
                        st.sidebar.success("Conocimiento Inicial Extraído")
                        with st.sidebar.expander("Ver Categorías Aprendidas"): st.write(st.session_state.knowledge['categorias'])

                        # Entrenar modelo
                        model, vectorizer, report = train_classifier_std(df_std_hist)
                        if model and vectorizer:
                            st.session_state.model = model
                            st.session_state.vectorizer = vectorizer
                            st.session_state.model_trained = True
                            st.session_state.training_report = report
                            st.success("¡Modelo entrenado exitosamente!")
                            st.sidebar.subheader("Evaluación Modelo Base")
                            with st.sidebar.expander("Ver Informe"): st.text(st.session_state.training_report)
                        else: # Fallo el entrenamiento
                            st.error("Fallo en el entrenamiento del modelo base.")
                            st.session_state.model_trained = False
                            st.session_state.training_report = report # Guardar mensaje de error
                            st.sidebar.error("Entrenamiento Fallido")
                            st.sidebar.text(st.session_state.training_report)
                    # El else aquí significa que parse_historic falló o devolvió vacío
                    else:
                        st.error("No se pudo parsear el archivo histórico o no contenía datos válidos para entrenar.")
                        st.session_state.model_trained = False # Asegurar que el estado es correcto
                # El else aquí significa que read_sample_csv falló
                else:
                     st.error("No se pudo leer el archivo histórico.")
                     st.session_state.model_trained = False

# --- Fase 2: Aprendizaje de Formatos Bancarios ---
with st.expander("Fase 2: Aprender Formatos de Archivos Bancarios"):
    st.write("Sube un archivo CSV de ejemplo (sin categorizar) para cada banco cuyo formato quieras que la aplicación aprenda. Luego, define cómo mapear sus columnas a las estándar.")

    bank_options = ["SANTANDER", "EVO", "WIZINK", "AMEX"] # Añade más bancos aquí
    selected_bank_learn = st.selectbox("Selecciona Banco para Aprender Formato:", bank_options, key="bank_learn_f2")

    uploaded_sample_file = st.file_uploader(f"Cargar archivo CSV de ejemplo de {selected_bank_learn}", type="csv", key="sample_uploader_f2")

    if uploaded_sample_file:
        df_sample, detected_columns = read_sample_csv(uploaded_sample_file)

        if df_sample is not None:
            st.write(f"Columnas detectadas en el archivo de {selected_bank_learn}:")
            st.code(f"{detected_columns}") # Usar code para mejor visualización
            st.dataframe(df_sample.head(3)) # Mostrar solo unas pocas filas

            st.subheader("Mapeo de Columnas")
            st.write(f"Selecciona qué columna del archivo de {selected_bank_learn} corresponde a cada campo estándar necesario.")

            # Usar el mapeo guardado como valor inicial si existe
            saved_mapping = st.session_state.bank_mappings.get(selected_bank_learn, {'columns': {}})
            current_mapping_ui = {'columns': {}} # Mapeo temporal para esta sesión de UI

            cols_with_none = [None] + detected_columns

            # --- Mapeos Esenciales ---
            st.markdown("**Campos Esenciales:**")
            current_mapping_ui['columns'][CONCEPTO_STD] = st.selectbox(
                f"`{CONCEPTO_STD}` (Descripción)", cols_with_none,
                index=cols_with_none.index(saved_mapping['columns'].get(CONCEPTO_STD)) if saved_mapping['columns'].get(CONCEPTO_STD) in cols_with_none else 0,
                key=f"map_{CONCEPTO_STD}_{selected_bank_learn}"
            )
            current_mapping_ui['columns'][IMPORTE_STD] = st.selectbox(
                f"`{IMPORTE_STD}` (Valor)", cols_with_none,
                index=cols_with_none.index(saved_mapping['columns'].get(IMPORTE_STD)) if saved_mapping['columns'].get(IMPORTE_STD) in cols_with_none else 0,
                key=f"map_{IMPORTE_STD}_{selected_bank_learn}"
            )

            # --- Mapeo de Fecha ---
            st.markdown("**Campo de Fecha (elige una opción):**")
            # Determinar estado inicial del checkbox basado en el mapeo guardado
            is_single_date_saved = FECHA_STD in saved_mapping['columns']
            map_single_date = st.checkbox("La fecha está en una sola columna", value=is_single_date_saved, key=f"map_single_date_{selected_bank_learn}")

            if map_single_date:
                current_mapping_ui['columns'][FECHA_STD] = st.selectbox(
                    f"`{FECHA_STD}` (Columna Única)", cols_with_none,
                    index=cols_with_none.index(saved_mapping['columns'].get(FECHA_STD)) if saved_mapping['columns'].get(FECHA_STD) in cols_with_none else 0,
                    key=f"map_{FECHA_STD}_{selected_bank_learn}"
                )
                date_format_guess = st.text_input(
                    "Formato fecha (ej: %d/%m/%Y, %Y-%m-%d, %d-%b-%Y)",
                    value=saved_mapping.get('date_format', ''),
                    help="Usa los códigos de formato de Python (ver documentación de `strftime`)",
                    key=f"map_date_format_{selected_bank_learn}"
                )
                if date_format_guess: current_mapping_ui['date_format'] = date_format_guess.strip()
                # Eliminar mapeos AÑO/MES/DIA si se cambia a fecha única
                current_mapping_ui['columns'].pop(AÑO_STD, None); current_mapping_ui['columns'].pop(MES_STD, None); current_mapping_ui['columns'].pop(DIA_STD, None)
            else:
                current_mapping_ui['columns'][AÑO_STD] = st.selectbox(
                    f"`{AÑO_STD}`", cols_with_none,
                    index=cols_with_none.index(saved_mapping['columns'].get(AÑO_STD)) if saved_mapping['columns'].get(AÑO_STD) in cols_with_none else 0,
                    key=f"map_{AÑO_STD}_{selected_bank_learn}"
                )
                current_mapping_ui['columns'][MES_STD] = st.selectbox(
                    f"`{MES_STD}`", cols_with_none,
                    index=cols_with_none.index(saved_mapping['columns'].get(MES_STD)) if saved_mapping['columns'].get(MES_STD) in cols_with_none else 0,
                    key=f"map_{MES_STD}_{selected_bank_learn}"
                 )
                current_mapping_ui['columns'][DIA_STD] = st.selectbox(
                    f"`{DIA_STD}`", cols_with_none,
                     index=cols_with_none.index(saved_mapping['columns'].get(DIA_STD)) if saved_mapping['columns'].get(DIA_STD) in cols_with_none else 0,
                     key=f"map_{DIA_STD}_{selected_bank_learn}"
                )
                # Eliminar mapeo FECHA_STD si se cambia a A/M/D
                current_mapping_ui['columns'].pop(FECHA_STD, None); current_mapping_ui.pop('date_format', None)


            # --- Mapeos Opcionales ---
            st.markdown("**Campos Opcionales:**")
            current_mapping_ui['columns'][COMERCIO_STD] = st.selectbox(
                 f"`{COMERCIO_STD}` (Comercio/Entidad)", cols_with_none,
                 index=cols_with_none.index(saved_mapping['columns'].get(COMERCIO_STD)) if saved_mapping['columns'].get(COMERCIO_STD) in cols_with_none else 0,
                 key=f"map_{COMERCIO_STD}_{selected_bank_learn}"
            )

            # --- Configuración Importe ---
            st.markdown("**Configuración de Importe:**")
            current_mapping_ui['decimal_sep'] = st.text_input("Separador Decimal (ej: , ó .)", value=saved_mapping.get('decimal_sep', ','), key=f"map_decimal_{selected_bank_learn}")
            current_mapping_ui['thousands_sep'] = st.text_input("Separador de Miles (si aplica, ej: . ó ,)", value=saved_mapping.get('thousands_sep', ''), key=f"map_thousands_{selected_bank_learn}")


            # --- Validación y Guardado ---
            # Filtrar mapeos None antes de validar
            final_mapping_cols = {std: src for std, src in current_mapping_ui['columns'].items() if src is not None}
            valid_mapping = True
            # Chequear obligatorios (Concepto, Importe)
            if not final_mapping_cols.get(CONCEPTO_STD):
                 st.error("Falta mapear CONCEPTO_STD.")
                 valid_mapping = False
            if not final_mapping_cols.get(IMPORTE_STD):
                 st.error("Falta mapear IMPORTE_STD.")
                 valid_mapping = False
            # Chequear fecha
            if map_single_date:
                 if not final_mapping_cols.get(FECHA_STD):
                      st.error("Falta mapear FECHA_STD (columna única).")
                      valid_mapping = False
                 elif not current_mapping_ui.get('date_format'):
                      st.error("Falta especificar el formato de fecha.")
                      valid_mapping = False
            else: # Columnas A/M/D
                 if not all(final_mapping_cols.get(d) for d in [AÑO_STD, MES_STD, DIA_STD]):
                      st.error("Faltan mapeos para AÑO, MES o DIA.")
                      valid_mapping = False

            if valid_mapping:
                 # Crear el diccionario final limpio para guardar
                 mapping_to_save = {
                      'bank_name': selected_bank_learn,
                      'columns': final_mapping_cols, # Solo los mapeos válidos
                      'decimal_sep': current_mapping_ui.get('decimal_sep', ',').strip(),
                      # Guardar separador de miles solo si no está vacío
                      'thousands_sep': current_mapping_ui.get('thousands_sep', '').strip() or None,
                 }
                 if map_single_date and current_mapping_ui.get('date_format'):
                      mapping_to_save['date_format'] = current_mapping_ui['date_format']

                 if st.button(f"💾 Guardar Mapeo para {selected_bank_learn}", key="save_mapping_f2"):
                      st.session_state.bank_mappings[selected_bank_learn] = mapping_to_save
                      st.success(f"¡Mapeo para {selected_bank_learn} guardado!")
                      st.json(mapping_to_save)
            else:
                 st.warning("Revisa los errores en el mapeo antes de guardar.")


# Mostrar mapeos guardados en la barra lateral
st.sidebar.divider()
st.sidebar.subheader("Mapeos Bancarios Guardados")
if st.session_state.bank_mappings:
    try:
        # Usar json.dumps para asegurar serialización correcta antes de st.json
        st.sidebar.json(json.dumps(st.session_state.bank_mappings, indent=2), expanded=False)
    except Exception as e_json:
         st.sidebar.error(f"No se pueden mostrar los mapeos: {e_json}")
         st.sidebar.write(st.session_state.bank_mappings) # Fallback a escritura simple
else:
    st.sidebar.info("Aún no se han guardado mapeos.")


# --- Fase 3: Categorización ---
with st.expander("Fase 3: Categorizar Nuevos Archivos", expanded=True):
    if not st.session_state.model_trained:
        st.warning("⚠️ Modelo no entrenado (Ver Fase 1).")
    elif not st.session_state.bank_mappings:
        st.warning("⚠️ No se han aprendido formatos bancarios (Ver Fase 2).")
    else:
        st.write("Selecciona el banco y sube el archivo CSV **sin categorizar** que deseas procesar.")
        available_banks_for_pred = list(st.session_state.bank_mappings.keys())
        if not available_banks_for_pred:
             st.warning("No hay mapeos guardados para seleccionar un banco.")
        else:
            selected_bank_predict = st.selectbox("Banco del Nuevo Archivo:", available_banks_for_pred, key="bank_predict_f3")

            uploaded_final_file = st.file_uploader(f"Cargar archivo CSV NUEVO de {selected_bank_predict}", type="csv", key="final_uploader_f3")

            if uploaded_final_file and selected_bank_predict:
                mapping_to_use = st.session_state.bank_mappings.get(selected_bank_predict)
                if not mapping_to_use:
                     st.error(f"Error interno: No se encontró el mapeo para {selected_bank_predict} aunque debería existir.")
                else:
                     st.write("Procesando archivo nuevo...")
                     with st.spinner(f"Estandarizando datos con mapeo de {selected_bank_predict}..."):
                          df_raw_new, _ = read_sample_csv(uploaded_final_file)
                          df_std_new = None # Inicializar
                          if df_raw_new is not None:
                              df_std_new = standardize_data_with_mapping(df_raw_new.copy(), mapping_to_use)
                          else:
                              st.error(f"No se pudo leer el archivo: {uploaded_final_file.name}")

                     # Continuar solo si la estandarización fue exitosa
                     if df_std_new is not None and not df_std_new.empty:
                          st.success("Datos estandarizados.")
                          with st.spinner("Aplicando modelo de categorización..."):
                              try:
                                   if TEXTO_MODELO not in df_std_new.columns:
                                       st.error(f"Error crítico: La columna {TEXTO_MODELO} no se generó durante la estandarización.")
                                   else:
                                        df_pred = df_std_new.dropna(subset=[TEXTO_MODELO]).copy() # Trabajar con copia limpia
                                        if not df_pred.empty:
                                             X_new_vec = st.session_state.vectorizer.transform(df_pred[TEXTO_MODELO])
                                             predictions = st.session_state.model.predict(X_new_vec)
                                             df_pred[CATEGORIA_PREDICHA] = predictions.astype(str).str.capitalize()

                                             st.subheader("📊 Resultados de la Categorización")
                                             # Seleccionar y reordenar columnas para mostrar
                                             display_cols = [CATEGORIA_PREDICHA, CONCEPTO_STD, IMPORTE_STD, AÑO_STD, MES_STD, DIA_STD]
                                             # Insertar COMERCIO si existe después de estandarizar
                                             if COMERCIO_STD in df_pred.columns:
                                                  display_cols.insert(2, COMERCIO_STD)
                                             # Añadir columnas originales relevantes
                                             orig_cols = [c for c in df_pred.columns if c.startswith('ORIG_')]
                                             display_cols.extend(orig_cols)

                                             # Asegurarse que todas las columnas existen antes de mostrarlas
                                             final_display_cols = [col for col in display_cols if col in df_pred.columns]
                                             st.dataframe(df_pred[final_display_cols])

                                             # Descarga
                                             csv_output = df_pred.to_csv(index=False, sep=';', decimal=',').encode('utf-8')
                                             st.download_button(
                                                  label=f"📥 Descargar '{uploaded_final_file.name}' Categorizado",
                                                  data=csv_output,
                                                  file_name=f"categorizado_{uploaded_final_file.name}",
                                                  mime='text/csv',
                                                  key=f"download_final_{uploaded_final_file.name}"
                                             )
                                        else:
                                             st.warning("No quedaron filas válidas para categorizar después de limpiar NAs en datos de texto.")

                              except Exception as e:
                                   st.error(f"Error durante la predicción de categorías: {e}")
                                   st.error(f"Vectorizador: {st.session_state.vectorizer}")
                                   st.error(f"Texto (head): {df_pred[TEXTO_MODELO].head().tolist() if TEXTO_MODELO in df_pred else 'N/A'}")

                     elif df_std_new is not None and df_std_new.empty:
                         st.warning("El archivo no contenía datos válidos después de la estandarización.")
                     else: # df_std_new is None (fallo en standardize_data_with_mapping)
                         st.error("Fallo en la estandarización del archivo nuevo usando el mapeo guardado.")


# Sidebar Info
st.sidebar.divider()
st.sidebar.header("Acerca de")
st.sidebar.info(
    "1. Entrena con tu CSV histórico. "
    "2. Enseña a la app los formatos de tus bancos. "
    "3. Sube nuevos archivos para categorizarlos."
)
