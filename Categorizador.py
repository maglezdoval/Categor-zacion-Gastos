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
        st.write("Debug: Entrando en parse_historic_categorized") # Debug
        # Asegurar que los nombres de columna son strings antes de operar
        df_raw.columns = [str(col).upper().strip() for col in df_raw.columns]
        st.write(f"Debug: Columnas raw limpiadas: {df_raw.columns.tolist()}") # Debug

        required = ['CONCEPTO', 'CATEGORÍA', 'SUBCATEGORIA', 'IMPORTE', 'AÑO', 'MES', 'DIA']
        # Asegurarse que COMERCIO se maneja aunque no sea estrictamente 'required' aquí
        if 'COMERCIO' not in df_raw.columns:
             st.warning("Archivo histórico: Columna 'COMERCIO' no encontrada. Se continuará, pero puede afectar la precisión.")
             df_raw['COMERCIO'] = '' # Añadir columna vacía para evitar errores posteriores

        if not all(col in df_raw.columns for col in required):
            missing = [col for col in required if col not in df_raw.columns]
            st.error(f"Archivo histórico: Faltan columnas esenciales: {', '.join(missing)}")
            return None

        df_std = pd.DataFrame()

        # Procesar columnas de texto una por una con chequeos
        text_cols_mapping = {
            CONCEPTO_STD: 'CONCEPTO',
            COMERCIO_STD: 'COMERCIO',
            CATEGORIA_STD: 'CATEGORÍA',
            SUBCATEGORIA_STD: 'SUBCATEGORIA'
        }

        for std_col, raw_col in text_cols_mapping.items():
            st.write(f"Debug: Procesando columna de texto '{raw_col}' -> '{std_col}'")
            if raw_col in df_raw.columns:
                try:
                    # Paso 1: Seleccionar Serie y rellenar NaNs
                    series = df_raw[raw_col].fillna('')
                    # Paso 2: Convertir a string explícitamente
                    series_str = series.astype(str)
                    # Paso 3: Aplicar métodos de string con .str
                    series_lower = series_str.str.lower()
                    series_stripped = series_lower.str.strip()
                    df_std[std_col] = series_stripped
                    st.write(f"Debug: Columna '{raw_col}' procesada.")
                except AttributeError as ae:
                     st.error(f"Error de atributo procesando columna '{raw_col}'. ¿Contiene datos no textuales inesperados? Error: {ae}")
                     st.error(f"Primeros valores problemáticos:\n{df_raw[raw_col][~df_raw[raw_col].apply(lambda x: isinstance(x, (str, int, float, type(None))))].head()}")
                     return None
                except Exception as e:
                     st.error(f"Error inesperado procesando columna de texto '{raw_col}': {e}")
                     st.error(traceback.format_exc())
                     return None
            elif std_col == COMERCIO_STD: # Si la columna COMERCIO original no existe
                 df_std[COMERCIO_STD] = '' # Crearla vacía
                 st.write(f"Debug: Columna '{raw_col}' no encontrada, creada vacía como '{std_col}'.")
            # Si falta otra columna (Categoría, etc.), ya debería haber fallado en el check 'required'


        # Procesar Importe
        st.write("Debug: Procesando IMPORTE")
        try:
            df_std[IMPORTE_STD] = pd.to_numeric(df_raw['IMPORTE'].astype(str).str.replace(',', '.', regex=False), errors='coerce')
            if df_std[IMPORTE_STD].isnull().any():
                 st.warning("Archivo histórico: Algunos importes no pudieron convertirse a número.")
        except Exception as e:
            st.error(f"Error procesando IMPORTE en archivo histórico: {e}")
            return None

        # Procesar Fechas
        st.write("Debug: Procesando Fechas")
        try:
            for col in ['AÑO', 'MES', 'DIA']:
                df_std[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(0).astype(int)
        except Exception as e:
            st.error(f"Error procesando Fechas en archivo histórico: {e}")
            return None

        # Crear columna de texto para entrenamiento
        st.write("Debug: Creando TEXTO_MODELO")
        df_std[TEXTO_MODELO] = df_std[CONCEPTO_STD] + ' ' + df_std[COMERCIO_STD]
        df_std[TEXTO_MODELO] = df_std[TEXTO_MODELO].str.strip()

        # Filtrar filas sin categoría o importe válido
        st.write("Debug: Filtrando filas inválidas")
        initial_rows = len(df_std)
        df_std = df_std.dropna(subset=[IMPORTE_STD, CATEGORIA_STD]) # Asegurar que categoría no sea NaN
        df_std = df_std[df_std[CATEGORIA_STD] != ''] # Asegurar que categoría no sea vacía
        st.write(f"Debug: Filas antes: {initial_rows}, después de filtrar importe/categoría: {len(df_std)}")

        if df_std.empty:
            st.warning("No se encontraron filas válidas con categorías e importes en el archivo histórico después del filtrado.")
            # Devolvemos DF vacío para no romper la lógica posterior, pero el entrenamiento fallará
            return pd.DataFrame(columns=df_std.columns)

        st.write("Debug: parse_historic_categorized finalizado.")
        return df_std

    except Exception as e:
        st.error(f"Error general parseando archivo histórico: {e}")
        st.error(traceback.format_exc()) # Imprimir traceback completo
        return None

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

# Funciones de ML (adaptadas para usar columnas _STD)
@st.cache_data
def extract_knowledge_std(df_std):
    # Extrae conocimiento del DataFrame estandarizado
    knowledge = {'categorias': [], 'subcategorias': {}, 'comercios': {}}
    if df_std is None or CATEGORIA_STD not in df_std.columns or df_std.empty:
        st.warning("No hay datos estandarizados válidos para extraer conocimiento.")
        return knowledge
    knowledge['categorias'] = sorted([c for c in df_std[CATEGORIA_STD].dropna().unique() if c])
    for cat in knowledge['categorias']:
        # Subcategorías
        subcat_col = SUBCATEGORIA_STD
        if subcat_col in df_std.columns:
             subcats_series = df_std.loc[df_std[CATEGORIA_STD] == cat, subcat_col].dropna()
             knowledge['subcategorias'][cat] = sorted([s for s in subcats_series.unique() if s])
        else:
             knowledge['subcategorias'][cat] = []
        # Comercios
        comercio_col = COMERCIO_STD
        if comercio_col in df_std.columns:
             comers_series = df_std.loc[df_std[CATEGORIA_STD] == cat, comercio_col].dropna()
             comers_list = sorted([c for c in comers_series.unique() if c and c != 'n/a'])
             knowledge['comercios'][cat] = comers_list
        else:
            knowledge['comercios'][cat] = [] # Asegurar que existe la llave
    return knowledge


@st.cache_resource # Cachear el modelo y vectorizador
def train_classifier_std(df_std):
    # Entrena el clasificador usando datos estandarizados
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

        # Split (manejar pocos datos)
        test_available = False
        if len(y.unique()) > 1 and len(y) > 5: # Umbral simple para split
             try:
                 # Asegurar que y no contenga NaNs antes de estratificar
                 valid_idx = y.dropna().index
                 X_clean = X[valid_idx]
                 y_clean = y[valid_idx]
                 if len(y_clean.unique()) > 1 and len(y_clean) > 5 : # Re-chequear después de limpiar y
                    X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean)
                    test_available = True
                 else: # No estratificar si aún no cumple condiciones
                      X_train, X_test, y_train, y_test = train_test_split(X_clean, y_clean, test_size=0.2, random_state=42)
                      test_available = True

             except ValueError: # Si falla el split por alguna razón
                 st.warning("No se pudo realizar el split estratificado, usando split simple.")
                 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                 test_available = True # Se intentó split
        else: # No hacer split
             X_train, y_train = X, y
             X_test, y_test = pd.Series(dtype='str'), pd.Series(dtype='str') # Vacío para evitar error

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
                 # Obtener etiquetas presentes en ambos y_test y y_pred para evitar errores
                present_labels = sorted(list(set(y_test.unique()) | set(y_pred)))
                report = classification_report(y_test, y_pred, labels=present_labels, zero_division=0)
            except Exception as report_err:
                 report = f"Modelo entrenado, error en reporte: {report_err}"
        else:
            report = "Modelo entrenado (sin evaluación detallada por falta de datos de test)."

    except Exception as e:
         report = f"Error entrenamiento: {e}"
         model, vectorizer = None, None # Asegurar que no se devuelvan objetos inválidos

    return model, vectorizer, report

def standardize_data_with_mapping(df_raw, mapping):
    """Aplica el mapeo guardado para estandarizar un DataFrame nuevo."""
    try:
        df_std = pd.DataFrame()
        # Asegurar nombres de columna limpios en el DF raw
        df_raw.columns = [str(col).strip() for col in df_raw.columns]
        original_columns = df_raw.columns.tolist() # Lista de columnas originales limpias


        # 1. Mapear columnas básicas usando el mapping['columns']
        # Crear un df temporal solo con las columnas mapeadas
        temp_std_data = {}
        found_essential = {std_col: False for std_col in MANDATORY_STD_COLS if std_col != FECHA_STD} # Track esenciales (excluye fecha por ahora)
        found_optional = {std_col: False for std_col in OPTIONAL_STD_COLS}
        source_cols_used = [] # Track columnas fuente usadas en el mapeo

        for std_col, source_col in mapping['columns'].items():
             if source_col in original_columns:
                  temp_std_data[std_col] = df_raw[source_col]
                  source_cols_used.append(source_col)
                  if std_col in found_essential: found_essential[std_col] = True
                  if std_col in found_optional: found_optional[std_col] = True
             elif std_col in MANDATORY_STD_COLS and std_col != FECHA_STD: # Esencial (excluye FECHA) no encontrado
                  st.error(f"¡Error Crítico! La columna esencial mapeada '{source_col}' para '{std_col}' no existe en este archivo.")
                  return None
             elif std_col in OPTIONAL_STD_COLS or std_col == FECHA_STD: # Opcional o Fecha no encontrado
                  st.info(f"Columna opcional/fecha mapeada '{source_col}' para '{std_col}' no encontrada. Se omitirá o creará vacía.")
                  # No añadirla a temp_std_data si no se encuentra

        # Crear DataFrame estandarizado desde el diccionario temporal
        df_std = pd.DataFrame(temp_std_data)

        # Verificar si faltan esenciales después del mapeo inicial
        missing_essential_mapped = [k for k, v in found_essential.items() if not v]
        if missing_essential_mapped:
             st.error(f"Faltan columnas esenciales en el archivo que estaban mapeadas: {missing_essential_mapped}")
             return None


        # 2. Manejar Fecha
        fecha_col_source = mapping['columns'].get(FECHA_STD)
        año_col_source = mapping['columns'].get(AÑO_STD)
        mes_col_source = mapping['columns'].get(MES_STD)
        dia_col_source = mapping['columns'].get(DIA_STD)
        date_processed_ok = False

        if fecha_col_source and fecha_col_source in source_cols_used: # Opción 1: Columna única de fecha (ya está en df_std)
             date_format = mapping.get('date_format')
             if not date_format:
                  st.error(f"Error: Mapeo para '{FECHA_STD}' existe pero falta el formato de fecha.")
                  return None
             try:
                # Limpiar antes de convertir
                date_series = df_std[FECHA_STD].astype(str).str.strip()
                valid_dates = pd.to_datetime(date_series, format=date_format, errors='coerce')
                if valid_dates.isnull().all(): # Si *todas* las fechas fallan
                      st.error(f"Ninguna fecha en '{fecha_col_source}' coincide con el formato '{date_format}'. Verifique el formato y los datos.")
                      return None
                elif valid_dates.isnull().any():
                      st.warning(f"Algunas fechas en '{fecha_col_source}' no coinciden con formato '{date_format}' o son inválidas.")

                df_std[AÑO_STD] = valid_dates.dt.year.fillna(0).astype(int)
                df_std[MES_STD] = valid_dates.dt.month.fillna(0).astype(int)
                df_std[DIA_STD] = valid_dates.dt.day.fillna(0).astype(int)
                df_std = df_std.drop(columns=[FECHA_STD]) # Eliminar la columna estándar original
                date_processed_ok = True
             except ValueError as ve:
                 st.error(f"Error de formato de fecha para '{fecha_col_source}' con formato '{date_format}'. ¿Es correcto el formato? Error: {ve}")
                 return None
             except Exception as e_date:
                 st.error(f"Error inesperado procesando fecha única '{fecha_col_source}': {e_date}")
                 return None

        elif año_col_source and mes_col_source and dia_col_source: # Opción 2: Columnas separadas
            all_date_cols_found = True
            for col_std, col_source in zip([AÑO_STD, MES_STD, DIA_STD], [año_col_source, mes_col_source, dia_col_source]):
                 if col_source in source_cols_used: # Ya debería estar en df_std
                     try:
                         # Forzar a numérico, errores a NaN, llenar NaN con 0, luego a int
                         df_std[col_std] = pd.to_numeric(df_std[col_std], errors='coerce').fillna(0).astype(int)
                     except Exception as e_num:
                          st.error(f"Error convirtiendo columna de fecha '{col_source}' a número: {e_num}")
                          all_date_cols_found = False; break # Salir del bucle si una falla
                 else: # Mapeada pero no encontrada (error raro, pero chequear)
                      st.error(f"Error: La columna de fecha mapeada '{col_source}' para '{col_std}' no se encontró en los datos procesados.")
                      all_date_cols_found = False; break
            if all_date_cols_found: date_processed_ok = True

        if not date_processed_ok:
             st.error("Error crítico: No se pudo procesar la información de fecha según el mapeo.")
             return None

        # 3. Limpiar Importe
        importe_col_source = mapping['columns'].get(IMPORTE_STD) # El nombre original
        if IMPORTE_STD in df_std.columns: # Usar la columna estándar que ya existe en df_std
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
        else: # No debería pasar si la validación inicial funcionó
             st.error("Error interno: Falta la columna IMPORTE_STD mapeada.")
             return None


        # 4. Limpiar Concepto y Comercio (columnas estándar ya presentes en df_std)
        for col_std in [CONCEPTO_STD, COMERCIO_STD]:
            if col_std in df_std.columns:
                df_std[col_std] = df_std[col_std].fillna('').astype(str).str.lower().str.strip()
            elif col_std == COMERCIO_STD: # Si COMERCIO_STD no se mapeó/encontró
                 df_std[COMERCIO_STD] = ''


        # 5. Crear Texto para Modelo
        # Asegurar que ambas columnas existen antes de concatenar
        if CONCEPTO_STD not in df_std.columns: df_std[CONCEPTO_STD] = ''
        if COMERCIO_STD not in df_std.columns: df_std[COMERCIO_STD] = ''
        df_std[TEXTO_MODELO] = df_std[CONCEPTO_STD] + ' ' + df_std[COMERCIO_STD]
        df_std[TEXTO_MODELO] = df_std[TEXTO_MODELO].str.strip()

        # 6. Mantener columnas originales no usadas en el mapeo estándar
        # Comparar original_columns con source_cols_used
        original_cols_to_keep = [c for c in original_columns if c not in source_cols_used]
        for col in original_cols_to_keep:
              # Evitar sobrescribir si por casualidad una columna original
              # tiene el mismo nombre que una estándar ya creada (ej. AÑO)
              target_col_name = f"ORIG_{col}"
              if target_col_name not in df_std.columns:
                   df_std[target_col_name] = df_raw[col]
              else: # Añadir sufijo si ya existe
                   suffix = 1
                   while f"ORIG_{col}_{suffix}" in df_std.columns:
                       suffix += 1
                   df_std[f"ORIG_{col}_{suffix}"] = df_raw[col]


        # Devolver solo filas con importe válido y texto de modelo no vacío/NaN
        df_std = df_std.dropna(subset=[IMPORTE_STD, TEXTO_MODELO])
        df_std = df_std[df_std[TEXTO_MODELO]!='']

        return df_std

    except Exception as e:
        st.error(f"Error inesperado aplicando mapeo '{mapping.get('bank_name', 'Desconocido')}': {e}")
        st.error(traceback.format_exc()) # Imprimir traceback completo para depuración
        return None


# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🏦 Categorizador Bancario Multi-Formato v2")

# --- Fase 1: Entrenamiento Inicial ---
with st.expander("Fase 1: Entrenar Modelo con Datos Históricos Categorizados", expanded=True):
    st.write("Sube tu archivo CSV histórico (ej: `Gastos.csv`) que ya contiene las categorías y subcategorías asignadas. Este archivo entrena el modelo base.")

    uploaded_historic_file = st.file_uploader("Cargar Archivo Histórico Categorizado (.csv)", type="csv", key="historic_uploader_f1")

    if uploaded_historic_file:
        if st.button("🧠 Entrenar Modelo y Aprender Conocimiento Inicial", key="train_historic_f1"):
            with st.spinner("Procesando archivo histórico y entrenando..."):
                # Leer el archivo raw ANTES de pasarlo a la función de parseo
                df_raw_hist, _ = read_sample_csv(uploaded_historic_file)
                if df_raw_hist is not None:
                    df_std_hist = parse_historic_categorized(df_raw_hist.copy()) # Pasar copia
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
                        else:
                            st.error("Fallo en el entrenamiento del modelo base.")
                            st.session_state.model_trained = False
                            st.session_state.training_report = report # Guardar mensaje de error
                            st.sidebar.error("Entrenamiento Fallido")
                            st.sidebar.text(st.session_state.training_report)
                    else:
                        st.error("No se pudo parsear el archivo histórico o no contenía datos válidos.")
                else:
                     st.error("No se pudo leer el archivo histórico.")

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
    # Convertir a JSON para mostrar (evita problemas con objetos no serializables si los hubiera)
    try:
        st.sidebar.json(json.dumps(st.session_state.bank_mappings, indent=2), expanded=False) # Indentado para legibilidad
    except TypeError:
         st.sidebar.write("No se pueden mostrar los mapeos (posible objeto no serializable).") # Fallback
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
                                             if COMERCIO_STD in df_pred.columns: display_cols.insert(2, COMERCIO_STD)
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
