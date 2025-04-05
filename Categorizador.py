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
        df_raw.columns = [col.upper().strip() for col in df_raw.columns]
        required = ['CONCEPTO', 'CATEGORÍA', 'SUBCATEGORIA', 'IMPORTE', 'AÑO', 'MES', 'DIA']
        if not all(col in df_raw.columns for col in required):
            missing = [col for col in required if col not in df_raw.columns]
            st.error(f"Archivo histórico: Faltan columnas: {', '.join(missing)}")
            return None

        df_std = pd.DataFrame()
        df_std[CONCEPTO_STD] = df_raw['CONCEPTO'].fillna('').astype(str).str.lower().strip()
        df_std[COMERCIO_STD] = df_raw.get('COMERCIO', pd.Series(dtype=str)).fillna('').astype(str).str.lower().strip() # Comercio puede faltar
        df_std[IMPORTE_STD] = pd.to_numeric(df_raw['IMPORTE'].astype(str).str.replace(',', '.', regex=False), errors='coerce')
        df_std[AÑO_STD] = pd.to_numeric(df_raw['AÑO'], errors='coerce').fillna(0).astype(int)
        df_std[MES_STD] = pd.to_numeric(df_raw['MES'], errors='coerce').fillna(0).astype(int)
        df_std[DIA_STD] = pd.to_numeric(df_raw['DIA'], errors='coerce').fillna(0).astype(int)
        df_std[CATEGORIA_STD] = df_raw['CATEGORÍA'].fillna('').astype(str).str.lower().strip()
        df_std[SUBCATEGORIA_STD] = df_raw['SUBCATEGORIA'].fillna('').astype(str).str.lower().strip()

        # Crear columna de texto para entrenamiento
        df_std[TEXTO_MODELO] = df_std[CONCEPTO_STD] + ' ' + df_std[COMERCIO_STD]
        df_std[TEXTO_MODELO] = df_std[TEXTO_MODELO].str.strip()

        # Filtrar filas sin categoría o importe válido
        df_std = df_std.dropna(subset=[IMPORTE_STD])
        df_std = df_std[df_std[CATEGORIA_STD] != '']

        if df_std.empty:
            st.warning("No se encontraron filas válidas con categorías en el archivo histórico.")
            return None

        return df_std
    except Exception as e:
        st.error(f"Error parseando archivo histórico: {e}")
        return None

def read_sample_csv(uploaded_file):
    """Lee un CSV de muestra y devuelve el DataFrame y sus columnas."""
    if uploaded_file is None:
        return None, []
    try:
        bytes_data = uploaded_file.getvalue()
        # Intentar detectar separador común (puede fallar)
        # Usar decode con replace para evitar errores en sniffer si hay caracteres raros
        sniffer_content = bytes_data.decode('utf-8', errors='replace')
        sniffer = io.StringIO(sniffer_content)
        try:
             # Leer una porción más grande para mejorar detección, manejar final de archivo inesperado
             sample_data = sniffer.read(min(1024 * 20, len(sniffer_content))) # Leer hasta 20KB
             if not sample_data:
                  st.error("El archivo parece estar vacío.")
                  return None, []
             dialect = pd.io.parsers.readers.csv.Sniffer().sniff(sample_data)
             sep = dialect.delimiter
             st.info(f"Separador detectado: '{sep}'")
        except Exception as sniff_err:
             st.warning(f"No se pudo detectar separador (Error: {sniff_err}), asumiendo ';'.")
             sep = ';'

        # Leer con encoding flexible y separador detectado/asumido
        try:
            # Volver al inicio del stream de bytes
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
                 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
                 test_available = True
             except ValueError: # No se puede estratificar
                 X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                 test_available = True
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
                labels = sorted(list(set(y_test) | set(y_pred)))
                report = classification_report(y_test, y_pred, labels=labels, zero_division=0)
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
        original_columns = df_raw.columns

        # 1. Mapear columnas básicas usando el mapping['columns']
        for std_col, source_col in mapping['columns'].items():
            if source_col in original_columns:
                # Copiar la columna fuente a la columna estándar correspondiente
                df_std[std_col] = df_raw[source_col]
            elif std_col in OPTIONAL_STD_COLS or std_col == FECHA_STD: # Permitir que opcionales falten
                 st.info(f"Columna opcional/fecha mapeada '{source_col}' para '{std_col}' no encontrada en el archivo. Se omitirá o creará vacía.")
                 df_std[std_col] = pd.Series(dtype='object') # Crear vacía si es opcional
            elif std_col in MANDATORY_STD_COLS and std_col != FECHA_STD : # Esencial (excepto FECHA que se trata abajo)
                 st.error(f"¡Error Crítico! La columna esencial mapeada '{source_col}' para '{std_col}' no existe en este archivo.")
                 return None


        # 2. Manejar Fecha (Comprobar si se mapeó FECHA_STD o AÑO/MES/DIA)
        fecha_col_source = mapping['columns'].get(FECHA_STD)
        año_col_source = mapping['columns'].get(AÑO_STD)
        mes_col_source = mapping['columns'].get(MES_STD)
        dia_col_source = mapping['columns'].get(DIA_STD)

        if fecha_col_source and fecha_col_source in df_std.columns: # Opción 1: Columna única de fecha
             date_format = mapping.get('date_format')
             if not date_format:
                  st.error(f"Error: Se mapeó la columna de fecha '{fecha_col_source}' pero falta el formato de fecha en el mapeo.")
                  return None

             # Limpiar posibles valores no-fecha antes de convertir
             # Forzar a string, reemplazar múltiples espacios, etc. podría ser necesario
             df_std[FECHA_STD] = df_std[FECHA_STD].astype(str).str.strip()
             valid_dates = pd.to_datetime(df_std[FECHA_STD], format=date_format, errors='coerce')

             if valid_dates.isnull().any():
                  st.warning(f"Algunas fechas en '{fecha_col_source}' no coinciden con formato '{date_format}' o son inválidas. Se usarán valores nulos (0).")

             df_std[AÑO_STD] = valid_dates.dt.year.fillna(0).astype(int)
             df_std[MES_STD] = valid_dates.dt.month.fillna(0).astype(int)
             df_std[DIA_STD] = valid_dates.dt.day.fillna(0).astype(int)
             df_std = df_std.drop(columns=[FECHA_STD]) # Eliminar la columna estandarizada original

        elif año_col_source and mes_col_source and dia_col_source: # Opción 2: Columnas separadas
            for col_std, col_source in zip([AÑO_STD, MES_STD, DIA_STD], [año_col_source, mes_col_source, dia_col_source]):
                 if col_source in original_columns:
                     df_std[col_std] = pd.to_numeric(df_raw[col_source], errors='coerce').fillna(0).astype(int)
                 else:
                      st.error(f"Error: La columna de fecha mapeada '{col_source}' para '{col_std}' no existe en el archivo.")
                      return None
        else:
             st.error("Error: Mapeo de fecha inválido. Debe mapear una columna a FECHA_STD (con formato) O mapear columnas separadas a AÑO, MES y DIA.")
             return None


        # 3. Limpiar Importe (Usando separadores guardados)
        importe_col_source = mapping['columns'].get(IMPORTE_STD)
        if importe_col_source and importe_col_source in df_std.columns: # Usar df_std aquí porque la columna ya fue copiada
            # Forzar a string para limpieza robusta
            importe_str = df_std[IMPORTE_STD].fillna('0').astype(str)
            # Eliminar separador de miles ANTES de reemplazar decimal
            thousands_sep = mapping.get('thousands_sep')
            if thousands_sep: # Solo reemplazar si no está vacío
                 importe_str = importe_str.str.replace(thousands_sep, '', regex=False)
            # Reemplazar separador decimal con punto
            decimal_sep = mapping.get('decimal_sep', ',') # Default a coma si no se especifica
            importe_str = importe_str.str.replace(decimal_sep, '.', regex=False)
            # Convertir a numérico, errores a NaN
            df_std[IMPORTE_STD] = pd.to_numeric(importe_str, errors='coerce')
            if df_std[IMPORTE_STD].isnull().any():
                 st.warning(f"Algunos valores de importe en '{importe_col_source}' no pudieron convertirse tras aplicar separadores. Serán NaN.")
        elif importe_col_source: # Mapeado pero no encontrado (debería haber fallado antes, pero doble check)
            st.error(f"Error interno: Columna de importe '{importe_col_source}' mapeada pero no encontrada.")
            return None
        else: # No mapeado (debería haber fallado en la validación del mapeo)
             st.error("Error: La columna IMPORTE_STD no fue mapeada.")
             return None


        # 4. Limpiar Concepto y Comercio (ya deberían estar mapeados y en df_std)
        for col_std in [CONCEPTO_STD, COMERCIO_STD]:
            if col_std in df_std.columns:
                df_std[col_std] = df_std[col_std].fillna('').astype(str).str.lower().str.strip()
            elif col_std == COMERCIO_STD: # Si COMERCIO_STD no se mapeó (opcional), crearla vacía
                 df_std[COMERCIO_STD] = ''

        # 5. Crear Texto para Modelo (siempre CONCEPTO + COMERCIO, incluso si comercio está vacío)
        df_std[TEXTO_MODELO] = df_std[CONCEPTO_STD] + ' ' + df_std[COMERCIO_STD]
        df_std[TEXTO_MODELO] = df_std[TEXTO_MODELO].str.strip()

        # 6. Mantener columnas originales relevantes (que no fueron mapeadas a estándar)
        mapped_source_cols = list(mapping['columns'].values())
        original_cols_to_keep = [c for c in original_columns if c not in mapped_source_cols]
        for col in original_cols_to_keep:
              df_std[f"ORIG_{col}"] = df_raw[col]

        # Devolver solo filas con importe válido y texto de modelo no vacío
        df_std = df_std.dropna(subset=[IMPORTE_STD])
        df_std = df_std[df_std[TEXTO_MODELO]!='']

        return df_std

    except Exception as e:
        st.error(f"Error inesperado aplicando mapeo '{mapping.get('bank_name', 'Desconocido')}': {e}")
        import traceback
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
                df_raw_hist, _ = read_sample_csv(uploaded_historic_file) # Leerlo primero
                if df_raw_hist is not None:
                    df_std_hist = parse_historic_categorized(df_raw_hist)
                    if df_std_hist is not None and not df_std_hist.empty:
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
            current_mapping = {'columns': {}} # Empezar fresco para UI

            cols_with_none = [None] + detected_columns

            # --- Mapeos Esenciales ---
            st.markdown("**Campos Esenciales:**")
            current_mapping['columns'][CONCEPTO_STD] = st.selectbox(
                f"`{CONCEPTO_STD}` (Descripción)", cols_with_none,
                index=cols_with_none.index(saved_mapping['columns'].get(CONCEPTO_STD)) if saved_mapping['columns'].get(CONCEPTO_STD) in cols_with_none else 0,
                key=f"map_{CONCEPTO_STD}_{selected_bank_learn}"
            )
            current_mapping['columns'][IMPORTE_STD] = st.selectbox(
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
                current_mapping['columns'][FECHA_STD] = st.selectbox(
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
                if date_format_guess: current_mapping['date_format'] = date_format_guess.strip()
                # Eliminar mapeos AÑO/MES/DIA si se cambia a fecha única
                current_mapping['columns'].pop(AÑO_STD, None); current_mapping['columns'].pop(MES_STD, None); current_mapping['columns'].pop(DIA_STD, None)
            else:
                current_mapping['columns'][AÑO_STD] = st.selectbox(
                    f"`{AÑO_STD}`", cols_with_none,
                    index=cols_with_none.index(saved_mapping['columns'].get(AÑO_STD)) if saved_mapping['columns'].get(AÑO_STD) in cols_with_none else 0,
                    key=f"map_{AÑO_STD}_{selected_bank_learn}"
                )
                current_mapping['columns'][MES_STD] = st.selectbox(
                    f"`{MES_STD}`", cols_with_none,
                    index=cols_with_none.index(saved_mapping['columns'].get(MES_STD)) if saved_mapping['columns'].get(MES_STD) in cols_with_none else 0,
                    key=f"map_{MES_STD}_{selected_bank_learn}"
                 )
                current_mapping['columns'][DIA_STD] = st.selectbox(
                    f"`{DIA_STD}`", cols_with_none,
                     index=cols_with_none.index(saved_mapping['columns'].get(DIA_STD)) if saved_mapping['columns'].get(DIA_STD) in cols_with_none else 0,
                     key=f"map_{DIA_STD}_{selected_bank_learn}"
                )
                # Eliminar mapeo FECHA_STD si se cambia a A/M/D
                current_mapping['columns'].pop(FECHA_STD, None); current_mapping.pop('date_format', None)


            # --- Mapeos Opcionales ---
            st.markdown("**Campos Opcionales:**")
            current_mapping['columns'][COMERCIO_STD] = st.selectbox(
                 f"`{COMERCIO_STD}` (Comercio/Entidad)", cols_with_none,
                 index=cols_with_none.index(saved_mapping['columns'].get(COMERCIO_STD)) if saved_mapping['columns'].get(COMERCIO_STD) in cols_with_none else 0,
                 key=f"map_{COMERCIO_STD}_{selected_bank_learn}"
            )

            # --- Configuración Importe ---
            st.markdown("**Configuración de Importe:**")
            current_mapping['decimal_sep'] = st.text_input("Separador Decimal (ej: , ó .)", value=saved_mapping.get('decimal_sep', ','), key=f"map_decimal_{selected_bank_learn}")
            current_mapping['thousands_sep'] = st.text_input("Separador de Miles (si aplica, ej: . ó ,)", value=saved_mapping.get('thousands_sep', ''), key=f"map_thousands_{selected_bank_learn}")


            # --- Validación y Guardado ---
            # Filtrar mapeos None antes de validar
            final_mapping_cols = {std: src for std, src in current_mapping['columns'].items() if src is not None}
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
                 elif not current_mapping.get('date_format'):
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
                      'decimal_sep': current_mapping.get('decimal_sep', ',').strip(),
                      # Guardar separador de miles solo si no está vacío
                      'thousands_sep': current_mapping.get('thousands_sep', '').strip() or None,
                 }
                 if map_single_date and current_mapping.get('date_format'):
                      mapping_to_save['date_format'] = current_mapping['date_format']

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
        st.sidebar.json(json.dumps(st.session_state.bank_mappings), expanded=False)
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

                                         st.dataframe(df_pred[display_cols])

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
