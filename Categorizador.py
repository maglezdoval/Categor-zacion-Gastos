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
TEXTO_MODELO = 'TEXTO_MODELO'
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

# Funciones de Parseo/Estandarización (Simplificadas para el ejemplo inicial)
# El parseo REAL se hará usando el mapeo en la Fase 3
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
        sniffer = io.StringIO(bytes_data.decode('utf-8', errors='ignore'))
        try:
             dialect = pd.io.parsers.readers.csv.Sniffer().sniff(sniffer.read(1024*10)) # Read more data
             sep = dialect.delimiter
             st.info(f"Separador detectado: '{sep}'")
        except Exception:
             st.warning("No se pudo detectar separador, asumiendo ';'.")
             sep = ';'

        sniffer.seek(0) # Reset buffer

        # Leer con encoding flexible y separador detectado/asumido
        try:
            df = pd.read_csv(io.BytesIO(bytes_data), encoding='utf-8', sep=sep, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(io.BytesIO(bytes_data), encoding='latin1', sep=sep, low_memory=False)

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
    # (Sin cambios respecto a la versión anterior, ya usaba _STD)
    knowledge = {'categorias': [], 'subcategorias': {}, 'comercios': {}}
    if df_std is None or CATEGORIA_STD not in df_std.columns or df_std.empty: return knowledge
    knowledge['categorias'] = sorted([c for c in df_std[CATEGORIA_STD].dropna().unique() if c])
    for cat in knowledge['categorias']:
        subcats = df_std.loc[df_std[CATEGORIA_STD] == cat, SUBCATEGORIA_STD].dropna().unique()
        knowledge['subcategorias'][cat] = sorted([s for s in subcats if s])
        comers = df_std.loc[df_std[CATEGORIA_STD] == cat, COMERCIO_STD].dropna().unique()
        knowledge['comercios'][cat] = sorted([c for c in comers if c and c != 'n/a'])
    return knowledge

@st.cache_resource
def train_classifier_std(df_std):
    # (Sin cambios respecto a la versión anterior, ya usaba _STD y TEXTO_MODELO)
    report = "Modelo no entrenado."
    model = None; vectorizer = None
    required = [TEXTO_MODELO, CATEGORIA_STD]
    if df_std is None or df_std.empty or not all(c in df_std.columns for c in required): return model, vectorizer, report
    df_train = df_std.dropna(subset=required)
    df_train = df_train[df_train[CATEGORIA_STD] != '']
    if df_train.empty or len(df_train[CATEGORIA_STD].unique()) < 2: return model, vectorizer, report
    try:
        X = df_train[TEXTO_MODELO]; y = df_train[CATEGORIA_STD]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        vectorizer = TfidfVectorizer(); X_train_vec = vectorizer.fit_transform(X_train)
        model = MultinomialNB(); model.fit(X_train_vec, y_train)
        X_test_vec = vectorizer.transform(X_test); y_pred = model.predict(X_test_vec)
        labels = sorted(list(set(y_test) | set(y_pred)))
        report = classification_report(y_test, y_pred, labels=labels, zero_division=0)
    except Exception as e: report = f"Error entrenamiento: {e}"
    return model, vectorizer, report

def standardize_data_with_mapping(df_raw, mapping):
    """Aplica el mapeo guardado para estandarizar un DataFrame nuevo."""
    try:
        df_std = pd.DataFrame()
        original_columns = df_raw.columns

        # 1. Mapear columnas básicas
        for std_col, source_col in mapping['columns'].items():
            if source_col in original_columns:
                df_std[std_col] = df_raw[source_col]
            else:
                # Si una columna mapeada no existe en el nuevo archivo, crea una columna vacía
                st.warning(f"La columna mapeada '{source_col}' para '{std_col}' no se encontró en el archivo. Se creará vacía.")
                df_std[std_col] = pd.Series(dtype='object') # O float para importe

        # 2. Manejar Fecha (Asumiendo que FECHA_STD mapea a UNA columna de fecha)
        fecha_col_source = mapping['columns'].get(FECHA_STD)
        if fecha_col_source and fecha_col_source in df_std.columns:
             date_format = mapping.get('date_format')
             # Limpiar posibles valores no-fecha antes de convertir
             df_std[fecha_col_source] = df_std[fecha_col_source].astype(str) # Asegurar que es string
             valid_dates = pd.to_datetime(df_std[fecha_col_source], format=date_format, errors='coerce')

             if valid_dates.isnull().any():
                  st.warning(f"Algunas fechas en la columna '{fecha_col_source}' no coinciden con el formato esperado '{date_format}' o son inválidas. Esas filas tendrán fechas Nulas.")

             df_std[AÑO_STD] = valid_dates.dt.year.fillna(0).astype(int)
             df_std[MES_STD] = valid_dates.dt.month.fillna(0).astype(int)
             df_std[DIA_STD] = valid_dates.dt.day.fillna(0).astype(int)
             df_std = df_std.drop(columns=[FECHA_STD]) # Eliminar la columna de fecha original estandarizada
        elif mapping.get(AÑO_STD) and mapping.get(MES_STD) and mapping.get(DIA_STD):
             # Si se mapearon columnas separadas AÑO, MES, DIA
             for col_std in [AÑO_STD, MES_STD, DIA_STD]:
                 col_source = mapping['columns'].get(col_std)
                 if col_source and col_source in original_columns:
                     df_std[col_std] = pd.to_numeric(df_raw[col_source], errors='coerce').fillna(0).astype(int)
                 else: # Si falta una de las columnas mapeadas
                      st.warning(f"Columna mapeada '{col_source}' para '{col_std}' no encontrada. Se usará 0.")
                      df_std[col_std] = 0

        else:
             st.error("Mapeo de fecha incompleto. Se necesitan mapeos para AÑO, MES, DIA o para una única columna FECHA_STD con su formato.")
             return None


        # 3. Limpiar Importe (Usando separadores guardados)
        importe_col_source = mapping['columns'].get(IMPORTE_STD)
        if importe_col_source and importe_col_source in df_std.columns:
            importe_str = df_std[IMPORTE_STD].fillna('0').astype(str)
            # Eliminar separador de miles si existe y fue especificado
            if mapping.get('thousands_sep'):
                 importe_str = importe_str.str.replace(mapping['thousands_sep'], '', regex=False)
            # Reemplazar separador decimal
            importe_str = importe_str.str.replace(mapping.get('decimal_sep', ','), '.', regex=False)
            df_std[IMPORTE_STD] = pd.to_numeric(importe_str, errors='coerce')
        else:
             st.error(f"No se pudo procesar el importe. Columna fuente '{importe_col_source}' no encontrada o no mapeada.")
             return None

        # 4. Limpiar Concepto y Comercio (ya deberían estar mapeados)
        for col_std in [CONCEPTO_STD, COMERCIO_STD]:
            if col_std in df_std.columns:
                df_std[col_std] = df_std[col_std].fillna('').astype(str).str.lower().str.strip()
            elif col_std == COMERCIO_STD: # Si COMERCIO_STD no fue mapeado, crearla vacía
                 df_std[COMERCIO_STD] = ''


        # 5. Crear Texto para Modelo
        df_std[TEXTO_MODELO] = df_std[CONCEPTO_STD] + ' ' + df_std[COMERCIO_STD]
        df_std[TEXTO_MODELO] = df_std[TEXTO_MODELO].str.strip()

        # 6. Mantener columnas originales relevantes
        original_cols_to_keep = [c for c in original_columns if c not in mapping['columns'].values()]
        for col in original_cols_to_keep:
              df_std[f"ORIG_{col}"] = df_raw[col]

        return df_std.dropna(subset=[IMPORTE_STD]) # Devolver solo filas con importe válido

    except Exception as e:
        st.error(f"Error aplicando mapeo '{mapping.get('bank_name', 'Desconocido')}': {e}")
        return None

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("🏦 Categorizador Bancario Multi-Formato v2")

# --- Fase 1: Entrenamiento Inicial ---
st.header("Fase 1: Entrenar Modelo con Datos Históricos Categorizados")
st.write("Sube tu archivo CSV histórico (ej: `Gastos.csv`) que ya contiene las categorías y subcategorías asignadas. Este archivo entrena el modelo base.")

uploaded_historic_file = st.file_uploader("Cargar Archivo Histórico Categorizado (.csv)", type="csv", key="historic_uploader")

if uploaded_historic_file:
    if st.button("🧠 Entrenar Modelo y Aprender Conocimiento Inicial", key="train_historic"):
        with st.spinner("Procesando archivo histórico y entrenando..."):
            df_raw_hist = read_sample_csv(uploaded_historic_file)[0] # Leerlo primero
            if df_raw_hist is not None:
                df_std_hist = parse_historic_categorized(df_raw_hist)
                if df_std_hist is not None:
                    # Extraer conocimiento
                    st.session_state.knowledge = extract_knowledge_std(df_std_hist)
                    st.sidebar.success("Conocimiento Inicial Extraído")
                    with st.sidebar.expander("Ver Categorías Aprendidas"):
                        st.write(st.session_state.knowledge['categorias'])

                    # Entrenar modelo
                    model, vectorizer, report = train_classifier_std(df_std_hist)
                    if model and vectorizer:
                        st.session_state.model = model
                        st.session_state.vectorizer = vectorizer
                        st.session_state.model_trained = True
                        st.session_state.training_report = report
                        st.success("¡Modelo entrenado exitosamente!")
                        st.sidebar.subheader("Evaluación Modelo Base")
                        with st.sidebar.expander("Ver Informe"):
                             st.text(st.session_state.training_report)
                    else:
                        st.error("Fallo en el entrenamiento del modelo base.")
                        st.session_state.model_trained = False
                        st.session_state.training_report = report # Guardar mensaje de error
                        st.sidebar.error("Entrenamiento Fallido")
                        st.sidebar.text(st.session_state.training_report)
                else:
                    st.error("No se pudo parsear el archivo histórico.")
            else:
                 st.error("No se pudo leer el archivo histórico.")


# --- Fase 2: Aprendizaje de Formatos Bancarios ---
st.divider()
st.header("Fase 2: Aprender Formatos de Archivos Bancarios")
st.write("Sube un archivo CSV de ejemplo (sin categorizar) para cada banco cuyo formato quieras que la aplicación aprenda. Luego, define cómo mapear sus columnas a las estándar.")

bank_options = ["SANTANDER", "EVO", "WIZINK", "AMEX"] # Añade más bancos aquí
selected_bank_learn = st.selectbox("Selecciona Banco para Aprender Formato:", bank_options, key="bank_learn")

uploaded_sample_file = st.file_uploader(f"Cargar archivo CSV de ejemplo de {selected_bank_learn}", type="csv", key="sample_uploader")

if uploaded_sample_file:
    df_sample, detected_columns = read_sample_csv(uploaded_sample_file)

    if df_sample is not None:
        st.write(f"Columnas detectadas en el archivo de {selected_bank_learn}:")
        st.write(detected_columns)
        st.dataframe(df_sample.head())

        st.subheader("Mapeo de Columnas")
        st.write(f"Selecciona qué columna del archivo de {selected_bank_learn} corresponde a cada campo estándar necesario.")

        current_mapping = {'columns': {}} # Mapeo temporal para esta sesión de UI

        cols_with_none = [None] + detected_columns # Opción para no mapear

        # Mapeos obligatorios + opcionales
        st.write("**Campos Esenciales:**")
        current_mapping['columns'][CONCEPTO_STD] = st.selectbox(f"`{CONCEPTO_STD}` (Descripción de la transacción)", cols_with_none, key=f"map_{CONCEPTO_STD}_{selected_bank_learn}")
        current_mapping['columns'][IMPORTE_STD] = st.selectbox(f"`{IMPORTE_STD}` (Valor numérico)", cols_with_none, key=f"map_{IMPORTE_STD}_{selected_bank_learn}")

        st.write("**Campo de Fecha (elige una opción):**")
        map_single_date = st.checkbox("¿La fecha está en una sola columna?", key=f"map_single_date_{selected_bank_learn}")

        if map_single_date:
            current_mapping['columns'][FECHA_STD] = st.selectbox(f"`{FECHA_STD}` (Columna única de fecha)", cols_with_none, key=f"map_{FECHA_STD}_{selected_bank_learn}")
            # Pedir formato de fecha
            date_format_guess = st.text_input(f"Formato de fecha (ej: %d/%m/%Y, %Y-%m-%d, etc.)", key=f"map_date_format_{selected_bank_learn}")
            if date_format_guess:
                 current_mapping['date_format'] = date_format_guess
            # Eliminar mapeos de AÑO/MES/DIA si existen
            current_mapping['columns'].pop(AÑO_STD, None)
            current_mapping['columns'].pop(MES_STD, None)
            current_mapping['columns'].pop(DIA_STD, None)

        else:
            current_mapping['columns'][AÑO_STD] = st.selectbox(f"`{AÑO_STD}`", cols_with_none, key=f"map_{AÑO_STD}_{selected_bank_learn}")
            current_mapping['columns'][MES_STD] = st.selectbox(f"`{MES_STD}`", cols_with_none, key=f"map_{MES_STD}_{selected_bank_learn}")
            current_mapping['columns'][DIA_STD] = st.selectbox(f"`{DIA_STD}`", cols_with_none, key=f"map_{DIA_STD}_{selected_bank_learn}")
            # Eliminar mapeo de FECHA_STD si existe
            current_mapping['columns'].pop(FECHA_STD, None)
            current_mapping.pop('date_format', None)


        st.write("**Campos Opcionales (pero recomendados):**")
        current_mapping['columns'][COMERCIO_STD] = st.selectbox(f"`{COMERCIO_STD}` (Nombre del Comercio/Entidad)", cols_with_none, key=f"map_{COMERCIO_STD}_{selected_bank_learn}")

        st.write("**Configuración de Importe:**")
        current_mapping['decimal_sep'] = st.text_input("Separador Decimal usado en el archivo (ej: , o .)", value=",", key=f"map_decimal_{selected_bank_learn}")
        current_mapping['thousands_sep'] = st.text_input("Separador de Miles (si existe, ej: . o , o dejar vacío)", value="", key=f"map_thousands_{selected_bank_learn}")


        # Filtrar mapeos donde el usuario seleccionó None
        final_mapping_cols = {std: src for std, src in current_mapping['columns'].items() if src is not None}

        # Validar mapeo antes de guardar
        valid_mapping = True
        # Chequear obligatorios
        missing_mandatory_map = [std for std in MANDATORY_STD_COLS if std not in final_mapping_cols and not (std == FECHA_STD and AÑO_STD in final_mapping_cols)] # Chequeo especial fecha
        if missing_mandatory_map:
             st.error(f"Faltan mapeos obligatorios: {', '.join(missing_mandatory_map)}")
             valid_mapping = False
        # Chequear formato de fecha si se usa columna única
        if map_single_date and FECHA_STD in final_mapping_cols and not current_mapping.get('date_format'):
            st.error("Debes especificar el formato de fecha si usas una columna única.")
            valid_mapping = False

        if valid_mapping:
             # Crear el diccionario final para guardar
            mapping_to_save = {
                 'bank_name': selected_bank_learn,
                 'columns': final_mapping_cols,
                 'decimal_sep': current_mapping.get('decimal_sep', ','),
                 'thousands_sep': current_mapping.get('thousands_sep', None), # Guardar None si está vacío
            }
            if map_single_date and current_mapping.get('date_format'):
                 mapping_to_save['date_format'] = current_mapping['date_format']


            if st.button(f"💾 Guardar Mapeo para {selected_bank_learn}", key="save_mapping"):
                st.session_state.bank_mappings[selected_bank_learn] = mapping_to_save
                st.success(f"¡Mapeo para {selected_bank_learn} guardado!")
                st.json(mapping_to_save) # Mostrar lo que se guardó
                # Considerar limpiar el uploader aquí si es posible
        else:
            st.warning("Completa los mapeos obligatorios y la configuración necesaria antes de guardar.")

# Mostrar mapeos guardados
st.sidebar.divider()
st.sidebar.subheader("Mapeos Bancarios Guardados")
if st.session_state.bank_mappings:
    st.sidebar.json(st.session_state.bank_mappings, expanded=False)
else:
    st.sidebar.info("Aún no se han guardado mapeos.")


# --- Fase 3: Categorización ---
st.divider()
st.header("Fase 3: Categorizar Nuevos Archivos")

if not st.session_state.model_trained:
    st.warning("⚠️ Modelo no entrenado (Ver Fase 1).")
elif not st.session_state.bank_mappings:
    st.warning("⚠️ No se han aprendido formatos bancarios (Ver Fase 2).")
else:
    st.write("Selecciona el banco y sube el archivo CSV **sin categorizar** que deseas procesar.")
    available_banks_for_pred = list(st.session_state.bank_mappings.keys())
    selected_bank_predict = st.selectbox("Banco del Nuevo Archivo:", available_banks_for_pred, key="bank_predict_final")

    uploaded_final_file = st.file_uploader(f"Cargar archivo CSV NUEVO de {selected_bank_predict}", type="csv", key="final_uploader")

    if uploaded_final_file and selected_bank_predict:
        mapping_to_use = st.session_state.bank_mappings.get(selected_bank_predict)
        if not mapping_to_use:
             st.error(f"Error interno: No se encontró el mapeo para {selected_bank_predict} aunque debería existir.")
        else:
             st.write("Procesando archivo nuevo...")
             with st.spinner(f"Estandarizando datos con mapeo de {selected_bank_predict}..."):
                  df_raw_new, _ = read_sample_csv(uploaded_final_file)
                  if df_raw_new is not None:
                      df_std_new = standardize_data_with_mapping(df_raw_new.copy(), mapping_to_use)

             if df_std_new is not None and not df_std_new.empty:
                  st.success("Datos estandarizados.")
                  with st.spinner("Aplicando modelo de categorización..."):
                      try:
                           # Asegurar que la columna TEXTO_MODELO existe y está limpia
                           if TEXTO_MODELO not in df_std_new.columns:
                               st.error(f"Error crítico: La columna {TEXTO_MODELO} no se generó durante la estandarización.")
                           else:
                                df_std_new = df_std_new.dropna(subset=[TEXTO_MODELO]) # Eliminar filas donde el texto sea NaN
                                if not df_std_new.empty:
                                     X_new_vec = st.session_state.vectorizer.transform(df_std_new[TEXTO_MODELO])
                                     predictions = st.session_state.model.predict(X_new_vec)
                                     df_std_new[CATEGORIA_PREDICHA] = predictions.astype(str).str.capitalize()

                                     st.subheader("📊 Resultados de la Categorización")
                                     # Seleccionar y reordenar columnas para mostrar
                                     display_cols = [CATEGORIA_PREDICHA, CONCEPTO_STD, IMPORTE_STD, AÑO_STD, MES_STD, DIA_STD]
                                     if COMERCIO_STD in df_std_new.columns: display_cols.insert(2, COMERCIO_STD)
                                     # Añadir columnas originales relevantes
                                     orig_cols = [c for c in df_std_new.columns if c.startswith('ORIG_')]
                                     display_cols.extend(orig_cols)

                                     st.dataframe(df_std_new[display_cols])

                                     # Descarga
                                     csv_output = df_std_new.to_csv(index=False, sep=';', decimal=',').encode('utf-8')
                                     st.download_button(
                                          label=f"📥 Descargar '{uploaded_final_file.name}' Categorizado",
                                          data=csv_output,
                                          file_name=f"categorizado_{uploaded_final_file.name}",
                                          mime='text/csv',
                                          key=f"download_final_{uploaded_final_file.name}"
                                     )
                                else:
                                     st.warning("No quedaron filas válidas para categorizar después de limpiar datos de texto.")

                      except Exception as e:
                           st.error(f"Error durante la predicción de categorías: {e}")
                           st.error(f"Vectorizador esperado: {st.session_state.vectorizer}")
                           st.error(f"Texto de entrada (primeras filas): {df_std_new[TEXTO_MODELO].head().tolist() if TEXTO_MODELO in df_std_new else 'Columna no encontrada'}")


                  elif df_std_new is not None and df_std_new.empty:
                      st.warning("El archivo no contenía datos válidos después de la estandarización.")
                  else: # df_std_new is None
                      st.error("Fallo en la estandarización del archivo nuevo usando el mapeo guardado.")
             else: # df_raw_new is None
                 st.error("No se pudo leer el archivo nuevo.")

# Sidebar Info
st.sidebar.divider()
st.sidebar.header("Acerca de")
st.sidebar.info(
    "1. Entrena con tu CSV histórico. "
    "2. Enseña a la app los formatos de tus bancos. "
    "3. Sube nuevos archivos para categorizarlos."
)
