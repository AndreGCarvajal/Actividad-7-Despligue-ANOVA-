#Creamos el archivo de la APP del dashboard de Cadena de Suministro de CLV México
#Propuesta 2: Cadena de Suministro, Demanda SKU e Inventario
#Ejecutamos con: streamlit run app.py   (desde la carpeta Reto)
#####################################################
#Importamos librerias
from pathlib import Path
import unicodedata
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, confusion_matrix, accuracy_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
######################################################
#Configuramos la página con diseño amplio
st.set_page_config(page_title="CLV México | Supply Chain", page_icon="📦", layout="wide")

#Definimos las rutas a partir de la carpeta donde vive app.py (así funciona desde cualquier terminal)
BASE_DIR = Path(__file__).resolve().parent
RUTA_DEMANDA = BASE_DIR / "Datos Reales" / "Datos Reales Tratados" / "demanda_mensual_sku.csv"
RUTA_EXCEL = BASE_DIR / "Datos Reales" / "Datos_CLV_2026_Tec_v2.xlsx"
HOJA_EXISTENCIAS = "Existencias_04_09_2026"

#Definimos las marcas principales; el resto lo agrupamos como OTRAS para no saturar las gráficas
MARCAS_PRINCIPALES = ["VITA", "AMANN GIRRBACH", "CREATION", "AIDITE"]

#Limitamos los puntos de los diagramas de dispersión: el panel tiene ~240 mil filas y Plotly se vuelve lento
MAX_PUNTOS_GRAFICA = 5000
SEMILLA = 42


######################################################
#Creamos una función auxiliar para normalizar textos (quitamos acentos y espacios, pasamos a mayúsculas)
def normalizar_texto(valor):
    if pd.isna(valor):  #Si el valor viene vacío lo dejamos como nulo
        return np.nan
    texto = str(valor).strip().upper()  #Quitamos espacios en las orillas y pasamos a mayúsculas
    texto = " ".join(texto.split())  #Colapsamos espacios dobles internos
    #Eliminamos acentos para evitar duplicados como 'CREACIÓN' y 'CREACION'
    return "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")


######################################################
#Definimos la instancia de caché para no recargar los archivos en cada interacción
@st.cache_data
######################################################
#Creamos la función de carga de datos
def load_data():
    #Lectura del panel de demanda mensual por SKU (ya tratado por tratamiento_datos_inventario.py)
    demanda = pd.read_csv(RUTA_DEMANDA)
    #Lectura de la hoja de existencias; solo traemos las columnas que necesitamos
    existencias = pd.read_excel(
        RUTA_EXCEL,
        sheet_name=HOJA_EXISTENCIAS,
        usecols=["articulo_id", "nombre_articulo", "marca", "existencias", "valor_total_inventario"],
    )

    #Aseguramos que la llave articulo_id sea numérica en ambas tablas para que el merge coincida
    demanda["articulo_id"] = pd.to_numeric(demanda["articulo_id"], errors="coerce")
    existencias["articulo_id"] = pd.to_numeric(existencias["articulo_id"], errors="coerce")
    demanda = demanda.dropna(subset=["articulo_id"])  #Descartamos filas sin llave
    #Quitamos artículos repetidos en existencias para que el merge no duplique meses
    existencias = existencias.dropna(subset=["articulo_id"]).drop_duplicates("articulo_id")

    #Unimos por articulo_id con merge left: conservamos todos los meses del panel y traemos marca y existencias
    df = demanda.merge(existencias, on="articulo_id", how="left")

    #Convertimos las columnas numéricas por si vienen como texto
    columnas_numericas = ["salidas", "unidades_vendidas", "salidas_tope", "existencias", "valor_total_inventario"]
    for col in columnas_numericas:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    #Rellenamos nulos numéricos con ffill/bfill DENTRO de cada artículo (nunca entre artículos distintos,
    #porque eso copiaría las existencias de un SKU a otro)
    df = df.sort_values(["articulo_id", "periodo"])
    df[columnas_numericas] = df.groupby("articulo_id")[columnas_numericas].transform(lambda s: s.ffill().bfill())

    #Lo que siga vacío lo rellenamos con valores neutros
    df["existencias"] = df["existencias"].fillna(0)  #SKU sin registro en el corte -> 0 existencias
    df["valor_total_inventario"] = df["valor_total_inventario"].fillna(0)
    df[["salidas", "unidades_vendidas"]] = df[["salidas", "unidades_vendidas"]].fillna(0)
    df["salidas_tope"] = df["salidas_tope"].fillna(df["salidas"])  #Sin tope calculado -> usamos la salida real

    #Normalizamos la marca y agrupamos las secundarias en OTRAS
    df["marca_original"] = df["marca"].apply(normalizar_texto)
    df["marca"] = df["marca_original"].where(df["marca_original"].isin(MARCAS_PRINCIPALES), "OTRAS")

    #Aseguramos que es_pico sea booleano (en CSV puede venir como texto 'True'/'False')
    df["es_pico"] = (
        df["es_pico"].astype(str).str.strip().str.lower().map({"true": True, "1": True, "false": False, "0": False})
        .fillna(False).astype(bool)
    )

    #Convertimos periodo a fecha y aseguramos que anio y mes sean enteros
    df["periodo"] = pd.to_datetime(df["periodo"], errors="coerce")
    df["anio"] = pd.to_numeric(df["anio"], errors="coerce").fillna(df["periodo"].dt.year).astype(int)
    df["mes"] = pd.to_numeric(df["mes"], errors="coerce").fillna(df["periodo"].dt.month).astype(int)

    #Definimos las listas de variables que usaremos en los selectores
    Lista_categoricas = ["marca", "mes", "anio", "es_pico"]
    Lista_numericas = ["salidas", "unidades_vendidas", "existencias", "salidas_tope", "mes"]
    return df.reset_index(drop=True), Lista_categoricas, Lista_numericas


######################################################
#Creamos una función para tomar una muestra solo para graficar (los modelos sí usan todos los datos)
def muestra_para_grafica(datos):
    if len(datos) > MAX_PUNTOS_GRAFICA:
        return datos.sample(MAX_PUNTOS_GRAFICA, random_state=SEMILLA)
    return datos


#Creamos una función segura para la correlación (evitamos división por cero si una serie es constante)
def correlacion_segura(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


###############################################################################
#Cargamos los datos obtenidos de la función "load_data"
try:
    df, Lista_categoricas, Lista_numericas = load_data()
except FileNotFoundError as error:
    #Si falta un archivo avisamos qué ejecutar antes, en lugar de mostrar un error técnico
    st.error(f"No encontramos un archivo de datos: {error}. "
             "Primero ejecutamos `python3 tratamiento_datos_inventario.py` desde la carpeta Reto.")
    st.stop()

###############################################################################
#CREACIÓN DEL DASHBOARD
##############################################################################
#Generamos los encabezados para la barra lateral (sidebar)
st.sidebar.title("CLV MÉXICO | SUPPLY CHAIN")
#Widget 1: Selectbox
#Menú desplegable con las 5 vistas del dashboard
View = st.sidebar.selectbox(label="Tipo de Análisis", options=["Extracción de Características",
                                                             "Regresión Lineal", "Regresión No Lineal",
                                                             "Regresión Logística", "ANOVA"])

#Mostramos en la barra lateral el tamaño de los datos para dar contexto
st.sidebar.markdown("---")
st.sidebar.metric("Registros SKU × mes", f"{len(df):,}")
st.sidebar.metric("Artículos (SKU)", f"{df['articulo_id'].nunique():,}")

##############################################################################
# CONTENIDO DE LA VISTA 1
if View == "Extracción de Características":
    #EXTRACCIÓN DE CARACTERÍSTICAS
    #Select box de variables categóricas
    Variable_Cat = st.sidebar.selectbox(label="Variables", options=Lista_categoricas)
    #Obtenemos las frecuencias de las categorías de la variable seleccionada
    Tabla_frecuencias = df[Variable_Cat].astype(str).value_counts().reset_index()
    #Ajustamos los nombres de las cabeceras de las columnas
    Tabla_frecuencias.columns = ["categorias", "frecuencia"]
    #Para mes y año ordenamos cronológicamente; para marca y es_pico dejamos el orden por frecuencia
    if Variable_Cat in ["mes", "anio"]:
        Tabla_frecuencias = Tabla_frecuencias.sort_values("categorias", key=lambda s: s.astype(int))
    #Calculamos la frecuencia acumulada para el gráfico de área
    Tabla_frecuencias["frecuencia_acumulada"] = Tabla_frecuencias["frecuencia"].cumsum()

    #Generamos los encabezados para el dashboard
    st.title("Extracción de Características")
    st.caption(f"Distribución de la variable **{Variable_Cat}** en el panel de demanda mensual por SKU.")

    #Generamos el diseño del Layout deseado
    # Fila 1
    Contenedor_A, Contenedor_B = st.columns(2)
    with Contenedor_A:
        st.write("Gráfico de Barras")
        #GRAPH 1: BARPLOT
        #Desplegamos un bar plot, con las categorías en X y la frecuencia en Y
        figure1 = px.bar(data_frame=Tabla_frecuencias, x="categorias",
                         y="frecuencia", title="Frecuencia por categoría", color="frecuencia",
                         color_continuous_scale="Reds")
        figure1.update_xaxes(automargin=True, type="category")
        figure1.update_yaxes(automargin=True)
        figure1.update_layout(height=300)
        st.plotly_chart(figure1, width="stretch")

    with Contenedor_B:
        st.write("Gráfico de Pastel")
        #GRAPH 2: PIEPLOT
        #Desplegamos un pie plot con el porcentaje de cada categoría
        figure2 = px.pie(data_frame=Tabla_frecuencias, names="categorias",
                         values="frecuencia", title="Porcentaje por categoría",
                         color_discrete_sequence=px.colors.qualitative.Vivid)
        figure2.update_layout(height=300)
        st.plotly_chart(figure2, width="stretch")

    # Fila 2
    Contenedor_C, Contenedor_D = st.columns(2)
    with Contenedor_C:
        st.write("Gráfico de anillo o dona")
        #GRAPH 3: DONUT PLOT
        #Desplegamos un pie plot con hueco central (hole=0.4) para obtener la dona
        figure3 = px.pie(data_frame=Tabla_frecuencias, names="categorias",
                         values="frecuencia", hole=0.4, title="Frecuencia por categoría",
                         color_discrete_sequence=px.colors.qualitative.Bold)
        figure3.update_layout(height=300)
        st.plotly_chart(figure3, width="stretch")

    with Contenedor_D:
        st.write("Gráfico de área")
        #GRAPH 4: AREA PLOT
        #Desplegamos un area plot con la frecuencia acumulada de las categorías
        figure4 = px.area(data_frame=Tabla_frecuencias, x="categorias",
                          y="frecuencia_acumulada", title="Frecuencia acumulada por categoría",
                          color_discrete_sequence=px.colors.qualitative.Dark24)
        figure4.update_xaxes(type="category")
        figure4.update_layout(height=300)
        st.plotly_chart(figure4, width="stretch")

##############################################################################
# CONTENIDO DE LA VISTA 2
elif View == "Regresión Lineal":
    #REGRESIÓN LINEAL
    st.title("Regresión Lineal")
    #Selectores de variables: Y por defecto 'salidas' y X simple por defecto 'unidades_vendidas'
    Variable_Y = st.sidebar.selectbox("Variable dependiente (Y)", options=Lista_numericas,
                                      index=Lista_numericas.index("salidas"))
    opciones_x = [v for v in Lista_numericas if v != Variable_Y]  #No permitimos que X sea igual a Y
    Variable_X = st.sidebar.selectbox("Variable independiente simple (X)", options=opciones_x,
                                      index=opciones_x.index("unidades_vendidas") if "unidades_vendidas" in opciones_x else 0)
    opciones_multiple = [v for v in ["unidades_vendidas", "existencias", "salidas_tope", "mes"] if v != Variable_Y]
    Variables_X_multiple = st.sidebar.multiselect("Variables independientes (modelo múltiple)",
                                                  options=opciones_multiple, default=opciones_multiple[:2])

    #Avisamos cuando una variable se deriva de la propia Y (fuga de información)
    if Variable_Y == "salidas" and ("salidas_tope" in Variables_X_multiple or Variable_X == "salidas_tope"):
        st.warning("⚠️ 'salidas_tope' es la misma 'salidas' con los picos recortados: el R² será casi 1 "
                   "por construcción, no porque exista una relación predictiva.")

    Contenedor_A, Contenedor_B = st.columns(2)
    with Contenedor_A:
        st.subheader("Regresión lineal simple")
        #Quitamos nulos de las dos columnas que usaremos
        datos_simple = df[[Variable_X, Variable_Y]].dropna()
        if len(datos_simple) < 3 or datos_simple[Variable_X].std() == 0:
            st.warning("No hay suficientes datos con variación para ajustar el modelo.")
        else:
            #Ajustamos el modelo de regresión lineal simple
            modelo_simple = LinearRegression()
            modelo_simple.fit(datos_simple[[Variable_X]], datos_simple[Variable_Y])
            y_pred_simple = modelo_simple.predict(datos_simple[[Variable_X]])
            #Calculamos el coeficiente de correlación de Pearson (r) y el de determinación (R²)
            r_simple = correlacion_segura(datos_simple[Variable_X], datos_simple[Variable_Y])
            r2_simple = r2_score(datos_simple[Variable_Y], y_pred_simple)
            st.write(f"**Coeficiente de correlación (r):** {r_simple:.4f}")
            st.write(f"**Coeficiente de determinación (R²):** {r2_simple:.4f}")
            st.write(f"**Ecuación:** {Variable_Y} = {modelo_simple.coef_[0]:.4f} · {Variable_X} "
                     f"+ {modelo_simple.intercept_:.4f}")
            #Graficamos la dispersión con una muestra y superponemos la recta ajustada
            muestra = muestra_para_grafica(datos_simple)
            figure5 = px.scatter(muestra, x=Variable_X, y=Variable_Y, opacity=0.5,
                                 title=f"{Variable_Y} vs {Variable_X}")
            x_linea = np.linspace(datos_simple[Variable_X].min(), datos_simple[Variable_X].max(), 100)
            figure5.add_trace(go.Scatter(x=x_linea, y=modelo_simple.predict(pd.DataFrame({Variable_X: x_linea})),
                                         mode="lines", name="Modelo", line=dict(color="red")))
            st.plotly_chart(figure5, width="stretch")

    with Contenedor_B:
        st.subheader("Regresión lineal múltiple")
        if not Variables_X_multiple:
            st.info("Seleccionamos al menos una variable independiente en la barra lateral.")
        else:
            datos_multiple = df[Variables_X_multiple + [Variable_Y]].dropna()
            if len(datos_multiple) < len(Variables_X_multiple) + 2:
                st.warning("No hay suficientes datos para ajustar el modelo múltiple.")
            else:
                #Ajustamos el modelo de regresión lineal múltiple
                modelo_multiple = LinearRegression()
                modelo_multiple.fit(datos_multiple[Variables_X_multiple], datos_multiple[Variable_Y])
                y_pred_multiple = modelo_multiple.predict(datos_multiple[Variables_X_multiple])
                #Calculamos R² y la correlación múltiple (R = raíz de R², o r entre Y real y Y predicha)
                r2_multiple = r2_score(datos_multiple[Variable_Y], y_pred_multiple)
                r_multiple = correlacion_segura(datos_multiple[Variable_Y], y_pred_multiple)
                st.write(f"**Coeficiente de correlación múltiple (R):** {r_multiple:.4f}")
                st.write(f"**Coeficiente de determinación (R²):** {r2_multiple:.4f}")
                #Mostramos los coeficientes de cada variable
                st.dataframe(pd.DataFrame({"variable": Variables_X_multiple,
                                           "coeficiente": modelo_multiple.coef_}).round(4),
                             hide_index=True, width="stretch")
                #Graficamos Y real contra Y predicha: si el modelo fuera perfecto, los puntos caerían en la diagonal
                comparacion = pd.DataFrame({"Real": datos_multiple[Variable_Y].values, "Predicho": y_pred_multiple})
                figure6 = px.scatter(muestra_para_grafica(comparacion), x="Real", y="Predicho", opacity=0.5,
                                     title=f"{Variable_Y}: real vs predicho")
                limite = [comparacion.min().min(), comparacion.max().max()]
                figure6.add_trace(go.Scatter(x=limite, y=limite, mode="lines", name="Ajuste perfecto",
                                             line=dict(color="red", dash="dash")))
                st.plotly_chart(figure6, width="stretch")

##############################################################################
# CONTENIDO DE LA VISTA 3
elif View == "Regresión No Lineal":
    #REGRESIÓN NO LINEAL
    st.title("Regresión No Lineal")
    st.caption("Contrastamos artículos con pico de demanda contra artículos regulares. X = unidades_vendidas, Y = salidas.")
    Modelo = st.sidebar.selectbox("Modelo no lineal", options=["Función cuadrática", "Función exponencial"])

    #Definimos las dos funciones que ajustaremos
    def funcion_cuadratica(x, a, b, c):
        return a * x ** 2 + b * x + c

    def funcion_exponencial(x, a, b, c):
        return a * np.exp(-b * x) + c

    #Creamos una función que ajusta el modelo elegido y dibuja el resultado para un subgrupo
    def ajustar_y_graficar(datos, titulo):
        datos = datos[["unidades_vendidas", "salidas"]].dropna()
        x = datos["unidades_vendidas"].to_numpy(dtype=float)
        y = datos["salidas"].to_numpy(dtype=float)
        if len(datos) < 4 or np.std(x) == 0:
            st.warning(f"{titulo}: no hay suficientes datos con variación para ajustar el modelo.")
            return
        try:
            if Modelo == "Función cuadrática":
                #Ajustamos a·x² + b·x + c
                parametros, _ = curve_fit(funcion_cuadratica, x, y, maxfev=10000)
                predecir = lambda v: funcion_cuadratica(v, *parametros)
                ecuacion = f"{parametros[0]:.4f}·x² + {parametros[1]:.4f}·x + {parametros[2]:.4f}"
            else:
                #Ajustamos a·e^(-b·x) + c con valores iniciales razonables y b ≥ 0 para evitar desbordes numéricos
                p0 = (y.max() - y.min() or 1.0, 1.0 / (x.mean() or 1.0), y.mean())
                parametros, _ = curve_fit(funcion_exponencial, x, y, p0=p0, maxfev=10000,
                                          bounds=([-np.inf, 0, -np.inf], [np.inf, np.inf, np.inf]))
                predecir = lambda v: funcion_exponencial(v, *parametros)
                ecuacion = f"{parametros[0]:.4f}·e^(-{parametros[1]:.4f}·x) + {parametros[2]:.4f}"
        except (RuntimeError, ValueError) as error:
            #curve_fit lanza RuntimeError si no converge; lo mostramos sin romper el dashboard
            st.error(f"{titulo}: el modelo no convergió ({error}).")
            return

        #Calculamos la correlación entre Y real y Y predicha, y el R²
        y_pred = predecir(x)
        r = correlacion_segura(y, y_pred)
        st.write(f"**Coeficiente de correlación (r):** {r:.4f}")
        st.write(f"**R²:** {r2_score(y, y_pred):.4f}")
        st.write(f"**Ecuación:** y = {ecuacion}")
        st.write(f"**Observaciones:** {len(datos):,}")

        #Graficamos la dispersión (muestra) y la curva de predicción en rojo
        muestra = muestra_para_grafica(datos)
        x_curva = np.linspace(x.min(), x.max(), 200)
        figura = go.Figure()
        figura.add_trace(go.Scatter(x=muestra["unidades_vendidas"], y=muestra["salidas"], mode="markers",
                                    name="Datos", marker=dict(opacity=0.5)))
        figura.add_trace(go.Scatter(x=x_curva, y=predecir(x_curva), mode="lines", name="Predicción",
                                    line=dict(color="red")))
        figura.update_layout(title=titulo, xaxis_title="unidades_vendidas", yaxis_title="salidas", height=400)
        st.plotly_chart(figura, width="stretch")

    #Dividimos los datos en los dos subgrupos de negocio
    df_pico = df[df["es_pico"] == True]
    df_regular = df[df["es_pico"] == False]

    Contenedor_A, Contenedor_B = st.columns(2)
    with Contenedor_A:
        st.subheader("Artículos con pico de demanda")
        ajustar_y_graficar(df_pico, "Meses con pico")
    with Contenedor_B:
        st.subheader("Artículos regulares")
        ajustar_y_graficar(df_regular, "Meses regulares")

##############################################################################
# CONTENIDO DE LA VISTA 4
elif View == "Regresión Logística":
    #REGRESIÓN LOGÍSTICA
    st.title("Regresión Logística")
    st.caption("Objetivo: predecir si un artículo tendrá un pico de consumo en el mes (es_pico).")
    Variables_X = st.sidebar.multiselect("Variables independientes (X)",
                                         options=["salidas", "unidades_vendidas", "existencias", "salidas_tope", "mes"],
                                         default=["salidas", "unidades_vendidas", "existencias"])
    #Avisamos que es_pico se calcula a partir de salidas: usarla como X es fuga de información
    if "salidas" in Variables_X or "salidas_tope" in Variables_X:
        st.warning("⚠️ 'es_pico' se construyó a partir de 'salidas' (salidas > Q3 + 3·IQR del propio SKU). "
                   "Con 'salidas' o 'salidas_tope' como X, el modelo ve la respuesta en los datos; "
                   "para medir capacidad predictiva real conviene quitarlas.")

    if not Variables_X:
        st.info("Seleccionamos al menos una variable independiente en la barra lateral.")
        st.stop()

    #Preparamos X y Y quitando nulos
    datos = df[Variables_X + ["es_pico"]].dropna()
    X = datos[Variables_X]
    y = datos["es_pico"].astype(int)  #Convertimos True/False a 1/0

    #Verificamos que existan las dos clases con suficientes casos para estratificar
    if y.nunique() < 2 or y.value_counts().min() < 2:
        st.warning("La variable es_pico no tiene suficientes casos de ambas clases para entrenar el modelo.")
        st.stop()

    #Dividimos en entrenamiento y prueba (70/30), estratificando para conservar la proporción de picos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=SEMILLA, stratify=y)
    #Escalamos: ajustamos el escalador solo con entrenamiento para no filtrar información de prueba
    escalador = StandardScaler()
    X_train_esc = escalador.fit_transform(X_train)
    X_test_esc = escalador.transform(X_test)

    #Ajustamos el modelo; class_weight='balanced' compensa que los picos son muy pocos
    modelo_log = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEMILLA)
    modelo_log.fit(X_train_esc, y_train)
    y_pred = modelo_log.predict(X_test_esc)

    #Verificamos la matriz de confusión (forzamos el orden de etiquetas 0 y 1)
    matriz = confusion_matrix(y_test, y_pred, labels=[0, 1])
    TN, FP, FN, TP = matriz.ravel()

    Contenedor_A, Contenedor_B = st.columns(2)
    with Contenedor_A:
        st.subheader("Matriz de confusión")
        #Creamos el heatmap interactivo con colores naranjas y etiquetas TN, FP, FN, TP
        etiquetas = [[f"TN: {TN:,}", f"FP: {FP:,}"],
                     [f"FN: {FN:,}", f"TP: {TP:,}"]]
        figure7 = go.Figure(data=go.Heatmap(
            z=matriz,
            x=["Predicho: Regular (0)", "Predicho: Pico (1)"],
            y=["Real: Regular (0)", "Real: Pico (1)"],
            text=etiquetas, texttemplate="%{text}", textfont={"size": 16},
            colorscale="Oranges", showscale=True))
        figure7.update_yaxes(autorange="reversed")  #Dejamos la clase 0 arriba, como en la plantilla
        figure7.update_layout(height=400)
        st.plotly_chart(figure7, width="stretch")

    with Contenedor_B:
        st.subheader("Métricas de evaluación")
        #Calculamos la exactitud y la precisión de cada clase (zero_division=0 evita errores si no hay predicciones)
        exactitud = accuracy_score(y_test, y_pred)
        precision_0 = precision_score(y_test, y_pred, pos_label=0, zero_division=0)
        precision_1 = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        st.metric("Exactitud", f"{exactitud:.4f}")
        st.metric("Precisión clase 0 (Regular)", f"{precision_0:.4f}")
        st.metric("Precisión clase 1 (Pico)", f"{precision_1:.4f}")
        #Mostramos los coeficientes y su razón de momios (odds ratio = e^β) para interpretar
        st.dataframe(pd.DataFrame({"variable": Variables_X,
                                   "coeficiente": modelo_log.coef_[0],
                                   "odds_ratio": np.exp(modelo_log.coef_[0])}).round(4),
                     hide_index=True, width="stretch")
        st.caption(f"Proporción de picos en los datos: {y.mean():.2%}. Con clases tan desbalanceadas, "
                   "la exactitud alta no basta: revisamos también la precisión de la clase Pico.")

##############################################################################
# CONTENIDO DE LA VISTA 5
elif View == "ANOVA":
    #ANÁLISIS DE VARIANZA (ANOVA)
    st.title("ANOVA")
    Variable_Y = st.sidebar.selectbox("Variable dependiente numérica (Y)", options=["salidas", "existencias",
                                                                                     "unidades_vendidas"])
    Variable_X = st.sidebar.selectbox("Variable independiente categórica (X)", options=["marca", "mes", "anio"])

    #Formamos los grupos: una serie de valores de Y por cada categoría de X (mínimo 2 datos por grupo)
    datos = df[[Variable_X, Variable_Y]].dropna()
    grupos_dict = {categoria: valores[Variable_Y].to_numpy()
                   for categoria, valores in datos.groupby(Variable_X) if len(valores) >= 2}
    grupos = list(grupos_dict.values())

    Contenedor_A, Contenedor_B = st.columns(2)
    with Contenedor_A:
        st.subheader("Prueba estadística")
        if len(grupos) < 2:
            st.warning("Necesitamos al menos dos grupos con datos para aplicar ANOVA.")
        elif all(np.var(g) == 0 for g in grupos):
            st.warning("Todos los grupos son constantes: no hay varianza que analizar.")
        else:
            #Aplicamos ANOVA de una vía: H0 = todas las medias de los grupos son iguales
            f_valor, p_valor = stats.f_oneway(*grupos)
            st.write(f"**F-valor:** {f_valor:.3f}")
            st.write(f"**p-valor:** {p_valor:.4f}")
            st.write(f"**Grupos comparados:** {len(grupos)}")
            if p_valor < 0.05:
                st.success("¡Hay diferencias significativas entre los grupos!")
            else:
                st.warning("No hay diferencias significativas entre los grupos (no rechazamos H0).")
            if Variable_Y == "existencias":
                st.caption("Nota: 'existencias' es un solo corte (04/09/2026) repetido en cada mes del SKU, "
                           "por lo que cada artículo pesa tantas veces como meses de historia tenga.")

    with Contenedor_B:
        st.subheader("Media por grupo ± error estándar")
        if grupos:
            #Calculamos la media y el error estándar de la media (sem) de cada grupo
            resumen = datos.groupby(Variable_X)[Variable_Y].agg(["mean", "sem"]).fillna(0)
            resumen = resumen.loc[list(grupos_dict.keys())]  #Conservamos solo los grupos válidos
            #Graficamos las barras en Matplotlib con barras de error
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.bar(resumen.index.astype(str), resumen["mean"], yerr=resumen["sem"], capsize=5,
                   color="skyblue", edgecolor="black")
            ax.set_xlabel(Variable_X)
            ax.set_ylabel(f"Media de {Variable_Y}")
            ax.set_title(f"Media de {Variable_Y} por {Variable_X}")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)  #Cerramos la figura para liberar memoria entre interacciones
