#Creamos el dashboard del Avance 1 con el socio formador (CLV México)
#Diagnóstico de datos, segmentación del catálogo y hoja de ruta hacia Odoo
#Ejecutamos con: streamlit run dashboard_inventario.py   (desde la carpeta Reto)
#Necesita: tratamiento_datos_inventario.py (CSV + reporte) y el notebook Segmentacion_SKU.ipynb (segmentación + perfil)
#####################################################
#Importamos librerías
import os
import json
from pathlib import Path

import base64
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

######################################################
#Configuramos la página con diseño amplio
st.set_page_config(page_title="CLV México | Inventarios", page_icon="📦", layout="wide")

#Definimos las rutas a partir de la carpeta donde vive este archivo (igual que en app.py)
#CLV_DATOS y CLV_REPORTES son opcionales: nos sirven para probar con datos sintéticos sin tocar los reales
BASE_DIR = Path(__file__).resolve().parent
CARPETA_DATOS = Path(os.environ.get("CLV_DATOS", BASE_DIR / "Datos Reales" / "Datos Reales Tratados"))
CARPETA_REPORTES = Path(os.environ.get("CLV_REPORTES", BASE_DIR))
CARPETA_IMG = BASE_DIR / "img_reporte"
RUTA_LOGO_BLANCO = CARPETA_IMG / "logo_clv.png"
RUTA_LOGO_NAVY = CARPETA_IMG / "logo_clv_navy.png"


def logo_base64(ruta):
    """Convierte la imagen a base64 para incrustarla en encabezados HTML."""
    if ruta.exists():
        try:
            with open(ruta, "rb") as f:
                return base64.b64encode(f.read()).decode()
        except Exception:
            return ""
    return ""

CMD_TRATAMIENTO = "python3 tratamiento_datos_inventario.py"
CMD_NOTEBOOK = "el notebook Segmentacion_SKU.ipynb (Restart & Run All)"
ARCHIVOS = {
    "maestro": (CARPETA_DATOS / "maestro_sku.csv", CMD_TRATAMIENTO),
    "panel": (CARPETA_DATOS / "demanda_mensual_sku.csv", CMD_TRATAMIENTO),
    "segmentacion": (CARPETA_DATOS / "segmentacion_sku.csv", CMD_NOTEBOOK),
    "reporte": (CARPETA_REPORTES / "reporte_tratamiento_seguro.json", CMD_TRATAMIENTO),
    "perfil": (CARPETA_REPORTES / "perfil_segmentos_seguro.json", CMD_NOTEBOOK),
}

#Umbrales de Syntetos-Boylan (los mismos del script de tratamiento)
UMBRAL_ADI = 1.32
UMBRAL_CV2 = 0.49

#Umbrales de la lista de revisión (sección 4.3 del plan); los dejamos aquí para ajustarlos fácilmente
UMBRAL_COBERTURA_FALTANTE = 1      #Menos de 1 mes de cobertura en clase A/B = riesgo de faltante
UMBRAL_SOBREINVENTARIO = 12        #Más de 12 meses de cobertura = sobreinventario
CLASES_CRITICAS = ["A", "B"]

#Limitamos los puntos de las dispersiones para que Plotly no se vuelva lento (los cálculos usan todo)
MAX_PUNTOS_GRAFICA = 5000
SEMILLA = 42

#Porcentaje de fechas válidas en ventas antes del tratamiento (CONTEXTO_PROYECTO_CLV.md, sección 7)
PCT_FECHAS_VENTAS_ANTES = 34.54

#Paleta fija: un color por segmento en todas las vistas
COLORES_SEGMENTO = {
    "S0": "#8D99AE", "S1": "#C3C7D1",
    "C1": "#0FA3B1", "C2": "#F25F5C", "C3": "#FFB627", "C4": "#7B61FF",
    "C5": "#3BB273", "C6": "#E56399", "C7": "#2E86AB", "C8": "#A23B72",
}
COLORES_DEMANDA = {"SUAVE": "#3BB273", "ERRATICA": "#FFB627", "INTERMITENTE": "#0FA3B1",
                   "IRREGULAR": "#F25F5C", "INSUFICIENTE": "#8D99AE"}
COLORES_BANDERA = {"RIESGO_FALTANTE": "#F25F5C", "SIN_EXISTENCIA_CON_DEMANDA": "#FF8C42",
                   "INMOVILIZADO": "#8D99AE", "SOBREINVENTARIO": "#7B61FF", "OK": "#3BB273"}

#Políticas sugeridas por tipo de segmento (texto cualitativo; se ajusta cuando haya resultados reales)
#La llave es el nombre del segmento que asigna el notebook
POLITICAS = {
    "Sin movimiento": "Revisar con CLV si se descontinúa, se liquida o se maneja solo bajo pedido. "
                      "No cargar regla de reabastecimiento automático en Odoo.",
    "Historia insuficiente": "Artículos nuevos o con muy pocas salidas. Usar como referencia artículos parecidos "
                             "(misma marca y subcategoría) y revisar manualmente cada mes.",
    "Alta rotación (núcleo del negocio)": "Regla mínimo-máximo automática en Odoo, revisión semanal, "
                                          "máxima prioridad de disponibilidad y colchón para pedidos grandes.",
    "Consumibles baratos de pedido grande y esporádico": "Stock mínimo bajo; como cada pieza cuesta poco, "
                                                        "conviene comprar en lote cuando llegue un pedido grande. Revisión mensual.",
    "Ocasionales (casi sin movimiento)": "Sin reposición automática: comprar contra pedido del cliente. "
                                         "Candidatos a pasar a S0 si siguen sin moverse.",
    "Baja rotación constante (con sobreinventario)": "Demanda pequeña pero predecible: mínimo-máximo con mínimos bajos. "
                                                    "Frenar compras hasta bajar la cobertura.",
    "Equipos y artículos de alto costo": "Compra bajo pedido o stock de 1 pieza solo en modelos clave; "
                                         "cada pieza de más inmoviliza mucho dinero.",
    # Nombres previos por compatibilidad:
    "Alta rotación estable": "Revisión frecuente (semanal) y prioridad máxima de disponibilidad. "
                             "Son los mejores candidatos para reglas mínimo-máximo automáticas en Odoo.",
    "Alta rotación variable": "Se venden seguido pero en cantidades muy distintas. Revisión frecuente "
                              "con un colchón de seguridad mayor para absorber los pedidos grandes.",
    "Costosos de baja rotación": "Pedir bajo demanda o mantener un stock mínimo (1 a 2 piezas). "
                                 "Cada pieza de más inmoviliza mucho dinero.",
    "Esporádicos": "Se piden pocas veces al año. Mantener stock mínimo o comprar contra pedido "
                   "del cliente; revisar mensualmente.",
    "En declive": "Su demanda se está apagando. Dejar de reponer automáticamente, agotar existencias "
                  "y confirmar con ventas si siguen en catálogo.",
    "Rotación media": "Revisión quincenal con regla mínimo-máximo; ajustar el mínimo según el tiempo de "
                      "entrega del proveedor cuando CLV lo comparta.",
}
POLITICA_GENERICA = "Política por definir con el equipo de almacén a partir del perfil del segmento."

#Glosario para personal sin formación en inventarios
GLOSARIO = {
    "SKU": "Código único de cada artículo del catálogo. Un mismo producto en otra presentación es otro SKU.",
    "Rotación": "Qué tan rápido sale un artículo del almacén. Alta rotación = se vende seguido.",
    "Cobertura": "Para cuántos meses alcanzan las existencias actuales al ritmo de venta del último año.",
    "ABC": "Clasifica por importancia económica: A son los pocos artículos que suman el 80 % del valor vendido.",
    "XYZ": "Clasifica por estabilidad: X se vende parecido cada mes; Z cambia mucho de un mes a otro.",
    "Demanda intermitente": "Artículo que tiene muchos meses sin ventas entre una venta y otra.",
    "ADI": "Cada cuántos meses, en promedio, se vende el artículo. ADI = 1 significa que se vende cada mes.",
    "CV²": "Qué tan distintas son las cantidades de cada pedido. Un valor alto = pedidos muy desiguales.",
    "Stock de seguridad": "Piezas extra que se guardan para cubrir ventas inesperadas o retrasos del proveedor.",
    "Punto de reorden": "Nivel de existencias en el que hay que hacer un nuevo pedido para no quedarse sin piezas.",
    "Lead time": "Tiempo que tarda el proveedor desde que se hace el pedido hasta que llega al almacén.",
    "Nivel de servicio": "Porcentaje de veces que queremos tener el artículo disponible cuando el cliente lo pide.",
}

#Datos que necesitamos de CLV (sección 6 de PROPUESTA_STOCK_SEGURIDAD_ODOO.md + pendientes del avance)
DATOS_SOLICITAR = [
    ("Lead time promedio, mínimo y máximo por marca o proveedor", "Calcular el stock de seguridad y el punto de reorden."),
    ("Lote mínimo y múltiplo de empaque", "Definir la cantidad máxima y el múltiplo de la regla de Odoo."),
    ("Frecuencia con que revisan y piden", "Definir el periodo de revisión de la política mínimo-máximo."),
    ("Nivel de servicio deseado o lista de productos que nunca pueden faltar", "Fijar la protección por segmento."),
    ("¿Qué artículos se manejan bajo pedido (sin stock)?", "Separar los agotados reales de los artículos que no se almacenan."),
    ("Productos descontinuados y lanzamientos próximos", "Depurar el segmento sin movimiento y tratar aparte los nuevos."),
    ("Si las salidas de almacén incluyen traspasos o consumo interno", "No inflar la demanda."),
    ("Significado de los folios XP, S, XS, SCR y XU", "Saber si son ventas, devoluciones o movimientos internos."),
    ("Órdenes de compra con fecha de pedido y fecha de recepción", "Medir el lead time real y su variación."),
    ("Archivo «Relación de tiempos de entrega.xlsx»", "Puede cubrir buena parte del lead time."),
    ("Si las líneas idénticas dentro de una factura son legítimas", "No eliminar ni contar dos veces ventas reales."),
    ("Registro de eventos del proceso de compras (solicitud, aprobación, pedido, recepción, con fecha y responsable)",
     "Aplicar minería de procesos para encontrar cuellos de botella en el abastecimiento."),
    ("Versión de Odoo y fecha de arranque prevista", "Entregar las reglas de reabastecimiento en el formato correcto y a tiempo."),
]

VISTAS = [
    "1 · Resumen",
    "2 · Calidad de datos",
    "3 · Exploración y descubrimiento",
    "4 · Comportamiento de la demanda",
    "5 · Segmentación de artículos",
    "6 · Ficha por SKU",
    "7 · Hoja de ruta",
]


######################################################
#Estilos visuales: encabezado con degradado, tarjetas y cajas de interpretación
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Outfit', sans-serif; }
.encabezado {
    background: linear-gradient(120deg, #0B3C5D 0%, #0FA3B1 60%, #3BB273 100%);
    padding: 1.4rem 1.8rem; border-radius: 18px; color: white; margin-bottom: 1.2rem;
    box-shadow: 0 8px 24px rgba(11, 60, 93, 0.25);
}
.encabezado h1 { color: white; margin: 0; font-size: 2.0rem; font-weight: 700; }
.encabezado p { color: rgba(255,255,255,0.88); margin: 0.3rem 0 0 0; font-size: 1.02rem; }
div[data-testid="stMetric"] {
    background: linear-gradient(160deg, rgba(15,163,177,0.08), rgba(123,97,255,0.06));
    border: 1px solid rgba(15,163,177,0.25); border-radius: 14px; padding: 0.8rem 1rem;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
}
div[data-testid="stMetric"]:hover { transform: translateY(-2px); box-shadow: 0 6px 16px rgba(15,163,177,0.18); }
.interp {
    border-left: 4px solid #0FA3B1; background: rgba(15,163,177,0.07);
    padding: 0.55rem 0.9rem; border-radius: 0 10px 10px 0; margin: -0.3rem 0 1.2rem 0; font-size: 0.95rem;
}
.mensaje {
    background: rgba(255,182,39,0.12); border: 1px solid rgba(255,182,39,0.45);
    border-radius: 12px; padding: 0.8rem 1rem; margin-bottom: 0.6rem;
}
.tarjeta-titulo { font-weight: 700; font-size: 1.08rem; margin-bottom: 0.2rem; }
.franja { height: 6px; border-radius: 6px; margin-bottom: 0.6rem; }
</style>
""", unsafe_allow_html=True)

px.defaults.template = "plotly_white"


######################################################
#Creamos funciones auxiliares de formato e interpretación
def pesos(valor):
    return "—" if pd.isna(valor) else f"${valor:,.0f}"


def numero(valor, decimales=0):
    return "—" if pd.isna(valor) else f"{valor:,.{decimales}f}"


def interpretar(texto):
    #Cada gráfica lleva una línea de interpretación pensada para almacén
    st.markdown(f"<div class='interp'>💡 <b>¿Qué significa?</b> {texto}</div>", unsafe_allow_html=True)


def encabezado(titulo, subtitulo):
    b64 = logo_base64(RUTA_LOGO_BLANCO)
    img_tag = (
        f"<img src='data:image/png;base64,{b64}' style='height: 44px; max-width: 140px; object-fit: contain; "
        f"filter: drop-shadow(0 2px 6px rgba(0,0,0,0.25)); margin-left: 1rem;' alt='CLV Logo' />"
        if b64 else ""
    )
    st.markdown(
        f"<div class='encabezado' style='display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 0.8rem;'>"
        f"  <div style='flex: 1; min-width: 250px;'><h1>{titulo}</h1><p>{subtitulo}</p></div>"
        f"  {img_tag}"
        f"</div>",
        unsafe_allow_html=True,
    )


def a_booleano(serie):
    #En un CSV los booleanos pueden venir como texto: los convertimos de forma explícita
    mapa = {"true": True, "1": True, "1.0": True, "false": False, "0": False, "0.0": False}
    return serie.astype(str).str.strip().str.lower().map(mapa).fillna(False).astype(bool)


def muestra_para_grafica(datos):
    #Solo para graficar; los cálculos siempre usan todos los datos
    if len(datos) > MAX_PUNTOS_GRAFICA:
        return datos.sample(MAX_PUNTOS_GRAFICA, random_state=SEMILLA)
    return datos


def politica_de(nombre_segmento):
    return POLITICAS.get(str(nombre_segmento).strip(), POLITICA_GENERICA)


def orden_segmentos(segmentos):
    #Ordenamos S0, S1, C1, C2, ... para que todas las gráficas sigan el mismo orden
    return sorted(set(segmentos), key=lambda s: (s[0] != "S", int(s[1:]) if s[1:].isdigit() else 99))


######################################################
#Verificamos que existan todos los archivos de entrada antes de cargar
def verificar_archivos():
    faltantes = [(ruta.name, comando) for ruta, comando in ARCHIVOS.values() if not ruta.exists()]
    if faltantes:
        encabezado("CLV MÉXICO | INVENTARIOS", "Faltan archivos de entrada")
        lineas = "\n".join(f"- **{nombre}** → ejecuta {comando}" for nombre, comando in faltantes)
        st.error(f"No encontramos los siguientes archivos:\n\n{lineas}\n\n"
                 "Primero corre `python3 tratamiento_datos_inventario.py` y después el notebook "
                 "`Segmentacion_SKU.ipynb`; luego recarga esta página.")
        st.stop()


######################################################
#Definimos la carga con caché para no releer los archivos en cada interacción
@st.cache_data(show_spinner="Cargando datos de inventario…")
def cargar_datos(rutas):
    maestro = pd.read_csv(rutas["maestro"])
    panel = pd.read_csv(rutas["panel"])
    segmentacion = pd.read_csv(rutas["segmentacion"])
    with open(rutas["reporte"], encoding="utf-8") as archivo:
        reporte = json.load(archivo)
    with open(rutas["perfil"], encoding="utf-8") as archivo:
        perfil = json.load(archivo)

    #Aseguramos el mismo tipo de llave en las tres tablas
    for df in (maestro, panel, segmentacion):
        df["articulo_id"] = pd.to_numeric(df["articulo_id"], errors="coerce").astype("Int64")

    #Convertimos tipos de la segmentación
    for col in ["costo_imputado", "sin_registro_existencia"]:
        segmentacion[col] = a_booleano(segmentacion[col])
    numericas = ["demanda_12m", "costo_unitario", "adi_24m", "cv2_24m", "meses_sin_demanda", "existencias",
                 "valor_inventario_corte_ultimo", "cobertura_meses", "pc1", "pc2"]
    for col in numericas:
        segmentacion[col] = pd.to_numeric(segmentacion[col], errors="coerce")
    segmentacion["valor_inventario_corte_ultimo"] = segmentacion["valor_inventario_corte_ultimo"].fillna(0)
    segmentacion["marca"] = segmentacion["marca"].fillna("SIN MARCA")

    #Traemos del maestro las columnas que solo usamos en la ficha y en la vista de demanda
    extras = maestro[["articulo_id", "adi", "cv2", "fecha_ultima_compra", "dias_promedio_entre_compras",
                      "n_meses_historia", "clase_demanda"]].copy()
    for col in ["adi", "cv2", "dias_promedio_entre_compras", "n_meses_historia"]:
        extras[col] = pd.to_numeric(extras[col], errors="coerce")
    extras["fecha_ultima_compra"] = pd.to_datetime(extras["fecha_ultima_compra"], errors="coerce")
    #Si en segmentacion falta clase_demanda actualizada, tomamos la del maestro
    if "clase_demanda" in segmentacion.columns:
        extras = extras.drop(columns=["clase_demanda"])
    base = segmentacion.merge(extras, on="articulo_id", how="left")
    base = calcular_banderas(base)

    #Preparamos el panel y lo unimos con marca y segmento
    panel["periodo"] = pd.to_datetime(panel["periodo"], errors="coerce")
    for col in ["salidas", "unidades_vendidas"]:
        panel[col] = pd.to_numeric(panel[col], errors="coerce").fillna(0)
    panel["es_pico"] = a_booleano(panel["es_pico"])
    panel["anio"] = panel["periodo"].dt.year
    panel["mes"] = panel["periodo"].dt.month
    panel = panel.merge(base[["articulo_id", "marca", "segmento"]], on="articulo_id", how="left")
    return base, panel, reporte, perfil


######################################################
#Creamos las banderas de la lista de revisión (la primera regla que se cumple gana)
def calcular_banderas(df):
    df = df.copy()
    cobertura = df["cobertura_meses"]
    condiciones = [
        df["clase_abc"].isin(CLASES_CRITICAS) & (cobertura < UMBRAL_COBERTURA_FALTANTE),
        df["sin_registro_existencia"] & (df["demanda_12m"] > 0),
        (df["segmento"] == "S0") & (df["existencias"] > 0),
        cobertura > UMBRAL_SOBREINVENTARIO,
    ]
    banderas = ["RIESGO_FALTANTE", "SIN_EXISTENCIA_CON_DEMANDA", "INMOVILIZADO", "SOBREINVENTARIO"]
    df["bandera_revision"] = np.select(condiciones, banderas, default="OK")

    #Redactamos un motivo corto y legible para el almacén (C3: considerar opción bajo pedido)
    def motivo(fila):
        b, cob = fila["bandera_revision"], fila["cobertura_meses"]
        if b == "RIESGO_FALTANTE":
            return f"Artículo clase {fila['clase_abc']} con solo {cob:.1f} meses de cobertura: revisar si hay pedido en camino."
        if b == "SIN_EXISTENCIA_CON_DEMANDA":
            return (f"Salieron {fila['demanda_12m']:,.0f} piezas en el último año y hoy no hay existencias: "
                    "confirmar si es un agotado o si el artículo se maneja bajo pedido.")
        if b == "INMOVILIZADO":
            if str(fila.get("clase_demanda", "")).strip() == "SIN MOVIMIENTO DESDE 2021":
                return f"Sin salidas desde 2021 y tiene {fila['existencias']:,.0f} piezas: evaluar liquidar o descontinuar."
            return f"Lleva {fila['meses_sin_demanda']:,.0f} meses sin salidas y tiene {fila['existencias']:,.0f} piezas: evaluar liquidar o descontinuar."
        if b == "SOBREINVENTARIO":
            return f"Las existencias alcanzan para {cob:,.1f} meses al ritmo actual: frenar compras."
        return "Sin alerta por cobertura."
    df["motivo"] = df.apply(motivo, axis=1)
    return df


######################################################
#Cargamos los datos (o detenemos la app con un mensaje claro si falta algo)
verificar_archivos()
try:
    base, panel, reporte, perfil = cargar_datos({k: str(v[0]) for k, v in ARCHIVOS.items()})
except Exception as error:  #Si un archivo viene dañado mostramos un mensaje en lugar del traceback
    st.error(f"No pudimos leer los archivos de entrada ({type(error).__name__}: {error}). "
             f"Vuelve a ejecutar {CMD_TRATAMIENTO} y {CMD_NOTEBOOK}.")
    st.stop()

NOMBRES_SEGMENTO = base.groupby("segmento")["nombre_segmento"].first().to_dict()
SEGMENTOS = orden_segmentos(base["segmento"].dropna())


def etiqueta_segmento(seg):
    return f"{seg} · {NOMBRES_SEGMENTO.get(seg, '')}"


######################################################
#Barra lateral: logo, título, navegación, filtros globales y glosario
if RUTA_LOGO_NAVY.exists():
    st.sidebar.image(str(RUTA_LOGO_NAVY), width=160)
elif RUTA_LOGO_BLANCO.exists():
    st.sidebar.image(str(RUTA_LOGO_BLANCO), width=160)

st.sidebar.markdown("## INVENTARIOS")
st.sidebar.caption("Avance 1 · Diagnóstico y segmentación del catálogo")
vista = st.sidebar.radio("Navegación", VISTAS, key="navegacion")

st.sidebar.markdown("### Filtros")
st.sidebar.caption("Aplican a las vistas 1, 4 y 5 y a la lista de revisión.")
marcas_sel = st.sidebar.multiselect("Marca", sorted(base["marca"].unique()), placeholder="Todas las marcas")
segmentos_sel = st.sidebar.multiselect("Segmento", SEGMENTOS, format_func=etiqueta_segmento,
                                       placeholder="Todos los segmentos")

with st.sidebar.expander("Glosario"):
    for termino, definicion in GLOSARIO.items():
        st.markdown(f"**{termino}:** {definicion}")

#Aplicamos los filtros (una lista vacía significa "todos")
mascara = pd.Series(True, index=base.index)
if marcas_sel:
    mascara &= base["marca"].isin(marcas_sel)
if segmentos_sel:
    mascara &= base["segmento"].isin(segmentos_sel)
base_f = base[mascara]
panel_f = panel[panel["articulo_id"].isin(base_f["articulo_id"])]
hay_filtros = bool(marcas_sel or segmentos_sel)


def aviso_vacio():
    if base_f.empty:
        st.warning("No hay SKUs con la combinación de filtros elegida. Quita algún filtro en la barra lateral.")
        st.stop()


######################################################
#Vista 1 — Resumen
def vista_resumen():
    encabezado("Resumen del inventario", "Cómo está hoy el catálogo de CLV")
    aviso_vacio()
    activos = base_f["segmento"].str.startswith("C")
    s0 = base_f["segmento"] == "S0"
    valor_total = base_f["valor_inventario_corte_ultimo"].sum()
    valor_s0 = base_f.loc[s0, "valor_inventario_corte_ultimo"].sum()
    pct_intermitente = 100 * base_f["clase_demanda"].isin(["INTERMITENTE", "IRREGULAR"]).mean()

    # C2: Distinguir SKUs sin movimiento desde 2021 vs sin movimiento en 12+ meses
    sin_mov_2021 = s0 & (base_f["clase_demanda"] == "SIN MOVIMIENTO DESDE 2021")
    sin_salidas_12m = s0 & ~sin_mov_2021

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("SKUs analizados", numero(len(base_f)))
    c2.metric("SKUs activos (clusters)", numero(activos.sum()), f"{100 * activos.mean():.0f} % del catálogo", delta_color="off")
    c3.metric("Valor del inventario (último corte)", pesos(valor_total))
    c4.metric("Demanda intermitente o irregular", f"{pct_intermitente:.1f} %")
    c5, c6, c7, _ = st.columns(4)
    subtexto_s0 = f"{sin_mov_2021.sum():,} sin salidas desde 2021 · {sin_salidas_12m.sum():,} en 12+ meses" if sin_mov_2021.sum() > 0 else f"{s0.sum():,} sin salidas en 12+ meses"
    c5.metric("SKUs sin movimiento (S0)", numero(s0.sum()), subtexto_s0, delta_color="off")
    c6.metric("Valor inmovilizado en S0", pesos(valor_s0),
              f"{100 * valor_s0 / valor_total:.1f} % del valor" if valor_total else None, delta_color="off")
    c7.metric("Cobertura mediana", f"{numero(base_f['cobertura_meses'].median(), 1)} meses")

    #Barras de valor de inventario por segmento
    valor_seg = (base_f.groupby("segmento").agg(valor=("valor_inventario_corte_ultimo", "sum"),
                                                n_skus=("articulo_id", "size")).reset_index())
    valor_seg["etiqueta"] = valor_seg["segmento"].map(etiqueta_segmento)
    valor_seg = valor_seg.set_index("segmento").loc[orden_segmentos(valor_seg["segmento"])].reset_index()
    fig = px.bar(valor_seg, x="etiqueta", y="valor", color="segmento", color_discrete_map=COLORES_SEGMENTO,
                 text=valor_seg["valor"].map(pesos), hover_data={"n_skus": True, "segmento": False},
                 labels={"etiqueta": "Segmento", "valor": "Valor de inventario ($)", "n_skus": "SKUs"},
                 title="Valor de inventario por segmento")
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, height=430)
    st.plotly_chart(fig, width="stretch")
    mayor = valor_seg.sort_values("valor", ascending=False).iloc[0]
    interpretar(f"Cada barra es el dinero que hoy está en el almacén para ese grupo de artículos. "
                f"El grupo con más dinero es <b>{mayor['etiqueta']}</b> ({pesos(mayor['valor'])}). "
                f"Si una barra grande corresponde a artículos que casi no se venden, ahí hay capital detenido.")

    #Tres mensajes clave calculados con los datos
    st.markdown("### Mensajes clave")
    n_suave = (base_f["clase_demanda"] == "SUAVE").sum()
    riesgo = (base_f["bandera_revision"] == "RIESGO_FALTANTE").sum()
    sobre = (base_f["bandera_revision"] == "SOBREINVENTARIO").sum()
    mensajes = [
        f"<b>{pesos(valor_s0)}</b> ({100 * valor_s0 / valor_total if valor_total else 0:.1f} % del inventario) está en "
        f"{s0.sum():,} artículos que llevan 12 meses o más sin salir del almacén.",
        f"Solo <b>{n_suave:,} de {len(base_f):,} SKUs</b> ({100 * n_suave / len(base_f):.1f} %) tienen una demanda pareja; "
        f"el {pct_intermitente:.0f} % se vende de forma intermitente o irregular. Una sola regla de reorden no sirve para todos.",
        f"<b>{riesgo:,}</b> artículos importantes (clase A o B) tienen menos de {UMBRAL_COBERTURA_FALTANTE} mes de cobertura "
        f"y <b>{sobre:,}</b> tienen existencias para más de {UMBRAL_SOBREINVENTARIO} meses.",
    ]
    for m in mensajes:
        st.markdown(f"<div class='mensaje'>📌 {m}</div>", unsafe_allow_html=True)

    seccion_lista_revision()


######################################################
#Lista de revisión exportable (respeta los filtros activos)
def seccion_lista_revision():
    st.markdown("### Lista de revisión de inventario")
    st.info("Estas banderas son **señales de revisión** basadas en cobertura; todavía no consideran el "
            "tiempo de entrega (lead time) ni el stock de seguridad, que son la fase siguiente del proyecto.")
    conteo = (base_f["bandera_revision"].value_counts()
              .reindex(list(COLORES_BANDERA)).fillna(0).astype(int).reset_index())
    conteo.columns = ["bandera", "skus"]
    columnas = st.columns(len(conteo))
    for col, (_, fila) in zip(columnas, conteo.iterrows()):
        col.metric(fila["bandera"].replace("_", " ").capitalize(), numero(fila["skus"]))

    # C3: Nota visible sobre posibles artículos bajo pedido
    st.caption("Nota: Muchos de los artículos marcados con 'sin existencia con demanda' podrían ser "
               "productos que se manejan bajo pedido (sin stock); lo confirmaremos en las mesas de trabajo con CLV.")

    columnas_csv = ["articulo_id", "nombre_articulo", "marca", "segmento", "nombre_segmento", "clase_abc", "clase_xyz",
                    "clase_demanda", "existencias", "demanda_12m", "cobertura_meses", "valor_inventario_corte_ultimo",
                    "bandera_revision", "motivo"]
    lista = base_f[columnas_csv].sort_values(["bandera_revision", "valor_inventario_corte_ultimo"], ascending=[True, False])
    st.download_button("⬇️ Descargar lista_revision_inventario.csv",
                       data=lista.to_csv(index=False).encode("utf-8-sig"),
                       file_name="lista_revision_inventario.csv", mime="text/csv", key="descarga_lista")
    st.caption(f"El archivo incluye {len(lista):,} SKUs" + (" con los filtros activos." if hay_filtros else "."))


######################################################
#Vista 2 — Calidad de datos (todo desde reporte_tratamiento_seguro.json)
def vista_calidad():
    encabezado("Calidad de datos", "Qué encontramos en los datos, qué corregimos y qué riesgo implica para Odoo")
    hojas = reporte.get("hojas", {})
    marcas = reporte.get("marcas", {})
    llaves = reporte.get("llaves", {}).get("cobertura_articulo_id", {})
    maestro_rep = reporte.get("maestro_sku", {})

    pct_fechas = hojas.get("Ventas_CLV_2022_2026", {}).get("fechas_fecha_documento", {}).get("pct_validas")
    mapeos = marcas.get("mapeos_aplicados", {})
    pct_llaves = [v.get("pct_origen") for v in llaves.values() if v.get("pct_origen") is not None]
    prefijos = {}
    for hoja in hojas.values():
        for pref, n in hoja.get("prefijos_sin_tipo_pago", {}).items():
            prefijos[pref] = prefijos.get(pref, 0) + n
    duplicados = sum(h.get("duplicados_exactos", 0) for h in hojas.values())
    cierres = hojas.get("Negocios_CRM", {}).get("cierres_antes_de_creacion", 0)

    hallazgos = pd.DataFrame([
        ["Fechas de venta mal leídas",
         f"Fechas válidas en ventas: {PCT_FECHAS_VENTAS_ANTES} % → {pct_fechas} %",
         "Convertimos con formato día-mes-año explícito.",
         "Si Odoo importa fechas con el formato equivocado, el historial de ventas queda con día y mes invertidos."],
        ["Marcas escritas de varias formas",
         f"{len(mapeos)} abreviaturas homologadas ({', '.join(f'{k}→{v}' for k, v in list(mapeos.items())[:4])}…)",
         "Unificamos mayúsculas, acentos y abreviaturas.",
         "Proveedores duplicados en Odoo: compras y reportes partidos por marca."],
        ["Llaves de artículo distintas entre tablas",
         f"Cobertura de articulo_id entre tablas: {min(pct_llaves):.0f} % a {max(pct_llaves):.0f} %" if pct_llaves else "Sin dato",
         "Resolvimos SKU contra clave_articulo y marcamos los cruces.",
         "Artículos sin equivalencia se cargarían como productos nuevos y perderían su historial."],
        ["Prefijos de folio sin significado",
         ", ".join(f"{k}: {v:,}" for k, v in sorted(prefijos.items(), key=lambda t: -t[1])) + " líneas",
         "Quedaron como POR CONFIRMAR; no se asumió el tipo de pago.",
         "Si son devoluciones o traspasos, la demanda estaría inflada."],
        ["Registros duplicados", f"{duplicados:,} filas idénticas eliminadas",
         "Eliminamos duplicados exactos entre y dentro de las bases.",
         "Ventas contadas dos veces inflan pronósticos y compras."],
        ["SKUs sin costo o sin clasificación",
         f"{maestro_rep.get('sin_costo_unitario', 0):,} sin costo · {maestro_rep.get('sin_clasificacion', 0):,} sin categoría",
         "Imputamos costo con artículos parecidos y lo marcamos.",
         "Sin costo no se puede valorar el inventario ni calcular la inversión en Odoo."],
        ["Fechas del CRM incoherentes", f"{cierres:,} negocios cierran antes de su fecha de creación",
         "Dejamos vacío el ciclo de venta en esos casos.",
         "La fecha de creación parece ser la de una migración: hay que conservar las fechas reales al migrar."],
    ], columns=["Problema", "Evidencia", "Qué hicimos", "Riesgo para Odoo"])
    st.dataframe(hallazgos, width="stretch", hide_index=True)
    interpretar("Cada fila es un problema real que encontramos en los archivos del sistema actual. "
                "Ya lo corregimos para el análisis, pero si los datos se cargan así a Odoo, el problema se hereda.")

    #Gráfica de cobertura de llaves entre tablas
    if llaves:
        df_llaves = pd.DataFrame([{"cruce": k, "pct": v.get("pct_origen")} for k, v in llaves.items()]).sort_values("pct")
        fig = px.bar(df_llaves, x="pct", y="cruce", orientation="h", text=df_llaves["pct"].map(lambda x: f"{x:.0f} %"),
                     color="pct", color_continuous_scale=["#F25F5C", "#FFB627", "#3BB273"], range_color=[40, 100],
                     labels={"pct": "% de artículos de la tabla origen que se encuentran en la destino", "cruce": ""},
                     title="¿Qué tanto coinciden los artículos entre tablas?")
        fig.update_layout(height=560, coloraxis_showscale=False)
        st.plotly_chart(fig, width="stretch")
        interpretar("Una barra al 100 % significa que todos los artículos de una tabla existen en la otra. "
                    "Las barras cortas indican artículos que solo aparecen en un archivo: hay que depurarlos antes de cargarlos a Odoo.")

    st.markdown("### Alertas del tratamiento")
    for alerta in reporte.get("alertas", []):
        st.markdown(f"- {alerta}")

    st.warning("**Mensaje para CLV:** Odoo no limpia los datos por sí solo. Si los datos maestros (artículos, marcas, "
               "costos y fechas) entran sucios, el nuevo sistema hereda los mismos problemas y las reglas de "
               "reabastecimiento se calcularán sobre información equivocada.")


######################################################
#Vista 3 — Exploración y descubrimiento (C5: petición de los profesores)
RECORRIDO_CRISP = [
    {"Fase CRISP-DM": "Comprensión de datos", "Análisis": "Univariado de artículos y compras", "Archivo": "Analisis_Articulos.ipynb, Analisis_Compras.ipynb", "Aporte": "Estructura del catálogo y estacionalidad de compras"},
    {"Fase CRISP-DM": "Comprensión de datos", "Análisis": "Perfilado seguro de metadatos", "Archivo": "socio.ipynb, socio2.ipynb, perfil_*.json", "Aporte": "Calidad, llaves y cardinalidades sin exponer datos"},
    {"Fase CRISP-DM": "Preparación", "Análisis": "Tratamiento integral", "Archivo": "tratamiento_datos_inventario.py", "Aporte": "Fechas, marcas, llaves, panel de demanda y maestro de SKUs"},
    {"Fase CRISP-DM": "Exploración", "Análisis": "ANOVA (10 hipótesis)", "Archivo": "Analisis_ANOVA_CLV.ipynb", "Aporte": "La marca explica ~20 % de la variación de existencias"},
    {"Fase CRISP-DM": "Modelado", "Análisis": "Regresión lineal de ventas", "Archivo": "Regresion_Ventas_Analisis_Completo.ipynb", "Aporte": "Lección sobre variables que contienen la respuesta"},
    {"Fase CRISP-DM": "Modelado", "Análisis": "Regresión logística (CRM y compra por volumen)", "Archivo": "Regresion_Logistica_CLV.ipynb, A01738830_EvaluacionCLV.ipynb", "Aporte": "Manejo de clases desbalanceadas"},
    {"Fase CRISP-DM": "Modelado", "Análisis": "Clustering de clientes", "Archivo": "Actividad_Algoritmos_Agrupamiento.ipynb", "Aporte": "Tipos de cliente que generan la demanda"},
    {"Fase CRISP-DM": "Modelado", "Análisis": "Clustering de artículos", "Archivo": "Segmentacion_SKU.ipynb", "Aporte": "Base de la política de inventario"},
]

INTERPRETACION_ANOVA = {
    1: "El ticket cambia según la forma de pago: crédito y tienda en línea facturan más por línea que el contado. Efecto pequeño (η² = 0.03).",
    2: "El importe difiere entre los 5 estados principales, pero la ubicación geográfica explica muy poco de la variación total (η² = 0.016).",
    3: "La forma de pago importa distinto según la marca: las marcas de equipo (p. ej. Amann Girrbach) dependen estructuralmente del financiamiento a crédito; las de consumo diario (Vita) se venden igual en cualquier modalidad.",
    4: "No significativo (p = 0.87): el valor de una cotización no depende de si el cliente es nuevo ni de si se prospectó en ruta. Depende de la escala clínica del cliente.",
    5: "El valor previsto de los negocios cambia significativamente según el pipeline de HubSpot (η² = 0.053).",
    6: "Hallazgo principal para inventario: la marca comercial explica casi el 20 % de toda la variabilidad de existencias físicas en el almacén (efecto grande, η² = 0.1952). Justifica plenamente políticas de reorden diferenciadas por marca.",
    7: "El volumen comprado a proveedores cambia por marca y año con interacción: cada marca tuvo su propio ritmo y temporalidad de compras.",
    8: "Hay estacionalidad mensual estadísticamente significativa en las salidas, pero el efecto es débil a nivel de artículo individual (η² = 0.006). Se considera en el pronóstico sin esperar que domine.",
    9: "Hay representantes comerciales especializados por marca: la interacción vendedor × marca es altamente significativa.",
    10: "Las salidas de almacén difieren por marca comercial (η² = 0.059, casi mediano). Refuerza y complementa el Caso 6.",
}

IMAGENES_ANOVA = {
    1: "anova_caso1_ventas_pago.png",
    2: "anova_caso2_ventas_estado.png",
    3: "anova_caso3_marca_x_pago.png",
    4: "anova_caso4_crm_cliente_x_ruta.png",
    5: "anova_caso5_crm_pipeline.png",
    6: "anova_caso6_existencias_marca.png",
    7: "anova_caso7_compras_marca_x_ano.png",
    8: "anova_caso8_salidas_estacionalidad.png",
    9: "anova_caso9_vendedor_x_marca.png",
    10: "anova_caso10_movimientos_almacen.png",
}

AREA_CASOS = {
    1: "Ventas", 2: "Ventas", 3: "Ventas", 4: "CRM", 5: "CRM",
    6: "Inventario y compras", 7: "Inventario y compras", 8: "Inventario y compras",
    9: "Ventas", 10: "Inventario y compras"
}

def vista_exploracion():
    encabezado("Exploración y descubrimiento", "Hallazgos analíticos previos que fundamentan la estrategia de inventario")
    st.caption("Los filtros globales de marca y segmento no aplican a esta vista (resultados analíticos del Reto).")

    tab_recorrido, tab_anova, tab_clientes, tab_otros = st.tabs([
        "Recorrido del análisis", "ANOVA", "Segmentación de clientes", "Otros modelos del Reto"
    ])

    with tab_recorrido:
        st.markdown("### Mapa de análisis y modelos del Reto")
        st.dataframe(pd.DataFrame(RECORRIDO_CRISP), width="stretch", hide_index=True)
        interpretar("Este recorrido resume cómo cada fase previa aportó insumos metodológicos y de calidad de datos "
                    "para llegar a la segmentación y política de inventario actual.")

    with tab_anova:
        st.markdown("### Análisis de Varianza (ANOVA — 10 hipótesis de negocio)")
        ruta_anova = BASE_DIR / "resultados_anova_resumen.csv"
        if ruta_anova.exists():
            df_anova = pd.read_csv(ruta_anova)
            df_anova["Num_Caso"] = df_anova["Caso"].str.extract(r"Caso\s+(\d+)")[0].astype(int)
            df_anova["Área"] = df_anova["Num_Caso"].map(AREA_CASOS)
            df_anova["Tamaño del efecto"] = np.select(
                [df_anova["Eta_Cuadrado"] < 0.06, df_anova["Eta_Cuadrado"] < 0.14],
                ["Pequeño", "Mediano"], default="Grande"
            )
            df_anova["p"] = df_anova["P_Valor"].map(lambda p: f"{p:.2e}" if p < 0.001 else f"{p:.4f}")

            area_sel = st.selectbox("Filtrar por área temática", ["Todas", "Inventario y compras", "Ventas", "CRM"], key="area_anova")
            df_mostrar = df_anova if area_sel == "Todas" else df_anova[df_anova["Área"] == area_sel]

            columnas_tabla = ["Caso", "Área", "Variable_Y", "Factores", "F_Statistic", "p", "Eta_Cuadrado", "Tamaño del efecto", "Significativo"]
            st.dataframe(
                df_mostrar[columnas_tabla].rename(columns={"F_Statistic": "F", "Eta_Cuadrado": "η²"}),
                width="stretch", hide_index=True,
                column_config={"η²": st.column_config.NumberColumn(format="%.4f"), "F": st.column_config.NumberColumn(format="%.2f")}
            )

            # Gráfica de barras de Eta Cuadrado
            fig_eta = px.bar(
                df_anova.sort_values("Eta_Cuadrado", ascending=True),
                x="Eta_Cuadrado", y="Caso", orientation="h", color="Área",
                color_discrete_map={"Inventario y compras": "#0FA3B1", "Ventas": "#7B61FF", "CRM": "#FFB627"},
                labels={"Eta_Cuadrado": "Tamaño del efecto (η² parcial)", "Caso": ""},
                title="Magnitud del efecto por hipótesis (η² parcial)"
            )
            fig_eta.add_vline(x=0.06, line_dash="dash", line_color="gray", annotation_text="Efecto mediano (0.06)")
            fig_eta.add_vline(x=0.14, line_dash="dash", line_color="crimson", annotation_text="Efecto grande (0.14)")
            fig_eta.update_layout(height=480)
            st.plotly_chart(fig_eta, width="stretch")

            # Selector de caso interactivo para ver la gráfica
            st.markdown("#### Detalle visual por caso")
            caso_elegido = st.selectbox(
                "Selecciona un caso para inspeccionar su gráfica:",
                range(1, 11),
                format_func=lambda i: f"Caso {i}: {df_anova.loc[df_anova['Num_Caso'] == i, 'Caso'].iloc[0].split(': ', 1)[-1]}",
                key="caso_anova_detalle"
            )
            col_img, col_txt = st.columns([1.2, 1])
            with col_img:
                img_path = BASE_DIR / "img_reporte" / IMAGENES_ANOVA.get(caso_elegido, "")
                if img_path.exists():
                    st.image(str(img_path), caption=f"Caso {caso_elegido}", width="stretch")
                else:
                    st.info(f"Gráfica no encontrada en img_reporte ({img_path.name})")
            with col_txt:
                with st.container(border=True):
                    st.markdown("**Interpretación estadística y de negocio:**")
                    st.markdown(INTERPRETACION_ANOVA.get(caso_elegido, "Sin interpretación disponible."))
                    fila_c = df_anova[df_anova["Num_Caso"] == caso_elegido].iloc[0]
                    st.markdown(f"- **Área:** {fila_c['Área']}")
                    st.markdown(f"- **Variable dependiente ($Y$):** `{fila_c['Variable_Y']}`")
                    st.markdown(f"- **Factores analizados:** `{fila_c['Factores']}`")
                    st.markdown(f"- **Estadístico $F$:** {fila_c['F_Statistic']:,.2f} | **$p$-valor:** {fila_c['p']}")
                    st.markdown(f"- **Tamaño del efecto ($\\eta^2$):** {fila_c['Eta_Cuadrado']:.4f} ({fila_c['Tamaño del efecto']})")

            st.markdown(
                "<div class='mensaje'>📌 <b>Mensaje central:</b> Con más de 100 mil registros casi todo sale estadísticamente significativo (p < 0.05); "
                "el tamaño del efecto (η²) es lo que realmente dice cuánto importa cada factor. Solo la <b>marca comercial frente a las existencias "
                "(Caso 6, η² = 0.1952)</b> tiene un efecto grande en todo el estudio, explicando casi el 20 % de la variación del inventario. "
                "Esto respalda científicamente la necesidad de políticas de reorden diferenciadas por marca comercial.</div>",
                unsafe_allow_html=True
            )
            st.caption("Nota de método: Calculado sobre los datos antes del tratamiento (sin homologar marcas ni eliminar duplicados). "
                       "Las conclusiones de magnitud se mantienen; se recalculará con datos tratados en la siguiente fase.")
        else:
            st.warning("No se encontró resultados_anova_resumen.csv.")

    with tab_clientes:
        st.markdown("### Segmentación de clientes (Clustering de demanda)")
        ruta_perfil_cli = CARPETA_REPORTES / "perfil_clientes_seguro.json"
        ruta_csv_cli = CARPETA_DATOS / "segmentacion_clientes.csv"

        if not ruta_perfil_cli.exists() or not ruta_csv_cli.exists():
            st.info("Para generar y visualizar la segmentación de clientes actualizada sobre los datos tratados, "
                    "ejecuta en tu terminal:\n\n```bash\npython3 exportar_segmentacion_clientes.py\n```")
        else:
            with open(ruta_perfil_cli, encoding="utf-8") as f:
                perfil_cli = json.load(f)
            df_cli = pd.read_csv(ruta_csv_cli)

            c1, c2, c3 = st.columns(3)
            c1.metric("Clientes únicos analizados", f"{len(df_cli):,}")
            c2.metric("Clusters formados (k)", str(perfil_cli.get("k_elegido", 3)))
            c3.metric("Coincidencia K-Means vs Ward (ARI)", f"{perfil_cli.get('ari_kmeans_vs_ward', 0):.3f}")

            # Perfil por cluster de clientes
            st.markdown("#### Perfil de clientes por segmento")
            filas_cli = []
            for nom_c, d_c in perfil_cli.get("clusters", {}).items():
                med = d_c.get("medianas", {})
                filas_cli.append({
                    "Segmento de cliente": nom_c,
                    "Clientes": d_c.get("n_clientes"),
                    "% Clientes": f"{d_c.get('pct_clientes'):.1f} %",
                    "Ventas acumuladas": d_c.get("total_ventas_acumuladas"),
                    "% Ventas": f"{d_c.get('pct_ventas'):.1f} %",
                    "Ticket promedio (med.)": med.get("ticket_promedio"),
                    "Pedidos (med.)": med.get("frecuencia_pedidos"),
                    "Variedad SKUs (med.)": med.get("variedad_articulos"),
                })
            st.dataframe(pd.DataFrame(filas_cli), width="stretch", hide_index=True, column_config={
                "Ventas acumuladas": st.column_config.NumberColumn(format="$%,.0f"),
                "Ticket promedio (med.)": st.column_config.NumberColumn(format="$%,.0f"),
                "Clientes": st.column_config.NumberColumn(format="%,.0f"),
                "Pedidos (med.)": st.column_config.NumberColumn(format="%,.0f"),
                "Variedad SKUs (med.)": st.column_config.NumberColumn(format="%,.0f"),
            })

            # Gráficas de método del codo y silhouette por k (R3)
            metricas_k = perfil_cli.get("metricas_por_k", [])
            if metricas_k:
                st.markdown("#### Selección de clusters (k): Método del codo y Silhouette")
                df_k = pd.DataFrame(metricas_k)
                col_codo, col_sil = st.columns(2)
                with col_codo:
                    fig_codo = px.line(df_k, x="k", y="inercia", markers=True, title="Método del codo (Inercia)",
                                       labels={"k": "Número de clusters (k)", "inercia": "Inercia"})
                    fig_codo.add_vline(x=3, line_dash="dash", line_color="#F25F5C", annotation_text="k = 3 seleccionado")
                    fig_codo.update_layout(height=300)
                    st.plotly_chart(fig_codo, width="stretch")
                with col_sil:
                    fig_sil = px.line(df_k, x="k", y="silhouette", markers=True, title="Coeficiente de Silhouette por k",
                                      labels={"k": "Número de clusters (k)", "silhouette": "Silhouette score"})
                    fig_sil.add_vline(x=3, line_dash="dash", line_color="#F25F5C", annotation_text="k = 3 seleccionado")
                    fig_sil.update_layout(height=300)
                    st.plotly_chart(fig_sil, width="stretch")

                # Advertencia de método exigida en R3 con tamaños calculados desde el JSON
                tamanos_c = [f"{v['n_clientes']:,}" for v in perfil_cli.get("clusters", {}).values()]
                txt_tams = f"{tamanos_c[0]}, {tamanos_c[1]} y {tamanos_c[2]}" if len(tamanos_c) == 3 else ", ".join(tamanos_c)
                st.warning(
                    f"⚠️ **Nota metodológica:** Replicamos la actividad tal como se entregó (sin transformación logarítmica). "
                    f"La silhouette alta refleja que unos pocos clientes muy grandes se separan del resto, no grupos bien definidos: "
                    f"los clusters quedan de {txt_tams} clientes. En la siguiente fase se recalculará con logaritmo, como en la segmentación de artículos."
                )

            col_pareto, col_pca_cli = st.columns(2)
            with col_pareto:
                # Gráfico de concentración Pareto (% clientes vs % ventas)
                df_pareto = pd.DataFrame([
                    {"Segmento": k, "% Clientes": v.get("pct_clientes"), "% Ventas": v.get("pct_ventas")}
                    for k, v in perfil_cli.get("clusters", {}).items()
                ])
                fig_pareto = px.bar(
                    df_pareto.melt(id_vars="Segmento", value_vars=["% Clientes", "% Ventas"], var_name="Métrica", value_name="Porcentaje"),
                    x="Segmento", y="Porcentaje", color="Métrica", barmode="group",
                    color_discrete_map={"% Clientes": "#8D99AE", "% Ventas": "#0FA3B1"},
                    title="Concentración de ventas por cluster de clientes"
                )
                fig_pareto.update_layout(height=420)
                st.plotly_chart(fig_pareto, width="stretch")

                # R1: Texto dinámico generado desde el JSON
                clusters_dict = perfil_cli.get("clusters", {})
                c_frec = None
                c_ocas = None
                for nom_c, datos_c in clusters_dict.items():
                    nom_lower = nom_c.lower()
                    if "frecuente" in nom_lower or (datos_c.get("pct_ventas", 0) > 30 and datos_c.get("pct_clientes", 0) < 10):
                        c_frec = (nom_c, datos_c)
                    elif "ocasional" in nom_lower or datos_c.get("pct_clientes", 0) > 80:
                        c_ocas = (nom_c, datos_c)

                if c_frec and c_ocas:
                    nom_f, dat_f = c_frec
                    nom_o, dat_o = c_ocas
                    texto_pareto = (
                        f"<b>{dat_f['n_clientes']:,} clientes frecuentes</b> ({dat_f['pct_clientes']:.0f} % de la cartera) "
                        f"generan el <b>{dat_f['pct_ventas']:.0f} % de la facturación</b>; los {dat_o['n_clientes']:,} ocasionales "
                        f"({dat_o['pct_clientes']:.0f} %) generan el {dat_o['pct_ventas']:.0f} %."
                    )
                else:
                    partes = [f"{d['n_clientes']:,} {n.lower()} ({d['pct_clientes']:.0f} % cartera) concentran el {d['pct_ventas']:.0f} % ventas"
                              for n, d in sorted(clusters_dict.items(), key=lambda x: x[1].get("pct_ventas", 0), reverse=True)]
                    texto_pareto = "; ".join(partes) + "."
                interpretar(texto_pareto)

            with col_pca_cli:
                # Dispersión PCA de clientes sin nombres en el hover
                muestra_cli = muestra_para_grafica(df_cli)
                fig_pca_c = px.scatter(
                    muestra_cli, x="pc1", y="pc2", color="nombre_cluster",
                    opacity=0.6,
                    hover_data={"nombre_cluster": True, "total_ventas": ":$,.0f", "total_unidades": True, "frecuencia_pedidos": True, "pc1": False, "pc2": False},
                    title="Mapa PCA de clusters de clientes (sin nombres)"
                )
                fig_pca_c.update_layout(height=420, legend=dict(orientation="h", y=-0.2))
                st.plotly_chart(fig_pca_c, width="stretch")
                st.caption("Por privacidad, no se exponen nombres de clientes ni razones sociales.")

            # Heatmap Puente: Cluster de clientes x Segmento de artículos (% de unidades)
            matriz_puente = perfil_cli.get("matriz_puente_unidades_pct", {})
            if matriz_puente:
                st.markdown("#### Puente con inventario: ¿Qué productos consume cada tipo de cliente?")
                df_puente = pd.DataFrame(matriz_puente).T  # filas: clusters de clientes, columnas: segmentos SKU
                cols_orden = [c for c in ["C1", "C2", "C3", "C4", "C5", "S0", "S1"] if c in df_puente.columns] + [c for c in df_puente.columns if c not in ["C1", "C2", "C3", "C4", "C5", "S0", "S1"]]
                df_puente = df_puente.reindex(columns=cols_orden).fillna(0)

                fig_puente = px.imshow(
                    df_puente, text_auto=".1f", color_continuous_scale="Viridis", aspect="auto",
                    labels={"x": "Segmento de artículo (SKU)", "y": "Cluster de cliente", "color": "% Unidades compradas"},
                    title="Distribución de unidades compradas por segmento de artículo (%)"
                )
                fig_puente.update_layout(height=350)
                st.plotly_chart(fig_puente, width="stretch")

                # R3: Interpretación calculada dinámicamente desde matriz_puente_unidades_pct
                c1_vals = [datos.get("C1", 0) for datos in matriz_puente.values()]
                min_c1 = min(c1_vals) if c1_vals else 0
                max_c1 = max(c1_vals) if c1_vals else 0
                max_s0 = max(matriz_puente.items(), key=lambda x: x[1].get("S0", 0))
                max_c4 = max(matriz_puente.items(), key=lambda x: x[1].get("C4", 0))
                txt_puente = (
                    f"El núcleo de alta rotación (C1) representa entre el <b>{min_c1:.0f} % y el {max_c1:.0f} %</b> de las piezas que compra "
                    f"cada tipo de cliente: es transversal a toda la cartera. "
                    f"Los clientes {max_s0[0].lower()} concentran proporcionalmente más artículos sin movimiento "
                    f"(S0, {max_s0[1].get('S0', 0):.0f} %) y de baja rotación (C4, {max_c4[1].get('C4', 0):.0f} %)."
                )
                interpretar(txt_puente)

            st.caption("Recalculado con los datos tratados; las cifras pueden diferir ligeramente de la actividad entregada.")

    with tab_otros:
        st.markdown("### Otros modelos desarrollados en el Reto")
        st.caption("Lecciones aprendidas y transferibles al modelo de inventarios.")

        c_reg, c_crm, c_vol = st.columns(3)
        with c_reg:
            with st.container(border=True):
                st.markdown("#### 1. Regresión lineal de ventas")
                st.markdown("**Objetivo:** Predecir facturación a partir de características de transacción.")
                st.markdown("**Resultado clave:** $R^2 = 0.98$ al incluir `precio_unitario`.")
                st.markdown("**Lección metodológica:** El $R^2$ sale alto por construcción matemática (`total_neto` = `precio` × `unidades`). "
                            "No es una relación predictiva real del negocio. Por eso, para pronóstico de demanda, "
                            "se deben usar rezagos históricos temporales y no variables contemporáneas de la misma fila.")
                with st.expander("Ver figuras analíticas (fig1 – fig4)"):
                    for num in range(1, 5):
                        matches = list((BASE_DIR / "img_reporte").glob(f"fig{num}*.png"))
                        if matches:
                            st.image(str(matches[0]), caption=f"{matches[0].stem} · Figuras de la Evaluación (Histórico_Ventas)", width="stretch")

        with c_crm:
            with st.container(border=True):
                st.markdown("#### 2. Regresión logística: CRM")
                st.markdown("**Objetivo:** Identificar factores que detonan la pérdida de cotizaciones en HubSpot.")
                st.markdown("**Resultado clave (modelo de 3 variables en CRM):**")
                # Cifras agregadas de Regresion_Logistica_CLV.ipynb (modelo de 3 variables)
                df_crm_or = pd.DataFrame([
                    {"Variable": "cliente_nuevo", "Odds Ratio": 3.55, "Lectura": "Un cliente nuevo tiene 3.5× más probabilidad de que se pierda la cotización."},
                    {"Variable": "ruta_si", "Odds Ratio": 0.31, "Lectura": "Prospectar en ruta reduce la probabilidad de pérdida ~70 %."},
                    {"Variable": "pipeline_ventas_main", "Odds Ratio": 1.58, "Lectura": "El pipeline principal tiene 58 % más probabilidad de pérdida."},
                ])
                st.dataframe(df_crm_or, width="stretch", hide_index=True)
                st.caption("Métricas del modelo: Exactitud 0.733 · Precisión (pérdida) 0.293 · Sensibilidad (pérdida) 0.514.")
                st.markdown("**Lección metodológica:** Se debieron excluir variables que revelan el resultado a posteriori "
                            "(etapa final, motivo de cierre, fecha de cierre) para evitar **fuga de datos** (*data leakage*). "
                            "Principio idéntico aplicado en el panel de demanda.")

        with c_vol:
            with st.container(border=True):
                st.markdown("#### 3. Regresión logística: Volumen")
                st.markdown("**Objetivo:** Clasificar si un cliente compra por volumen mayorista.")
                st.markdown("**Resultado clave:** Oversampling balanceado logrando Recall 69 %, F1 0.62 y AUC 0.70.")
                st.markdown("**Lección metodológica:** Con clases fuertemente desbalanceadas, la exactitud (*Accuracy*) engaña. "
                            "El desempeño real se evalúa mediante sensibilidad (Recall) y curvas ROC. "
                            "Mismo reto que enfrenta la demanda intermitente (77.6 % meses en cero).")
                with st.expander("Ver figuras analíticas (fig5 – fig7)"):
                    for num in [5, 6, 7]:
                        matches = list((BASE_DIR / "img_reporte").glob(f"fig{num}*.png"))
                        if matches:
                            st.image(str(matches[0]), caption=f"{matches[0].stem} · Compra por volumen", width="stretch")

        st.markdown(
            "<div class='mensaje'>💡 <b>Conclusión transferible:</b> Estos ejercicios nos enseñaron a evitar la fuga de información "
            "y a evaluar modelos con métricas robustas ante desbalance severo; ambas lecciones se aplican directamente al diseño del "
            "pronóstico de demanda en la siguiente fase.</div>",
            unsafe_allow_html=True
        )


######################################################
#Vista 4 — Comportamiento de la demanda
def vista_demanda():
    encabezado("Comportamiento de la demanda", "Cómo salen los artículos del almacén mes a mes")
    aviso_vacio()

    #Serie mensual con selector de marca
    marcas_disponibles = ["Todas"] + sorted(panel_f["marca"].dropna().unique())
    marca_serie = st.selectbox("Marca para la serie mensual", marcas_disponibles, key="marca_serie")
    datos_serie = panel_f if marca_serie == "Todas" else panel_f[panel_f["marca"] == marca_serie]
    serie = datos_serie.groupby("periodo", as_index=False)["salidas"].sum()
    fig = px.line(serie, x="periodo", y="salidas", markers=True,
                  labels={"periodo": "Mes", "salidas": "Salidas de almacén (piezas)"},
                  title=f"Salidas mensuales — {marca_serie}")
    fig.update_traces(line_color="#0FA3B1", line_width=2.5)
    fig.update_layout(height=400)
    st.plotly_chart(fig, width="stretch")
    if len(serie) >= 24:
        ult12 = serie.tail(12)["salidas"].sum()
        prev12 = serie.iloc[-24:-12]["salidas"].sum()
        cambio = 100 * (ult12 - prev12) / prev12 if prev12 else np.nan
        texto_cambio = f"En los últimos 12 meses salieron {ult12:,.0f} piezas, {cambio:+.0f} % frente a los 12 meses previos."
    else:
        texto_cambio = ""
    interpretar(f"Es el total de piezas que salieron del almacén cada mes. {texto_cambio} "
                "Los picos aislados suelen ser pedidos grandes de un cliente, no una tendencia.")

    col_a, col_b = st.columns(2)
    with col_a:
        #Estacionalidad: promedio de las salidas totales de cada mes del año
        por_mes = panel_f.groupby(["anio", "mes"], as_index=False)["salidas"].sum()
        estacional = por_mes.groupby("mes", as_index=False)["salidas"].mean()
        estacional["nombre_mes"] = estacional["mes"].map(dict(enumerate(
            ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Sep", "Oct", "Nov", "Dic"], start=1)))
        fig = px.bar(estacional, x="nombre_mes", y="salidas", color_discrete_sequence=["#7B61FF"],
                     labels={"nombre_mes": "Mes", "salidas": "Salidas promedio (piezas)"},
                     title="Estacionalidad: salidas promedio por mes del año")
        fig.update_layout(height=400)
        st.plotly_chart(fig, width="stretch")
        alto = estacional.loc[estacional["salidas"].idxmax(), "nombre_mes"]
        bajo = estacional.loc[estacional["salidas"].idxmin(), "nombre_mes"]
        interpretar(f"En promedio, <b>{alto}</b> es el mes con más salidas y <b>{bajo}</b> el de menos. "
                    "Conviene tener inventario listo antes de los meses fuertes.")
    with col_b:
        #Número de SKUs por patrón de demanda
        conteo = base_f["clase_demanda"].value_counts().reindex(list(COLORES_DEMANDA)).fillna(0).reset_index()
        conteo.columns = ["clase_demanda", "skus"]
        fig = px.bar(conteo, x="clase_demanda", y="skus", color="clase_demanda", color_discrete_map=COLORES_DEMANDA,
                     text="skus", labels={"clase_demanda": "Patrón de demanda", "skus": "Número de SKUs"},
                     title="SKUs por patrón de demanda")
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, width="stretch")
        interpretar("Suave = se vende parecido cada mes. Intermitente = muchos meses sin venta. "
                    "Errática = se vende seguido pero en cantidades muy distintas. Irregular = ambas cosas a la vez.")

    #Dispersión ADI vs CV² con las líneas de Syntetos-Boylan
    disp = base_f.dropna(subset=["adi", "cv2"])
    disp = disp[disp["adi"] > 0]
    if not disp.empty:
        muestra = muestra_para_grafica(disp)
        tope_cv2 = max(disp["cv2"].quantile(0.99), UMBRAL_CV2 * 2)
        fig = px.scatter(muestra, x="adi", y="cv2", color="clase_demanda", color_discrete_map=COLORES_DEMANDA,
                         log_x=True, opacity=0.6, hover_data={"marca": True, "segmento": True},
                         labels={"adi": "ADI: cada cuántos meses se vende (escala log)", "cv2": "CV²: variación del tamaño del pedido",
                                 "clase_demanda": "Patrón"},
                         title="Mapa de patrones de demanda (Syntetos-Boylan)")
        fig.add_vline(x=UMBRAL_ADI, line_dash="dash", line_color="#333")
        fig.add_hline(y=UMBRAL_CV2, line_dash="dash", line_color="#333")
        x_max = disp["adi"].max()
        for x, y, texto in [(1.0, UMBRAL_CV2 * 0.4, "SUAVE"), (1.0, tope_cv2 * 0.92, "ERRÁTICA"),
                            (min(x_max, UMBRAL_ADI * 6), UMBRAL_CV2 * 0.4, "INTERMITENTE"),
                            (min(x_max, UMBRAL_ADI * 6), tope_cv2 * 0.92, "IRREGULAR")]:
            fig.add_annotation(x=np.log10(x), y=y, text=f"<b>{texto}</b>", showarrow=False,
                               font=dict(size=13, color="#333"), bgcolor="rgba(255,255,255,0.75)")
        fig.update_yaxes(range=[0, tope_cv2])
        fig.update_layout(height=520)
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Se grafican {len(muestra):,} de {len(disp):,} SKUs. Los SKUs de historia insuficiente no tienen ADI/CV² y no aparecen.")
        interpretar("Cada punto es un artículo. Las líneas punteadas dividen el mapa en cuatro zonas: solo los de la "
                    "esquina inferior izquierda (suave) se pueden manejar con la fórmula clásica de stock de seguridad. "
                    "Por eso una sola regla como \"pedir cuando haya 5 o menos\" no sirve para todo el catálogo: "
                    "para un artículo que se vende cada 6 meses es demasiado, y para uno que sale diario es muy poco.")


######################################################
#Vista 5 — Segmentación de artículos
def tabla_perfil():
    #Usamos el perfil del JSON; si hay filtro de marca lo recalculamos con los SKUs filtrados
    if not marcas_sel:
        filas = []
        for seg, datos in perfil.get("segmentos", {}).items():
            if segmentos_sel and seg not in segmentos_sel:
                continue
            med = datos.get("medianas", {})
            filas.append({"Segmento": seg, "Nombre": datos.get("nombre_segmento"), "SKUs": datos.get("n_skus"),
                          "Demanda 12m (med.)": med.get("demanda_12m"), "Costo unitario (med.)": med.get("costo_unitario"),
                          "ADI 24m (med.)": med.get("adi_24m"), "CV² 24m (med.)": med.get("cv2_24m"),
                          "Meses sin demanda (med.)": med.get("meses_sin_demanda"),
                          "Valor inventario": datos.get("valor_inventario"), "Cobertura (med., meses)": datos.get("cobertura_mediana")})
        return pd.DataFrame(filas), "Fuente: perfil_segmentos_seguro.json"
    tabla = base_f.groupby("segmento").agg(
        Nombre=("nombre_segmento", "first"), SKUs=("articulo_id", "size"),
        **{"Demanda 12m (med.)": ("demanda_12m", "median"), "Costo unitario (med.)": ("costo_unitario", "median"),
           "ADI 24m (med.)": ("adi_24m", "median"), "CV² 24m (med.)": ("cv2_24m", "median"),
           "Meses sin demanda (med.)": ("meses_sin_demanda", "median"),
           "Valor inventario": ("valor_inventario_corte_ultimo", "sum"), "Cobertura (med., meses)": ("cobertura_meses", "median")})
    tabla = tabla.loc[orden_segmentos(tabla.index)].reset_index().rename(columns={"segmento": "Segmento"})
    return tabla, "Recalculado con el filtro de marca activo"


def vista_segmentacion():
    encabezado("Segmentación de artículos", "No todos los artículos merecen la misma política de inventario")
    aviso_vacio()
    k = perfil.get("k_elegido")
    sil = next((m.get("silhouette") for m in perfil.get("metricas_por_k", []) if m.get("k") == k), None)
    ari = perfil.get("ari_kmeans_vs_ward")
    st.caption(f"Modelo: K-Means con k = {k} sobre {perfil.get('tamanos', {}).get('universo_clustering', 0):,} SKUs activos"
               + (f" · silhouette = {sil:.2f}" if sil is not None else "")
               + (f" · coincidencia con jerárquico Ward (ARI) = {ari:.2f}" if ari is not None else ""))

    tabla, fuente = tabla_perfil()
    st.dataframe(tabla, width="stretch", hide_index=True, column_config={
        "Valor inventario": st.column_config.NumberColumn(format="$%,.0f"),
        "Costo unitario (med.)": st.column_config.NumberColumn(format="$%,.0f"),
        "Demanda 12m (med.)": st.column_config.NumberColumn(format="%,.0f"),
        "ADI 24m (med.)": st.column_config.NumberColumn(format="%.1f"),
        "CV² 24m (med.)": st.column_config.NumberColumn(format="%.2f"),
        "Meses sin demanda (med.)": st.column_config.NumberColumn(format="%.0f"),
        "Cobertura (med., meses)": st.column_config.NumberColumn(format="%.1f"),
    })
    st.caption(fuente)
    interpretar("Cada fila resume un grupo de artículos con valores típicos (mediana): cuánto se vende al año, cuánto cuesta, "
                "cada cuántos meses se pide y para cuántos meses alcanza lo que hay en almacén.")

    col_a, col_b = st.columns(2)
    with col_a:
        pca = base_f.dropna(subset=["pc1", "pc2"])
        if not pca.empty:
            muestra = muestra_para_grafica(pca).copy()
            muestra["Segmento"] = muestra["segmento"].map(etiqueta_segmento)
            mapa_colores = {etiqueta_segmento(s): c for s, c in COLORES_SEGMENTO.items()}
            fig = px.scatter(muestra.sort_values("segmento"), x="pc1", y="pc2", color="Segmento",
                             color_discrete_map=mapa_colores, opacity=0.65, hover_data={"marca": True, "clase_abc": True},
                             labels={"pc1": "Componente 1", "pc2": "Componente 2"}, title="Mapa de clusters (PCA)")
            fig.update_layout(height=470, legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, width="stretch")
            interpretar("Cada punto es un artículo activo, acomodado según sus 5 características resumidas en dos ejes. "
                        "Los puntos del mismo color se comportan parecido; grupos bien separados = segmentación clara.")
        else:
            st.info("Los segmentos filtrados (S0 y S1) no participan en el clustering, así que no tienen mapa PCA.")
    with col_b:
        #Matriz ABC × XYZ con selector entre número de SKUs y valor de inventario
        medida = st.radio("Medida de la matriz ABC × XYZ", ["Número de SKUs", "Valor de inventario"],
                          horizontal=True, key="medida_abc_xyz")
        abc_xyz = base_f[base_f["clase_abc"].isin(["A", "B", "C"]) & base_f["clase_xyz"].isin(["X", "Y", "Z"])]
        if medida == "Número de SKUs":
            matriz = pd.crosstab(abc_xyz["clase_abc"], abc_xyz["clase_xyz"])
        else:
            matriz = abc_xyz.pivot_table(index="clase_abc", columns="clase_xyz",
                                         values="valor_inventario_corte_ultimo", aggfunc="sum", fill_value=0)
        matriz = matriz.reindex(index=["A", "B", "C"], columns=["X", "Y", "Z"]).fillna(0)
        fig = px.imshow(matriz, text_auto=",.0f", color_continuous_scale="Teal", aspect="auto",
                        labels={"x": "XYZ (estabilidad)", "y": "ABC (importancia)", "color": medida},
                        title=f"Matriz ABC × XYZ — {medida.lower()}")
        fig.update_layout(height=430)
        st.plotly_chart(fig, width="stretch")
        interpretar("Arriba a la izquierda (AX) están los artículos importantes y predecibles: los más fáciles de automatizar. "
                    "Abajo a la derecha (CZ) los de poco valor y muy variables: conviene stock mínimo o compra bajo pedido.")

    #Heatmap segmento × clase ABC
    cruce = pd.crosstab(base_f["segmento"], base_f["clase_abc"])
    cruce = cruce.loc[orden_segmentos(cruce.index)]
    cruce.index = [etiqueta_segmento(s) for s in cruce.index]
    fig = px.imshow(cruce, text_auto=True, color_continuous_scale="Blues", aspect="auto",
                    labels={"x": "Clase ABC", "y": "Segmento", "color": "SKUs"}, title="Segmento × clase ABC (número de SKUs)")
    fig.update_layout(height=120 + 45 * len(cruce))
    st.plotly_chart(fig, width="stretch")
    interpretar("La clasificación ABC solo mira el valor vendido. Si una misma clase (por ejemplo C) aparece repartida en "
                "varios segmentos, significa que dentro de ella hay artículos que necesitan políticas distintas: "
                "por eso la segmentación funciona como un ABC con varios criterios a la vez.")

    #Tarjetas con la política sugerida por segmento
    st.markdown("### Política sugerida por segmento")
    segmentos_vista = orden_segmentos(base_f["segmento"].dropna())
    columnas = st.columns(3)
    for i, seg in enumerate(segmentos_vista):
        datos = base_f[base_f["segmento"] == seg]
        with columnas[i % 3]:
            with st.container(border=True):
                st.markdown(f"<div class='franja' style='background:{COLORES_SEGMENTO.get(seg, '#999')}'></div>"
                            f"<div class='tarjeta-titulo'>{etiqueta_segmento(seg)}</div>", unsafe_allow_html=True)
                st.caption(f"{len(datos):,} SKUs · {pesos(datos['valor_inventario_corte_ultimo'].sum())} en inventario")
                st.markdown(politica_de(NOMBRES_SEGMENTO.get(seg, "")))


######################################################
#Vista 6 — Ficha por SKU
def vista_ficha():
    encabezado("Ficha por SKU", "Toda la información de un artículo en una sola pantalla")
    opciones = base.sort_values("demanda_12m", ascending=False)
    etiquetas = dict(zip(opciones["articulo_id"], opciones["articulo_id"].astype(str) + " · " + opciones["nombre_articulo"].fillna("")))
    sku = st.selectbox("Busca por número de artículo o por nombre", list(etiquetas), format_func=etiquetas.get,
                       key="buscador_sku")
    if sku is None:
        st.stop()
    fila = base[base["articulo_id"] == sku].iloc[0]
    serie = panel[panel["articulo_id"] == sku].sort_values("periodo")

    #Gráfica de salidas mensuales con picos resaltados
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=serie["periodo"], y=serie["salidas"], mode="lines+markers", name="Salidas de almacén",
                             line=dict(color=COLORES_SEGMENTO.get(fila["segmento"], "#0FA3B1"), width=2.5)))
    fig.add_trace(go.Scatter(x=serie["periodo"], y=serie["unidades_vendidas"], mode="lines", name="Unidades vendidas",
                             line=dict(color="#888", dash="dot")))
    picos = serie[serie["es_pico"]]
    if not picos.empty:
        fig.add_trace(go.Scatter(x=picos["periodo"], y=picos["salidas"], mode="markers", name="Mes pico",
                                 marker=dict(color="#F25F5C", size=13, symbol="star")))
    fig.update_layout(title=f"Salidas mensuales — {etiquetas[sku]}", height=420, xaxis_title="Mes", yaxis_title="Piezas",
                      legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig, width="stretch")
    interpretar("La línea de color son las piezas que salieron del almacén cada mes (la demanda que usaremos para pronosticar); "
                "la línea gris punteada son las unidades facturadas, solo como referencia. Las estrellas rojas son meses "
                "con salidas muy por encima de lo normal para este artículo.")

    #Indicadores del artículo
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Segmento", fila["segmento"], fila["nombre_segmento"], delta_color="off")
    c2.metric("ABC / XYZ", f"{fila['clase_abc']} / {fila['clase_xyz']}")
    c3.metric("Patrón de demanda", str(fila["clase_demanda"]).capitalize())
    c4.metric("Existencias", numero(fila["existencias"]),
              "Sin registro en el corte" if fila["sin_registro_existencia"] else None, delta_color="off")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("Cobertura", f"{numero(fila['cobertura_meses'], 1)} meses")
    c6.metric("Costo unitario", pesos(fila["costo_unitario"]), "Imputado" if fila["costo_imputado"] else None, delta_color="off")
    c7.metric("Última compra", fila["fecha_ultima_compra"].strftime("%d/%m/%Y") if pd.notna(fila["fecha_ultima_compra"]) else "—")
    c8.metric("Días promedio entre compras", numero(fila["dias_promedio_entre_compras"]))

    #Interpretación automática en una o dos frases
    patrones = {
        "SUAVE": "se vende de forma pareja casi todos los meses",
        "ERRATICA": "se vende seguido pero en cantidades muy distintas",
        "INTERMITENTE": "tiene muchos meses sin ventas",
        "IRREGULAR": "tiene meses sin ventas y pedidos de tamaño muy distinto",
        "INSUFICIENTE": "todavía no tiene historia suficiente para describir su patrón",
        "SIN MOVIMIENTO DESDE 2021": "no tiene salidas registradas desde 2021",
    }
    frases = []
    if pd.notna(fila["cobertura_meses"]):
        frases.append(f"Este artículo tiene cobertura para {fila['cobertura_meses']:.1f} meses al ritmo del último año")
    elif fila["demanda_12m"] == 0:
        if str(fila.get("clase_demanda", "")).strip() == "SIN MOVIMIENTO DESDE 2021":
            frases.append("Este artículo no tiene salidas registradas desde 2021")
        else:
            frases.append("Este artículo no tuvo salidas en los últimos 12 meses")
    else:
        frases.append("No pudimos calcular su cobertura")
    frases[0] += f"; su demanda {patrones.get(fila['clase_demanda'], 'no está clasificada')}."
    if fila["bandera_revision"] != "OK":
        frases.append(f"Señal de revisión: {fila['motivo']}")
    frases.append(f"Política sugerida para su segmento: {politica_de(fila['nombre_segmento'])}")
    st.markdown(f"<div class='mensaje'>🧭 {' '.join(frases)}</div>", unsafe_allow_html=True)


######################################################
#Vista 7 — Hoja de ruta
def vista_hoja_ruta():
    encabezado("Hoja de ruta", "Hacia dónde va el proyecto y qué necesitamos de CLV")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        with st.container(border=True):
            st.markdown("### 1 · Pronóstico de demanda")
            st.markdown("- Un modelo distinto para cada segmento.\n"
                        "- Líneas base (promedio móvil, ingenuo estacional).\n"
                        "- Croston, SBA y TSB para la demanda intermitente.\n"
                        "- ETS / SARIMA para la demanda regular y por marca.\n"
                        "- Random Forest y XGBoost como modelos globales.\n"
                        "- Validación con *backtesting*: simulamos el pasado para medir el error real.")
    with col_b:
        with st.container(border=True):
            st.markdown("### 2 · Stock de seguridad")
            st.markdown("- Stock de seguridad y punto de reorden por SKU.\n"
                        "- Simulación Monte Carlo con el error del pronóstico y el tiempo de entrega.\n"
                        "- Nivel de servicio distinto por segmento.\n"
                        "- Escenarios de lead time (15 / 30 / 60 días) mientras llega el dato real.")
    with col_c:
        with st.container(border=True):
            st.markdown("### 3 · Entregables para Odoo")
            st.markdown("- Reglas de reabastecimiento (mínimo, máximo, múltiplo) listas para importar.\n"
                        "- Propuesta de pedido.\n"
                        "- Excel explicado por SKU.\n"
                        "- Simulador de nivel de servicio contra inversión.\n"
                        "- Botón de pronóstico a 3–6 meses con intervalos.")
    interpretar("Hoy entregamos el diagnóstico y la segmentación. La segmentación es la base: decide qué modelo de "
                "pronóstico y qué nivel de protección recibe cada artículo en las siguientes fases.")

    st.markdown("### Datos que necesitamos de CLV")
    st.dataframe(pd.DataFrame(DATOS_SOLICITAR, columns=["Dato", "Para qué lo necesitamos"]), width="stretch", hide_index=True)
    st.info("Si el lead time no llega a tiempo, el modelo corre con escenarios por marca y deja el parámetro listo "
            "para cuando se tenga. Se presentará como supuesto, no como resultado.")


######################################################
#Mostramos la vista elegida (agregar una vista nueva = agregar su función a este diccionario y a VISTAS)
FUNCIONES_VISTA = dict(zip(VISTAS, [
    vista_resumen,
    vista_calidad,
    vista_exploracion,
    vista_demanda,
    vista_segmentacion,
    vista_ficha,
    vista_hoja_ruta,
]))
FUNCIONES_VISTA[vista]()
