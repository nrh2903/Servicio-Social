import io
import zipfile

import geoglows
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
from scipy.stats import gumbel_r

st.set_page_config(page_title="Predicción de desbordamientos", page_icon="🌊", layout="wide")

COLORES_RP = {2: 'yellow', 5: 'orange', 10: 'red', 25: 'purple', 50: 'brown', 100: 'magenta'}
NOMBRES_COLUMNAS = {
    'flow_min': 'Mínimo', 'flow_25p': 'Percentil 25', 'flow_avg': 'Promedio',
    'flow_med': 'Mediana', 'flow_75p': 'Percentil 75',
    'flow_uncertainty_upper': 'Incertidumbre superior', 'flow_max': 'Máximo'
}


# ============================================================
# DESCARGA DE DATOS (con caché para que la página sea rápida)
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def cargar_pronostico(reach_id, fecha_str):
    """Pronóstico de la fecha dada; si no está publicado, retrocede hasta 3 días."""
    fecha = pd.Timestamp(fecha_str)
    for dias_atras in range(4):
        fecha_try = fecha - pd.Timedelta(days=dias_atras)
        try:
            df = geoglows.data.forecast_stats(reach_id, date=fecha_try.strftime('%Y%m%d'))
            df.index = pd.to_datetime(df.index).tz_localize(None)
            return df, fecha_try
        except Exception:
            continue
    raise RuntimeError("No se encontró ningún pronóstico en los últimos 4 días.")


@st.cache_data(ttl=86400, show_spinner=False)
def cargar_historico(reach_id):
    df = geoglows.data.retrospective(reach_id, resolution='daily')
    df.index = pd.to_datetime(df.index).tz_localize(None)
    return df


def extraer_rp(df):
    """Devuelve {años_de_retorno: caudal} sin importar el formato del DataFrame."""
    rps = {}
    if len(df.columns) == 1:
        try:
            for idx, val in df.iloc[:, 0].items():
                rps[int(float(idx))] = float(val)
            if rps:
                return rps
        except (ValueError, TypeError):
            rps = {}
    for col in df.columns:
        digitos = ''.join(ch for ch in str(col) if ch.isdigit())
        if digitos:
            rps[int(digitos)] = float(df[col].iloc[0])
    return rps


@st.cache_data(ttl=86400, show_spinner=False)
def cargar_rp_oficial(reach_id):
    """Periodos de retorno oficiales de GEOGLOWS; None si ninguna llamada funciona."""
    for kwargs in ({'method': 'gumbel'}, {'distribution': 'gumbel'}):
        try:
            rp = extraer_rp(geoglows.data.return_periods(reach_id, **kwargs))
            if rp:
                return rp
        except Exception:
            continue
    return None


def rp_gumbel(historico):
    """Respaldo: Gumbel ajustado a los máximos anuales del histórico."""
    serie = historico.iloc[:, 0]
    maximos = serie.groupby(serie.index.year).max()
    loc, scale = gumbel_r.fit(maximos.values)
    return {t: float(gumbel_r.ppf(1 - 1 / t, loc=loc, scale=scale)) for t in [2, 5, 10, 25, 50, 100]}


# ============================================================
# EVALUACIÓN DE RIESGO REAL
# ============================================================
def evaluar_riesgo(pron, rp_dict, rp_min, dias_min):
    """Riesgo real = caudal PROMEDIO del ensamble sobre el periodo rp_min
    durante al menos dias_min días consecutivos."""
    col = next((c for c in ('flow_avg', 'flow_med') if c in pron.columns), pron.columns[0])
    diario = pron[col].resample('D').mean()
    pico, fecha_pico = diario.max(), diario.idxmax()

    nivel = 0
    for rp, q in sorted(rp_dict.items()):
        if pico >= q:
            nivel = rp

    q_min = rp_dict.get(rp_min)
    consecutivos, max_consec = 0, 0
    if q_min is not None:
        for v in diario:
            consecutivos = consecutivos + 1 if v >= q_min else 0
            max_consec = max(max_consec, consecutivos)

    hay_riesgo = (q_min is not None) and (nivel >= rp_min) and (max_consec >= dias_min)
    return hay_riesgo, pico, fecha_pico, nivel, max_consec


# ============================================================
# GRÁFICAS
# ============================================================
def fig_prediccion(pron, titulo):
    fig, ax = plt.subplots(figsize=(10, 5))
    pron.plot(ax=ax)
    ax.set_title(titulo)
    ax.set_ylabel("Caudal (m³/s)")
    ax.set_xlabel("Fecha")
    ax.grid(True)
    fig.tight_layout()
    return fig


def fig_historico(hist_rec, titulo):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(hist_rec.index, hist_rec.iloc[:, 0], color='purple', label='Caudal histórico (último año)')
    ax.set_title(titulo)
    ax.set_ylabel("Caudal (m³/s)")
    ax.set_xlabel("Fecha")
    ax.legend()
    ax.grid(True)
    fig.tight_layout()
    return fig


def fig_combinada(pron, hist_rec, rp_dict, inicio, fin, hay_riesgo, pico, fecha_pico, nivel, titulo):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(hist_rec.index, hist_rec.iloc[:, 0], color='black', label='Caudal histórico (ERA5)', linewidth=2)
    pron.plot(ax=ax)

    for num_rp, q in sorted(rp_dict.items()):
        ax.axhline(y=q, color=COLORES_RP.get(num_rp, 'gray'), linestyle='--', linewidth=1.5,
                   label=f'Periodo de retorno {num_rp} años')

    if hay_riesgo:
        ax.axvspan(inicio, fin, color='red', alpha=0.08)
        ax.annotate(
            f"⚠ ALERTA: pico {pico:.0f} m³/s ({fecha_pico.date()})\nsupera el periodo de retorno de {nivel} años",
            xy=(fecha_pico, pico), xytext=(0.02, 0.95), textcoords='axes fraction',
            va='top', fontsize=10, color='darkred', fontweight='bold',
            bbox=dict(boxstyle='round', fc='white', ec='red'),
            arrowprops=dict(arrowstyle='->', color='red'))

    ax.set_title(titulo + ("  [ALERTA]" if hay_riesgo else ""))
    ax.set_ylabel("Caudal (m³/s)")
    ax.set_xlabel("Fecha")
    ax.grid(True)
    ax.set_xlim(left=inicio - pd.DateOffset(months=1), right=fin)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    return fig


def fig_evento(historico, rp_dict, f_ini, f_fin, f_evento, titulo):
    ev = historico.loc[f_ini:f_fin].iloc[:, 0]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ev.index, ev.values, color='black', linewidth=2, label='Caudal simulado (ERA5)')
    for num_rp, q in sorted(rp_dict.items()):
        ax.axhline(y=q, color=COLORES_RP.get(num_rp, 'gray'), linestyle='--', linewidth=1.5,
                   label=f'Periodo de retorno {num_rp} años')
    ax.axvline(f_evento, color='red', linestyle=':', linewidth=2, label=f'Evento {f_evento.date()}')
    ax.set_title(titulo)
    ax.set_ylabel("Caudal (m³/s)")
    ax.set_xlabel("Fecha")
    ax.grid(True)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    fig.tight_layout()
    return fig, ev


def construir_zip(figs, tablas, texto_alerta):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as z:
        for nombre, fig in figs.items():
            b = io.BytesIO()
            fig.savefig(b, format='png', dpi=150, bbox_inches='tight')
            z.writestr(f'{nombre}.png', b.getvalue())
        for nombre, df in tablas.items():
            z.writestr(f'{nombre}.csv', df.to_csv())
        if texto_alerta:
            z.writestr('ALERTA.txt', texto_alerta)
    return buf.getvalue()


# ============================================================
# INTERFAZ
# ============================================================
st.title("🌊 Predicción de desbordamientos - Poza Rica, Veracruz")
st.caption("Pronóstico de caudal a 15 días con GEOGLOWS y alertas por periodos de retorno.")

with st.sidebar:
    st.header("Parámetros")
    reach_id = int(st.number_input("ID del tramo", value=770229421, step=1, format="%d"))
    fecha = st.date_input("Fecha del pronóstico", value=pd.Timestamp.now(tz='UTC').date())
    st.subheader("Criterio de alerta")
    rp_min = st.selectbox("Alertar desde el periodo de retorno de (años)", [2, 5, 10], index=0)
    dias_min = st.slider("Días consecutivos mínimos", 1, 5, 2)
    if st.button("Generar pronóstico", type="primary"):
        st.session_state["generado"] = True

if not st.session_state.get("generado"):
    st.info("Elige los parámetros en el panel izquierdo y pulsa **Generar pronóstico**.")
    st.stop()

# --- Descarga ---
try:
    with st.spinner("Descargando datos de GEOGLOWS (la primera vez tarda un poco)..."):
        pronostico_raw, fecha_usada = cargar_pronostico(reach_id, pd.Timestamp(fecha).strftime('%Y-%m-%d'))
        historico = cargar_historico(reach_id)
        rp_dict = cargar_rp_oficial(reach_id)
except Exception as e:
    st.error(f"No se pudieron descargar los datos: {e}")
    st.stop()

rp_origen = "GEOGLOWS (oficial)"
if rp_dict is None:
    rp_dict = rp_gumbel(historico)
    rp_origen = "Gumbel calculado desde el histórico ERA5 (NO oficial)"

if fecha_usada != pd.Timestamp(fecha):
    st.warning(f"Se usó el pronóstico del {fecha_usada.date()}: el de {fecha} aún no está disponible.")
if "NO oficial" in rp_origen:
    st.warning("Los periodos de retorno se calcularon desde el histórico (no son los oficiales de GEOGLOWS).")

# --- Evaluación ---
hay_riesgo, pico, fecha_pico, nivel, dias_sobre = evaluar_riesgo(pronostico_raw, rp_dict, rp_min, dias_min)
pronostico = pronostico_raw.rename(columns=NOMBRES_COLUMNAS)

inicio = pronostico.index.min()
fin = pronostico.index.max()
hist_rec = historico.loc[inicio - pd.DateOffset(years=1):inicio]

texto_alerta = None
if hay_riesgo:
    texto_alerta = (f"ALERTA - Tramo {reach_id}: caudal promedio pronosticado de {pico:.0f} m3/s "
                    f"el {fecha_pico.date()}, supera el periodo de retorno de {nivel} años "
                    f"durante {dias_sobre} días consecutivos.\nPeriodos de retorno: {rp_origen}\n")

# --- Gráficas ---
titulo_base = f"Tramo {reach_id} - Poza Rica"
f1 = fig_prediccion(pronostico, f"1. Hidrograma de predicción (15 días, desde {inicio.date()}) - {titulo_base}")
f2 = fig_historico(hist_rec, f"2. Hidrograma histórico - {titulo_base}")
f3 = fig_combinada(pronostico, hist_rec, rp_dict, inicio, fin, hay_riesgo, pico, fecha_pico, nivel,
                   f"3. Hidrograma combinado y alertas - {titulo_base}")
figs = {'grafica1_prediccion': f1, 'grafica2_historico': f2, 'grafica3_combinada': f3}
tablas = {
    'pronostico': pronostico,
    'historico_reciente': hist_rec,
    'periodos_retorno': pd.Series(rp_dict, name='caudal_m3s').rename_axis('periodo_retorno_anios').to_frame(),
}

# --- Pestañas ---
tab_res, tab1, tab2, tab3, tab4, tab_met = st.tabs(
    ["Resumen", "Predicción", "Histórico", "Combinada", "Caso de estudio", "Metodología"])

with tab_res:
    if hay_riesgo:
        st.error(f"⚠ ALERTA: pico de {pico:.0f} m³/s el {fecha_pico.date()}, supera el periodo de retorno "
                 f"de {nivel} años durante {dias_sobre} días consecutivos.")
    else:
        st.success("✔ Sin riesgo real en los próximos 15 días con el criterio seleccionado.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pico promedio pronosticado", f"{pico:.0f} m³/s")
    c2.metric("Fecha del pico", str(fecha_pico.date()))
    c3.metric("Periodo de retorno superado", f"{nivel} años" if nivel else "Ninguno")
    c4.metric("Pronóstico emitido", str(fecha_usada.date()))

    st.subheader("Periodos de retorno del tramo")
    st.caption(f"Origen: {rp_origen}")
    st.dataframe(pd.Series(rp_dict, name="Caudal (m³/s)").rename_axis("Periodo de retorno (años)").round(1).to_frame())

with tab1:
    st.pyplot(f1)
with tab2:
    st.pyplot(f2)
with tab3:
    st.pyplot(f3)

with tab4:
    st.write("Comparación del caudal simulado con los periodos de retorno durante un evento pasado.")
    c1, c2, c3 = st.columns(3)
    f_ini = pd.Timestamp(c1.date_input("Inicio de la ventana", value=pd.Timestamp('2025-09-25').date()))
    f_fin = pd.Timestamp(c2.date_input("Fin de la ventana", value=pd.Timestamp('2025-10-25').date()))
    f_evt = pd.Timestamp(c3.date_input("Fecha del evento", value=pd.Timestamp('2025-10-10').date()))
    f4, ev = fig_evento(historico, rp_dict, f_ini, f_fin, f_evt,
                        f"4. Caso de estudio ({f_evt.date()}) - {titulo_base}")
    st.pyplot(f4)
    figs['grafica4_evento'] = f4
    if len(ev):
        superados = [rp for rp, q in sorted(rp_dict.items()) if ev.max() >= q]
        st.info(f"Pico en la ventana: {ev.max():.0f} m³/s el {ev.idxmax().date()} | "
                f"Periodo de retorno superado: {superados[-1] if superados else 'ninguno'} años")
    else:
        st.warning("El histórico no tiene datos en esa ventana de fechas.")

with tab_met:
    st.markdown("""
**Datos:** pronóstico, histórico y periodos de retorno del modelo hidrológico GEOGLOWS (caudales simulados, no medidos en estación).

**Pronóstico:** estadísticas de un ensamble de 51 simulaciones a 15 días. La incertidumbre aumenta con el tiempo.

**Criterio de alerta:** hay alerta cuando el caudal *promedio* del ensamble supera el periodo de retorno elegido durante
al menos el número de días consecutivos indicado. Se usa el promedio y no el máximo para evitar falsas alarmas.

**Limitaciones:** el caudal es simulado con datos de clima (ERA5); los periodos de retorno son estadísticos y dependen del modelo.
Si no se pueden obtener los oficiales, se calculan con Gumbel sobre el histórico y la app lo indica.
""")

# --- Descarga de resultados ---
st.sidebar.divider()
st.sidebar.download_button(
    "⬇ Descargar resultados (.zip)",
    data=construir_zip(figs, tablas, texto_alerta),
    file_name=f"resultados_{reach_id}_{inicio.strftime('%Y%m%d')}.zip",
    mime="application/zip",
)

plt.close('all')
