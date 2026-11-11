import io
import zipfile

import geoglows
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import streamlit as st
from scipy.stats import gumbel_r

st.set_page_config(page_title="Predicción de desbordamientos", page_icon="🌊", layout="wide")

COLORES_RP = {2: 'yellow', 5: 'orange', 10: 'red', 25: 'purple', 50: 'brown', 100: 'magenta'}
COLOR_ESTADO = {'🔴 Alerta': 'red', '🟠 Atención': 'orange',
                '🟡 Vigilancia': 'gold', '🟢 Sin riesgo': 'green'}
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
    for kwargs in ({'method': 'gumbel'},
