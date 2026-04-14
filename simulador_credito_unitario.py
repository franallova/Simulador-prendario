import io
import json
import math
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

# Meses en español para etiquetas del cashflow
MESES_ES = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
]
import streamlit as st

# CSV de cartera en disco (mismo directorio que este script). El export «desde abril» reemplaza al de marzo.
VENCIMIENTOS_CSV_DISCO_DEFAULT = "Vencimientos desde Abril2026.csv"


def calcular_irr(flujos: List[float], precision: float = 1e-6, max_iter: int = 10_000) -> float:
    """
    Calcula una TIR aproximada por método de bisección.
    Devuelve tasa MENSUAL (por ejemplo 0.03 = 3% mensual).
    """

    def npv(tasa: float) -> float:
        total = 0.0
        for t, cf in enumerate(flujos):
            total += cf / ((1 + tasa) ** t)
        return total

    low, high = -0.99, 5.0  # -99% a 500% mensual
    npv_low, npv_high = npv(low), npv(high)
    if npv_low * npv_high > 0:
        return 0.0

    for _ in range(max_iter):
        mid = (low + high) / 2
        npv_mid = npv(mid)
        if abs(npv_mid) < precision:
            return mid
        if npv_low * npv_mid < 0:
            high = mid
            npv_high = npv_mid
        else:
            low = mid
            npv_low = npv_mid
    return (low + high) / 2


def calcular_van(flujos: List[float], tasa_mensual: float) -> float:
    """VAN (NPV) de la serie de flujos descontados a tasa_mensual por período."""
    return sum(cf / ((1 + tasa_mensual) ** t) for t, cf in enumerate(flujos))


def formato_pesos(valor: float) -> str:
    try:
        return f"$ {valor:,.0f}".replace(",", ".")
    except (TypeError, ValueError):
        return ""


def _parse_num(s: str) -> float:
    """Convierte string numérico (ej. '161440,27' o '418960,04') a float."""
    if pd.isna(s) or s == "" or s is None:
        return 0.0
    s = str(s).strip().replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return 0.0


_VENCIMIENTOS_COLS = [
    "Fecha_Vto",
    "Capital",
    "Interes_Ordinario",
    "IVA_Interes_Ordinario",
    "Seguro",
    "Gastos",
    "Interes_Gastos",
    "IVA_Interes_Gastos",
]


def _procesar_dataframe_vencimientos(df: pd.DataFrame, fecha_desde: Optional[date] = None) -> Optional[pd.DataFrame]:
    """Agrupa vencimientos por mes calendario de Fecha_Vto. Espera columnas estándar ya leídas."""
    if df is None or df.empty:
        return None
    for c in _VENCIMIENTOS_COLS:
        if c not in df.columns:
            return None
    # Parsear fechas (ej. "01/03/2026 12:00:00 a.m.")
    df = df.copy()
    df["Fecha_Vto"] = pd.to_datetime(df["Fecha_Vto"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Fecha_Vto"])
    if fecha_desde is not None:
        try:
            desde = pd.to_datetime(date(fecha_desde.year, fecha_desde.month, 1))
            df = df[df["Fecha_Vto"] >= desde]
        except Exception:
            pass
    if df.empty:
        return None
    for col in [
        "Capital",
        "Interes_Ordinario",
        "IVA_Interes_Ordinario",
        "Seguro",
        "Gastos",
        "Interes_Gastos",
        "IVA_Interes_Gastos",
    ]:
        df[col] = df[col].astype(str).apply(_parse_num)
    df["año"] = df["Fecha_Vto"].dt.year
    df["mes"] = df["Fecha_Vto"].dt.month
    ag = df.groupby(["año", "mes"]).agg(
        Capital=("Capital", "sum"),
        Interes_Ordinario=("Interes_Ordinario", "sum"),
        IVA_Interes_Ordinario=("IVA_Interes_Ordinario", "sum"),
        Seguro=("Seguro", "sum"),
        Gastos=("Gastos", "sum"),
        Interes_Gastos=("Interes_Gastos", "sum"),
        IVA_Interes_Gastos=("IVA_Interes_Gastos", "sum"),
    ).reset_index()
    ag["Cobranza_Total"] = (
        ag["Capital"] + ag["Interes_Ordinario"] + ag["IVA_Interes_Ordinario"]
        + ag["Seguro"] + ag["Gastos"] + ag["Interes_Gastos"] + ag["IVA_Interes_Gastos"]
    )
    ag["IVA_Total"] = ag["IVA_Interes_Ordinario"] + ag["IVA_Interes_Gastos"]
    ag["Seguro_Total"] = ag["Seguro"].copy()
    return ag


def _leer_csv_vencimientos_fuente(fuente: Union[str, bytes]) -> Optional[pd.DataFrame]:
    """Lee CSV de vencimientos desde ruta de archivo o contenido binario (export del core)."""
    try:
        if isinstance(fuente, bytes):
            return pd.read_csv(
                io.BytesIO(fuente),
                sep=";",
                encoding="latin-1",
                usecols=_VENCIMIENTOS_COLS,
                dtype=str,
                on_bad_lines="skip",
            )
        return pd.read_csv(
            fuente,
            sep=";",
            encoding="latin-1",
            usecols=_VENCIMIENTOS_COLS,
            dtype=str,
            on_bad_lines="skip",
        )
    except Exception:
        return None


def leer_vencimientos_real(ruta_csv: str, fecha_desde: Optional[date] = None) -> Optional[pd.DataFrame]:
    """
    Lee el archivo de vencimientos desde disco.
    Agrupa por mes de vencimiento y suma Capital, Interés, IVA, Seguro, Gastos (sin punitorios ni compensatorios).
    """
    if not ruta_csv or not os.path.isfile(ruta_csv):
        return None
    df = _leer_csv_vencimientos_fuente(ruta_csv)
    return _procesar_dataframe_vencimientos(df, fecha_desde=fecha_desde)


def leer_vencimientos_desde_bytes(contenido: bytes, fecha_desde: Optional[date] = None) -> Optional[pd.DataFrame]:
    """Igual que leer_vencimientos_real pero desde bytes (archivo subido en Streamlit)."""
    df = _leer_csv_vencimientos_fuente(contenido)
    return _procesar_dataframe_vencimientos(df, fecha_desde=fecha_desde)


def calcular_cuota_total(
    capital_financiado: float,
    tasa_mensual: float,
    cuotas: int,
    gasto_cuota: float,
    iva_interes_pct: float,
    iva_gastos_pct: float,
) -> float:
    """
    Calcula la CUOTA TOTAL constante (sistema francés "completo"):
    - Cuota total = capital + interés + gastos + IVA_int + IVA_gastos
    - Interés = saldo * tasa_mensual
    - Gastos = gasto_cuota (fijo por cuota)
    - IVA_int = interés * iva_interes_pct
    - IVA_gastos = gasto_cuota * iva_gastos_pct

    Busca la cuota_total tal que el saldo final quede lo más cercano posible a 0.
    """

    if cuotas <= 0:
        return 0.0

    # Si la tasa es 0, cuota total aproximada: capital/ cuotas + gastos + IVA_gastos
    if tasa_mensual == 0:
        base = capital_financiado / cuotas
        iva_gastos = gasto_cuota * iva_gastos_pct
        return base + gasto_cuota + iva_gastos

    def saldo_final(cuota_total: float) -> float:
        saldo = capital_financiado
        for _ in range(cuotas):
            interes = saldo * tasa_mensual
            iva_int = interes * iva_interes_pct
            iva_gastos = gasto_cuota * iva_gastos_pct
            capital_pago = cuota_total - (interes + iva_int + gasto_cuota + iva_gastos)
            saldo -= capital_pago
        return saldo

    # Buscamos cuota_total por bisección entre límites razonables
    low, high = 0.0, capital_financiado * 2.0  # cota alta generosa
    s_low, s_high = saldo_final(low), saldo_final(high)

    # Ajustamos cota alta si hace falta para cambiar de signo
    intentos = 0
    while s_low * s_high > 0 and intentos < 10:
        high *= 2
        s_high = saldo_final(high)
        intentos += 1

    if s_low * s_high > 0:
        # Si no encontramos cambio de signo, usamos fórmula francesa aproximada
        cuota_aprox = (
            capital_financiado
            * (tasa_mensual * (1 + tasa_mensual) ** cuotas)
            / ((1 + tasa_mensual) ** cuotas - 1)
        )
        iva_gastos = gasto_cuota * iva_gastos_pct
        return cuota_aprox + gasto_cuota + iva_gastos

    for _ in range(10_000):
        mid = (low + high) / 2
        s_mid = saldo_final(mid)
        if abs(s_mid) < 1e-3:
            return mid
        if s_low * s_mid < 0:
            high = mid
            s_high = s_mid
        else:
            low = mid
            s_low = s_mid

    return (low + high) / 2


# --- Persistencia de inputs (archivo JSON junto al script) ---
_INPUT_DATE_KEYS = frozenset({"fecha_otorgamiento", "fecha_inicio_cf"})


def _defaults_inputs_simulador() -> Dict[str, Any]:
    """Valores por defecto cuando no hay archivo guardado."""
    return {
        "fecha_otorgamiento": date.today(),
        "capital_solicitado": 10_000_000.0,
        "cuotas": 18,
        "tasa_anual_pct": 78.0,
        # Planificación desde abril: el CSV se filtra desde este mes (marzo no entra).
        "fecha_inicio_cf": date(2026, 4, 1),
        "horizonte_meses": 48,
        "credito_promedio": 10_000_000.0,
        "capital_colocado_inicial": 250_000_000.0,
        "crecimiento_colocacion_mensual": 30_000_000.0,
        "capital_colocado_objetivo": 250_000_000.0,
        "comision_comercial_pct": 7.0,
        "gastos_fijos_mensuales": 20_000_000.0,
        "iibb_pct": 5.0,
        "incobrabilidad_pct": 6.0,
        "tasa_descuento_van_cf": 3.0,
        "valor_residual_cartera_pct": 80.0,
        "deuda_inicial_financ": 850_000_000.0,
        "tna_financ_pct": 34.0,
        "pct_cobranza_existente_pct": 90.0,
        # Nombre del archivo en la carpeta del simulador
        "vencimientos_nombre_archivo_disco": VENCIMIENTOS_CSV_DISCO_DEFAULT,
        # True = ya se aplicó el salto a abril 2026 en prefs viejos (no repetir)
        "migracion_planificacion_desde_abril2026": False,
    }


def _coercer_valor_input(key: str, val: Any) -> Any:
    if val is None:
        return _defaults_inputs_simulador().get(key)
    if key in _INPUT_DATE_KEYS:
        if isinstance(val, date):
            return val
        if isinstance(val, str):
            try:
                return date.fromisoformat(val)
            except ValueError:
                return _defaults_inputs_simulador().get(key)
        return _defaults_inputs_simulador().get(key)
    if key in ("cuotas", "horizonte_meses"):
        return int(round(float(val)))
    if key == "vencimientos_nombre_archivo_disco":
        s = str(val).strip() if val is not None else ""
        return s if s else VENCIMIENTOS_CSV_DISCO_DEFAULT
    if key == "migracion_planificacion_desde_abril2026":
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.strip().lower() in ("1", "true", "yes", "si", "sí")
        return bool(val)
    return float(val)


def _serializar_valor_input(val: Any) -> Any:
    if isinstance(val, date):
        return val.isoformat()
    return val


def aplicar_inputs_guardados(ruta_prefs: str) -> None:
    """Carga JSON en st.session_state; completa claves nuevas al actualizar el simulador sin perder la sesión."""
    defaults = _defaults_inputs_simulador()
    prefs: Dict[str, Any] = {}
    if os.path.isfile(ruta_prefs):
        try:
            with open(ruta_prefs, encoding="utf-8") as f:
                prefs = json.load(f)
        except Exception:
            prefs = {}
    primera_vez = not st.session_state.get("_inputs_prefs_aplicados")
    if primera_vez:
        st.session_state["_inputs_prefs_aplicados"] = True
    for key, dflt in defaults.items():
        if primera_vez or key not in st.session_state:
            raw = prefs.get(key, dflt)
            st.session_state[key] = _coercer_valor_input(key, raw)

    # Una sola vez (si el JSON viejo no tenía esta marca): planificación desde abril 2026 — no se usa marzo en cartera
    if "migracion_planificacion_desde_abril2026" not in prefs:
        st.session_state["fecha_inicio_cf"] = date(2026, 4, 1)
        st.session_state["migracion_planificacion_desde_abril2026"] = True

    # Inicio simulación: no usar marzo 2026 (planificación desde abril). Unifica criterio en Cashflow y Cartera existente.
    fi0 = st.session_state.get("fecha_inicio_cf")
    if isinstance(fi0, date) and fi0.year == 2026 and fi0.month == 3:
        st.session_state["fecha_inicio_cf"] = date(2026, 4, 1)

    # Versión vieja: había un segundo date_input en «Cartera existente»; ya no se usa (evita que quede «marzo» solo ahí).
    st.session_state.pop("cartera_existente_desde", None)

    # Dejar de leer el export de marzo: usar «Vencimientos desde Abril2026» y olvidar CSV subido viejo.
    _nom_v = str(st.session_state.get("vencimientos_nombre_archivo_disco", "")).strip()
    if _nom_v and "marzo2026" in _nom_v.lower():
        st.session_state["vencimientos_nombre_archivo_disco"] = VENCIMIENTOS_CSV_DISCO_DEFAULT
        st.session_state.pop("vencimientos_cartera_upload", None)


def guardar_inputs_actuales(ruta_prefs: str) -> bool:
    """Persiste los valores actuales de los inputs (mismas keys que defaults). Devuelve True si se escribió bien."""
    defaults = _defaults_inputs_simulador()
    out: Dict[str, Any] = {}
    try:
        for key in defaults:
            if key not in st.session_state:
                continue
            out[key] = _serializar_valor_input(st.session_state[key])
        with open(ruta_prefs, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        st.session_state.pop("_prefs_guardado_error", None)
        return True
    except Exception as ex:
        st.session_state["_prefs_guardado_error"] = str(ex)
        return False


def cargar_dataframe_vencimientos(
    archivo_subido: Optional[Any],
    ruta_disco: str,
    fecha_desde: Optional[date],
) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Cartera existente: si hay CSV subido en la app, tiene prioridad (útil para un export al día);
    si no, se usa el archivo en la carpeta del simulador.
    """
    if archivo_subido is not None:
        ag = leer_vencimientos_desde_bytes(archivo_subido.getvalue(), fecha_desde=fecha_desde)
        nombre = getattr(archivo_subido, "name", "archivo.csv")
        return ag, f"CSV subido ({nombre})"
    ag = leer_vencimientos_real(ruta_disco, fecha_desde=fecha_desde)
    if ag is not None:
        return ag, f"Archivo en carpeta ({os.path.basename(ruta_disco)})"
    return None, "(sin archivo)"


def _filtrar_agrupado_desde_mes(ag: Optional[pd.DataFrame], desde: date) -> Optional[pd.DataFrame]:
    """Solo filas (año, mes) >= mes de inicio elegido (evita que quede marzo si el inicio es abril)."""
    if ag is None or ag.empty:
        return ag
    y0, m0 = desde.year, desde.month
    mask = (ag["año"] > y0) | ((ag["año"] == y0) & (ag["mes"] >= m0))
    out = ag.loc[mask].copy()
    return out if not out.empty else None


def main() -> None:
    st.set_page_config(
        page_title="Simulador prendario",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.title("Simulador de crédito prendario")

    # Reducimos un poco el tamaño visual para que entre más información en pantalla
    st.markdown(
        """
        <style>
        .block-container {
            zoom: 0.8;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    _dir_sim = os.path.dirname(os.path.abspath(__file__))
    ruta_prefs_inputs = os.path.join(_dir_sim, "simulador_inputs.json")
    aplicar_inputs_guardados(ruta_prefs_inputs)

    tab_conf, tab_unitario, tab_cashflow, tab_cartera = st.tabs(
        ["CSV y mes de inicio", "Crédito unitario", "Cashflow", "Cartera existente"]
    )

    with tab_conf:
        st.header("Archivo CSV y mes de inicio")
        st.warning(
            "**Primera solapa:** todo lo que define la cartera existente está **acá**. "
            "Si solo abrís «Cartera existente» y ves la tabla, **no** vas a ver nombre de archivo ni fecha: vení a esta solapa."
        )
        st.subheader("Archivo de cartera (carpeta + opcional subida)")
        st.text_input(
            "Nombre del archivo CSV (misma carpeta que este programa)",
            key="vencimientos_nombre_archivo_disco",
            help=f"Por defecto suele ser {VENCIMIENTOS_CSV_DISCO_DEFAULT}",
        )
        _nom_csv = str(st.session_state.get("vencimientos_nombre_archivo_disco", VENCIMIENTOS_CSV_DISCO_DEFAULT)).strip()
        ruta_venc_default = os.path.normpath(os.path.join(_dir_sim, _nom_csv or VENCIMIENTOS_CSV_DISCO_DEFAULT))
        st.code(str(Path(ruta_venc_default).resolve()), language=None)
        if os.path.isfile(ruta_venc_default):
            try:
                mtime = os.path.getmtime(ruta_venc_default)
                mtxt = datetime.fromtimestamp(mtime).strftime("%d/%m/%Y %H:%M:%S")
            except OSError:
                mtxt = "?"
            st.success(f"Archivo en carpeta encontrado — última modificación en disco: **{mtxt}**")
        else:
            st.error("No existe ese archivo en esa ruta. Revisá el nombre o que el CSV esté en la carpeta del simulador.")

        st.caption(
            "Parámetros (fechas, montos…) se guardan en **simulador_inputs.json** en esa carpeta al recargar la página."
        )
        _err_p = st.session_state.get("_prefs_guardado_error")
        if _err_p:
            st.error(f"No se pudo escribir el archivo de preferencias: {_err_p}")

        archivo_venc_upload = st.file_uploader(
            "Opcional: subir un .csv distinto (mientras esté cargado, va por delante del archivo de la carpeta)",
            type=["csv"],
            key="vencimientos_cartera_upload",
            help="Separador ; , encoding latin-1. Columnas: Fecha_Vto, Capital, Interes_Ordinario, etc.",
        )
        if archivo_venc_upload is not None:
            st.warning(
                f"**Modo subida activo:** se está usando **{archivo_venc_upload.name}**, no el CSV de la ruta de arriba."
            )
        else:
            st.info("**Modo carpeta:** se lee el archivo de la ruta que figura arriba.")

        if st.button(
            "Descartar archivo subido y volver al CSV de la carpeta",
            key="btn_csv_solo_carpeta",
            help="Borra la subida del navegador y recarga leyendo el .csv de la ruta de arriba.",
        ):
            st.session_state.pop("vencimientos_cartera_upload", None)
            st.rerun()

        st.markdown("---")
        st.subheader("Mes de inicio del modelo (cartera existente + cashflow)")
        fecha_inicio_cf = st.date_input(
            "Inicio simulación — primer día del mes (mes 0)",
            key="fecha_inicio_cf",
            help="La tabla y el cashflow solo muestran meses desde acá (ej. 1 abr 2026 para no incluir marzo).",
        )

    with tab_unitario:
        st.header("Simulación de crédito unitario")

        st.subheader("Parámetros del crédito")
        p1, p2 = st.columns(2)
        with p1:
            fecha_otorgamiento = st.date_input(
                "Fecha de otorgamiento del crédito",
                key="fecha_otorgamiento",
            )
            c1a, c1b = st.columns([2, 1])
            with c1a:
                capital_solicitado = st.number_input(
                    "Capital promedio solicitado ($)",
                    min_value=0.0,
                    value=10_000_000.0,
                    step=100_000.0,
                    format="%.0f",
                    key="capital_solicitado",
                )
            with c1b:
                st.markdown(f"**{formato_pesos(capital_solicitado)}**")
            cuotas = st.number_input(
                "Cantidad de cuotas mensuales",
                min_value=1,
                max_value=120,
                value=18,
                step=1,
                key="cuotas",
            )
            c2a, c2b = st.columns([2, 1])
            with c2a:
                tasa_anual_pct = st.number_input(
                    "Tasa anual TNA (%)",
                    min_value=0.0,
                    max_value=300.0,
                    value=78.0,
                    step=1.0,
                    key="tasa_anual_pct",
                )
                tasa_anual = tasa_anual_pct / 100.0
            with c2b:
                st.markdown(f"**{tasa_anual_pct:.2f} %**")
        with p2:
            # Valores fijos según el sistema (no editables)
            quebranto_pct = 0.03  # 3%
            gasto_admin_pct = 0.0015  # 0,15% por cuota
            iva_interes_pct = 0.21  # 21%
            iva_gastos_pct = 0.21  # 21%
            st.caption("Sistema: quebranto 3%, gastos adm. 0,15%, IVA 21%.")

        # El quebranto se AGREGA al capital solicitado.
        capital_financiado = capital_solicitado * (1 + quebranto_pct)

        # Simplificación: tasa mensual lineal (TNA / 12)
        tasa_mensual = tasa_anual / 12 if tasa_anual > 0 else 0.0

        # Gastos administrativos por cuota = % del capital financiado (igual todos los meses)
        gasto_cuota = capital_financiado * gasto_admin_pct

        # Calculamos la CUOTA TOTAL constante que deja saldo final ~ 0
        cuota_total_const = calcular_cuota_total(
            capital_financiado,
            tasa_mensual,
            cuotas,
            gasto_cuota,
            iva_interes_pct,
            iva_gastos_pct,
        )

        # Armamos el cuadro de amortización (descomponiendo la cuota total)
        filas = []
        saldo = capital_financiado
        for n in range(1, cuotas + 1):
            saldo_inicial = saldo
            interes = saldo_inicial * tasa_mensual
            iva_interes = interes * iva_interes_pct
            iva_gastos = gasto_cuota * iva_gastos_pct

            amortizacion = cuota_total_const - (interes + iva_interes + gasto_cuota + iva_gastos)
            saldo_final = saldo_inicial - amortizacion
            fecha_vto = pd.to_datetime(fecha_otorgamiento) + pd.DateOffset(months=n)

            filas.append(
                {
                    "Cuota N°": n,
                    "Fecha vto.": fecha_vto.date(),
                    "Saldo inicial": saldo_inicial,
                    "Cuota base": cuota_total_const,
                    "Interés": interes,
                    "Amortización": amortizacion,
                    "Gastos": gasto_cuota,
                    "IVA intereses": iva_interes,
                    "IVA gastos": iva_gastos,
                    "Cuota total": cuota_total_const,
                    "Saldo final": max(saldo_final, 0.0),
                }
            )
            saldo = saldo_final

        df = pd.DataFrame(filas)

        flujo_inicial = -capital_solicitado
        flujos = [flujo_inicial] + df["Cuota total"].tolist()
        tir_mensual = calcular_irr(flujos)
        tir_anual = (1 + tir_mensual) ** 12 - 1 if tir_mensual != 0 else 0.0

        total_cuota = df["Cuota total"].sum()
        total_interes = df["Interés"].sum()
        total_amortizacion = df["Amortización"].sum()
        total_gastos = df["Gastos"].sum()
        total_iva_int = df["IVA intereses"].sum()
        total_iva_gastos = df["IVA gastos"].sum()

        totals_row = {
            "Cuota N°": "Total",
            "Fecha vto.": "",
            "Saldo inicial": None,
            "Cuota base": total_cuota,
            "Interés": total_interes,
            "Amortización": total_amortizacion,
            "Gastos": total_gastos,
            "IVA intereses": total_iva_int,
            "IVA gastos": total_iva_gastos,
            "Cuota total": total_cuota,
            "Saldo final": 0.0,
        }

        df_con_totales = pd.concat([df, pd.DataFrame([totals_row])], ignore_index=True)

        st.subheader("Resultados")
        # TNA = tasa nominal anual; TEA = tasa efectiva anual (capitalización mensual); CFT = costo financiero total (TIR anual, incluye todo)
        tea = (1 + tasa_mensual) ** 12 - 1 if tasa_mensual != 0 else 0.0
        cft = tir_anual  # CFT = costo financiero total para el cliente (misma TIR anual del flujo del crédito)

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Cuota total mensual", formato_pesos(df.loc[0, "Cuota total"]))
        with col2:
            st.metric("TIR mensual del crédito", f"{tir_mensual * 100:.2f} %")
        with col3:
            st.metric("TIR anual del crédito", f"{tir_anual * 100:.2f} %")
        with col4:
            st.metric("Capital financiado", formato_pesos(capital_financiado))
        with col5:
            st.metric("Total del crédito (suma de cuotas)", formato_pesos(total_cuota))

        r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns(5)
        with r2c1:
            st.metric("TNA (tasa nominal anual)", f"{tasa_anual_pct:.2f} %")
        with r2c2:
            st.metric("TEA (tasa efectiva anual)", f"{tea * 100:.2f} %")
        with r2c3:
            st.metric("CFT (costo financiero total)", f"{cft * 100:.2f} %")
        st.caption("TNA = tasa que ingresaste. TEA = (1 + TNA/12)^12 − 1. CFT = costo financiero total anual para el cliente (TIR del flujo de la operación).")

        st.subheader("Cuadro de cuotas")
        st.dataframe(
            df_con_totales.set_index("Cuota N°").style.format(
                {
                    "Saldo inicial": formato_pesos,
                    "Cuota base": formato_pesos,
                    "Interés": formato_pesos,
                    "Amortización": formato_pesos,
                    "Gastos": formato_pesos,
                    "IVA intereses": formato_pesos,
                    "IVA gastos": formato_pesos,
                    "Cuota total": formato_pesos,
                    "Saldo final": formato_pesos,
                }
            )
        )

    with tab_cashflow:
        st.header("Cashflow de colocación mensual")
        st.info(
            "**Cartera ya colocada:** del CSV se toman solo las filas con **Fecha_Vto desde el mes de inicio** (por defecto **abril 2026**; lo anterior no se usa). "
            "Encima va la **nueva colocación** simulada. Cuotas nuevas a **mes vencido** (en el mes 0 de cada cohorte la cobranza nueva suele ser 0). "
            f"TNA perfil unitario: **{tasa_anual * 100:.2f} %** anual."
        )

        st.subheader("Parámetros de colocación")
        st.caption(
            f"**Mes inicio del modelo:** {MESES_ES[fecha_inicio_cf.month - 1]} {fecha_inicio_cf.year} "
            "(lo cambiás en la solapa **CSV y mes de inicio**)."
        )
        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            horizonte_meses = st.number_input(
                "Horizonte (meses)", min_value=1, max_value=120, value=48, step=1, key="horizonte_meses"
            )
        with r1c2:
            credito_promedio = st.number_input(
                "Crédito promedio ($)",
                min_value=1_000_000.0,
                value=10_000_000.0,
                step=100_000.0,
                format="%.0f",
                key="credito_promedio",
            )
            st.caption(formato_pesos(credito_promedio))
        with r1c3:
            capital_colocado_inicial = st.number_input(
                "Capital colocado inicial mes 0 ($)",
                min_value=0.0,
                value=250_000_000.0,
                step=10_000_000.0,
                format="%.0f",
                key="capital_colocado_inicial",
            )
            st.caption(formato_pesos(capital_colocado_inicial))

        r1b1, r1b2 = st.columns(2)
        with r1b1:
            crecimiento_colocacion_mensual = st.number_input(
                "Crecimiento colocación mensual ($)",
                min_value=0.0,
                value=30_000_000.0,
                step=5_000_000.0,
                format="%.0f",
                key="crecimiento_colocacion_mensual",
            )
            crecimiento_mm = crecimiento_colocacion_mensual / 1_000_000 if crecimiento_colocacion_mensual else 0.0
            st.caption(
                f"Incremento por mes: {formato_pesos(crecimiento_colocacion_mensual)} "
                f"({crecimiento_mm:,.1f} millones)".replace(",", "X").replace(".", ",").replace("X", ".")
            )
        with r1b2:
            capital_colocado_objetivo = st.number_input(
                "Capital colocado objetivo/mes ($)",
                min_value=0.0,
                value=250_000_000.0,
                step=10_000_000.0,
                format="%.0f",
                key="capital_colocado_objetivo",
            )
            objetivo_mm = capital_colocado_objetivo / 1_000_000 if capital_colocado_objetivo else 0.0
            st.caption(
                "Tope de colocación mensual (se alcanza y luego se mantiene) "
                f"= {objetivo_mm:,.1f} millones".replace(",", "X").replace(".", ",").replace("X", ".")
            )

        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        with r2c1:
            comision_comercial_pct = st.number_input(
                "Comisión colocación (%)",
                min_value=0.0,
                max_value=100.0,
                value=7.0,
                step=1.0,
                key="comision_comercial_pct",
            )
        with r2c2:
            gastos_fijos_mensuales = st.number_input(
                "Gastos fijos/mes ($)",
                min_value=0.0,
                value=20_000_000.0,
                step=1_000_000.0,
                format="%.0f",
                key="gastos_fijos_mensuales",
            )
            st.caption(formato_pesos(gastos_fijos_mensuales))
        with r2c3:
            iibb_pct = st.number_input(
                "IIBB sobre intereses (%)",
                min_value=0.0,
                max_value=100.0,
                value=5.0,
                step=1.0,
                key="iibb_pct",
            )
        with r2c4:
            incobrabilidad_pct = st.number_input(
                "Incobrabilidad (%)",
                min_value=0.0,
                max_value=90.0,
                value=6.0,
                step=1.0,
                key="incobrabilidad_pct",
            )

        # Tasa de descuento para el VAN (mensual), usada luego en los indicadores
        tasa_descuento_pct = st.number_input(
            "Tasa desc. VAN (% mensual)",
            min_value=0.0,
            max_value=20.0,
            value=3.0,
            step=0.25,
            key="tasa_descuento_van_cf",
        )
        # Porcentaje del saldo de cartera a cobrar al cierre que se reconoce como valor residual
        valor_residual_cartera_pct = st.number_input(
            "% valor residual de cartera al cierre",
            min_value=0.0,
            max_value=150.0,
            value=80.0,
            step=5.0,
            key="valor_residual_cartera_pct",
        )

        imp_debcred_pct = 0.012

        with st.expander("Financiamiento (mercado de capitales)", expanded=False):
            f1, f2 = st.columns(2)
            with f1:
                deuda_inicial_financ = st.number_input(
                    "Deuda inicial ($)",
                    min_value=0.0,
                    value=850_000_000.0,
                    step=50_000_000.0,
                    format="%.0f",
                    key="deuda_inicial_financ",
                )
                st.caption(formato_pesos(deuda_inicial_financ))
            with f2:
                tna_financ_pct = st.number_input(
                    "TNA financiamiento (%)",
                    min_value=0.0,
                    max_value=200.0,
                    value=34.0,
                    step=1.0,
                    key="tna_financ_pct",
                )
            st.caption("Si flujo < 0 se toma deuda; si > 0 se abona. Intereses: (TNA/100)/12 sobre saldo.")

        # Cartera existente: subida (prioridad) o archivo en carpeta
        ag_vencimientos, _origen_cartera = cargar_dataframe_vencimientos(
            archivo_venc_upload, ruta_venc_default, fecha_desde=fecha_inicio_cf
        )
        ag_vencimientos = _filtrar_agrupado_desde_mes(ag_vencimientos, fecha_inicio_cf)
        c1, c2 = st.columns([1, 1])
        with c1:
            pct_cobranza_existente_pct = st.number_input(
                "% cobranza efectiva cartera existente",
                min_value=50.0,
                max_value=100.0,
                value=90.0,
                step=1.0,
                key="pct_cobranza_existente_pct",
            )
        with c2:
            if ag_vencimientos is not None:
                st.caption(
                    f"Cartera existente: {_origen_cartera} — {len(ag_vencimientos)} meses "
                    "(tras filtrar desde inicio simulación)."
                )
                y0, m0 = fecha_inicio_cf.year, fecha_inicio_cf.month
                hit = ag_vencimientos[
                    (ag_vencimientos["año"] == y0) & (ag_vencimientos["mes"] == m0)
                ]
                if not hit.empty:
                    bruto = float(hit["Cobranza_Total"].iloc[0])
                    st.caption(
                        f"Control **mes inicio** ({MESES_ES[m0 - 1]} {y0}): Cobranza_Total en CSV = **{formato_pesos(bruto)}** "
                        f"(después se aplica el % cobranza efectiva a la izquierda)."
                    )
            else:
                st.caption(
                    "No hay cartera existente: revisá el **nombre del CSV** y la **ruta** en la sección «Archivo de cartera» arriba del todo."
                )

        if st.button("Calcular cashflow de colocación mensual"):
            meses = list(range(int(horizonte_meses)))

            # Convertimos porcentajes de entrada (0-100) a proporciones (0-1)
            comision_comercial_prop = comision_comercial_pct / 100.0
            iibb_prop = iibb_pct / 100.0
            incobrabilidad_prop = incobrabilidad_pct / 100.0
            tna_financ = tna_financ_pct / 100.0
            pct_cobranza_existente = pct_cobranza_existente_pct / 100.0

            _bruto_mes0 = None
            if ag_vencimientos is not None and not ag_vencimientos.empty:
                _hm = ag_vencimientos[
                    (ag_vencimientos["año"] == fecha_inicio_cf.year)
                    & (ag_vencimientos["mes"] == fecha_inicio_cf.month)
                ]
                if not _hm.empty:
                    _bruto_mes0 = float(_hm["Cobranza_Total"].iloc[0])
            if _bruto_mes0 is not None:
                st.success(
                    f"**Archivo usado en este cálculo:** `{Path(ruta_venc_default).resolve()}`  \n"
                    f"Mes inicio (**{MESES_ES[fecha_inicio_cf.month - 1]} {fecha_inicio_cf.year}**): "
                    f"Cobranza_Total **bruta** en CSV = **{formato_pesos(_bruto_mes0)}** → "
                    f"con **{pct_cobranza_existente_pct:.0f}%** cobranza efectiva = **{formato_pesos(_bruto_mes0 * pct_cobranza_existente)}**."
                )
            else:
                st.error(
                    "No hay fila de cartera existente para el **mes inicio** en el CSV (o no se leyó el archivo). "
                    "Revisá la ruta del CSV arriba (sección «Archivo de cartera») y que el export tenga vencimientos para ese mes."
                )

            # Diccionario (año, mes) -> cobranza real, IVA y Seguro para cartera existente
            real_por_mes = {}
            if ag_vencimientos is not None and not ag_vencimientos.empty:
                for _, row in ag_vencimientos.iterrows():
                    y, m = int(row["año"]), int(row["mes"])
                    real_por_mes[(y, m)] = {
                        "Cobranza_Total": float(row["Cobranza_Total"]),
                        "IVA_Total": float(row["IVA_Total"]),
                        "Seguro_Total": float(row["Seguro_Total"]),
                        "Interes_Ordinario": float(row["Interes_Ordinario"]),
                    }

            # Perfil de cuota del tab "unitario" escalado al ticket promedio de esta simulación.
            # Sin esto, si capital_solicitado ≠ credito_promedio, el pagaré/cobranza queda mal (ej. mucha colocación y poco pagaré).
            escala_ticket = (credito_promedio / capital_solicitado) if capital_solicitado and capital_solicitado > 0 else 1.0
            interes_por_vida = (df["Interés"] * escala_ticket).tolist()
            cuota_total_por_vida = (df["Cuota total"] * escala_ticket).tolist()
            # Gastos por cuota (igual en todas) para IVA sobre gastos cobrados
            gasto_cuota_promedio = credito_promedio * (1 + quebranto_pct) * gasto_admin_pct
            if abs(credito_promedio - capital_solicitado) > 1.0:
                st.caption(
                    f"Escala aplicada al perfil del crédito unitario: crédito promedio ÷ capital del unitario = **{escala_ticket:,.4f}** "
                    "(así el pagaré y las cuotas coinciden con el ticket promedio).".replace(",", "X").replace(".", ",").replace("X", ".")
                )

            # Flujos por mes calendario
            fechas_cf = []
            capital_colocado_mes = []
            operaciones_mes = []
            cobranza_cuotas_nueva_mes = []
            cobranza_existente_mes = []
            cobranza_total_mes = []
            intereses_cobrados_mes = []
            iva_intereses_mes = []
            iva_gastos_mes = []
            iva_pagado_existente_mes = []
            seguros_pagados_mes = []
            comisiones_mes = []
            gastos_fijos_mes = []
            imp_debcred_mes = []
            iibb_mes = []
            egresos_totales_mes = []
            flujo_neto_mes = []
            flujo_acumulado_mes = []

            # Vistas para la evolución de la cartera a cobrar (solo cartera nueva)
            pagare_generado_mes = []
            intereses_devengados_nueva_mes = []
            caida_cartera_cobrada_mes = []
            saldo_cartera_mes = []
            saldo_cartera = 0.0

            comision_mes_anterior = 0.0
            flujo_acum = 0.0

            for t in meses:
                fecha_t = pd.to_datetime(fecha_inicio_cf) + pd.DateOffset(months=t)
                fechas_cf.append(fecha_t.date())

                # Cartera existente (vencimientos reales): cobranza del mes; IVA y seguros se pagan al mes siguiente
                # Se aplica % de cobranza (ej. 85-90%): solo ese % se percibe efectivamente; el resto cae en meses siguientes
                cobranza_existente_t = real_por_mes.get((fecha_t.year, fecha_t.month), {}).get("Cobranza_Total", 0.0) * pct_cobranza_existente
                intereses_existente_t = real_por_mes.get((fecha_t.year, fecha_t.month), {}).get("Interes_Ordinario", 0.0) * pct_cobranza_existente
                if t >= 1:
                    fecha_ant = pd.to_datetime(fecha_inicio_cf) + pd.DateOffset(months=t - 1)
                    iva_pagado_existente_t = real_por_mes.get((fecha_ant.year, fecha_ant.month), {}).get("IVA_Total", 0.0) * pct_cobranza_existente
                    seguros_pagados_t = real_por_mes.get((fecha_ant.year, fecha_ant.month), {}).get("Seguro_Total", 0.0) * pct_cobranza_existente
                else:
                    iva_pagado_existente_t = 0.0
                    seguros_pagados_t = 0.0
                cobranza_existente_mes.append(cobranza_existente_t)
                iva_pagado_existente_mes.append(iva_pagado_existente_t)
                seguros_pagados_mes.append(seguros_pagados_t)

                # Egresos de colocación del mes t
                # Colocación mensual creciente: parte de un nivel inicial y crece linealmente hasta un objetivo.
                col_mes = capital_colocado_inicial + crecimiento_colocacion_mensual * t
                if capital_colocado_objetivo > 0:
                    col_mes = min(col_mes, capital_colocado_objetivo)
                capital_colocado_mes.append(col_mes)
                # Cantidad de operaciones nuevas del mes (créditos originados)
                n_creditos_mes_t = col_mes / credito_promedio if credito_promedio > 0 else 0.0
                # Importante: NO redondear a entero para "cantidad", porque distorsiona la simulación
                # cuando se usa un crédito promedio y una colocación mensual (pueden existir "operaciones equivalentes").
                operaciones_mes.append(float(n_creditos_mes_t))

                # Comisiones comerciales (mes vencido): se pagan sobre la colocación del mes anterior
                com_mes = comision_mes_anterior
                comisiones_mes.append(com_mes)
                # Se calcula la comisión del mes actual para pagar el próximo
                comision_mes_anterior = col_mes * comision_comercial_prop

                # Gastos fijos
                gf_mes = gastos_fijos_mensuales
                gastos_fijos_mes.append(gf_mes)

                # Ingresos por cobranza de cuotas (todas las cohortes hasta ese mes)
                # NOTA: las cuotas se pagan a mes vencido.
                cobranza_t = 0.0
                intereses_brutos_t = 0.0
                gastos_cobrados_brutos_t = 0.0
                for origen in range(t):
                    edad = t - 1 - origen  # mes de vida del crédito (0 = primera cuota)
                    if 0 <= edad < cuotas:
                        # Créditos originados en el mes "origen" (proporcional a la colocación de ese mes)
                        col_origen = capital_colocado_mes[origen]
                        n_creditos_origen = col_origen / credito_promedio if credito_promedio > 0 else 0.0
                        cobranza_t += n_creditos_origen * cuota_total_por_vida[edad]
                        intereses_brutos_t += n_creditos_origen * interes_por_vida[edad]
                        gastos_cobrados_brutos_t += n_creditos_origen * gasto_cuota_promedio

                # Aplicamos incobrabilidad sobre la caída mensual de la cartera nueva
                cobranza_neta_t = cobranza_t * (1 - incobrabilidad_prop)
                intereses_netos_t = intereses_brutos_t * (1 - incobrabilidad_prop)
                gastos_cobrados_netos_t = gastos_cobrados_brutos_t * (1 - incobrabilidad_prop)

                # Intereses devengados (cartera nueva): incluye el mes de la colocación (1er mes de vida = interés sobre saldo inicial).
                # Distinto de "intereses cobrados" en caja: la cobranza de cuotas sigue siendo a mes vencido (primer cobro en t+1).
                intereses_devengados_t = 0.0
                for origen in range(t + 1):
                    edad = t - origen
                    if 0 <= edad < cuotas:
                        col_origen = capital_colocado_mes[origen]
                        n_co = col_origen / credito_promedio if credito_promedio > 0 else 0.0
                        intereses_devengados_t += n_co * interes_por_vida[edad]

                # Cartera a cobrar (solo cartera nueva):
                # Pagaré generado en el mes = créditos del mes × suma de cuotas de un crédito al ticket promedio (ya escalado).
                pagare_por_credito = sum(cuota_total_por_vida)
                pagare_gen_t = n_creditos_mes_t * pagare_por_credito
                pagare_generado_mes.append(pagare_gen_t)
                intereses_devengados_nueva_mes.append(intereses_devengados_t)
                caida_cartera_cobrada_mes.append(cobranza_neta_t)
                saldo_cartera += pagare_gen_t - cobranza_neta_t
                saldo_cartera_mes.append(saldo_cartera)

                # Cobranza: separar nueva vs existente y luego totalizar
                cobranza_total_t = cobranza_neta_t + cobranza_existente_t
                cobranza_cuotas_nueva_mes.append(cobranza_neta_t)
                cobranza_total_mes.append(cobranza_total_t)
                intereses_cobrados_mes.append(intereses_netos_t + intereses_existente_t)

                # IVA por percibido (cartera nueva): sobre intereses y gastos cobrados (egreso)
                iva_intereses_t = iva_interes_pct * intereses_netos_t
                iva_gastos_t = iva_gastos_pct * gastos_cobrados_netos_t
                iva_intereses_mes.append(iva_intereses_t)
                iva_gastos_mes.append(iva_gastos_t)

                # Impuestos: incluye IVA y Seguros pagados (cartera existente, al mes siguiente de cobro)
                egresos_antes_debcred = (
                    col_mes + com_mes + gf_mes + iva_intereses_t + iva_gastos_t
                    + iva_pagado_existente_t + seguros_pagados_t
                )
                debcred_t = egresos_antes_debcred * imp_debcred_pct
                imp_debcred_mes.append(debcred_t)

                # IIBB sobre intereses cobrados (nueva + existente)
                iibb_t = (intereses_netos_t + intereses_existente_t) * iibb_prop
                iibb_mes.append(iibb_t)

                egresos_tot_t = egresos_antes_debcred + debcred_t + iibb_t
                egresos_totales_mes.append(egresos_tot_t)

                flujo_t = cobranza_total_t - egresos_tot_t
                flujo_neto_mes.append(flujo_t)
                flujo_acum += flujo_t
                flujo_acumulado_mes.append(flujo_acum)

            # Etiquetas de mes: "Marzo 2026", "Abril 2026", ...
            etiquetas_meses = [f"{MESES_ES[d.month - 1]} {d.year}" for d in fechas_cf]

            # Financiamiento: deuda inicial, intereses mensuales (TNA/12), cuando flujo < 0 se toma más deuda, cuando > 0 se paga deuda
            tasa_financ_mensual = tna_financ / 12.0 if tna_financ > 0 else 0.0
            saldo_deuda = deuda_inicial_financ
            intereses_financ_mes = []
            pago_deuda_mes = []
            saldo_deuda_mes = []
            for t in meses:
                intereses_t = saldo_deuda * tasa_financ_mensual
                deuda_despues_intereses = saldo_deuda + intereses_t
                flujo_operativo_t = flujo_neto_mes[t]
                if flujo_operativo_t < 0:
                    # Se toma más deuda para cubrir el faltante
                    saldo_deuda = deuda_despues_intereses + (-flujo_operativo_t)
                    pago_t = 0.0
                else:
                    # Se usa el flujo positivo para pagar deuda hasta quedar saneada
                    pago_t = min(flujo_operativo_t, deuda_despues_intereses)
                    saldo_deuda = max(0.0, deuda_despues_intereses - pago_t)
                intereses_financ_mes.append(intereses_t)
                pago_deuda_mes.append(pago_t)
                saldo_deuda_mes.append(saldo_deuda)

            # Mes de saneamiento (primera vez que saldo deuda <= 0)
            mes_saneamiento_idx = None
            for t, sal in enumerate(saldo_deuda_mes):
                if sal <= 0 and deuda_inicial_financ > 0:
                    mes_saneamiento_idx = t
                    break
            mes_saneamiento_texto = etiquetas_meses[mes_saneamiento_idx] if mes_saneamiento_idx is not None else "No saneado en el horizonte"

            # Flujo del negocio después de financiamiento: resta intereses de la deuda.
            # El acumulado arranca restando la deuda inicial (posición de partida).
            flujo_neto_despues_financ_mes = [flujo_neto_mes[t] - intereses_financ_mes[t] for t in meses]
            flujo_acum_despues_financ = -deuda_inicial_financ
            flujo_acumulado_despues_financ_mes = []
            for f in flujo_neto_despues_financ_mes:
                flujo_acum_despues_financ += f
                flujo_acumulado_despues_financ_mes.append(flujo_acum_despues_financ)

            # Valor residual de cartera al cierre del horizonte (solo cartera nueva)
            saldo_cartera_final = saldo_cartera_mes[-1] if saldo_cartera_mes else 0.0
            valor_residual_cartera = saldo_cartera_final * (valor_residual_cartera_pct / 100.0)

            # Indicadores financieros del cashflow (incluyen valor residual de cartera)
            flujos_con_residual = flujo_neto_mes.copy()
            if flujos_con_residual:
                flujos_con_residual[-1] += valor_residual_cartera

            tir_cf_mensual = calcular_irr(flujos_con_residual)
            tir_cf_anual = (1 + tir_cf_mensual) ** 12 - 1 if tir_cf_mensual != 0 else 0.0
            flujo_total = sum(flujos_con_residual) if flujos_con_residual else 0.0
            # VAN descontado a la tasa elegida en los parámetros
            tasa_descuento_mensual = tasa_descuento_pct / 100.0
            van_descontado = calcular_van(flujos_con_residual, tasa_descuento_mensual)
            # Mes de recupero: primer mes con flujo acumulado >= 0
            mes_recupero_idx = None
            for t, ac in enumerate(flujo_acumulado_mes):
                if ac >= 0:
                    mes_recupero_idx = t
                    break
            mes_recupero_texto = etiquetas_meses[mes_recupero_idx] if mes_recupero_idx is not None else "No recuperado en el horizonte"

            # Capital restante para breakeven: suma de los flujos de caja negativos hasta que la empresa empiece a repagar
            primer_pago_idx = None
            for t in meses:
                if pago_deuda_mes[t] > 0:
                    primer_pago_idx = t
                    break
            capital_restante_breakeven = 0.0
            if primer_pago_idx is not None:
                for t in range(primer_pago_idx):
                    if flujo_neto_mes[t] < 0:
                        capital_restante_breakeven += -flujo_neto_mes[t]
            else:
                # No hubo repago en el horizonte: sumar todos los flujos negativos del periodo
                for t in meses:
                    if flujo_neto_mes[t] < 0:
                        capital_restante_breakeven += -flujo_neto_mes[t]

            # Ingresos y egresos totales del periodo
            ingresos_totales = sum(cobranza_total_mes)
            egresos_totales = sum(egresos_totales_mes)
            total_operaciones_nuevas = float(sum(operaciones_mes))

            # Mes de breakeven por flujo neto: primer mes con flujo neto de caja positivo
            mes_breakeven_idx = None
            for t, f in enumerate(flujo_neto_mes):
                if f > 0:
                    mes_breakeven_idx = t
                    break
            mes_breakeven_texto = (
                etiquetas_meses[mes_breakeven_idx] if mes_breakeven_idx is not None else "Sin flujo neto positivo"
            )

            st.subheader("Indicadores financieros del cashflow")

            # Fila 1: TIRs y VAN
            fila1_col1, fila1_col2, fila1_col3 = st.columns(3)
            with fila1_col1:
                st.metric("TIR mensual", f"{tir_cf_mensual * 100:.2f} %")
            with fila1_col2:
                st.metric("TIR anual", f"{tir_cf_anual * 100:.2f} %")
            with fila1_col3:
                st.metric("VAN", formato_pesos(van_descontado))
            st.caption(
                f"Tasa desc. VAN: {tasa_descuento_pct:.2f} % mensual. "
                f"Se considera como valor residual el {valor_residual_cartera_pct:.0f}% "
                f"del saldo de cartera a cobrar al cierre."
            )

            # Fila 2: capital restante y meses clave
            fila2_col1, fila2_col2, fila2_col3 = st.columns(3)
            with fila2_col1:
                st.metric("Capital restante para breakeven", formato_pesos(capital_restante_breakeven))
            with fila2_col2:
                st.metric("Mes de breakeven (flujo neto)", mes_breakeven_texto)
            with fila2_col3:
                st.metric("Mes de recupero", mes_recupero_texto)

            # Fila 3: flujo total, ingresos y egresos
            fila3_col1, fila3_col2, fila3_col3 = st.columns(3)
            with fila3_col1:
                st.metric("Flujo total (periodo)", formato_pesos(flujo_total))
            with fila3_col2:
                st.metric("Ingresos totales (cobranzas)", formato_pesos(ingresos_totales))
            with fila3_col3:
                st.metric("Egresos totales", formato_pesos(egresos_totales))

            # Fila 4: saldo deuda final, total operaciones y mes de saneamiento
            fila4_col1, fila4_col2, fila4_col3 = st.columns(3)
            with fila4_col1:
                if deuda_inicial_financ > 0:
                    st.metric(
                        "Saldo deuda al cierre del horizonte",
                        formato_pesos(saldo_deuda_mes[-1]) if saldo_deuda_mes else formato_pesos(0),
                    )
                else:
                    st.metric("Saldo deuda al cierre del horizonte", formato_pesos(0))
            with fila4_col2:
                st.metric(
                    "Total operaciones nuevas (equivalentes)",
                    f"{total_operaciones_nuevas:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."),
                )
            with fila4_col3:
                st.metric("Mes de saneamiento (deuda en 0)", mes_saneamiento_texto)

            # Fila 5: saldo de cartera y valor residual al cierre
            fila5_col1, fila5_col2, fila5_col3 = st.columns(3)
            with fila5_col1:
                st.metric("Saldo cartera a cobrar (fin horizonte)", formato_pesos(saldo_cartera_final))
            with fila5_col2:
                st.metric("Valor residual de cartera (incluido en TIR/VAN)", formato_pesos(valor_residual_cartera))
            # fila5_col3 queda libre por ahora

            df_cf = pd.DataFrame(
                {
                    "Mes": etiquetas_meses,
                    "Fecha": fechas_cf,
                    "nueva originacion": cobranza_cuotas_nueva_mes,
                    "cartera existente": cobranza_existente_mes,
                    "Cobranza total (nueva + existente)": cobranza_total_mes,
                    "Capital colocado": capital_colocado_mes,
                    "Operaciones nuevas": operaciones_mes,
                    "Intereses cobrados": intereses_cobrados_mes,
                    "Comisiones comerciales": comisiones_mes,
                    "Gastos fijos": gastos_fijos_mes,
                    "IVA intereses": iva_intereses_mes,
                    "IVA gastos": iva_gastos_mes,
                    "IVA pagado (cartera exist.)": iva_pagado_existente_mes,
                    "Seguros pagados (cartera exist.)": seguros_pagados_mes,
                    "Imp. débito/crédito": imp_debcred_mes,
                    "IIBB": iibb_mes,
                    "Egresos totales": egresos_totales_mes,
                    "Flujo neto": flujo_neto_mes,
                    "Flujo acumulado": flujo_acumulado_mes,
                    "Flujo neto (después financ.)": flujo_neto_despues_financ_mes,
                    "Resultado acumulado después de deuda": flujo_acumulado_despues_financ_mes,
                    "Intereses financiamiento": intereses_financ_mes,
                    "Pago a deuda (financ.)": pago_deuda_mes,
                    "Saldo deuda (fin de mes)": saldo_deuda_mes,
                }
            )

            st.subheader("Resumen mensual")
            df_resumen = pd.DataFrame(
                {
                    "Mes": etiquetas_meses,
                    "nueva originacion": cobranza_cuotas_nueva_mes,
                    "cartera existente": cobranza_existente_mes,
                    "Total cobranza": cobranza_total_mes,
                    "Capital colocado": capital_colocado_mes,
                    "Flujo neto": flujo_neto_mes,
                    "Flujo acumulado": flujo_acumulado_mes,
                }
            )
            st.dataframe(
                df_resumen.style.format(
                    {
                        "nueva originacion": formato_pesos,
                        "cartera existente": formato_pesos,
                        "Total cobranza": formato_pesos,
                        "Capital colocado": formato_pesos,
                        "Flujo neto": formato_pesos,
                        "Flujo acumulado": formato_pesos,
                    }
                )
            )
            st.caption(
                "**nueva originacion:** cuotas de la colocación simulada (a mes vencido; el mes 0 suele ser $0). "
                "**cartera existente:** archivo de vencimientos × % cobranza efectiva. **Total cobranza** = suma de ambas."
            )

            with st.expander("Detalle técnico (más columnas)", expanded=False):
                # Meses en columnas, cada fila = un concepto (sin Fecha: el mes ya está en el encabezado)
                _df_tec = (
                    df_cf.drop(columns=["Fecha"], errors="ignore")
                    .set_index("Mes")
                    .T.rename_axis("Detalle")
                    .reset_index()
                )
                _fmt_tec = {c: formato_pesos for c in _df_tec.columns if c != "Detalle"}
                st.dataframe(_df_tec.style.format(_fmt_tec))
                st.caption("Cada **fila** es un rubro; cada **columna** es un mes (salvo la primera: concepto).")

            with st.expander("Neto vs acumulado (financiamiento)", expanded=False):
                df_cruzado = pd.DataFrame(
                    {
                        "Mes": etiquetas_meses,
                        "Flujo neto": flujo_neto_mes,
                        "Flujo acumulado": flujo_acumulado_mes,
                        "Flujo neto (después financ.)": flujo_neto_despues_financ_mes,
                        "Resultado acumulado después de deuda": flujo_acumulado_despues_financ_mes,
                    }
                )
                st.dataframe(
                    df_cruzado.style.format(
                        {
                            "Flujo neto": formato_pesos,
                            "Flujo acumulado": formato_pesos,
                            "Flujo neto (después financ.)": formato_pesos,
                            "Resultado acumulado después de deuda": formato_pesos,
                        }
                    )
                )
                st.caption(
                    "Flujo acumulado = suma operativa (cobranzas − egresos). "
                    "Resultado acumulado después de deuda incluye deuda inicial e intereses del financiamiento."
                )

            columnas_meses = etiquetas_meses

            # Cuadro de evolución de cartera a cobrar (cartera nueva)
            with st.expander("Evolución cartera nueva (colocación, pagaré, caída, saldo)", expanded=False):
                st.caption(
                    "**Pagaré del mes** = operaciones equivalentes del mes × suma de cuotas de un crédito al **crédito promedio** "
                    "(perfil del unitario escalado). **Caída cobrada** = caja a mes vencido (mes 0 suele ser 0). "
                    "**Intereses devengados** incluye el 1er mes de la colocación del mes (no es lo mismo que intereses cobrados)."
                )
                df_cartera = pd.DataFrame(
                    index=[
                        "Colocación (capital colocado)",
                        "Intereses devengados (cartera nueva)",
                        "Pagaré generado en el mes (nueva colocación)",
                        "Caída cobrada (nueva cartera, caja)",
                        "Saldo cartera nueva a cobrar (fin de mes)",
                    ],
                    columns=columnas_meses,
                )

                for t in meses:
                    col = columnas_meses[t]
                    df_cartera.at["Colocación (capital colocado)", col] = capital_colocado_mes[t]
                    df_cartera.at["Intereses devengados (cartera nueva)", col] = intereses_devengados_nueva_mes[t]
                    df_cartera.at["Pagaré generado en el mes (nueva colocación)", col] = pagare_generado_mes[t]
                    df_cartera.at["Caída cobrada (nueva cartera, caja)", col] = caida_cartera_cobrada_mes[t]
                    df_cartera.at["Saldo cartera nueva a cobrar (fin de mes)", col] = saldo_cartera_mes[t]

                st.dataframe(df_cartera.style.format(formato_pesos))

            st.subheader("Gráfico de flujo de caja: neto vs acumulado")
            # Usar Fecha como índice para que el eje X quede en orden cronológico (no alfabético)
            df_chart = df_cf.copy()
            df_chart["Fecha"] = pd.to_datetime(df_chart["Fecha"])
            df_chart = df_chart.set_index("Fecha")[
                ["Flujo neto", "Flujo acumulado", "Resultado acumulado después de deuda"]
            ]
            st.line_chart(df_chart)
            st.caption("Flujo acumulado = operativo. Resultado acumulado después de deuda = operativo menos intereses y considerando la deuda inicial.")

    with tab_cartera:
        st.header("Caída de cartera existente (vencimientos)")
        st.warning(
            "El **nombre del CSV**, la **subida opcional** y la **fecha de inicio** están en la solapa **CSV y mes de inicio** (primera)."
        )
        _fi_tab = st.session_state.get("fecha_inicio_cf", date(2026, 4, 1))
        if not isinstance(_fi_tab, date):
            _fi_tab = date(2026, 4, 1)
        _fi_label = f"{MESES_ES[_fi_tab.month - 1]} {_fi_tab.year}"
        st.info(
            f"**Mes de corte:** **{_fi_label}** (lo cambiás en la solapa **CSV y mes de inicio**)."
        )
        ag_v, _origen_tab_v = cargar_dataframe_vencimientos(
            archivo_venc_upload, ruta_venc_default, fecha_desde=_fi_tab
        )
        ag_v = _filtrar_agrupado_desde_mes(ag_v, _fi_tab)
        st.caption(
            f"Meses mostrados: **{_fi_label}** en adelante. Fuente: {_origen_tab_v}"
        )
        if ag_v is None:
            st.warning(
                "No hay datos: revisá la sección **Archivo de cartera** arriba (nombre del .csv y ruta), "
                "separador `;`, columnas Fecha_Vto, Capital, Interes_Ordinario, etc."
            )
        else:
            # Ordenamos por año y mes y armamos una fecha de referencia (primer día de cada mes)
            ag_v = ag_v.sort_values(["año", "mes"]).copy()
            # Nombres de mes en español (strftime %b depende del locale y a veces muestra «Mar» en inglés).
            ag_v["Mes"] = [
                f"{MESES_ES[int(m) - 1]} {int(y)}" for y, m in zip(ag_v["año"], ag_v["mes"])
            ]
            cols_monetarias = [
                "Capital",
                "Interes_Ordinario",
                "IVA_Interes_Ordinario",
                "Seguro",
                "Gastos",
                "Interes_Gastos",
                "IVA_Interes_Gastos",
                "Cobranza_Total",
                "IVA_Total",
                "Seguro_Total",
            ]
            st.subheader("Caída mensual (resumen por mes)")
            st.dataframe(
                ag_v[
                    ["Mes", "Capital", "Interes_Ordinario", "IVA_Interes_Ordinario", "Seguro", "Gastos"]
                    + ["Interes_Gastos", "IVA_Interes_Gastos", "Cobranza_Total", "IVA_Total", "Seguro_Total"]
                ].style.format({col: formato_pesos for col in cols_monetarias})
            )

    guardar_inputs_actuales(ruta_prefs_inputs)


if __name__ == "__main__":
    main()

