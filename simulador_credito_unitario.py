import math
import os
import tempfile
from datetime import date
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Meses en español para etiquetas del cashflow
MESES_ES = [
    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre",
]
import streamlit as st


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


def leer_vencimientos_real(ruta_csv: str) -> Optional[pd.DataFrame]:
    """
    Lee el archivo de vencimientos (ej. 'Vencimientos desde Marzo2026.csv').
    Agrupa por mes de vencimiento y suma Capital, Interés, IVA, Seguro, Gastos (sin punitorios ni compensatorios).
    Devuelve DataFrame con index = (año, mes) y columnas:
    Capital, Interes_Ordinario, IVA_Interes, IVA_Gastos, Seguro, Gastos, Interes_Gastos,
    Cobranza_Total, IVA_Total (para pago al mes sig.), Seguro_Total (para pago al mes sig.).
    """
    if not ruta_csv or not os.path.isfile(ruta_csv):
        return None
    try:
        df = pd.read_csv(
            ruta_csv,
            sep=";",
            encoding="latin-1",
            usecols=[
                "Fecha_Vto",
                "Capital",
                "Interes_Ordinario",
                "IVA_Interes_Ordinario",
                "Seguro",
                "Gastos",
                "Interes_Gastos",
                "IVA_Interes_Gastos",
            ],
            dtype=str,
            on_bad_lines="skip",
        )
    except Exception:
        return None
    if df.empty:
        return None
    # Parsear fechas (ej. "01/03/2026 12:00:00 a.m.")
    df["Fecha_Vto"] = pd.to_datetime(df["Fecha_Vto"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Fecha_Vto"])
    # Solo desde marzo 2026
    df = df[df["Fecha_Vto"] >= "2026-03-01"]
    if df.empty:
        return None
    # Parsear números (coma decimal)
    for col in ["Capital", "Interes_Ordinario", "IVA_Interes_Ordinario", "Seguro", "Gastos", "Interes_Gastos", "IVA_Interes_Gastos"]:
        if col in df.columns:
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


def main() -> None:
    st.set_page_config(page_title="Simulador prendario", layout="wide")
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

    tab_unitario, tab_cashflow, tab_cartera = st.tabs(["Crédito unitario", "Cashflow", "Cartera existente"])

    with tab_unitario:
        st.header("Simulación de crédito unitario")

        st.subheader("Parámetros del crédito")
        p1, p2 = st.columns(2)
        with p1:
            fecha_otorgamiento = st.date_input(
                "Fecha de otorgamiento del crédito",
                value=date.today(),
            )
            c1a, c1b = st.columns([2, 1])
            with c1a:
                capital_solicitado = st.number_input(
                    "Capital promedio solicitado ($)",
                    min_value=0.0,
                    value=10_000_000.0,
                    step=100_000.0,
                    format="%.0f",
                )
            with c1b:
                st.markdown(f"**{formato_pesos(capital_solicitado)}**")
            cuotas = st.number_input(
                "Cantidad de cuotas mensuales",
                min_value=1,
                max_value=120,
                value=18,
                step=1,
            )
            c2a, c2b = st.columns([2, 1])
            with c2a:
                tasa_anual_pct = st.number_input(
                    "Tasa anual TNA (%)",
                    min_value=0.0,
                    max_value=300.0,
                    value=78.0,
                    step=1.0,
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
        st.caption("Usa los parámetros del crédito unitario (solapa anterior). Tasa: " + f"{tasa_anual * 100:.2f} % anual.")

        st.subheader("Parámetros de colocación")
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        with r1c1:
            fecha_inicio_cf = st.date_input("Inicio simulación (mes 0)", value=date(2026, 3, 1))
        with r1c2:
            horizonte_meses = st.number_input("Horizonte (meses)", min_value=1, max_value=120, value=48, step=1)
        with r1c3:
            credito_promedio = st.number_input(
                "Crédito promedio ($)",
                min_value=1_000_000.0,
                value=capital_solicitado if capital_solicitado > 0 else 10_000_000.0,
                step=100_000.0,
                format="%.0f",
            )
            st.caption(formato_pesos(credito_promedio))
        with r1c4:
            capital_colocado_inicial = st.number_input(
                "Capital colocado inicial mes 0 ($)",
                min_value=0.0,
                value=250_000_000.0,
                step=10_000_000.0,
                format="%.0f",
            )
            st.caption(formato_pesos(capital_colocado_inicial))

        r1b1, r1b2 = st.columns(2)
        with r1b1:
            crecimiento_colocacion_mensual = st.number_input(
                "Crecimiento colocación mensual ($)",
                min_value=0.0,
                value=50_000_000.0,
                step=5_000_000.0,
                format="%.0f",
            )
            st.caption(f"Incremento lineal por mes: {formato_pesos(crecimiento_colocacion_mensual)}")
        with r1b2:
            capital_colocado_objetivo = st.number_input(
                "Capital colocado objetivo/mes ($)",
                min_value=0.0,
                value=250_000_000.0,
                step=10_000_000.0,
                format="%.0f",
            )
            st.caption("Tope de colocación mensual (se alcanza y luego se mantiene)")

        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        with r2c1:
            comision_comercial_pct = st.number_input(
                "Comisión colocación (%)",
                min_value=0.0,
                max_value=100.0,
                value=7.0,
                step=1.0,
            )
        with r2c2:
            gastos_fijos_mensuales = st.number_input(
                "Gastos fijos/mes ($)",
                min_value=0.0,
                value=20_000_000.0,
                step=1_000_000.0,
                format="%.0f",
            )
            st.caption(formato_pesos(gastos_fijos_mensuales))
        with r2c3:
            iibb_pct = st.number_input(
                "IIBB sobre intereses (%)",
                min_value=0.0,
                max_value=100.0,
                value=5.0,
                step=1.0,
            )
        with r2c4:
            incobrabilidad_pct = st.number_input(
                "Incobrabilidad (%)",
                min_value=0.0,
                max_value=90.0,
                value=6.0,
                step=1.0,
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
                )
                st.caption(formato_pesos(deuda_inicial_financ))
            with f2:
                tna_financ_pct = st.number_input(
                    "TNA financiamiento (%)",
                    min_value=0.0,
                    max_value=200.0,
                    value=34.0,
                    step=1.0,
                )
            st.caption("Si flujo < 0 se toma deuda; si > 0 se abona. Intereses: (TNA/100)/12 sobre saldo.")

        # Cartera existente: se lee de un CSV incluido en el proyecto, si está presente
        ruta_venc_default = os.path.join(os.path.dirname(__file__), "Vencimientos desde Marzo2026.csv")
        ag_vencimientos = leer_vencimientos_real(ruta_venc_default)
        c1, c2 = st.columns([1, 1])
        with c1:
            pct_cobranza_existente_pct = st.number_input(
                "% cobranza efectiva cartera existente",
                min_value=50.0,
                max_value=100.0,
                value=90.0,
                step=1.0,
            )
        with c2:
            if ag_vencimientos is not None:
                st.caption(f"Cartera existente cargada desde archivo interno ({len(ag_vencimientos)} meses).")
            else:
                st.caption("No se encontró el archivo interno de vencimientos; la cartera existente no se incluye.")

        if st.button("Calcular cashflow de colocación mensual"):
            meses = list(range(int(horizonte_meses)))

            # Convertimos porcentajes de entrada (0-100) a proporciones (0-1)
            comision_comercial_prop = comision_comercial_pct / 100.0
            iibb_prop = iibb_pct / 100.0
            incobrabilidad_prop = incobrabilidad_pct / 100.0
            tna_financ = tna_financ_pct / 100.0
            pct_cobranza_existente = pct_cobranza_existente_pct / 100.0

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

            # Extraemos de df unitario los arrays por mes de vida (1..cuotas)
            interes_por_vida = df["Interés"].tolist()
            cuota_total_por_vida = df["Cuota total"].tolist()
            # Gastos por cuota (igual en todas) para IVA sobre gastos cobrados
            gasto_cuota_promedio = credito_promedio * (1 + quebranto_pct) * gasto_admin_pct

            # Flujos por mes calendario
            fechas_cf = []
            capital_colocado_mes = []
            operaciones_mes = []
            cobranza_cuotas_mes = []
            cobranza_existente_mes = []
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
            intereses_generados_mes = []
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
                ops_mes = int(round(n_creditos_mes_t)) if n_creditos_mes_t > 0 else 0
                operaciones_mes.append(ops_mes)

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

                # Cartera a cobrar (solo cartera nueva):
                # - Pagaré generado: capital + intereses + gastos futuros de la nueva colocación
                #   aproximado como (suma de todas las cuotas) * número de créditos del mes.
                pagare_por_credito = sum(cuota_total_por_vida)
                pagare_gen_t = n_creditos_mes_t * pagare_por_credito
                pagare_generado_mes.append(pagare_gen_t)
                intereses_generados_mes.append(intereses_brutos_t)
                caida_cartera_cobrada_mes.append(cobranza_neta_t)
                saldo_cartera += pagare_gen_t - cobranza_neta_t
                saldo_cartera_mes.append(saldo_cartera)

                # Cobranza total = cartera nueva + cartera existente (real)
                cobranza_total_t = cobranza_neta_t + cobranza_existente_t
                cobranza_cuotas_mes.append(cobranza_total_t)
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

            # Indicadores financieros del cashflow
            tir_cf_mensual = calcular_irr(flujo_neto_mes)
            tir_cf_anual = (1 + tir_cf_mensual) ** 12 - 1 if tir_cf_mensual != 0 else 0.0
            flujo_total = flujo_acumulado_mes[-1] if flujo_acumulado_mes else 0.0
            van_cero = calcular_van(flujo_neto_mes, 0.0)
            # VAN descontado a la tasa elegida en los parámetros
            tasa_descuento_mensual = tasa_descuento_pct / 100.0
            van_descontado = calcular_van(flujo_neto_mes, tasa_descuento_mensual)
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
            ingresos_totales = sum(cobranza_cuotas_mes)
            egresos_totales = sum(egresos_totales_mes)
            total_operaciones_nuevas = int(sum(operaciones_mes))

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
            st.caption(f"Tasa desc. VAN: {tasa_descuento_pct:.2f} % mensual")

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
                st.metric("Total operaciones nuevas", f"{total_operaciones_nuevas:,}".replace(",", "."))
            with fila4_col3:
                st.metric("Mes de saneamiento (deuda en 0)", mes_saneamiento_texto)

            df_cf = pd.DataFrame(
                {
                    "Mes": etiquetas_meses,
                    "Fecha": fechas_cf,
                    "Capital colocado": capital_colocado_mes,
                    "Operaciones nuevas": operaciones_mes,
                    "Cobranza cuotas": cobranza_cuotas_mes,
                    "Cobranza cartera existente": cobranza_existente_mes,
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

            # Tabla cruzada: flujo neto vs acumulado por mes (operativo y después de financiamiento)
            st.subheader("Flujo de caja: neto vs acumulado por mes")
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
            st.caption("Flujo acumulado = suma operativa (cobranzas − egresos). Resultado acumulado después de deuda = mismo menos intereses de la deuda e incluyendo la deuda inicial; refleja el efecto del financiamiento en el negocio.")

            st.subheader("Flujo de caja mensual de la colocación simulada (detalle)")
            fmt_cf = {
                "Capital colocado": formato_pesos,
                "Cobranza cuotas": formato_pesos,
                "Cobranza cartera existente": formato_pesos,
                "Intereses cobrados": formato_pesos,
                "Comisiones comerciales": formato_pesos,
                "Gastos fijos": formato_pesos,
                "IVA intereses": formato_pesos,
                "IVA gastos": formato_pesos,
                "IVA pagado (cartera exist.)": formato_pesos,
                "Seguros pagados (cartera exist.)": formato_pesos,
                "Imp. débito/crédito": formato_pesos,
                "IIBB": formato_pesos,
                "Egresos totales": formato_pesos,
                "Flujo neto": formato_pesos,
                "Flujo acumulado": formato_pesos,
                "Flujo neto (después financ.)": formato_pesos,
                "Resultado acumulado después de deuda": formato_pesos,
                "Intereses financiamiento": formato_pesos,
                "Pago a deuda (financ.)": formato_pesos,
                "Saldo deuda (fin de mes)": formato_pesos,
            }
            st.dataframe(df_cf.style.format(fmt_cf))

            # Vista horizontal: meses en columnas, filas separando ingresos y egresos
            st.subheader("Flujo de caja (meses en columnas, ingresos y egresos separados)")
            columnas_meses = etiquetas_meses

            df_horizontal = pd.DataFrame(
                index=[
                    "INGRESOS - Cobranza cuotas (total)",
                    "INGRESOS - Cobranza cartera existente",
                    "INGRESOS - Operaciones nuevas (cantidad)",
                    "EGRESOS - Capital colocado",
                    "EGRESOS - Comisiones comerciales",
                    "EGRESOS - Gastos fijos",
                    "EGRESOS - IVA intereses",
                    "EGRESOS - IVA gastos",
                    "EGRESOS - IVA pagado (cartera exist.)",
                    "EGRESOS - Seguros pagados (cartera exist.)",
                    "EGRESOS - Imp. débito/crédito",
                    "EGRESOS - IIBB",
                    "FINANCIAMIENTO - Intereses",
                    "FINANCIAMIENTO - Pago a deuda",
                    "FINANCIAMIENTO - Saldo deuda (fin mes)",
                    "Flujo neto",
                    "Flujo acumulado",
                    "Resultado acumulado después de deuda",
                ],
                columns=columnas_meses,
            )

            for t in meses:
                col = columnas_meses[t]
                df_horizontal.at["INGRESOS - Cobranza cuotas (total)", col] = cobranza_cuotas_mes[t]
                df_horizontal.at["INGRESOS - Cobranza cartera existente", col] = cobranza_existente_mes[t]
                df_horizontal.at["INGRESOS - Operaciones nuevas (cantidad)", col] = operaciones_mes[t]
                df_horizontal.at["EGRESOS - Capital colocado", col] = capital_colocado_mes[t]
                df_horizontal.at["EGRESOS - Comisiones comerciales", col] = comisiones_mes[t]
                df_horizontal.at["EGRESOS - Gastos fijos", col] = gastos_fijos_mes[t]
                df_horizontal.at["EGRESOS - IVA intereses", col] = iva_intereses_mes[t]
                df_horizontal.at["EGRESOS - IVA gastos", col] = iva_gastos_mes[t]
                df_horizontal.at["EGRESOS - IVA pagado (cartera exist.)", col] = iva_pagado_existente_mes[t]
                df_horizontal.at["EGRESOS - Seguros pagados (cartera exist.)", col] = seguros_pagados_mes[t]
                df_horizontal.at["EGRESOS - Imp. débito/crédito", col] = imp_debcred_mes[t]
                df_horizontal.at["EGRESOS - IIBB", col] = iibb_mes[t]
                df_horizontal.at["FINANCIAMIENTO - Intereses", col] = intereses_financ_mes[t]
                df_horizontal.at["FINANCIAMIENTO - Pago a deuda", col] = pago_deuda_mes[t]
                df_horizontal.at["FINANCIAMIENTO - Saldo deuda (fin mes)", col] = saldo_deuda_mes[t]
                df_horizontal.at["Flujo neto", col] = flujo_neto_mes[t]
                df_horizontal.at["Flujo acumulado", col] = flujo_acumulado_mes[t]
                df_horizontal.at["Resultado acumulado después de deuda", col] = flujo_acumulado_despues_financ_mes[t]

            st.dataframe(df_horizontal.style.format(formato_pesos))

            # Cuadro de evolución de cartera a cobrar (cartera nueva)
            st.subheader("Evolución de la cartera a cobrar (cartera nueva, meses en columnas)")
            df_cartera = pd.DataFrame(
                index=[
                    "Colocación (capital colocado)",
                    "Intereses generados (mes)",
                    "Pagaré generado (nuevas colocaciones)",
                    "Caída de cartera cobrada (nueva cartera)",
                    "Saldo cartera a cobrar (fin de mes)",
                ],
                columns=columnas_meses,
            )

            for t in meses:
                col = columnas_meses[t]
                df_cartera.at["Colocación (capital colocado)", col] = capital_colocado_mes[t]
                df_cartera.at["Intereses generados (mes)", col] = intereses_generados_mes[t]
                df_cartera.at["Pagaré generado (nuevas colocaciones)", col] = pagare_generado_mes[t]
                df_cartera.at["Caída de cartera cobrada (nueva cartera)", col] = caida_cartera_cobrada_mes[t]
                df_cartera.at["Saldo cartera a cobrar (fin de mes)", col] = saldo_cartera_mes[t]

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
        ruta_venc_default = os.path.join(os.path.dirname(__file__), "Vencimientos desde Marzo2026.csv")
        ag_v = leer_vencimientos_real(ruta_venc_default)
        if ag_v is None:
            st.warning(
                "No se encontró el archivo 'Vencimientos desde Marzo2026.csv' en la carpeta del simulador. "
                "Si querés ver la caída de la cartera existente, copialo a la misma carpeta del archivo .py y volvé a desplegar."
            )
        else:
            # Ordenamos por año y mes y armamos una fecha de referencia (primer día de cada mes)
            ag_v = ag_v.sort_values(["año", "mes"]).copy()
            ag_v["Mes"] = pd.to_datetime(
                dict(year=ag_v["año"], month=ag_v["mes"], day=1)
            ).dt.strftime("%b %Y")
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


if __name__ == "__main__":
    main()

