import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────────────────────
#  Configuración y estilo
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Refugio Huellitas · EDO con Heun y RK4",
    page_icon="🐾",
    layout="wide",
)

CYAN, PINK, LIME, VIOLET, AMBER = "#22d3ee", "#f472b6", "#a3e635", "#a78bfa", "#fbbf24"

st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(1200px 600px at 10% -10%, #2a1458 0%, transparent 60%),
                    radial-gradient(1000px 500px at 100% 0%, #0b3b52 0%, transparent 55%),
                    #0b0b1a;
    }
    h1.titulo {
        font-size: 2.6rem; font-weight: 800; margin-bottom: 0;
        background: linear-gradient(90deg, #22d3ee, #a78bfa, #f472b6);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    p.subtitulo { color: #b8b8d9; font-size: 1.05rem; margin-top: .2rem; }
    .card {
        background: rgba(255,255,255,.05); border: 1px solid rgba(167,139,250,.35);
        border-radius: 16px; padding: 1rem 1.2rem; margin-bottom: .8rem;
    }
    .card.cyan  { border-color: rgba(34,211,238,.5); }
    .card.pink  { border-color: rgba(244,114,182,.5); }
    .card.lime  { border-color: rgba(163,230,53,.5); }
    .card.amber { border-color: rgba(251,191,36,.5); }
    div[data-testid="stMetric"] {
        background: rgba(255,255,255,.05); border: 1px solid rgba(34,211,238,.35);
        border-radius: 14px; padding: .7rem 1rem;
    }
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────
#  Modelo y métodos numéricos
# ──────────────────────────────────────────────────────────────
def f(t, y, a, k):
    """Lado derecho de la EDO: dy/dt = a - k*y."""
    return a - k * y


def exacta(t, y0, a, k):
    """Solución analítica: y(t) = a/k + (y0 - a/k) e^{-kt}."""
    ye = a / k
    return ye + (y0 - ye) * np.exp(-k * t)


def heun(y0, a, k, h, n):
    """Método de Heun (Runge-Kutta de orden 2)."""
    t = np.arange(n + 1) * h
    y = np.empty(n + 1)
    y[0] = y0
    for i in range(n):
        k1 = f(t[i], y[i], a, k)
        k2 = f(t[i] + h, y[i] + h * k1, a, k)
        y[i + 1] = y[i] + (h / 2) * (k1 + k2)
    return t, y


def rk4(y0, a, k, h, n):
    """Método clásico de Runge-Kutta de orden 4."""
    t = np.arange(n + 1) * h
    y = np.empty(n + 1)
    y[0] = y0
    for i in range(n):
        k1 = f(t[i], y[i], a, k)
        k2 = f(t[i] + h / 2, y[i] + h / 2 * k1, a, k)
        k3 = f(t[i] + h / 2, y[i] + h / 2 * k2, a, k)
        k4 = f(t[i] + h, y[i] + h * k3, a, k)
        y[i + 1] = y[i] + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return t, y


def simular(y0, a, k, h, T):
    n = int(round(T / h))
    t, yh = heun(y0, a, k, h, n)
    _, yr = rk4(y0, a, k, h, n)
    ye = exacta(t, y0, a, k)
    return t, ye, yh, yr


def error_rel(abs_err, exacto):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(exacto != 0, abs_err / np.abs(exacto) * 100, np.nan)


def cruce(t, y, umbral):
    """Primer instante en que la curva cruza el umbral (interpolación lineal)."""
    for i in range(len(y) - 1):
        if (y[i] - umbral) * (y[i + 1] - umbral) <= 0 and y[i] != y[i + 1]:
            return t[i] + (umbral - y[i]) * (t[i + 1] - t[i]) / (y[i + 1] - y[i])
    return None


def layout(fig, titulo, xt, yt, h=430):
    fig.update_layout(
        template="plotly_dark",
        title=titulo,
        xaxis_title=xt,
        yaxis_title=yt,
        height=h,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.03)",
        legend=dict(orientation="h", y=-0.22),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    return fig


# ──────────────────────────────────────────────────────────────
#  Barra lateral: entradas y validación
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎛️ Parámetros")
    st.caption("Valores **hipotéticos** para simular. Ajusta y la app recalcula todo.")

    y0 = st.number_input("Animales iniciales y₀ (animales)", 0.0, 100000.0, 120.0, 5.0)
    a = st.number_input("Ingreso semanal a (animales/semana)", 0.0, 10000.0, 10.0, 1.0)
    k = st.number_input("Tasa de adopción k (1/semana)", 0.001, 5.0, 0.20, 0.01, format="%.3f")
    C = st.number_input("Capacidad objetivo C (animales)", 0.0, 100000.0, 70.0, 5.0)

    st.markdown("---")
    T = st.number_input("Tiempo de simulación T (semanas)", 1.0, 520.0, 20.0, 1.0)
    h = st.number_input("Tamaño de paso h (semanas)", 0.01, 52.0, 2.0, 0.25, format="%.2f")
    h2 = st.number_input("Paso de comparación h₂ (semanas)", 0.01, 52.0, 1.0, 0.25, format="%.2f")

    st.markdown("---")
    st.caption("📐 Unidades: tiempo en **semanas**, población en **animales**.")

errores = []


def divide(paso):
    n = T / paso
    return abs(n - round(n)) < 1e-9 and round(n) >= 1


if h > T:
    errores.append("h no puede ser mayor que el tiempo de simulación T.")
if h2 > T:
    errores.append("h₂ no puede ser mayor que T.")
if not errores and not divide(h):
    errores.append(f"T/h = {T / h:.3f} no es entero. Elige un h que divida exactamente a T (ej. h = 2 con T = 20).")
if not errores and not divide(h2):
    errores.append(f"T/h₂ = {T / h2:.3f} no es entero. Elige un h₂ que divida exactamente a T.")
if not errores and (T / min(h, h2) > 20000):
    errores.append("Demasiados pasos (más de 20 000). Aumenta h o reduce T.")
if abs(h - h2) < 1e-12:
    errores.append("h y h₂ deben ser distintos para poder comparar dos tamaños de paso.")

if errores:
    for e in errores:
        st.error("⚠️ " + e)
    st.stop()

# ──────────────────────────────────────────────────────────────
#  Cálculos
# ──────────────────────────────────────────────────────────────
ystar = a / k
t, ye, yh, yr = simular(y0, a, k, h, T)
eh, er = np.abs(ye - yh), np.abs(ye - yr)
rh, rr = error_rel(eh, ye), error_rel(er, ye)

t2, ye2, yh2, yr2 = simular(y0, a, k, h2, T)
eh2, er2 = np.abs(ye2 - yh2), np.abs(ye2 - yr2)

# ──────────────────────────────────────────────────────────────
#  Encabezado
# ──────────────────────────────────────────────────────────────
st.markdown('<h1 class="titulo">🐾 Refugio Huellitas</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitulo">Modelando la población de un refugio de mascotas con una '
    "ecuación diferencial · Comparación de <b>Heun</b>, <b>Runge-Kutta 4</b> y la "
    "<b>solución analítica</b></p>",
    unsafe_allow_html=True,
)

tabs = st.tabs(
    [
        "🐶 Problema",
        "📐 Modelo y solución",
        "📈 Simulación",
        "🎯 Errores",
        "💬 Interpretación",
        "🧮 Métodos",
    ]
)

# ───────────── TAB 1: Problema ─────────────
with tabs[0]:
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown(
            """
            <div class="card cyan">
            <h3>El problema</h3>
            Un refugio comunitario de mascotas recibe animales rescatados todas las semanas
            (abandonados, extraviados, heridos) y los entrega en adopción. Si ingresan más
            animales de los que se adoptan, el refugio se <b>satura</b>: hay menos espacio,
            más gasto en alimento y veterinario, y peor bienestar para los animales.
            </div>
            <div class="card pink">
            <h3>Población beneficiaria</h3>
            Los animales alojados en el refugio y el equipo de voluntarios que los cuida;
            indirectamente, la comunidad que evita animales en situación de calle.
            </div>
            <div class="card lime">
            <h3>Pregunta del desafío</h3>
            ¿En cuántas semanas el refugio logra bajar a su <b>capacidad objetivo C</b>,
            y a qué cantidad de animales se estabiliza si el ritmo de ingreso y de adopción
            se mantiene?
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="card amber">
            <h3>⚠️ Aviso importante</h3>
            Todos los parámetros son <b>hipotéticos</b>. Esta simulación ayuda a entender el
            comportamiento del sistema, pero <b>no es una predicción comprobada</b> porque no
            se calibró con datos reales de un refugio.
            </div>
            <div class="card">
            <h3>Meta cuantificable</h3>
            Llegar a y(t) ≤ C animales y sostenerlo en el tiempo.
            </div>
            """,
            unsafe_allow_html=True,
        )

# ───────────── TAB 2: Modelo y solución analítica ─────────────
with tabs[1]:
    st.markdown("### Variables y parámetros")
    st.dataframe(
        pd.DataFrame(
            {
                "Símbolo": ["t", "y(t)", "a", "k", "y₀", "C"],
                "Significado": [
                    "Tiempo (variable independiente)",
                    "Animales en el refugio (variable dependiente)",
                    "Ingreso constante de animales",
                    "Tasa de adopción por animal",
                    "Animales en t = 0",
                    "Capacidad objetivo",
                ],
                "Unidad": ["semanas", "animales", "animales/semana", "1/semana", "animales", "animales"],
                "Valor actual": [f"0 a {T:g}", "—", f"{a:g}", f"{k:g}", f"{y0:g}", f"{C:g}"],
            }
        ),
        hide_index=True,
        width="stretch",
    )

    st.markdown("### La EDO y la condición inicial")
    st.latex(r"\frac{dy}{dt} = a - k\,y \qquad y(0) = y_0")
    st.markdown(
        """
        **¿Por qué esta ecuación?** La población cambia por dos efectos:
        - **Entran** `a` animales por semana (constante).
        - **Salen por adopción** `k·y` animales por semana: mientras más animales hay, más
          opciones tienen los adoptantes, así que las adopciones son proporcionales a `y`.

        **Supuestos:** ingreso constante, tasa de adopción constante, población tratada como
        variable continua y sin otras causas de salida (como traslados o fallecimientos).
        """
    )

    st.markdown("### Solución analítica (ecuación lineal de primer orden)")
    st.markdown("Reescribimos como `y' + k·y = a` y usamos factor integrante `μ(t) = e^{kt}`:")
    st.latex(r"\frac{d}{dt}\left(e^{kt}y\right) = a\,e^{kt}\;\Rightarrow\; e^{kt}y = \frac{a}{k}e^{kt} + K")
    st.latex(r"\boxed{\,y(t) = \frac{a}{k} + \left(y_0 - \frac{a}{k}\right)e^{-kt}\,}")
    st.markdown(f"Con tus valores: equilibrio **y\\* = a/k = {ystar:g} animales**.")

    st.markdown("### Verificación por sustitución")
    st.markdown("**1) Condición inicial:**")
    st.latex(r"y(0) = \frac{a}{k} + \left(y_0 - \frac{a}{k}\right)\cdot 1 = y_0 \;\checkmark")
    st.markdown("**2) La ecuación:** derivamos y sustituimos")
    st.latex(r"y'(t) = -k\left(y_0 - \tfrac{a}{k}\right)e^{-kt}")
    st.latex(
        r"a - k\,y(t) = a - k\cdot\frac{a}{k} - k\left(y_0-\tfrac{a}{k}\right)e^{-kt}"
        r"= -k\left(y_0-\tfrac{a}{k}\right)e^{-kt} = y'(t)\;\checkmark"
    )
    tt = np.linspace(0, T, 200)
    dy = -k * (y0 - ystar) * np.exp(-k * tt)
    residuo = np.max(np.abs(dy - f(tt, exacta(tt, y0, a, k), a, k)))
    st.success(f"Comprobación numérica con tus parámetros: máx |y′ − (a − k·y)| = {residuo:.2e} (≈ 0 ✅)")

    st.markdown("### Tiempo para alcanzar la capacidad objetivo")
    st.latex(r"y(t^*) = C \;\Rightarrow\; t^* = -\frac{1}{k}\,\ln\!\left(\frac{C - a/k}{y_0 - a/k}\right)")

# ───────────── TAB 3: Simulación ─────────────
with tabs[2]:
    st.markdown(f"### Resultados con h = {h:g} semanas ({len(t) - 1} pasos)")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("y(T) exacta", f"{ye[-1]:.3f}")
    m2.metric("y(T) Heun", f"{yh[-1]:.3f}", f"error {eh[-1]:.2e}", delta_color="off")
    m3.metric("y(T) RK4", f"{yr[-1]:.3f}", f"error {er[-1]:.2e}", delta_color="off")
    m4.metric("Equilibrio a/k", f"{ystar:.2f}")

    tf = np.linspace(0, T, 400)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tf, y=exacta(tf, y0, a, k), name="Analítica", line=dict(color=CYAN, width=3)))
    fig.add_trace(
        go.Scatter(x=t, y=yh, name="Heun", mode="lines+markers",
                   line=dict(color=PINK, dash="dash"), marker=dict(size=8, symbol="diamond"))
    )
    fig.add_trace(
        go.Scatter(x=t, y=yr, name="RK4", mode="lines+markers",
                   line=dict(color=LIME, dash="dot"), marker=dict(size=7))
    )
    fig.add_hline(y=C, line=dict(color=AMBER, dash="dash"), annotation_text=f"Capacidad C = {C:g}")
    fig.add_hline(y=ystar, line=dict(color=VIOLET, dash="dot"), annotation_text=f"Equilibrio = {ystar:g}",
                  annotation_position="bottom right")
    layout(fig, "Población del refugio en el tiempo", "Tiempo (semanas)", "Animales en el refugio")
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### Tabla comparativa por pasos")
    df = pd.DataFrame(
        {
            "Paso": np.arange(len(t)),
            "t (sem)": t,
            "Exacta": ye,
            "Heun": yh,
            "RK4": yr,
            "Error abs Heun": eh,
            "Error abs RK4": er,
            "Error rel Heun (%)": rh,
            "Error rel RK4 (%)": rr,
        }
    )
    st.dataframe(
        df.style.format(
            {
                "t (sem)": "{:.2f}", "Exacta": "{:.5f}", "Heun": "{:.5f}", "RK4": "{:.5f}",
                "Error abs Heun": "{:.3e}", "Error abs RK4": "{:.3e}",
                "Error rel Heun (%)": "{:.5f}", "Error rel RK4 (%)": "{:.5f}",
            },
            na_rep="—",
        ),
        hide_index=True,
        width="stretch",
        height=330,
    )
    st.download_button("⬇️ Descargar tabla (CSV)", df.to_csv(index=False).encode("utf-8"),
                       "comparacion_heun_rk4.csv", "text/csv")

# ───────────── TAB 4: Errores ─────────────
with tabs[3]:
    st.markdown("### Error frente a la solución analítica")
    cA, cB = st.columns(2)
    with cA:
        logy = st.checkbox("Escala logarítmica en el eje Y", value=False)
        f1 = go.Figure()
        f1.add_trace(go.Scatter(x=t, y=eh, name="Heun", line=dict(color=PINK, width=3), mode="lines+markers"))
        f1.add_trace(go.Scatter(x=t, y=er, name="RK4", line=dict(color=LIME, width=3), mode="lines+markers"))
        if logy:
            f1.update_yaxes(type="log")
        layout(f1, f"Error absoluto (h = {h:g})", "Tiempo (semanas)", "|exacta − aproximada| (animales)", 380)
        st.plotly_chart(f1, width="stretch")
    with cB:
        st.write("")
        st.write("")
        f2 = go.Figure()
        f2.add_trace(go.Scatter(x=t[1:], y=rh[1:], name="Heun", line=dict(color=PINK, width=3), mode="lines+markers"))
        f2.add_trace(go.Scatter(x=t[1:], y=rr[1:], name="RK4", line=dict(color=LIME, width=3), mode="lines+markers"))
        if logy:
            f2.update_yaxes(type="log")
        layout(f2, f"Error relativo porcentual (h = {h:g})", "Tiempo (semanas)", "Error relativo (%)", 380)
        st.plotly_chart(f2, width="stretch")

    st.markdown("### Comparación de dos tamaños de paso")
    filas = []
    for nombre, hh, e_h, e_r, tt_, ye_ in [("h", h, eh, er, t, ye), ("h₂", h2, eh2, er2, t2, ye2)]:
        filas.append(
            {
                "Paso": f"{nombre} = {hh:g}",
                "N.º de pasos": len(tt_) - 1,
                "Máx error Heun": e_h.max(),
                "Máx error RK4": e_r.max(),
                "Error final Heun": e_h[-1],
                "Error final RK4": e_r[-1],
                "Error rel final Heun (%)": e_h[-1] / abs(ye_[-1]) * 100 if ye_[-1] != 0 else np.nan,
                "Error rel final RK4 (%)": e_r[-1] / abs(ye_[-1]) * 100 if ye_[-1] != 0 else np.nan,
            }
        )
    dfc = pd.DataFrame(filas)
    st.dataframe(
        dfc.style.format({c: "{:.3e}" for c in dfc.columns[2:6]} | {c: "{:.5f}" for c in dfc.columns[6:]}, na_rep="—"),
        hide_index=True,
        width="stretch",
    )

    st.markdown("### ¿Qué tan rápido mejora el error al achicar h? (orden de convergencia)")
    hs = [max(h, h2) / 2**j for j in range(5)]
    hs = [x for x in hs if divide(x)]
    rows, errs_h, errs_r = [], [], []
    for x in hs:
        tx, yex, yhx, yrx = simular(y0, a, k, x, T)
        errs_h.append(abs(yex[-1] - yhx[-1]))
        errs_r.append(abs(yex[-1] - yrx[-1]))
    ordenes_h, ordenes_r = [np.nan], [np.nan]
    for i in range(1, len(hs)):
        ordenes_h.append(math.log(errs_h[i - 1] / errs_h[i], hs[i - 1] / hs[i]) if errs_h[i] > 1e-13 and errs_h[i - 1] > 0 else np.nan)
        ordenes_r.append(math.log(errs_r[i - 1] / errs_r[i], hs[i - 1] / hs[i]) if errs_r[i] > 1e-13 and errs_r[i - 1] > 0 else np.nan)
    dfo = pd.DataFrame({"h": hs, "Error final Heun": errs_h, "Orden Heun": ordenes_h,
                        "Error final RK4": errs_r, "Orden RK4": ordenes_r})
    f3 = go.Figure()
    f3.add_trace(go.Scatter(x=hs, y=np.maximum(errs_h, 1e-16), name="Heun", mode="lines+markers", line=dict(color=PINK, width=3)))
    f3.add_trace(go.Scatter(x=hs, y=np.maximum(errs_r, 1e-16), name="RK4", mode="lines+markers", line=dict(color=LIME, width=3)))
    f3.update_xaxes(type="log")
    f3.update_yaxes(type="log")
    layout(f3, "Error final vs h (escala log-log)", "h (semanas)", "Error absoluto final", 380)
    cX, cY = st.columns([1, 1])
    cX.plotly_chart(f3, width="stretch")
    with cY:
        st.dataframe(
            dfo.style.format({"h": "{:.4f}", "Error final Heun": "{:.3e}", "Error final RK4": "{:.3e}",
                              "Orden Heun": "{:.2f}", "Orden RK4": "{:.2f}"}, na_rep="—"),
            hide_index=True, width="stretch")
        st.caption("Teoría: Heun ≈ orden 2 (el error baja ~4× al dividir h entre 2) y "
                   "RK4 ≈ orden 4 (baja ~16×). Si el error llega a ~1e-13, la precisión de la computadora limita la medición.")

# ───────────── TAB 5: Interpretación ─────────────
with tabs[4]:
    st.markdown("### ¿Qué significan los resultados para la comunidad?")

    # tiempo para alcanzar C
    tstar = None
    razon = (C - ystar) / (y0 - ystar) if y0 != ystar else None
    if razon is not None and 0 < razon < 1:
        tstar = -math.log(razon) / k
    t_h = cruce(t, yh, C)
    t_r = cruce(t, yr, C)

    if y0 > ystar:
        tendencia = (f"El refugio está **por encima** de su equilibrio: ingresan {a:g} animales/semana, pero hay "
                     f"tantos animales que las adopciones superan los ingresos. La población **baja** de {y0:g} hacia {ystar:g}.")
    elif y0 < ystar:
        tendencia = (f"El refugio está **por debajo** de su equilibrio: los ingresos superan a las adopciones, "
                     f"así que la población **sube** de {y0:g} hacia {ystar:g}.")
    else:
        tendencia = f"El refugio parte justo en equilibrio ({ystar:g} animales): la población se mantiene constante."
    st.markdown(f'<div class="card cyan">{tendencia}</div>', unsafe_allow_html=True)

    if tstar is not None and tstar <= T:
        st.markdown(
            f'<div class="card lime"><b>Respuesta a la pregunta del desafío:</b> la población cruza la capacidad '
            f"C = {C:g} aproximadamente a las <b>{tstar:.2f} semanas</b> (valor exacto). "
            f"Con las aproximaciones numéricas: Heun ≈ {t_h:.2f} sem y RK4 ≈ {t_r:.2f} sem.</div>",
            unsafe_allow_html=True,
        )
    elif tstar is not None:
        st.markdown(
            f'<div class="card amber">El umbral C = {C:g} se cruzaría en t* = {tstar:.2f} semanas, que queda '
            f"<b>fuera</b> del tiempo simulado (T = {T:g}). Aumenta T para verlo.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="card amber">Con estos parámetros la población <b>nunca cruza</b> C = {C:g}: '
            f"se estabiliza en {ystar:g}. Para que el refugio baje a C hace falta que el equilibrio a/k quede por debajo "
            f"de C (más adopciones, mayor k, o menos ingresos, menor a).</div>",
            unsafe_allow_html=True,
        )

    if ystar > C:
        st.warning(f"A largo plazo el refugio se estabiliza en {ystar:g} animales, por encima de su capacidad "
                   f"objetivo ({C:g}). Se necesita **aumentar adopciones** o **reducir ingresos** (esterilización, "
                   f"campañas de tenencia responsable).")
    else:
        st.success(f"A largo plazo el refugio se estabiliza en {ystar:g} animales, dentro de su capacidad ({C:g}).")

    mejor = "RK4" if er.max() < eh.max() else "Heun"
    st.markdown(
        f"""
        <div class="card pink">
        <b>¿Qué método fue más preciso?</b><br>
        Con h = {h:g}: error máximo de Heun = <b>{eh.max():.2e}</b> animales y de RK4 = <b>{er.max():.2e}</b> animales.
        Resultó más preciso <b>{mejor}</b>. Al reducir el paso a h₂ = {h2:g}, los errores máximos pasan a
        {eh2.max():.2e} (Heun) y {er2.max():.2e} (RK4).
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Limitaciones del modelo")
    st.markdown(
        """
        - Los parámetros son **hipotéticos**; sin datos reales no es una predicción.
        - Ingreso y tasa de adopción se suponen **constantes**, aunque en la realidad hay campañas y temporadas.
        - Se trata a los animales como una cantidad continua (un valor como 70,4 animales es solo una aproximación).
        - No considera capacidad máxima física, traslados, ni diferencias entre perros y gatos.
        """
    )

# ───────────── TAB 6: Métodos ─────────────
with tabs[5]:
    st.markdown("### Cómo funcionan los dos métodos")
    st.markdown("Ambos avanzan en el tiempo con pasos de tamaño `h` partiendo de `y₀`, usando `f(t, y) = a − k·y`.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="card pink"><h4>Heun (RK de orden 2)</h4>'
                    "Promedia la pendiente al inicio del paso con la pendiente al final (estimada con Euler).</div>",
                    unsafe_allow_html=True)
        st.latex(r"k_1 = f(t_i, y_i)")
        st.latex(r"k_2 = f(t_i + h,\; y_i + h\,k_1)")
        st.latex(r"y_{i+1} = y_i + \tfrac{h}{2}(k_1 + k_2)")
        st.caption("Error global ≈ O(h²)")
    with c2:
        st.markdown('<div class="card lime"><h4>Runge-Kutta 4 (clásico)</h4>'
                    "Promedia cuatro pendientes (inicio, dos al medio y final), con más peso en las del medio.</div>",
                    unsafe_allow_html=True)
        st.latex(r"k_1 = f(t_i, y_i),\quad k_2 = f(t_i+\tfrac h2,\, y_i+\tfrac h2 k_1)")
        st.latex(r"k_3 = f(t_i+\tfrac h2,\, y_i+\tfrac h2 k_2),\quad k_4 = f(t_i+h,\, y_i+h k_3)")
        st.latex(r"y_{i+1} = y_i + \tfrac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)")
        st.caption("Error global ≈ O(h⁴)")

    with st.expander("🔍 Primer paso calculado a mano con tus parámetros"):
        k1 = f(0, y0, a, k)
        k2h = f(h, y0 + h * k1, a, k)
        y1h = y0 + h / 2 * (k1 + k2h)
        r1 = f(0, y0, a, k)
        r2 = f(h / 2, y0 + h / 2 * r1, a, k)
        r3 = f(h / 2, y0 + h / 2 * r2, a, k)
        r4 = f(h, y0 + h * r3, a, k)
        y1r = y0 + h / 6 * (r1 + 2 * r2 + 2 * r3 + r4)
        st.markdown(
            f"""
            **Heun:** k₁ = {k1:.5f}, k₂ = {k2h:.5f} → y₁ = {y0:g} + {h:g}/2·({k1:.5f} + {k2h:.5f}) = **{y1h:.5f}**

            **RK4:** k₁ = {r1:.5f}, k₂ = {r2:.5f}, k₃ = {r3:.5f}, k₄ = {r4:.5f} → y₁ = **{y1r:.5f}**

            **Exacta:** y(h) = **{exacta(h, y0, a, k):.5f}**
            """
        )

st.markdown("---")
st.caption("Desafío individual de Métodos Numéricos · Parámetros hipotéticos · Hecho con Streamlit 🐾")
