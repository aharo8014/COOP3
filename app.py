"""
COOPE Analytics — Dashboard de Créditos 2024
Version 4.0 · UI/UX Pro Max · Alexander Haro · 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.stats import chi2_contingency

# ══════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="COOPE Analytics 2024",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════
# DESIGN TOKENS
# ══════════════════════════════════════════════
BG        = "#0A0E1A"
SURFACE   = "#111827"
SURFACE2  = "#1F2937"
BORDER    = "rgba(255,255,255,0.09)"
PRIMARY   = "#F59E0B"
SECONDARY = "#8B5CF6"
ACCENT3   = "#10B981"
ACCENT4   = "#06B6D4"
DANGER    = "#EF4444"
TEXT      = "#F8FAFC"
MUTED     = "#94A3B8"
PALETTE   = [PRIMARY, SECONDARY, ACCENT3, ACCENT4, "#F97316", "#EC4899", "#FBBF24", "#A78BFA", "#34D399"]

# Orden correcto de monto para gráficos
MONTO_ORDER = [
    "Hasta mil dolares",
    "Mayor a 1 hasta 5 mil dolares",
    "Mayor a 5 hasta 10 mil dolares",
    "Mayor a 10 hasta 50 mil dolares",
    "Mayor a 50 hasta 100 mil dolares",
    "Mayor a 100 hasta 200 mil dolares",
    "Mayor a 200 mil dolares",
]

# ══════════════════════════════════════════════
# PLOTLY THEME HELPER
# ══════════════════════════════════════════════
def pt(fig, title="", h=None):
    kw = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(17,24,39,0.45)",
        font=dict(family="IBM Plex Sans, sans-serif", color=TEXT, size=12),
        title=dict(text=title, font=dict(size=14, color=TEXT), x=0.01, pad=dict(l=0,t=4)),
        colorway=PALETTE,
        xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor=BORDER,
                   tickcolor=BORDER, tickfont=dict(color=MUTED, size=11)),
        yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor=BORDER,
                   tickcolor=BORDER, tickfont=dict(color=MUTED, size=11)),
        legend=dict(bgcolor="rgba(17,24,39,0.7)", bordercolor=BORDER,
                    borderwidth=1, font=dict(color=MUTED, size=11)),
        margin=dict(l=10, r=10, t=46, b=10),
        hoverlabel=dict(bgcolor="#1e293b", bordercolor=BORDER,
                        font_color=TEXT, font_size=12),
    )
    if h:
        kw["height"] = h
    fig.update_layout(**kw)
    return fig


# ══════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════
def inject_css():
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {{
        font-family: 'IBM Plex Sans', sans-serif !important;
    }}
    .stApp {{
        background: {BG} !important;
    }}
    .main .block-container {{
        padding: 0 1.8rem 3rem 1.8rem !important;
        max-width: 100% !important;
    }}

    /* Sidebar */
    [data-testid="stSidebar"] > div:first-child {{
        background: {SURFACE} !important;
        border-right: 1px solid {BORDER};
        padding-top: 0 !important;
    }}
    [data-testid="stSidebarContent"] {{
        padding: 1rem 1rem 2rem 1rem !important;
    }}

    /* Tabs */
    [data-testid="stTabs"] [role="tablist"] {{
        background: {SURFACE} !important;
        border-radius: 12px !important;
        padding: 4px !important;
        border: 1px solid {BORDER} !important;
        gap: 2px !important;
        flex-wrap: wrap !important;
        overflow-x: auto !important;
    }}
    [data-testid="stTabs"] [role="tab"] {{
        color: {MUTED} !important;
        font-weight: 500 !important;
        font-size: 0.79rem !important;
        border-radius: 8px !important;
        padding: 6px 16px !important;
        transition: all 0.2s !important;
        border: none !important;
        background: transparent !important;
        white-space: nowrap !important;
    }}
    [data-testid="stTabs"] [role="tab"][aria-selected="true"] {{
        background: linear-gradient(135deg, {PRIMARY}, {SECONDARY}) !important;
        color: #fff !important;
        font-weight: 600 !important;
    }}
    [data-testid="stTabs"] [role="tabpanel"] {{
        padding-top: 1.2rem !important;
    }}

    /* Metrics */
    [data-testid="metric-container"] {{
        background: {SURFACE} !important;
        border: 1px solid {BORDER} !important;
        border-radius: 12px !important;
        padding: 1rem !important;
    }}

    /* Divider */
    hr {{
        border: none !important;
        border-top: 1px solid {BORDER} !important;
        margin: 1.4rem 0 !important;
    }}

    /* Hide defaults */
    #MainMenu, footer {{ visibility: hidden !important; }}
    [data-testid="stDeployButton"] {{ display: none !important; }}

    /* Scrollbar */
    ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
    ::-webkit-scrollbar-track {{ background: {BG}; }}
    ::-webkit-scrollbar-thumb {{ background: {SURFACE2}; border-radius: 3px; }}

    /* Download button */
    [data-testid="stDownloadButton"] button {{
        background: linear-gradient(135deg, {PRIMARY}, {SECONDARY}) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
    }}

    /* Expander */
    [data-testid="stExpander"] {{
        border: 1px solid {BORDER} !important;
        border-radius: 10px !important;
        background: {SURFACE} !important;
    }}

    /* Dataframe */
    [data-testid="stDataFrame"] {{
        border: 1px solid {BORDER} !important;
        border-radius: 10px !important;
        overflow: hidden !important;
    }}

    /* Button primary */
    [data-testid="stButton"] button {{
        border-radius: 8px !important;
        font-weight: 500 !important;
        font-size: 0.82rem !important;
        transition: opacity 0.2s !important;
    }}
    [data-testid="stButton"] button:hover {{
        opacity: 0.85 !important;
    }}
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# SPLASH SCREEN
# ══════════════════════════════════════════════
def render_splash():
    st.markdown(f"""
    <style>
    #spl {{
        position:fixed; top:0;left:0;right:0;bottom:0; z-index:999999;
        background:radial-gradient(ellipse at 50% 35%, #1a0630 0%, {BG} 55%, #000508 100%);
        display:flex; flex-direction:column; align-items:center; justify-content:center;
        overflow:hidden;
        animation: splOut 1s cubic-bezier(0.4,0,1,1) 5.2s forwards;
    }}
    @keyframes splOut {{
        0%   {{ opacity:1; transform:scale(1); }}
        100% {{ opacity:0; transform:scale(1.04); pointer-events:none; }}
    }}
    .sg {{ position:absolute;inset:0;
        background-image:linear-gradient(rgba(245,158,11,.05) 1px,transparent 1px),
                         linear-gradient(90deg,rgba(245,158,11,.05) 1px,transparent 1px);
        background-size:55px 55px;
        transform:perspective(500px) rotateX(12deg) scale(1.2);
        transform-origin:bottom; animation:gd 7s linear infinite; }}
    @keyframes gd {{ from{{background-position:0 0}} to{{background-position:0 55px}} }}

    .o {{ position:absolute;border-radius:50%;filter:blur(90px);animation:op 5s ease-in-out infinite alternate; }}
    .o1 {{ width:420px;height:420px;background:radial-gradient(circle,rgba(139,92,246,.28),transparent);top:-120px;right:-100px; }}
    .o2 {{ width:320px;height:320px;background:radial-gradient(circle,rgba(245,158,11,.22),transparent);bottom:-90px;left:-80px;animation-delay:1.5s; }}
    .o3 {{ width:200px;height:200px;background:radial-gradient(circle,rgba(16,185,129,.18),transparent);top:55%;left:48%;transform:translate(-50%,-50%);animation-delay:.8s; }}
    @keyframes op {{ from{{opacity:.4;transform:scale(1)}} to{{opacity:.9;transform:scale(1.18)}} }}

    .rw {{ position:relative;width:168px;height:168px;perspective:700px;margin-bottom:30px;
           animation:rv .8s cubic-bezier(.16,1,.3,1) .1s both; }}
    @keyframes rv {{ from{{opacity:0;transform:translateY(30px) scale(.8)}} to{{opacity:1;transform:none}} }}
    .rn {{ position:absolute;inset:0;border-radius:50%;border:2px solid transparent; }}
    .r1 {{ border-top-color:{PRIMARY};border-right-color:{PRIMARY}44;animation:rr 2.4s linear infinite; }}
    .r2 {{ inset:15px;border-top-color:{SECONDARY};border-left-color:{SECONDARY}44;animation:rr 3.4s linear infinite reverse; }}
    .r3 {{ inset:30px;border-top-color:{ACCENT3};border-right-color:{ACCENT3}44;animation:rr 4.2s linear infinite; }}
    @keyframes rr {{ from{{transform:rotateZ(0) rotateX(62deg)}} to{{transform:rotateZ(360deg) rotateX(62deg)}} }}
    .dm {{ position:absolute;top:50%;left:50%;width:36px;height:36px;margin:-18px 0 0 -18px;
           background:linear-gradient(135deg,{PRIMARY},{SECONDARY});transform:rotate(45deg);
           border-radius:5px;box-shadow:0 0 28px {PRIMARY}88,0 0 65px {SECONDARY}44;
           animation:dp 2.2s ease-in-out infinite; }}
    @keyframes dp {{
        0%,100%{{ box-shadow:0 0 20px {PRIMARY}77,0 0 45px {SECONDARY}33; }}
        50%    {{ box-shadow:0 0 42px {PRIMARY}cc,0 0 85px {SECONDARY}66; }}
    }}

    .tt {{ font-family:'IBM Plex Sans',sans-serif;font-size:clamp(1.9rem,5vw,3.1rem);
           font-weight:700;letter-spacing:-.02em;color:{TEXT};text-align:center;
           line-height:1.1;margin-bottom:5px;animation:tu .8s cubic-bezier(.16,1,.3,1) .4s both; }}
    .tt span {{ background:linear-gradient(135deg,{PRIMARY} 0%,#FBBF24 45%,{SECONDARY} 100%);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text; }}
    .sb {{ font-family:'IBM Plex Sans',sans-serif;font-size:clamp(.75rem,1.8vw,.9rem);
           font-weight:400;color:{MUTED};text-align:center;letter-spacing:.17em;
           text-transform:uppercase;margin-bottom:5px;animation:tu .8s cubic-bezier(.16,1,.3,1) .55s both; }}
    .au {{ font-family:'IBM Plex Sans',sans-serif;font-size:.82rem;font-weight:500;
           color:{PRIMARY}cc;text-align:center;letter-spacing:.08em;margin-bottom:36px;
           animation:tu .8s cubic-bezier(.16,1,.3,1) .65s both; }}
    @keyframes tu {{ from{{opacity:0;transform:translateY(18px)}} to{{opacity:1;transform:none}} }}

    .bw {{ width:250px;height:2px;background:rgba(255,255,255,.07);border-radius:2px;
           overflow:hidden;animation:tu .5s ease .7s both; }}
    .bf {{ height:100%;background:linear-gradient(90deg,{PRIMARY},{SECONDARY});border-radius:2px;
           animation:bfill 5s cubic-bezier(.4,0,.2,1) .4s forwards;width:0%; }}
    @keyframes bfill {{ from{{width:0%}} to{{width:100%}} }}

    .dots {{ display:flex;gap:6px;margin-top:14px;animation:tu .5s ease .85s both; }}
    .dots span {{ width:5px;height:5px;border-radius:50%;background:{PRIMARY};
                 animation:db 1.3s ease-in-out infinite; }}
    .dots span:nth-child(2){{animation-delay:.15s;background:{SECONDARY};}}
    .dots span:nth-child(3){{animation-delay:.30s;background:{ACCENT3};}}
    @keyframes db {{ 0%,80%,100%{{transform:scale(.6);opacity:.35}} 40%{{transform:scale(1);opacity:1}} }}

    .vr {{ position:absolute;bottom:24px;font-size:.67rem;
           color:rgba(148,163,184,.32);letter-spacing:.12em; }}

    /* Floating particles */
    .fp {{ position:absolute;width:3px;height:3px;border-radius:50%;
           background:{PRIMARY};opacity:0;animation:pfl linear infinite; }}
    @keyframes pfl {{
        0%  {{ opacity:0;transform:translateY(0) scale(0); }}
        8%  {{ opacity:.9;transform:translateY(-12px) scale(1); }}
        90% {{ opacity:.3; }}
        100%{{ opacity:0;transform:translateY(-170px) scale(.4); }}
    }}
    </style>

    <div id="spl">
      <div class="sg"></div>
      <div class="o o1"></div><div class="o o2"></div><div class="o o3"></div>
      <div class="fp" style="left:8%;top:78%;animation-duration:3.6s;animation-delay:.2s"></div>
      <div class="fp" style="left:18%;top:83%;animation-duration:4.1s;animation-delay:.9s;background:{SECONDARY}"></div>
      <div class="fp" style="left:33%;top:76%;animation-duration:3.2s;animation-delay:1.3s"></div>
      <div class="fp" style="left:52%;top:80%;animation-duration:4.6s;animation-delay:.5s;background:{ACCENT3}"></div>
      <div class="fp" style="left:67%;top:84%;animation-duration:3.9s;animation-delay:1.6s;background:{SECONDARY}"></div>
      <div class="fp" style="left:82%;top:77%;animation-duration:3.3s;animation-delay:.4s"></div>
      <div class="fp" style="left:44%;top:88%;animation-duration:4.3s;animation-delay:1.1s;background:#FBBF24"></div>
      <div class="rw">
        <div class="rn r1"></div><div class="rn r2"></div><div class="rn r3"></div>
        <div class="dm"></div>
      </div>
      <div class="tt">COOPE <span>Analytics</span></div>
      <div class="sb">Sistema de Análisis de Créditos · 2024</div>
      <div class="au">Elaborado por Alexander Haro &nbsp;·&nbsp; 2026</div>
      <div class="bw"><div class="bf"></div></div>
      <div class="dots"><span></span><span></span><span></span></div>
      <div class="vr">v4.0 &nbsp;·&nbsp; COOPE Risk Intelligence Platform &nbsp;·&nbsp; 2026</div>
    </div>
    <script>
      setTimeout(function(){{
        var e = document.getElementById('spl');
        if (e) e.remove();
      }}, 6400);
    </script>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# CARGA Y NORMALIZACIÓN DE DATOS
# ══════════════════════════════════════════════
_FIXES = {
    "CRÃ‰DITO": "CRÉDITO", "CRÉDITO": "CRÉDITO",
    "Ã‰": "É", "Ã©": "é", "Ã³": "ó", "Ã¡": "á",
    "Ã­": "í", "Ãº": "ú", "Ã±": "ñ",
}

def _fix(s):
    for b, g in _FIXES.items():
        s = s.replace(b, g)
    return s.strip()

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    df = pd.read_excel("2024.xlsx", sheet_name="2024", engine="openpyxl")
    df.columns = [_fix(c) for c in df.columns]
    # Seleccionar columnas disponibles
    want = ["SEGMENTO","PROVINCIA","CANTON","FECHA DE CORTE","INSTRUCCION",
            "SEXO","DESTINO FINANCIERO","RANGO EDAD",
            "RANGO MONTO CREDITO CONCEDIDO","RANGO PLAZO ORIGINAL CONCESION",
            "TIPO PERSONA","TIPO DE CRÉDITO"]
    df = df[[c for c in want if c in df.columns]].copy()
    # Strings
    str_cols = [c for c in df.columns if c != "FECHA DE CORTE"]
    for c in str_cols:
        df[c] = df[c].astype(str).str.strip().replace("nan","Sin dato")
    # Fechas
    if "FECHA DE CORTE" in df.columns:
        df["FECHA DE CORTE"] = pd.to_datetime(df["FECHA DE CORTE"], errors="coerce")
        df["MES"]     = df["FECHA DE CORTE"].dt.strftime("%b %Y")
        df["MES_ORD"] = df["FECHA DE CORTE"].dt.to_period("M").astype(str)
    return df


# ══════════════════════════════════════════════
# UI HELPERS
# ══════════════════════════════════════════════
def kpi_card(title, value, sub="", color=PRIMARY, delta=None):
    delta_html = ""
    if delta is not None:
        dc = ACCENT3 if delta >= 0 else DANGER
        arrow = "▲" if delta >= 0 else "▼"
        delta_html = f"<span style='color:{dc};font-size:.72rem;font-weight:600;'>{arrow} {abs(delta):.1f}%</span>"
    return f"""
    <div style="background:linear-gradient(145deg,{SURFACE},{SURFACE2});
        border:1px solid {BORDER};border-top:2.5px solid {color};
        border-radius:14px;padding:1.1rem 1.3rem;position:relative;overflow:hidden;height:100%;">
      <div style="position:absolute;top:0;right:0;width:60px;height:60px;
          background:radial-gradient(circle at top right,{color}18,transparent);
          border-radius:0 14px 0 60px;"></div>
      <div style="font-size:.67rem;font-weight:600;color:{MUTED};text-transform:uppercase;
          letter-spacing:.09em;margin-bottom:6px;">{title}</div>
      <div style="font-size:1.8rem;font-weight:700;color:{TEXT};line-height:1;
          margin-bottom:4px;">{value}</div>
      <div style="display:flex;align-items:center;gap:8px;">
        {"<div style='font-size:.72rem;color:"+MUTED+";'>"+sub+"</div>" if sub else ""}
        {delta_html}
      </div>
    </div>"""

def sec(title, sub=""):
    st.markdown(f"""
    <div style="margin-bottom:.85rem;margin-top:.3rem;">
      <h3 style="font-size:.97rem;font-weight:600;color:{TEXT};margin:0 0 3px 0;
          display:flex;align-items:center;gap:8px;">
        <span style="display:inline-block;width:3px;height:16px;flex-shrink:0;
            background:linear-gradient(180deg,{PRIMARY},{SECONDARY});
            border-radius:2px;"></span>{title}</h3>
      {"<p style='font-size:.76rem;color:"+MUTED+";margin:0 0 0 11px;'>"+sub+"</p>" if sub else ""}
    </div>""", unsafe_allow_html=True)

def insight(text, kind="info"):
    c = {"info":(f"{SECONDARY}1a",SECONDARY),"success":(f"{ACCENT3}1a",ACCENT3),
         "warning":(f"{PRIMARY}1a",PRIMARY),"danger":(f"{DANGER}1a",DANGER)}
    bg, bd = c.get(kind, c["info"])
    st.markdown(f"""
    <div style="background:{bg};border-left:3px solid {bd};border-radius:0 8px 8px 0;
        padding:.6rem 1rem;margin:.3rem 0 .85rem 0;font-size:.79rem;color:{MUTED};">{text}</div>
    """, unsafe_allow_html=True)

def empty_state(msg):
    st.markdown(f"""
    <div style="text-align:center;padding:3rem 2rem;border:1px dashed {BORDER};
        border-radius:14px;margin:1rem 0;">
      <div style="font-size:2rem;opacity:.25;margin-bottom:10px;">◈</div>
      <div style="color:{MUTED};font-size:.85rem;">{msg}</div>
    </div>""", unsafe_allow_html=True)

def dashboard_header():
    st.markdown(f"""
    <div style="padding:1.8rem 0 1.1rem 0;border-bottom:1px solid {BORDER};margin-bottom:1.2rem;">
      <div style="display:flex;align-items:center;gap:14px;margin-bottom:8px;">
        <div style="width:42px;height:42px;
            background:linear-gradient(135deg,{PRIMARY},{SECONDARY});
            border-radius:11px;display:flex;align-items:center;justify-content:center;
            box-shadow:0 0 22px {PRIMARY}44;flex-shrink:0;">
          <div style="width:17px;height:17px;background:#fff;
              transform:rotate(45deg);border-radius:3px;"></div>
        </div>
        <div>
          <div style="font-size:1.55rem;font-weight:700;color:{TEXT};
              letter-spacing:-.025em;line-height:1.1;">
            COOPE
            <span style="background:linear-gradient(135deg,{PRIMARY},{SECONDARY});
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                background-clip:text;">Analytics</span>
          </div>
          <div style="font-size:.71rem;color:{MUTED};letter-spacing:.09em;text-transform:uppercase;">
            Sistema de Análisis de Créditos 2024 &nbsp;·&nbsp; Alexander Haro &nbsp;·&nbsp; 2026
          </div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════
def build_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        # Header
        st.markdown(f"""
        <div style="background:linear-gradient(135deg,{PRIMARY}22,{SECONDARY}22);
            border-radius:10px;padding:.9rem 1rem;margin-bottom:1rem;
            border:1px solid {BORDER};">
          <div style="font-size:1rem;font-weight:700;color:{TEXT};">◈ COOPE Analytics</div>
          <div style="font-size:.65rem;color:{MUTED};text-transform:uppercase;
              letter-spacing:.07em;margin-top:2px;">Panel de filtros · 2024</div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"<div style='font-size:.72rem;font-weight:600;color:{PRIMARY};text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;'>📍 Geografía</div>", unsafe_allow_html=True)

        provs = sorted(df["PROVINCIA"].unique().tolist())
        sel_prov = st.selectbox("Provincia", ["Todas"] + provs)

        df_geo = df[df["PROVINCIA"] == sel_prov] if sel_prov != "Todas" else df
        cants = sorted(df_geo["CANTON"].unique().tolist())
        sel_cant = st.selectbox("Cantón", ["Todos"] + cants)

        st.divider()
        st.markdown(f"<div style='font-size:.72rem;font-weight:600;color:{SECONDARY};text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;'>👤 Perfil del solicitante</div>", unsafe_allow_html=True)

        segs   = sorted(df["SEGMENTO"].unique().tolist())
        sel_seg = st.selectbox("Segmento", ["Todos"] + segs)

        sexos   = sorted(df["SEXO"].unique().tolist())
        sel_sexo = st.selectbox("Sexo", ["Todos"] + sexos)

        instruc  = sorted(df["INSTRUCCION"].unique().tolist())
        sel_ins  = st.selectbox("Instrucción", ["Todas"] + instruc)

        edades   = sorted(df["RANGO EDAD"].unique().tolist())
        sel_edad = st.selectbox("Rango de edad", ["Todos"] + edades)

        st.divider()
        st.markdown(f"<div style='font-size:.72rem;font-weight:600;color:{ACCENT3};text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;'>💳 Producto crediticio</div>", unsafe_allow_html=True)

        tipos_col = "TIPO DE CRÉDITO"
        tipos = sorted(df[tipos_col].unique().tolist())
        sel_tipo = st.multiselect("Tipo de crédito", tipos, placeholder="Todos los tipos")

        dest_col = "DESTINO FINANCIERO"
        if dest_col in df.columns:
            dests = sorted(df[dest_col].unique().tolist())
            sel_dest = st.multiselect("Destino financiero", dests, placeholder="Todos")
        else:
            sel_dest = []

        st.divider()
        st.markdown(f"<div style='font-size:.72rem;font-weight:600;color:{ACCENT4};text-transform:uppercase;letter-spacing:.06em;margin-bottom:6px;'>📅 Período</div>", unsafe_allow_html=True)

        if "MES" in df.columns:
            meses = sorted(df["MES_ORD"].unique().tolist())
            mes_labels = {m: df[df["MES_ORD"]==m]["MES"].iloc[0] for m in meses}
            sel_mes = st.multiselect("Mes de corte",
                                     [mes_labels[m] for m in meses],
                                     placeholder="Todos los meses")
        else:
            sel_mes = []

        st.divider()

        # Stats rápidas
        total = len(df)
        st.markdown(f"""
        <div style="background:{SURFACE2};border:1px solid {BORDER};border-radius:8px;
            padding:.7rem .9rem;font-size:.72rem;color:{MUTED};">
          <div style="color:{TEXT};font-weight:600;margin-bottom:4px;">Dataset completo</div>
          <div>{total:,} registros &nbsp;·&nbsp; 6 meses</div>
          <div>24 provincias &nbsp;·&nbsp; 219 cantones</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
        if st.button("↺ Restablecer filtros", use_container_width=True):
            st.rerun()

    # ─── Aplicar filtros ───
    dff = df.copy()
    if sel_prov != "Todas":
        dff = dff[dff["PROVINCIA"] == sel_prov]
    if sel_cant != "Todos":
        dff = dff[dff["CANTON"] == sel_cant]
    if sel_seg != "Todos":
        dff = dff[dff["SEGMENTO"] == sel_seg]
    if sel_sexo != "Todos":
        dff = dff[dff["SEXO"] == sel_sexo]
    if sel_ins != "Todas":
        dff = dff[dff["INSTRUCCION"] == sel_ins]
    if sel_edad != "Todos":
        dff = dff[dff["RANGO EDAD"] == sel_edad]
    if sel_tipo:
        dff = dff[dff[tipos_col].isin(sel_tipo)]
    if sel_dest and dest_col in dff.columns:
        dff = dff[dff[dest_col].isin(sel_dest)]
    if sel_mes and "MES" in dff.columns:
        dff = dff[dff["MES"].isin(sel_mes)]
    return dff


# ══════════════════════════════════════════════
# KPI ROW
# ══════════════════════════════════════════════
def render_kpis(df: pd.DataFrame, df_full: pd.DataFrame):
    n = len(df)
    pct = n / len(df_full) * 100
    np_ = df["PROVINCIA"].nunique()
    nc  = df["CANTON"].nunique()
    tc  = df["TIPO DE CRÉDITO"] if "TIPO DE CRÉDITO" in df.columns else pd.Series(dtype=str)
    top_t   = tc.value_counts().idxmax()[:18] if len(df) else "—"
    top_t_p = tc.value_counts(normalize=True).max()*100 if len(df) else 0
    top_sx  = df["SEXO"].value_counts().idxmax() if len(df) else "—"
    ns      = df["SEGMENTO"].nunique()

    cards = [
        kpi_card("Registros filtrados", f"{n:,}", f"{pct:.1f}% del total", PRIMARY),
        kpi_card("Provincias",          str(np_), "en selección actual",  SECONDARY),
        kpi_card("Cantones",            str(nc),  "en selección actual",  ACCENT3),
        kpi_card("Segmentos",           str(ns),  "activos",              ACCENT4),
        kpi_card("Tipo crédito top",    top_t,    f"{top_t_p:.0f}% del total", "#F97316"),
        kpi_card("Sexo predominante",   top_sx,   "",                     "#EC4899"),
    ]
    cols = st.columns(6)
    for col, card in zip(cols, cards):
        with col:
            st.markdown(card, unsafe_allow_html=True)
    st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 1 — RESUMEN GENERAL
# ══════════════════════════════════════════════
def tab_resumen(df: pd.DataFrame):
    if df.empty:
        empty_state("Sin datos para la selección actual. Ajusta los filtros.")
        return

    c1, c2 = st.columns(2)
    with c1:
        sec("Créditos por Tipo", "Volumen de operaciones por categoría")
        tc = df["TIPO DE CRÉDITO"].value_counts().reset_index()
        tc.columns = ["Tipo","Cantidad"]
        fig = px.bar(tc.sort_values("Cantidad"), x="Cantidad", y="Tipo",
                     orientation="h", color="Tipo", color_discrete_sequence=PALETTE)
        fig = pt(fig, h=380)
        fig.update_layout(showlegend=False)
        fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x:,}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sec("Distribución por Segmento")
        sg = df["SEGMENTO"].value_counts().reset_index()
        sg.columns = ["Segmento","Cantidad"]
        fig2 = px.pie(sg, values="Cantidad", names="Segmento",
                      hole=0.58, color_discrete_sequence=PALETTE)
        fig2 = pt(fig2, h=380)
        fig2.update_traces(textposition="outside", textfont_size=11,
                           hovertemplate="<b>%{label}</b><br>%{value:,} (%{percent})<extra></extra>")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    with c3:
        sec("Top 12 Provincias", "Por volumen de créditos")
        pv = df["PROVINCIA"].value_counts().head(12).reset_index()
        pv.columns = ["Provincia","Cantidad"]
        fig3 = px.bar(pv.sort_values("Cantidad"), x="Cantidad", y="Provincia",
                      orientation="h", color="Cantidad",
                      color_continuous_scale=[[0,"rgba(139,92,246,.35)"],[1,PRIMARY]])
        fig3 = pt(fig3, h=380)
        fig3.update_coloraxes(showscale=False)
        fig3.update_traces(hovertemplate="<b>%{y}</b><br>%{x:,}<extra></extra>")
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        sec("Sexo × Tipo de Persona")
        sx = df.groupby(["SEXO","TIPO PERSONA"]).size().reset_index(name="Cantidad")
        fig4 = px.bar(sx, x="SEXO", y="Cantidad", color="TIPO PERSONA",
                      barmode="group", color_discrete_sequence=PALETTE)
        fig4 = pt(fig4, h=380)
        fig4.update_traces(hovertemplate="<b>%{x}</b> · %{fullData.name}<br>%{y:,}<extra></extra>")
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Treemap jerárquico
    sec("Mapa de Árbol — Jerarquía de Créditos",
        "Segmento → Tipo de Crédito → Rango de Monto")
    df_tm = df.groupby(["SEGMENTO","TIPO DE CRÉDITO","RANGO MONTO CREDITO CONCEDIDO"]).size().reset_index(name="Cantidad")
    fig5 = px.treemap(df_tm,
                      path=["SEGMENTO","TIPO DE CRÉDITO","RANGO MONTO CREDITO CONCEDIDO"],
                      values="Cantidad", color="Cantidad",
                      color_continuous_scale=[[0,SURFACE2],[0.4,"rgba(139,92,246,0.75)"],[1,PRIMARY]])
    fig5 = pt(fig5, h=440)
    fig5.update_traces(hovertemplate="<b>%{label}</b><br>%{value:,}<extra></extra>",
                       textfont_size=12)
    st.plotly_chart(fig5, use_container_width=True)

    # Descarga
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("↓  Descargar datos filtrados (CSV)", data=csv,
                       file_name="coope_creditos_2024.csv", mime="text/csv")


# ══════════════════════════════════════════════
# TAB 2 — ANÁLISIS DESCRIPTIVO
# ══════════════════════════════════════════════
def tab_descriptivo(df: pd.DataFrame):
    if df.empty:
        empty_state("Sin datos. Ajusta los filtros.")
        return

    # Monto — ordenado
    sec("Distribución por Rango de Monto de Crédito",
        "Concentración por monto concedido (orden ascendente)")
    col_m = "RANGO MONTO CREDITO CONCEDIDO"
    dm = df[col_m].value_counts().reset_index()
    dm.columns = ["Rango","Cantidad"]
    dm["order"] = dm["Rango"].map({v:i for i,v in enumerate(MONTO_ORDER)})
    dm = dm.sort_values("order")

    fig = px.bar(dm, x="Rango", y="Cantidad", color="Cantidad",
                 color_continuous_scale=[[0,"rgba(139,92,246,.35)"],[1,PRIMARY]])
    fig = pt(fig, h=340)
    fig.update_coloraxes(showscale=False)
    fig.update_traces(hovertemplate="<b>%{x}</b><br>%{y:,}<extra></extra>")
    st.plotly_chart(fig, use_container_width=True)

    top_r = dm.sort_values("Cantidad", ascending=False).iloc[0]
    insight(f"El rango <b>{top_r['Rango']}</b> lidera con <b>{top_r['Cantidad']:,}</b> operaciones "
            f"({top_r['Cantidad']/len(df)*100:.1f}% del total filtrado).", "success")

    st.markdown("<hr>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        sec("Por Nivel de Instrucción")
        di = df["INSTRUCCION"].value_counts().reset_index()
        di.columns = ["Instrucción","Cantidad"]
        fig2 = px.bar(di.sort_values("Cantidad"), x="Cantidad", y="Instrucción",
                      orientation="h", color_discrete_sequence=[SECONDARY])
        fig2 = pt(fig2, h=290)
        fig2.update_traces(hovertemplate="<b>%{y}</b><br>%{x:,}<extra></extra>")
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        sec("Por Rango de Edad")
        de = df["RANGO EDAD"].value_counts().reset_index()
        de.columns = ["Edad","Cantidad"]
        fig3 = px.bar(de, x="Edad", y="Cantidad", color_discrete_sequence=[ACCENT3])
        fig3 = pt(fig3, h=290)
        fig3.update_traces(hovertemplate="<b>%{x}</b><br>%{y:,}<extra></extra>")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    with c3:
        sec("Por Rango de Plazo")
        dp = df["RANGO PLAZO ORIGINAL CONCESION"].value_counts().reset_index()
        dp.columns = ["Plazo","Cantidad"]
        fig4 = px.bar(dp, x="Plazo", y="Cantidad", color_discrete_sequence=[ACCENT4])
        fig4 = pt(fig4, h=290)
        fig4.update_xaxes(tickangle=-20)
        fig4.update_traces(hovertemplate="<b>%{x}</b><br>%{y:,}<extra></extra>")
        st.plotly_chart(fig4, use_container_width=True)

    with c4:
        sec("Instrucción × Sexo (Stacked)")
        si = df.groupby(["INSTRUCCION","SEXO"]).size().reset_index(name="Cantidad")
        fig5 = px.bar(si, x="INSTRUCCION", y="Cantidad", color="SEXO",
                      barmode="stack", color_discrete_sequence=PALETTE)
        fig5 = pt(fig5, h=290)
        fig5.update_xaxes(tickangle=-20)
        fig5.update_traces(hovertemplate="<b>%{x}</b> · %{fullData.name}<br>%{y:,}<extra></extra>")
        st.plotly_chart(fig5, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Sunburst
    sec("Sunburst — Segmento → Instrucción → Sexo",
        "Vista radial de la composición del portafolio")
    df_sb = df.groupby(["SEGMENTO","INSTRUCCION","SEXO"]).size().reset_index(name="Cantidad")
    fig6 = px.sunburst(df_sb, path=["SEGMENTO","INSTRUCCION","SEXO"],
                       values="Cantidad", color="SEGMENTO",
                       color_discrete_sequence=PALETTE)
    fig6 = pt(fig6, h=480)
    fig6.update_traces(hovertemplate="<b>%{label}</b><br>%{value:,}<extra></extra>",
                       textfont_size=11)
    st.plotly_chart(fig6, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3 — EVOLUCIÓN TEMPORAL
# ══════════════════════════════════════════════
def tab_temporal(df: pd.DataFrame):
    if "MES_ORD" not in df.columns or df.empty:
        empty_state("Sin datos temporales disponibles.")
        return

    sec("Evolución mensual — Julio a Diciembre 2024",
        "Volumen de créditos mes a mes por tipo")

    ev = (df.groupby(["MES_ORD","MES","TIPO DE CRÉDITO"])
            .size().reset_index(name="Cantidad").sort_values("MES_ORD"))
    fig = px.line(ev, x="MES", y="Cantidad", color="TIPO DE CRÉDITO",
                  markers=True, color_discrete_sequence=PALETTE)
    fig = pt(fig, h=400)
    fig.update_traces(line=dict(width=2.5), marker=dict(size=7),
                      hovertemplate="<b>%{fullData.name}</b><br>%{x}: %{y:,}<extra></extra>")
    st.plotly_chart(fig, use_container_width=True)

    # Totales por mes + delta
    tot = (df.groupby(["MES_ORD","MES"]).size()
             .reset_index(name="Cantidad").sort_values("MES_ORD"))
    if len(tot) > 1:
        delta = (tot["Cantidad"].iloc[-1] - tot["Cantidad"].iloc[-2]) / tot["Cantidad"].iloc[-2] * 100
        arrow = "↑" if delta > 0 else "↓"
        kind  = "success" if delta > 0 else "danger"
        insight(f"<b>{tot['MES'].iloc[-1]}</b>: {tot['Cantidad'].iloc[-1]:,} créditos. "
                f"Variación vs. período anterior: <b>{arrow} {abs(delta):.1f}%</b>", kind)

    st.markdown("<hr>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        sec("Volumen total por mes")
        fig2 = px.bar(tot, x="MES", y="Cantidad", color="Cantidad",
                      color_continuous_scale=[[0,"rgba(139,92,246,.3)"],[1,PRIMARY]],
                      text="Cantidad")
        fig2 = pt(fig2, h=310)
        fig2.update_coloraxes(showscale=False)
        fig2.update_traces(texttemplate="%{text:,}", textposition="outside",
                           textfont_size=10, textfont_color=MUTED,
                           hovertemplate="<b>%{x}</b><br>%{y:,}<extra></extra>")
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        sec("Segmento × Mes")
        es = (df.groupby(["MES_ORD","MES","SEGMENTO"]).size()
                .reset_index(name="Cantidad").sort_values("MES_ORD"))
        fig3 = px.bar(es, x="MES", y="Cantidad", color="SEGMENTO",
                      barmode="stack", color_discrete_sequence=PALETTE)
        fig3 = pt(fig3, h=310)
        fig3.update_traces(hovertemplate="<b>%{x}</b> · %{fullData.name}<br>%{y:,}<extra></extra>")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Heatmap actividad mes × provincia
    sec("Heatmap de actividad crediticia — Mes × Provincia (Top 15)")
    top15 = df["PROVINCIA"].value_counts().head(15).index.tolist()
    df_hm = df[df["PROVINCIA"].isin(top15)]
    hmp = (df_hm.groupby(["MES_ORD","MES","PROVINCIA"]).size()
                .reset_index(name="Cantidad").sort_values("MES_ORD"))
    # Ordenar meses
    mes_orden = hmp.drop_duplicates("MES_ORD").sort_values("MES_ORD")["MES"].tolist()
    pivot = hmp.pivot(index="PROVINCIA", columns="MES", values="Cantidad").fillna(0)
    pivot = pivot[[c for c in mes_orden if c in pivot.columns]]

    fig4 = go.Figure(go.Heatmap(
        z=pivot.values, x=pivot.columns.tolist(), y=pivot.index.tolist(),
        colorscale=[[0,BG],[0.3,"rgba(139,92,246,0.67)"],[1,PRIMARY]],
        text=[[f"{int(v):,}" for v in row] for row in pivot.values],
        texttemplate="%{text}", textfont=dict(size=10, color=TEXT),
        hovertemplate="<b>%{y}</b> · %{x}<br>Créditos: %{z:,}<extra></extra>",
        colorbar=dict(
            title=dict(text="Créditos", font=dict(color=MUTED, size=11)),
            tickfont=dict(color=MUTED, size=10),
        )
    ))
    fig4 = pt(fig4, h=440)
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Área acumulada
    sec("Área acumulada — Tipo de Crédito por Mes")
    fig5 = px.area(ev, x="MES", y="Cantidad", color="TIPO DE CRÉDITO",
                   color_discrete_sequence=PALETTE)
    fig5 = pt(fig5, h=360)
    fig5.update_traces(hovertemplate="<b>%{fullData.name}</b><br>%{x}: %{y:,}<extra></extra>")
    st.plotly_chart(fig5, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 4 — DESTINO FINANCIERO
# ══════════════════════════════════════════════
def tab_destino(df: pd.DataFrame):
    if "DESTINO FINANCIERO" not in df.columns or df.empty:
        empty_state("Sin datos de destino financiero.")
        return

    dest_col = "DESTINO FINANCIERO"
    dest = df[dest_col].value_counts().reset_index()
    dest.columns = ["Destino","Cantidad"]
    dest["Pct"] = dest["Cantidad"] / len(df) * 100
    dest["Etiqueta"] = dest["Destino"].str[:45]

    sec("Ranking de Destinos Financieros",
        "Clasificación de créditos según su destino económico")

    fig = px.bar(dest.sort_values("Cantidad"), x="Cantidad", y="Etiqueta",
                 orientation="h", color="Cantidad",
                 color_continuous_scale=[[0,"rgba(16,185,129,.3)"],[1,ACCENT3]],
                 text=dest.sort_values("Cantidad")["Pct"].apply(lambda x: f"{x:.1f}%"))
    fig = pt(fig, h=max(420, len(dest)*34))
    fig.update_coloraxes(showscale=False)
    fig.update_traces(textposition="outside", textfont=dict(size=10, color=MUTED),
                      hovertemplate="<b>%{y}</b><br>%{x:,}<extra></extra>")
    st.plotly_chart(fig, use_container_width=True)

    top_d = dest.iloc[0]
    insight(f"Destino principal: <b>{top_d['Destino'][:70]}</b> — "
            f"<b>{top_d['Cantidad']:,}</b> operaciones ({top_d['Pct']:.1f}%).", "success")

    st.markdown("<hr>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        sec("Destino × Segmento (Top 8)")
        top8 = dest.head(8)["Destino"].tolist()
        ds8 = df[df[dest_col].isin(top8)]
        dsg = ds8.groupby([dest_col,"SEGMENTO"]).size().reset_index(name="Cantidad")
        dsg["Etiqueta"] = dsg[dest_col].str[:28]
        fig2 = px.bar(dsg, x="Cantidad", y="Etiqueta", color="SEGMENTO",
                      orientation="h", barmode="stack", color_discrete_sequence=PALETTE)
        fig2 = pt(fig2, h=400)
        fig2.update_layout(yaxis=dict(autorange="reversed"))
        fig2.update_traces(hovertemplate="%{fullData.name}<br>%{x:,}<extra></extra>")
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        sec("Destino × Tipo de Crédito (Top 6)")
        top6 = dest.head(6)["Destino"].tolist()
        dt6 = df[df[dest_col].isin(top6)]
        dtg = dt6.groupby([dest_col,"TIPO DE CRÉDITO"]).size().reset_index(name="Cantidad")
        dtg["Etiqueta"] = dtg[dest_col].str[:22]
        fig3 = px.bar(dtg, x="Etiqueta", y="Cantidad", color="TIPO DE CRÉDITO",
                      barmode="group", color_discrete_sequence=PALETTE)
        fig3 = pt(fig3, h=400)
        fig3.update_xaxes(tickangle=-30)
        fig3.update_traces(hovertemplate="<b>%{x}</b> · %{fullData.name}<br>%{y:,}<extra></extra>")
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Sunburst Destino → Sexo → Instrucción
    sec("Sunburst — Destino → Sexo → Instrucción (Top 6 destinos)")
    df_sun = df[df[dest_col].isin(top6)].copy()
    df_sun["DestCorto"] = df_sun[dest_col].str[:30]
    df_sb = df_sun.groupby(["DestCorto","SEXO","INSTRUCCION"]).size().reset_index(name="Cantidad")
    fig4 = px.sunburst(df_sb, path=["DestCorto","SEXO","INSTRUCCION"],
                       values="Cantidad", color_discrete_sequence=PALETTE)
    fig4 = pt(fig4, h=480)
    fig4.update_traces(hovertemplate="<b>%{label}</b><br>%{value:,}<extra></extra>")
    st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 5 — PCA
# ══════════════════════════════════════════════
PCA_BASE = ["INSTRUCCION","SEXO","RANGO EDAD","RANGO MONTO CREDITO CONCEDIDO",
            "TIPO PERSONA","TIPO DE CRÉDITO","SEGMENTO"]

def tab_pca(df: pd.DataFrame):
    pca_cols = [c for c in PCA_BASE if c in df.columns]
    if len(pca_cols) < 2 or len(df) < 20:
        empty_state("Se necesitan al menos 20 registros y 2 variables para el PCA.")
        return

    sec("Análisis de Componentes Principales (PCA)",
        "Reducción dimensional — máximo 20 000 registros por rendimiento")

    sample_n = min(len(df), 20000)
    df_s = df.sample(sample_n, random_state=42)
    df_p = df_s[pca_cols].copy()
    les = {}
    for c in pca_cols:
        le = LabelEncoder()
        df_p[c] = le.fit_transform(df_p[c].astype(str))
        les[c] = le

    X = StandardScaler().fit_transform(df_p)
    n_comp = min(3, len(pca_cols)-1, len(df_p)-1)
    pca = PCA(n_components=n_comp)
    res = pca.fit_transform(X)
    exp = pca.explained_variance_ratio_

    lines = [f"<b>PC{i+1}</b>: {exp[i]*100:.1f}%" for i in range(n_comp)]
    insight(f"Varianza explicada — {' · '.join(lines)} · "
            f"Total: <b>{sum(exp)*100:.1f}%</b>  (muestra: {sample_n:,} registros)", "success")

    df_res = pd.DataFrame(res[:, :2], columns=["PC1","PC2"])
    df_res["TIPO DE CRÉDITO"] = df_s["TIPO DE CRÉDITO"].values
    df_res["SEGMENTO"]        = df_s["SEGMENTO"].values

    c1, c2 = st.columns(2)
    with c1:
        sec("PCA — Tipo de Crédito")
        fig = px.scatter(df_res, x="PC1", y="PC2", color="TIPO DE CRÉDITO",
                         opacity=0.60, color_discrete_sequence=PALETTE,
                         labels={"PC1":f"PC1 ({exp[0]*100:.1f}%)",
                                 "PC2":f"PC2 ({exp[1]*100:.1f}%)"})
        fig = pt(fig, h=420)
        fig.update_traces(marker=dict(size=4),
                          hovertemplate="<b>%{marker.color}</b><br>PC1:%{x:.2f} PC2:%{y:.2f}<extra></extra>")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        sec("PCA — Segmento")
        fig2 = px.scatter(df_res, x="PC1", y="PC2", color="SEGMENTO",
                          opacity=0.60, color_discrete_sequence=PALETTE,
                          labels={"PC1":f"PC1 ({exp[0]*100:.1f}%)",
                                  "PC2":f"PC2 ({exp[1]*100:.1f}%)"})
        fig2 = pt(fig2, h=420)
        fig2.update_traces(marker=dict(size=4))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    sec("Contribución de Variables a los Componentes Principales",
        "Mayor barra = mayor influencia en la reducción dimensional")

    df_imp = pd.DataFrame({"Variable": pca_cols})
    for i in range(n_comp):
        df_imp[f"PC{i+1}"] = pca.components_[i]
    df_imp["Importancia"] = df_imp[[f"PC{i+1}" for i in range(n_comp)]].abs().sum(axis=1)
    df_imp = df_imp.sort_values("Importancia", ascending=False)

    fig3 = go.Figure()
    for i, col in enumerate([PRIMARY, SECONDARY, ACCENT3][:n_comp]):
        fig3.add_trace(go.Bar(name=f"PC{i+1}", x=df_imp["Variable"],
                              y=df_imp[f"PC{i+1}"].abs(), marker_color=col))
    fig3.update_layout(barmode="group")
    fig3 = pt(fig3, h=320)
    st.plotly_chart(fig3, use_container_width=True)

    # Varianza explicada acumulada
    sec("Varianza Explicada Acumulada")
    var_df = pd.DataFrame({
        "Componente": [f"PC{i+1}" for i in range(n_comp)],
        "Varianza %": exp * 100,
        "Acumulada %": np.cumsum(exp) * 100,
    })
    fig4 = go.Figure()
    fig4.add_trace(go.Bar(x=var_df["Componente"], y=var_df["Varianza %"],
                          name="Varianza individual", marker_color=PRIMARY))
    fig4.add_trace(go.Scatter(x=var_df["Componente"], y=var_df["Acumulada %"],
                              name="Acumulada", mode="lines+markers",
                              line=dict(color=SECONDARY, width=2.5),
                              marker=dict(size=8)))
    fig4 = pt(fig4, h=280)
    st.plotly_chart(fig4, use_container_width=True)

    with st.expander("Ver tabla de cargas completa"):
        st.dataframe(df_imp.set_index("Variable").round(4), use_container_width=True)


# ══════════════════════════════════════════════
# TAB 6 — CHI-CUADRADO
# ══════════════════════════════════════════════
CHI_BASE = ["INSTRUCCION","SEXO","RANGO EDAD","RANGO MONTO CREDITO CONCEDIDO",
            "TIPO PERSONA","TIPO DE CRÉDITO","DESTINO FINANCIERO","SEGMENTO"]

def _fmt_p(v):
    if np.isnan(v) or v > 0.99:
        return "—"
    if v < 0.001:
        return "<0.001"
    return f"{v:.3f}"

def tab_chi2(df: pd.DataFrame):
    chi_cols = [c for c in CHI_BASE if c in df.columns]
    if len(chi_cols) < 2 or len(df) < 5:
        empty_state("Se necesitan al menos 5 registros y 2 variables.")
        return

    # Muestra para rendimiento con Chi²
    sample_n = min(len(df), 50000)
    df_c = df.sample(sample_n, random_state=42) if len(df) > sample_n else df

    sec("Análisis de Asociación Chi-Cuadrado",
        "Prueba de independencia estadística entre pares de variables (α = 0.05)")
    if len(df) > sample_n:
        insight(f"Calculado sobre muestra de {sample_n:,} registros para rendimiento.", "warning")

    records = []
    for i, v1 in enumerate(chi_cols):
        for j, v2 in enumerate(chi_cols):
            if i >= j:
                continue
            try:
                ct = pd.crosstab(df_c[v1], df_c[v2])
                if ct.shape[0] < 2 or ct.shape[1] < 2:
                    continue
                chi2, p, dof, _ = chi2_contingency(ct)
                # Cramér's V
                n_ct = ct.sum().sum()
                k = min(ct.shape) - 1
                cramers_v = np.sqrt(chi2 / (n_ct * k)) if k > 0 else 0
                records.append({
                    "Variable 1": v1, "Variable 2": v2,
                    "Chi²": round(chi2, 2), "p-valor": p,
                    "G.L.": dof,
                    "Cramér V": round(cramers_v, 4),
                    "Asociación": "Sí ✓" if p < 0.05 else "No ✗",
                })
            except Exception:
                continue

    if not records:
        empty_state("No se pudo calcular Chi² con los datos actuales.")
        return

    df_chi = pd.DataFrame(records).sort_values("Chi²", ascending=False)
    sig = (df_chi["p-valor"] < 0.05).sum()
    insight(f"De <b>{len(df_chi)}</b> pares evaluados, <b>{sig}</b> muestran asociación "
            f"significativa (p < 0.05). Se incluye <b>Cramér's V</b> como medida de fuerza.",
            "success" if sig > 0 else "warning")

    # ─── Heatmap Cramér's V (más informativo que p-valores cero) ───
    sec("Heatmap de Cramér's V",
        "Fuerza de asociación entre variables — 0 = independencia, 1 = asociación perfecta")

    cv_mat = pd.DataFrame(0.0, index=chi_cols, columns=chi_cols)
    for _, r in df_chi.iterrows():
        cv_mat.loc[r["Variable 1"], r["Variable 2"]] = r["Cramér V"]
        cv_mat.loc[r["Variable 2"], r["Variable 1"]] = r["Cramér V"]
    np.fill_diagonal(cv_mat.values, 1.0)

    short = {c: c.replace("RANGO ","").replace(" CONCEDIDO","").replace(" CONCESION","")
                 .replace("DESTINO FINANCIERO","DESTINO").replace("INSTRUCCION","INSTRUC.")
                 for c in chi_cols}
    cm = cv_mat.rename(index=short, columns=short)

    text_cv = [[f"{cm.iloc[i,j]:.3f}" if cm.iloc[i,j] < 0.999 else "—"
                for j in range(len(cm.columns))]
               for i in range(len(cm.index))]

    fig_hm = go.Figure(go.Heatmap(
        z=cm.values, x=cm.columns.tolist(), y=cm.index.tolist(),
        colorscale=[[0,"#0d1117"],[0.2,SURFACE2],[0.5,"rgba(139,92,246,0.67)"],[1,PRIMARY]],
        zmin=0, zmax=1,
        text=text_cv,
        texttemplate="%{text}", textfont=dict(size=10, color=TEXT),
        hovertemplate="<b>%{y}</b> × <b>%{x}</b><br>Cramér's V: %{z:.4f}<extra></extra>",
        colorbar=dict(
            title=dict(text="Cramér's V", font=dict(color=MUTED, size=11)),
            tickfont=dict(color=MUTED, size=10),
        )
    ))
    fig_hm = pt(fig_hm, h=440)
    st.plotly_chart(fig_hm, use_container_width=True)

    insight("Un Cramér's V <b>> 0.3</b> indica asociación moderada. "
            "<b>> 0.5</b> asociación fuerte. La diagonal (valor 1.0) se excluye del análisis.", "info")

    st.markdown("<hr>", unsafe_allow_html=True)

    # Ranking Chi² con Cramér's V
    sec("Ranking de Asociaciones", "Ordenado por Chi² — color indica significancia")
    df_top = df_chi.head(20).copy()
    df_top["Par"] = df_top["Variable 1"].str[:14] + " × " + df_top["Variable 2"].str[:14]

    fig_bar = go.Figure(go.Bar(
        x=df_top["Chi²"], y=df_top["Par"], orientation="h",
        marker=dict(
            color=df_top["Cramér V"],
            colorscale=[[0,SURFACE2],[0.3,SECONDARY],[1,PRIMARY]],
            showscale=True,
            colorbar=dict(
                title=dict(text="Cramér V", font=dict(color=MUTED, size=10)),
                tickfont=dict(color=MUTED, size=9), thickness=12,
            )
        ),
        text=[f"V={r['Cramér V']:.3f}" for _, r in df_top.iterrows()],
        textposition="outside", textfont=dict(size=10, color=MUTED),
        hovertemplate="<b>%{y}</b><br>Chi²: %{x:.2f}<extra></extra>",
    ))
    fig_bar = pt(fig_bar, h=max(400, len(df_top)*28))
    fig_bar.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    sec("Tabla completa de resultados")
    df_show = df_chi.copy()
    df_show["p-valor"] = df_show["p-valor"].apply(lambda p: "<0.001" if p < 0.001 else f"{p:.4f}")
    st.dataframe(
        df_show.style.map(
            lambda v: f"color:{ACCENT3}" if v == "Sí ✓" else f"color:{MUTED}",
            subset=["Asociación"]
        ),
        use_container_width=True, height=360,
    )


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main():
    # Splash (solo primera carga de sesión)
    if "splash" not in st.session_state:
        st.session_state.splash = False
    if not st.session_state.splash:
        render_splash()
        st.session_state.splash = True

    inject_css()

    # Carga de datos
    with st.spinner("Cargando 372 000 registros..."):
        try:
            df_full = load_data()
        except FileNotFoundError:
            st.error("No se encontró `2024.xlsx`. Ejecuta desde la carpeta `Dash-main/`.")
            st.stop()
        except Exception as e:
            st.error(f"Error al cargar datos: {e}")
            st.stop()

    # Sidebar + filtros
    df_f = build_sidebar(df_full)

    # Header
    dashboard_header()

    # Alerta si filtros dejan dataset vacío
    if df_f.empty:
        empty_state("La combinación de filtros no retorna datos. Ajusta los filtros en el panel izquierdo.")
        return

    # KPIs
    render_kpis(df_f, df_full)

    # Tabs
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "Resumen General",
        "Descriptivo",
        "Evolución Temporal",
        "Destino Financiero",
        "PCA",
        "Chi-Cuadrado",
    ])
    with t1: tab_resumen(df_f)
    with t2: tab_descriptivo(df_f)
    with t3: tab_temporal(df_f)
    with t4: tab_destino(df_f)
    with t5: tab_pca(df_f)
    with t6: tab_chi2(df_f)


if __name__ == "__main__":
    main()
