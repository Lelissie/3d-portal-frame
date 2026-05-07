"""
3D Portal Frame — Parametric Designer
Single-story pitched portal frame in glued laminated timber (GLT).

UX: Workflow tabs on the left (Project Setting, Geometry, Loading,
Analysis, Design, Report) with a persistent 3D viewport on the right
(Architectural Model / Structural Model toggle).

Run:
    pip install -r requirements.txt
    streamlit run app.py
"""
import math
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from modules.geometry      import FrameGeometry, Section
from modules.materials     import GLT_GRADES, KMOD, GAMMA_M
from modules.load_takedown import SurfaceLoads, build_loads, combine
from modules.design_glt    import design_beam, design_column


# ====================================================================
# Page setup
# ====================================================================
st.set_page_config(page_title="3D Portal Frame",
                   page_icon="🏛️",
                   layout="wide",
                   initial_sidebar_state="collapsed")

st.markdown("""
<style>
    /* page-wide */
    .main .block-container {padding-top: 1rem; padding-bottom: 2rem;
                            max-width: 100%;}
    h1 {color: #1565C0; margin: 0 0 0.4rem 0;}
    h2,h3 {color: #1565C0;}

    /* tab strip — clean look that connects to the panel below */
    [data-baseweb="tab-list"] {
        gap: 0;
        border: 1px solid #B0BEC5;
        border-bottom: none;
        border-radius: 6px 6px 0 0;
        padding: 0.25rem 0.25rem 0;
        background: #F5F7FA;
        margin-bottom: 0 !important;
    }
    [data-baseweb="tab"] {
        padding: 0.55rem 1.1rem !important;
        border-radius: 4px 4px 0 0 !important;
        background: transparent !important;
    }
    [data-baseweb="tab"][aria-selected="true"] {
        background: white !important;
    }

    /* >>> THE MAIN TAB CONTENT PANEL — visible bordered card  <<< */
    [data-baseweb="tab-panel"] {
        border: 1px solid #B0BEC5;
        border-radius: 0 0 6px 6px;
        padding: 1.25rem 1.5rem !important;
        background: white;
        min-height: 640px;
        margin-top: 0 !important;
    }

    /* group headers in the parameter panel */
    .group-header {
        font-size: 0.78rem;
        font-weight: 700;
        color: #455A64;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        border-bottom: 1px solid #ECEFF1;
        margin: 0.6rem 0 0.4rem 0;
        padding-bottom: 0.2rem;
    }

    /* >>> Prominent Architectural / Structural toggle <<<
       Targets every horizontal radio group (the viewport one is the most
       prominent, and the other tabs don't use horizontal radios). */
    div[data-testid="stRadio"] > div[role="radiogroup"][aria-orientation="horizontal"] {
        background: #F5F7FA;
        border: 2px solid #1565C0;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        gap: 2rem;
        justify-content: center;
        margin-bottom: 0.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    div[data-testid="stRadio"] > div[role="radiogroup"][aria-orientation="horizontal"] label {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        color: #263238 !important;
    }

    .pass {color:#2E7D32; font-weight:600;}
    .fail {color:#C62828; font-weight:600;}
</style>
""", unsafe_allow_html=True)


# ====================================================================
# Session-state defaults
# ====================================================================
ss = st.session_state
ss.setdefault("project", {
    "name": "3D portal Frame",
    "location": "Ethiopia",
    "engineer": "Lelissie",
    "code": "EN 1995-1-1",
    "service_class": 1,
    "load_duration": "medium_term",
    "kmod": KMOD["medium_term"],
    "gamma_M": GAMMA_M,
    "combo": "ULS (1.35G + 1.5Q + 0.9W)",
})
ss.setdefault("geom_params", {
    "H": 9.0,
    "a": 5.25,
    "L_a": 14.0,
    "H_s": 6.0,
    "H_f": 9.0,
    "h_a_cm": 35.0,
    "h_f_cm": 30.0,
    "h_1_cm": 90.0,
    "l_zw_cm": 40.0,
    "n_bays": 4,
    "n_purlins_per_slope": 3,
    "col_b": 200, "col_h": 800, "col_grade": "GL32h",
    "raf_b": 200, "raf_h": 800, "raf_grade": "GL32h",
    "pur_b": 115, "pur_h": 240, "pur_grade": "GL24h",
})
ss.setdefault("appearance", {"view_mode": "Architectural Model"})
ss.setdefault("surface_loads", {"g_roof": 2.00, "q_snow": 1.00, "w_wind": 0.50})
ss.setdefault("results", {})
ss.setdefault("design_results", {})


@st.cache_data(show_spinner=False)
def _build_geometry_cached(n_bays, a, L_a, H, H_s, H_f,
                            h_a_cm, h_f_cm, h_1_cm, l_zw_cm,
                            n_purlins_per_slope,
                            col_b, col_h, col_grade,
                            raf_b, raf_h, raf_grade,
                            pur_b, pur_h, pur_grade):
    g = FrameGeometry(
        H=H, B=a * n_bays, a=a, L_a=L_a,
        H_s=H_s, H_f=H_f,
        h_a_cm=h_a_cm, h_f_cm=h_f_cm,
        h_1_cm=h_1_cm, l_zw_cm=l_zw_cm,
        n_bays=n_bays, n_purlins_per_slope=n_purlins_per_slope,
        column_section=Section("C-section", col_b, col_h, col_grade),
        rafter_section=Section("R-section", raf_b, raf_h, raf_grade),
        purlin_section=Section("P-section", pur_b, pur_h, pur_grade),
    )
    g.build()
    return g


@st.cache_data(show_spinner=False)
def _build_viewport_fig(n_bays, a, L_a, H, H_s, H_f,
                         h_a_cm, h_f_cm, h_1_cm, l_zw_cm,
                         n_purlins_per_slope,
                         col_b, col_h, col_grade,
                         raf_b, raf_h, raf_grade,
                         pur_b, pur_h, pur_grade,
                         view_mode):
    from modules.visualization import fig_architectural, fig_structural
    g = _build_geometry_cached(n_bays, a, L_a, H, H_s, H_f,
                                h_a_cm, h_f_cm, h_1_cm, l_zw_cm,
                                n_purlins_per_slope,
                                col_b, col_h, col_grade,
                                raf_b, raf_h, raf_grade,
                                pur_b, pur_h, pur_grade)
    if view_mode == "Architectural Model":
        return fig_architectural(g)
    return fig_structural(g)


def build_geometry() -> FrameGeometry:
    p = ss.geom_params
    return _build_geometry_cached(
        p["n_bays"], p["a"], p["L_a"], p["H"], p["H_s"], p["H_f"],
        p["h_a_cm"], p["h_f_cm"], p["h_1_cm"], p["l_zw_cm"],
        p["n_purlins_per_slope"],
        p["col_b"], p["col_h"], p["col_grade"],
        p["raf_b"], p["raf_h"], p["raf_grade"],
        p["pur_b"], p["pur_h"], p["pur_grade"],
    )


# ====================================================================
# TITLE
# ====================================================================
st.title("🏛️ 3D Portal Frame")


# --------------------------------------------------------------------
# Page-level layout: LEFT (tabs + content)  |  RIGHT (3D viewport)
# --------------------------------------------------------------------
LEFT, RIGHT = st.columns([1, 1.25], gap="large")


# ====================================================================
# RIGHT — persistent 3D viewport (rendered ONCE; visible across all tabs)
# ====================================================================
with RIGHT:
    # Prominent Architectural / Structural toggle (styled via global CSS)
    ss.appearance["view_mode"] = st.radio(
        "View mode",
        ("Architectural Model", "Structural Model"),
        index=0 if ss.appearance["view_mode"] == "Architectural Model" else 1,
        horizontal=True,
        label_visibility="collapsed",
        key="view_mode_radio",
    )

    p = ss.geom_params
    fig = _build_viewport_fig(
        p["n_bays"], p["a"], p["L_a"], p["H"], p["H_s"], p["H_f"],
        p["h_a_cm"], p["h_f_cm"], p["h_1_cm"], p["l_zw_cm"],
        p["n_purlins_per_slope"],
        p["col_b"], p["col_h"], p["col_grade"],
        p["raf_b"], p["raf_h"], p["raf_grade"],
        p["pur_b"], p["pur_h"], p["pur_grade"],
        ss.appearance["view_mode"],
    )

    fig.update_layout(title=None, height=680,
                      margin=dict(l=0, r=0, t=10, b=0),
                      legend=dict(orientation="h", yanchor="bottom",
                                  y=0.0, x=0.5, xanchor="center"))
    st.plotly_chart(fig, width='stretch')


# ====================================================================
# LEFT — workflow tabs
# ====================================================================
with LEFT:
    WORKFLOW = ["Project Setting", "Geometry", "Loading",
                "Analysis", "Design", "Report"]
    tabs = st.tabs(WORKFLOW)

    # --- helpers ---
    def slider_with_value(label, key_dict, key, *, mn, mx, step, fmt="%.2f",
                          help=None, kind="float"):
        state_key = f"_s_{key}"
        if state_key not in ss:
            ss[state_key] = key_dict[key]
        if kind == "int":
            new = st.slider(label, min_value=int(mn), max_value=int(mx),
                            step=int(step), help=help, key=state_key)
            new = int(new)
        else:
            new = st.slider(label, min_value=float(mn), max_value=float(mx),
                            step=float(step), format=fmt, help=help, key=state_key)
        key_dict[key] = new
        return new

    def group_header(text):
        st.markdown(f'<div class="group-header">{text}</div>',
                    unsafe_allow_html=True)

    # ================================================================
    # TAB 1 — PROJECT SETTING
    # ================================================================
    with tabs[0]:
        st.markdown("##### Project Setting")
        st.caption("General data and Eurocode design parameters.")
        c1, c2 = st.columns(2)
        with c1:
            ss.project["name"]     = st.text_input("Project name",   ss.project["name"])
            ss.project["location"] = st.text_input("Location",       ss.project["location"])
            ss.project["engineer"] = st.text_input("Engineer",       ss.project["engineer"])
            ss.project["code"]     = st.text_input("Design code",    ss.project["code"])
        with c2:
            ss.project["service_class"] = st.selectbox(
                "Service class", [1, 2, 3],
                index=[1,2,3].index(ss.project["service_class"]))
            dur = st.selectbox(
                "Load duration class", list(KMOD.keys()),
                index=list(KMOD.keys()).index(ss.project["load_duration"]))
            ss.project["load_duration"] = dur
            ss.project["kmod"] = KMOD[dur]
            st.markdown(
                f'<p style="font-size:0.85rem;margin-bottom:0.1rem;color:#555">k_mod</p>'
                f'<p style="font-size:0.85rem;margin-top:0">{ss.project["kmod"]:.2f}</p>',
                unsafe_allow_html=True)
            ss.project["gamma_M"] = st.number_input(
                "γ_M (material partial factor)",
                value=ss.project["gamma_M"],
                min_value=1.0, max_value=2.0, step=0.05)
            ss.project["combo"] = st.text_input(
                "Default combination", ss.project["combo"])

        with st.expander("📋 Scope & assumptions"):
            st.markdown("""
            - Only UDL is considered for the roof.
            - Self-weight of cladding is captured via the *permanent surface load*;
              structural self-weight is added automatically from member sections × density.
            - **Pinned supports** at all column bases.
            - One-way load distribution (purlin → rafter → column → footing).
            - Concentric beam loading is assumed for the columns.
            - No lateral restraints except where explicitly modelled.
            """)

    # ================================================================
    # TAB 2 — GEOMETRY  (parametric sliders)
    # ================================================================
    with tabs[1]:
        p = ss.geom_params
        st.markdown("##### Geometry")
        st.caption("All sliders update the 3D model on the right in real time.")

        group_header("Structure")
        slider_with_value("Number of bays", p, "n_bays",
                          mn=1, mx=10, step=1, fmt="%d", kind="int",
                          help="Number of bays along the building depth.")
        slider_with_value("Frame spacing  a (m)", p, "a",
                          mn=2.0, mx=10.0, step=0.25,
                          help="Centre-to-centre spacing between portal frames.")
        slider_with_value("Frame span  L_a (m)", p, "L_a",
                          mn=4.0, mx=40.0, step=0.5,
                          help="Total span across the ridge.")
        slider_with_value("Eaves height  H_s (m)", p, "H_s",
                          mn=2.0, mx=12.0, step=0.1,
                          help="Height to the knee / eaves.")
        slider_with_value("Ridge height  H_f (m)", p, "H_f",
                          mn=2.5, mx=20.0, step=0.1,
                          help="Apex height. Must exceed eaves height.")
        if p["H_f"] <= p["H_s"]:
            p["H_f"] = p["H_s"] + 1.0
            ss["_s_H_f"] = p["H_f"]
            st.warning(f"Ridge height auto-adjusted to {p['H_f']:.1f} m "
                       f"(must exceed eaves {p['H_s']:.1f} m).")
        p["H"] = max(p["H"], p["H_f"])

        B = p["a"] * p["n_bays"]
        roof_slope_deg = math.degrees(math.atan2(p["H_f"]-p["H_s"], p["L_a"]/2))
        st.caption(f"📏 Building depth **B = a × n_bays = {B:.2f} m**  ·  "
                   f"Roof slope **{roof_slope_deg:.1f}°**")

        slider_with_value("Purlins per slope", p, "n_purlins_per_slope",
                          mn=0, mx=8, step=1, fmt="%d", kind="int",
                          help="Intermediate purlins (excludes eaves & ridge).")

        group_header("Haunched cross-section")
        c1, c2 = st.columns(2)
        with c1:
            slider_with_value("Footing  h_a (cm)", p, "h_a_cm",
                              mn=10.0, mx=120.0, step=1.0, fmt="%.0f")
            slider_with_value("Knee  h_1 (cm)", p, "h_1_cm",
                              mn=20.0, mx=200.0, step=1.0, fmt="%.0f")
        with c2:
            slider_with_value("Ridge  h_f (cm)", p, "h_f_cm",
                              mn=10.0, mx=120.0, step=1.0, fmt="%.0f")
            slider_with_value("Wedge  l_zw (cm)", p, "l_zw_cm",
                              mn=10.0, mx=200.0, step=1.0, fmt="%.0f")

        group_header("Members & GL grades")
        grades = list(GLT_GRADES.keys())

        # Initialize session-state keys for these number_inputs once
        for k in ("col_b","col_h","raf_b","raf_h","pur_b","pur_h"):
            if k not in ss:
                ss[k] = p[k]
        for k in ("col_g","raf_g","pur_g"):
            if k not in ss:
                ss[k] = p[k.replace("_g","_grade")]

        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st.caption("**Column**")
            p["col_h"] = st.number_input("h — depth (mm)", min_value=100, max_value=1000,
                                         step=10, key="col_h",
                                         help="Structural depth — faces the span; resists in-plane portal bending (strong axis).")
            p["col_b"] = st.number_input("b — width (mm)", min_value=80, max_value=1000,
                                         step=10, key="col_b",
                                         help="Section width — out-of-plane dimension.")
            p["col_grade"] = st.selectbox("grade", grades, key="col_g")
        with cc2:
            st.caption("**Rafter**")
            p["raf_h"] = st.number_input("h — depth (mm)", min_value=100, max_value=1000,
                                         step=10, key="raf_h",
                                         help="Structural depth — perpendicular to rafter axis, in-plane; resists bending (strong axis).")
            p["raf_b"] = st.number_input("b — width (mm)", min_value=80, max_value=1000,
                                         step=10, key="raf_b",
                                         help="Section width — out-of-plane dimension (building depth direction).")
            p["raf_grade"] = st.selectbox("grade", grades, key="raf_g")
        with cc3:
            st.caption("**Purlin**")
            p["pur_h"] = st.number_input("h — depth (mm)", min_value=100, max_value=400,
                                         step=10, key="pur_h",
                                         help="Purlin depth — vertical dimension; resists gravity bending.")
            p["pur_b"] = st.number_input("b — width (mm)", min_value=60, max_value=240,
                                         step=10, key="pur_b",
                                         help="Purlin width — horizontal dimension.")
            p["pur_grade"] = st.selectbox("grade", grades, key="pur_g")

        with st.expander("Parameter legend"):
            st.markdown("""
            | Symbol | Meaning |
            |---|---|
            | a    | Frame (truss) spacing |
            | L_a  | Portal-frame span across the ridge |
            | H_s  | Eaves (knee) height |
            | H_f  | Ridge (apex) height |
            | h_a  | Cross-section depth at footing |
            | h_f  | Cross-section depth at ridge |
            | h_1  | Cross-section depth at knee |
            | l_zw | Inserted wedge / haunch length |
            """)

    # ================================================================
    # TAB 3 — LOADING
    # ================================================================
    with tabs[2]:
        st.markdown("##### Loading")
        st.caption("Surface loads on the roof and load take-down to line elements.")
        geom = build_geometry()
        sl = ss.surface_loads
        c1, c2, c3 = st.columns(3)
        with c1:
            sl["g_roof"] = st.number_input("G — Permanent (kN/m²)",
                                           value=sl["g_roof"],
                                           min_value=0.0, step=0.05,
                                           help="Cladding, insulation, membrane")
        with c2:
            sl["q_snow"] = st.number_input("Q — Snow (kN/m², on plan)",
                                           value=sl["q_snow"],
                                           min_value=0.0, step=0.05,
                                           help="Characteristic snow on plan")
        with c3:
            sl["w_wind"] = st.number_input("W — Wind (kN/m², ⟂ roof)",
                                           value=sl["w_wind"],
                                           min_value=0.0, step=0.05,
                                           help="Characteristic suction normal to slope")

        with st.expander("Load take-down logic"):
            st.markdown("""
            1. **Surface → rafter line load**: *w = q × tributary frame-spacing*
               (interior frames: a; end frames: a/2)
            2. **Snow on plan** projected onto rafter: *w = q_snow × trib × cos α*
            3. **Wind**: char. suction ⟂ roof — resolved into global Y, Z.
            4. **Self-weight**: ρ_mean × g × A on every line element.
            """)

        LM = build_loads(geom, SurfaceLoads(**sl), include_selfweight=True)
        rows = []
        for case, items in LM.cases.items():
            for L in items:
                e = next(x for x in geom.elements if x.id == L.element_id)
                rows.append([case, e.id, e.member_type, e.frame_id,
                             f"{L.wx:+.3f}", f"{L.wy:+.3f}", f"{L.wz:+.3f}"])
        df = pd.DataFrame(rows, columns=["Case","Elem","Type","Frame",
                                         "wx","wy","wz"])
        st.markdown(f"**Generated line loads**: {len(rows)} entries across "
                    f"{len(LM.cases)} cases (G, Q, W)")
        st.dataframe(df, height=320, use_container_width=True)

    # ================================================================
    # TAB 4 — ANALYSIS
    # ================================================================
    with tabs[3]:
        st.markdown("##### Analysis")
        st.caption("Linear-elastic 3D analysis with OpenSeesPy.")
        geom = build_geometry()
        LM   = build_loads(geom, SurfaceLoads(**ss.surface_loads))

        combos = st.multiselect(
            "Load combinations",
            ["G_only", "Q_only", "W_only", "ULS", "SLS_char", "SLS_qp"],
            default=["G_only", "ULS", "SLS_char"]
        )

        if st.button("▶ Run analysis", type="primary",
                     use_container_width=True):
            from modules.analysis import run_combo
            ss.results = {}
            progress = st.progress(0.0)
            for i, c in enumerate(combos):
                with st.spinner(f"Running {c} ..."):
                    loads = combine(LM, combo=c)
                    ss.results[c] = run_combo(geom, loads, combo_name=c)
                progress.progress((i+1)/len(combos))
            st.success(f"✅ Completed {len(combos)} analysis runs.")

        if ss.results:
            from modules.analysis import envelope
            combo_to_show = st.selectbox("View combo", list(ss.results.keys()))
            res = ss.results[combo_to_show]

            umax = max(np.linalg.norm(d[:3]) for d in res.node_disp.values())
            st.metric("Max nodal displacement", f"{umax*1000:.2f} mm")

            with st.expander("Reactions (kN)"):
                rxn_rows = []
                for nid, R in res.reactions.items():
                    rxn_rows.append([nid,
                                     f"{R[0]/1000:.2f}",
                                     f"{R[1]/1000:.2f}",
                                     f"{R[2]/1000:.2f}"])
                st.dataframe(pd.DataFrame(rxn_rows,
                                          columns=["Node","Rx","Ry","Rz"]),
                             use_container_width=True, height=280)

            with st.expander("Member-force envelope", expanded=True):
                env = envelope(ss.results, [e.id for e in geom.elements])
                env_rows = []
                for e in geom.elements:
                    env_rows.append([e.id, e.member_type, e.frame_id,
                                     f"{env[e.id]['N']/1000:.2f}",
                                     f"{env[e.id]['Vy']/1000:.2f}",
                                     f"{env[e.id]['Vz']/1000:.2f}",
                                     f"{env[e.id]['My']/1000:.2f}",
                                     f"{env[e.id]['Mz']/1000:.2f}"])
                st.dataframe(pd.DataFrame(env_rows,
                                          columns=["Elem","Type","Frame",
                                                   "N (kN)","Vy (kN)","Vz (kN)",
                                                   "My (kNm)","Mz (kNm)"]),
                             height=320, use_container_width=True)

            st.markdown("**Force diagrams**")
            from modules.visualization import fig_force_diagram
            diag_col1, diag_col2 = st.columns(2)
            with diag_col1:
                st.caption("Bending Moment — Mz")
                st.plotly_chart(
                    fig_force_diagram(geom, res, force_key="Mz", scale=0.5),
                    use_container_width=True)
            with diag_col2:
                st.caption("Shear Force — Vz")
                st.plotly_chart(
                    fig_force_diagram(geom, res, force_key="Vz", scale=0.5),
                    use_container_width=True)
            with st.expander("Other force components"):
                other_col1, other_col2, other_col3 = st.columns(3)
                with other_col1:
                    st.caption("Bending Moment — My")
                    st.plotly_chart(
                        fig_force_diagram(geom, res, force_key="My", scale=0.5),
                        use_container_width=True)
                with other_col2:
                    st.caption("Axial Force — N")
                    st.plotly_chart(
                        fig_force_diagram(geom, res, force_key="N", scale=0.5),
                        use_container_width=True)
                with other_col3:
                    st.caption("Shear Force — Vy")
                    st.plotly_chart(
                        fig_force_diagram(geom, res, force_key="Vy", scale=0.5),
                        use_container_width=True)

    # ================================================================
    # TAB 5 — DESIGN
    # ================================================================
    with tabs[4]:
        st.markdown("##### Design")
        st.caption("GLT member verification per EN 1995-1-1.")
        if not ss.results:
            st.warning("Run the analysis first (Analysis tab).")
        else:
            from modules.analysis import envelope

            geom = build_geometry()
            env = envelope(ss.results, [e.id for e in geom.elements])
            kmod = ss.project["kmod"]

            beams = [e for e in geom.elements if e.member_type == "rafter"]
            cols  = [e for e in geom.elements if e.member_type == "column"]

            beam_results = []
            for f_id in range(geom.n_bays + 1):
                frame_rafters = [e for e in beams if e.frame_id == f_id]
                if not frame_rafters: continue
                worst = max(frame_rafters,
                            key=lambda e: max(env[e.id]["My"], env[e.id]["Mz"]))
                L = math.hypot(geom.L_a/2.0, geom.H_f - geom.H_s)
                w_inst_mm = 0.0
                if "SLS_char" in ss.results:
                    res = ss.results["SLS_char"]
                    ys = {geom.nodes[e.j_node-1].id for e in frame_rafters}
                    ys |= {geom.nodes[e.i_node-1].id for e in frame_rafters}
                    wmax = max(abs(res.node_disp[nid][2]) for nid in ys)
                    w_inst_mm = wmax * 1000.0
                Mmax = max(env[worst.id]["My"], env[worst.id]["Mz"])
                Vmax = max(env[worst.id]["Vy"], env[worst.id]["Vz"])
                Nmax = env[worst.id]["N"]
                beam_results.append(
                    design_beam(label=f"Rafter (Frame {f_id})",
                                sec=worst.section, grade=worst.section.grade,
                                L=L, L_ef=L,
                                M_Ed_kNm=Mmax/1000.0,
                                V_Ed_kN=Vmax/1000.0,
                                N_Ed_kN=Nmax/1000.0,
                                kmod=kmod, w_inst_mm=w_inst_mm,
                                deflection_limit_ratio=300.0))

            col_results = []
            for f_id in range(geom.n_bays + 1):
                frame_cols = [e for e in cols if e.frame_id == f_id]
                if not frame_cols: continue
                worst = max(frame_cols, key=lambda e: env[e.id]["N"])
                L = geom.H_s
                N    = env[worst.id]["N"]    / 1000.0
                Mz   = env[worst.id]["Mz"]   / 1000.0
                My   = env[worst.id]["My"]   / 1000.0
                col_results.append(
                    design_column(label=f"Column (Frame {f_id})",
                                  sec=worst.section, grade=worst.section.grade,
                                  L=L, L_eff_y=L, L_eff_z=L,
                                  N_Ed_kN=N, M_y_Ed_kNm=My, M_z_Ed_kNm=Mz,
                                  kmod=kmod))

            ss.design_results = {"beams": beam_results, "columns": col_results}

            st.markdown("**Beam (rafter) checks**")
            st.dataframe(pd.DataFrame([{
                "Member": r.member_label,
                "Sec (mm)": f"{r.section.h_mm:.0f}×{r.section.b_mm:.0f}",
                "Grade": r.grade,
                "M_Ed": round(r.M_Ed,2),
                "V_Ed": round(r.V_Ed,2),
                "Bend": round(r.bending_UR,2),
                "Shear": round(r.shear_UR,2),
                "LTB": round(r.LTB_UR,2),
                "Defl": round(r.deflection_UR,2),
                "Result": "✅" if r.pass_fail else "❌",
            } for r in beam_results]), use_container_width=True, height=200)

            st.markdown("**Column checks**")
            st.dataframe(pd.DataFrame([{
                "Member": r.member_label,
                "Sec (mm)": f"{r.section.h_mm:.0f}×{r.section.b_mm:.0f}",
                "Grade": r.grade,
                "N_Ed": round(r.N_Ed,2),
                "M_y": round(r.M_y_Ed,2),
                "M_z": round(r.M_z_Ed,2),
                "λ_y": round(r.lambda_rel_y,2),
                "λ_z": round(r.lambda_rel_z,2),
                "UR(6.23)": round(r.UR_y,2),
                "UR(6.24)": round(r.UR_z,2),
                "Result": "✅" if r.pass_fail else "❌",
            } for r in col_results]), use_container_width=True, height=200)

            n_pass = sum(1 for r in beam_results+col_results if r.pass_fail)
            n_tot  = len(beam_results) + len(col_results)
            if n_pass == n_tot:
                st.success(f"All {n_tot} member checks pass ✅")
            else:
                st.error(f"{n_tot - n_pass} of {n_tot} member checks fail ❌")

    # ================================================================
    # TAB 6 — REPORT
    # ================================================================
    with tabs[5]:
        st.markdown("##### Report")
        st.caption("Generate downloadable PDF report and Excel results.")
        if not ss.design_results:
            st.warning("Run analysis (Analysis tab) and design (Design tab) first.")
        else:
            from modules.report import generate_report
            geom = build_geometry()
            out_dir = Path("outputs"); out_dir.mkdir(exist_ok=True)

            cb1, cb2 = st.columns(2)
            with cb1:
                if st.button("📄 Generate PDF", type="primary",
                             use_container_width=True):
                    pdf_path = out_dir / f"PortalFrame_Report_{datetime.now():%Y%m%d_%H%M%S}.pdf"
                    generate_report(str(pdf_path), ss.project, geom,
                                    ss.surface_loads,
                                    ss.design_results["beams"],
                                    ss.design_results["columns"])
                    with open(pdf_path, "rb") as f:
                        st.download_button("⬇ Download PDF", f.read(),
                                           file_name=pdf_path.name,
                                           mime="application/pdf",
                                           use_container_width=True)
                    st.success(f"Saved {pdf_path.name}")

            with cb2:
                if st.button("📊 Generate Excel",
                             use_container_width=True):
                    xlsx_path = out_dir / f"PortalFrame_Results_{datetime.now():%Y%m%d_%H%M%S}.xlsx"
                    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
                        pd.DataFrame([ss.geom_params]).T.to_excel(
                            xw, sheet_name="Geometry")
                        pd.DataFrame([ss.project]).T.to_excel(
                            xw, sheet_name="Project")
                        pd.DataFrame([ss.surface_loads]).T.to_excel(
                            xw, sheet_name="SurfaceLoads")
                        pd.DataFrame([r.__dict__ for r in ss.design_results["beams"]]
                            ).drop(columns=["section","notes"], errors="ignore"
                            ).to_excel(xw, sheet_name="Beams", index=False)
                        pd.DataFrame([r.__dict__ for r in ss.design_results["columns"]]
                            ).drop(columns=["section","notes"], errors="ignore"
                            ).to_excel(xw, sheet_name="Columns", index=False)
                    with open(xlsx_path, "rb") as f:
                        st.download_button("⬇ Download Excel", f.read(),
                                           file_name=xlsx_path.name,
                                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                           use_container_width=True)
                    st.success(f"Saved {xlsx_path.name}")
