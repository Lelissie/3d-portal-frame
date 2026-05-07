# 3D Portal Frame

A Streamlit web-app that performs **parametric 3D modelling, structural analysis, and Eurocode 5 design** of a single-story pitched portal frame in **glued laminated timber (GLT)**.

The UX is inspired by [BuildSpec](https://buildspec.app): a parameter sidebar on the left (sliders + grouped sections) drives a large interactive 3D viewport on the right, with a single radio toggle to switch between **Architectural Model** (default) and **Structural Model** views. Up to **10 bays** are supported; the default frame span is **12 m**.

The full workflow follows the same six steps as the load-linking template: Project Setting → Geometry → Loading → Analysis → Design → Report.

---

## 1. What the app does

| Stage | Engine / source |
|---|---|
| Parametric 3D model | Pure NumPy + Plotly (architectural view = extruded boxes; structural view = line elements) |
| Load take-down | Tributary-width method, one-way action (per the load-linking template's scope) |
| Analysis | **OpenSeesPy** — 3D linear-elastic frame, `elasticBeamColumn` line elements, pinned bases |
| Design — beams | EN 1995-1-1 §6.1.6 (bending) · §6.1.7 (shear, k_cr=0.67) · §6.3.3 (LTB, k_crit) · §7.2 deflections |
| Design — columns | EN 1995-1-1 §6.3.2 — combined bending + axial-compression with stability (eqs. 6.23 & 6.24) |
| Material values | EN 14080:2013 — GL24h, GL28h, GL32h |
| Report | Reportlab PDF + openpyxl Excel export |

## 2. Frame definition (matches the RFEM screenshot)

Symbol | Meaning
-------|--------
H | Building height to eaves
B | Building depth (= n_bays · a)
a | Frame (truss) spacing
L_a | Portal-frame span across the ridge
H_s | Eaves height
H_f | Ridge height
h_a | Cross-section height at footing
h_f | Cross-section height at ridge
h_1 | Cross-section height at frame joint (knee)
l_zw | Length of inserted wedge (haunch)

Coordinate system: **X** along building depth (B), **Y** across the span (L_a), **Z** up. Each portal frame lies in a plane of constant X.

## 3. Install & run

```bash
git clone <this-folder>
cd portal_frame_app
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Then open <http://localhost:8501> in your browser.

> **Note** OpenSeesPy is shipped as a wheel and is not currently published for every Python+OS combo. If `pip install openseespy` fails, use Python 3.9–3.12 on Windows / Linux / macOS-Intel as recommended on the project page. The rest of the app (geometry, loads, design, viz) works without it; only Page 4 needs OpenSees.

## 4. Project structure

```
portal_frame_app/
├── app.py                      # Streamlit UI — 6 pages
├── requirements.txt
├── README.md
├── modules/
│   ├── __init__.py
│   ├── materials.py            # GL24h / GL28h / GL32h, kmod, γ_M, kdef
│   ├── geometry.py             # FrameGeometry → nodes, line elements
│   ├── load_takedown.py        # Surface → rafter line loads, combo factors
│   ├── analysis.py             # OpenSeesPy 3D linear-elastic
│   ├── design_glt.py           # EC5 beam + column verifications
│   ├── visualization.py        # Plotly 3D — architectural + structural + diagrams
│   └── report.py               # PDF report generator
└── outputs/                    # auto-created for PDF / Excel exports
```

## 5. Engineering logic — load take-down

Following the template scope (one-way slab, UDL only, pinned bases, concentric column loading):

1. **Surface loads** are entered on Page 3 as G (permanent), Q (snow on plan), W (wind normal to slope).
2. They are converted to **line loads on each rafter** using the tributary frame-spacing (interior frame = a, end frames = a/2).
3. Snow (a load on plan area) is multiplied by `cos α` to get the load per metre of rafter; permanent G is per metre of rafter directly.
4. Wind (suction normal to roof) is resolved into global Y/Z components per slope side.
5. **Self-weight** of every line element is added automatically as `ρ_mean · g · A` (kN/m, –Z).
6. The analysis assembles `1.35 G + 1.5 Q + 1.5·ψ_0,W·W` for ULS (default), plus `G + Q + ψ_0,W·W` for SLS.
7. Forces from the rafter are transmitted into the eaves node and then to the foundation through the column — no extra hand-calculation needed; OpenSees does it automatically once the line loads are applied.

## 6. Design references

- **EN 1995-1-1:2004 + A2:2014** Eurocode 5: Design of timber structures — General — Common rules and rules for buildings
- **EN 14080:2013** Timber structures — Glued laminated timber — Requirements
- *Design of Timber Structures, Vol. 1*, Swedish Wood (free online)
- Eurocode-Applied online calculators ([eurocodeapplied.com/design/en1995](https://eurocodeapplied.com/design/en1995)) — used to cross-check the algorithm logic for k_c, k_crit, and the combined-action checks.

## 7. Limitations (from the template's scope)

- Pinned bases only (no fixed bases or rotational springs yet).
- Linear-elastic analysis (no second-order or P-Δ; for slender frames consider §5.4.4 amplifier on demand).
- Wind is applied as a normal pressure on the rafter only — gable wall loading is not modelled.
- Members are line elements with constant cross-section per element. The haunch profile is reflected in the design **demands** (max moments/shears) but a dedicated tapered-beam check (§6.4) is not part of this release.
- No connection design (roof–column, base, ridge).
- Service classes 1 and 2 supported (kdef table available; cumulative deflections w_creep can be added if needed).

## 8. How to extend

- **Floors / slabs**: Add a `floor.py` module that mirrors `load_takedown.py`'s rafter routine but for a CLT panel; transfer reactions to beams just as the Excel "Floor 1 → B1 → C1" chain does.
- **Multi-story**: Replicate the column-eaves chain at each level and link with floor diaphragms.
- **Tapered haunch design** (§6.4 EC5): in `design_glt.py` add `design_tapered_beam(...)` using `geom.haunched_depth(s, L)`.
- **Connections**: Use the env reactions on Page 4 directly as input to a pin/dowel design helper.
