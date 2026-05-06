"""
Load take-down / load linking for a single-story 3-bay portal frame.

Engineering logic (one-way slab assumption, per the Excel sample's scope):

    Input:  Surface loads on the roof  (kN/m²)
              - permanent G  (self-weight of cladding / sheathing / membrane)
              - imposed Q    (snow, maintenance, etc.)
              - wind W       (suction / pressure - applied perpendicular to roof)
            Self-weight of structural members is computed automatically from
            the GLT density and member cross-sections.

    Step 1: Convert surface loads to LINE LOADS on the purlins.
            Tributary width of a purlin = half-distance to the neighbouring purlins.
            Edge purlins (eaves, ridge) get half the interior tributary width.
            -> w_purlin (kN/m) = q (kN/m²) * t_purlin (m)

    Step 2: Convert purlin reactions to LINE LOADS on the rafters.
            Because purlins span between adjacent frames (length = a),
            each purlin delivers point loads at its end nodes equal to
            (w_purlin * a / 2) onto the rafter node.
            For a continuous-purlin idealisation we instead distribute the
            purlin loads as an equivalent UDL on the rafter:
                w_rafter (kN/m_along_rafter) = q (kN/m²) * a   (interior frame)
                w_rafter_edge                = q (kN/m²) * a/2 (edge frame)

    Step 3: Rafter reactions go into the eaves (knee) node, which the column
            transfers to the foundation.

This module returns load dictionaries the analysis module can consume directly.
Only line loads on line elements are produced -- which matches the user
requirement to model elements as line elements with line loads.

Load-case keys: 'G' (permanent), 'Q' (imposed/snow), 'W' (wind).
ULS combo (EN 1990 6.10): 1.35 G + 1.5 Q + 1.5*0.6 W (default).
SLS char: G + Q + 0.6 W.
"""
from dataclasses import dataclass, field
from typing import Dict, List
import math

from .geometry import FrameGeometry, Element
from .materials import GLT_GRADES


# Eurocode partial / combination factors (default for residential / category A)
PSI_0 = {"Q": 0.7, "W": 0.6}   # combination factor
GAMMA_G = 1.35
GAMMA_Q = 1.50

# ------------------------------------------------------------------
@dataclass
class SurfaceLoads:
    g_roof: float = 0.50   # kN/m^2 permanent (cladding+insulation, excl. structural self-weight)
    q_snow: float = 1.00   # kN/m^2 snow (characteristic, on plan)
    w_wind: float = 0.50   # kN/m^2 wind suction (perpendicular to roof, characteristic)


@dataclass
class ElementLoad:
    """A uniformly distributed load on a single line element, in kN/m, in global axes.
       The convention used by the analysis module: (wx, wy, wz) global components."""
    element_id: int
    wx: float = 0.0
    wy: float = 0.0
    wz: float = 0.0
    case: str = "G"


@dataclass
class LoadModel:
    surface: SurfaceLoads
    cases: Dict[str, List[ElementLoad]] = field(default_factory=dict)

    def add(self, case: str, load: ElementLoad):
        self.cases.setdefault(case, []).append(load)


# ------------------------------------------------------------------
def _self_weight_kN_per_m(elem: Element) -> float:
    rho = GLT_GRADES[elem.section.grade]["rho_mean"]   # kg/m^3
    return rho * 9.81 * elem.section.A / 1000.0        # kN/m


def _purlin_tributary_widths(geom: FrameGeometry) -> Dict[str, float]:
    """Tributary width along the slope for each purlin label (m).
       Purlin labels are evenly spaced between eaves and ridge."""
    n_p = geom.n_purlins_per_slope
    # rafter slope length
    L_slope = math.hypot(geom.L_a / 2.0, geom.H_f - geom.H_s)
    spacing = L_slope / (n_p + 1)
    widths = {}
    # eaves & ridge edge purlins -> half spacing (one-sided tributary)
    for side in ("L", "R"):
        widths[f"eaves_{side}"] = spacing / 2.0
    widths["ridge"] = spacing   # ridge serves both slopes -> spacing/2 from each side
    for k in range(1, n_p + 1):
        for side in ("L", "R"):
            widths[f"raft_{side}_{k}"] = spacing
    return widths


def build_loads(geom: FrameGeometry, surface: SurfaceLoads,
                include_selfweight: bool = True) -> LoadModel:
    """Generate line-load model on every element from surface loads + self-weight."""
    LM = LoadModel(surface=surface, cases={"G": [], "Q": [], "W": []})

    # ---- Self-weight: gravity on every element (G case) ----
    if include_selfweight:
        for e in geom.elements:
            sw = _self_weight_kN_per_m(e)   # kN/m, acting in -Z global
            LM.add("G", ElementLoad(e.id, 0.0, 0.0, -sw, "G"))

    # ---- Roof loads ----
    # We distribute roof surface loads onto the rafters as line loads (kN/m along rafter axis),
    # using tributary frame-spacing.  This is exact for one-way action over purlins ->
    # rafters when purlins are continuous; it is the standard simplification when
    # a sub-frame analysis of purlins isn't required.
    # Tributary depth (along building length B) for a frame:
    #     interior frame: a   (half spacing each side)
    #     end frame:      a/2
    n_frames = geom.n_bays + 1

    # Surface load on plan -> on rafter slope: kN/m of rafter = q (kN/m^2) * trib (m).
    # The vertical loads (G, Q on plan) act in -Z; we split into projected components later.
    # We give analysis the global (wx, wy, wz). For UDL on a rafter the resultant is in -Z.
    for e in geom.elements:
        if e.member_type != "rafter":
            continue

        # tributary width = a (interior) or a/2 (edge frame)
        is_edge = e.frame_id in (0, n_frames - 1)
        trib = geom.a / 2.0 if is_edge else geom.a

        # G (cladding) — vertical, kN/m of rafter, acting -Z
        wG = surface.g_roof * trib
        if wG > 0:
            LM.add("G", ElementLoad(e.id, 0.0, 0.0, -wG, "G"))

        # Q (snow on plan) — vertical -Z; load is given on plan area, so multiply by trib
        # and project onto rafter length. Acting on rafter line (in plan view per m of rafter
        # the plan-tributary length is cos(alpha) * 1m of rafter), so:
        #     w_rafter_Q = q_snow * trib * cos(alpha)   (kN/m of rafter, acting -Z)
        L_slope = math.hypot(geom.L_a/2.0, geom.H_f - geom.H_s)
        cos_a = (geom.L_a/2.0) / L_slope
        wQ = surface.q_snow * trib * cos_a
        if wQ > 0:
            LM.add("Q", ElementLoad(e.id, 0.0, 0.0, -wQ, "Q"))

        # W (wind) — perpendicular to roof surface; resolve to global X/Y/Z
        # Take suction (negative pressure -> uplift). Direction: outward normal of roof.
        # For symmetric pitched roof, normal of left slope: (0, -sin(α), +cos(α))? Actually
        # left slope rises from y=0 to y=L_a/2; outward normal points away from inside,
        # i.e. up-and-left. Interior pressure -> outward; suction -> we keep it as outward.
        # For simplicity, apply uplift component in +Z (suction lifts roof).
        sin_a = (geom.H_f - geom.H_s) / L_slope
        # Determine slope side from element y midpoint
        ni = geom.nodes[e.i_node - 1]
        nj = geom.nodes[e.j_node - 1]
        y_mid = 0.5 * (ni.y + nj.y)
        side_sign = -1.0 if y_mid < geom.L_a / 2.0 else +1.0   # left slope -> -y normal
        wW_total = surface.w_wind * trib   # kN/m of rafter, perpendicular to slope (suction -> outward)
        # outward normal components (suction = positive outward = uplift in z, sideways in y)
        # For uplift we want +Z; for left slope normal points (0, -cos_a, +sin_a)? Re-derive:
        # rafter goes from (x, 0, H_s) to (x, L_a/2, H_f); slope tangent t = (0, cos_a, sin_a)
        # frame-perpendicular = X axis (1,0,0); normal in YZ plane = (0, -sin_a, cos_a) for left slope
        # right slope tangent (0, -cos_a, sin_a) -> normal (0, +sin_a, cos_a)
        ny = -sin_a * (1.0 if side_sign < 0 else -1.0)   # left=-sin_a, right=+sin_a
        nz =  cos_a
        # As suction (outward), force on roof is in -normal direction for pressure,
        # +normal for suction. Use suction (uplift) by default:
        wWy = wW_total * ny
        wWz = wW_total * nz
        if abs(wW_total) > 0:
            LM.add("W", ElementLoad(e.id, 0.0, wWy, wWz, "W"))

    return LM


# ------------------------------------------------------------------
def combine(LM: LoadModel,
            combo: str = "ULS",
            psi_q: float = PSI_0["Q"],
            psi_w: float = PSI_0["W"]) -> List[ElementLoad]:
    """Return a flat list of factored element loads for a given combo.
       combo: 'ULS', 'SLS_char', 'SLS_qp', 'G_only', 'Q_only', 'W_only'."""
    out = []

    def _scaled(case_loads, factor):
        return [ElementLoad(L.element_id, L.wx*factor, L.wy*factor, L.wz*factor, L.case)
                for L in case_loads]

    G = LM.cases.get("G", [])
    Q = LM.cases.get("Q", [])
    W = LM.cases.get("W", [])

    if combo == "ULS":
        out += _scaled(G, GAMMA_G)
        out += _scaled(Q, GAMMA_Q)
        out += _scaled(W, GAMMA_Q * psi_w)
    elif combo == "SLS_char":
        out += _scaled(G, 1.0)
        out += _scaled(Q, 1.0)
        out += _scaled(W, psi_w)
    elif combo == "SLS_qp":
        out += _scaled(G, 1.0)
        out += _scaled(Q, 0.3)         # psi2 for snow (alt < 1000 m)
        out += _scaled(W, 0.0)
    elif combo == "G_only":
        out += _scaled(G, 1.0)
    elif combo == "Q_only":
        out += _scaled(Q, 1.0)
    elif combo == "W_only":
        out += _scaled(W, 1.0)
    else:
        raise ValueError(f"Unknown combo {combo}")
    return out
