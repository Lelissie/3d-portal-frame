"""
3D visualization for the parametric portal frame.

Two views:
    - 'structural': line elements as colored lines + node markers + supports + loads
    - 'architectural': solid extruded cross-sections (boxes) along each line element,
                       roof plane, and column volumes.
Result overlays:
    - deformed shape (scaled)
    - axial force / bending-moment diagrams along selected members
    - utilization ratio coloring (red-amber-green)
"""
from typing import Dict, List
import numpy as np
import plotly.graph_objects as go

from .geometry import FrameGeometry, Element, Node


_COLOR = {
    "column": "#2E7D32",
    "rafter": "#1565C0",
    "purlin": "#FF8F00",
    "support": "#212121",
    "load": "#D81B60",
    "deformed": "#E91E63",
    "ridge": "#5D4037",
}


# --------------------------------------------------------------------
def _rect_box_mesh(p_i: np.ndarray, p_j: np.ndarray,
                   b: float, h: float, local_z: np.ndarray):
    """Return arrays (x, y, z, i, j, k) for a Plotly Mesh3d rectangular prism
       extruded along axis p_i -> p_j with width b (local-z) and depth h (local-y)."""
    ex = p_j - p_i
    L = np.linalg.norm(ex)
    if L < 1e-12:
        return None
    ex /= L
    ez = local_z - np.dot(local_z, ex) * ex
    if np.linalg.norm(ez) < 1e-9:
        ez = np.array([0, 0, 1.0]) if abs(ex[2]) < 0.99 else np.array([1.0, 0, 0])
        ez -= np.dot(ez, ex) * ex
    ez /= np.linalg.norm(ez)
    ey = np.cross(ez, ex)

    # 8 corners
    corners = []
    for sign_along in (0.0, 1.0):
        for sign_y in (-0.5, +0.5):
            for sign_z in (-0.5, +0.5):
                p = p_i + sign_along * (p_j - p_i) + sign_y * h * ey + sign_z * b * ez
                corners.append(p)
    P = np.array(corners)
    # Triangulation of a box (12 triangles, 6 faces)
    faces = [
        (0,1,3),(0,3,2),     # i face
        (4,6,7),(4,7,5),     # j face
        (0,4,5),(0,5,1),     # bottom
        (2,3,7),(2,7,6),     # top
        (0,2,6),(0,6,4),     # -y
        (1,5,7),(1,7,3),     # +y
    ]
    return P, faces


def _add_box(fig, p_i, p_j, b, h, local_z, color, name=None, opacity=0.85):
    out = _rect_box_mesh(np.array(p_i), np.array(p_j), b, h, np.array(local_z))
    if out is None: return
    P, faces = out
    fig.add_trace(go.Mesh3d(
        x=P[:,0], y=P[:,1], z=P[:,2],
        i=[f[0] for f in faces], j=[f[1] for f in faces], k=[f[2] for f in faces],
        color=color, opacity=opacity, name=name or "", showlegend=False,
        hoverinfo="name",
    ))


# --------------------------------------------------------------------
def fig_structural(geom: FrameGeometry, deformed: Dict[int, np.ndarray] = None,
                   scale: float = 100.0,
                   utilization: Dict[int, float] = None,
                   show_loads: bool = False) -> go.Figure:
    fig = go.Figure()

    # nodes
    xs = [n.x for n in geom.nodes]
    ys = [n.y for n in geom.nodes]
    zs = [n.z for n in geom.nodes]
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=zs, mode="markers",
        marker=dict(size=3, color="#424242"), name="nodes",
        text=[f"N{n.id}" for n in geom.nodes], hoverinfo="text"
    ))

    # supports
    sup_x, sup_y, sup_z = [], [], []
    for n in geom.nodes:
        if any(n.fixity):
            sup_x.append(n.x); sup_y.append(n.y); sup_z.append(n.z)
    if sup_x:
        fig.add_trace(go.Scatter3d(
            x=sup_x, y=sup_y, z=sup_z, mode="markers",
            marker=dict(size=8, color=_COLOR["support"], symbol="diamond"),
            name="supports"))

    # elements (lines colored by member type or utilization)
    by_type = {}
    for e in geom.elements:
        by_type.setdefault(e.member_type, []).append(e)

    for mt, elems in by_type.items():
        line_x, line_y, line_z, hover = [], [], [], []
        for e in elems:
            ni = geom.nodes[e.i_node - 1]; nj = geom.nodes[e.j_node - 1]
            line_x += [ni.x, nj.x, None]
            line_y += [ni.y, nj.y, None]
            line_z += [ni.z, nj.z, None]
            ur = utilization.get(e.id) if utilization else None
            tag = (f"{mt} #{e.id} | {e.section.name} | {e.section.grade}"
                   + (f" | UR={ur:.2f}" if ur is not None else ""))
            hover += [tag, tag, ""]
        # color by UR if provided, else by type
        if utilization:
            line_color = _ur_to_color(np.mean([utilization.get(e.id, 0.0) for e in elems]))
        else:
            line_color = _COLOR.get(mt, "#000")
        fig.add_trace(go.Scatter3d(
            x=line_x, y=line_y, z=line_z, mode="lines",
            line=dict(color=line_color, width=6),
            name=mt, text=hover, hoverinfo="text"))

    # deformed shape
    if deformed is not None and len(deformed) > 0:
        for e in geom.elements:
            ni = geom.nodes[e.i_node - 1]; nj = geom.nodes[e.j_node - 1]
            di = deformed.get(ni.id, np.zeros(6))
            dj = deformed.get(nj.id, np.zeros(6))
            fig.add_trace(go.Scatter3d(
                x=[ni.x + scale*di[0], nj.x + scale*dj[0]],
                y=[ni.y + scale*di[1], nj.y + scale*dj[1]],
                z=[ni.z + scale*di[2], nj.z + scale*dj[2]],
                mode="lines",
                line=dict(color=_COLOR["deformed"], width=3, dash="dot"),
                name="deformed", showlegend=(e.id == geom.elements[0].id),
                hoverinfo="skip"))

    _layout(fig, "Structural Model")
    return fig


def fig_architectural(geom: FrameGeometry) -> go.Figure:
    fig = go.Figure()
    _MT_COLOR = {"column": "#A1887F", "rafter": "#8D6E63",
                 "purlin": "#D7CCC8", "ridge": "#BCAAA4"}

    # Merge all boxes of the same member type into one Mesh3d trace
    # so Plotly only handles 3-4 traces instead of one per element.
    by_type: dict = {}
    for e in geom.elements:
        ni = geom.nodes[e.i_node - 1]; nj = geom.nodes[e.j_node - 1]
        out = _rect_box_mesh(np.array([ni.x, ni.y, ni.z]),
                             np.array([nj.x, nj.y, nj.z]),
                             e.section.b, e.section.h, np.array(e.local_z))
        if out is None:
            continue
        P, faces = out
        by_type.setdefault(e.member_type, ([], []))
        verts, tris = by_type[e.member_type]
        offset = sum(len(v) for v in verts)  # total vertices so far
        verts.append(P)
        tris.extend([(f[0]+offset, f[1]+offset, f[2]+offset) for f in faces])

    for mt, (verts, tris) in by_type.items():
        if not verts:
            continue
        P = np.vstack(verts)
        fig.add_trace(go.Mesh3d(
            x=P[:, 0], y=P[:, 1], z=P[:, 2],
            i=[t[0] for t in tris], j=[t[1] for t in tris], k=[t[2] for t in tris],
            color=_MT_COLOR.get(mt, "#BCAAA4"), opacity=0.92,
            name=mt, showlegend=True, hoverinfo="name",
        ))

    # roof skin (semi-transparent panels per bay between adjacent frames)
    n_frames = geom.n_bays + 1
    fn = geom._frame_nodes
    n_p = geom.n_purlins_per_slope
    chain_L = ["eaves_L"] + [f"raft_L_{k}" for k in range(1, n_p+1)] + ["ridge"]
    chain_R = ["eaves_R"] + [f"raft_R_{k}" for k in range(1, n_p+1)] + ["ridge"]
    for f in range(n_frames - 1):
        for chain in (chain_L, chain_R):
            for k in range(len(chain) - 1):
                a = geom.nodes[fn[f][chain[k]] - 1]
                b = geom.nodes[fn[f][chain[k+1]] - 1]
                c = geom.nodes[fn[f+1][chain[k+1]] - 1]
                d = geom.nodes[fn[f+1][chain[k]] - 1]
                fig.add_trace(go.Mesh3d(
                    x=[a.x, b.x, c.x, d.x],
                    y=[a.y, b.y, c.y, d.y],
                    z=[a.z, b.z, c.z, d.z],
                    i=[0, 0], j=[1, 2], k=[2, 3],
                    color="#FFB74D", opacity=0.35, name="roof", showlegend=False,
                    hoverinfo="skip"))

    _layout(fig, "Architectural Model")
    return fig


def fig_force_diagram(geom: FrameGeometry, results, force_key: str = "Mz",
                      scale: float = 0.5) -> go.Figure:
    """Plot member end forces as a sticks diagram alongside elements.
       force_key: 'N','Vy','Vz','My','Mz','T' (uses index in elem_forces array)."""
    idx_map = {"N":0, "Vy":1, "Vz":2, "T":3, "My":4, "Mz":5}
    idx = idx_map[force_key]

    fig = fig_structural(geom)

    # find reference for scaling
    fmax = 1e-9
    for f in results.elem_forces.values():
        fmax = max(fmax, abs(f[idx]), abs(f[idx+6]))

    for e in geom.elements:
        f = results.elem_forces.get(e.id)
        if f is None: continue
        ni = geom.nodes[e.i_node - 1]; nj = geom.nodes[e.j_node - 1]
        # local axes
        ex = np.array([nj.x - ni.x, nj.y - ni.y, nj.z - ni.z]); L = np.linalg.norm(ex); ex /= L
        ez0 = np.array(e.local_z); ez = ez0 - np.dot(ez0, ex)*ex; ez /= np.linalg.norm(ez)
        ey = np.cross(ez, ex)
        # offset direction depends on which force component
        offset_dir = ey if force_key in ("Vy","Mz") else ez
        Fi = f[idx]; Fj = -f[idx+6]   # j-end sign flipped for end-action convention
        s = scale / fmax
        pi = np.array([ni.x, ni.y, ni.z])
        pj = np.array([nj.x, nj.y, nj.z])
        pi_o = pi + Fi*s*offset_dir
        pj_o = pj + Fj*s*offset_dir
        fig.add_trace(go.Scatter3d(
            x=[pi.x if False else pi[0], pi_o[0], pj_o[0], pj[0]],
            y=[pi[1], pi_o[1], pj_o[1], pj[1]],
            z=[pi[2], pi_o[2], pj_o[2], pj[2]],
            mode="lines",
            line=dict(color="#C2185B", width=3),
            name=force_key, showlegend=(e.id == geom.elements[0].id),
            hoverinfo="skip"))
    _layout(fig, f"{force_key} Diagram")
    return fig


# --------------------------------------------------------------------
def _ur_to_color(ur: float) -> str:
    if ur <= 0.5:  return "#2E7D32"
    if ur <= 0.85: return "#FBC02D"
    if ur <= 1.0:  return "#EF6C00"
    return "#C62828"


def _layout(fig: go.Figure, title: str):
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X (m) — building depth (B)",
            yaxis_title="Y (m) — span (L_a)",
            zaxis_title="Z (m) — height",
            aspectmode="data",
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.0)),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        height=620,
        legend=dict(orientation="h", yanchor="bottom", y=0.0, x=0.5, xanchor="center"),
    )
