"""
3D linear-elastic frame analysis — pure NumPy direct stiffness method.
Drop-in replacement for the OpenSeesPy-based version; no C extensions needed.

Unit system:
  Geometry  : m
  E, G      : Pa  (MPa × 10⁶, as stored in materials.py)
  Loads in  : kN/m  (from load_takedown.py) — converted × 1000 → N/m internally
  Outputs   : node_disp [m], elem_forces [N / N·m], reactions [N]
"""
from dataclasses import dataclass, field
from typing import Dict, List
import numpy as np

from .geometry import FrameGeometry
from .load_takedown import ElementLoad
from .materials import GLT_GRADES


@dataclass
class AnalysisResults:
    combo_name: str
    node_disp:   Dict[int, np.ndarray] = field(default_factory=dict)  # node_id -> 6-dof
    elem_forces: Dict[int, np.ndarray] = field(default_factory=dict)  # elem_id -> 12 local forces
    reactions:   Dict[int, np.ndarray] = field(default_factory=dict)  # node_id -> 6-dof


# ---------------------------------------------------------------------------
def _local_axes(ni, nj, local_z):
    """3×3 R where columns = local unit vectors [ex, ey, ez] in global coords; also returns L."""
    ex = np.array([nj.x - ni.x, nj.y - ni.y, nj.z - ni.z])
    L = float(np.linalg.norm(ex))
    if L < 1e-12:
        raise ValueError("Zero-length element")
    ex /= L
    ez0 = np.array(local_z, dtype=float)
    ez = ez0 - np.dot(ez0, ex) * ex
    if np.linalg.norm(ez) < 1e-9:
        ez0 = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(ez0, ex)) > 0.99:
            ez0 = np.array([0.0, 1.0, 0.0])
        ez = ez0 - np.dot(ez0, ex) * ex
    ez /= np.linalg.norm(ez)
    ey = np.cross(ez, ex)
    return np.column_stack([ex, ey, ez]), L


def _local_stiffness(E, G, A, Iy, Iz, J, L):
    """12×12 Euler-Bernoulli 3-D beam stiffness in local coordinates.

    DOF order per node: [u, v, w, θx, θy, θz]
      u  = axial        Iy = 2nd moment about local y (strong if h > b)
      v  = trans-y      Iz = 2nd moment about local z (weak)
      w  = trans-z      J  = St-Venant torsion constant
      θy sign convention: θy = −dw/dx  (right-hand rule, matches OpenSees)
    """
    EA   = E * A / L
    GJL  = G * J / L
    L2, L3 = L * L, L * L * L

    iy1 = 12*E*Iy/L3;  iy2 = 6*E*Iy/L2;  iy3 = 4*E*Iy/L;  iy4 = 2*E*Iy/L
    iz1 = 12*E*Iz/L3;  iz2 = 6*E*Iz/L2;  iz3 = 4*E*Iz/L;  iz4 = 2*E*Iz/L

    k = np.zeros((12, 12))

    # Axial  (dofs 0, 6)
    k[0, 0] =  EA;  k[0, 6] = -EA
    k[6, 0] = -EA;  k[6, 6] =  EA

    # Torsion  (dofs 3, 9)
    k[3, 3] =  GJL;  k[3, 9] = -GJL
    k[9, 3] = -GJL;  k[9, 9] =  GJL

    # Bending about z — v and θz  (dofs 1, 5, 7, 11)
    k[1,  1] =  iz1;  k[1,  5] =  iz2;  k[1,  7] = -iz1;  k[1, 11] =  iz2
    k[5,  1] =  iz2;  k[5,  5] =  iz3;  k[5,  7] = -iz2;  k[5, 11] =  iz4
    k[7,  1] = -iz1;  k[7,  5] = -iz2;  k[7,  7] =  iz1;  k[7, 11] = -iz2
    k[11, 1] =  iz2;  k[11, 5] =  iz4;  k[11, 7] = -iz2;  k[11,11] =  iz3

    # Bending about y — w and θy  (dofs 2, 4, 8, 10)
    # coupling terms are negative due to θy = −dw/dx convention
    k[2,  2] =  iy1;  k[2,  4] = -iy2;  k[2,  8] = -iy1;  k[2, 10] = -iy2
    k[4,  2] = -iy2;  k[4,  4] =  iy3;  k[4,  8] =  iy2;  k[4, 10] =  iy4
    k[8,  2] = -iy1;  k[8,  4] =  iy2;  k[8,  8] =  iy1;  k[8, 10] =  iy2
    k[10, 2] = -iy2;  k[10, 4] =  iy4;  k[10, 8] =  iy2;  k[10,10] =  iy3

    return k


def _transform(R):
    """12×12 block-diagonal transformation  d_local = T @ d_global,  T = diag(Rᵀ ×4)."""
    T = np.zeros((12, 12))
    RT = R.T
    for b in range(4):
        s = 3 * b
        T[s:s+3, s:s+3] = RT
    return T


def _fixed_end_forces(wx_l, wy_l, wz_l, L):
    """Equivalent nodal forces for UDL in local coords [N/m → N, N·m].
    Returns 12-vector: [Fx,Fy,Fz,Tx,My,Mz] at i then j.
    """
    L2 = L * L
    return np.array([
        wx_l*L/2,  wy_l*L/2,  wz_l*L/2,  0.0, -wz_l*L2/12,  wy_l*L2/12,
        wx_l*L/2,  wy_l*L/2,  wz_l*L/2,  0.0,  wz_l*L2/12, -wy_l*L2/12,
    ])


# ---------------------------------------------------------------------------
def run_combo(geom: FrameGeometry, loads: List[ElementLoad],
              combo_name: str = "Combo") -> AnalysisResults:
    """Assemble and solve the 3-D frame for a single set of factored line loads."""
    n_dof    = 6 * len(geom.nodes)
    node_idx = {n.id: i for i, n in enumerate(geom.nodes)}
    elem_map = {e.id: e for e in geom.elements}

    K = np.zeros((n_dof, n_dof))
    F = np.zeros(n_dof)

    # ---------- global stiffness matrix ----------
    cache = {}   # elem_id -> (T, K_loc, dofs)
    for e in geom.elements:
        ni = geom.nodes[node_idx[e.i_node]]
        nj = geom.nodes[node_idx[e.j_node]]
        R, L = _local_axes(ni, nj, e.local_z)
        mat  = GLT_GRADES[e.section.grade]
        E    = mat["E_0_mean"] * 1.0e6   # MPa → Pa
        G    = mat["G_mean"]   * 1.0e6
        K_loc = _local_stiffness(E, G, e.section.A, e.section.Iy,
                                 e.section.Iz, e.section.J, L)
        T     = _transform(R)
        K_glo = T.T @ K_loc @ T
        dofs  = ([6*node_idx[e.i_node] + k for k in range(6)] +
                 [6*node_idx[e.j_node] + k for k in range(6)])
        K[np.ix_(dofs, dofs)] += K_glo
        cache[e.id] = (T, K_loc, dofs)

    # ---------- equivalent nodal forces from distributed loads ----------
    ld_by_elem: Dict[int, List[ElementLoad]] = {}
    for ld in loads:
        ld_by_elem.setdefault(ld.element_id, []).append(ld)

    fef_local: Dict[int, np.ndarray] = {}
    for eid, ld_list in ld_by_elem.items():
        e   = elem_map[eid]
        ni  = geom.nodes[node_idx[e.i_node]]
        nj  = geom.nodes[node_idx[e.j_node]]
        R, L = _local_axes(ni, nj, e.local_z)
        T, _, dofs = cache[eid]

        # loads are in kN/m — convert to N/m for SI consistency with E in Pa
        wx_g = sum(ld.wx for ld in ld_list) * 1000.0
        wy_g = sum(ld.wy for ld in ld_list) * 1000.0
        wz_g = sum(ld.wz for ld in ld_list) * 1000.0

        w_loc = R.T @ np.array([wx_g, wy_g, wz_g])   # global → local
        fef   = _fixed_end_forces(*w_loc, L)           # N, N·m
        fef_local[eid] = fef
        F[dofs] += T.T @ fef                           # transform to global

    # ---------- boundary conditions ----------
    fixed = set()
    for n in geom.nodes:
        ni = node_idx[n.id]
        for k in range(6):
            if n.fixity[k]:
                fixed.add(6*ni + k)
    free = [d for d in range(n_dof) if d not in fixed]

    # ---------- solve ----------
    u = np.zeros(n_dof)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])

    # ---------- harvest ----------
    res = AnalysisResults(combo_name=combo_name)

    for n in geom.nodes:
        ni = node_idx[n.id]
        res.node_disp[n.id] = u[6*ni:6*ni+6].copy()
        if any(n.fixity):
            r = np.zeros(6)
            for k in range(6):
                if n.fixity[k]:
                    dof = 6*ni + k
                    # reaction = K_row @ u − F_ext (FEF component at support)
                    r[k] = K[dof, :] @ u - F[dof]
            res.reactions[n.id] = r

    for e in geom.elements:
        T, K_loc, dofs = cache[e.id]
        d_loc = T @ u[dofs]
        fef   = fef_local.get(e.id, np.zeros(12))
        res.elem_forces[e.id] = K_loc @ d_loc - fef

    return res


# ---------------------------------------------------------------------------
def envelope(results_by_combo: Dict[str, AnalysisResults],
             elem_ids: List[int]) -> Dict[int, Dict[str, float]]:
    """Max-absolute-value envelope across all supplied combos."""
    env: Dict[int, Dict[str, float]] = {}
    for eid in elem_ids:
        N = Vy = Vz = My = Mz = T = 0.0
        for res in results_by_combo.values():
            f = res.elem_forces.get(eid)
            if f is None:
                continue
            Ni, Vyi, Vzi, Ti, Myi, Mzi = f[0], f[1], f[2], f[3], f[4], f[5]
            Nj, Vyj, Vzj, Tj, Myj, Mzj = f[6], f[7], f[8], f[9], f[10], f[11]
            N  = max(N,  abs(Ni), abs(Nj))
            Vy = max(Vy, abs(Vyi), abs(Vyj))
            Vz = max(Vz, abs(Vzi), abs(Vzj))
            T  = max(T,  abs(Ti),  abs(Tj))
            My = max(My, abs(Myi), abs(Myj))
            Mz = max(Mz, abs(Mzi), abs(Mzj))
        env[eid] = {"N": N, "Vy": Vy, "Vz": Vz, "My": My, "Mz": Mz, "T": T}
    return env
