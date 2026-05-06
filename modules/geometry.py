"""
Parametric 3D geometry generator for a single-story, 3-bay pitched portal frame.

Frame parameters mirror RFEM/Dlubal nomenclature:
    H   : building height to eaves (m)
    B   : building depth (along ridge) (m)
    a   : frame (truss) spacing along the building depth (m)  ->  3 bays => B = 3a
    L_a : frame (portal) span, gable-to-gable across the ridge (m)
    H_s : height on edge / eaves height (m)
    H_f : height in the middle / ridge height (m)
    h_a : cross-section height at footing (cm)
    h_f : cross-section height at ridge (cm)
    h_1 : cross-section height at frame joint / knee (cm) -- max depth of haunched zone
    l_zw: length of inserted wedge / haunch length (cm) along the rafter

Output: a Frame3D object containing nodes, line elements, and member metadata.
Coordinate system: X along the building depth (B), Y across the span (L_a), Z up.
Each portal frame lies in a plane of constant X.
"""
from dataclasses import dataclass, field
from typing import Literal
import numpy as np


@dataclass
class Section:
    name: str
    b_mm: float            # width (mm)
    h_mm: float            # depth (mm)
    grade: str             # GLT grade key

    @property
    def b(self) -> float: return self.b_mm / 1000.0
    @property
    def h(self) -> float: return self.h_mm / 1000.0
    @property
    def A(self) -> float: return self.b * self.h
    @property
    def Iy(self) -> float: return self.b * self.h**3 / 12.0    # bending about local y (strong)
    @property
    def Iz(self) -> float: return self.h * self.b**3 / 12.0    # bending about local z (weak)
    @property
    def J(self) -> float:
        # St. Venant torsion constant for solid rectangle (Roark)
        a = max(self.b, self.h) / 2.0
        bb = min(self.b, self.h) / 2.0
        return a * bb**3 * (16.0/3.0 - 3.36 * (bb/a) * (1 - (bb**4)/(12*a**4)))


@dataclass
class Node:
    id: int
    x: float
    y: float
    z: float
    fixity: tuple = (0, 0, 0, 0, 0, 0)   # 1 = restrained, 0 = free  (Ux,Uy,Uz,Rx,Ry,Rz)


@dataclass
class Element:
    id: int
    i_node: int
    j_node: int
    section: Section
    member_type: Literal["column", "rafter", "purlin", "wall_plate", "ridge"]
    frame_id: int = 0      # which portal frame (0..n_bays)
    bay_id: int = 0        # which bay along B
    # local axis vector (z-axis of the section, used for orientation)
    local_z: tuple = (1.0, 0.0, 0.0)


@dataclass
class FrameGeometry:
    H: float = 9.0
    B: float = 21.0
    a: float = 5.25
    L_a: float = 14.0
    H_s: float = 6.0
    H_f: float = 9.0
    h_a_cm: float = 35.0
    h_f_cm: float = 30.0
    h_1_cm: float = 90.0
    l_zw_cm: float = 40.0

    column_section: Section = None
    rafter_section: Section = None
    purlin_section: Section = None

    n_bays: int = 4
    n_purlins_per_slope: int = 3   # intermediate purlins (excluding eaves & ridge)

    nodes: list = field(default_factory=list)
    elements: list = field(default_factory=list)

    def __post_init__(self):
        if self.column_section is None:
            self.column_section = Section("C-default", 400, 700, "GL32h")
        if self.rafter_section is None:
            self.rafter_section = Section("R-default", 400, 700, "GL32h")
        if self.purlin_section is None:
            self.purlin_section = Section("P-default", 200, 200, "GL24h")
        # B is computed from n_bays * a if user hasn't overridden consistently
        # but keep explicit B as authoritative; also expose computed
        self.B = self.n_bays * self.a

    # ------------------------------------------------------------------
    def build(self) -> "FrameGeometry":
        """Generate nodes and elements for the full 3-bay portal frame structure."""
        self.nodes.clear()
        self.elements.clear()

        n_frames = self.n_bays + 1   # 3 bays => 4 frames
        x_frames = [i * self.a for i in range(n_frames)]

        # Per-frame key node Y coordinates (across the span)
        y_left  = 0.0
        y_right = self.L_a
        y_ridge = self.L_a / 2.0

        node_id = 1
        # frame_nodes[f] = dict of {label: node_id}
        frame_nodes = []

        for f, xf in enumerate(x_frames):
            fn = {}
            # Footings (pinned)
            self.nodes.append(Node(node_id, xf, y_left,  0.0,
                                   fixity=(1,1,1,0,0,0)))
            fn["base_L"] = node_id; node_id += 1

            self.nodes.append(Node(node_id, xf, y_right, 0.0,
                                   fixity=(1,1,1,0,0,0)))
            fn["base_R"] = node_id; node_id += 1

            # Eaves (knee)
            self.nodes.append(Node(node_id, xf, y_left,  self.H_s))
            fn["eaves_L"] = node_id; node_id += 1

            self.nodes.append(Node(node_id, xf, y_right, self.H_s))
            fn["eaves_R"] = node_id; node_id += 1

            # Ridge
            self.nodes.append(Node(node_id, xf, y_ridge, self.H_f))
            fn["ridge"] = node_id; node_id += 1

            # Intermediate purlin nodes on rafters (left & right slopes)
            n_p = self.n_purlins_per_slope
            for k in range(1, n_p + 1):
                t = k / (n_p + 1)
                # left rafter: from eaves_L to ridge
                yL = y_left  + t * (y_ridge - y_left)
                zL = self.H_s + t * (self.H_f - self.H_s)
                self.nodes.append(Node(node_id, xf, yL, zL))
                fn[f"raft_L_{k}"] = node_id; node_id += 1
                # right rafter
                yR = y_right + t * (y_ridge - y_right)
                zR = self.H_s + t * (self.H_f - self.H_s)
                self.nodes.append(Node(node_id, xf, yR, zR))
                fn[f"raft_R_{k}"] = node_id; node_id += 1

            frame_nodes.append(fn)

        # Build elements for each frame
        elem_id = 1
        for f, fn in enumerate(frame_nodes):
            # Columns (vertical, in YZ frame plane): local_z = Y (span dir) so depth h
            # faces the span direction → strong axis Iy resists in-plane portal bending.
            self.elements.append(Element(elem_id, fn["base_L"], fn["eaves_L"],
                                         self.column_section, "column", f, 0,
                                         local_z=(0.0, 1.0, 0.0)))
            elem_id += 1
            self.elements.append(Element(elem_id, fn["base_R"], fn["eaves_R"],
                                         self.column_section, "column", f, 0,
                                         local_z=(0.0, 1.0, 0.0)))
            elem_id += 1

            # Rafters segmented at purlin locations.
            # local_z = Z (vertical) → after Gram-Schmidt ez ≈ in-plane ⊥ to rafter,
            # ey = [-1,0,0] (out-of-plane) → h faces in-frame direction, strong axis
            # Iy carries in-plane portal bending.
            n_p = self.n_purlins_per_slope
            # Left slope: eaves_L -> raft_L_1 -> ... -> ridge
            chain_L = ["eaves_L"] + [f"raft_L_{k}" for k in range(1, n_p+1)] + ["ridge"]
            chain_R = ["eaves_R"] + [f"raft_R_{k}" for k in range(1, n_p+1)] + ["ridge"]
            for chain in (chain_L, chain_R):
                for k in range(len(chain) - 1):
                    self.elements.append(Element(elem_id, fn[chain[k]], fn[chain[k+1]],
                                                 self.rafter_section, "rafter", f, 0,
                                                 local_z=(0.0, 0.0, 1.0)))
                    elem_id += 1

        # Purlins: connect equivalent rafter nodes between adjacent frames
        purlin_labels = (["eaves_L"]
                         + [f"raft_L_{k}" for k in range(1, self.n_purlins_per_slope+1)]
                         + ["ridge"]
                         + [f"raft_R_{k}" for k in range(self.n_purlins_per_slope, 0, -1)]
                         + ["eaves_R"])
        for f in range(n_frames - 1):
            for label in purlin_labels:
                i = frame_nodes[f][label]
                j = frame_nodes[f+1][label]
                # local z for a purlin lies in the rafter plane perpendicular to purlin axis
                self.elements.append(Element(elem_id, i, j, self.purlin_section,
                                             "purlin", f, f,
                                             local_z=(0.0, 0.0, 1.0)))
                elem_id += 1

        self._frame_nodes = frame_nodes
        return self

    # ------------------------------------------------------------------
    def haunched_depth(self, s: float, L_rafter: float) -> float:
        """
        Linearly interpolated cross-section depth (mm) along a rafter for visualization
        and tapered-section design checks.
            s = 0   at eaves (depth = h_1)
            s = l_zw  end of haunch (depth = h_a or back to nominal)
            s = L_rafter at ridge (depth = h_f)
        """
        h_eaves = self.h_1_cm
        h_ridge = self.h_f_cm
        h_post  = self.rafter_section.h_mm / 10.0   # back to nominal cross-section depth (cm)
        l_zw_m  = self.l_zw_cm / 100.0
        if s <= l_zw_m:
            t = s / max(l_zw_m, 1e-9)
            return (h_eaves * (1 - t) + h_post * t) * 10.0  # mm
        # post-haunch linear taper to ridge
        t = (s - l_zw_m) / max(L_rafter - l_zw_m, 1e-9)
        return (h_post * (1 - t) + h_ridge * t) * 10.0
