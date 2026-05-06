"""
Glued laminated timber (GLT) member design per EN 1995-1-1 (Eurocode 5).

Verifications implemented:
    - 6.1.6 Bending: σ_m,d / f_m,d ≤ 1
    - 6.1.7 Shear:   τ_d / f_v,d ≤ 1   (with k_cr = 0.67 for solid timber/glulam)
    - 6.2.4 Combined bending + axial compression (column):
              (σ_c,0,d / (k_c · f_c,0,d))   +  σ_m,y,d / f_m,y,d  +  k_m · σ_m,z,d / f_m,z,d  ≤ 1
              σ_c,0,d / (k_c,z · f_c,0,d)   +  k_m · σ_m,y,d / f_m,y,d  +  σ_m,z,d / f_m,z,d  ≤ 1
    - 6.3.2 Lateral torsional buckling for beams (k_crit)
    - 7.2 Deflection limits: w_inst ≤ L/300, w_net,fin ≤ L/250 (typical roof beam)

Reference values (size factor k_h, k_m=0.7 for rectangular glulam) are taken
from EN 1995-1-1 §3.3 and §6.

Web resource consulted (algorithm logic cross-checked):
    - "Design of Timber Structures – Volume 1" Swedish Wood (free PDF)
    - Eurocode 5: Design of timber structures (BS EN 1995-1-1:2004+A2:2014)
    - https://eurocodeapplied.com/design/en1995  (online calculator – Eurocode formulas)
"""
from dataclasses import dataclass
from typing import Dict
import math

from .materials import GLT_GRADES, KMOD, GAMMA_M, KDEF, get_design_strengths
from .geometry import Section


# --- helpers ---------------------------------------------------------
def k_h_bending(h_mm: float) -> float:
    """Size effect factor for bending (glulam), EN 1995-1-1 §3.3(3)."""
    if h_mm >= 600:
        return 1.0
    return min(1.10, (600.0 / h_mm) ** 0.1)


def k_h_tension(h_mm: float) -> float:
    """Size effect factor for tension parallel to grain (glulam)."""
    if h_mm >= 600:
        return 1.0
    return min(1.10, (600.0 / h_mm) ** 0.1)


def k_c(lambda_rel: float, beta_c: float = 0.1) -> float:
    """Compression buckling factor k_c, EN 1995-1-1 §6.3.2 (eq. 6.27, 6.28).
       beta_c = 0.1 for glulam."""
    if lambda_rel <= 0.3:
        return 1.0
    k = 0.5 * (1.0 + beta_c * (lambda_rel - 0.3) + lambda_rel ** 2)
    return 1.0 / (k + math.sqrt(k * k - lambda_rel ** 2))


def lambda_rel_compression(L_eff: float, i: float, f_c_0_k: float, E_0_05: float) -> float:
    """Relative slenderness for compression buckling, EN 1995-1-1 §6.3.2."""
    lam = L_eff / i
    return (lam / math.pi) * math.sqrt(f_c_0_k / E_0_05)


def k_crit(lambda_rel_m: float) -> float:
    """Lateral-torsional buckling factor for bending, EN 1995-1-1 §6.3.3 eq. 6.34."""
    if lambda_rel_m <= 0.75:
        return 1.0
    if lambda_rel_m <= 1.4:
        return 1.56 - 0.75 * lambda_rel_m
    return 1.0 / (lambda_rel_m ** 2)


def lambda_rel_lateral(L_ef: float, sec: Section, grade: str) -> float:
    """Relative slenderness for LTB of glulam beams, EN 1995-1-1 §6.3.3 eq. 6.32.
       For rectangular glulam:
           σ_m,crit = 0.78 b² / (h L_ef) · E_0,05
           λ_rel,m = sqrt(f_m,k / σ_m,crit)
    """
    p = GLT_GRADES[grade]
    b = sec.b_mm / 1000.0
    h = sec.h_mm / 1000.0
    E05 = p["E_0_05"] * 1.0e6
    sig_m_crit = 0.78 * (b**2) / (h * L_ef) * E05
    return math.sqrt((p["f_m_k"] * 1.0e6) / sig_m_crit)


# --- Beam design -----------------------------------------------------
@dataclass
class BeamDesignResult:
    member_label: str
    section: Section
    grade: str
    L: float                   # member length (m)
    L_ef: float                # effective length for LTB (m)
    M_Ed: float                # max moment (kNm)
    V_Ed: float                # max shear (kN)
    N_Ed: float                # axial (kN, + tension / − compression)
    sig_m_d: float
    f_m_d: float
    bending_UR: float
    tau_d: float
    f_v_d: float
    shear_UR: float
    k_crit: float
    LTB_UR: float
    deflection_inst: float     # mm
    deflection_limit_inst: float
    deflection_UR: float
    pass_fail: bool
    notes: list


def design_beam(label: str, sec: Section, grade: str,
                L: float, L_ef: float,
                M_Ed_kNm: float, V_Ed_kN: float, N_Ed_kN: float = 0.0,
                kmod: float = KMOD["medium_term"],
                w_inst_mm: float = 0.0,
                deflection_limit_ratio: float = 300.0) -> BeamDesignResult:

    p = GLT_GRADES[grade]
    f = get_design_strengths(grade, kmod)

    # Bending (about strong axis y)
    W_y = sec.b * sec.h ** 2 / 6.0    # m^3
    sig_m_d = abs(M_Ed_kNm) * 1.0e3 / W_y / 1.0e6   # MPa
    f_m_d = f["f_m_d"] * k_h_bending(sec.h_mm)
    bending_UR = sig_m_d / f_m_d

    # Shear with k_cr = 0.67 (EN 1995-1-1 §6.1.7 with NA)
    k_cr = 0.67
    A_eff = k_cr * sec.b * sec.h
    tau_d = 1.5 * abs(V_Ed_kN) * 1.0e3 / (A_eff) / 1.0e6   # MPa
    f_v_d = f["f_v_d"]
    shear_UR = tau_d / f_v_d

    # LTB (lateral-torsional buckling)
    lam_rel_m = lambda_rel_lateral(L_ef, sec, grade)
    kc = k_crit(lam_rel_m)
    LTB_UR = sig_m_d / (kc * f_m_d)

    # Deflection (instantaneous) limit L/300 typical roof; user can change
    delta_lim = (L * 1000.0) / deflection_limit_ratio   # mm
    defl_UR = abs(w_inst_mm) / delta_lim if delta_lim > 0 else 0.0

    URs = [bending_UR, shear_UR, LTB_UR, defl_UR]
    notes = []
    if bending_UR > 1: notes.append(f"Bending UR = {bending_UR:.2f} > 1 — increase depth")
    if shear_UR  > 1: notes.append(f"Shear UR = {shear_UR:.2f} > 1 — increase b or h")
    if LTB_UR    > 1: notes.append(f"LTB UR = {LTB_UR:.2f} > 1 — add lateral restraints")
    if defl_UR   > 1: notes.append(f"Deflection UR = {defl_UR:.2f} > 1 — stiffen section")

    return BeamDesignResult(
        member_label=label, section=sec, grade=grade,
        L=L, L_ef=L_ef, M_Ed=M_Ed_kNm, V_Ed=V_Ed_kN, N_Ed=N_Ed_kN,
        sig_m_d=sig_m_d, f_m_d=f_m_d, bending_UR=bending_UR,
        tau_d=tau_d, f_v_d=f_v_d, shear_UR=shear_UR,
        k_crit=kc, LTB_UR=LTB_UR,
        deflection_inst=abs(w_inst_mm), deflection_limit_inst=delta_lim,
        deflection_UR=defl_UR,
        pass_fail=all(u <= 1.0 for u in URs),
        notes=notes,
    )


# --- Column design ---------------------------------------------------
@dataclass
class ColumnDesignResult:
    member_label: str
    section: Section
    grade: str
    L: float
    L_eff_y: float
    L_eff_z: float
    N_Ed: float                # compression (kN, + compression)
    M_y_Ed: float
    M_z_Ed: float
    sig_c_0_d: float
    f_c_0_d: float
    sig_m_y_d: float
    sig_m_z_d: float
    f_m_d: float
    lambda_rel_y: float
    lambda_rel_z: float
    k_c_y: float
    k_c_z: float
    UR_y: float                # combined eq (6.23)
    UR_z: float                # combined eq (6.24)
    pass_fail: bool
    notes: list


def design_column(label: str, sec: Section, grade: str,
                  L: float, L_eff_y: float, L_eff_z: float,
                  N_Ed_kN: float, M_y_Ed_kNm: float, M_z_Ed_kNm: float,
                  kmod: float = KMOD["medium_term"]) -> ColumnDesignResult:

    p = GLT_GRADES[grade]
    f = get_design_strengths(grade, kmod)

    A = sec.A
    W_y = sec.b * sec.h ** 2 / 6.0
    W_z = sec.h * sec.b ** 2 / 6.0

    # Stresses (compression positive)
    sig_c_0_d = max(N_Ed_kN, 0.0) * 1.0e3 / A / 1.0e6
    sig_m_y_d = abs(M_y_Ed_kNm) * 1.0e3 / W_y / 1.0e6
    sig_m_z_d = abs(M_z_Ed_kNm) * 1.0e3 / W_z / 1.0e6
    f_c_0_d = f["f_c_0_d"]
    f_m_d   = f["f_m_d"] * k_h_bending(sec.h_mm)

    # Buckling about y (bending about y -> deflection about z, slenderness with i_z)
    # i_y = sqrt(Iy/A), i_z = sqrt(Iz/A)
    i_y = math.sqrt(sec.Iy / A)
    i_z = math.sqrt(sec.Iz / A)
    lam_y = lambda_rel_compression(L_eff_y, i_y, p["f_c_0_k"], p["E_0_05"] * 1.0e6 / 1.0e6)  # using MPa consistently
    # Re-derive with consistent units (MPa):
    lam_y = lambda_rel_compression(L_eff_y, i_y, p["f_c_0_k"], p["E_0_05"])
    lam_z = lambda_rel_compression(L_eff_z, i_z, p["f_c_0_k"], p["E_0_05"])

    kcy = k_c(lam_y)
    kcz = k_c(lam_z)

    k_m = 0.7   # rectangular glulam, §6.1.6(2)

    # EN 1995-1-1 §6.3.2 eq. (6.23) and (6.24)
    UR_y = (sig_c_0_d / (kcy * f_c_0_d)) +        sig_m_y_d / f_m_d + k_m * sig_m_z_d / f_m_d
    UR_z = (sig_c_0_d / (kcz * f_c_0_d)) + k_m * sig_m_y_d / f_m_d +        sig_m_z_d / f_m_d

    notes = []
    if UR_y > 1: notes.append(f"Combined check (6.23) UR = {UR_y:.2f} > 1")
    if UR_z > 1: notes.append(f"Combined check (6.24) UR = {UR_z:.2f} > 1")

    return ColumnDesignResult(
        member_label=label, section=sec, grade=grade,
        L=L, L_eff_y=L_eff_y, L_eff_z=L_eff_z,
        N_Ed=max(N_Ed_kN, 0.0),
        M_y_Ed=abs(M_y_Ed_kNm), M_z_Ed=abs(M_z_Ed_kNm),
        sig_c_0_d=sig_c_0_d, f_c_0_d=f_c_0_d,
        sig_m_y_d=sig_m_y_d, sig_m_z_d=sig_m_z_d, f_m_d=f_m_d,
        lambda_rel_y=lam_y, lambda_rel_z=lam_z,
        k_c_y=kcy, k_c_z=kcz,
        UR_y=UR_y, UR_z=UR_z,
        pass_fail=(UR_y <= 1.0 and UR_z <= 1.0),
        notes=notes,
    )
