"""
GLT material properties per EN 14080:2013.
Characteristic values (5-percentile) in N/mm² (MPa) and kg/m³.
Source: EN 14080:2013 Table 5 (homogeneous glulam GLxxh).
"""

GLT_GRADES = {
    "GL24h": {
        "f_m_k":   24.0,   # bending strength (MPa)
        "f_t_0_k": 19.2,   # tension parallel
        "f_t_90_k": 0.5,
        "f_c_0_k": 24.0,   # compression parallel
        "f_c_90_k": 2.5,
        "f_v_k":   3.5,    # shear
        "E_0_mean":   11500.0,
        "E_0_05":      9600.0,
        "E_90_mean":    300.0,
        "G_mean":       650.0,
        "rho_k":        385.0,
        "rho_mean":     420.0,
    },
    "GL28h": {
        "f_m_k":   28.0,
        "f_t_0_k": 22.3,
        "f_t_90_k": 0.5,
        "f_c_0_k": 28.0,
        "f_c_90_k": 2.5,
        "f_v_k":   3.5,
        "E_0_mean":   12600.0,
        "E_0_05":     10500.0,
        "E_90_mean":    300.0,
        "G_mean":       650.0,
        "rho_k":        425.0,
        "rho_mean":     460.0,
    },
    "GL32h": {
        "f_m_k":   32.0,
        "f_t_0_k": 25.6,
        "f_t_90_k": 0.5,
        "f_c_0_k": 32.0,
        "f_c_90_k": 2.5,
        "f_v_k":   3.5,
        "E_0_mean":   13700.0,
        "E_0_05":     11400.0,
        "E_90_mean":    300.0,
        "G_mean":       650.0,
        "rho_k":        440.0,
        "rho_mean":     490.0,
    },
}

# kmod factors per EN 1995-1-1 Table 3.1 (Service Class 1 & 2, glued laminated timber)
KMOD = {
    "permanent":      0.60,
    "long_term":      0.70,
    "medium_term":    0.80,
    "short_term":     0.90,
    "instantaneous":  1.10,
}

# Partial factor for glulam, gamma_M per EN 1995-1-1 Table 2.3
GAMMA_M = 1.25

# Deformation factor kdef, Service Class 1 & 2 for glulam
KDEF = {1: 0.60, 2: 0.80, 3: 2.00}


def get_design_strengths(grade: str, kmod: float, gamma_M: float = GAMMA_M) -> dict:
    """Compute design strengths f_d = kmod * f_k / gamma_M."""
    p = GLT_GRADES[grade]
    return {
        "f_m_d":    kmod * p["f_m_k"]    / gamma_M,
        "f_t_0_d":  kmod * p["f_t_0_k"]  / gamma_M,
        "f_t_90_d": kmod * p["f_t_90_k"] / gamma_M,
        "f_c_0_d":  kmod * p["f_c_0_k"]  / gamma_M,
        "f_c_90_d": kmod * p["f_c_90_k"] / gamma_M,
        "f_v_d":    kmod * p["f_v_k"]    / gamma_M,
    }
