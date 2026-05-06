"""
PDF report generator for the portal frame analysis & design.
Uses reportlab (no external network needed).
"""
from datetime import datetime
from typing import Dict, List
from io import BytesIO

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, PageBreak, Image)

from .geometry import FrameGeometry
from .design_glt import BeamDesignResult, ColumnDesignResult


def _h(text, styles, lvl=1):
    style = styles["Heading%d" % lvl]
    return Paragraph(text, style)


def _kv_table(data, col_widths=(60*mm, 80*mm)):
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ("FONT", (0,0), (-1,-1), "Helvetica", 9),
        ("LINEABOVE", (0,0), (-1,0), 0.5, colors.grey),
        ("LINEBELOW", (0,-1), (-1,-1), 0.5, colors.grey),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING", (0,0), (-1,-1), 4),
        ("BACKGROUND", (0,0), (0,-1), colors.HexColor("#F5F5F5")),
    ]))
    return t


def _check_table(rows):
    t = Table(rows, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONT", (0,0), (-1,-1), "Helvetica", 9),
        ("FONT", (0,0), (-1,0), "Helvetica-Bold", 9),
        ("BACKGROUND", (0,0), (-1,0), colors.HexColor("#1565C0")),
        ("TEXTCOLOR", (0,0), (-1,0), colors.white),
        ("GRID", (0,0), (-1,-1), 0.25, colors.grey),
        ("BOTTOMPADDING", (0,0), (-1,-1), 4),
        ("TOPPADDING", (0,0), (-1,-1), 4),
    ]))
    return t


def generate_report(out_path: str,
                    project: dict,
                    geom: FrameGeometry,
                    surface_loads: dict,
                    beam_results: List[BeamDesignResult],
                    column_results: List[ColumnDesignResult]):

    doc = SimpleDocTemplate(out_path, pagesize=A4,
                            leftMargin=18*mm, rightMargin=18*mm,
                            topMargin=18*mm, bottomMargin=18*mm,
                            title="Portal Frame Design Report",
                            author=project.get("engineer", "Engineer"))
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="Small", parent=styles["BodyText"], fontSize=8))
    story = []

    # -------- Title page
    story.append(_h("Single-Story 3-Bay GLT Portal Frame — Design Report", styles, 1))
    story.append(Paragraph(
        f"Project: <b>{project.get('name','-')}</b><br/>"
        f"Location: {project.get('location','-')}<br/>"
        f"Engineer: {project.get('engineer','-')}<br/>"
        f"Date: {datetime.now():%Y-%m-%d %H:%M}<br/>"
        f"Code: {project.get('code','EN 1995-1-1')}",
        styles["BodyText"]))
    story.append(Spacer(1, 6*mm))

    # -------- Project setting
    story.append(_h("1. Project Setting", styles, 2))
    rows = [
        ["Parameter", "Value"],
        ["Project name", project.get("name", "-")],
        ["Service class",  str(project.get("service_class", 1))],
        ["Load duration",  project.get("load_duration", "medium-term")],
        ["k_mod",          f"{project.get('kmod', 0.80):.2f}"],
        ["γ_M",            f"{project.get('gamma_M', 1.25):.2f}"],
        ["Combination",    project.get("combo", "ULS (1.35G + 1.5Q + 0.9W)")],
    ]
    story.append(_check_table(rows))

    story.append(Spacer(1, 4*mm))

    # -------- Geometry
    story.append(_h("2. Geometry", styles, 2))
    story.append(_kv_table([
        ["Building height H (m)",     f"{geom.H:.2f}"],
        ["Building depth B (m)",      f"{geom.B:.2f}"],
        ["Frame spacing a (m)",       f"{geom.a:.2f}"],
        ["Number of bays",            f"{geom.n_bays}"],
        ["Frame span L_a (m)",        f"{geom.L_a:.2f}"],
        ["Eaves height H_s (m)",      f"{geom.H_s:.2f}"],
        ["Ridge height H_f (m)",      f"{geom.H_f:.2f}"],
        ["Footing CS height h_a (cm)",f"{geom.h_a_cm:.1f}"],
        ["Ridge CS height h_f (cm)",  f"{geom.h_f_cm:.1f}"],
        ["Knee CS height h_1 (cm)",   f"{geom.h_1_cm:.1f}"],
        ["Wedge length l_zw (cm)",    f"{geom.l_zw_cm:.1f}"],
        ["Column section",
            f"{geom.column_section.b_mm}×{geom.column_section.h_mm} mm, {geom.column_section.grade}"],
        ["Rafter section",
            f"{geom.rafter_section.b_mm}×{geom.rafter_section.h_mm} mm, {geom.rafter_section.grade}"],
        ["Purlin section",
            f"{geom.purlin_section.b_mm}×{geom.purlin_section.h_mm} mm, {geom.purlin_section.grade}"],
    ]))

    story.append(Spacer(1, 4*mm))

    # -------- Loads
    story.append(_h("3. Loads (surface, characteristic)", styles, 2))
    story.append(_kv_table([
        ["Permanent G (kN/m²)",  f"{surface_loads.get('g_roof',0):.2f}"],
        ["Snow Q (kN/m²)",       f"{surface_loads.get('q_snow',0):.2f}"],
        ["Wind W (kN/m²)",       f"{surface_loads.get('w_wind',0):.2f}"],
        ["Self-weight",          "Auto from member sections × ρ_mean"],
        ["Tributary depth",      "interior frame: a;  edge frame: a/2"],
    ]))

    story.append(PageBreak())

    # -------- Beam design
    story.append(_h("4. Beam (Rafter) Design — EN 1995-1-1", styles, 2))
    rows = [["Member","Sec b×h (mm)","Grade","M_Ed (kNm)","V_Ed (kN)",
             "Bend UR","Shear UR","LTB UR","Defl UR","Result"]]
    for r in beam_results:
        rows.append([r.member_label,
                     f"{r.section.b_mm:.0f}×{r.section.h_mm:.0f}",
                     r.grade, f"{r.M_Ed:.1f}", f"{r.V_Ed:.1f}",
                     f"{r.bending_UR:.2f}", f"{r.shear_UR:.2f}",
                     f"{r.LTB_UR:.2f}", f"{r.deflection_UR:.2f}",
                     "PASS" if r.pass_fail else "FAIL"])
    story.append(_check_table(rows))
    for r in beam_results:
        if r.notes:
            story.append(Paragraph("<b>%s notes:</b> %s" % (
                r.member_label, "; ".join(r.notes)), styles["Small"]))

    story.append(Spacer(1, 4*mm))

    # -------- Column design
    story.append(_h("5. Column Design — EN 1995-1-1", styles, 2))
    rows = [["Member","Sec b×h (mm)","Grade","N_Ed (kN)","M_y (kNm)",
             "M_z (kNm)","λ_rel,y","λ_rel,z","UR (6.23)","UR (6.24)","Result"]]
    for r in column_results:
        rows.append([r.member_label,
                     f"{r.section.b_mm:.0f}×{r.section.h_mm:.0f}",
                     r.grade, f"{r.N_Ed:.1f}",
                     f"{r.M_y_Ed:.1f}", f"{r.M_z_Ed:.1f}",
                     f"{r.lambda_rel_y:.2f}", f"{r.lambda_rel_z:.2f}",
                     f"{r.UR_y:.2f}", f"{r.UR_z:.2f}",
                     "PASS" if r.pass_fail else "FAIL"])
    story.append(_check_table(rows))

    story.append(Spacer(1, 4*mm))
    story.append(Paragraph(
        "Verifications carried out: bending (§6.1.6, including k_h size factor), "
        "shear (§6.1.7 with k_cr = 0.67), lateral-torsional buckling (§6.3.3), "
        "combined bending + axial compression with stability (§6.3.2 eqs. 6.23 and 6.24, "
        "k_m = 0.7 for rectangular glulam). Material values per EN 14080:2013.",
        styles["Small"]))

    doc.build(story)
    return out_path
