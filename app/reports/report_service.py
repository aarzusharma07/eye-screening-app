import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

from app.config import settings
from app.models.models import Visit, Patient, User, RetinalImage, GlaucomaPrediction, ReferralRecord, ReferralTier


class ReportService:
    @staticmethod
    def generate_pdf_report(visit: Visit, db_session) -> str:
        """
        Generates a professional clinical PDF report for a patient visit.
        Includes full patient demographics, image quality findings, model outputs,
        4-tier referral triage, grounded advisory note, and Grad-CAM visualizations.
        """
        os.makedirs(settings.REPORT_DIR, exist_ok=True)

        report_filename = f"report_visit_{visit.id}_{int(datetime.utcnow().timestamp())}.pdf"
        report_path = os.path.join(settings.REPORT_DIR, report_filename)

        doc = SimpleDocTemplate(
            report_path,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        story = []
        styles = getSampleStyleSheet()

        # ── Custom Styles ──────────────────────────────────────────────
        title_style = ParagraphStyle(
            name="ReportTitle", parent=styles["Normal"],
            fontName="Helvetica-Bold", fontSize=20,
            textColor=colors.HexColor("#1565C0"),
            spaceAfter=12, alignment=1
        )
        subtitle_style = ParagraphStyle(
            name="Subtitle", parent=styles["Normal"],
            fontName="Helvetica", fontSize=10,
            textColor=colors.HexColor("#607D8B"),
            spaceAfter=4, alignment=1
        )
        section_style = ParagraphStyle(
            name="SectionTitle", parent=styles["Normal"],
            fontName="Helvetica-Bold", fontSize=13,
            textColor=colors.HexColor("#00897B"),
            spaceBefore=14, spaceAfter=6
        )
        normal_style = styles["Normal"]
        bold_style = ParagraphStyle(
            name="BoldText", parent=normal_style,
            fontName="Helvetica-Bold"
        )
        advisory_style = ParagraphStyle(
            name="Advisory", parent=normal_style,
            fontName="Helvetica", fontSize=10,
            textColor=colors.HexColor("#212121"),
            leading=16
        )

        # ── Patient & Visit Details ─────────────────────────────────────
        patient: Patient = visit.patient
        created_by: User = visit.created_by

        report_title = "Comprehensive Eye Screening (Glaucoma & DR) — Clinical Report"

        # ── Header ─────────────────────────────────────────────────────
        story.append(Paragraph("RURAL HEALTHCARE AI SCREENING NETWORK", subtitle_style))
        story.append(Paragraph(report_title, title_style))
        story.append(Paragraph(f"Generated: {datetime.utcnow().strftime('%d %B %Y, %H:%M UTC')}", subtitle_style))
        story.append(Spacer(1, 6))

        # Blue divider
        divider = Table([[""]], colWidths=[540], rowHeights=[3])
        divider.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#1565C0")),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
        ]))
        story.append(divider)
        story.append(Spacer(1, 14))

        hypertension_label = "Yes" if patient.has_hypertension else "No"
        prev_eye_label = "Yes" if patient.has_prev_eye_issues else "No"
        prev_eye_notes = patient.prev_eye_notes or "—"
        phone_label = patient.phone or "—"
        village_label = patient.village or "—"
        diabetes_label = f"{patient.diabetes_duration} years" if patient.diabetes_duration > 0 else "None"

        details_data = [
            [
                Paragraph("<b>Patient Name:</b>", normal_style), Paragraph(patient.name, normal_style),
                Paragraph("<b>Visit Date:</b>", normal_style), Paragraph(visit.visit_date.strftime("%Y-%m-%d %H:%M"), normal_style)
            ],
            [
                Paragraph("<b>Age / Gender:</b>", normal_style), Paragraph(f"{patient.age} years / {patient.gender}", normal_style),
                Paragraph("<b>Screened By:</b>", normal_style), Paragraph(f"{created_by.full_name} ({created_by.role.value})", normal_style)
            ],
            [
                Paragraph("<b>Phone:</b>", normal_style), Paragraph(phone_label, normal_style),
                Paragraph("<b>Village:</b>", normal_style), Paragraph(village_label, normal_style)
            ],
            [
                Paragraph("<b>Diabetes Duration:</b>", normal_style), Paragraph(diabetes_label, normal_style),
                Paragraph("<b>Hypertension:</b>", normal_style), Paragraph(hypertension_label, normal_style)
            ],
            [
                Paragraph("<b>Prior Eye Issues:</b>", normal_style), Paragraph(prev_eye_label, normal_style),
                Paragraph("<b>Eye Notes:</b>", normal_style), Paragraph(prev_eye_notes, normal_style)
            ],
        ]

        details_table = Table(details_data, colWidths=[110, 160, 110, 160])
        details_table.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E0E0")),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor("#F5F9FF")),
            ('BACKGROUND', (2, 0), (2, -1), colors.HexColor("#F5F9FF")),
            ('PADDING', (0, 0), (-1, -1), 6),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))

        story.append(Paragraph("Patient & Visit Information", section_style))
        story.append(details_table)
        story.append(Spacer(1, 14))

        # Capture notes
        cap_notes = visit.capture_notes
        if cap_notes:
            cn_data = [
                [Paragraph("<b>Capture Notes (Nurse):</b>", bold_style)],
                [Paragraph(cap_notes, normal_style)]
            ]
            cn_table = Table(cn_data, colWidths=[540])
            cn_table.setStyle(TableStyle([
                ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#BDBDBD")),
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FFFDE7")),
                ('PADDING', (0, 0), (-1, -1), 7),
            ]))
            story.append(cn_table)
            story.append(Spacer(1, 10))

        # ── Eye Findings ────────────────────────────────────────────────
        story.append(Paragraph("Multi-Disease Retinal Findings (Simultaneous Screening)", section_style))

        images = visit.images
        eye_flows = []

        for img in images:
            pred: GlaucomaPrediction = img.prediction
            eye_title = f"{img.eye.value} Eye — Retinal Examination"

            eye_flows.append(Paragraph(eye_title, ParagraphStyle(
                name=f"EyeTitle_{img.eye.value}", parent=section_style,
                fontSize=11, textColor=colors.HexColor("#1565C0"),
                spaceBefore=8
            )))

            # Quality row
            quality_color = colors.HexColor("#C8E6C9") if img.quality == "Good" else colors.HexColor("#FFCDD2")
            quality_text = (
                f"<b>Quality:</b> {img.quality} &nbsp;&nbsp; "
                f"<b>Gradability:</b> {img.gradability} &nbsp;&nbsp; "
                f"Focus: {img.focus_score:.2f} &nbsp; "
                f"Lighting: {img.lighting_score:.2f} &nbsp; "
                f"Clarity: {img.clarity_score:.2f}"
            )

            if pred:
                # Glaucoma
                g_prob_pct = f"{pred.probability * 100:.1f}%"
                g_risk_col = {"High": "#D32F2F", "Medium": "#F57C00", "Low": "#388E3C"}.get(pred.risk_category, "#616161")
                g_pred_text = (
                    f"<b>🩺 GLAUCOMA ASSESSMENT:</b> {pred.prediction_class.upper()} &nbsp;|&nbsp; "
                    f"Probability: <b>{g_prob_pct}</b> &nbsp;|&nbsp; "
                    f"<font color='{g_risk_col}'><b>Risk: {pred.risk_category}</b></font>"
                )

                # Diabetic Retinopathy
                dr_prob_val = pred.dr_probability if pred.dr_probability is not None else 0.0
                dr_risk_val = pred.dr_risk_category or "Low"
                dr_cls_val = pred.dr_prediction_class or "No DR"
                dr_prob_pct = f"{dr_prob_val * 100:.1f}%"
                dr_risk_col = {"High": "#D32F2F", "Medium": "#F57C00", "Low": "#388E3C"}.get(dr_risk_val, "#616161")
                dr_pred_text = (
                    f"<b>🩸 DIABETIC RETINOPATHY:</b> {dr_cls_val.upper()} &nbsp;|&nbsp; "
                    f"Probability: <b>{dr_prob_pct}</b> &nbsp;|&nbsp; "
                    f"<font color='{dr_risk_col}'><b>Risk: {dr_risk_val}</b></font>"
                )
            else:
                g_pred_text = "<b>Glaucoma:</b> Analysis pending"
                dr_pred_text = "<b>Diabetic Retinopathy:</b> Analysis pending"

            findings_data = [
                [Paragraph(quality_text, normal_style)],
                [Paragraph(g_pred_text, normal_style)],
                [Paragraph(dr_pred_text, normal_style)]
            ]
            findings_table = Table(findings_data, colWidths=[540])
            findings_table.setStyle(TableStyle([
                ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E0E0")),
                ('BACKGROUND', (0, 0), (0, 0), quality_color),
                ('BACKGROUND', (0, 1), (0, 1), colors.HexColor("#F0F7FF")),
                ('BACKGROUND', (0, 2), (0, 2), colors.HexColor("#FFF8F0")),
                ('PADDING', (0, 0), (-1, -1), 6),
            ]))
            eye_flows.append(findings_table)
            eye_flows.append(Spacer(1, 8))

            # Images: Original + Glaucoma Heatmap + DR Heatmap side by side (3 columns: 175pt each)
            img_row = []
            if os.path.exists(img.image_path):
                try:
                    img_row.append(RLImage(img.image_path, width=1.75 * inch, height=1.75 * inch))
                except Exception:
                    img_row.append(Paragraph("[Original Missing]", normal_style))
            else:
                img_row.append(Paragraph("[Original Missing]", normal_style))

            # Glaucoma Heatmap
            if pred and pred.heatmap_path and os.path.exists(pred.heatmap_path):
                try:
                    img_row.append(RLImage(pred.heatmap_path, width=1.75 * inch, height=1.75 * inch))
                except Exception:
                    img_row.append(Paragraph("[Glaucoma Map Err]", normal_style))
            else:
                img_row.append(Paragraph("[Glaucoma Map Err]", normal_style))

            # DR Heatmap
            dr_hm_path = pred.dr_heatmap_path if (pred and pred.dr_heatmap_path) else (pred.heatmap_path if pred else None)
            if dr_hm_path and os.path.exists(dr_hm_path):
                try:
                    img_row.append(RLImage(dr_hm_path, width=1.75 * inch, height=1.75 * inch))
                except Exception:
                    img_row.append(Paragraph("[DR Map Err]", normal_style))
            else:
                img_row.append(Paragraph("[DR Map Err]", normal_style))

            label_row = [
                Paragraph("<font size=7 color=grey>Original Fundus Scan</font>", normal_style),
                Paragraph("<font size=7 color=grey>Glaucoma Grad-CAM (Disc)</font>", normal_style),
                Paragraph("<font size=7 color=grey>DR Grad-CAM (Lesions)</font>", normal_style)
            ]
            img_table = Table([img_row, label_row], colWidths=[180, 180, 180])
            img_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                ('TOPPADDING', (0, 1), (-1, 1), 3),
            ]))
            eye_flows.append(img_table)
            eye_flows.append(Spacer(1, 12))

        story.extend(eye_flows)

        # ── 4-Tier Referral Decision ────────────────────────────────────
        story.append(Paragraph("Referral Decision & Triage", section_style))

        tier = visit.referral_tier
        tier_label = tier.value if tier else "Not assessed"

        tier_config = {
            ReferralTier.URGENT: ("#D32F2F", "#FFEBEE", "🔴 URGENT REFERRAL"),
            ReferralTier.REFER: ("#E65100", "#FFF3E0", "🟠 REFERRAL RECOMMENDED"),
            ReferralTier.MONITOR: ("#F9A825", "#FFFDE7", "🟡 MONITOR — FOLLOW-UP REQUIRED"),
            ReferralTier.REASSURE: ("#2E7D32", "#E8F5E9", "🟢 REASSURE — NO REFERRAL NEEDED"),
        }
        border_col, bg_col, tier_banner = tier_config.get(
            tier,
            ("#616161", "#F5F5F5", "⚪ ASSESSMENT PENDING")
        )

        referral = db_session.query(ReferralRecord).filter(ReferralRecord.visit_id == visit.id).first()
        facility = referral.referred_to_facility if referral else "N/A"

        ref_data = [
            [
                Paragraph(f"<b>{tier_banner}</b>", ParagraphStyle(
                    name="TierBanner", parent=bold_style, fontSize=12,
                    textColor=colors.HexColor(border_col)
                )),
                Paragraph(f"<b>Referred To:</b> {facility}", normal_style)
            ],
            [
                Paragraph(f"<b>Reason:</b> {visit.referral_reason or 'See advisory below.'}", normal_style),
                ""
            ]
        ]

        ref_table = Table(ref_data, colWidths=[270, 270])
        ref_table.setStyle(TableStyle([
            ('SPAN', (0, 1), (1, 1)),
            ('BOX', (0, 0), (-1, -1), 2.0, colors.HexColor(border_col)),
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(bg_col)),
            ('PADDING', (0, 0), (-1, -1), 9),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        story.append(ref_table)
        story.append(Spacer(1, 14))

        # ── Grounded Advisory Note ──────────────────────────────────────
        if visit.advisory_note:
            story.append(Paragraph("Clinical Advisory Note", section_style))
            advisory_data = [
                [Paragraph(visit.advisory_note, advisory_style)]
            ]
            advisory_table = Table(advisory_data, colWidths=[540])
            advisory_table.setStyle(TableStyle([
                ('BOX', (0, 0), (-1, -1), 1.0, colors.HexColor(border_col)),
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(bg_col)),
                ('PADDING', (0, 0), (-1, -1), 10),
                ('LEFTPADDING', (0, 0), (-1, -1), 14),
            ]))
            story.append(advisory_table)
            story.append(Spacer(1, 14))

        # ── Doctor / Nurse Notes ────────────────────────────────────────
        notes_text = visit.notes if visit.notes else "No clinical notes provided."
        notes_data = [
            [Paragraph("<b>Clinical Notes & Assessment:</b>", bold_style)],
            [Paragraph(notes_text, normal_style)]
        ]
        notes_table = Table(notes_data, colWidths=[540])
        notes_table.setStyle(TableStyle([
            ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#E0E0E0")),
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FAFAFA")),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(notes_table)
        story.append(Spacer(1, 30))

        # ── Sign-off ────────────────────────────────────────────────────
        sig_data = [
            ["", "____________________________________"],
            ["", "Authorized Signature"],
            ["", f"Date: {datetime.utcnow().strftime('%Y-%m-%d')}"],
            ["", f"Role: {created_by.role.value}"]
        ]
        sig_table = Table(sig_data, colWidths=[300, 240])
        sig_table.setStyle(TableStyle([
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('FONTNAME', (1, 1), (1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (1, 1), (1, -1), 9),
            ('TOPPADDING', (1, 1), (1, -1), 3),
        ]))
        story.append(KeepTogether([sig_table]))

        # ── Footer disclaimer ───────────────────────────────────────────
        story.append(Spacer(1, 20))
        story.append(Paragraph(
            "<font size=8 color=grey>"
            "This report is generated by an AI-assisted screening tool and is intended to support, "
            "not replace, clinical judgment. Final diagnosis must be confirmed by a licensed ophthalmologist."
            "</font>",
            ParagraphStyle(name="Disclaimer", parent=normal_style, alignment=1)
        ))

        doc.build(story)
        return report_path
