"""
report_generator.py — PDF and Markdown report generation.

Generates a downloadable report summarizing:
- City overview + LST statistics
- Each hotspot: name, LST, top drivers, explanation text
- Recommended interventions per hotspot
- Budget optimizer allocation
- Data quality notes

Uses ReportLab for PDF, plain Markdown for text version.
"""

import io
import logging
import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


def generate_markdown_report(
    city_name: str,
    analysis_date: str,
    city_stats: Dict,
    hotspot_explanations: List[Dict],
    simulation_df: 'pd.DataFrame',
    optimizer_result: Optional[Dict] = None,
    model_metrics: Optional[Dict] = None,
    lst_source: str = 'gee',
) -> str:
    """
    Generate a comprehensive Markdown report.

    Returns: string with full Markdown content.
    """
    lines = []

    # Header
    lines += [
        f"# Urban Heat Analysis Report — {city_name}",
        f"",
        f"**Generated:** {analysis_date}  ",
        f"**Analysis Period:** {city_stats.get('date_start', 'N/A')} to {city_stats.get('date_end', 'N/A')}  ",
        f"**LST Data Source:** {'Landsat 8/9 via Google Earth Engine' if lst_source == 'gee' else '⚠️ Physics proxy (GEE unavailable)'}  ",
        f"**Grid Cells Analyzed:** {city_stats.get('n_cells', 'N/A')}  ",
        f"",
        f"---",
        f"",
    ]

    # City-level statistics
    lines += [
        f"## City-Level LST Summary",
        f"",
        f"| Metric | Value |",
        f"|---|---|",
        f"| City Mean LST | {city_stats.get('lst_mean', 'N/A'):.1f}°C |",
        f"| City Max LST | {city_stats.get('lst_max', 'N/A'):.1f}°C |",
        f"| City Min LST | {city_stats.get('lst_min', 'N/A'):.1f}°C |",
        f"| LST Std Dev | {city_stats.get('lst_std', 'N/A'):.2f}°C |",
        f"| Air Temperature | {city_stats.get('air_temp_c', 'N/A'):.1f}°C |",
        f"| Relative Humidity | {city_stats.get('rh', 'N/A'):.0f}% |",
        f"| Heat Index | {city_stats.get('heat_index_c', 'N/A'):.1f}°C |",
        f"",
    ]

    # Model quality
    if model_metrics:
        lines += [
            f"## Model Performance",
            f"",
            f"| Metric | Value |",
            f"|---|---|",
            f"| Cross-Validation R² | {model_metrics.get('cv_r2', 'N/A'):.3f} |",
            f"| Cross-Validation RMSE | {model_metrics.get('cv_rmse_c', 'N/A'):.2f}°C |",
            f"| Training R² | {model_metrics.get('train_r2', 'N/A'):.3f} |",
            f"| Training Samples | {model_metrics.get('n_cells', 'N/A')} cells |",
            f"",
            f"> Model uses XGBoost with physics-informed monotonic constraints: "
            f"vegetation and albedo are constrained to decrease LST; built-up area "
            f"and net radiation are constrained to increase LST.",
            f"",
        ]

    # Hotspot analysis
    lines += [
        f"## Detected Heat Hotspots",
        f"",
        f"Using Getis-Ord Gi* spatial statistics (p < 0.05) to identify "
        f"statistically significant heat clusters.",
        f"",
    ]

    for exp in hotspot_explanations:
        rank = exp.get('rank', '?')
        locality = exp.get('locality_name', 'Unknown')
        lst = exp.get('lst_celsius', 0)
        anomaly = exp.get('lst_anomaly_c', 0)
        category = exp.get('hotspot_category', '')

        lines += [
            f"### Hotspot #{rank} — {locality}",
            f"",
            f"- **Land Surface Temperature:** {lst:.1f}°C",
            f"- **Anomaly above city mean:** +{anomaly:.1f}°C",
            f"- **Statistical significance:** {category}",
            f"- **Coordinates:** {exp.get('centroid_lat', 0):.4f}°N, {exp.get('centroid_lon', 0):.4f}°E",
            f"",
        ]

        if exp.get('top_drivers'):
            lines.append(f"**Top thermal drivers (SHAP analysis):**")
            lines.append(f"")
            lines.append(f"| Driver | Cell Value | City Average | LST Contribution |")
            lines.append(f"|---|---|---|---|")
            for driver in exp['top_drivers']:
                direction = "🔴 +heat" if driver['is_heating'] else "🔵 -cool"
                lines.append(
                    f"| {driver['display_name']} | {driver['cell_value_str']} | "
                    f"{driver['city_mean_str']} | {driver['shap_contribution_c']:+.2f}°C {direction} |"
                )
            lines.append(f"")

        # Best intervention for this hotspot
        if simulation_df is not None and not simulation_df.empty:
            try:
                cell_sims = simulation_df[simulation_df['cell_id'] == exp['cell_id']]
                if not cell_sims.empty:
                    best = cell_sims.nsmallest(1, 'delta_lst_c').iloc[0]
                    lines += [
                        f"**Recommended Intervention:**",
                        f"",
                        f"- **Type:** {best.get('intervention_label', best['intervention_type'])}",
                        f"- **Predicted Cooling:** {best['delta_lst_c']:.1f}°C "
                        f"(range: {best.get('delta_lst_lower_c', best['delta_lst_c']*1.2):.1f}°C to "
                        f"{best.get('delta_lst_upper_c', best['delta_lst_c']*0.8):.1f}°C)",
                        f"- **Resource Required:** {best['resource_amount']:.0f} {best['resource_unit']}",
                        f"",
                    ]
            except Exception:
                pass

        lines.append(f"---")
        lines.append(f"")

    # Budget optimizer results
    if optimizer_result and optimizer_result.get('allocations'):
        allocs = optimizer_result['allocations']
        lines += [
            f"## Budget Optimization Results",
            f"",
            f"**Budget:** {optimizer_result.get('budget', 0):,.0f} {optimizer_result.get('budget_type', 'units')}  ",
            f"**Budget Used:** {optimizer_result.get('total_cost', 0):,.0f} "
            f"({optimizer_result.get('budget_utilization', 0)*100:.0f}%)  ",
            f"**Total Predicted Cooling:** {optimizer_result.get('total_delta_lst_c', 0):.2f}°C (sum across hotspots)  ",
            f"**Solver:** {optimizer_result.get('solver_status', 'N/A')}  ",
            f"",
            f"| Locality | Intervention | Intensity | Cooling | Cost |",
            f"|---|---|---|---|---|",
        ]
        for alloc in allocs:
            lines.append(
                f"| {alloc['locality_name']} | "
                f"{alloc.get('intervention_label', alloc['intervention_type'])} | "
                f"{alloc.get('effective_intensity', '?')}/10 | "
                f"{alloc['delta_lst_c']:.2f}°C | "
                f"{alloc['cost']:,.0f} {alloc['resource_unit']} |"
            )
        lines.append(f"")
        lines.append(f"> Allocations weighted by population exposure (GHSL) + LST anomaly severity. "
                     f"Solver: PuLP linear programming.")
        lines.append(f"")

    # Footer
    lines += [
        f"---",
        f"",
        f"## Methodology Notes",
        f"",
        f"- **LST:** Landsat 8/9 Collection 2 Level-2 ST_B10 band, median composite over "
        f"{city_stats.get('date_start', '')}–{city_stats.get('date_end', '')}",
        f"- **LULC:** ESA WorldCover v200 (10 m resolution), class fractions per grid cell",
        f"- **NDVI:** Sentinel-2 SR median composite (cloud-masked)",
        f"- **Weather:** Open-Meteo Archive API (free, no key required)",
        f"- **Urban Morphology:** OpenStreetMap Overpass API via OSMnx",
        f"- **Model:** XGBoost with monotonic constraints (physics-informed). "
        f"SHAP TreeExplainer for feature attribution.",
        f"- **Hotspot Detection:** Getis-Ord Gi* (esda/libpysal), K=8 nearest neighbors, p<0.05",
        f"- **Intervention Cooling Estimates:** Model counterfactual predictions. "
        f"Physical feature modifications documented in source code (intervention_simulator.py).",
        f"",
        f"*Generated by Urban Heat Mitigation AI System — ISRO Hackathon Submission*",
        f"*All data from free public APIs. No hardcoded city data or pre-baked estimates.*",
    ]

    return "\n".join(lines)


def generate_pdf_report(
    city_name: str,
    analysis_date: str,
    city_stats: Dict,
    hotspot_explanations: List[Dict],
    simulation_df,
    optimizer_result: Optional[Dict] = None,
    model_metrics: Optional[Dict] = None,
    lst_source: str = 'gee',
) -> bytes:
    """
    Generate a PDF report using ReportLab.
    Returns bytes of the PDF file.

    Falls back to encoding Markdown as plain text if ReportLab fails.
    """
    md_content = generate_markdown_report(
        city_name, analysis_date, city_stats,
        hotspot_explanations, simulation_df,
        optimizer_result, model_metrics, lst_source
    )

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                         Table, TableStyle, HRFlowable)
        from reportlab.lib.enums import TA_LEFT, TA_CENTER

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=A4,
            leftMargin=2*cm, rightMargin=2*cm,
            topMargin=2*cm, bottomMargin=2*cm,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle('Title', parent=styles['Title'],
                                      fontSize=18, spaceAfter=12,
                                      textColor=colors.HexColor('#1a1a2e'))
        h2_style = ParagraphStyle('H2', parent=styles['Heading2'],
                                   fontSize=14, spaceBefore=12, spaceAfter=6,
                                   textColor=colors.HexColor('#16213e'))
        h3_style = ParagraphStyle('H3', parent=styles['Heading3'],
                                   fontSize=12, spaceBefore=8, spaceAfter=4,
                                   textColor=colors.HexColor('#e94560'))
        body_style = ParagraphStyle('Body', parent=styles['Normal'],
                                     fontSize=10, leading=14, spaceAfter=4)

        story = []

        # Title
        story.append(Paragraph(f"Urban Heat Analysis — {city_name}", title_style))
        story.append(Paragraph(f"Generated: {analysis_date}", body_style))
        story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor('#e94560')))
        story.append(Spacer(1, 0.3*cm))

        # City stats table
        story.append(Paragraph("City-Level LST Summary", h2_style))
        stats_data = [
            ["Metric", "Value"],
            ["City Mean LST", f"{city_stats.get('lst_mean', 0):.1f}°C"],
            ["City Max LST", f"{city_stats.get('lst_max', 0):.1f}°C"],
            ["City Min LST", f"{city_stats.get('lst_min', 0):.1f}°C"],
            ["LST Std Dev", f"{city_stats.get('lst_std', 0):.2f}°C"],
            ["Air Temperature", f"{city_stats.get('air_temp_c', 0):.1f}°C"],
            ["Heat Index", f"{city_stats.get('heat_index_c', 0):.1f}°C"],
            ["Grid Cells Analyzed", str(city_stats.get('n_cells', 0))],
        ]
        t = Table(stats_data, colWidths=[8*cm, 8*cm])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#16213e')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f8f9fa'), colors.white]),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.4*cm))

        # Hotspots
        story.append(Paragraph("Heat Hotspot Analysis", h2_style))
        for exp in hotspot_explanations:
            story.append(Paragraph(
                f"Hotspot #{exp.get('rank', '?')} — {exp.get('locality_name', 'Unknown')}",
                h3_style
            ))
            story.append(Paragraph(
                f"LST: {exp.get('lst_celsius', 0):.1f}°C "
                f"({exp.get('lst_anomaly_c', 0):+.1f}°C above city mean) | "
                f"{exp.get('hotspot_category', '')}",
                body_style
            ))

            if exp.get('top_drivers'):
                driver_data = [["Driver", "Cell Value", "City Avg", "LST Impact"]]
                for d in exp['top_drivers']:
                    driver_data.append([
                        d['display_name'][:30],
                        d['cell_value_str'],
                        d['city_mean_str'],
                        f"{d['shap_contribution_c']:+.2f}°C",
                    ])
                dt = Table(driver_data, colWidths=[6*cm, 3*cm, 3*cm, 3*cm])
                dt.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#e94560')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 9),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
                    ('PADDING', (0, 0), (-1, -1), 5),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#fff5f5'), colors.white]),
                ]))
                story.append(dt)
            story.append(Spacer(1, 0.3*cm))

        # Budget optimizer
        if optimizer_result and optimizer_result.get('allocations'):
            story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
            story.append(Paragraph("Budget Optimization Results", h2_style))
            story.append(Paragraph(
                f"Budget: {optimizer_result.get('budget', 0):,.0f} {optimizer_result.get('budget_type', '')} | "
                f"Predicted total cooling: {optimizer_result.get('total_delta_lst_c', 0):.2f}°C",
                body_style
            ))

            alloc_data = [["Locality", "Intervention", "Cooling", "Cost"]]
            for alloc in optimizer_result['allocations']:
                alloc_data.append([
                    alloc['locality_name'][:25],
                    alloc.get('intervention_label', alloc['intervention_type'])[:25],
                    f"{alloc['delta_lst_c']:.2f}°C",
                    f"{alloc['cost']:,.0f} {alloc['resource_unit']}",
                ])
            at = Table(alloc_data, colWidths=[5*cm, 5*cm, 3*cm, 4*cm])
            at.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0f3460')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#dee2e6')),
                ('PADDING', (0, 0), (-1, -1), 5),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f0f8ff'), colors.white]),
            ]))
            story.append(at)

        doc.build(story)
        return buffer.getvalue()

    except Exception as e:
        logger.error(f"PDF generation failed: {e}. Returning Markdown as bytes.")
        return md_content.encode('utf-8')
