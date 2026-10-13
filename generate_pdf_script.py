"""
Generates a mobile-optimized, beautifully formatted PDF of the Loom presentation script.
"""
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    KeepTogether,
)

OUTPUT_PDF = Path(__file__).resolve().parent / "Spotter_Loom_Presentation_Script.pdf"

def build_pdf():
    # Margins: 36 points = 0.5 inch, ideal for phone readability
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=letter,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom styles tailored for mobile reading
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#064A56"),
        alignment=1, # Center
        spaceAfter=4,
    )

    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#4A5568"),
        alignment=1, # Center
        spaceAfter=15,
    )

    badge_style = ParagraphStyle(
        "SectionBadge",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.white,
    )

    screen_title_style = ParagraphStyle(
        "ScreenTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=14,
        textColor=colors.HexColor("#1A365D"),
    )

    screen_text_style = ParagraphStyle(
        "ScreenText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=13.5,
        textColor=colors.HexColor("#2D3748"),
    )

    say_title_style = ParagraphStyle(
        "SayTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#22543D"),
    )

    say_text_style = ParagraphStyle(
        "SayText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=16,
        textColor=colors.HexColor("#1A202C"),
    )

    tip_style = ParagraphStyle(
        "TipText",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#2D3748"),
    )

    story = []

    # Title & Header
    story.append(Paragraph("SPOTTER ML FREIGHT RATE CHALLENGE", title_style))
    story.append(Paragraph("<b>2-3 Minute Loom Presentation Script</b> &bull; Candidate: <b>Talha Rauf</b> &bull; Target Time: <b>3:30 - 4:00 mins</b>", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#064A56"), spaceAfter=14))

    # Helper function to generate clean section cards
    def add_section(time_range, title, screen_instruction, spoken_text):
        card_content = []

        # 1. Badge Header
        header_p = Paragraph(f"<b>[{time_range}] &nbsp; {title.upper()}</b>", badge_style)
        header_table = Table([[header_p]], colWidths=[doc.width])
        header_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#064A56")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("CORNERPAD", (0, 0), (-1, -1), 4),
        ]))
        card_content.append(header_table)
        card_content.append(Spacer(1, 4))

        # 2. Laptop Screen Cue (Light Blue box)
        screen_p1 = Paragraph("<b>&#128187; WHAT TO SHOW ON LAPTOP SCREEN:</b>", screen_title_style)
        screen_p2 = Paragraph(screen_instruction, screen_text_style)
        screen_table = Table([[screen_p1], [screen_p2]], colWidths=[doc.width])
        screen_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#EBF8FF")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#BEE3F8")),
            ("TOPPADDING", (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))
        card_content.append(screen_table)
        card_content.append(Spacer(1, 5))

        # 3. Spoken Words (Light Green / Crisp box)
        say_p1 = Paragraph("<b>&#127908; WHAT TO SAY TO THE CAMERA / MIC:</b>", say_title_style)
        say_p2 = Paragraph(spoken_text, say_text_style)
        say_table = Table([[say_p1], [say_p2]], colWidths=[doc.width])
        say_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F0FFF4")),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#C6F6D5")),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ]))
        card_content.append(say_table)
        card_content.append(Spacer(1, 14))

        story.append(KeepTogether(card_content))

    # SECTION 1
    add_section(
        time_range="0:00 - 0:35",
        title="Part 1: Introduction & Problem Statement",
        screen_instruction="Show the project root in VS Code and open <b>README.md</b>.",
        spoken_text=(
            "\"Hi everyone, my name is <b>Talha Rauf</b>. In this video, I'll walk you through my "
            "end-to-end Machine Learning solution for the Spotter Freight Rate Prediction Challenge.<br/><br/>"
            "The objective of this assessment is to predict the spot market rate (<code>posted_rate</code>) "
            "for full-truckload shipments across the United States. In freight logistics, quote accuracy is critical: "
            "underpricing leads to carrier rejections and service failures, while overpricing leads to lost freight volume.<br/><br/>"
            "Our dataset includes 48,000 labeled development loads spanning January through October 2025, and our goal "
            "is to accurately predict 12,000 unseen future loads in November and December, as well as model a 31-day contract "
            "scenario for December.\""
        ),
    )

    # SECTION 2
    add_section(
        time_range="0:35 - 1:15",
        title="Part 2: Key Findings from Exploring the Data (EDA)",
        screen_instruction="Switch tab to show <b>scorer_results/eda_distributions.png</b> and <b>scorer_results/eda_equipment_rates.png</b>.",
        spoken_text=(
            "\"During Exploratory Data Analysis, I identified three fundamental domain patterns:<br/><br/>"
            "First, <b>distance is the overwhelming linehaul driver</b>, exhibiting a <b>0.9085 linear correlation</b> "
            "with the total posted rate. Across the entire dataset, the average Rate-Per-Mile (RPM) is <b>$2.22 per mile</b>.<br/><br/>"
            "Second, we observe a clear <b>equipment pricing hierarchy</b>: Dry Van averages <b>$2.12 per mile</b>, "
            "while Flatbed commands an 8% premium at <b>$2.29 per mile</b> due to specialized tarping and tie-downs, and refrigerated "
            "Reefer commands a 12.3% premium at <b>$2.38 per mile</b> due to continuous fuel burn and temperature liability.<br/><br/>"
            "Third, while 82.5% of validation lanes overlap with the training data, <b>17.5% are brand new, unseen lanes</b>. "
            "That means our model cannot simply memorize origin-destination pairs; it must learn true spatial geometry using geographic coordinates.\""
        ),
    )

    # SECTION 3
    add_section(
        time_range="1:15 - 1:50",
        title="Part 3: Data-Quality Issues & How They Were Addressed",
        screen_instruction="Open <b>src/features.py</b> (lines 120-150 showing imputation logic).",
        spoken_text=(
            "\"Next, my automated audit flagged two specific data quality anomalies:<br/><br/>"
            "<b>1. Missing Cargo Weight:</b> 300 records in development and 165 in validation were missing weight. "
            "Instead of a naive global mean, I used a <b>conditional median imputation grouped by equipment type</b>, "
            "because a flatbed carrying steel has a fundamentally different density profile than a dry van. I also added a boolean "
            "<code>weight_is_na</code> indicator flag.<br/><br/>"
            "<b>2. Missing Market Index:</b> 374 development and 249 validation records were missing <code>market_index</code>. "
            "Since this represents daily macroeconomic market tightness, I applied a <b>calendar-aligned temporal rolling forward and backward fill</b>, "
            "accompanied by a <code>market_index_is_na</code> indicator.<br/><br/>"
            "<b>3. Strictly Positive Target:</b> Lastly, the target <code>posted_rate</code> is strictly positive. To respect Spotter's "
            "evaluation constraint where predictions must be strictly positive, our pipeline models <b>Rate-Per-Mile</b> and enforces positive bounds.\""
        ),
    )

    # SECTION 4
    add_section(
        time_range="1:50 - 2:35",
        title="Part 4: Training & Validation Strategy (Anti-Leakage Split)",
        screen_instruction="Open <b>REPORT.md</b> (scroll to Section 4: Training & Validation Strategy).",
        spoken_text=(
            "\"The most critical architectural decision was how to split and validate the development data.<br/><br/>"
            "A standard random K-fold split is <b>fundamentally flawed</b> here. Our training data runs from January to October 2025, "
            "while our evaluation data runs from November to December. If you randomly shuffle the data, the model trains on future October market "
            "conditions to predict past October loads—causing severe temporal leakage and artificially inflated scores that collapse in production.<br/><br/>"
            "Instead, I instituted a strict <b>Out-of-Time (OOT) Chronological Holdout Split</b>:<br/>"
            "&bull; <b>Training Window:</b> January 1 to August 31 (8 months, 38,477 loads, 80.2%).<br/>"
            "&bull; <b>Validation Window:</b> September 1 to October 31 (2 months, 9,523 loads, 19.8%).<br/><br/>"
            "This 2-month validation horizon <b>identically mimics the actual 2-month test horizon</b>, and ensures zero future "
            "information leaks into our feature preprocessors.\""
        ),
    )

    # SECTION 5
    add_section(
        time_range="2:35 - 3:20",
        title="Part 5: Model Selection & Empirical Benchmark",
        screen_instruction="Show the Benchmark Table in <b>REPORT.md</b> or open <b>scorer_results/model_benchmark_comparison.png</b>.",
        spoken_text=(
            "\"For modeling, rather than predicting the raw total rate directly, I formulated the target as <b>Rate-Per-Mile (RPM = rate / distance)</b>. "
            "Because distance explains 80% of the variance linearly, modeling RPM allows our trees to focus 100% of their capacity on pricing volatility, "
            "equipment surcharges, and holiday surges.<br/><br/>"
            "I benchmarked six configurations on our out-of-time holdout set:<br/>"
            "&bull; <b>Naive Baseline:</b> $270.71 MAE, 11.3% WAPE.<br/>"
            "&bull; <b>Ridge Regression:</b> $186.71 MAE, 7.8% WAPE.<br/>"
            "&bull; <b>XGBoost Regressor:</b> $152.72 MAE, 6.38% WAPE.<br/>"
            "&bull; <b>LightGBM Regressor:</b> $151.80 MAE, 6.34% WAPE.<br/>"
            "&bull; And <b>CatBoost Regressor:</b> achieved an outstanding <b>$117.42 MAE</b> and <b>4.91% WAPE</b> with an R&sup2; of <b>0.8268</b>.<br/><br/>"
            "CatBoost excelled because its ordered boosting and symmetric trees natively capture high-cardinality geographic interactions without overfitting.<br/><br/>"
            "For our final production deployment, I built a <b>Blended Ensemble</b> combining CatBoost, LightGBM, and XGBoost to maximize stability across market regimes.\""
        ),
    )

    # SECTION 6
    add_section(
        time_range="3:20 - 4:00",
        title="Part 6: Code Walkthrough & Scorer Verification",
        screen_instruction="Briefly show the <code>src/</code> folder in sidebar, then open <b>scorer_results/candidate_december.png</b>.",
        spoken_text=(
            "\"Turning to the codebase: everything is organized modularly under <code>src/</code>:<br/>"
            "&bull; <code>features.py</code> computes Haversine distances, circuitousness ratios, directional bearings, ton-miles, cyclical sine/cosine calendar components, and market interaction terms.<br/>"
            "&bull; <code>inference.py</code> trains the production ensemble on all 48,000 historical loads, generates <code>validation_predictions.csv</code> for all 12,000 loads, and fills <code>december_chart_inputs.csv</code>.<br/><br/>"
            "Here on screen is the official December chart produced by Spotter's <code>score.py</code>:<br/>"
            "For this fixed 360-mile Dry Van lane from Lexington to Fort Wayne, our model predicts an average rate of <b>$779.13</b> (roughly $2.16/mile, perfectly aligning with our historical Dry Van baseline).<br/><br/>"
            "Notice how naturally it captures market seasonality: rates surge in early December post-Thanksgiving, hit a mid-month peak around December 17 for e-commerce shipping deadlines, spike right before Christmas due to driver holiday capacity shortages, and drop sharply right after December 25.<br/><br/>"
            "All 12,000 predictions and December inputs passed Spotter's <code>score.py</code> assertions with zero errors, and all code, dependencies, and documentation are committed and pushed to the <b>assessment</b> branch.<br/><br/>"
            "Thank you for your time, and I look forward to discussing the solution further!\""
        ),
    )

    # Quick Recording Tips Box
    tip_p1 = Paragraph("<b>&#128161; PRO TIPS FOR RECORDING ON LOOM:</b>", ParagraphStyle("H", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=10.5, textColor=colors.HexColor("#7B341E")))
    tip_p2 = Paragraph(
        "1. Open the PDF on your phone and hold it at eye level beside your laptop camera.<br/>"
        "2. Pre-open these 4 tabs on your laptop: Tab 1 (<code>README.md</code>), Tab 2 (<code>eda_distributions.png</code>), Tab 3 (<code>REPORT.md</code>), Tab 4 (<code>candidate_december.png</code>).<br/>"
        "3. Speak with a natural, steady pace. You can easily click between tabs when transitioning sections.",
        tip_style,
    )
    tips_table = Table([[tip_p1], [tip_p2]], colWidths=[doc.width])
    tips_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#FFFAF0")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#FBD38D")),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(Spacer(1, 4))
    story.append(tips_table)

    doc.build(story)
    print(f"PDF successfully built: {OUTPUT_PDF}")

if __name__ == "__main__":
    build_pdf()
