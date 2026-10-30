"""
generate_pdf.py
Compiles the RSCOE CS3203T-A Data Mining Mini-Project Report to exactly 5 pages in PDF format using ReportLab.
Ensures perfect vertical spacing, typography, alignment, and formatting matching the college standard.
Student: Vinit Shinde (Roll No: 37, PRN: RBT24CS178)
"""

import sys
from pathlib import Path
import pypdf
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def make_footer(canvas, doc):
    """Draws running footer with line and page number on every page."""
    canvas.saveState()
    # Bottom dividing rule
    canvas.setStrokeColor(colors.HexColor('#B7C9D6'))
    canvas.setLineWidth(0.75)
    canvas.line(18 * mm, 39.69, 595.28 - 18 * mm, 39.69)
    
    # Bottom labels
    canvas.setFont('Helvetica', 8)
    canvas.setFillColor(colors.HexColor('#555555'))
    canvas.drawString(18 * mm, 25.51, 'Data Mining and Analytics | CS3203T-A')
    canvas.drawRightString(595.28 - 18 * mm, 25.51, f'Page {doc.page}')
    canvas.restoreState()

def build_pdf(output_path: str):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=15 * mm,
        bottomMargin=19 * mm
    )
    
    # Usable width: 595.28 - 36mm = 493.24 pt
    USABLE_W = 595.28 - 36 * mm

    styles = getSampleStyleSheet()

    # Define refined paragraph styles
    c_navy = colors.HexColor('#17365D')
    c_charcoal = colors.HexColor('#1F2937')
    c_muted = colors.HexColor('#4B5563')

    style_college_title = ParagraphStyle(
        'CollegeTitle',
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#111827'),
        spaceAfter=2
    )
    style_college_sub = ParagraphStyle(
        'CollegeSub',
        fontName='Helvetica',
        fontSize=10,
        leading=13,
        alignment=TA_CENTER,
        textColor=c_muted,
        spaceAfter=0
    )
    style_report_type = ParagraphStyle(
        'ReportType',
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        alignment=TA_CENTER,
        textColor=c_navy,
        spaceAfter=6
    )
    style_project_title = ParagraphStyle(
        'ProjectTitle',
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        alignment=TA_CENTER,
        textColor=c_navy,
        spaceAfter=6
    )
    style_project_sub = ParagraphStyle(
        'ProjectSub',
        fontName='Helvetica',
        fontSize=9.5,
        leading=13.5,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#374151'),
        spaceAfter=0
    )
    style_meta_label = ParagraphStyle(
        'MetaLabel',
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=14,
        textColor=c_navy
    )
    style_meta_val = ParagraphStyle(
        'MetaVal',
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=c_charcoal
    )
    style_student_name = ParagraphStyle(
        'StudentName',
        fontName='Helvetica-Bold',
        fontSize=13,
        leading=16,
        textColor=colors.HexColor('#0F172A')
    )
    style_student_sub = ParagraphStyle(
        'StudentSub',
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=c_charcoal
    )
    style_h1 = ParagraphStyle(
        'Heading1_Custom',
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=13.5,
        textColor=c_navy,
        spaceBefore=6,
        spaceAfter=3,
        keepWithNext=True
    )
    style_h2 = ParagraphStyle(
        'Heading2_Custom',
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=c_navy,
        spaceBefore=4,
        spaceAfter=2,
        keepWithNext=True
    )
    style_body = ParagraphStyle(
        'Body_Custom',
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.2,
        textColor=c_charcoal,
        alignment=TA_JUSTIFY,
        spaceAfter=4
    )
    style_bullet = ParagraphStyle(
        'Bullet_Custom',
        fontName='Helvetica',
        fontSize=8.5,
        leading=11.0,
        textColor=c_charcoal,
        alignment=TA_JUSTIFY,
        leftIndent=12,
        firstLineIndent=-12,
        spaceAfter=2.5
    )
    style_table_h = ParagraphStyle(
        'TableHead',
        fontName='Helvetica-Bold',
        fontSize=8.0,
        leading=10,
        textColor=c_navy
    )
    style_table_c = ParagraphStyle(
        'TableCell',
        fontName='Helvetica',
        fontSize=7.8,
        leading=9.8,
        textColor=c_charcoal
    )
    style_table_c_bold = ParagraphStyle(
        'TableCellBold',
        fontName='Helvetica-Bold',
        fontSize=7.8,
        leading=9.8,
        textColor=c_navy
    )
    style_footer_note = ParagraphStyle(
        'FooterNote',
        fontName='Helvetica',
        fontSize=8.0,
        leading=10,
        alignment=TA_CENTER,
        textColor=colors.HexColor('#6B7280')
    )

    story = []

    # =========================================================================
    # PAGE 1: COVER PAGE
    # =========================================================================
    story.append(Spacer(1, 15 * mm))
    story.append(Paragraph("RAJARSHI SHAHU COLLEGE OF ENGINEERING", style_college_title))
    story.append(Paragraph("Tathawade, Pune &ndash; 411033", style_college_sub))
    story.append(Paragraph("Department of Computer Engineering", style_college_sub))
    
    story.append(Spacer(1, 20 * mm))
    story.append(Paragraph("MINI PROJECT REPORT", style_report_type))
    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph("MOVIE / OTT RECOMMENDATION AND TREND ANALYSIS", style_project_title))
    story.append(Paragraph(
        "An Integrated Analytical System Using Fact Constellation Warehousing, Apriori Association Rules, "
        "Multi-Modal Classification, Regularized Regression, and ARIMA Time Series",
        style_project_sub
    ))
    
    story.append(Spacer(1, 22 * mm))
    
    meta_table_data = [
        [Paragraph("Subject:", style_meta_label), Paragraph("Data Mining and Analytics", style_meta_val)],
        [Paragraph("Subject Code:", style_meta_label), Paragraph("CS3203T-A", style_meta_val)],
        [Paragraph("Academic Year:", style_meta_label), Paragraph("2026&ndash;2027", style_meta_val)],
        [Paragraph("Semester:", style_meta_label), Paragraph("1", style_meta_val)],
        [Spacer(1, 4 * mm), Spacer(1, 4 * mm)],
        [Paragraph("Submitted by:", style_meta_label), Paragraph("<b>Vinit Shinde</b>", style_student_name)],
        [Paragraph("Roll No:", style_meta_label), Paragraph("37", style_student_sub)],
        [Paragraph("PRN:", style_meta_label), Paragraph("RBT24CS178", style_student_sub)],
        [Paragraph("Branch:", style_meta_label), Paragraph("B.Tech &ndash; Computer Engineering", style_student_sub)],
        [Spacer(1, 4 * mm), Spacer(1, 4 * mm)],
        [Paragraph("Faculty Guide:", style_meta_label), Paragraph("Prof. Jit Shendekar", style_meta_val)],
    ]
    meta_table = Table(meta_table_data, colWidths=[35 * mm, 80 * mm])
    meta_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
        ('TOPPADDING', (0,0), (-1,-1), 1.5),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
    ]))
    
    # Center meta table
    outer_table = Table([[meta_table]], colWidths=[USABLE_W])
    outer_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('LEFTPADDING', (0,0), (-1,-1), 25 * mm),
    ]))
    story.append(outer_table)
    
    story.append(Spacer(1, 20 * mm))
    story.append(Paragraph(
        "Autonomous Institute Affiliated to Savitribai Phule Pune University &bull; Academic Year 2026&ndash;27",
        style_footer_note
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 2: CONTENTS, 1. ABSTRACT, 2. INTRODUCTION, 3. PROBLEM STATEMENT
    # =========================================================================
    story.append(Paragraph("CONTENTS", style_h1))
    
    toc_data = [
        [Paragraph("Sec", style_table_h), Paragraph("Title", style_table_h), Paragraph("Sec", style_table_h), Paragraph("Title", style_table_h)],
        [Paragraph("1", style_table_c_bold), Paragraph("Abstract", style_table_c), Paragraph("8", style_table_c_bold), Paragraph("Methodology and Algorithm", style_table_c)],
        [Paragraph("2", style_table_c_bold), Paragraph("Introduction", style_table_c), Paragraph("9", style_table_c_bold), Paragraph("System Architecture", style_table_c)],
        [Paragraph("3", style_table_c_bold), Paragraph("Problem Statement", style_table_c), Paragraph("10", style_table_c_bold), Paragraph("Software and Tools", style_table_c)],
        [Paragraph("4", style_table_c_bold), Paragraph("Aim and Objectives", style_table_c), Paragraph("11", style_table_c_bold), Paragraph("Experimental Output and Evaluation", style_table_c)],
        [Paragraph("5", style_table_c_bold), Paragraph("Scope of the Project", style_table_c), Paragraph("12", style_table_c_bold), Paragraph("Limitations and Future Scope", style_table_c)],
        [Paragraph("6", style_table_c_bold), Paragraph("Syllabus Mapping", style_table_c), Paragraph("13", style_table_c_bold), Paragraph("Conclusion", style_table_c)],
        [Paragraph("7", style_table_c_bold), Paragraph("Dataset Description", style_table_c), Paragraph("14", style_table_c_bold), Paragraph("References", style_table_c)],
    ]
    toc_table = Table(toc_data, colWidths=[12 * mm, 70 * mm, 12 * mm, 79.24 * mm])
    toc_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#D9EAF7')),
        ('TEXTCOLOR', (0, 0), (-1, 0), c_navy),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0, 0), (-1, -1), 2.2),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.2),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
    ]))
    story.append(toc_table)
    story.append(Spacer(1, 3 * mm))

    story.append(Paragraph("1. ABSTRACT", style_h1))
    story.append(Paragraph(
        "With the rapid expansion of digital streaming media platforms, entertainment distributors face fundamental "
        "challenges in catalog discovery, customer churn reduction, and acquisition planning. This mini-project presents "
        "an end-to-end analytical data mining architecture that unifies theatrical viewer preferences from the "
        "<b>MovieLens 100k</b> dataset (100,836 ratings across 9,742 movies) with commercial video-on-demand distribution patterns "
        "from the <b>Netflix Titles</b> catalog (8,807 streaming records). The system executes an automated, idempotent ETL pipeline "
        "loading raw records into a conformed <b>Fact Constellation Data Warehouse</b> in SQLite across 5,340 conformed dates "
        "with complete relational foreign key integrity.",
        style_body
    ))
    story.append(Paragraph(
        "Upon this multidimensional warehouse, five complementary data mining modules are executed: (1) Apriori market-basket "
        "mining discovering 935 high-lift co-viewing association rules (Lift up to 6.15x); (2) Multinomial Naive Bayes natural language "
        "processing classifying plot summaries into genres with 50.49% accuracy; (3) Balanced Random Forest predicting movie rating tiers "
        "with 63.01% accuracy without data leakage; (4) Regularized Lasso regression performing sparse feature selection (Test RMSE: 0.4683); "
        "and (5) Box-Jenkins ARIMA(3,1,3) time series modeling achieving a 22.47% forecast improvement over naive persistence. Finally, "
        "a weighted hybrid recommender engine evaluated offline via Leave-One-Out Hit-Rate@10 (0.0467) and served through an interactive "
        "Flask web application provides practical, explainable deployment.",
        style_body
    ))

    story.append(Paragraph("2. INTRODUCTION", style_h1))
    story.append(Paragraph(
        "Modern media distribution relies extensively on data mining algorithms to extract actionable intelligence from vast, "
        "heterogeneous consumer interaction logs. Online streaming services generate continuous transaction feeds, descriptive metadata, "
        "and temporal catalog releases. Examining these multidimensional datasets manually is infeasible due to extreme sparsity and "
        "combinatorial complexity. Data mining bridges this analytical gap by uncovering latent associations between titles, classifying "
        "unstructured narrative synopses, projecting temporal catalog growth, and personalizing viewing queues.",
        style_body
    ))
    story.append(Paragraph(
        "By integrating traditional market-basket analysis (Apriori) with statistical time series forecasting (ARIMA), text mining "
        "(TF-IDF vectorization), and supervised ensemble learning (Random Forest), content providers can reliably predict audience reception, "
        "optimize procurement budgets, and deliver transparent, explainable recommendations to subscribers without black-box opacity.",
        style_body
    ))

    story.append(Paragraph("3. PROBLEM STATEMENT", style_h1))
    story.append(Paragraph(
        "To ingest, cleanse, and structure heterogeneous movie rating logs and streaming catalog metadata into an analytical Fact "
        "Constellation warehouse, and to apply association rule mining, text classification, predictive regression, and ARIMA temporal "
        "modeling so that co-viewing affinities, rating tiers, and platform catalog trends can be reliably discovered, evaluated, and served "
        "through an explainable recommendation interface.",
        style_body
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 3: 4. AIM & OBJECTIVES, 5. SCOPE, 6. SYLLABUS MAPPING, 7. DATASET
    # =========================================================================
    story.append(Paragraph("4. AIM AND OBJECTIVES", style_h1))
    story.append(Paragraph(
        "<b>Aim:</b> To design, develop, and empirically evaluate an integrated Movie and OTT Recommendation and Trend Analysis system "
        "using multidimensional warehousing and core data mining algorithms.",
        style_body
    ))
    story.append(Paragraph("&bull; Construct an idempotent ETL pipeline to assemble a SQLite Fact Constellation schema with conformed date dimensions and 0 orphan keys.", style_bullet))
    story.append(Paragraph("&bull; Formulate user viewing records into transactional baskets and apply Apriori to mine frequent itemsets and co-viewing rules.", style_bullet))
    story.append(Paragraph("&bull; Develop a text mining pipeline using NLTK Porter Stemming, TF-IDF vectorization, and Multinomial Naive Bayes to classify plot synopses.", style_bullet))
    story.append(Paragraph("&bull; Engineer leak-free metadata features and train Decision Tree and Random Forest classifiers to predict movie rating categories.", style_bullet))
    story.append(Paragraph("&bull; Train and benchmark Linear, Ridge, and Lasso regression models to estimate continuous movie ratings and analyze active L1 feature weights.", style_bullet))
    story.append(Paragraph("&bull; Conduct ADF stationarity tests and fit ARIMA models on monthly additions to forecast catalog expansion with 95% confidence intervals.", style_bullet))
    story.append(Paragraph("&bull; Synthesize Content-based, Item-Item Collaborative Filtering, and Apriori rules into an explainable Hybrid Recommender (evaluated via Hit-Rate@10).", style_bullet))
    story.append(Paragraph("&bull; Deploy an interactive Flask web application and Chart.js analytical dashboard for real-time inference.", style_bullet))

    story.append(Paragraph("5. SCOPE OF THE PROJECT", style_h1))
    story.append(Paragraph(
        "The project encompasses end-to-end data processing, from raw CSV cleansing to analytical warehousing, model training, offline "
        "mathematical evaluation, and interactive dashboard deployment. All reported metrics reflect empirical test evaluations on fixed splits "
        "without data leakage. The project operates on benchmark historical catalogs (MovieLens 100k and Netflix Titles); it does not establish "
        "causal human behavior nor perform automated live video transcoding.",
        style_body
    ))

    story.append(Paragraph("6. SYLLABUS MAPPING", style_h1))
    syllabus_data = [
        [Paragraph("Syllabus Unit / Topic", style_table_h), Paragraph("Implementation in this Project", style_table_h), Paragraph("Outcome", style_table_h)],
        [Paragraph("Unit 1: Warehousing & Preprocessing", style_table_c_bold), Paragraph("Fact Constellation schema in SQLite (warehouse.db), conformed dim_date, OLAP queries.", style_table_c), Paragraph("CO1", style_table_c_bold)],
        [Paragraph("Unit 2: Transformation & Reduction", style_table_c_bold), Paragraph("Regex release year extraction, genre one-hot encoding, log transform, Porter Stemming, TF-IDF.", style_table_c), Paragraph("CO1, CO2", style_table_c_bold)],
        [Paragraph("Unit 3: Regression Analysis", style_table_c_bold), Paragraph("OLS, Ridge (L2 shrinkage), Lasso (L1 feature selection), Decision Tree Regressor for continuous ratings.", style_table_c), Paragraph("CO3, CO4", style_table_c_bold)],
        [Paragraph("Unit 4: Association Rules & Classification", style_table_c_bold), Paragraph("Apriori algorithm (support/confidence/lift), Multinomial Naive Bayes, Decision Trees, Random Forest.", style_table_c), Paragraph("CO2, CO4", style_table_c_bold)],
        [Paragraph("Unit 5: Advanced Mining & Recommenders", style_table_c_bold), Paragraph("ADF stationarity differencing (d=1), ARIMA(3,1,3) forecasting, Content-based TF-IDF, Item-CF, Hit-Rate@10.", style_table_c), Paragraph("CO5", style_table_c_bold)],
    ]
    syllabus_table = Table(syllabus_data, colWidths=[42 * mm, 111 * mm, 20.24 * mm])
    syllabus_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#D9EAF7')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0, 0), (-1, -1), 2.0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.0),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
    ]))
    story.append(syllabus_table)
    story.append(Spacer(1, 2 * mm))

    story.append(Paragraph("7. DATASET DESCRIPTION", style_h1))
    dataset_data = [
        [Paragraph("Dataset File", style_table_h), Paragraph("Records / Shape", style_table_h), Paragraph("Raw Attributes", style_table_h), Paragraph("Transformation & Cleansing", style_table_h)],
        [Paragraph("ratings.csv", style_table_c_bold), Paragraph("100,836 rows (610 users)", style_table_c), Paragraph("userId, movieId, rating, timestamp", style_table_c), Paragraph("Converted epoch to UTC ISO date; mapped rating tiers: Low (<3), Med (3-3.9), High (>=4). 0 orphan keys.", style_table_c)],
        [Paragraph("movies.csv", style_table_c_bold), Paragraph("9,742 movies", style_table_c), Paragraph("movieId, title, genres", style_table_c), Paragraph("Extracted 4-digit year via regex; split pipe genres into conformed dim_genre; cleaned titles.", style_table_c)],
        [Paragraph("tags.csv & links.csv", style_table_c_bold), Paragraph("3,683 tags / 9,742 links", style_table_c), Paragraph("userId, movieId, tag, imdbId", style_table_c), Paragraph("Cleaned text, lowercased, merged for content-based TF-IDF document profiles.", style_table_c)],
        [Paragraph("netflix_titles.csv", style_table_c_bold), Paragraph("8,807 titles", style_table_c), Paragraph("show_id, type, director, cast, country, date_added, duration, description", style_table_c), Paragraph("Stripped whitespace, parsed date_added into conformed dates, parsed duration, split multi-genres into bridge table.", style_table_c)],
    ]
    dataset_table = Table(dataset_data, colWidths=[28 * mm, 34 * mm, 45 * mm, 66.24 * mm])
    dataset_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#D9EAF7')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0, 0), (-1, -1), 2.0),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2.0),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
    ]))
    story.append(dataset_table)
    story.append(PageBreak())

    # =========================================================================
    # PAGE 4: 8. METHODOLOGY, 9. ARCHITECTURE, 10. SOFTWARE & TOOLS
    # =========================================================================
    story.append(Paragraph("8. METHODOLOGY AND ALGORITHM", style_h1))
    story.append(Paragraph("<b>Step 1 &ndash; Ingestion &amp; Discovery:</b> Robust automated path discovery scanning nested directories, reading raw CSVs into DataFrames.", style_bullet))
    story.append(Paragraph("<b>Step 2 &ndash; Fact Constellation ETL:</b> Automated idempotent relational pipeline loading 11 normalized tables into SQLite with primary/foreign keys.", style_bullet))
    story.append(Paragraph("<b>Step 3 &ndash; Basket Assembly &amp; Apriori Mining:</b> Formulated 599 users &times; 300 movies binary co-viewing matrix. Extracted frequent itemsets and association rules using mlxtend.", style_bullet))
    story.append(Paragraph("<b>Step 4 &ndash; Text Mining &amp; Naive Bayes:</b> Plot synopsis cleaning, Porter stemming, TF-IDF vectorization (20,000 max features), and MultinomialNB tuning.", style_bullet))
    story.append(Paragraph("<b>Step 5 &ndash; Rating Tier Classification:</b> Engineered 22 leak-free features (19 genre flags, release year, log ratings, tag counts) and trained Random Forest.", style_bullet))
    story.append(Paragraph("<b>Step 6 &ndash; Regularized Regression:</b> Evaluated Linear, Ridge, and Lasso regression models with cross-validation to analyze L1 coefficient sparsity.", style_bullet))
    story.append(Paragraph("<b>Step 7 &ndash; Time Series Differencing &amp; ARIMA:</b> Stationarity verification via ADF unit-root test (d=1), grid search over (p,1,q), and 12-month holdout evaluation.", style_bullet))
    story.append(Paragraph("<b>Step 8 &ndash; Hybrid Recommender &amp; Serving:</b> Linear ensemble of Content-based, Item-CF, and Apriori rules served through a responsive Flask web app.", style_bullet))
    
    story.append(Paragraph("Key Measures and Mathematical Formulations", style_h2))
    math_data = [
        [Paragraph("Measure / Algorithm", style_table_h), Paragraph("Mathematical Formula / Objective Function", style_table_h), Paragraph("Interpretation in Project Context", style_table_h)],
        [Paragraph("Support", style_table_c_bold), Paragraph("support(A &rarr; B) = P(A &cup; B) = Count(A &cup; B) / N", style_table_c), Paragraph("Fraction of user viewing baskets containing both titles.", style_table_c)],
        [Paragraph("Confidence", style_table_c_bold), Paragraph("confidence(A &rarr; B) = P(A &cup; B) / P(A)", style_table_c), Paragraph("Conditional probability that a user likes title B given they liked A.", style_table_c)],
        [Paragraph("Lift", style_table_c_bold), Paragraph("lift(A &rarr; B) = confidence(A &rarr; B) / support(B)", style_table_c), Paragraph("Affinity strength: Lift > 1 indicates positive association above independence.", style_table_c)],
        [Paragraph("Naive Bayes", style_table_c_bold), Paragraph("P(c | d) &prop; P(c) &prod; P(w_i | c)", style_table_c), Paragraph("Posterior genre probability given word tokens in plot synopsis.", style_table_c)],
        [Paragraph("Lasso (L1)", style_table_c_bold), Paragraph("min (1/2n)||y - X&beta;||<sup>2</sup> + &lambda;&Sigma;|&beta;<sub>j</sub>|", style_table_c), Paragraph("L1-penalized regression driving uninformative genre weights to exactly 0.", style_table_c)],
        [Paragraph("ARIMA(p,d,q)", style_table_c_bold), Paragraph("&phi;(B)(1-B)<sup>d</sup> y<sub>t</sub> = &theta;(B)&epsilon;<sub>t</sub>", style_table_c), Paragraph("Autoregressive integrated moving average for monthly OTT additions.", style_table_c)],
        [Paragraph("Hybrid Score", style_table_c_bold), Paragraph("S_h = 0.40 S_content + 0.30 S_cf + 0.30 S_rule", style_table_c), Paragraph("Dynamic linear combination with weight normalization for missing signals.", style_table_c)],
    ]
    math_table = Table(math_data, colWidths=[30 * mm, 68 * mm, 75.24 * mm])
    math_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#D9EAF7')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0, 0), (-1, -1), 1.8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
    ]))
    story.append(math_table)

    story.append(Paragraph("9. SYSTEM ARCHITECTURE", style_h1))
    arch_data = [
        [Paragraph("Pipeline Stage", style_table_h), Paragraph("Component & Implementation Mechanism", style_table_h), Paragraph("Generated Artifact", style_table_h)],
        [Paragraph("1. Warehouse ETL", style_table_c_bold), Paragraph("SQLite Fact Constellation schema with 11 relational tables & 11 indexes", style_table_c), Paragraph("warehouse.db", style_table_c)],
        [Paragraph("2. Association Engine", style_table_c_bold), Paragraph("mlxtend Apriori frequent itemsets on 599x300 co-viewing matrix", style_table_c), Paragraph("models/rules.joblib", style_table_c)],
        [Paragraph("3. Classification", style_table_c_bold), Paragraph("MultinomialNB (genre text) & Balanced Random Forest (rating tier)", style_table_c), Paragraph("models/*_classifier.joblib", style_table_c)],
        [Paragraph("4. Trend & Regression", style_table_c_bold), Paragraph("LassoCV rating predictor & statsmodels ARIMA(3,1,3) monthly additions", style_table_c), Paragraph("models/arima_models.joblib", style_table_c)],
        [Paragraph("5. Serving Engine", style_table_c_bold), Paragraph("Hybrid Recommender + Flask REST API + Jinja2 + Chart.js dashboard", style_table_c), Paragraph("app.py / web interface", style_table_c)],
    ]
    arch_table = Table(arch_data, colWidths=[35 * mm, 98 * mm, 40.24 * mm])
    arch_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#D9EAF7')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0, 0), (-1, -1), 1.8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.8),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
    ]))
    story.append(arch_table)

    story.append(Paragraph("10. SOFTWARE AND TOOLS", style_h1))
    story.append(Paragraph(
        "<b>Core Language:</b> Python 3.12 &bull; "
        "<b>Data Handling:</b> Pandas 2.3.3, NumPy 2.5.2 &bull; "
        "<b>Database:</b> SQLite 3 (ACID-compliant Fact Constellation) &bull; "
        "<b>Machine Learning:</b> Scikit-Learn 1.6.1 &bull; "
        "<b>Pattern Mining:</b> mlxtend 0.25.0 &bull; "
        "<b>Time Series:</b> statsmodels 0.15.0 &bull; "
        "<b>NLP:</b> NLTK 3.10.4 &bull; "
        "<b>Web & UI:</b> Flask 3.1.3, Jinja2, Chart.js &bull; "
        "<b>Testing:</b> Pytest 9.1.1 (19/19 passing), Playwright 1.64.0.",
        style_body
    ))
    story.append(PageBreak())

    # =========================================================================
    # PAGE 5: 11. EVALUATION, 12. LIMITATIONS, 13. CONCLUSION, 14. REFERENCES
    # =========================================================================
    story.append(Paragraph("11. EXPERIMENTAL OUTPUT AND EVALUATION", style_h1))
    story.append(Paragraph(
        "All metrics reported below were produced by the automated training pipeline (<code>python train_all.py</code>) on reproducible holdout splits:",
        style_body
    ))
    
    # Benchmarks table
    eval_data = [
        [Paragraph("Module", style_table_h), Paragraph("Algorithm", style_table_h), Paragraph("Primary Metric", style_table_h), Paragraph("Secondary Metric", style_table_h), Paragraph("Optimal Parameter", style_table_h)],
        [Paragraph("Genre Text Mining", style_table_c_bold), Paragraph("Multinomial Naive Bayes", style_table_c), Paragraph("<b>Accuracy: 50.49%</b>", style_table_c), Paragraph("Macro F1: 0.4757", style_table_c), Paragraph("&alpha; = 0.1", style_table_c)],
        [Paragraph("Genre Baseline", style_table_c_bold), Paragraph("Logistic Regression", style_table_c), Paragraph("Accuracy: 51.54%", style_table_c), Paragraph("Macro F1: 0.5215", style_table_c), Paragraph("C = 1.0", style_table_c)],
        [Paragraph("Rating Tier Class", style_table_c_bold), Paragraph("Decision Tree", style_table_c), Paragraph("Accuracy: 50.68%", style_table_c), Paragraph("Macro F1: 0.4766", style_table_c), Paragraph("max_depth = 10", style_table_c)],
        [Paragraph("Rating Tier Class", style_table_c_bold), Paragraph("<b>Random Forest</b>", style_table_c), Paragraph("<b>Accuracy: 63.01%</b>", style_table_c), Paragraph("Macro F1: 0.5576", style_table_c), Paragraph("n_est=200, depth=10", style_table_c)],
        [Paragraph("Rating Regression", style_table_c_bold), Paragraph("<b>Lasso (L1)</b>", style_table_c), Paragraph("<b>Test RMSE: 0.4683</b>", style_table_c), Paragraph("Test R&sup2;: 0.3105", style_table_c), Paragraph("&alpha; = 0.00041", style_table_c)],
        [Paragraph("Rating Regression", style_table_c_bold), Paragraph("Ridge (L2)", style_table_c), Paragraph("Test RMSE: 0.4684", style_table_c), Paragraph("Test R&sup2;: 0.3103", style_table_c), Paragraph("&alpha; = 10.0", style_table_c)],
        [Paragraph("Time Series ARIMA", style_table_c_bold), Paragraph("ARIMA(3,1,3) on Netflix", style_table_c), Paragraph("<b>Test RMSE: 41.42</b>", style_table_c), Paragraph("Naive RMSE: 53.42", style_table_c), Paragraph("<b>+22.47% vs. Naive</b>", style_table_c)],
    ]
    eval_table = Table(eval_data, colWidths=[33 * mm, 42 * mm, 36 * mm, 32 * mm, 30.24 * mm])
    eval_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#D9EAF7')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0, 0), (-1, -1), 1.6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
    ]))
    story.append(eval_table)
    story.append(Spacer(1, 2 * mm))

    # Recommender table & Apriori
    rec_data = [
        [Paragraph("Recommendation Method", style_table_h), Paragraph("Hit-Rate @ 10", style_table_h), Paragraph("Precision @ 10", style_table_h), Paragraph("Recall @ 10", style_table_h), Paragraph("Top Discovered Apriori Rule", style_table_h)],
        [Paragraph("Content-Based (TF-IDF)", style_table_c), Paragraph("0.0067", style_table_c), Paragraph("0.0007", style_table_c), Paragraph("0.0067", style_table_c), Paragraph("Kill Bill: Vol. 1 &rarr; Kill Bill: Vol. 2", style_table_c_bold)],
        [Paragraph("Apriori Association Rules", style_table_c), Paragraph("0.0267", style_table_c), Paragraph("0.0027", style_table_c), Paragraph("0.0267", style_table_c), Paragraph("Support: 10.85% &bull; Confidence: 73.86%", style_table_c)],
        [Paragraph("Item-Item Collaborative", style_table_c), Paragraph("<b>0.0467</b>", style_table_c), Paragraph("0.0047", style_table_c), Paragraph("0.0467", style_table_c), Paragraph("<b>Lift: 6.15x (Strongest Rule)</b>", style_table_c_bold)],
        [Paragraph("<b>Hybrid Ensemble Engine</b>", style_table_c_bold), Paragraph("<b>0.0467</b>", style_table_c_bold), Paragraph("<b>0.0047</b>", style_table_c_bold), Paragraph("<b>0.0467</b>", style_table_c_bold), Paragraph("The Godfather & Goodfellas &rarr; Part II (Lift: 4.43x)", style_table_c)],
    ]
    rec_table = Table(rec_data, colWidths=[38 * mm, 24 * mm, 25 * mm, 23 * mm, 63.24 * mm])
    rec_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#D9EAF7')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('TOPPADDING', (0, 0), (-1, -1), 1.6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.6),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
    ]))
    story.append(rec_table)

    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph("12. LIMITATIONS AND FUTURE SCOPE", style_h1))
    story.append(Paragraph(
        "<b>Limitations:</b> (1) The MovieLens 100k catalog spans 1996&ndash;2018 theatrical releases whereas Netflix additions span 2008&ndash;2021, reflecting divergent distribution channels. (2) Netflix synopses are concise (~25 words), constraining lexical separation for subtle genres. (3) Natural user rating skew required class-balanced loss weighting during classification.",
        style_body
    ))
    story.append(Paragraph(
        "<b>Future Scope:</b> (1) Transition from cosine similarity to Matrix Factorization (SVD) and Neural Collaborative Filtering (NCF). (2) Ingest streaming clickstream telemetry (completion rate, pause points) via Kafka. (3) Containerized Kubernetes microservices deployment with continuous retraining.",
        style_body
    ))

    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph("13. CONCLUSION", style_h1))
    story.append(Paragraph(
        "This mini-project successfully engineered an integrated Data Mining and Analytics architecture fulfilling all course outcomes (CO1&ndash;CO5). An automated ETL pipeline structured 100,836 ratings and 8,807 streaming titles into an analytical SQLite Fact Constellation warehouse. Association mining mined 935 high-lift co-viewing rules, text mining classified plot synopses, Random Forest predicted rating tiers with 63.01% accuracy, Lasso regression isolated critical genre features, and ARIMA(3,1,3) forecasted monthly platform additions with a 22.47% error reduction over baseline persistence. The hybrid recommender and responsive Flask dashboard demonstrate how multidimensional warehousing and core data mining algorithms unite into a practical production system.",
        style_body
    ))

    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph("14. REFERENCES", style_h1))
    refs = [
        "1. Agrawal, R., Imielinski, T., &amp; Swami, A. (1993). <i>Mining Association Rules between Sets of Items in Large Databases</i>. ACM SIGMOD.",
        "2. Box, G. E., Jenkins, G. M., Reinsel, G. C., &amp; Ljung, G. M. (2015). <i>Time Series Analysis: Forecasting and Control</i> (5th ed.). Wiley.",
        "3. Breiman, L. (2001). <i>Random Forests</i>. Machine Learning, 45(1), 5&ndash;32.",
        "4. Harper, F. M., &amp; Konstan, J. A. (2015). <i>The MovieLens Datasets: History and Context</i>. ACM TiiS, 5(4), 1&ndash;19.",
        "5. Tibshirani, R. (1996). <i>Regression Shrinkage and Selection via the Lasso</i>. J. R. Statist. Soc. B, 58(1), 267&ndash;288.",
        "6. Savitribai Phule Pune University &bull; CS3203T-A Data Mining and Analytics Course Curriculum.",
    ]
    for r in refs:
        story.append(Paragraph(r, style_bullet))

    story.append(Spacer(1, 2 * mm))
    story.append(Paragraph(
        "<i>Submission Verification: Submitted by Vinit Shinde (Roll No: 37, PRN: RBT24CS178). All metrics verified via train_all.py test runs.</i>",
        style_footer_note
    ))

    # Build document
    doc.build(story, onFirstPage=make_footer, onLaterPages=make_footer)

    # Verification of page count
    reader = pypdf.PdfReader(output_path)
    page_count = len(reader.pages)
    print(f"[SUCCESS] Compiled '{output_path}' -> Exact Page Count: {page_count}")
    return page_count

if __name__ == '__main__':
    out = 'reports/PROJECT_REPORT.pdf'
    if len(sys.argv) > 1:
        out = sys.argv[1]
    cnt = build_pdf(out)
    if cnt != 5:
        print(f"[WARNING] Expected 5 pages, but got {cnt}!")
        sys.exit(1)
    else:
        print("[PERFECT] Verified exactly 5 pages!")
