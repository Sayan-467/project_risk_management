"""
generate_architecture_diagrams.py
=================================
Generates publication-ready IEEE architecture diagrams for the research paper:
"AI-Powered IT Project Risk Management Using a Multi-Agent Intelligence Framework"

Deliverables:
1. results/architecture/figure1_overall_architecture.png (>= 300 DPI)
2. results/architecture/figure1_overall_architecture.svg (vector)
3. results/architecture/figure2_multi_agent_workflow.png (>= 300 DPI)
4. results/architecture/figure2_multi_agent_workflow.svg (vector)

Design Specifications:
- Strict IEEE formatting standards: pure white background, crisp vector geometry, high-contrast academic palette.
- Clear distinction of implementation paradigms: Procedural Python components, Domain Lexicon NLP, TF-IDF Vector retrieval, and Scikit-Learn ML.
- Mathematically aligned, non-intersecting dataflow and control connectors.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle

OUTPUT_DIR = os.path.join("results", "architecture")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Color Palette (IEEE Standard: High-Contrast, Colorblind-Friendly)
# -----------------------------------------------------------------------------
PALETTE = {
    "bg": "#FFFFFF",
    "text_dark": "#0F172A",
    "text_body": "#334155",
    "text_muted": "#64748B",
    
    # Layer 1: Data Sources (Slate Blue)
    "l1_bg": "#F8FAFC",
    "l1_card": "#FFFFFF",
    "l1_border": "#94A3B8",
    "l1_header": "#1E293B",
    
    # Layer 2: Preprocessing (Sky Blue)
    "l2_bg": "#F0F9FF",
    "l2_card": "#FFFFFF",
    "l2_border": "#0284C7",
    "l2_header": "#0369A1",
    
    # Layer 3: Coordination (Royal Purple / Indigo)
    "l3_bg": "#FAF5FF",
    "l3_card": "#FFFFFF",
    "l3_border": "#7E22CE",
    "l3_header": "#581C87",
    
    # Layer 4: Analytical Components (Teal & Blue)
    "l4_bg": "#F0FDFA",
    "l4_card": "#FFFFFF",
    "l4_border_teal": "#0D9488",
    "l4_header_teal": "#115E59",
    "l4_border_blue": "#0284C7",
    "l4_header_blue": "#0369A1",
    
    # Layer 5: Risk Assessment (Amber / Ochre)
    "l5_bg": "#FFFBEB",
    "l5_card": "#FFFFFF",
    "l5_border": "#D97706",
    "l5_header": "#92400E",
    
    # Layer 6: Reporting (Emerald / Forest)
    "l6_bg": "#F0FDF4",
    "l6_card": "#FFFFFF",
    "l6_border": "#16A34A",
    "l6_header": "#166534",
    
    # Layer 7: Final Outputs (Deep Slate & Mint)
    "l7_bg": "#F8FAFC",
    "l7_card": "#FFFFFF",
    "l7_border": "#475569",
    "l7_header": "#0F172A",
    
    # Badges
    "badge_procedural": "#6B21A8",   # Deep Purple
    "badge_lexicon": "#0369A1",      # Deep Sky
    "badge_retrieval": "#0F766E",    # Deep Teal
    "badge_ml": "#B45309",           # Deep Amber
    "badge_pipeline": "#334155",     # Slate
    "badge_output": "#15803D",       # Forest Green
    "badge_input": "#475569",        # Slate Muted
    
    # Arrows
    "arrow_dark": "#1E293B",
    "arrow_blue": "#0284C7",
    "arrow_purple": "#7E22CE",
    "arrow_teal": "#0D9488"
}


def draw_card(
    ax, x, y, w, h,
    title, subtitle=None, badge=None, badge_color="#475569",
    bullets=None,
    bg_color="#FFFFFF", border_color="#94A3B8", header_color="#0F172A",
    corner_radius=0.12, lw=1.2, zorder=2, font_size_title=10.5
):
    """Draws a clean, publication-ready card with a header, badge, and items."""
    # Main card box
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={corner_radius}",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=lw,
        zorder=zorder
    )
    ax.add_patch(box)
    
    # Header Title
    title_y = y + h - 0.26
    ax.text(
        x + 0.22, title_y, title,
        fontsize=font_size_title, fontweight="bold", color=header_color,
        va="center", ha="left", zorder=zorder+1, fontfamily="Arial"
    )
    
    # Badge (upper right)
    if badge:
        badge_w = len(badge) * 0.10 + 0.28
        badge_h = 0.26
        badge_x = x + w - badge_w - 0.18
        badge_y = title_y - badge_h / 2
        badge_box = FancyBboxPatch(
            (badge_x, badge_y), badge_w, badge_h,
            boxstyle=f"round,pad=0,rounding_size=0.06",
            facecolor=badge_color,
            edgecolor=badge_color,
            linewidth=0.5,
            zorder=zorder+1
        )
        ax.add_patch(badge_box)
        ax.text(
            badge_x + badge_w / 2, title_y, badge,
            fontsize=7.2, fontweight="bold", color="#FFFFFF",
            va="center", ha="center", zorder=zorder+2, fontfamily="Arial"
        )
        
    # Subtitle / description line
    curr_y = title_y - 0.24
    if subtitle:
        ax.text(
            x + 0.22, curr_y, subtitle,
            fontsize=8.0, fontstyle="italic", color="#64748B",
            va="center", ha="left", zorder=zorder+1, fontfamily="Arial"
        )
        curr_y -= 0.22
    else:
        curr_y -= 0.06
        
    # Divider line under title
    ax.plot([x + 0.18, x + w - 0.18], [curr_y + 0.08, curr_y + 0.08], color=border_color, lw=0.75, zorder=zorder+1)
    
    # Bullets
    if bullets:
        bullet_spacing = (curr_y - (y + 0.12)) / max(len(bullets), 1)
        bullet_spacing = min(bullet_spacing, 0.24)
        
        for i, bullet in enumerate(bullets):
            by = curr_y - (i * bullet_spacing)
            if by < y + 0.08:
                break
            # Bullet marker
            ax.plot([x + 0.28], [by], marker="s", markersize=2.8, color=header_color, zorder=zorder+1)
            # Bullet text
            ax.text(
                x + 0.42, by, bullet,
                fontsize=7.8, color="#334155",
                va="center", ha="left", zorder=zorder+1, fontfamily="Arial"
            )


def draw_container_layer(
    ax, x, y, w, h,
    layer_num, layer_title, layer_sub="",
    bg_color="#F8FAFC", border_color="#CBD5E1", header_color="#334155"
):
    """Draws a surrounding layer frame."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0,rounding_size=0.18",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=1.1,
        linestyle="--",
        zorder=1
    )
    ax.add_patch(box)
    
    # Layer Tab / Badge on top left
    tab_w = 4.3
    tab_h = 0.34
    tab_x = x + 0.25
    tab_y = y + h - tab_h / 2
    tab = FancyBboxPatch(
        (tab_x, tab_y), tab_w, tab_h,
        boxstyle="round,pad=0,rounding_size=0.08",
        facecolor=header_color,
        edgecolor=header_color,
        linewidth=0.5,
        zorder=3
    )
    ax.add_patch(tab)
    
    full_title = f"LAYER {layer_num}: {layer_title.upper()}"
    ax.text(
        tab_x + tab_w / 2, tab_y + tab_h / 2, full_title,
        fontsize=8.2, fontweight="bold", color="#FFFFFF",
        va="center", ha="center", zorder=4, fontfamily="Arial"
    )
    
    if layer_sub:
        ax.text(
            tab_x + tab_w + 0.2, tab_y + tab_h / 2, f"({layer_sub})",
            fontsize=7.8, fontstyle="italic", color="#64748B",
            va="center", ha="left", zorder=3, fontfamily="Arial"
        )


def draw_arrow(
    ax, x1, y1, x2, y2,
    label=None, color="#1E293B", lw=1.4,
    label_side="top", label_offset=0.12, fontsize=7.2,
    connectionstyle=None, zorder=5
):
    """Draws an IEEE-compliant arrow with an optional label."""
    if connectionstyle:
        arrowprops = dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            mutation_scale=11,
            connectionstyle=connectionstyle
        )
    else:
        arrowprops = dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            mutation_scale=11
        )
        
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=arrowprops,
        zorder=zorder
    )
    
    if label:
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        
        lx, ly = mx, my
        va, ha = "center", "center"
        if label_side == "top":
            ly += label_offset
            va = "bottom"
        elif label_side == "bottom":
            ly -= label_offset
            va = "top"
        elif label_side == "left":
            lx -= label_offset
            ha = "right"
        elif label_side == "right":
            lx += label_offset
            ha = "left"
            
        ax.text(
            lx, ly, label,
            fontsize=fontsize, fontweight="semibold", color=color,
            va=va, ha=ha, zorder=zorder+1, fontfamily="Arial",
            bbox=dict(boxstyle="round,pad=0.15,rounding_size=0.05", facecolor="#FFFFFF", edgecolor=color, alpha=0.92, lw=0.6)
        )


# =============================================================================
# FIGURE 1: OVERALL SYSTEM ARCHITECTURE
# =============================================================================
def generate_figure1():
    print("Generating Figure 1: Overall System Architecture...")
    fig, ax = plt.subplots(figsize=(19, 14.5), dpi=300)
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.set_xlim(0, 19)
    ax.set_ylim(0, 14.5)
    ax.axis('off')
    
    # -------------------------------------------------------------------------
    # Header Title & Subtitle
    # -------------------------------------------------------------------------
    ax.text(
        9.5, 14.15,
        "Figure 1: Overall System Architecture of the AI-Powered IT Project Risk Management Framework",
        fontsize=13.5, fontweight="bold", color=PALETTE["text_dark"],
        ha="center", va="center", fontfamily="Arial"
    )
    ax.text(
        9.5, 13.85,
        "Hierarchical multi-layer architecture integrating procedural agent dispatch, domain lexicon NLP, TF-IDF vector retrieval, and supervised ML classification",
        fontsize=9.0, fontstyle="italic", color=PALETTE["text_muted"],
        ha="center", va="center", fontfamily="Arial"
    )
    
    # -------------------------------------------------------------------------
    # LAYER 1: Input Data Sources (y: 11.65 - 13.45, h = 1.80)
    # -------------------------------------------------------------------------
    draw_container_layer(
        ax, 0.4, 11.65, 18.2, 1.80,
        1, "Input Data Sources", "Multi-Modal Ingestion: Telemetry, Unstructured Text, & Historical Cases",
        bg_color=PALETTE["l1_bg"], border_color=PALETTE["l1_border"], header_color=PALETTE["l1_header"]
    )
    
    # Card 1.1: Structured IT Project Data (Column 1)
    draw_card(
        ax, 0.65, 11.80, 5.65, 1.45,
        title="Structured IT Project Telemetry",
        subtitle="19 Project Features (17 Numerical + 2 Categorical)",
        badge="[Tabular Modality]", badge_color=PALETTE["badge_input"],
        bullets=[
            "Execution Progress: Completion %, Schedule Delay, Budget Variance",
            "Resource Allocations: Team Size, Velocity, Experience, Attrition Rate",
            "Software Quality: Defect Density, Open Bugs, Critical Issues",
            "Categorical Factors: Project Complexity & Communication Health"
        ],
        border_color=PALETTE["l1_border"], header_color=PALETTE["l1_header"]
    )
    
    # Card 1.2: Project Communications & Notes (Column 2)
    draw_card(
        ax, 6.65, 11.80, 5.65, 1.45,
        title="Project Communications & Feedback",
        subtitle="Unstructured Textual Telemetry",
        badge="[Textual Modality]", badge_color=PALETTE["badge_input"],
        bullets=[
            "Sprint Retrospective Meeting Logs & Standup Status Notes",
            "Stakeholder Communications, Review Feedback & Escalations",
            "Team Sentiment Expressions & Organizational Morale Signals",
            "Real-time Qualitative Early-Warning Risk Indicators"
        ],
        border_color=PALETTE["l1_border"], header_color=PALETTE["l1_header"]
    )
    
    # Card 1.3: Historical Case Studies (Column 3)
    draw_card(
        ax, 12.65, 11.80, 5.65, 1.45,
        title="Historical IT Project Case Studies",
        subtitle="15 Curated Software Engineering Cases",
        badge="[Knowledge Base]", badge_color=PALETTE["badge_input"],
        bullets=[
            "Past IT Projects Across Healthcare, Finance, ERP & Web Platforms",
            "Documented Root Causes, Failure Modes & Architecture Shifts",
            "Empirical Schedule / Cost Overrun Post-Mortem Records",
            "Contextual Mitigation Strategies & Observed Lessons Learned"
        ],
        border_color=PALETTE["l1_border"], header_color=PALETTE["l1_header"]
    )
    
    # -------------------------------------------------------------------------
    # LAYER 2: Data Preprocessing & Preparation Pipeline (y: 9.75 - 11.20, h = 1.45)
    # -------------------------------------------------------------------------
    draw_container_layer(
        ax, 0.4, 9.75, 18.2, 1.45,
        2, "Data Preprocessing & Preparation", "Modality-Specific Feature Transformation & Corpus Preparation",
        bg_color=PALETTE["l2_bg"], border_color=PALETTE["l2_border"], header_color=PALETTE["l2_header"]
    )
    
    # Card 2.1: Tabular Preprocessing (Column 1)
    draw_card(
        ax, 0.65, 9.90, 5.65, 1.12,
        title="Tabular Preprocessing Pipeline",
        subtitle="SimpleImputer & StandardScaler (Scikit-Learn)",
        badge="[Leakage-Free]", badge_color=PALETTE["badge_pipeline"],
        bullets=[
            "Median imputation for numericals; Mode imputation for categoricals",
            "Z-score standardization (mean=0, variance=1) for ML models",
            "OneHotEncoder(handle_unknown='ignore') for project domain/type",
            "Strictly fitted on stratified 80% train split (No test leakage)"
        ],
        border_color=PALETTE["l2_border"], header_color=PALETTE["l2_header"]
    )
    
    # Card 2.2: Text Preparation (Column 2)
    draw_card(
        ax, 6.65, 9.90, 5.65, 1.12,
        title="Text Ingestion & Tokenizer Setup",
        subtitle="String Cleaning, Tokenization & Lexicon Mapping",
        badge="[NLP Prep]", badge_color=PALETTE["badge_pipeline"],
        bullets=[
            "Strips noise, punctuation, formatting artifacts & casing",
            "Extracts unigrams and multi-word software domain phrases",
            "Maps tokens to domain sentiment & risk polarity dictionary",
            "Prepares normalized text stream for Market Analysis Agent"
        ],
        border_color=PALETTE["l2_border"], header_color=PALETTE["l2_header"]
    )
    
    # Card 2.3: Historical Corpus Preparation (Column 3)
    draw_card(
        ax, 12.65, 9.90, 5.65, 1.12,
        title="Historical Corpus Vectorizer",
        subtitle="TF-IDF Term Weighting & Corpus Indexing",
        badge="[Vector Prep]", badge_color=PALETTE["badge_pipeline"],
        bullets=[
            "Indexes 15 historical project case narratives and risk logs",
            "Extracts sublinear TF-IDF term weights (unigram + bigram)",
            "Builds sparse document-term vector matrix for fast lookup",
            "Zero ChromaDB server dependency (Local in-memory vectors)"
        ],
        border_color=PALETTE["l2_border"], header_color=PALETTE["l2_header"]
    )
    
    # -------------------------------------------------------------------------
    # LAYER 3: Coordination Layer (y: 7.90 - 9.30, h = 1.40)
    # -------------------------------------------------------------------------
    draw_container_layer(
        ax, 0.4, 7.90, 18.2, 1.40,
        3, "Coordination Layer", "Centralized Procedural Orchestration & Latency Profiling",
        bg_color=PALETTE["l3_bg"], border_color=PALETTE["l3_border"], header_color=PALETTE["l3_header"]
    )
    
    draw_card(
        ax, 0.65, 8.05, 17.65, 1.08,
        title="Project Risk Manager (Coordinator)",
        subtitle="Deterministic Procedural Python Dispatcher • High-Resolution Latency Profiling (time.perf_counter)",
        badge="[Procedural Python Component]", badge_color=PALETTE["badge_procedural"],
        bullets=[
            "Multi-Modal Dispatch: Ingests validated assessment request; synchronously triggers status tracking, market analysis, and retrieval",
            "Query Formulation: Synthesizes contextual search query ('schedule delay budget overrun...') from detected project telemetry anomalies",
            "Signal Routing: Aggregates intermediate status flags, sentiment polarity, and historical matches; passes bundle to Risk Scoring Agent",
            "Execution Integrity: Pure procedural execution eliminating LLM hallucination and multi-agent network overhead (< 0.05 ms latency)"
        ],
        border_color=PALETTE["l3_border"], header_color=PALETTE["l3_header"],
        font_size_title=11.0
    )
    
    # -------------------------------------------------------------------------
    # LAYER 4: Analytical Components (y: 5.00 - 7.45, h = 2.45)
    # -------------------------------------------------------------------------
    draw_container_layer(
        ax, 0.4, 5.00, 18.2, 2.45,
        4, "Analytical Components", "Domain-Specific Telemetry Tracking, Lexicon NLP, & TF-IDF Retrieval",
        bg_color=PALETTE["l4_bg"], border_color=PALETTE["l4_border_teal"], header_color=PALETTE["l4_header_teal"]
    )
    
    # 4.1 Project Status Tracking Agent (Column 1)
    draw_card(
        ax, 0.65, 5.18, 4.15, 2.05,
        title="Project Status Tracking Agent",
        subtitle="Internal Variance Monitor",
        badge="[Procedural Agent]", badge_color=PALETTE["badge_procedural"],
        bullets=[
            "Analyzes raw project telemetry",
            "Checks deterministic thresholds:",
            "  - Schedule delay > 15 days",
            "  - Budget overrun > 15%",
            "  - Critical issues > 3",
            "  - Team attrition > 12%",
            "Computes milestone progress index",
            "Emits structured status deviation flags"
        ],
        border_color=PALETTE["l4_border_teal"], header_color=PALETTE["l4_header_teal"]
    )
    
    # 4.2 Market Analysis Agent (Column 2 - Left)
    draw_card(
        ax, 5.05, 5.18, 4.15, 2.05,
        title="Market Analysis Agent",
        subtitle="Qualitative Context Monitor",
        badge="[Procedural Agent]", badge_color=PALETTE["badge_procedural"],
        bullets=[
            "Evaluates team communication health:",
            "  - Classifies as 'Good', 'Fair', or 'Poor'",
            "Monitors stakeholder satisfaction rating",
            "Delegates communication notes to NLP",
            "Computes external environment risk score",
            "Synthesizes qualitative context payload"
        ],
        border_color=PALETTE["l4_border_teal"], header_color=PALETTE["l4_header_teal"]
    )
    
    # 4.3 NLP Sentiment Analysis Component (Column 2 - Right)
    draw_card(
        ax, 9.45, 5.18, 4.15, 2.05,
        title="NLP Sentiment Component",
        subtitle="Domain Lexicon Engine",
        badge="[Domain Lexicon]", badge_color=PALETTE["badge_lexicon"],
        bullets=[
            "Software engineering risk lexicon",
            "Identifies positive & negative polarity:",
            "  - Positive: 'stable', 'ahead', 'resolved'",
            "  - Negative: 'blocker', 'attrition', 'delay'",
            "Calculates continuous polarity [-1.0, +1.0]",
            "Counts positive/negative risk tokens",
            "Deterministic rule engine (No HF)"
        ],
        border_color=PALETTE["l4_border_blue"], header_color=PALETTE["l4_header_blue"]
    )
    
    # 4.4 Historical Knowledge Retrieval Component (Column 3)
    draw_card(
        ax, 13.85, 5.18, 4.45, 2.05,
        title="Historical Retrieval Component",
        subtitle="Vector Similarity Store",
        badge="[TF-IDF Cosine]", badge_color=PALETTE["badge_retrieval"],
        bullets=[
            "Receives formulated risk query",
            "Transforms query into TF-IDF vector",
            "Computes cosine similarity across 15 cases",
            "Ranks and extracts Top-2 relevant projects",
            "Retrieves observed failure root causes",
            "Extracts empirical mitigation strategies",
            "Zero ChromaDB dependency (Vector math)"
        ],
        border_color=PALETTE["l4_border_blue"], header_color=PALETTE["l4_header_blue"]
    )
    
    # -------------------------------------------------------------------------
    # LAYER 5: Risk Assessment Engine (y: 3.10 - 4.55, h = 1.45)
    # -------------------------------------------------------------------------
    draw_container_layer(
        ax, 0.4, 3.10, 18.2, 1.45,
        5, "Risk Assessment Engine", "Supervised Machine Learning & Multi-Factor Risk Scoring Fusion",
        bg_color=PALETTE["l5_bg"], border_color=PALETTE["l5_border"], header_color=PALETTE["l5_header"]
    )
    
    draw_card(
        ax, 0.65, 3.25, 17.65, 1.12,
        title="Risk Scoring Agent (Machine Learning Classification & Multi-Factor Fusion)",
        subtitle="Dual Supervised Classifiers + Multi-Signal Heuristic Risk Formula",
        badge="[Scikit-Learn ML + Procedural Fusion]", badge_color=PALETTE["badge_ml"],
        bullets=[
            "Supervised ML Classifiers: Logistic Regression (L2 penalty, liblinear) & Random Forest (100 trees, max_depth=10)",
            "Probability Estimation: Generates 3-class posterior distribution across {At Risk (0), In Progress (1), On Track (2)}",
            "Multi-Signal Risk Fusion: Integrates ML class probabilities with status deviation flags, sentiment polarity, and historical risk benchmarks",
            "Consolidated Scoring: Computes composite Risk Score (0.0 to 100.0) and assigns discrete final risk category"
        ],
        border_color=PALETTE["l5_border"], header_color=PALETTE["l5_header"],
        font_size_title=11.0
    )
    
    # -------------------------------------------------------------------------
    # LAYER 6: Reporting Layer (y: 1.70 - 2.80, h = 1.10)
    # -------------------------------------------------------------------------
    draw_container_layer(
        ax, 0.4, 1.70, 18.2, 1.10,
        6, "Reporting Layer", "Automated Narrative Generation & Mitigation Strategy Mapping",
        bg_color=PALETTE["l6_bg"], border_color=PALETTE["l6_border"], header_color=PALETTE["l6_header"]
    )
    
    draw_card(
        ax, 0.65, 1.82, 17.65, 0.82,
        title="Reporting Agent",
        subtitle="Contextual Synthesis • Primary Risk Driver Ranking • Prioritized Mitigation Recommendation",
        badge="[Procedural Python Component]", badge_color=PALETTE["badge_procedural"],
        bullets=[
            "Synthesizes quantitative risk scores, categorical classifications, and telemetry variance metrics into an executive summary",
            "Ranks primary risk drivers across schedule, budget, staffing, and defect dimensions based on detected threshold breaches",
            "Maps empirical mitigation recommendations from retrieved historical cases to the current project's operational profile"
        ],
        border_color=PALETTE["l6_border"], header_color=PALETTE["l6_header"],
        font_size_title=10.5
    )
    
    # -------------------------------------------------------------------------
    # LAYER 7: Final Outputs & Deliverables (y: 0.65 - 1.45, h = 0.80)
    # -------------------------------------------------------------------------
    draw_container_layer(
        ax, 0.4, 0.65, 18.2, 0.82,
        7, "Final Outputs & Deliverables", "Decision-Support Artifacts for IT Project Governance",
        bg_color=PALETTE["l7_bg"], border_color=PALETTE["l7_border"], header_color=PALETTE["l7_header"]
    )
    
    outputs = [
        ("Consolidated Risk Score", "Continuous metric: 0.0 - 100.0\n(Calibrated composite index)", 0.65, 4.15),
        ("Categorical Risk Label", "On Track (<35) | In Progress (35-64)\nAt Risk (>=65)", 5.05, 4.15),
        ("Structured Risk Report", "Comprehensive JSON / Markdown\nTelemetry audit & executive summary", 9.45, 4.15),
        ("Prioritized Mitigations", "Actionable interventions\nContextualized from past cases", 13.85, 4.45),
    ]
    
    for title, desc, ox, ow in outputs:
        box = FancyBboxPatch(
            (ox, 0.73), ow, 0.60,
            boxstyle="round,pad=0,rounding_size=0.10",
            facecolor="#FFFFFF",
            edgecolor=PALETTE["l7_border"],
            linewidth=1.1,
            zorder=2
        )
        ax.add_patch(box)
        ax.text(
            ox + ow/2, 1.15, title,
            fontsize=8.8, fontweight="bold", color=PALETTE["text_dark"],
            ha="center", va="center", zorder=3, fontfamily="Arial"
        )
        ax.text(
            ox + ow/2, 0.88, desc,
            fontsize=7.2, color=PALETTE["text_body"],
            ha="center", va="center", zorder=3, fontfamily="Arial"
        )
        
    # -------------------------------------------------------------------------
    # ARROWS & DATA FLOWS (Figure 1: Clean, Parallel, Non-Intersecting)
    # -------------------------------------------------------------------------
    # 1. Layer 1 -> Layer 2 (Three Parallel Modality Channels)
    draw_arrow(ax, 3.48, 11.80, 3.48, 11.02, label="Raw Telemetry", color=PALETTE["arrow_dark"])
    draw_arrow(ax, 9.48, 11.80, 9.48, 11.02, label="Communication Notes", color=PALETTE["arrow_dark"])
    draw_arrow(ax, 15.48, 11.80, 15.48, 11.02, label="15 Case Studies", color=PALETTE["arrow_dark"])
    
    # 2. Layer 2 -> Layer 3 (Three Prepared Channels into Coordinator)
    draw_arrow(ax, 3.48, 9.90, 3.48, 9.13, label="Preprocessed Vectors", color=PALETTE["arrow_dark"])
    draw_arrow(ax, 9.48, 9.90, 9.48, 9.13, label="Normalized Text Stream", color=PALETTE["arrow_dark"])
    draw_arrow(ax, 15.48, 9.90, 15.48, 9.13, label="Indexed Vector Corpus", color=PALETTE["arrow_dark"])
    
    # 3. Layer 3 (Coordinator) -> Layer 4 (Analytical Components Dispatch)
    draw_arrow(ax, 2.72, 8.05, 2.72, 7.23, label="Project Features", color=PALETTE["arrow_purple"])
    draw_arrow(ax, 7.12, 8.05, 7.12, 7.23, label="Text & Sat Metrics", color=PALETTE["arrow_purple"])
    draw_arrow(ax, 16.08, 8.05, 16.08, 7.23, label="Formulated Risk Query", color=PALETTE["arrow_purple"])
    
    # 4. Market Analyst <-> NLP Sentiment Component (Horizontal Interaction in Layer 4)
    draw_arrow(ax, 9.20, 6.45, 9.45, 6.45, label="Raw Text", color=PALETTE["arrow_teal"], label_side="top", label_offset=0.08)
    draw_arrow(ax, 9.45, 5.85, 9.20, 5.85, label="Polarity & Terms", color=PALETTE["arrow_teal"], label_side="bottom", label_offset=0.08)
    
    # 5. Layer 4 (Analytical Components) -> Layer 5 (Risk Assessment Engine)
    draw_arrow(ax, 2.72, 5.18, 2.72, 4.37, label="Deviation Alert Flags", color=PALETTE["arrow_dark"])
    draw_arrow(ax, 7.12, 5.18, 7.12, 4.37, label="Qualitative Sentiment Score", color=PALETTE["arrow_dark"])
    draw_arrow(ax, 16.08, 5.18, 16.08, 4.37, label="Top-2 Case Matches & Mitigations", color=PALETTE["arrow_dark"])
    
    # 6. Layer 5 (Risk Scoring Agent) -> Layer 6 (Reporting Agent)
    draw_arrow(ax, 9.5, 3.25, 9.5, 2.64, label="Composite Risk Score (0-100), Probability Distribution & Drivers", color=PALETTE["arrow_dark"])
    
    # 7. Layer 6 (Reporting Agent) -> Layer 7 (Final Deliverables)
    draw_arrow(ax, 2.72, 1.82, 2.72, 1.33, color=PALETTE["arrow_dark"])
    draw_arrow(ax, 7.12, 1.82, 7.12, 1.33, color=PALETTE["arrow_dark"])
    draw_arrow(ax, 11.52, 1.82, 11.52, 1.33, color=PALETTE["arrow_dark"])
    draw_arrow(ax, 16.08, 1.82, 16.08, 1.33, color=PALETTE["arrow_dark"])
    
    # -------------------------------------------------------------------------
    # BOTTOM LEGEND
    # -------------------------------------------------------------------------
    legend_y = 0.20
    legend_items = [
        ("[Procedural Python Component]", PALETTE["badge_procedural"], "Deterministic Python logic (No LLM / No CrewAI)"),
        ("[Domain Lexicon Engine]", PALETTE["badge_lexicon"], "Keyword polarity scoring (No HuggingFace)"),
        ("[TF-IDF Cosine Similarity]", PALETTE["badge_retrieval"], "Vector retrieval engine (No ChromaDB)"),
        ("[Scikit-Learn ML Model]", PALETTE["badge_ml"], "Logistic Regression & Random Forest"),
        ("[Data Pipeline / Imputer]", PALETTE["badge_pipeline"], "Preprocessing & leak-free scaling"),
        ("[Output Deliverable]", PALETTE["badge_output"], "Decision support artifacts")
    ]
    
    lx = 0.5
    for tag, color, desc in legend_items:
        bw = len(tag) * 0.085 + 0.2
        box = FancyBboxPatch(
            (lx, legend_y), bw, 0.22,
            boxstyle="round,pad=0,rounding_size=0.04",
            facecolor=color, edgecolor=color, zorder=3
        )
        ax.add_patch(box)
        ax.text(
            lx + bw/2, legend_y + 0.11, tag,
            fontsize=6.5, fontweight="bold", color="#FFFFFF",
            ha="center", va="center", zorder=4, fontfamily="Arial"
        )
        ax.text(
            lx + bw + 0.1, legend_y + 0.11, desc,
            fontsize=6.8, color="#475569",
            ha="left", va="center", zorder=3, fontfamily="Arial"
        )
        lx += bw + len(desc) * 0.045 + 0.35

    # Save figure
    png_path = os.path.join(OUTPUT_DIR, "figure1_overall_architecture.png")
    svg_path = os.path.join(OUTPUT_DIR, "figure1_overall_architecture.svg")
    root_png_path = "fig 1.png"
    
    plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig(svg_path, format="svg", bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig(root_png_path, format="png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"Saved: {png_path} (>= 300 DPI)")
    print(f"Saved: {svg_path} (vector)")
    print(f"Saved: {root_png_path} (root >= 300 DPI)")


# =============================================================================
# FIGURE 2: MULTI-AGENT WORKFLOW
# =============================================================================
def generate_figure2():
    print("Generating Figure 2: Multi-Agent Workflow...")
    fig, ax = plt.subplots(figsize=(22, 10.5), dpi=300)
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["bg"])
    ax.set_xlim(0, 22)
    ax.set_ylim(0, 10.5)
    ax.axis('off')
    
    # -------------------------------------------------------------------------
    # Header Title & Subtitle
    # -------------------------------------------------------------------------
    ax.text(
        11.0, 10.15,
        "Figure 2: Execution and Dataflow Architecture of the Multi-Agent IT Project Risk Management Framework",
        fontsize=13.5, fontweight="bold", color=PALETTE["text_dark"],
        ha="center", va="center", fontfamily="Arial"
    )
    ax.text(
        11.0, 9.85,
        "Step-by-step sequential and concurrent procedural workflow illustrating data ingestion, agent coordination, analytical enrichment, ML scoring, and report synthesis",
        fontsize=9.0, fontstyle="italic", color=PALETTE["text_muted"],
        ha="center", va="center", fontfamily="Arial"
    )
    
    # -------------------------------------------------------------------------
    # STAGES 1 TO 7 (Horizontal Columns)
    # -------------------------------------------------------------------------
    stages = [
        (0.5,  2.6, "STAGE 1", "Assessment Trigger", "Input Ingestion & Payload Packing",
         PALETTE["l1_bg"], PALETTE["l1_border"], PALETTE["l1_header"]),
        (3.4,  2.7, "STAGE 2", "Coordination", "Workflow Timing & Task Routing",
         PALETTE["l3_bg"], PALETTE["l3_border"], PALETTE["l3_header"]),
        (6.4,  3.5, "STAGE 3", "Telemetry & Context", "Concurrent Procedural Agents",
         PALETTE["l4_bg"], PALETTE["l4_border_teal"], PALETTE["l4_header_teal"]),
        (10.2, 3.5, "STAGE 4", "Enrichment", "Lexicon NLP & TF-IDF Retrieval",
         PALETTE["l4_bg"], PALETTE["l4_border_blue"], PALETTE["l4_header_blue"]),
        (14.0, 2.7, "STAGE 5", "Risk Assessment", "ML Inference & Signal Fusion",
         PALETTE["l5_bg"], PALETTE["l5_border"], PALETTE["l5_header"]),
        (17.0, 2.5, "STAGE 6", "Reporting", "Driver Ranking & Mitigation",
         PALETTE["l6_bg"], PALETTE["l6_border"], PALETTE["l6_header"]),
        (19.8, 1.8, "STAGE 7", "Outputs", "Governance Deliverables",
         PALETTE["l7_bg"], PALETTE["l7_border"], PALETTE["l7_header"]),
    ]
    
    container_y = 1.05
    container_h = 8.45
    
    for sx, sw, snum, stitle, sdesc, sbg, sborder, sheader in stages:
        box = FancyBboxPatch(
            (sx, container_y), sw, container_h,
            boxstyle="round,pad=0,rounding_size=0.18",
            facecolor=sbg, edgecolor=sborder, linewidth=1.1, linestyle="--", zorder=1
        )
        ax.add_patch(box)
        
        tab_h = 0.36
        tab_y = container_y + container_h - tab_h / 2
        tab = FancyBboxPatch(
            (sx + 0.15, tab_y), sw - 0.30, tab_h,
            boxstyle="round,pad=0,rounding_size=0.08",
            facecolor=sheader, edgecolor=sheader, linewidth=0.5, zorder=3
        )
        ax.add_patch(tab)
        
        ax.text(
            sx + sw/2, tab_y + tab_h/2, f"{snum}: {stitle.upper()}",
            fontsize=8.0, fontweight="bold", color="#FFFFFF",
            ha="center", va="center", zorder=4, fontfamily="Arial"
        )
        
        ax.text(
            sx + sw/2, container_y + container_h - 0.35, sdesc,
            fontsize=6.8, fontstyle="italic", color="#64748B",
            ha="center", va="center", zorder=3, fontfamily="Arial"
        )
        
    # -------------------------------------------------------------------------
    # STAGE 1 CONTENT: Project Assessment Request
    # -------------------------------------------------------------------------
    draw_card(
        ax, 0.65, 4.8, 2.3, 4.0,
        title="Project Request Payload",
        subtitle="assess_project(data, text)",
        badge="[Input]", badge_color=PALETTE["badge_input"],
        bullets=[
            "Project ID ('PRJ-001')",
            "Telemetry Dictionary (19 Features):",
            "  - completion_pct, planned_completion",
            "  - budget_variance_pct, budget_util",
            "  - schedule_delay_days, duration_days",
            "  - unresolved_issue_count, bug_count",
            "  - critical_issue_count, test_pass_pct",
            "  - team_attrition_rate, satisfaction",
            "  - project_complexity, comm_health",
            "Communication Text Notes",
            "Variant flag ('complete')"
        ],
        border_color=PALETTE["l1_border"], header_color=PALETTE["l1_header"],
        corner_radius=0.12, font_size_title=9.5
    )
    
    # -------------------------------------------------------------------------
    # STAGE 2 CONTENT: Project Risk Manager (Coordinator)
    # -------------------------------------------------------------------------
    draw_card(
        ax, 3.55, 3.2, 2.4, 5.6,
        title="Project Risk Manager",
        subtitle="Coordinator Agent",
        badge="[Procedural]", badge_color=PALETTE["badge_procedural"],
        bullets=[
            "Execution Controller:",
            "  - Initializes timer:",
            "    time.perf_counter()",
            "  - Validates feature dict",
            "  - Ingests variant config",
            "Task Dispatching:",
            "  1. Calls Status Tracker",
            "  2. Calls Market Analyst",
            "  3. Formulates query",
            "  4. Calls Risk Scorer",
            "  5. Calls Reporter",
            "Execution Auditing:",
            "  - Collects agent latencies",
            "  - Measures total runtime",
            "  - Negligible coordinator",
            "    overhead (< 0.05 ms)"
        ],
        border_color=PALETTE["l3_border"], header_color=PALETTE["l3_header"],
        corner_radius=0.12, font_size_title=9.5
    )
    
    # -------------------------------------------------------------------------
    # STAGE 3 CONTENT: Telemetry & Context Agents
    # -------------------------------------------------------------------------
    # Top Card: Project Status Tracking Agent
    draw_card(
        ax, 6.55, 5.7, 3.2, 3.1,
        title="Project Status Tracking Agent",
        subtitle="Internal Variance Monitor",
        badge="[Procedural Agent]", badge_color=PALETTE["badge_procedural"],
        bullets=[
            "Ingests raw project telemetry metrics",
            "Executes deterministic threshold checks:",
            "  - schedule_delay > 15 days -> Delay Flag",
            "  - budget_variance > 15% -> Overrun Flag",
            "  - critical_issues > 3 -> Quality Flag",
            "  - attrition_rate > 12% -> Staffing Flag",
            "Computes milestone completion index",
            "Emits structured status deviation payload"
        ],
        border_color=PALETTE["l4_border_teal"], header_color=PALETTE["l4_header_teal"],
        corner_radius=0.12, font_size_title=9.5
    )
    
    # Bottom Card: Market Analysis Agent
    draw_card(
        ax, 6.55, 1.5, 3.2, 3.6,
        title="Market Analysis Agent",
        subtitle="Qualitative Context Monitor",
        badge="[Procedural Agent]", badge_color=PALETTE["badge_procedural"],
        bullets=[
            "Ingests qualitative communication text",
            "Ingests stakeholder satisfaction scores",
            "Evaluates team communication health:",
            "  - 'Good', 'Fair', or 'Poor'",
            "Delegates text to NLP Sentiment Engine",
            "Calculates external environmental risk score",
            "Integrates stakeholder perception flags",
            "Synthesizes qualitative context payload"
        ],
        border_color=PALETTE["l4_border_teal"], header_color=PALETTE["l4_header_teal"],
        corner_radius=0.12, font_size_title=9.5
    )
    
    # -------------------------------------------------------------------------
    # STAGE 4 CONTENT: NLP Sentiment & Historical Knowledge Retrieval
    # -------------------------------------------------------------------------
    # Top Card: Historical Knowledge Retrieval
    draw_card(
        ax, 10.35, 5.7, 3.2, 3.1,
        title="Historical Retrieval Component",
        subtitle="TF-IDF Cosine Similarity",
        badge="[TF-IDF Engine]", badge_color=PALETTE["badge_retrieval"],
        bullets=[
            "Ingests semantic risk query from deviations:",
            "  e.g. 'schedule delay budget overrun'",
            "Transforms query into TF-IDF vector space",
            "Computes cosine similarity across 15 cases",
            "Ranks and retrieves Top-2 matched projects",
            "Extracts past root cause analyses",
            "Extracts validated historical mitigations",
            "Zero ChromaDB dependency (Vector math)"
        ],
        border_color=PALETTE["l4_border_blue"], header_color=PALETTE["l4_header_blue"],
        corner_radius=0.12, font_size_title=9.5
    )
    
    # Bottom Card: NLP Sentiment Analysis
    draw_card(
        ax, 10.35, 1.5, 3.2, 3.6,
        title="NLP Sentiment Analysis Component",
        subtitle="Domain Lexicon Engine",
        badge="[Domain Lexicon]", badge_color=PALETTE["badge_lexicon"],
        bullets=[
            "Receives unstructured text from Market Analyst",
            "Tokenizes and normalizes message strings",
            "Matches tokens against domain risk lexicon:",
            "  - Positive: 'stable', 'ahead', 'on-track'",
            "  - Negative: 'blocker', 'attrition', 'delay'",
            "Computes continuous polarity score [-1.0, +1.0]",
            "Counts positive and negative keyword matches",
            "Deterministic keyword rule engine (No HF)"
        ],
        border_color=PALETTE["l4_border_blue"], header_color=PALETTE["l4_header_blue"],
        corner_radius=0.12, font_size_title=9.5
    )
    
    # -------------------------------------------------------------------------
    # STAGE 5 CONTENT: Risk Scoring Agent
    # -------------------------------------------------------------------------
    draw_card(
        ax, 14.15, 2.0, 2.4, 6.8,
        title="Risk Scoring Agent",
        subtitle="ML Inference & Fusion",
        badge="[Procedural + ML]", badge_color=PALETTE["badge_ml"],
        bullets=[
            "Ingests 4 distinct inputs:",
            "  1. Preprocessed feature vector",
            "  2. Status deviation flags",
            "  3. NLP sentiment score",
            "  4. Historical retrieval matches",
            "ML Supervised Inference:",
            "  - Logistic Regression /",
            "    Random Forest (100 trees)",
            "  - Emits class probabilities:",
            "    P(At Risk), P(In Progress),",
            "    P(On Track)",
            "Multi-Signal Risk Formula:",
            "  - Base score from ML prob",
            "  - Status deviation penalty",
            "  - Sentiment polarity offset",
            "  - Historical risk alignment",
            "Outputs:",
            "  - Composite Score: [0 - 100]",
            "  - Discrete Category Label"
        ],
        border_color=PALETTE["l5_border"], header_color=PALETTE["l5_header"],
        corner_radius=0.12, font_size_title=9.5
    )
    
    # -------------------------------------------------------------------------
    # STAGE 6 CONTENT: Reporting Agent
    # -------------------------------------------------------------------------
    draw_card(
        ax, 17.15, 2.4, 2.2, 6.0,
        title="Reporting Agent",
        subtitle="Report Synthesizer",
        badge="[Procedural]", badge_color=PALETTE["badge_procedural"],
        bullets=[
            "Aggregates full assessment:",
            "  - Risk score & category",
            "  - Probability breakdown",
            "  - Status deviation flags",
            "  - Sentiment indicators",
            "  - Top-2 historical cases",
            "Synthesizes Deliverables:",
            "  - Identifies top risk drivers",
            "  - Ranks impact severity",
            "  - Tailors mitigations from",
            "    retrieved case studies",
            "  - Generates comprehensive",
            "    narrative report"
        ],
        border_color=PALETTE["l6_border"], header_color=PALETTE["l6_header"],
        corner_radius=0.12, font_size_title=9.5
    )
    
    # -------------------------------------------------------------------------
    # STAGE 7 CONTENT: Outputs & Deliverables
    # -------------------------------------------------------------------------
    output_cards = [
        ("Consolidated Risk Score", "Continuous [0.0 - 100.0]\nCalibrated Risk Metric", 7.0, 1.5),
        ("Risk Classification", "Discrete Label:\n{On Track, In Progress,\nAt Risk}", 5.2, 1.55),
        ("Primary Risk Drivers", "Ranked severity audit\nSchedule, Budget, Staffing", 3.5, 1.45),
        ("Prioritized Mitigations", "Actionable interventions\nFrom historical cases", 1.8, 1.45),
    ]
    
    for otitle, odesc, oy, oh in output_cards:
        box = FancyBboxPatch(
            (19.95, oy), 1.5, oh,
            boxstyle="round,pad=0,rounding_size=0.10",
            facecolor="#FFFFFF", edgecolor=PALETTE["l7_border"], linewidth=1.1, zorder=2
        )
        ax.add_patch(box)
        ax.text(
            19.95 + 0.75, oy + oh - 0.24, otitle,
            fontsize=8.0, fontweight="bold", color=PALETTE["text_dark"],
            ha="center", va="center", zorder=3, fontfamily="Arial"
        )
        ax.text(
            19.95 + 0.75, oy + (oh - 0.24)/2, odesc,
            fontsize=6.8, color=PALETTE["text_body"],
            ha="center", va="center", zorder=3, fontfamily="Arial"
        )
        
    # -------------------------------------------------------------------------
    # ARROWS & DATA FLOWS (Figure 2: Strictly Adjacent & Non-Crossing)
    # -------------------------------------------------------------------------
    # 1. Request -> Coordinator (Stage 1 -> Stage 2)
    draw_arrow(
        ax, 2.95, 6.2, 3.55, 6.2,
        label="Project Data & Text", color=PALETTE["arrow_dark"]
    )
    
    # 2. Coordinator -> Status Tracker (Stage 2 -> Stage 3 Top)
    draw_arrow(
        ax, 5.95, 7.2, 6.55, 7.2,
        label="Telemetry Dict", color=PALETTE["arrow_purple"]
    )
    
    # 3. Coordinator -> Market Analyst (Stage 2 -> Stage 3 Bottom)
    draw_arrow(
        ax, 5.95, 3.3, 6.55, 3.3,
        label="Comms Text & Ratings", color=PALETTE["arrow_purple"]
    )
    
    # 4. Status Tracker -> Historical Retrieval (Stage 3 Top -> Stage 4 Top)
    # Status deviations trigger query formulation for retrieval
    draw_arrow(
        ax, 9.75, 7.2, 10.35, 7.2,
        label="Detected Deviations -> Formulated Query",
        color=PALETTE["arrow_teal"]
    )
    
    # 5. Market Analyst <-> NLP Sentiment Engine (Stage 3 Bottom -> Stage 4 Bottom)
    draw_arrow(
        ax, 9.75, 3.8, 10.35, 3.8,
        label="Communication Text", color=PALETTE["arrow_teal"], label_side="top", label_offset=0.10
    )
    draw_arrow(
        ax, 10.35, 2.6, 9.75, 2.6,
        label="Polarity Score & Tokens", color=PALETTE["arrow_teal"], label_side="bottom", label_offset=0.10
    )
    
    # 6. Historical Retrieval -> Risk Scoring Agent (Stage 4 Top -> Stage 5)
    draw_arrow(
        ax, 13.55, 7.2, 14.15, 7.2,
        label="Top-2 Matched Historical Cases", color=PALETTE["arrow_dark"]
    )
    
    # 7. NLP Sentiment / Qualitative -> Risk Scoring Agent (Stage 4 Bottom -> Stage 5)
    draw_arrow(
        ax, 13.55, 3.3, 14.15, 3.3,
        label="Sentiment Polarity & External Risk", color=PALETTE["arrow_dark"]
    )
    
    # 8. Risk Scoring Agent -> Reporting Agent (Stage 5 -> Stage 6)
    draw_arrow(
        ax, 16.55, 5.4, 17.15, 5.4,
        label="Risk Score, Probs & Drivers", color=PALETTE["arrow_dark"]
    )
    
    # 9. Reporting Agent -> Stage 7 Outputs (Stage 6 -> Stage 7)
    draw_arrow(ax, 19.35, 7.6, 19.95, 7.6, color=PALETTE["arrow_dark"])
    draw_arrow(ax, 19.35, 5.9, 19.95, 5.9, color=PALETTE["arrow_dark"])
    draw_arrow(ax, 19.35, 4.2, 19.95, 4.2, color=PALETTE["arrow_dark"])
    draw_arrow(ax, 19.35, 2.5, 19.95, 2.5, color=PALETTE["arrow_dark"])
    
    # -------------------------------------------------------------------------
    # BOTTOM BENCHMARK TIMING BAR
    # -------------------------------------------------------------------------
    timing_bar_y = 0.28
    timing_bar_w = 20.8
    timing_box = FancyBboxPatch(
        (0.6, timing_bar_y), timing_bar_w, 0.48,
        boxstyle="round,pad=0,rounding_size=0.10",
        facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.0, zorder=2
    )
    ax.add_patch(timing_box)
    
    timing_text = (
        "EMPIRICAL WORKFLOW RUNTIME BENCHMARK (Mean Latency):  "
        "Status Tracker: 0.016 ms  |  "
        "Market Analyst & NLP Lexicon: 0.051 ms  |  "
        "Historical TF-IDF Retrieval: 1.314 ms  |  "
        "Risk Scorer & ML Inference: 16.231 ms  |  "
        "Reporting Agent: 0.027 ms  |  "
        "Coordinator Overhead: 0.007 ms  |  "
        "TOTAL END-TO-END WORKFLOW: 17.646 ms (~56.6 req/s)"
    )
    ax.text(
        0.6 + timing_bar_w/2, timing_bar_y + 0.24, timing_text,
        fontsize=7.8, fontweight="bold", color="#1E293B",
        ha="center", va="center", zorder=3, fontfamily="Arial"
    )

    # Save figure
    png_path = os.path.join(OUTPUT_DIR, "figure2_multi_agent_workflow.png")
    svg_path = os.path.join(OUTPUT_DIR, "figure2_multi_agent_workflow.svg")
    root_png_path = "fig 2.png"
    
    plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig(svg_path, format="svg", bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig(root_png_path, format="png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"Saved: {png_path} (>= 300 DPI)")
    print(f"Saved: {svg_path} (vector)")
    print(f"Saved: {root_png_path} (root >= 300 DPI)")


if __name__ == "__main__":
    generate_figure1()
    generate_figure2()
    print("Both diagrams generated successfully!")
