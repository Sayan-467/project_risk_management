"""
generate_simple_architecture_diagrams.py
========================================
Generates publication-ready, clean, simplified architecture diagrams strictly matching
the user's original reference visual design, updated with post-audit project details:
- 19 IT project features (schedule, budget, quality, resources, complexity, comm health)
- Sprint retrospectives, team communications, and 15 historical IT case studies
- Dual supervised ML models: Logistic Regression and Random Forest
- Domain-specific sentiment lexicon (Software Engineering vocabulary)
- TF-IDF Vector Cosine Semantic Retrieval (replacing ChromaDB)
- 3 validated target classes: On Track, In Progress, At Risk
- Outputs: Continuous Risk Score (0-100), Risk Classification, Risk Drivers, Mitigations

Outputs:
- fig 1.png (Root repository)
- fig 2.png (Root repository)
- results/architecture/figure1_overall_architecture.png & .svg
- results/architecture/figure2_multi_agent_workflow.png & .svg
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Polygon
import numpy as np

OUTPUT_DIR = os.path.join("results", "architecture")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# Soft Academic Palette (faithfully matching reference images)
# -----------------------------------------------------------------------------
COLORS = {
    "bg": "#FFFFFF",
    "text_dark": "#1E293B",
    "text_navy": "#0F2942",
    "text_body": "#334155",
    "text_muted": "#5A6E85",
    
    # Outer Containers
    "container_fill": "#EBF2F8",
    "container_border": "#A4BAD2",
    
    # Header Banners
    "header_fill": "#BCCEE4",
    "header_border": "#96ACC6",
    
    # Agent Nodes & Accent Cards
    "agent_fill": "#D9E6F4",
    "agent_border": "#8CA8C8",
    
    # Inner White Cards
    "card_white": "#FFFFFF",
    "card_border": "#CBD8E6",
    
    # Application / Output Accent
    "accent_fill": "#C5D6EA",
    "accent_border": "#85A3C4",
    
    # Arrows & Connectors
    "arrow_color": "#476A94",
    "arrow_text": "#1E3A5F",
    
    # Soft Badge / Icon Accents
    "icon_blue": "#2B6CB0",
    "icon_green": "#2F855A",
    "icon_amber": "#C05621",
    "icon_purple": "#6B46C1"
}


# -----------------------------------------------------------------------------
# Drawing Helpers
# -----------------------------------------------------------------------------
def draw_rounded_box(ax, x, y, w, h, facecolor, edgecolor, radius=0.15, lw=1.2, zorder=2, linestyle='-'):
    """Draws a clean rounded rectangle with FancyBboxPatch."""
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=lw,
        linestyle=linestyle,
        zorder=zorder
    )
    ax.add_patch(box)
    return box


def draw_arrow(ax, x1, y1, x2, y2, color=COLORS["arrow_color"], lw=1.5, zorder=4, mutation_scale=12, label=None, label_pos=(0, 0), label_size=8.5):
    """Draws a crisp directed connection arrow."""
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            lw=lw,
            mutation_scale=mutation_scale,
            shrinkA=0, shrinkB=0
        ),
        zorder=zorder
    )
    if label:
        lx = (x1 + x2) / 2 + label_pos[0]
        ly = (y1 + y2) / 2 + label_pos[1]
        ax.text(
            lx, ly, label,
            fontsize=label_size, fontweight="bold", color=COLORS["arrow_text"],
            ha="center", va="center", zorder=zorder+1, fontfamily="Arial"
        )


def draw_double_arrow(ax, x1, y1, x2, y2, color=COLORS["arrow_color"], lw=1.5, zorder=4, mutation_scale=11):
    """Draws a bidirectional arrow <->."""
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="<|-|>",
            color=color,
            lw=lw,
            mutation_scale=mutation_scale,
            shrinkA=0, shrinkB=0
        ),
        zorder=zorder
    )


def draw_simple_icon(ax, x, y, icon_type, color=COLORS["icon_blue"]):
    """Draws minimal geometric vector glyphs for folder, globe, chart, etc."""
    if icon_type == "folder":
        # Folder glyph
        ax.add_patch(FancyBboxPatch((x-0.16, y-0.12), 0.32, 0.24, boxstyle="round,pad=0,rounding_size=0.03",
                                    facecolor=color, edgecolor=color, zorder=5))
        ax.add_patch(Rectangle((x-0.16, y+0.04), 0.14, 0.08, facecolor=color, edgecolor=color, zorder=5))
    elif icon_type == "globe":
        # Globe glyph
        circle = plt.Circle((x, y), 0.14, facecolor="none", edgecolor=color, lw=1.4, zorder=5)
        ax.add_patch(circle)
        ax.plot([x-0.14, x+0.14], [y, y], color=color, lw=1.2, zorder=5)
        ax.plot([x, x], [y-0.14, y+0.14], color=color, lw=1.2, zorder=5)
    elif icon_type == "chart":
        # Bar chart glyph
        ax.add_patch(Rectangle((x-0.14, y-0.12), 0.07, 0.12, facecolor=color, edgecolor=color, zorder=5))
        ax.add_patch(Rectangle((x-0.04, y-0.12), 0.07, 0.19, facecolor=color, edgecolor=color, zorder=5))
        ax.add_patch(Rectangle((x+0.06, y-0.12), 0.07, 0.25, facecolor=color, edgecolor=color, zorder=5))
    elif icon_type == "chat":
        # Chat bubble glyph
        ax.add_patch(FancyBboxPatch((x-0.14, y-0.08), 0.28, 0.20, boxstyle="round,pad=0,rounding_size=0.05",
                                    facecolor=color, edgecolor=color, zorder=5))
        pts = np.array([[x-0.06, y-0.08], [x-0.12, y-0.16], [x, y-0.08]])
        ax.add_patch(Polygon(pts, facecolor=color, edgecolor=color, zorder=5))
    elif icon_type == "gauge":
        # Gauge glyph
        ax.add_patch(FancyBboxPatch((x-0.14, y-0.12), 0.28, 0.24, boxstyle="round,pad=0,rounding_size=0.10",
                                    facecolor="none", edgecolor=color, lw=1.4, zorder=5))
        ax.plot([x, x+0.08], [y-0.04, y+0.06], color=color, lw=1.5, zorder=6)
    elif icon_type == "database":
        # Database cylinder glyph
        ax.add_patch(FancyBboxPatch((x-0.13, y-0.13), 0.26, 0.26, boxstyle="round,pad=0,rounding_size=0.04",
                                    facecolor=color, edgecolor=color, zorder=5))
        ax.plot([x-0.13, x+0.13], [y-0.04, y-0.04], color="#FFFFFF", lw=1.0, zorder=6)
        ax.plot([x-0.13, x+0.13], [y+0.05, y+0.05], color="#FFFFFF", lw=1.0, zorder=6)
    elif icon_type == "progress":
        # Progress bar glyph
        ax.add_patch(FancyBboxPatch((x-0.15, y-0.06), 0.30, 0.12, boxstyle="round,pad=0,rounding_size=0.05",
                                    facecolor="#CBD5E1", edgecolor=color, lw=1.0, zorder=5))
        ax.add_patch(FancyBboxPatch((x-0.15, y-0.06), 0.18, 0.12, boxstyle="round,pad=0,rounding_size=0.05",
                                    facecolor=color, edgecolor=color, lw=0.5, zorder=6))


# =============================================================================
# FIGURE 1: OVERALL SYSTEM ARCHITECTURE (Faithful to Reference Image 1)
# =============================================================================
def generate_figure1():
    print("Generating Figure 1: Overall System Architecture (Simplified Reference Design)...")
    fig, ax = plt.subplots(figsize=(8.5, 11.5), dpi=300)
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 8.5)
    ax.set_ylim(0, 11.5)
    ax.axis('off')

    # -------------------------------------------------------------------------
    # 1. TOP CONTAINER: DATA SOURCES (y: 9.35 to 11.15, h=1.80)
    # -------------------------------------------------------------------------
    c1_x, c1_y, c1_w, c1_h = 0.5, 9.35, 7.5, 1.80
    draw_rounded_box(ax, c1_x, c1_y, c1_w, c1_h, COLORS["container_fill"], COLORS["container_border"], radius=0.12, lw=1.2)
    
    # Header Banner
    draw_rounded_box(ax, c1_x, c1_y + c1_h - 0.40, c1_w, 0.40, COLORS["header_fill"], COLORS["header_border"], radius=0.08, lw=1.0)
    ax.text(c1_x + c1_w/2, c1_y + c1_h - 0.20, "DATA SOURCES",
            fontsize=11.5, fontweight="bold", color=COLORS["text_navy"], ha="center", va="center", fontfamily="Arial")

    # Card 1.1: Internal Project Data (Left Card)
    card1_w = 3.35
    card_h = 1.18
    draw_rounded_box(ax, 0.75, 9.50, card1_w, card_h, COLORS["card_white"], COLORS["card_border"], radius=0.08, lw=1.0)
    draw_simple_icon(ax, 1.15, 10.38, "folder", color="#4A6584")
    ax.text(1.50, 10.38, "Internal Project Data", fontsize=10.0, fontweight="bold", color=COLORS["text_navy"], va="center", fontfamily="Arial")
    
    bullets1 = [
        "• Schedule (Delay Days)",
        "• Budget (Variance %)",
        "• Tasks & Issues (Bugs, Criticals)",
        "• Resources (Team Attrition, Velocity)"
    ]
    for i, b in enumerate(bullets1):
        ax.text(1.10, 10.08 - i*0.18, b, fontsize=8.2, color=COLORS["text_body"], va="center", fontfamily="Arial")

    # Card 1.2: External Context & History (Right Card)
    draw_rounded_box(ax, 4.40, 9.50, card1_w, card_h, COLORS["card_white"], COLORS["card_border"], radius=0.08, lw=1.0)
    draw_simple_icon(ax, 4.80, 10.38, "globe", color="#2B6CB0")
    ax.text(5.15, 10.38, "External Context & History", fontsize=10.0, fontweight="bold", color=COLORS["text_navy"], va="center", fontfamily="Arial")
    
    bullets2 = [
        "• Team Communications & Standups",
        "• Sprint Retrospective Notes",
        "• Stakeholder Satisfaction Feedback",
        "• Historical IT Project Cases (15 Projects)"
    ]
    for i, b in enumerate(bullets2):
        ax.text(4.75, 10.08 - i*0.18, b, fontsize=8.2, color=COLORS["text_body"], va="center", fontfamily="Arial")

    # Arrow from Data Sources -> Preprocessing
    draw_arrow(ax, 4.25, 9.35, 4.25, 8.80)

    # -------------------------------------------------------------------------
    # 2. DATA PREPROCESSING (Pill / Text)
    # -------------------------------------------------------------------------
    prep_w, prep_h = 2.8, 0.36
    draw_rounded_box(ax, 4.25 - prep_w/2, 8.44, prep_w, prep_h, COLORS["container_fill"], COLORS["container_border"], radius=0.10, lw=1.0)
    ax.text(4.25, 8.62, "DATA PREPROCESSING", fontsize=9.5, fontweight="bold", color=COLORS["text_navy"], ha="center", va="center", fontfamily="Arial")

    # Arrow from Preprocessing -> Middle Layer
    draw_arrow(ax, 4.25, 8.44, 4.25, 7.92)

    # -------------------------------------------------------------------------
    # 3. MIDDLE SECTION: MULTI-AGENT LAYER (Left) & AI COMPONENTS (Right)
    # (y: 3.80 to 7.92, h=4.12)
    # -------------------------------------------------------------------------
    # Left Container: MULTI-AGENT ORCHESTRATION LAYER
    ma_x, ma_y, ma_w, ma_h = 0.5, 3.80, 5.0, 4.12
    draw_rounded_box(ax, ma_x, ma_y, ma_w, ma_h, COLORS["container_fill"], COLORS["container_border"], radius=0.12, lw=1.2)
    
    # Header Banner
    draw_rounded_box(ax, ma_x, ma_y + ma_h - 0.40, ma_w, 0.40, COLORS["header_fill"], COLORS["header_border"], radius=0.08, lw=1.0)
    ax.text(ma_x + ma_w/2, ma_y + ma_h - 0.20, "MULTI-AGENT ORCHESTRATION LAYER",
            fontsize=10.5, fontweight="bold", color=COLORS["text_navy"], ha="center", va="center", fontfamily="Arial")

    # Coordinator Box
    coord_w, coord_h = 4.2, 0.48
    draw_rounded_box(ax, ma_x + (ma_w - coord_w)/2, 6.85, coord_w, coord_h, COLORS["agent_fill"], COLORS["agent_border"], radius=0.08, lw=1.1)
    ax.text(ma_x + ma_w/2, 6.85 + coord_h/2, "Project Risk Manager (Coordinator)",
            fontsize=9.8, fontweight="bold", color=COLORS["text_navy"], ha="center", va="center", fontfamily="Arial")

    # Downward branching arrows from Coordinator to the 3 agents
    # Agent cards:
    ag_y = 5.25
    ag_w = 1.32
    ag_h = 1.15
    ag1_x = 0.72
    ag2_x = 2.34
    ag3_x = 3.96
    
    # Coordinator dispatch arrows
    draw_arrow(ax, 1.38, 6.85, 1.38, 6.40)
    draw_arrow(ax, 3.00, 6.85, 3.00, 6.40)
    draw_arrow(ax, 4.62, 6.85, 4.62, 6.40)

    # Agent 1: Status Agent
    draw_rounded_box(ax, ag1_x, ag_y, ag_w, ag_h, COLORS["card_white"], COLORS["card_border"], radius=0.08, lw=1.0)
    draw_simple_icon(ax, ag1_x + ag_w/2, ag_y + ag_h - 0.30, "progress", color="#3182CE")
    ax.text(ag1_x + ag_w/2, ag_y + 0.50, "Status\nAgent", fontsize=9.2, fontweight="bold", color=COLORS["text_navy"], ha="center", va="center", fontfamily="Arial")
    ax.text(ag1_x + ag_w/2, ag_y + 0.18, "(Telemetry Checks)", fontsize=7.2, color=COLORS["text_muted"], ha="center", va="center", fontfamily="Arial")

    # Agent 2: Market Analysis Agent
    draw_rounded_box(ax, ag2_x, ag_y, ag_w, ag_h, COLORS["card_white"], COLORS["card_border"], radius=0.08, lw=1.0)
    draw_simple_icon(ax, ag2_x + ag_w/2, ag_y + ag_h - 0.30, "chart", color="#38A169")
    ax.text(ag2_x + ag_w/2, ag_y + 0.50, "Market\nAnalysis Agent", fontsize=9.2, fontweight="bold", color=COLORS["text_navy"], ha="center", va="center", fontfamily="Arial")
    ax.text(ag2_x + ag_w/2, ag_y + 0.18, "(Qualitative Context)", fontsize=7.2, color=COLORS["text_muted"], ha="center", va="center", fontfamily="Arial")

    # Agent 3: Risk Scoring Agent
    draw_rounded_box(ax, ag3_x, ag_y, ag_w, ag_h, COLORS["card_white"], COLORS["card_border"], radius=0.08, lw=1.0)
    draw_simple_icon(ax, ag3_x + ag_w/2, ag_y + ag_h - 0.30, "gauge", color="#DD6B20")
    ax.text(ag3_x + ag_w/2, ag_y + 0.50, "Risk Scoring\nAgent", fontsize=9.2, fontweight="bold", color=COLORS["text_navy"], ha="center", va="center", fontfamily="Arial")
    ax.text(ag3_x + ag_w/2, ag_y + 0.18, "(ML + Fusion)", fontsize=7.2, color=COLORS["text_muted"], ha="center", va="center", fontfamily="Arial")

    # Horizontal flow arrows between agents: Status -> Market -> Risk Scoring
    draw_arrow(ax, ag1_x + ag_w, ag_y + 0.55, ag2_x, ag_y + 0.55, lw=1.2, mutation_scale=10)
    draw_arrow(ax, ag2_x + ag_w, ag_y + 0.55, ag3_x, ag_y + 0.55, lw=1.2, mutation_scale=10)

    # Reporting Agent (wide box at the bottom of Multi-Agent Layer)
    rep_w = 4.56
    rep_h = 0.55
    draw_rounded_box(ax, ma_x + (ma_w - rep_w)/2, 4.05, rep_w, rep_h, COLORS["card_white"], COLORS["card_border"], radius=0.08, lw=1.0)
    ax.text(ma_x + ma_w/2, 4.05 + rep_h/2, "Reporting Agent  (Narrative Synthesis & Mitigations)",
            fontsize=9.2, fontweight="bold", color=COLORS["text_navy"], ha="center", va="center", fontfamily="Arial")

    # Arrow from Risk Scoring Agent down to Reporting Agent
    draw_arrow(ax, ag3_x + ag_w/2, ag_y, ag3_x + ag_w/2, 4.60)

    # -------------------------------------------------------------------------
    # Right Container: AI INTELLIGENCE COMPONENTS (y: 3.80 to 7.92)
    # -------------------------------------------------------------------------
    ai_x, ai_y, ai_w, ai_h = 5.75, 3.80, 2.25, 4.12
    draw_rounded_box(ax, ai_x, ai_y, ai_w, ai_h, COLORS["container_fill"], COLORS["container_border"], radius=0.12, lw=1.2)
    
    # Header Banner
    draw_rounded_box(ax, ai_x, ai_y + ai_h - 0.40, ai_w, 0.40, COLORS["header_fill"], COLORS["header_border"], radius=0.08, lw=1.0)
    ax.text(ai_x + ai_w/2, ai_y + ai_h - 0.20, "AI INTELLIGENCE\nCOMPONENTS",
            fontsize=8.5, fontweight="bold", color=COLORS["text_navy"], ha="center", va="center", fontfamily="Arial")

    # 3 Stacked Cards inside AI Components
    aic_w = 1.95
    aic_h = 0.95
    aic_x = ai_x + (ai_w - aic_w)/2
    
    # AI Card 1: ML Risk Model
    draw_rounded_box(ax, aic_x, 6.35, aic_w, aic_h, COLORS["card_white"], COLORS["card_border"], radius=0.08, lw=1.0)
    draw_simple_icon(ax, aic_x + aic_w/2, 6.35 + aic_h - 0.28, "chart", color="#2B6CB0")
    ax.text(aic_x + aic_w/2, 6.35 + 0.40, "ML Risk Models", fontsize=8.8, fontweight="bold", color=COLORS["text_navy"], ha="center", va="center", fontfamily="Arial")
    ax.text(aic_x + aic_w/2, 6.35 + 0.18, "(Logistic Reg & Random Forest)", fontsize=6.8, color=COLORS["text_muted"], ha="center", va="center", fontfamily="Arial")

    # AI Card 2: NLP / Sentiment Analysis
    draw_rounded_box(ax, aic_x, 5.15, aic_w, aic_h, COLORS["card_white"], COLORS["card_border"], radius=0.08, lw=1.0)
    draw_simple_icon(ax, aic_x + aic_w/2, 5.15 + aic_h - 0.28, "chat", color="#2F855A")
    ax.text(aic_x + aic_w/2, 5.15 + 0.40, "NLP / Sentiment Analysis", fontsize=8.5, fontweight="bold", color=COLORS["text_navy"], ha="center", va="center", fontfamily="Arial")
    ax.text(aic_x + aic_w/2, 5.15 + 0.18, "(Domain Risk Lexicon)", fontsize=6.8, color=COLORS["text_muted"], ha="center", va="center", fontfamily="Arial")

    # AI Card 3: Semantic Retrieval
    draw_rounded_box(ax, aic_x, 3.95, aic_w, aic_h, COLORS["card_white"], COLORS["card_border"], radius=0.08, lw=1.0)
    draw_simple_icon(ax, aic_x + aic_w/2, 3.95 + aic_h - 0.28, "database", color="#4A5568")
    ax.text(aic_x + aic_w/2, 3.95 + 0.40, "Semantic Retrieval", fontsize=8.8, fontweight="bold", color=COLORS["text_navy"], ha="center", va="center", fontfamily="Arial")
    ax.text(aic_x + aic_w/2, 3.95 + 0.18, "(TF-IDF Vector Store / 15 Cases)", fontsize=6.6, color=COLORS["text_muted"], ha="center", va="center", fontfamily="Arial")

    # Bidirectional connector between Multi-Agent Layer and AI Components
    draw_double_arrow(ax, 5.50, 5.86, 5.75, 5.86, lw=1.5, mutation_scale=11)

    # Arrow from Multi-Agent Layer down to Application Layer
    draw_arrow(ax, 3.0, 3.80, 3.0, 3.25)

    # -------------------------------------------------------------------------
    # 4. APPLICATION LAYER (y: 2.30 to 3.25, h=0.95)
    # -------------------------------------------------------------------------
    app_x, app_y, app_w, app_h = 0.5, 2.30, 7.5, 0.95
    draw_rounded_box(ax, app_x, app_y, app_w, app_h, COLORS["container_fill"], COLORS["container_border"], radius=0.12, lw=1.2)
    
    # Header Banner
    draw_rounded_box(ax, app_x, app_y + app_h - 0.35, app_w, 0.35, COLORS["header_fill"], COLORS["header_border"], radius=0.08, lw=1.0)
    ax.text(app_x + app_w/2, app_y + app_h - 0.175, "APPLICATION LAYER",
            fontsize=10.5, fontweight="bold", color=COLORS["text_navy"], ha="center", va="center", fontfamily="Arial")

    # Two Cards inside Application Layer: FastAPI Backend & Streamlit Dashboard
    app_card_w = 3.35
    app_card_h = 0.45
    draw_rounded_box(ax, 0.75, 2.40, app_card_w, app_card_h, COLORS["card_white"], COLORS["card_border"], radius=0.08, lw=1.0)
    ax.text(0.75 + app_card_w/2, 2.40 + app_card_h/2, "FastAPI Backend",
            fontsize=9.2, fontweight="bold", color=COLORS["text_navy"], ha="center", va="center", fontfamily="Arial")

    draw_rounded_box(ax, 4.40, 2.40, app_card_w, app_card_h, COLORS["card_white"], COLORS["card_border"], radius=0.08, lw=1.0)
    ax.text(4.40 + app_card_w/2, 2.40 + app_card_h/2, "Streamlit Dashboard",
            fontsize=9.2, fontweight="bold", color=COLORS["text_navy"], ha="center", va="center", fontfamily="Arial")

    # Arrow from Application Layer down to Outputs
    draw_arrow(ax, 4.25, 2.30, 4.25, 1.85)

    # -------------------------------------------------------------------------
    # 5. OUTPUTS CONTAINER (y: 0.40 to 1.85, h=1.45)
    # -------------------------------------------------------------------------
    out_x, out_y, out_w, out_h = 0.5, 0.40, 7.5, 1.45
    draw_rounded_box(ax, out_x, out_y, out_w, out_h, COLORS["container_fill"], COLORS["container_border"], radius=0.12, lw=1.2)
    
    # Header Banner
    draw_rounded_box(ax, out_x, out_y + out_h - 0.35, out_w, 0.35, COLORS["header_fill"], COLORS["header_border"], radius=0.08, lw=1.0)
    ax.text(out_x + out_w/2, out_y + out_h - 0.175, "OUTPUTS",
            fontsize=10.5, fontweight="bold", color=COLORS["text_navy"], ha="center", va="center", fontfamily="Arial")

    # 6 Output Cards in 2 rows of 3 columns
    outputs_grid = [
        ("Risk Score (0–100)", 0.75, 1.00),
        ("Classification (3 Classes)", 3.10, 1.00),
        ("Project Risk Report", 5.45, 1.00),
        ("Risk Factor Alerts", 0.75, 0.52),
        ("Mitigation Recommendations", 3.10, 0.52),
        ("Decision Support Chatbot", 5.45, 0.52)
    ]
    og_w, og_h = 2.20, 0.38
    for otitle, ox, oy in outputs_grid:
        draw_rounded_box(ax, ox, oy, og_w, og_h, COLORS["card_white"], COLORS["card_border"], radius=0.06, lw=0.9)
        ax.text(ox + og_w/2, oy + og_h/2, otitle,
                fontsize=8.2, fontweight="bold", color=COLORS["text_navy"], ha="center", va="center", fontfamily="Arial")

    # -------------------------------------------------------------------------
    # Save Figure 1
    # -------------------------------------------------------------------------
    root_png = "fig 1.png"
    arch_png = os.path.join(OUTPUT_DIR, "figure1_overall_architecture.png")
    arch_svg = os.path.join(OUTPUT_DIR, "figure1_overall_architecture.svg")
    
    plt.savefig(root_png, format="png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig(arch_png, format="png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig(arch_svg, format="svg", bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"Saved Figure 1 to {root_png}, {arch_png}, and {arch_svg}")


# =============================================================================
# FIGURE 2: MULTI-AGENT ARCHITECTURE / WORKFLOW (Faithful to Reference Image 2)
# =============================================================================
def generate_figure2():
    print("Generating Figure 2: Multi-Agent Architecture (Simplified Reference Design)...")
    fig, ax = plt.subplots(figsize=(8.5, 10.5), dpi=300)
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])
    ax.set_xlim(0, 8.5)
    ax.set_ylim(0, 10.5)
    ax.axis('off')

    # -------------------------------------------------------------------------
    # Title at the top
    # -------------------------------------------------------------------------
    ax.text(
        4.25, 10.12,
        "Multi-Agent Architecture for IT Project Risk Management",
        fontsize=13.0, fontweight="bold", color=COLORS["text_navy"],
        ha="center", va="center", fontfamily="Arial"
    )

    # -------------------------------------------------------------------------
    # 1. TOP CARD: INPUT / DATA CONTEXT (y: 8.60 to 9.80, h=1.20)
    # -------------------------------------------------------------------------
    inp_x, inp_y, inp_w, inp_h = 2.45, 8.60, 3.60, 1.15
    draw_rounded_box(ax, inp_x, inp_y, inp_w, inp_h, COLORS["card_white"], COLORS["container_border"], radius=0.10, lw=1.2)
    # Header Banner
    draw_rounded_box(ax, inp_x, inp_y + inp_h - 0.32, inp_w, 0.32, COLORS["header_fill"], COLORS["header_border"], radius=0.08, lw=1.0)
    ax.text(inp_x + inp_w/2, inp_y + inp_h - 0.16, "INPUT / DATA CONTEXT",
            fontsize=9.5, fontweight="bold", color=COLORS["text_navy"], ha="center", va="center", fontfamily="Arial")
    
    inp_items = [
        "Internal Project Data (Telemetry: 19 Features)",
        "External Context (Communications & Retros)",
        "Historical Project Knowledge (15 IT Cases)"
    ]
    for i, item in enumerate(inp_items):
        ax.text(inp_x + inp_w/2, inp_y + 0.62 - i*0.22, item,
                fontsize=8.0, color=COLORS["text_body"], ha="center", va="center", fontfamily="Arial")

    # Arrow from Input Context -> Coordinator
    draw_arrow(ax, 4.25, inp_y, 4.25, 7.95)

    # -------------------------------------------------------------------------
    # 2. COORDINATOR AGENT (y: 7.30 to 7.95, h=0.65)
    # -------------------------------------------------------------------------
    coord_x, coord_y, coord_w, coord_h = 2.45, 7.30, 3.60, 0.65
    draw_rounded_box(ax, coord_x, coord_y, coord_w, coord_h, COLORS["agent_fill"], COLORS["agent_border"], radius=0.10, lw=1.2)
    ax.text(coord_x + coord_w/2, coord_y + 0.42, "Project Risk Manager",
            fontsize=9.8, fontweight="bold", color=COLORS["text_navy"], ha="center", va="center", fontfamily="Arial")
    ax.text(coord_x + coord_w/2, coord_y + 0.20, "(Coordinator Agent)",
            fontsize=8.5, fontstyle="italic", color=COLORS["text_muted"], ha="center", va="center", fontfamily="Arial")

    # -------------------------------------------------------------------------
    # 3. MIDDLE AGENTS: Status Agent (Left), Market Analyst (Middle), AI Components (Right)
    # -------------------------------------------------------------------------
    # Project Status Tracking Agent (Left)
    sta_x, sta_y, sta_w, sta_h = 0.80, 5.85, 2.25, 0.85
    draw_rounded_box(ax, sta_x, sta_y, sta_w, sta_h, COLORS["agent_fill"], COLORS["agent_border"], radius=0.10, lw=1.2)
    ax.text(sta_x + sta_w/2, sta_y + 0.52, "Project Status",
            fontsize=9.2, fontweight="bold", color=COLORS["text_navy"], ha="center", va="center", fontfamily="Arial")
    ax.text(sta_x + sta_w/2, sta_y + 0.30, "Tracking Agent",
            fontsize=9.2, fontweight="bold", color=COLORS["text_navy"], ha="center", va="center", fontfamily="Arial")

    # Market Analysis Agent (Middle)
    mkt_x, mkt_y, mkt_w, mkt_h = 3.25, 5.85, 2.25, 0.85
    draw_rounded_box(ax, mkt_x, mkt_y, mkt_w, mkt_h, COLORS["agent_fill"], COLORS["agent_border"], radius=0.10, lw=1.2)
    ax.text(mkt_x + mkt_w/2, mkt_y + 0.52, "Market Analysis",
            fontsize=9.2, fontweight="bold", color=COLORS["text_navy"], ha="center", va="center", fontfamily="Arial")
    ax.text(mkt_x + mkt_w/2, mkt_y + 0.30, "Agent",
            fontsize=9.2, fontweight="bold", color=COLORS["text_navy"], ha="center", va="center", fontfamily="Arial")

    # AI Intelligence Components Stack (Right)
    aic_x, aic_y, aic_w, aic_h = 5.80, 4.00, 2.25, 2.70
    draw_rounded_box(ax, aic_x, aic_y, aic_w, aic_h, COLORS["container_fill"], COLORS["container_border"], radius=0.10, lw=1.2)
    
    # AI Header
    draw_rounded_box(ax, aic_x, aic_y + aic_h - 0.40, aic_w, 0.40, COLORS["header_fill"], COLORS["header_border"], radius=0.08, lw=1.0)
    ax.text(aic_x + aic_w/2, aic_y + aic_h - 0.20, "AI Intelligence\nComponents",
            fontsize=8.5, fontweight="bold", color=COLORS["text_navy"], ha="center", va="center", fontfamily="Arial")

    # Stacked components inside AI Components
    comp_h = 0.65
    # 1. ML Risk Model
    draw_rounded_box(ax, aic_x + 0.12, 5.55, aic_w - 0.24, comp_h, COLORS["card_white"], COLORS["card_border"], radius=0.06, lw=0.9)
    ax.text(aic_x + aic_w/2, 5.55 + 0.40, "ML Risk Models", fontsize=8.2, fontweight="bold", color=COLORS["text_navy"], ha="center", va="center", fontfamily="Arial")
    ax.text(aic_x + aic_w/2, 5.55 + 0.20, "(Logistic Reg / RF)", fontsize=7.0, color=COLORS["text_muted"], ha="center", va="center", fontfamily="Arial")

    # 2. NLP / Sentiment Analysis
    draw_rounded_box(ax, aic_x + 0.12, 4.80, aic_w - 0.24, comp_h, COLORS["card_white"], COLORS["card_border"], radius=0.06, lw=0.9)
    ax.text(aic_x + aic_w/2, 4.80 + 0.40, "NLP / Sentiment", fontsize=8.2, fontweight="bold", color=COLORS["text_navy"], ha="center", va="center", fontfamily="Arial")
    ax.text(aic_x + aic_w/2, 4.80 + 0.20, "(Domain Risk Lexicon)", fontsize=7.0, color=COLORS["text_muted"], ha="center", va="center", fontfamily="Arial")

    # 3. Semantic Retrieval
    draw_rounded_box(ax, aic_x + 0.12, 4.05, aic_w - 0.24, comp_h, COLORS["card_white"], COLORS["card_border"], radius=0.06, lw=0.9)
    ax.text(aic_x + aic_w/2, 4.05 + 0.40, "Semantic Retrieval", fontsize=8.2, fontweight="bold", color=COLORS["text_navy"], ha="center", va="center", fontfamily="Arial")
    ax.text(aic_x + aic_w/2, 4.05 + 0.20, "(TF-IDF Vector Store)", fontsize=7.0, color=COLORS["text_muted"], ha="center", va="center", fontfamily="Arial")

    # Connectors from Coordinator down to agents
    draw_arrow(ax, 3.20, coord_y, 1.92, sta_y + sta_h)
    draw_arrow(ax, 4.25, coord_y, 4.37, mkt_y + mkt_h)
    # Connector from Coordinator to AI Components
    ax.plot([5.30, 6.92, 6.92], [coord_y + 0.32, coord_y + 0.32, aic_y + aic_h], color=COLORS["arrow_color"], lw=1.2, zorder=3)
    draw_arrow(ax, 6.92, aic_y + aic_h + 0.08, 6.92, aic_y + aic_h, lw=1.2)

    # Bidirectional arrow between Market Analysis Agent and NLP Sentiment
    draw_double_arrow(ax, mkt_x + mkt_w, 5.15, aic_x, 5.15, lw=1.3, mutation_scale=10)

    # -------------------------------------------------------------------------
    # 4. RISK SCORING AGENT (y: 4.45 to 5.15, h=0.70)
    # -------------------------------------------------------------------------
    rsk_x, rsk_y, rsk_w, rsk_h = 2.45, 4.45, 2.50, 0.70
    draw_rounded_box(ax, rsk_x, rsk_y, rsk_w, rsk_h, COLORS["agent_fill"], COLORS["agent_border"], radius=0.10, lw=1.2)
    ax.text(rsk_x + rsk_w/2, rsk_y + 0.44, "Risk Scoring Agent",
            fontsize=9.8, fontweight="bold", color=COLORS["text_navy"], ha="center", va="center", fontfamily="Arial")
    ax.text(rsk_x + rsk_w/2, rsk_y + 0.20, "(ML Inference + Heuristic Fusion)",
            fontsize=7.2, color=COLORS["text_muted"], ha="center", va="center", fontfamily="Arial")

    # Arrows from Status Tracking Agent & Market Analysis Agent into Risk Scoring Agent
    draw_arrow(ax, 1.92, sta_y, rsk_x + 0.5, rsk_y + rsk_h)
    draw_arrow(ax, 4.37, mkt_y, rsk_x + rsk_w - 0.5, rsk_y + rsk_h)

    # Bidirectional arrow between Risk Scoring Agent and AI Components (ML / Retrieval)
    draw_double_arrow(ax, rsk_x + rsk_w, 4.80, aic_x, 4.80, lw=1.3, mutation_scale=10)

    # -------------------------------------------------------------------------
    # 5. REPORTING AGENT (y: 3.20 to 3.85, h=0.65)
    # -------------------------------------------------------------------------
    rep_x, rep_y, rep_w, rep_h = 2.45, 3.20, 2.50, 0.65
    draw_rounded_box(ax, rep_x, rep_y, rep_w, rep_h, COLORS["agent_fill"], COLORS["agent_border"], radius=0.10, lw=1.2)
    ax.text(rep_x + rep_w/2, rep_y + 0.40, "Reporting Agent",
            fontsize=9.8, fontweight="bold", color=COLORS["text_navy"], ha="center", va="center", fontfamily="Arial")
    ax.text(rep_x + rep_w/2, rep_y + 0.18, "(Risk Drivers & Actionable Mitigations)",
            fontsize=7.2, color=COLORS["text_muted"], ha="center", va="center", fontfamily="Arial")

    # Arrow from Risk Scoring Agent -> Reporting Agent
    draw_arrow(ax, rsk_x + rsk_w/2, rsk_y, rep_x + rep_w/2, rep_y + rep_h)

    # Arrow from Semantic Retrieval (AI Components) into Reporting Agent
    ax.plot([aic_x, 5.25, 5.25], [4.35, 4.35, rep_y + rep_h/2], color=COLORS["arrow_color"], lw=1.2, zorder=3)
    draw_arrow(ax, 5.25, rep_y + rep_h/2, rep_x + rep_w, rep_y + rep_h/2, lw=1.2)

    # -------------------------------------------------------------------------
    # 6. RISK ASSESSMENT & REPORT (y: 2.05 to 2.65, h=0.60)
    # -------------------------------------------------------------------------
    rar_x, rar_y, rar_w, rar_h = 2.25, 2.05, 2.90, 0.60
    draw_rounded_box(ax, rar_x, rar_y, rar_w, rar_h, COLORS["header_fill"], COLORS["header_border"], radius=0.10, lw=1.2)
    ax.text(rar_x + rar_w/2, rar_y + rar_h/2, "Risk Assessment & Report",
            fontsize=10.0, fontweight="bold", color=COLORS["text_navy"], ha="center", va="center", fontfamily="Arial")

    # Arrow from Reporting Agent -> Risk Assessment & Report
    draw_arrow(ax, rep_x + rep_w/2, rep_y, rar_x + rar_w/2, rar_y + rar_h)

    # -------------------------------------------------------------------------
    # 7. FINAL THREE DELIVERABLE BOXES (y: 0.75 to 1.40, h=0.65)
    # -------------------------------------------------------------------------
    deliv_w = 2.30
    deliv_h = 0.65
    d1_x = 0.60
    d2_x = 3.10
    d3_x = 5.60
    deliv_y = 0.75
    
    # Branching arrows from Risk Assessment & Report
    draw_arrow(ax, 2.70, rar_y, d1_x + deliv_w/2, deliv_y + deliv_h)
    draw_arrow(ax, 3.70, rar_y, d2_x + deliv_w/2, deliv_y + deliv_h)
    draw_arrow(ax, 4.70, rar_y, d3_x + deliv_w/2, deliv_y + deliv_h)

    # Deliverable 1: Risk Score & Classification
    draw_rounded_box(ax, d1_x, deliv_y, deliv_w, deliv_h, COLORS["card_white"], COLORS["container_border"], radius=0.08, lw=1.1)
    ax.text(d1_x + deliv_w/2, deliv_y + 0.42, "Risk Score &", fontsize=8.8, fontweight="bold", color=COLORS["text_navy"], ha="center", va="center", fontfamily="Arial")
    ax.text(d1_x + deliv_w/2, deliv_y + 0.20, "Classification (3 Classes)", fontsize=8.0, color=COLORS["text_body"], ha="center", va="center", fontfamily="Arial")

    # Deliverable 2: Mitigation Recommendations
    draw_rounded_box(ax, d2_x, deliv_y, deliv_w, deliv_h, COLORS["card_white"], COLORS["container_border"], radius=0.08, lw=1.1)
    ax.text(d2_x + deliv_w/2, deliv_y + 0.42, "Mitigation", fontsize=8.8, fontweight="bold", color=COLORS["text_navy"], ha="center", va="center", fontfamily="Arial")
    ax.text(d2_x + deliv_w/2, deliv_y + 0.20, "Recommendations", fontsize=8.0, color=COLORS["text_body"], ha="center", va="center", fontfamily="Arial")

    # Deliverable 3: Project Risk Report
    draw_rounded_box(ax, d3_x, deliv_y, deliv_w, deliv_h, COLORS["card_white"], COLORS["container_border"], radius=0.08, lw=1.1)
    ax.text(d3_x + deliv_w/2, deliv_y + 0.42, "Project Risk", fontsize=8.8, fontweight="bold", color=COLORS["text_navy"], ha="center", va="center", fontfamily="Arial")
    ax.text(d3_x + deliv_w/2, deliv_y + 0.20, "Report (Executive Summary)", fontsize=7.8, color=COLORS["text_body"], ha="center", va="center", fontfamily="Arial")

    # -------------------------------------------------------------------------
    # Save Figure 2
    # -------------------------------------------------------------------------
    root_png = "fig 2.png"
    arch_png = os.path.join(OUTPUT_DIR, "figure2_multi_agent_workflow.png")
    arch_svg = os.path.join(OUTPUT_DIR, "figure2_multi_agent_workflow.svg")
    
    plt.savefig(root_png, format="png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig(arch_png, format="png", dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor='none')
    plt.savefig(arch_svg, format="svg", bbox_inches="tight", facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"Saved Figure 2 to {root_png}, {arch_png}, and {arch_svg}")


if __name__ == "__main__":
    generate_figure1()
    generate_figure2()
    print("Both simplified figures generated successfully!")
