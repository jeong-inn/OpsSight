import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

def draw_box(ax, x, y, w, h, label, color="#1f77b4", fontsize=9):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                         facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.9)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, label, ha='center', va='center',
            color='white', fontsize=fontsize, fontweight='bold')

def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="white", lw=1.5))

# ── 1. System Architecture ──────────────────────────────
fig, ax = plt.subplots(figsize=(16, 12))
ax.set_xlim(0, 16)
ax.set_ylim(0, 12)
ax.axis('off')
fig.patch.set_facecolor('#0e1117')

draw_box(ax, 5.5, 10.3, 5, 0.8, "Multi-Signal Input Data\n(590 sensors, 1,567 samples)", "#2c3e50", 8)
draw_box(ax, 5.5, 9.1, 5, 0.8, "Preprocessing\n(Missing value, StandardScaler)", "#34495e", 8)
arrow(ax, 8, 10.3, 8, 9.9)

draw_box(ax, 1.5, 7.7, 3.5, 0.9, "SPC Control Chart\n(3-sigma UCL/LCL)", "#1a5276", 8)
draw_box(ax, 6.5, 7.7, 3.5, 0.9, "Isolation Forest\n(+ PCA 50dim)", "#1a5276", 8)
draw_box(ax, 11.0, 7.7, 3.5, 0.9, "Digital Twin\nSimulator", "#1a6644", 8)
arrow(ax, 8, 9.1, 3.5, 8.6)
arrow(ax, 8, 9.1, 8, 8.6)
arrow(ax, 8, 9.1, 12.5, 8.6)

draw_box(ax, 5.5, 6.4, 5, 0.9, "Pre-failure Risk Scoring\n(GBM + SMOTE + Threshold)", "#6e2fa0", 8)
arrow(ax, 8, 7.7, 8, 7.3)

draw_box(ax, 5.5, 5.1, 5, 0.9, "SHAP Feature Importance\n(Process Contribution Analysis)", "#0e6655", 8)
arrow(ax, 8, 6.4, 8, 6.0)

agent_colors = ["#922b21", "#784212", "#145a32", "#1a3a5c"]
agent_labels = ["Detection\nAgent", "Diagnosis\nAgent", "Action\nAgent", "Report Agent\n(GPT-4o-mini)"]
for i, (label, color) in enumerate(zip(agent_labels, agent_colors)):
    draw_box(ax, 0.8 + i*3.8, 3.3, 3.0, 1.0, label, color, 8)
    if i < 3:
        arrow(ax, 3.8 + i*3.8, 3.8, 4.0 + i*3.8, 3.8)
arrow(ax, 8, 5.1, 8, 4.3)

draw_box(ax, 3.0, 1.8, 10, 1.0, "Streamlit Dashboard\n(7 tabs: Monitoring / SPC / Anomaly / SHAP / Agent / Log / Simulator)", "#17202a", 8)
arrow(ax, 8, 3.3, 8, 2.8)

draw_box(ax, 3.0, 0.5, 4.5, 0.9, "Operation Log\n(Auto saved)", "#2c3e50", 8)
draw_box(ax, 8.5, 0.5, 4.5, 0.9, "Agent Report\n(LLM auto generated)", "#2c3e50", 8)
arrow(ax, 6, 1.8, 5.5, 1.4)
arrow(ax, 10, 1.8, 10.5, 1.4)

plt.tight_layout()
plt.savefig("docs/architecture.png", dpi=150, bbox_inches='tight', facecolor='#0e1117')
print("Saved: docs/architecture.png")
plt.close()

# ── 2. Agent Flow ────────────────────────────────────────
fig2, ax2 = plt.subplots(figsize=(14, 8))
ax2.set_xlim(0, 14)
ax2.set_ylim(0, 8)
ax2.axis('off')
fig2.patch.set_facecolor('#0e1117')

draw_box(ax2, 4.5, 6.3, 5, 0.8, "Sensor Data + SHAP Results", "#2c3e50", 9)
draw_box(ax2, 4.5, 5.1, 5, 0.8, "LLM (GPT-4o-mini)\nTool Call Decision", "#8e44ad", 9)
arrow(ax2, 7, 6.3, 7, 5.9)

tool_colors = ["#922b21", "#784212", "#145a32", "#1a3a5c"]
tool_labels = [
    "Tool 1\nanalyze_anomaly\n(Severity)",
    "Tool 2\ndiagnose_root_cause\n(Process)",
    "Tool 3\nget_action_plan\n(Actions)",
    "Tool 4\ngenerate_report\n(LLM Report)"
]
for i, (label, color) in enumerate(zip(tool_labels, tool_colors)):
    draw_box(ax2, 0.3 + i*3.4, 3.2, 3.0, 1.4, label, color, 8)
    arrow(ax2, 7, 5.1, 1.8 + i*3.4, 4.6)

draw_box(ax2, 4.5, 2.0, 5, 0.8, "Context Accumulation\n(Each Tool result -> Next Tool input)", "#0e6655", 9)
for i in range(4):
    arrow(ax2, 1.8 + i*3.4, 3.2, 6.5, 2.8)

draw_box(ax2, 4.5, 0.7, 5, 0.9, "Final Operator Report\n(Auto-generated Markdown)", "#17202a", 9)
arrow(ax2, 7, 2.0, 7, 1.6)

plt.tight_layout()
plt.savefig("docs/agent_flow.png", dpi=150, bbox_inches='tight', facecolor='#0e1117')
print("Saved: docs/agent_flow.png")
plt.close()
