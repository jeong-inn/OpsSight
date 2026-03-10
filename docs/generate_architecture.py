import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')
fig.patch.set_facecolor('#0e1117')

def draw_box(ax, x, y, w, h, label, sublabel="", color="#1f77b4"):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                         facecolor=color, edgecolor="white", linewidth=1.5, alpha=0.9)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2 + (0.15 if sublabel else 0),
            label, ha='center', va='center', color='white', fontsize=10, fontweight='bold')
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.22, sublabel, ha='center', va='center',
                color='#cccccc', fontsize=7.5)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color="white", lw=1.5))

ax.text(7, 9.5, "FabSight - System Architecture",
        ha='center', va='center', color='white', fontsize=14, fontweight='bold')

draw_box(ax, 5, 8.5, 4, 0.7, "SECOM Sensor Data", "590 sensors / 1,567 samples", "#2c3e50")
draw_box(ax, 5, 7.4, 4, 0.7, "Preprocessing", "Missing value / StandardScaler", "#34495e")
arrow(ax, 7, 8.5, 7, 8.1)

draw_box(ax, 2.0, 6.1, 3.5, 0.8, "SPC Control Chart", "3-sigma UCL/LCL", "#1a5276")
draw_box(ax, 7.0, 6.1, 3.5, 0.8, "Isolation Forest", "Unsupervised Anomaly Detection", "#1a5276")
arrow(ax, 7, 7.4, 5.0, 6.9)
arrow(ax, 7, 7.4, 8.5, 6.9)

draw_box(ax, 5, 4.9, 4, 0.8, "Pre-failure Risk Scoring", "GBM, Risk Score 0 to 1", "#6e2fa0")
arrow(ax, 8.5, 6.1, 7.5, 5.7)

draw_box(ax, 5, 3.7, 4, 0.8, "SHAP Feature Importance", "Top 5 Sensors / Process Mapping", "#0e6655")
arrow(ax, 7, 4.9, 7, 4.5)

agent_colors = ["#922b21", "#784212", "#145a32", "#1a3a5c"]
agent_labels = ["Detection\nAgent", "Diagnosis\nAgent", "Action\nAgent", "Report\nAgent"]
for i, (label, color) in enumerate(zip(agent_labels, agent_colors)):
    draw_box(ax, 1 + i*3, 2.3, 2.5, 0.9, label, "", color)
    if i < 3:
        arrow(ax, 3.5 + i*3, 2.75, 4 + i*3, 2.75)
arrow(ax, 7, 3.7, 7, 3.2)

draw_box(ax, 3.0, 1.0, 8, 0.9, "Streamlit Dashboard",
         "FAB Monitoring / Alert System / Stream Simulator / Operation Log", "#17202a")
arrow(ax, 7, 2.3, 7, 1.9)

legend = [("#2c3e50","Data"),("#1a5276","Detection"),("#6e2fa0","Prediction"),
          ("#0e6655","Analysis"),("#922b21","Agent"),("#17202a","Dashboard")]
for i, (c, l) in enumerate(legend):
    rect = FancyBboxPatch((0.3+i*2.3, 0.2), 0.4, 0.3,
                          boxstyle="round,pad=0.05", facecolor=c, edgecolor="white", linewidth=1)
    ax.add_patch(rect)
    ax.text(0.85+i*2.3, 0.35, l, color='white', fontsize=7.5, va='center')

plt.tight_layout()
plt.savefig("/Users/jeonginn/projects/fabsight/docs/architecture.png",
            dpi=150, bbox_inches='tight', facecolor='#0e1117')
print("Done!")
