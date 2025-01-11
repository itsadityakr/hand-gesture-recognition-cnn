import matplotlib.pyplot as plt
import numpy as np

# Data for Performance Metrics
performance_metrics = {
    "Models": ["Proposed CNN", "YOLOv2", "AlexNet", "Myoelectric", "EMG-based"],
    "Accuracy": [95, 88, 85, 80, 82],
    "Processing Speed": [90, 75, 70, 85, 80],
    "Gesture Range": [95, 70, 65, 60, 55],
}

# Data for Cost-Benefit Analysis
cost_benefit_data = {
    "Categories": ["Implementation Cost", "Maintenance Cost", "Training Time (hours)"],
    "Proposed CNN": [200, 50, 24],
    "YOLOv2": [800, 200, 48],
    "AlexNet": [750, 180, 40],
    "Myoelectric": [1200, 300, 36],
    "EMG-based": [1000, 250, 32],
}

# Data for Radar Metrics
radar_metrics = {
    "Metrics": ["Accuracy", "Real-time Performance", "Gesture Range", "Cost Efficiency", "Power Efficiency", "Ease of Implementation"],
    "Proposed CNN": [95, 90, 95, 95, 85, 85],
    "Average Models": [84, 77, 63, 50, 68, 70],
}

# Function to save performance metrics bar chart
def save_performance_metrics_chart():
    labels = performance_metrics["Models"]
    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(x - width, performance_metrics["Accuracy"], width, label="Accuracy")
    ax.bar(x, performance_metrics["Processing Speed"], width, label="Processing Speed")
    ax.bar(x + width, performance_metrics["Gesture Range"], width, label="Gesture Range")

    ax.set_ylabel("Scores")
    ax.set_title("Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.legend()
    fig.tight_layout()
    plt.savefig("performance_metrics.png")
    plt.close(fig)


# Function to save cost-benefit analysis bar chart
def save_cost_benefit_chart():
    labels = cost_benefit_data["Categories"]
    x = np.arange(len(labels))
    width = 0.15

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - 2 * width, cost_benefit_data["Proposed CNN"], width, label="Proposed CNN")
    ax.bar(x - width, cost_benefit_data["YOLOv2"], width, label="YOLOv2")
    ax.bar(x, cost_benefit_data["AlexNet"], width, label="AlexNet")
    ax.bar(x + width, cost_benefit_data["Myoelectric"], width, label="Myoelectric")
    ax.bar(x + 2 * width, cost_benefit_data["EMG-based"], width, label="EMG-based")

    ax.set_ylabel("Cost/Time")
    ax.set_title("Cost-Benefit Analysis")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    fig.tight_layout()
    plt.savefig("cost_benefit_analysis.png")
    plt.close(fig)


# Function to save radar chart
def save_radar_chart():
    metrics = radar_metrics["Metrics"]
    proposed = radar_metrics["Proposed CNN"]
    average = radar_metrics["Average Models"]

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    proposed += proposed[:1]
    average += average[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.fill(angles, proposed, color="blue", alpha=0.25, label="Proposed CNN")
    ax.fill(angles, average, color="gray", alpha=0.25, label="Average Models")
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_title("Comprehensive Model Comparison", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    plt.savefig("radar_chart.png")
    plt.close(fig)


# Save all charts
save_performance_metrics_chart()
save_cost_benefit_chart()
save_radar_chart()

"Charts saved successfully as PNG files."
