import re
import matplotlib.pyplot as plt

# === 1. Set your log file path here ===
log_file_path = "work_dirs/local-SMDA_TS_IterUpdate/250615_2213_SMDA_EMD_Exp_2_urban50rural_2rural_pseudo_lr06_tsw0.8_baw_05_mask_psmulJun_15_b5c6a/20250615_221413.log"  # <-- Replace this with your actual file path

# === 2. Read the log file ===
with open(log_file_path, "r") as f:
    log_data = f.read()

# === 3. Extract relevant lines ===
pattern = r"Trust score updated at iteration (\d+): \[(.*?)\]"
matches = re.findall(pattern, log_data)

# === 4. Parse iteration and trust scores ===
iterations = []
scores_per_class = [[] for _ in range(7)]  # 7 classes

for it, score_str in matches:
    iterations.append(int(it))
    scores = list(map(float, score_str.split(", ")))
    for i in range(7):
        scores_per_class[i].append(scores[i])

# === 5. Plot ===
class_names = ['background', 'building', 'road', 'water', 'barren', 'forest', 'agricultural']
plt.figure(figsize=(10, 6))

for i in range(7):
    plt.plot(iterations, scores_per_class[i], label=class_names[i])

plt.xlabel("Iteration")
plt.ylabel("Trust Score")
plt.title("Trust Score Changes per Class")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plot_trust_scores.png")

