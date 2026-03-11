import os
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv_path = "results/plot_ready_summary.csv"
save_dir = "results/figures"
os.makedirs(save_dir, exist_ok=True)

df = pd.read_csv(csv_path)

method_map = {
    "base_model": "Base",
    "dora_final": "DoRA",
    "lora_final": "LoRA",
    "lora_prior_final": "LoRA + Prior"
}
df["method"] = df["method"].map(method_map)

domain_map = {
    "Anime_Faces": "Anime Faces",
    "flower_birdofparadise": "Bird of Paradise",
    "stanford_car": "Stanford Cars"
}
df["domain"] = df["domain"].map(domain_map)

method_order = ["Base", "DoRA", "LoRA", "LoRA + Prior"]
domain_order = ["Anime Faces", "Bird of Paradise", "Stanford Cars"]

sns.set_theme(style="whitegrid", font_scale=1.1)

df_5shot = df[df["shots"] == 5].copy()

plt.figure(figsize=(11,6))
ax = sns.barplot(
    data=df_5shot,
    x="domain",
    y="domain_fidelity_fid",
    hue="method",
    hue_order=method_order,
    order=domain_order
)
ax.set_title("Domain Fidelity (FID) Across Methods (5-shot)")
ax.set_xlabel("")
ax.set_ylabel("FID (lower is better)")
plt.legend(title="Method", bbox_to_anchor=(1.02,1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(save_dir,"fig1_fid_bar.png"), dpi=300, bbox_inches="tight")
plt.close()



df_method_avg = (
    df.groupby("method", as_index=False)
    .agg({
        "generative_diversity": "mean",
        "domain_fidelity_clip": "mean"
    })
)

df_method_avg["method"] = pd.Categorical(
    df_method_avg["method"],
    categories=method_order,
    ordered=True
)
df_method_avg = df_method_avg.sort_values("method")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))

sns.barplot(
    data=df_method_avg,
    x="method",
    y="generative_diversity",
    order=method_order,
    ax=axes[0]
)
axes[0].set_title("Average Generative Diversity")
axes[0].set_xlabel("")
axes[0].set_ylabel("Higher is better")
axes[0].tick_params(axis="x", rotation=15)

for container in axes[0].containers:
    axes[0].bar_label(container, fmt="%.3f", padding=3, fontsize=9)

sns.barplot(
    data=df_method_avg,
    x="method",
    y="domain_fidelity_clip",
    order=method_order,
    ax=axes[1]
)
axes[1].set_title("Average Domain Fidelity CLIP")
axes[1].set_xlabel("")
axes[1].set_ylabel("Higher is better")
axes[1].tick_params(axis="x", rotation=15)

for container in axes[1].containers:
    axes[1].bar_label(container, fmt="%.3f", padding=3, fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "fig2_method_average_bars.png"), dpi=300, bbox_inches="tight")
plt.close()

df_method_avg = (
    df.groupby("method", as_index=False)
    .agg({
        "generative_diversity": "mean",
        "domain_fidelity_clip": "mean"
    })
)

plot_df = df_method_avg.melt(
    id_vars="method",
    value_vars=["generative_diversity", "domain_fidelity_clip"],
    var_name="metric",
    value_name="score"
)

metric_map = {
    "generative_diversity": "Generative Diversity",
    "domain_fidelity_clip": "Domain Fidelity CLIP"
}
plot_df["metric"] = plot_df["metric"].map(metric_map)

plt.figure(figsize=(9, 5))
ax = sns.barplot(
    data=plot_df,
    x="method",
    y="score",
    hue="metric",
    order=method_order
)

ax.set_title("Average Performance by Method")
ax.set_xlabel("")
ax.set_ylabel("Score (higher is better)")
ax.tick_params(axis="x", rotation=15)

for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", padding=2, fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "fig2_method_grouped_bars.png"), dpi=300, bbox_inches="tight")
plt.close()


df_anime = df[df["domain"] == "Anime Faces"].copy()

plt.figure(figsize=(8,6))
ax = sns.barplot(
    data=df_anime,
    x="shots",
    y="domain_fidelity_fid",
    hue="method",
    hue_order=method_order
)
ax.set_title("Effect of Number of Shots on FID (Anime Faces)")
ax.set_xlabel("Number of Shots")
ax.set_ylabel("FID (lower is better)")
plt.legend(title="Method", bbox_to_anchor=(1.02,1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(save_dir,"fig3_shots_vs_fid_bar.png"), dpi=300, bbox_inches="tight")
plt.close()





plt.figure(figsize=(11,6))
ax = sns.barplot(
    data=df_5shot,
    x="domain",
    y="prompt_controllability",
    hue="method",
    hue_order=method_order,
    order=domain_order
)
ax.set_title("Prompt Controllability Across Methods (5-shot)")
ax.set_xlabel("")
ax.set_ylabel("Prompt Controllability (higher is better)")
plt.legend(title="Method", bbox_to_anchor=(1.02,1), loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(save_dir,"fig4_prompt_controllability.png"), dpi=300, bbox_inches="tight")
plt.close()

print("All figures saved to:", save_dir)