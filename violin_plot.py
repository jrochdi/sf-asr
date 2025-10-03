import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

csv_path = "streaming_results_fp32_rnnt_80ms.csv"
df = pd.read_csv(csv_path)
field = "first_partial_ms"
df = df.dropna(subset=[field])

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "text.usetex": True,  
})

plt.figure(figsize=(6, 4))
sns.violinplot(y=df[field], inner="box", color="skyblue")

plt.ylabel(f"{field.replace('_',' ').title()} (ms)")
plt.title(f"Distribution of {field.replace('_',' ').title()}")
plt.tight_layout()

out_path = f"violin_{field}.pdf"
plt.savefig(out_path)
plt.close()
print(f"Saved violin plot to {out_path}")