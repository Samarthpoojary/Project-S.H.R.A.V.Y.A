import pandas as pd
import os

BASE_DIR = r"C:\Users\shrey\OneDrive\Desktop\ANC_Benchmark"
RESULTS_DIR = os.path.join(BASE_DIR, "results")

files = {
    "SNR": "snr_206_results.csv",
    "PESQ": "pesq_206_results.csv",
    "STOI": "stoi_206_results.csv",
    "Noise Reduction": "noise_reduction_206_results.csv"
}

averages = []

for metric, filename in files.items():
    path = os.path.join(RESULTS_DIR, filename)
    df = pd.read_csv(path)

    # Take the average of every numeric column
    numeric_cols = df.select_dtypes(include="number").columns

    for col in numeric_cols:
        averages.append({
            "Metric": metric,
            "Value": col,
            "Average": df[col].mean()
        })

result = pd.DataFrame(averages)

output_path = os.path.join(RESULTS_DIR, "final_averages.csv")
result.to_csv(output_path, index=False)

print("\n===== FINAL AVERAGES =====")
print(result.to_string(index=False))
print(f"\nSaved to: {output_path}")