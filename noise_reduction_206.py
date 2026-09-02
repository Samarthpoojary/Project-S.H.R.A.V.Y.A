import os
import numpy as np
import soundfile as sf
import pandas as pd

base_path = r"C:\Users\shrey\OneDrive\Desktop\ANC_Benchmark\data"
results_path = r"C:\Users\shrey\OneDrive\Desktop\ANC_Benchmark\results"

results = []

for i in range(1, 207):
    folder = os.path.join(base_path, f"Sample_{i:03d}")

    clean, _ = sf.read(os.path.join(folder, "1_Clean_Voice.wav"))
    noisy, _ = sf.read(os.path.join(folder, "2_Noisy_Input.wav"))
    enhanced, _ = sf.read(os.path.join(folder, "3_Enhanced_Output.wav"))

    min_len = min(len(clean), len(noisy), len(enhanced))

    clean = clean[:min_len]
    noisy = noisy[:min_len]
    enhanced = enhanced[:min_len]

    original_noise = noisy - clean
    remaining_noise = enhanced - clean

    original_noise_power = np.mean(original_noise ** 2)
    remaining_noise_power = np.mean(remaining_noise ** 2)

    noise_reduction_db = 10 * np.log10(
        original_noise_power / remaining_noise_power
    )

    results.append([
        f"Sample_{i:03d}",
        noise_reduction_db
    ])

    print(
        f"Sample {i:03d}: "
        f"Noise Reduction = {noise_reduction_db:.2f} dB"
    )

df = pd.DataFrame(
    results,
    columns=["Sample", "Noise_Reduction_dB"]
)

output_file = os.path.join(results_path, "noise_reduction_206_results.csv")
df.to_csv(output_file, index=False)

print("\n✅ Noise reduction calculation completed.")
print("Saved to:", output_file)