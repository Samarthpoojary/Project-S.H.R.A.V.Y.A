import os
import soundfile as sf
from pystoi import stoi
import pandas as pd

base_path = r"C:\Users\shrey\OneDrive\Desktop\ANC_Benchmark\data"

results = []

for i in range(1, 207):

    folder = os.path.join(base_path, f"Sample_{i:03d}")

    clean, sr_clean = sf.read(os.path.join(folder, "1_Clean_Voice.wav"))
    noisy, sr_noisy = sf.read(os.path.join(folder, "2_Noisy_Input.wav"))
    enhanced, sr_enhanced = sf.read(os.path.join(folder, "3_Enhanced_Output.wav"))

    min_len = min(len(clean), len(noisy), len(enhanced))

    clean = clean[:min_len]
    noisy = noisy[:min_len]
    enhanced = enhanced[:min_len]

    noisy_stoi = stoi(clean, noisy, sr_clean, extended=False)
    enhanced_stoi = stoi(clean, enhanced, sr_clean, extended=False)

    improvement = enhanced_stoi - noisy_stoi

    results.append([
        f"Sample_{i:03d}",
        noisy_stoi,
        enhanced_stoi,
        improvement
    ])

    print(
        f"Sample_{i:03d}: "
        f"Noisy STOI = {noisy_stoi:.3f} | "
        f"Enhanced STOI = {enhanced_stoi:.3f} | "
        f"Improvement = {improvement:.3f}"
    )

df = pd.DataFrame(
    results,
    columns=[
        "Sample",
        "Noisy_STOI",
        "Enhanced_STOI",
        "STOI_Improvement"
    ]
)

output_path = r"C:\Users\shrey\OneDrive\Desktop\ANC_Benchmark\results\stoi_206_results.csv"

df.to_csv(output_path, index=False)

print("\n✅ STOI calculation completed for 206 samples.")
print("Saved to:", output_path)