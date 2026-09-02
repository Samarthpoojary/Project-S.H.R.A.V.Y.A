import os
import numpy as np
import soundfile as sf
from pesq import pesq

base_path = r"C:\Users\shrey\OneDrive\Desktop\ANC_Benchmark\data"

results = []

for i in range(1, 207):

    folder = os.path.join(base_path, f"Sample_{i:03d}")

    clean_path = os.path.join(folder, "1_Clean_Voice.wav")
    noisy_path = os.path.join(folder, "2_Noisy_Input.wav")
    enhanced_path = os.path.join(folder, "3_Enhanced_Output.wav")

    clean, sr_clean = sf.read(clean_path)
    noisy, sr_noisy = sf.read(noisy_path)
    enhanced, sr_enhanced = sf.read(enhanced_path)

    # Make all signals the same length
    min_len = min(len(clean), len(noisy), len(enhanced))

    clean = clean[:min_len]
    noisy = noisy[:min_len]
    enhanced = enhanced[:min_len]

    # PESQ requires 16000 Hz for wideband mode
    noisy_pesq = pesq(16000, clean, noisy, "wb")
    enhanced_pesq = pesq(16000, clean, enhanced, "wb")

    improvement = enhanced_pesq - noisy_pesq

    results.append([
        f"Sample_{i:03d}",
        noisy_pesq,
        enhanced_pesq,
        improvement
    ])

    print(
        f"Sample_{i:03d}: "
        f"Noisy PESQ = {noisy_pesq:.2f} | "
        f"Enhanced PESQ = {enhanced_pesq:.2f} | "
        f"Improvement = {improvement:.2f}"
    )

# Save results
import pandas as pd

df = pd.DataFrame(
    results,
    columns=[
        "Sample",
        "Noisy_PESQ",
        "Enhanced_PESQ",
        "PESQ_Improvement"
    ]
)

output_path = r"C:\Users\shrey\OneDrive\Desktop\ANC_Benchmark\results\pesq_206_results.csv"

df.to_csv(output_path, index=False)

print("\n✅ PESQ calculation completed for 206 samples.")
print("Saved to:", output_path)