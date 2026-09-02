import os
import numpy as np
import soundfile as sf
import csv

# Dataset location
DATA_FOLDER = r"C:\Users\shrey\OneDrive\Desktop\ANC_Benchmark\data"

# Store results
results = []

# Process all 206 samples
for i in range(1, 207):

    folder = os.path.join(DATA_FOLDER, f"Sample_{i:03d}")

    clean_file = os.path.join(folder, "1_Clean_Voice.wav")
    noisy_file = os.path.join(folder, "2_Noisy_Input.wav")
    enhanced_file = os.path.join(folder, "3_Enhanced_Output.wav")

    # Read audio files
    clean, clean_sr = sf.read(clean_file)
    noisy, noisy_sr = sf.read(noisy_file)
    enhanced, enhanced_sr = sf.read(enhanced_file)

    # Make all signals the same length
    min_length = min(len(clean), len(noisy), len(enhanced))

    clean = clean[:min_length]
    noisy = noisy[:min_length]
    enhanced = enhanced[:min_length]

    # Calculate noise
    noisy_noise = noisy - clean
    enhanced_noise = enhanced - clean

    # Calculate Noisy SNR
    noisy_snr = 10 * np.log10(
        np.sum(clean ** 2) / np.sum(noisy_noise ** 2)
    )

    # Calculate Enhanced SNR
    enhanced_snr = 10 * np.log10(
        np.sum(clean ** 2) / np.sum(enhanced_noise ** 2)
    )

    # Calculate SNR Improvement
    snr_improvement = enhanced_snr - noisy_snr

    # Store results
    results.append([
        i,
        noisy_snr,
        enhanced_snr,
        snr_improvement
    ])

    # Display result
    print(
        f"Sample {i:03d}: "
        f"Noisy SNR = {noisy_snr:.2f} dB | "
        f"Enhanced SNR = {enhanced_snr:.2f} dB | "
        f"Improvement = {snr_improvement:.2f} dB"
    )


# Save results to CSV
output_file = os.path.join(
    DATA_FOLDER,
    "..",
    "results",
    "snr_206_results.csv"
)

with open(output_file, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "Sample",
        "Noisy_SNR_dB",
        "Enhanced_SNR_dB",
        "SNR_Improvement_dB"
    ])

    writer.writerows(results)


print("\n✅ SNR calculation completed for 206 samples.")
print(f"Results saved to: {output_file}")