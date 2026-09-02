import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

csv_path = os.path.join(BASE_DIR, "data", "all audio.csv")
output_path = os.path.join(BASE_DIR, "data", "dataset_pairs.csv")

# Load original CSV
df = pd.read_csv(csv_path)

# Keep only the columns we need
pairs = df[[
    "mixed_filename",
    "clean_source",
    "noise_source",
    "snr_db"
]].copy()

# Add full paths
pairs["noisy_path"] = pairs["mixed_filename"].apply(
    lambda x: os.path.join("data", "noisy", x)
)

pairs["clean_path"] = pairs["clean_source"].apply(
    lambda x: os.path.join("data", "clean", x)
)

# Save
pairs.to_csv(output_path, index=False)

print("✅ Dataset preparation complete!")
print("Total audio pairs:", len(pairs))
print("Saved to:", output_path)