import os
import time
import csv
import numpy as np
import librosa
import tensorflow as tf

BASE_DIR = r"C:\Users\shrey\OneDrive\Desktop\ANC_Benchmark"

model_path = os.path.join(
    BASE_DIR, "scripts", "tinyml_gate_int8.lite"
)

# Load TinyML model
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

frame_size = 3168
sample_rate = 16000

processing_times = []
total_frames = 0

# Test all 206 samples
for i in range(1, 207):

    input_path = os.path.join(
        BASE_DIR,
        "data",
        f"Sample_{i:03d}",
        "2_Noisy_Input.wav"
    )

    audio, sr = librosa.load(input_path, sr=sample_rate)

    for j in range(0, len(audio), frame_size):

        frame = audio[j:j + frame_size]

        if len(frame) < frame_size:
            frame = np.pad(
                frame,
                (0, frame_size - len(frame))
            )

        start = time.perf_counter()

        # TinyML processing
        input_data = np.array([frame], dtype=np.int8)

        if len(input_details[0]["shape"]) == 3:
            input_data = np.expand_dims(input_data, axis=-1)

        interpreter.set_tensor(
            input_details[0]["index"],
            input_data
        )

        interpreter.invoke()

        interpreter.get_tensor(
            output_details[0]["index"]
        )

        end = time.perf_counter()

        processing_times.append(
            (end - start) * 1000
        )

        total_frames += 1


# 3168 samples at 16 kHz = 198 ms
frame_duration_ms = (frame_size / sample_rate) * 1000

average_latency = np.mean(processing_times)
maximum_latency = np.max(processing_times)
real_time_factor = average_latency / frame_duration_ms

print("\n===== LATENCY BENCHMARK =====")

print("Number of samples:", 206)
print("Total frames:", total_frames)
print(
    "Frame duration:",
    round(frame_duration_ms, 3),
    "ms"
)

print(
    "Average processing latency:",
    round(average_latency, 3),
    "ms"
)

print(
    "Maximum processing latency:",
    round(maximum_latency, 3),
    "ms"
)

print(
    "Real-time factor:",
    round(real_time_factor, 4)
)

if average_latency < frame_duration_ms:
    print("\n✅ TinyML processing is faster than real-time")
else:
    print("\n⚠️ TinyML processing is slower than real-time")


# Save latency results
output_path = os.path.join(
    BASE_DIR,
    "results",
    "latency_206_results.csv"
)

with open(output_path, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "Metric",
        "Value",
        "Unit",
        "Note"
    ])

    writer.writerow([
        "Average Latency",
        round(average_latency, 3),
        "ms",
        "TinyML inference only"
    ])

    writer.writerow([
        "Maximum Latency",
        round(maximum_latency, 3),
        "ms",
        "TinyML inference only"
    ])

    writer.writerow([
        "Frame Duration",
        round(frame_duration_ms, 3),
        "ms",
        "3168 samples at 16 kHz"
    ])

    writer.writerow([
        "Real-Time Factor",
        round(real_time_factor, 4),
        "ratio",
        "TinyML inference only"
    ])

    writer.writerow([
        "Samples",
        206,
        "count",
        "206-sample evaluation set"
    ])

    writer.writerow([
        "Total Frames",
        total_frames,
        "count",
        "206-sample evaluation set"
    ])

print("\nLatency results saved to:")
print(output_path)