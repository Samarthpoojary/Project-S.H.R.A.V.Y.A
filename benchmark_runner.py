import os
import time
import soundfile as sf
import warnings
# Ignore librosa/numba warnings for cleaner terminal output
warnings.filterwarnings('ignore')

# Import the modules we just built
from spectral_filter import spectral_subtraction
from metrics_calc import evaluate_audio

def run_evaluation_suite():
    print("🚀 Initiating Project SHRAVYA Benchmark Pipeline...\n")
    
    data_dir = "data/mixed_test_sets"
    clean_ref = os.path.join(data_dir, "clean_speech_ref.wav")
    
    # Ensure the clean reference exists
    if not os.path.exists(clean_ref):
        print(f"Error: Clean reference not found at {clean_ref}")
        return

    # Look for all the noisy synthesized files
    noise_profiles = ["stationary_engine", "helicopter", "impulsive_blast"]
    target_snr = -5.0
    
    # Table Header for the terminal
    print(f"{'Noise Profile':<20} | {'Algorithm':<15} | {'STOI (>0.85)':<12} | {'SI-SNR (dB)':<12} | {'Latency (ms)':<12}")
    print("-" * 80)
    
    for n_type in noise_profiles:
        noisy_file = os.path.join(data_dir, f"{n_type}_noisy_{target_snr}dB.wav")
        enhanced_file = os.path.join(data_dir, f"{n_type}_enhanced_spectral.wav")
        
        if not os.path.exists(noisy_file):
            continue
            
        # 1. Evaluate Raw Noisy Baseline
        raw_metrics = evaluate_audio(clean_ref, noisy_file)
        print(f"{n_type:<20} | {'Raw Noisy':<15} | {raw_metrics['STOI (Target > 0.85)']:<12.3f} | {raw_metrics['SI-SNR (dB) (Target > 15)']:<12.2f} | {'-':<12}")
        
        # 2. Run Classical DSP (Spectral Subtraction) & Measure Latency
        noisy_sig, sr = sf.read(noisy_file)
        
        start_time = time.time()
        enhanced_sig = spectral_subtraction(noisy_sig, sr=sr)
        process_time_ms = (time.time() - start_time) * 1000
        
        # Save enhanced audio to disk
        sf.write(enhanced_file, enhanced_sig, sr)
        
        # 3. Evaluate Enhanced Output
        enhanced_metrics = evaluate_audio(clean_ref, enhanced_file)
        
        # Calculate per-frame latency assuming a 16ms frame size
        num_frames = len(noisy_sig) / (sr * 0.016)
        latency_per_frame = process_time_ms / num_frames if num_frames > 0 else 0
        
        print(f"{'':<20} | {'Spectral Filter':<15} | {enhanced_metrics['STOI (Target > 0.85)']:<12.3f} | {enhanced_metrics['SI-SNR (dB) (Target > 15)']:<12.2f} | {latency_per_frame:<12.2f}")
        print("-" * 80)

if __name__ == "__main__":
    run_evaluation_suite()
    print("\n✅ Benchmark Complete. The metrics above can be directly copied to your SIH Slide Deck.")