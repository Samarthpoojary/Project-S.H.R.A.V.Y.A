import numpy as np
import librosa
from pystoi import stoi

def calculate_si_snr(reference, estimated, eps=1e-8):
    """
    Calculate Scale-Invariant Signal-to-Noise Ratio (SI-SNR).
    """
    reference = reference - np.mean(reference)
    estimated = estimated - np.mean(estimated)
    
    # Scale reference to match estimated signal
    ref_energy = np.sum(reference ** 2) + eps
    alpha = np.sum(reference * estimated) / ref_energy
    target_scaled = alpha * reference
    
    # Calculate noise (residual)
    noise = estimated - target_scaled
    
    # Calculate SI-SNR in dB
    val = 10 * np.log10((np.sum(target_scaled ** 2) + eps) / (np.sum(noise ** 2) + eps))
    return val

def evaluate_audio(clean_file, processed_file, sr=16000):
    """
    Evaluate processed audio against the clean reference.
    """
    clean_sig, _ = librosa.load(clean_file, sr=sr)
    processed_sig, _ = librosa.load(processed_file, sr=sr)
    
    # Ensure equal length
    min_len = min(len(clean_sig), len(processed_sig))
    clean_sig = clean_sig[:min_len]
    processed_sig = processed_sig[:min_len]
    
    # Calculate metrics
    stoi_score = stoi(clean_sig, processed_sig, sr, extended=False)
    si_snr_score = calculate_si_snr(clean_sig, processed_sig)
    
    return {
        "STOI (Target > 0.85)": round(stoi_score, 3),
        "SI-SNR (dB) (Target > 15)": round(si_snr_score, 2)
    }

if __name__ == "__main__":
    print("Initializing SIH Metrics Validation (STOI & SI-SNR)...")
    # Generate dummy arrays to test the math
    target_sr = 16000
    t = np.linspace(0, 1, target_sr)
    
    dummy_clean = np.sin(2 * np.pi * 400 * t)
    # Add slight noise to simulate a "processed" output
    dummy_processed = dummy_clean + 0.1 * np.random.randn(len(t))
    
    # Test metrics manually using the numpy arrays
    test_stoi = stoi(dummy_clean, dummy_processed, target_sr, extended=False)
    test_snr = calculate_si_snr(dummy_clean, dummy_processed)
    
    print(f"Test STOI: {test_stoi:.3f}")
    print(f"Test SI-SNR: {test_snr:.2f} dB")
    print("Metrics pipeline ready for Pair 2! You can now benchmark the audio outputs.")
