import numpy as np
import librosa
import soundfile as sf
import os

def adjust_snr(clean_sig, noise_sig, target_snr_db):
    """
    Scale noise power relative to clean speech to achieve exact target SNR.
    """
    # Match signal lengths
    if len(noise_sig) < len(clean_sig):
        # Loop noise if it is shorter than speech
        repeats = int(np.ceil(len(clean_sig) / len(noise_sig)))
        noise_sig = np.tile(noise_sig, repeats)[:len(clean_sig)]
    else:
        noise_sig = noise_sig[:len(clean_sig)]

    # Compute signal powers
    clean_power = np.mean(clean_sig ** 2)
    noise_power = np.mean(noise_sig ** 2)

    if noise_power == 0 or clean_power == 0:
        return clean_sig, noise_sig

    # Calculate required noise scaling factor
    target_noise_power = clean_power / (10 ** (target_snr_db / 10.0))
    scale_factor = np.sqrt(target_noise_power / noise_power)
    scaled_noise = noise_sig * scale_factor

    mixed_sig = clean_sig + scaled_noise

    # Normalize output to prevent digital clipping [-1.0, 1.0]
    max_val = np.max(np.abs(mixed_sig))
    if max_val > 1.0:
        mixed_sig = mixed_sig / max_val
        clean_sig = clean_sig / max_val
        scaled_noise = scaled_noise / max_val

    return mixed_sig, scaled_noise, clean_sig

def generate_synthetic_defense_noise(duration_sec=3, sr=16000, noise_type="helicopter"):
    """
    Synthesize mathematical defense noise profiles without external files.
    """
    t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
    
    if noise_type == "stationary_engine":
        # Multi-harmonic engine idle (50 Hz, 100 Hz, 200 Hz) + broad rumble
        noise = (0.5 * np.sin(2 * np.pi * 50 * t) +
                 0.3 * np.sin(2 * np.pi * 100 * t) +
                 0.2 * np.sin(2 * np.pi * 200 * t) +
                 0.15 * np.random.randn(len(t)))
        
    elif noise_type == "helicopter":
        # Low rumble with periodic blade-pass amplitude modulation (15 Hz modulation)
        carrier = 0.6 * np.sin(2 * np.pi * 120 * t) + 0.2 * np.random.randn(len(t))
        modulation = (np.sin(2 * np.pi * 15 * t) + 1.0) / 2.0
        noise = carrier * modulation
        
    elif noise_type == "impulsive_blast":
        # Ambient floor with sharp transient spikes (gunfire/artillery bursts)
        noise = 0.05 * np.random.randn(len(t))
        # Add 2 transient shockwaves with exponential decay
        for blast_time in [0.8, 2.0]:
            idx = int(blast_time * sr)
            burst_len = int(0.15 * sr)
            decay = np.exp(-np.linspace(0, 10, burst_len))
            transient = np.random.randn(burst_len) * decay * 2.5
            noise[idx : idx + burst_len] += transient

    return noise

if __name__ == "__main__":
    print("Initializing Defense Data Synthesis Lab...")
    target_sr = 16000
    os.makedirs("data/mixed_test_sets", exist_ok=True)
    
    # Generate 3 seconds of dummy speech (multi-tone harmonic to simulate voice formants)
    t = np.linspace(0, 3, 3 * target_sr, endpoint=False)
    simulated_speech = (0.4 * np.sin(2 * np.pi * 220 * t) + 
                        0.3 * np.sin(2 * np.pi * 440 * t) + 
                        0.2 * np.sin(2 * np.pi * 880 * t))
    
    # Synthesize test mixtures across 3 defense profiles at -5 dB SNR
    noise_profiles = ["stationary_engine", "helicopter", "impulsive_blast"]
    target_snr = -5.0  # Harsh battlefield condition
    
    for n_type in noise_profiles:
        noise = generate_synthetic_defense_noise(duration_sec=3, sr=target_sr, noise_type=n_type)
        mixed, scaled_noise, clean = adjust_snr(simulated_speech, noise, target_snr)
        
        sf.write(f"data/mixed_test_sets/{n_type}_noisy_{target_snr}dB.wav", mixed, target_sr)
        sf.write(f"data/mixed_test_sets/{n_type}_reference_noise.wav", scaled_noise, target_sr)
        sf.write("data/mixed_test_sets/clean_speech_ref.wav", clean, target_sr)
        print(f"Generated: {n_type} mix at {target_snr} dB SNR")
        
    print("All defense noise profiles synthesized in `data/mixed_test_sets/`.")