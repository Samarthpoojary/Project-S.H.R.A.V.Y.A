import numpy as np
import librosa
import soundfile as sf

def spectral_subtraction(noisy_sig, sr=16000, noise_duration_sec=0.5):
    """
    Single-channel STFT Spectral Subtraction.
    Assumes the first `noise_duration_sec` of the audio contains only background noise.
    """
    # Convert audio to frequency domain (STFT)
    n_fft = 512
    hop_length = 256
    S = librosa.stft(noisy_sig, n_fft=n_fft, hop_length=hop_length)
    
    magnitude = np.abs(S)
    phase = np.angle(S)
    
    # Calculate how many frames represent the initial noise profile
    noise_frames = int((noise_duration_sec * sr) / hop_length)
    if noise_frames == 0 or noise_frames > magnitude.shape[1]:
        noise_frames = min(10, magnitude.shape[1])
        
    # Estimate the stationary noise profile from the first few frames
    noise_profile = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
    
    # Subtract noise profile, flooring to avoid negative spectral power artifacts (musical noise)
    enhanced_magnitude = np.maximum(magnitude - (1.5 * noise_profile), 0.05 * noise_profile)
    
    # Reconstruct the audio (Inverse STFT)
    enhanced_S = enhanced_magnitude * np.exp(1j * phase)
    enhanced_sig = librosa.istft(enhanced_S, hop_length=hop_length)
    
    return enhanced_sig

if __name__ == "__main__":
    print("Testing Single-Channel Spectral Subtraction...")
    target_sr = 16000
    t = np.linspace(0, 3, 3 * target_sr, endpoint=False)
    
    # Simulate 0.5s of pure noise, then 2.5s of speech + noise
    clean_speech = np.where(t > 0.5, 0.5 * np.sin(2 * np.pi * 400 * t), 0)
    noise = 0.3 * np.sin(2 * np.pi * 1500 * t) + 0.1 * np.random.randn(len(t))
    
    primary = clean_speech + noise
    
    enhanced = spectral_subtraction(primary, sr=target_sr, noise_duration_sec=0.5)
    
    sf.write("dummy_spectral_input.wav", primary, target_sr)
    sf.write("dummy_spectral_output.wav", enhanced, target_sr)
    
    print("Success! Check your folder for 'dummy_spectral_input.wav' and 'dummy_spectral_output.wav'.")