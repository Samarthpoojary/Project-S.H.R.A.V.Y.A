import numpy as np
import librosa
import soundfile as sf

def apply_nlms(primary_sig, reference_sig, filter_length=256, mu=0.1, eps=1e-8):
    """
    Normalized Least Mean Squares (NLMS) Adaptive Filter.
    
    Parameters:
    primary_sig: 1D array, the noisy speech signal (speech + noise)
    reference_sig: 1D array, the reference noise signal (mostly noise)
    filter_length: int, number of filter taps (higher = better resolution but more latency)
    mu: float, step size (adaptation rate, 0 < mu < 2)
    eps: float, small constant to prevent division by zero
    
    Returns:
    enhanced_speech: 1D array, the denoised speech signal
    """
    # Ensure signals are the same length
    num_samples = min(len(primary_sig), len(reference_sig))
    
    # Initialize weights and output array
    weights = np.zeros(filter_length)
    enhanced_speech = np.zeros(num_samples)
    
    # Adaptive filtering loop
    for n in range(filter_length, num_samples):
        # Extract the current block of reference noise (reversed for convolution)
        x_block = reference_sig[n - filter_length : n][::-1]
        
        # Calculate filter output (the estimated noise in the primary channel)
        estimated_noise = np.dot(weights, x_block)
        
        # Calculate the error (this is our target enhanced speech)
        error = primary_sig[n] - estimated_noise
        enhanced_speech[n] = error
        
        # Compute normalization factor (power of the reference block)
        power = np.dot(x_block, x_block) + eps
        
        # Update filter weights dynamically
        weights = weights + (mu / power) * x_block * error
        
    return enhanced_speech

if __name__ == "__main__":
    print("Initializing NLMS DSP baseline...")
    
    # Generate 3 seconds of dummy audio at 16kHz
    target_sr = 16000
    t = np.linspace(0, 3, 3 * target_sr, endpoint=False)
    
    # Simulate 'speech' (a 400 Hz tone) and 'stationary noise' (a 1500 Hz tone)
    clean_speech = 0.5 * np.sin(2 * np.pi * 400 * t)
    noise = 0.5 * np.sin(2 * np.pi * 1500 * t)
    
    # Primary channel: Noisy speech (Speech + Noise)
    primary = clean_speech + noise
    
    # Reference channel: Isolated noise source
    reference = noise
    
    print("Processing audio through the NLMS adaptive filter...")
    enhanced = apply_nlms(primary, reference, filter_length=256, mu=0.1)
    
    # Export the files to your local directory for playback
    sf.write("dummy_noisy_input.wav", primary, target_sr)
    sf.write("dummy_enhanced_output.wav", enhanced, target_sr)
    
    print("Success! Check your folder for 'dummy_noisy_input.wav' and 'dummy_enhanced_output.wav'.")
    # Quick test harness for your local machine
    print("Initializing NLMS DSP baseline...")
    # Example usage (Uncomment and add real file paths once Shriya provides them)
    
    # target_sr = 16000
    # primary, _ = librosa.load("data/mixed_test_sets/noisy_speech.wav", sr=target_sr)
    # reference, _ = librosa.load("data/noise_profiles/stationary_engine.wav", sr=target_sr)
    
    # enhanced = apply_nlms(primary, reference, filter_length=256, mu=0.1)
    
    # sf.write("data/mixed_test_sets/nlms_enhanced_output.wav", enhanced, target_sr)
    # print("Filtering complete. Ready for Pair 1 ML fusion.")