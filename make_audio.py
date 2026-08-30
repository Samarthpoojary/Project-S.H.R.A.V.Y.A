import numpy as np
import soundfile as sf

# Generate a 3-second simple beep (sine wave)
samplerate = 16000
t = np.linspace(0., 3., samplerate * 3)
data = 0.5 * np.sin(2. * np.pi * 440. * t) # 440 Hz tone

sf.write('dummy_audio.wav', data, samplerate)
print("✅ dummy_audio.wav created successfully!")