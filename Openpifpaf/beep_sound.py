import numpy as np
from scipy.io.wavfile import write

duration = 0.2  # seconds
freq = 440.0  # A4 note
sample_rate = 44100

t = np.linspace(0., duration, int(sample_rate * duration))
waveform = 0.5 * np.sin(2. * np.pi * freq * t)

# Convert to 16-bit PCM format
waveform_int16 = np.int16(waveform * 32767)
write("static/beep.wav", sample_rate, waveform_int16)
