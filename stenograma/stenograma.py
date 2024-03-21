import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

# Step 3: Load Audio File
sample_rate, samples = wavfile.read("Baldurs.wav")

# Ensure the samples are mono (if it's stereo)
if len(samples.shape) > 1:
    samples = samples.mean(axis=1)

# Step 4: Compute Spectrogram
plt.figure(figsize=(8, 6))
plt.specgram(samples, Fs=sample_rate)
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Spectrogram')
plt.colorbar(label='Intensity (dB)')
plt.ylim(0, sample_rate // 2)  # Limit y-axis to the Nyquist frequency
plt.savefig('spectrogram.png')

# Show or save the plot
plt.show()
