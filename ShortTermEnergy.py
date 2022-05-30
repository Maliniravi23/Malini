from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
fs, signal = wavfile.read('s_a1.wav')
signal = signal / max(abs(signal))
sampsPerMilli = int(fs / 1000)
millisPerFrame = 20
sampsPerFrame = sampsPerMilli * millisPerFrame
nFrames = int(len(signal) / sampsPerFrame)
STEs = []
for k in range(nFrames):
    startIdx = k * sampsPerFrame
    stopIdx = startIdx + sampsPerFrame
    window = np.zeros(signal.shape)
    window[startIdx:stopIdx] = 1	# rectangular window
    STE = sum((signal ** 2) * (window ** 2))
    STEs.append(STE)
plt.plot(STEs)
plt.title('Short-Time Energy')
plt.ylabel('ENERGY')
plt.xlabel('FRAME')
plt.autoscale(tight='both')