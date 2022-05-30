import librosa as lbr
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
x , fs = lbr.load('a.wav')
sample = 8000
a = lbr.lpc(x,12)
x_padding = np.pad(a,(12,8000),'constant')
print(len(x_padding))
y = np.abs(np.fft.fft(x,sample))
lpc = np.fft.fft(x_padding,sample)
lpc_rep = np.abs(np.reciprocal(lpc))
half = fs/2
plt.plot(np.arange(4000)/4000*half,np.log(np.abs(y[:4000])),label='signal')
plt.plot(np.arange(4000)/4000*half,np.log(np.abs(lpc_rep[:4000])),label='lpc spectrum')
plt.xlabel('Frequency')
plt.ylabel('log magnitude')
plt.legend()
plt.show()