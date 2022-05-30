from scipy.io import wavfile
fs, signal = wavfile.read('s_a1.wav')
print(fs,'Hz') print(len(signal),'samples')
signal = signal / max(abs(signal))
print(signal)
DC = np.mean(signal)
newSignal = signal - DC
print(DC,newSignal)
sampsPerMilli = int(fs / 1000)
millisPerFrame =10
sampsPerFrame = sampsPerMilli * millisPerFrame
nFrames = int(len(signal) / sampsPerFrame)
print(sampsPerMilli)
print(sampsPerFrame)
print(nFrames)
ZCCs = []
for i in range(nFrames):
    startIdx = i * sampsPerFrame
    stopIdx = startIdx + sampsPerFrame
    frame = newSignal[startIdx:stopIdx]
    ZCC = 0
    for k in range(1, len(frame)):
        ZCC += 0.5 * abs(np.sign(frame[k]) - np.sign(frame[k - 1]))
        ZCCs.append(ZCC)
plt.plot(ZCCs)
plt.title('Zero Crossing Rate')
plt.ylabel('ZCC') plt.xlabel('FRAME')
framerate = 16000

wave_data = np.fromstring(str_data, dtype=np.short)
time3 = np.arange(0, len(STEs)) * (len(wave_data)/len(STEs) / framerate)
time4 = np.arange(0, len(ZCCs)) * (len(wave_data)/len(ZCCs) / framerate)
time5 = np.arange(0, len(signal)) * (len(wave_data)/len(signal) / framerate)
plt.plot(time3,STEs,'r',label='ste')
plt.plot(time2,zcr,'y',label='zcr')
plt.plot(time5,signal,'b',label='wave file')
plt.autoscale(tight='both')
plt.legend()
plt.show()