from typing import List, Any

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf


def applyLowpassFilter(signal_data, N):
    lowpassFilter = [1/N] * N
    return np.convolve(lowpassFilter, np.abs(signal_data))


def findNearestGainIndex(array, value):
    return (np.abs(array - value)).argmin()


def getFilterOrder(w):
    gain = np.power(10, -3/20)
    h0 = 1
    hGain = []
    for M in range(1, 1000, 1):
        sum = 0
        for i in range(0, M, 1):
            sum += np.exp(-1j*0*i)
        a = h0/np.real(sum)
        currentGain = 0
        for k in range(0, M, 1):
            currentGain += np.exp(-1j*w*k)
        hGain.append(np.abs(a*currentGain))
    N = findNearestGainIndex(hGain, gain) + 1

    print("Lowpass Filter Order: " + str(N))
    return N


x, fs = sf.read('./note_guitare_lad.wav')
N = len(x)
window = np.hanning(N)
windowed_x = window * x
nb_sinusoids = 32

X = np.fft.fft(windowed_x)
freqs = np.fft.fftfreq(N) * fs

index_lad = np.argmax(abs(X))
fundamental = freqs[index_lad]

# Get amplitudes at harmonics
index_harms = [index_lad * i for i in range(0, nb_sinusoids + 1)]
harm_freqs = [freqs[i] for i in index_harms]
harmonics = [np.abs(X[i]) for i in index_harms]
phases = [np.angle(X[i]) for i in index_harms]

filterOrder = getFilterOrder(np.pi/1000)
enveloppeTemporelle = applyLowpassFilter(x, filterOrder)


plt.plot(enveloppeTemporelle)
plt.show()


# ts = np.linspace(0, 2, int(fs * 2))
# audio = []
# for t in ts:
#     total = 0
#     for i in range(len(harmonics)):
#         total += harmonics[i] * np.sin(2*np.pi*fundamental*i*t + phases[i])
#
#     audio.append(total)



# FONCTION AFFICHER PLOTS ##########################################

# fig, (frams, fft) = plt.subplots(2)
# fig, (harm, phas) = plt.subplots(2)
#
# frams.plot(x)
# # frams.set_xlim(0, 140000)
# frams.set_title("Échantillons audios initiaux")
# frams.set_xlabel("Échantillons")
# frams.set_ylabel("Amplitude (normalisée à 1)")
#
# fft.stem(np.real(X))
# fft.set_xlim(0, len(X) // 2)
# fft.set_yscale("log")
# fft.set_title("FFT du signal")
# fft.set_xlabel("Échantillons fréquentiels")
# fft.set_ylabel("Amplitude")
#
# harm.stem(harm_freqs, harmonics)
# harm.set_yscale("log")
# harm.set_title("Amplitude des harmoniques")
# harm.set_xlabel("Fréquence (Hz)")
# harm.set_ylabel("Amplitude")
# phas.stem(harm_freqs, phases)
# phas.set_title("Phase des harmoniques")
# phas.set_xlabel("Fréquence (Hz)")
# phas.set_ylabel("Amplitude")
#
# plt.show()

# FIN FONCTION AFFICHER PLOTS ##########################################

# plt.subplot(2, 1, 1)
# plt.plot(x)
# plt.subplot(2, 1, 2)
# plt.plot(new_x)
# plt.show()
#
# sf.write('test.wav', audio, samplerate=fs)

