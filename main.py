import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

x, fe = sf.read('./note_guitare_lad.wav')
N = len(x)
window = np.hanning(N)
windowed_x = window * x
Nb_sinusoids = 32

X = np.fft.fft(windowed_x)
freqs = np.fft.fftfreq(N) * fe  #

index_lad = np.argmax(abs(X))
fundamental = freqs[index_lad]

# Get amplitudes at harmonics
index_harms = [index_lad * i for i in range(0, Nb_sinusoids + 1)]
harm_freqs = [freqs[i] for i in index_harms]
harmonics = [np.abs(X[i]) for i in index_harms]
phases = [np.angle(X[i]) for i in index_harms]

fig, (frams, fft) = plt.subplots(2)
fig, (harm, phas) = plt.subplots(2)

frams.plot(x)
# frams.set_xlim(0, 140000)
frams.set_title("Échantillons audios initiaux")
frams.set_xlabel("Échantillons")
frams.set_ylabel("Amplitude (normalisée à 1)")

fft.stem(np.real(X))
fft.set_xlim(0, len(X) // 2)
fft.set_yscale("log")
fft.set_title("FFT du signal")
fft.set_xlabel("Échantillons fréquentiels")
fft.set_ylabel("Amplitude")

harm.stem(harm_freqs, harmonics)
harm.set_yscale("log")
harm.set_title("Amplitude des harmoniques")
harm.set_xlabel("Fréquence (Hz)")
harm.set_ylabel("Amplitude")
phas.stem(harm_freqs, phases)
phas.set_title("Phase des harmoniques")
phas.set_xlabel("Fréquence (Hz)")
phas.set_ylabel("Amplitude")

plt.show()

# plt.subplot(2, 1, 1)
# plt.plot(freqs, np.abs(X))
# plt.subplot(2, 1, 2)
# plt.plot(np.angle(X))
# plt.show()

