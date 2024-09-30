from typing import List, Any

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import wave
import struct


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

def generateNotes(lad_freq):
    la_freq = lad_freq / 1.06

    frequencies = {
        "do": la_freq * 0.595,
        "do#": la_freq * 0.630,
        "re": la_freq * 0.667,
        "re#": la_freq * 0.707,
        "mi": la_freq * 0.749,
        "fa": la_freq * 0.794,
        "fa#": la_freq * 0.841,
        "sol": la_freq * 0.891,
        "sol#": la_freq * 0.944,
        "la": la_freq,
        "la#": lad_freq,
        "si": la_freq * 1.123
    }

    return frequencies

def synthesizeNoteAudio(fs, harmonics, note_freq, phases, enveloppe, duration=2):
    ts = np.linspace(0, duration, int(fs * duration))
    audio = []
    for t in ts:
        total = 0
        for i in range(len(harmonics)):
            total += harmonics[i] * np.sin(2 * np.pi * note_freq * i * t + phases[i])

        audio.append(total)

    new_env = enveloppe[0:len(audio)]
    new_env[-int(0.01 * fs):] = np.linspace(new_env[-int(0.01 * fs)], 0, int(0.01 * fs))
    audio = np.multiply(audio, new_env)

    return audio

def createWav(audio, sampleRate, filename):
    with wave.open(filename, "w") as wav:
        nchannels = 1
        sampwidth = 2
        nframes   = len(audio)
        wav.setparams((nchannels, sampwidth, sampleRate, nframes, "NONE", "not compressed"))

        for sample in audio:
            wav.writeframes(struct.pack('h', int(sample)*10))

def createSilence(sampleRate, duration_s = 1):
    return [0 for t in np.linspace(0, duration_s , int(sampleRate * duration_s))]

def synthetizeBeethoven(fs, harmonics, phases, enveloppe):
    solAudio = synthesizeNoteAudio(fs, harmonics, noteFrequencies["sol"], phases, enveloppe, 0.4)
    miAudio = synthesizeNoteAudio(fs, harmonics, noteFrequencies["re#"], phases, enveloppe, 1.5)
    faAudio = synthesizeNoteAudio(fs, harmonics, noteFrequencies["fa"], phases, enveloppe, 0.4)
    reAudio = synthesizeNoteAudio(fs, harmonics, noteFrequencies["re"], phases, enveloppe, 1.5)

    silence = np.zeros(int(fs*0.2))

    fullAudio = np.concatenate((solAudio, silence, solAudio, silence, solAudio, silence, miAudio, silence,
                                faAudio, silence, faAudio, silence, faAudio, silence, reAudio), axis=None)

    createWav(fullAudio, fs, "5thSymphony.wav")


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

noteFrequencies = generateNotes(fundamental)

# LaD_audio = synthesizeNoteAudio(fs, harmonics, fundamental, phases, enveloppeTemporelle)
# do_audio = synthesizeNoteAudio(fs, harmonics, noteFrequencies["do"], phases, enveloppeTemporelle)
# re_audio = synthesizeNoteAudio(fs, harmonics, noteFrequencies["re"], phases, enveloppeTemporelle)
# mi_audio = synthesizeNoteAudio(fs, harmonics, noteFrequencies["mi"], phases, enveloppeTemporelle, 1)
# fa_audio = synthesizeNoteAudio(fs, harmonics, noteFrequencies["fa"], phases, enveloppeTemporelle)
#
# createWav(LaD_audio, fs, "laD.wav")
# createWav(do_audio, fs, "do.wav")
# createWav(re_audio, fs, "re.wav")
# createWav(mi_audio, fs, "mi.wav")
# createWav(fa_audio, fs, "fa.wav")

synthetizeBeethoven(fs, harmonics, phases, enveloppeTemporelle)


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
