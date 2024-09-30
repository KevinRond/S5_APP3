import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import wave
import struct


def getSignalParameters(soundFilePath):
    x, fs = sf.read(soundFilePath)
    N = len(x)
    nb_sinusoids = 32

    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(N) * fs

    index_lad = np.argmax(abs(X))
    fundamental = freqs[index_lad]

    index_harms = [index_lad * i for i in range(0, nb_sinusoids + 1)]

    harm_freqs = [freqs[i] for i in index_harms]
    harmonics = [np.abs(X[i]) for i in index_harms]
    phases = [np.angle(X[i]) for i in index_harms]
    return x, fs, fundamental, harm_freqs, harmonics, phases

def applyLowpassFilter(signal_data, N):
    lowpassFilter = [1/N] * N
    window = np.hanning(N)
    lowpassFilter = lowpassFilter * window
    return np.convolve(lowpassFilter, np.abs(signal_data))

def findNearestGainIndex(array, value):
    return (np.abs(array - value)).argmin()

def getFilterOrder(w):
    gain = np.power(10, -3/20)
    hGain = []
    for N in range(1, 1000, 1):
        sum = 0
        for n in range(N):
            sum += np.exp(-1j*w*n)
        hGain.append(np.abs(sum) * 1/N)

    N = findNearestGainIndex(hGain, gain)

    print("Ordre du filtre: ", N)
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
    audio = np.int16(audio / np.max(np.abs(audio)) * 32767)  # normalize to 16-bit PCM


    with wave.open(filename, "w") as wav:
        nchannels = 1
        sampwidth = 2
        nframes   = len(audio)
        wav.setparams((nchannels, sampwidth, sampleRate, nframes, "NONE", "not compressed"))

        for sample in audio:
            wav.writeframes(struct.pack('h', sample))

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

def passe_bas(n, N, K):
    hn = np.sin(np.pi*n*K/N)/(N*np.sin(np.pi*n/N))
    return hn

def coupe_bande(n, N, K, w0):
    cb = 1-2*K/N
    if n!=0:
        cb = -2*passe_bas(n, N, K)*np.cos(w0*n)
    return cb

def filter1kWave():
    audio, rate = sf.read("./note_basson_plus_sinus_1000_hz.wav")

    N = 6000
    fc = 1000
    delta = 40
    fmax = 22050
    fe = 2 * fmax

    w0 = 2 * np.pi * fc / fe
    w1 = 2 * np.pi * delta / fe
    K = (w1 * N / np.pi) + 1

    index = np.linspace(-(N / 2) + 1, (N / 2), N)
    filtre = [coupe_bande(n, N, K, w0) for n in index]
    window = np.hanning(N)
    filtre = filtre * window

    audio = np.convolve(filtre, audio)

    createWav(audio, rate, "basson_filtre.wav")


filter1kWave()

signal, fs, fundamental, harmonicFrequencies, harmonics, phases = getSignalParameters('./note_guitare_lad.wav')

filterOrder = getFilterOrder(np.pi/1000)
enveloppeTemporelle = applyLowpassFilter(signal, filterOrder)

noteFrequencies = generateNotes(fundamental)

synthetizeBeethoven(fs, harmonics, phases, enveloppeTemporelle)

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
