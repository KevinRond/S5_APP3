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

    plotSignal(x, X, harmonics, harm_freqs, phases)
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

def synthesizeAllNotesAudio(fs, harmonics, frequencies, phases, enveloppe):
    for frequency in frequencies:
        audio = synthesizeNoteAudio(fs, harmonics, frequencies[frequency], phases, enveloppe)
        createWav(audio, fs, f"{frequency}.wav")


def synthesizeNoteAudio(fs, harmonics, note_freq, phases, enveloppe, duration=2):
    ts = np.linspace(0, duration, int(fs * duration))
    audio = []
    for t in ts:
        total = 0
        for i in range(len(harmonics)):
            total += harmonics[i] * np.sin(2 * np.pi * note_freq * i * t + phases[i])

        audio.append(total)

    new_env = enveloppe[0:len(audio)]
    # new_env[-int(0.01 * fs):] = np.linspace(new_env[-int(0.01 * fs)], 0, int(0.01 * fs))

    audio = np.multiply(audio, new_env)

    return audio

def createWav(audio, sampleRate, filename):
    audio = np.int16(audio / np.max(np.abs(audio)) * 32767)  # normalize to 16-bit PCM


    with wave.open(filename, "w") as wav:
        nchannels = 1
        sampwidth = 2
        nframes = len(audio)
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

def plotSignal(x, X, harmonics, harmonic_frequencies, phases):
    fig, ((signalPlt, fftPlt), (harmAmpPlt, harmPhasesPlt)) = plt.subplots(2, 2, figsize=(16, 8))

    signalPlt.plot(x)
    signalPlt.set_title("Signal initial")
    signalPlt.set_xlabel("Échantillons")
    signalPlt.set_ylabel("Amplitude")

    fftPlt.stem(np.abs(X))
    fftPlt.set_title("FFT du signal initial")
    fftPlt.set_xlabel("Fréquence")
    fftPlt.set_xlim(0, len(X) // 2)
    fftPlt.set_ylabel("Amplitude")
    fftPlt.set_yscale("log")

    harmAmpPlt.stem(harmonic_frequencies, harmonics)
    harmAmpPlt.set_title("Amplitude des harmoniques")
    harmAmpPlt.set_xlabel("Fréquence")
    harmAmpPlt.set_ylabel("Amplitude")

    harmPhasesPlt.stem(harmonic_frequencies, phases)
    harmPhasesPlt.set_title("Phase des harmoniques")
    harmPhasesPlt.set_xlabel("Fréquence")
    harmPhasesPlt.set_ylabel("Phase")

    plt.tight_layout(pad=1.0)

    fig, ax = plt.subplots(figsize=(10, 8))

    tableContent = []
    for i in range(len(harmonics)):
        tableContent.append([harmonic_frequencies[i], harmonics[i], phases[i]])
    table = ax.table(cellText=tableContent, colLabels=["Fréquence (Hz)", "Amplitude", "Phase"], loc="center")
    ax.axis('off')
    table.get_celld()[(0, 0)].set_text_props(weight="bold")
    table.get_celld()[(0, 1)].set_text_props(weight="bold")
    table.get_celld()[(0, 2)].set_text_props(weight="bold")

    plt.show()

filter1kWave()

signal, fs, fundamental, harmonicFrequencies, harmonics, phases = getSignalParameters('./note_guitare_lad.wav')

filterOrder = getFilterOrder(np.pi/1000)
enveloppeTemporelle = applyLowpassFilter(signal, filterOrder)

noteFrequencies = generateNotes(fundamental)

# synthetizeBeethoven(fs, harmonics, phases, enveloppeTemporelle)
# synthesizeAllNotesAudio(fs, harmonics, noteFrequencies, phases, enveloppeTemporelle)
