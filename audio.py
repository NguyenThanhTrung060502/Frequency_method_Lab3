import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq
import time
import os 
import librosa
import soundfile as sf

def plot_func(X, func, color, caption, title, legend=['Restored function']):
    ymin = min(func) 
    ymax = max(func) 
    ymax = ymax + 0.1 * (ymax - ymin)
    ymin = ymin - 0.1 * (ymax - ymin)
    plt.figure(figsize=(8, 5)) 
    plt.ylim(ymin, ymax)
    plt.plot(X, func.real, color=f'{color}')
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.legend(legend, loc='upper right')
    # add caption
    plt.title(caption)
    plt.grid(color='black', linestyle='--', linewidth=0.25)
    if title != '':
        plt.savefig(title + '.png')
    

def plot_image(X, func, caption, title):
    ymin = min(func.real.min(), func.imag.min())
    ymax = max(func.real.max(), func.imag.max())
    ymax = ymax + 0.1 * (ymax - ymin)
    ymin = ymin - 0.1 * (ymax - ymin)
    plt.figure(figsize=(8, 5)) 
    plt.ylim(ymin, ymax)
    plt.plot(X, func.real, color='green')
    plt.plot(X, func.imag, color='red')
    plt.xlabel('\u03C9')
    plt.ylabel('f(\u03C9)')
    plt.legend(['Real', 'Imag'], loc='upper right')
    # add caption
    plt.title(caption)
    plt.grid(color='black', linestyle='--', linewidth=0.25)
    if title != '':
        plt.savefig(title + '.png')


def cmp_func(X, func1, func2, caption, title, legend=['Source function', 'Restored function']):
    ymin = min(min(func1), min(func2))
    ymax = max(max(func1), max(func2)) 

    ymax = ymax + 0.1 * (ymax - ymin)
    ymin = ymin - 0.1 * (ymax - ymin)

    plt.figure(figsize=(8, 5)) 

    plt.ylim(ymin, ymax)
    plt.plot(X, func1.real, color='black')
    plt.plot(X, func2.real, color='tomato')
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.legend(legend, loc='upper right')
    # add caption
    plt.title(caption)
    plt.grid(color='black', linestyle='--', linewidth=0.25)
    if title != '':
        plt.savefig(title + '.png')


def dot_product(X, f, g):
    dx = X[1] - X[0]
    return np.dot(f, g) * dx


get_fourier_image = lambda X, V, func: np.array([1 / (np.sqrt(2 * np.pi)) * dot_product(X, func, (lambda t: np.e ** (-1j * 2 * np.pi * image_clip * t))(X)) for image_clip in V])
get_fourier_function = lambda X, V, image: np.array([1 / (np.sqrt(2 * np.pi)) * dot_product(V, image, (lambda t: np.e ** (1j * 2 * np.pi * x * t))(V)) for x in X])


clip_delta_image_values = lambda pivot, delta: lambda X, func: np.array([0 if (pivot - delta <= X[i] <= pivot + delta  or pivot - delta <= -X[i] <= pivot + delta)else func[i] for i in range(len(func))])


def audio_fm(n):
    if not os.path.exists(f'./audio/{n}'):
        os.makedirs(f'./audio/{n}')
    # read audio file
    samples, sr = librosa.load('MUHA.wav')
    samples = librosa.resample(samples, orig_sr=sr, target_sr=10000)
    sr = 10000 

    # plot audio file
    # plot_func(np.linspace(0, len(samples) / sr, len(samples)), samples, 'black', f'Аудио файлы', f'./audio/{n}/Source waveform')

    # get image 
    image = fft(samples)
    V = fftfreq(len(samples), 1 / sr)
    # plot_image(V, image, 'Фурье-образ', f'./audio/{n}/Fourier image')

    clipped_image = clip_delta_image_values(pivot=200, delta=200)(V, image)

    # plot_image(V, clipped_image, 'Обрезанный образ', f'./audio/{n}/Clipped image')

    restored = ifft(clipped_image).real


    # plot restored audio file
    plot_func(np.linspace(0, len(samples) / sr, len(samples)), restored, 'seagreen', 'Восстановленная функция', f'./audio/{n}/Restored audio file')

    sf.write('TrungHip.wav', restored, sr, subtype='PCM_24')

audio_fm(1)
# audio_fm(2)

# plt.show()