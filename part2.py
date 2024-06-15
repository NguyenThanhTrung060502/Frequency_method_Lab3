import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq
import time
import os 
import librosa
import soundfile as sf

# Vẽ hàm số 
def plot_func(t, func, title, legend=['Исходный сигнал']):
    ymin = min(func) 
    ymax = max(func) 

    ymax = ymax + 0.1 * (ymax - ymin)
    ymin = ymin - 0.1 * (ymax - ymin)

    plt.figure(figsize=(8, 5)) 

    plt.ylim(ymin, ymax)
    plt.plot(t, func.real)
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.legend(legend, loc='upper right')
    plt.title(title)
    plt.grid()


# So sánh 2 hàm số với nhau 
def cmp_func(t, func1, func2, title, legend=['Исходный сигнал', 'Восстановленный сигнал']):
    ymin = min(min(func1), min(func2))
    ymax = max(max(func1), max(func2)) 

    ymax = ymax + 0.1 * (ymax - ymin)
    ymin = ymin - 0.1 * (ymax - ymin)

    plt.figure(figsize=(8, 5)) 

    plt.ylim(ymin, ymax)
    plt.plot(t, func1.real)
    plt.plot(t, func2.real)
    plt.xlabel('t')
    plt.ylabel('f(t)')
    plt.legend(legend, loc='upper right')
    plt.title(title)
    plt.grid()


# Vẽ Фурье-образ
def plot_image(t, func, title):
    ymin = min(func.real.min(), func.imag.min())
    ymax = max(func.real.max(), func.imag.max())
    ymax = ymax + 0.1 * (ymax - ymin)
    ymin = ymin - 0.1 * (ymax - ymin)

    plt.figure(figsize=(8, 5)) 

    plt.ylim(ymin, ymax)
    plt.plot(t, func.real)
    plt.plot(t, func.imag)
    plt.xlabel('\u03C9')
    plt.ylabel('f(\u03C9)')
    plt.legend(['Real', 'Imag'], loc='upper right')
    plt.title(title)
    plt.grid()


# Tích vô hướng của 2 hàm số 
def dot_product(t, f, g):
    dt = t[1] - t[0]
    return np.dot(f, g) * dt


get_fourier_image = lambda X, V, func: np.array([1 / (np.sqrt(2 * np.pi)) * dot_product(X, func, (lambda t: np.e ** (-1j * 2 * np.pi * image_clip * t))(X)) for image_clip in V])
get_fourier_function = lambda X, V, image: np.array([1 / (np.sqrt(2 * np.pi)) * dot_product(V, image, (lambda t: np.e ** (1j * 2 * np.pi * x * t))(V)) for x in X])


clip_delta_image_values = lambda pivot, delta: lambda X, func: np.array([0 if (pivot - delta <= X[i] <= pivot + delta  or pivot - delta <= -X[i] <= pivot + delta)else func[i] for i in range(len(func))])



# Đọc file âm thanh 
samples, sr = librosa.load('MUHA.wav')
# samples = librosa.resample(samples, orig_sr=sr, target_sr=10000)
# sr = 5000 

# Vẽ đồ thị của sóng âm 
plot_func(np.linspace(0, len(samples) / sr, len(samples)), samples, title='Audio file')

# Tìm Fourier image 
image = fft(samples)
V = fftfreq(len(samples), 1 / sr)
plot_image(V, image, title='Fourier image')

clipped_image = clip_delta_image_values(pivot=150, delta=150)(V, image)

plot_image(V, clipped_image, title='Clipped image')

restored = ifft(clipped_image).real


# Vẽ Fourier image của hàm được khôi phục 
plot_func(np.linspace(0, len(samples) / sr, len(samples)), restored, title='Restored audio file')

sf.write('HA.wav', restored, sr, subtype='PCM_24')

plt.show()