import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft


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


# Hàm ban đầu và tín hiệu nhiễu 
function = lambda a, t1, t2: np.vectorize(lambda t: a if t1 <= t <= t2 else 0, otypes=[complex])
noised_signal = lambda t, func, b, c, d: func + b * (np.random.rand(t.size) - 0.5) + c * np.sin(d * t)


# Tìm Fourier image và Function source 
get_fourier_image = lambda t, V, func: np.array([1 / (np.sqrt(2 * np.pi)) * dot_product(t, func, (lambda t: np.e ** (-1j * image_clip * t))(t)) for image_clip in V])
get_fourier_function = lambda t, V, image: np.array([1 / (np.sqrt(2 * np.pi)) * dot_product(V, image, (lambda t: np.e ** (1j * T * t))(V)) for T in t])


# Các cắt ......
clip_outer_image_values = lambda image_clip:   lambda t, func: np.array([func[i] if abs(t[i]) <= image_clip else 0 for i in range(len(func))])
clip_inner_image_values = lambda image_clip:   lambda t, func: np.array([func[i] if abs(t[i]) >= image_clip else 0 for i in range(len(func))])
clip_delta_image_values = lambda pivot, delta: lambda t, func: np.array([0 if (pivot - delta  <= t[i] <= pivot + delta  or pivot - delta <= -t[i] <= pivot + delta) else func[i] for i in range(len(func))])



def results(a, b, c, d, t1, t2, T, image_limits, clip_function=clip_outer_image_values(image_clip=5)):

    t = np.linspace(-T/2, T/2, 1000)     # get t values array
    g = function(a, t1, t2)(t)           # get source function g(t) 

    noised_wave = noised_signal(t, g, b=b, c=c, d=d) # get noised function G(t) 

    # Vẽ hàm ban đầu và hàm nhiễu 
    plot_func(t, g, title='Исходный сигнал', legend=["g(t)"])                  # source function g(t)  
    plot_func(t, noised_wave, title='Зашумлённый сигнал',  legend=["u(t)"])    # noised function u(t)

    v = np.linspace(-image_limits, image_limits, 1000)
    wave_image = get_fourier_image(t, v, g)
    noised_signal_image = get_fourier_image(t, v, noised_wave)
    
    # Vẽ Фурье-образ của hàm ban đầu và hàm nhiễu
    plot_image(v, wave_image, title='Фурье-образ исходного сигнала')                 # source function image
    plot_image(v, noised_signal_image, title='Фурье-образ зашумлённого сигнала')     # noised function image
    
    # Vẽ Фурье-образ của hàm nhiễu bị cắt
    noised_signal_image_clipped = clip_function(v, noised_signal_image) 
    # plot_image(v, noised_signal_image_clipped, title='Фурье-образа зашумлённого обрезанного сигнала')   # noised function image clipped

    # Vẽ hàm khôi phục từ Фурье-образ của hàm ban đầu 
    noised_signal_restored = get_fourier_function(t, v, noised_signal_image)
    # plot_func(t, noised_signal_restored, title='Восстановленный сигнал', legend=["Восстановленный"])    # restored function

    # Vẽ hàm khôi phục từ Фурье-образ của hàm nhiễu bị cắt 
    noised_signal_clipped_restored = get_fourier_function(t, v, noised_signal_image_clipped)
    # plot_func(t, noised_signal_clipped_restored, title='Восстановленный обрезанного сигнал', legend=["Восстановленный"])            # restored clipped function

    
    noised_signal_clipped_restored_image = get_fourier_image(t, v, noised_signal_clipped_restored)
    # plot_image(v, noised_signal_clipped_restored_image, title='Фурье-образа фильтрованного обрезанного сигнал', )                   # restored clipped function image

    # Модуль Фурье-образа 
    # plot_image(v, np.abs(wave_image), title='Модуль Фурье-образа исходного сигнала')                                                # source function image abs
    # plot_image(v, np.abs(noised_signal_clipped_restored_image), title='Модуль фильтрованного Фурье-образа обрезанного сигнала')     # restored clipped function image abs

    # Source and restored clipped function comparison
    # cmp_func(t, g, noised_signal_clipped_restored, title='Сравнительные графики исходного и фильтрованного обрезанного сигналов', legend=["Исходный", "Фильтрованный"]) 



# results(a=3, b=1,   c=0,   d=8, t1=-2, t2=4, T=10, image_limits=20, clip_function=clip_outer_image_values(image_clip=10))
# results(a=3, b=1,   c=0,   d=8, t1=-2, t2=4, T=10, image_limits=20, clip_function=clip_outer_image_values(image_clip=5))
# results(a=3, b=1,   c=0,   d=8, t1=-2, t2=4, T=10, image_limits=20, clip_function=clip_outer_image_values(image_clip=2))
# results(a=3, b=1,   c=0,   d=8, t1=-2, t2=4, T=10, image_limits=20, clip_function=clip_outer_image_values(image_clip=20))
# results(a=3, b=0.5, c=0,   d=8, t1=-2, t2=4, T=10, image_limits=20, clip_function=clip_outer_image_values(image_clip=10))
# results(a=3, b=2,   c=0,   d=8, t1=-2, t2=4, T=10, image_limits=20, clip_function=clip_outer_image_values(image_clip=10))
# results(a=3, b=0.2, c=0,   d=8, t1=-2, t2=4, T=10, image_limits=20, clip_function=clip_outer_image_values(image_clip=10))
# results(a=3, b=4,   c=0,   d=8, t1=-2, t2=4, T=10, image_limits=20, clip_function=clip_outer_image_values(image_clip=10))
# results(a=3, b=0.5, c=0.8, d=8, t1=-2, t2=4, T=10, image_limits=20, clip_function=clip_outer_image_values(image_clip=15))

# clip_image_combine = lambda X, func: clip_outer_image_values(image_clip=15)(X, clip_delta_image_values(pivot=8, delta=0.7)(X, func))
# calc(a=3, b=0.5, c=0.8, d=8, t1=-2, t2=4, T=10, image_limits=20, clip_function=clip_image_combine, n=10)
# calc(a=3, b=0, c=1, d=10, t1=-2, t2=4, T=10, image_limits=20, clip_function=clip_delta_image_values(pivot=10, delta=1), n=11)


# clip_image_combine = lambda X, func: clip_outer_image_values(image_clip=15)(X, clip_delta_image_values(pivot=10, delta=1)(X, func))
# calc(a=3, b=2, c=1, d=10, t1=-2, t2=4, T=10, image_limits=20, clip_function=clip_image_combine, n=12)


# clip_image_combine = lambda X, func: clip_outer_image_values(image_clip=15)(X, clip_delta_image_values(pivot=10, delta=1)(X, func))
# calc(a=3, b=2, c=1, d=10, t1=-2, t2=4, T=10, image_limits=20, clip_function=clip_image_combine, n=13)


# clip_image_combine = lambda X, func: clip_outer_image_values(image_clip=15)(X, clip_delta_image_values(pivot=5, delta=1)(X, func))
# calc(a=3, b=0.5, c=1, d=5, t1=-2, t2=4, T=10, image_limits=20, clip_function=clip_image_combine, n=14)


# calc(a=3, b=0.3, c=1, d=10, t1=-2, t2=4, T=10, image_limits=20, clip_function=clip_inner_image_values(image_clip=8), n=15)

# plt.show()