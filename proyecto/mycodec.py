import numpy as np
import cv2 as cv
from scipy.signal import medfilt
from scipy import fftpack

def denoise(frame):
    # Eliminacion ruido 1 (Impulsivo)
    
    #Filtro mediana
    frame1 = medfilt(frame, 5)
    
    
    # Eliminacion ruido 2 (Periódico)
    S_img = fftpack.fftshift(fftpack.fft2(frame1))
    espectro_filtrado = S_img*create_mask(S_img.shape, 0.0591)   
    # Reconstrucción
    framef = np.uint8(fftpack.ifft2(fftpack.ifftshift(espectro_filtrado)))
    return framef

def code(frame):
    #
    # Implementa en esta función el bloque transmisor: Transformación + Cuantización + Codificación de fuente
    #    
    return frame


def decode(message):
    #
    # Reemplaza la linea 24...
    #
    frame = np.frombuffer(bytes(memoryview(message)), dtype='uint8').reshape(480, 848)
    #
    # ...con tu implementación del bloque receptor: decodificador + transformación inversa
    #    
    return frame

def create_mask(dims, frequency, size=10):
    freq_int = int(frequency*dims[0])
    mask = np.ones(shape=(dims[0], dims[1]))
    mask[dims[0]//2-size-freq_int:dims[0]//2+size-freq_int, dims[1]//2-size:dims[1]//2+size] = 0 
    mask[dims[0]//2-size+freq_int:dims[0]//2+size+freq_int, dims[1]//2-size:dims[1]//2+size] = 0
    return mask