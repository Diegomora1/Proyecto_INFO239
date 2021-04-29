import numpy as np
import cv2
from scipy.signal import medfilt
from scipy import fftpack
from collections import Counter
import heapq
import json
import math


Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
              [12, 12, 14, 19, 26, 58, 60, 55],
              [14, 13, 16, 24, 40, 57, 69, 56],
              [14, 17, 22, 29, 51, 87, 80, 62],
              [18, 22, 37, 56, 68, 109, 103, 77],
              [24, 35, 55, 64, 81, 104, 113, 92],
              [49, 64, 78, 87, 103, 121, 120, 101],
              [72, 92, 95, 98, 112, 100, 103, 99]])


def denoise(frame):
    
    # Eliminacion ruido 1 (Impulsivo)
    
    #Filtro mediana
    frame1 = medfilt(frame, 5)
    
    # Eliminacion ruido 2 (Periódico)
    S_img = fftpack.fftshift(fftpack.fft2(frame1))
    espectro_filtrado = S_img*create_mask(S_img.shape, 0.03)   
    # Reconstrucción
    framef = np.uint8(fftpack.ifft2(fftpack.ifftshift(espectro_filtrado)))
    
    #print(framef[0:8,0:8])
    return framef



def code(frame):
    #
    # Implementa en esta función el bloque transmisor: Transformación + Cuantización + Codificación de fuente
    #
    calidad = 80 # porcentaje
    imsize = frame.shape
    dct_matrix = np.zeros(shape=imsize)
    
    #TRANSFORMACION DCT

    DCT2 = lambda g, norm='ortho': fftpack.dct( fftpack.dct(g, axis=0, norm=norm), axis=1, norm=norm)

    imsize = frame.shape
    dct_matrix = np.zeros(shape=imsize)
    tira_zigzag = np.zeros(0)

    for i in range(0, imsize[0], 8):
        for j in range(0, imsize[1], 8):
            dct_matrix[i:(i+8),j:(j+8)] = DCT2(frame[i:(i+8),j:(j+8)])

    #print(dct_matrix)
    #bloques_cuantizados = np.zeros(imsize)
    #nnz = np.zeros(dct_matrix.shape)
    if (calidad < 50):
        S = 5000/calidad
    else:
        S = 200 - 2*calidad 
    Q_dyn = np.floor((S*Q + 50) / 100)
    Q_dyn[Q_dyn == 0] = 1
    #recorremos la imagen en bloques de  8x8
    for i in range(0, imsize[0], 8):
        for j in range(0, imsize[1], 8):
            quant = np.round(dct_matrix[i:(i+8),j:(j+8)]/Q_dyn)
            #bloques_cuantizados[i:(i+8),j:(j+8)] = IDCT2(quant)
            # creo que todavia no es necesario aplicar la transformada inversa, el profe la aplica para mostrar diferencias
            # al momento de cuantizar...
            #bloques_cuantizados[i:(i+8),j:(j+8)] = quant
            # reordenamos cada bloque de 8x8 en tiras con el patron zig zag
            
            
            #print(quant)
            #print("----------------")
            zz = zigzag2 (quant)
            #print(zz)
            #print("----------------")
            #zinv = inverse_zigzag(zz)
            #print(zinv)
            #print("----------------")
            #waitKey(0)

            tira_zigzag = np.append(tira_zigzag, zz)

    #print('debug')
    # Aplicamos Run Length encoding (RLE) a cada tira 
    img_rle = rle(tira_zigzag, imsize[0]*imsize[1])

    #print(img_rle, img_rle.size)

    imh = huffmann(img_rle)

    #print(type(imh))

    imhs = json.dumps(imh, indent=2).encode('utf-8')

    return imhs


def decode(message):
    IDCT2 = lambda G, norm='ortho': (fftpack.idct( fftpack.idct(G, axis=0, norm=norm), axis=1, norm=norm))
    #
    # Reemplaza la linea 24...
    #
    idct_matrix = np.zeros((480,848))
    dendo = json.loads(message)
    #print(type(dendo))
    data = dendo[1]
    dicc = dendo[0]

    # Decodificacion
    dicc_inverso = {codigo: simbolo for simbolo, codigo in dicc.items()}
    codigo = ""
    frame_codif = []
    for bit in data:
        codigo += bit
        if codigo in dicc_inverso:
            frame_codif.append(dicc_inverso[codigo])
            codigo = ""
    
    
    # RLE inverso
    frame1 = rle_inverso(frame_codif)

    # Zig-zag inverso
    #framef = inverse_zigzag(frame1)


    k = 0
    for i in range(0,480,8):
        for j in range(0,848,8):
            idct_matrix[i:(i+8), j:(j+8)] = inverse_zigzag(frame1[k:k+64])
            k+=64
    # Decuantizacion y Detransformacion
    calidad = 80
    if (calidad < 50):
        S = 5000/calidad
    else:
        S = 200 - 2*calidad 
    frame_dq = np.zeros((480, 848))
    Q_dyn = np.floor((S*Q + 50) / 100)
    Q_dyn[Q_dyn == 0] = 1
    #recorremos la imagen en bloques de  8x8
    for i in range(0, 480, 8):
        for j in range(0, 848, 8):
            dequant = idct_matrix[i:(i+8),j:(j+8)] * Q_dyn
            frame_dq[i:(i+8),j:(j+8)] = IDCT2(dequant)
    #print(frame_dq)
    return frame_dq


def huffmann (tira):
    # Construir dendograma con las probabilidades ordenadas
    dendograma = [[frequencia/len(tira), [simbolo, ""]] for simbolo, frequencia in Counter(tira).items()]
    #print(dendograma)
    heapq.heapify(dendograma)
    # Crear el código
    while len(dendograma) > 1:
        lo = heapq.heappop(dendograma)
        hi = heapq.heappop(dendograma)
        for codigo in lo[1:]:
            codigo[1] = '0' + codigo[1]
        for codigo in hi[1:]:
            codigo[1] = '1' + codigo[1]
        heapq.heappush(dendograma, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    # Convertir código a diccionario
    dendograma = sorted(heapq.heappop(dendograma)[1:])
    dendograma = {simbolo : codigo for simbolo, codigo in dendograma} 
    #print(dendograma)

    #tira_codificada = ""
    tira_codificada = [] #np.zeros(0)
    for valor in tira:
        #tira_codificada = np.r_[tira_codificada, [dendograma[valor]]]
        tira_codificada += dendograma[valor]
    #print('hola')
    #tira_codificada = np.array(tira_codificada)
    #print(type(dendograma))

    dendograma_byte = dendograma #json.dumps(dendograma, indent=2).encode('utf-8')
    #dnp = np.array(dendograma_byte)
    #tupla = (tira_codificada, dendograma_byte)
    #tupla2 = json.dumps(tupla)
    #print(type(dendograma_byte))
    #print(dendograma_byte)

    #envio = np.array([tira_codificada, dendograma_byte])
    return dendograma_byte, tira_codificada


def dehuffman(data, dendograma):
    #print(dendograma)

    dendograma_inverso =  {codigo: simbolo for simbolo, codigo in dendograma.items()}

    #print(dendograma_inverso)

    codigo = ""
    texto = ""
    for bit in data:
        codigo += bit
        if codigo in dendograma_inverso:
            texto += dendograma_inverso[codigo]
            codigo = ""

    floats = [float(x) for x in texto.split()]

    return floats


def create_mask(dims, frequency, size=10):
    freq_int = int(frequency*dims[0])
    mask = np.ones(shape=(dims[0], dims[1]))
    mask[dims[0]//2-size-freq_int:dims[0]//2+size-freq_int, dims[1]//2-size:dims[1]//2+size] = 0 
    mask[dims[0]//2-size+freq_int:dims[0]//2+size+freq_int, dims[1]//2-size:dims[1]//2+size] = 0
    return mask

def rle(message, n):
    #encoded_message = np.zeros(0)
    encoded_message = []
    i = 0
    while (i <= n-1):
        count = 1
        ch = message[i]
        j = i
        while (j < n-1):
            if (message[j] == message[j+1]):
                count = count+1
                j = j+1
            else:
                break
        #encoded_message = np.r_[encoded_message, [count, ch]]
        encoded_message += [count,ch]
        i = j+1
    
    encoded_message = np.array(encoded_message)

    return encoded_message


def rle_inverso(input):
    output = []
    j = 0
    n = len(input)
    for i in range(0,n,2):
        for k in range(j,j + math.floor(float(input[i]))):
            output.append(input[i+1])
            j = j + 1
    return output

def zigzag2(frameq):
    rows = 8
    columns = 8
    solution=[[] for i in range(rows+columns-1)]
    solution2 = []

    for i in range(rows):
        for j in range(columns):
            sum=i+j
            if(sum%2 ==0):
                #add at beginning
                solution[sum].insert(0,frameq[i][j])
            else:
                #add at end of the list
                solution[sum].append(frameq[i][j])

    for x in solution:
        solution2 += x
    solution2 = np.array(solution2)
    return solution2



def inverse_zigzag(input):	
	#print input.shape
	# initializing the variables
	#----------------------------------
    h = 0
    v = 0
    vmin = 0
    hmin = 0
    vmax = 8
    hmax = 8
    output = np.zeros((vmax, hmax))
    i = 0
    #----------------------------------
    while ((v < vmax) and (h < hmax)): 
        #print ('v:',v,', h:',h,', i:',i)   	
        if ((h + v) % 2) == 0:                 # going up            
            if (v == vmin):
                #print(1)                
                output[v, h] = input[i]        # if we got to the first line
                if (h == hmax):
                    v = v + 1
                else:
                    h = h + 1                        
                i = i + 1
            elif ((h == hmax -1 ) and (v < vmax)):   # if we got to the last column
                #print(2)
                output[v, h] = input[i] 
                v = v + 1
                i = i + 1
            elif ((v > vmin) and (h < hmax -1 )):    # all other cases
                #print(3)
                output[v, h] = input[i] 
                v = v - 1
                h = h + 1
                i = i + 1
        else:                                    # going down
            if ((v == vmax -1) and (h <= hmax -1)):       # if we got to the last line
                #print(4)
                output[v, h] = input[i] 
                h = h + 1
                i = i + 1
            elif (h == hmin):                  # if we got to the first column
                #print(5)
                output[v, h] = input[i] 
                if (v == vmax -1):
                    h = h + 1
                else:
                    v = v + 1
                i = i + 1                                
            elif((v < vmax -1) and (h > hmin)):     # all other cases
                output[v, h] = input[i] 
                v = v + 1
                h = h - 1
                i = i + 1
        if ((v == vmax-1) and (h == hmax-1)):          # bottom right element
            #print(7)        	
            output[v, h] = input[i] 
            break
    return output    

