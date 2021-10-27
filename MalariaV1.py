#-----------------------------------------
#PRIMER ENTREGA PROYECTO: PAULA CASTRO Y MICHAEL CONTRERAS
#-----------------------------------------
import cv2
import numpy as np
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt

if __name__ == '__main__':
    #Cargar imagen
    path = sys.argv[1]
    image_name = sys.argv[2]
    path_file = os.path.join(path, image_name)
    # Conversion de imagen a arreglo para manipular con librerias
    im = np.array(Image.open(path_file))
    # Craer copia de la original
    ima = im.copy()
    # Conversion de imagen a escala de grises
    image_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Aplica filtro Gaussiano
    gauss = cv2.GaussianBlur(image_gray, (3, 3), 0)

    # Aplicacion de metodo Otsu
    ret, Ibw_otsu = cv2.threshold(gauss, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Copia de imagen binarizada
    image_bin= np.copy(Ibw_otsu)
    # Extraccion de tamaño imagen alto y ancho
    h,w = image_bin.shape
    # Crear mascara y lenado de huecos para extraer objetos de interes y candidatos a posibles parasitos
    mask = np.zeros((h+2,w+2),np.uint8)
    cv2.floodFill(image_bin,mask,(0,0),255)
    imagem = cv2.bitwise_not(image_bin)

    # Detectamos los bordes con Canny
    canny = cv2.Canny(imagem, 50, 150)

    # Decteccion de contornos
    contours, hierarchy = cv2.findContours(imagem, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Ubicacion de punto central, caculo de area y perimetro de cada objeto encontrado
    for i in range(len(contours)):
        if len(contours[i]) > 2:
            cont = contours[i]
            M = cv2.moments(cont)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            area = cv2.contourArea(cont)
            perimetro = cv2.arcLength(cont,True)
            print(area)
            print(perimetro)
            cv2.circle(ima, (cx,cy),50, (0, 255, 0), 7)


    print("Se han encontrado {} objetos en la muestra".format(len(contours)))


    fig, axs = plt.subplots(2, 3, figsize=(7, 7))
    axs[0, 0].imshow(im)
    axs[0, 0].set_title('Imagen Original', fontweight="bold")
    axs[1, 0].imshow(mask)
    axs[1, 0].set_title('Mascara imagen', fontweight="bold")
    axs[0, 1].imshow(image_gray)
    axs[0, 1].set_title('Imagen escala grises', fontweight="bold")
    axs[1, 1].imshow(canny)
    axs[1, 1].set_title('Imagen deteccion de bordes', fontweight="bold")
    axs[0, 2].imshow(Ibw_otsu)
    axs[0, 2].set_title('Imagen Otsu', fontweight="bold")
    axs[1, 2].imshow(imagem)
    axs[1, 2].set_title('Imagen Objetos de interes', fontweight="bold")
    plt.show()

    plt.imshow(ima)
    plt.title('Deteccion de contornos externos', fontweight="bold")
    plt.show()