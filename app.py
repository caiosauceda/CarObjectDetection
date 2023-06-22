import cv2
import numpy as np
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox

def count_cars(image_path):
    # Carrega a imagem usando o caminho fornecido
    image = cv2.imread(image_path)
    
    # Verifica se a imagem foi carregada corretamente
    if image is None:
        print("Erro ao carregar a imagem. Verifique o caminho.")
        return
    
    # Detecta objetos comuns na imagem, retorna as caixas delimitadoras, rótulos e a contagem
    boxes, labels, count = cv.detect_common_objects(image)
    
    # Desenha as caixas delimitadoras em torno dos objetos detectados na imagem
    output = draw_bbox(image, boxes, labels, count)
    
    # Mostra a imagem resultante com as caixas delimitadoras
    plt.imshow(output)
    plt.show()
    
    # Conta o número de carros na imagem
    num_cars = labels.count('car')
    print("Número de carros na imagem: " + str(num_cars))

# Exemplo de uso:
image_path = "cars.jpeg"
count_cars(image_path)
