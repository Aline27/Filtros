import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import sqrt

def tam_filter(text, img_original):

	cont=0
	for i in range (0, len(text)):
		if (text[i].isdigit()):
			cont=cont+1
	tam_filter=int(sqrt(cont))
	return tam_filter	
	
def create_matrizAux(tam_filter,img_original):

	tam_img=img_original.shape
	print tam_img
	matriz_aux=np.zeros(((tam_img[0]+tam_filter-1),(tam_img[1]+tam_filter-1),3), dtype=np.uint8)
	tam_matriz_aux=matriz_aux.shape
	print tam_matriz_aux
	aux=int(tam_filter/2)
	matriz_aux[aux:tam_matriz_aux[0]-aux,aux:tam_matriz_aux[1]-aux,:]= img_original
	return matriz_aux


def filter_average(tam_filter,matriz_aux,img_size):

	size_matriz=matriz_aux.shape
	kernel=np.ones((tam_filter,tam_filter))
	kernel_sum = kernel.sum()
	print kernel_sum
	img_filtered=np.zeros((img_size[0],img_size[1],3), dtype=np.uint8)

	for i_mat in range(0,size_matriz[0]-tam_filter+1):
		for j_mat in range(0,size_matriz[1]-tam_filter+1):
			soma=0
			for i_kernel in range (0,tam_filter):
				for j_kernel in range (0, tam_filter):
					
					aux=matriz_aux[i_mat+i_kernel,j_mat+j_kernel]*kernel[i_kernel,j_kernel]
					soma=soma+aux

			img_filtered[i_mat,j_mat]=soma/kernel_sum

	return img_filtered

#---------------------------------------------------------------------------------------------------------

arq = open('filtro9.txt', 'r')
text = arq.read()
print(text)
arq.close()

img_original= mpimg.imread('3.jpg')
img_size=img_original.shape
plt.imshow(img_original)
plt.show()

tam=tam_filter(text, img_original)
matriz_result=create_matrizAux(tam,img_original)
plt.imshow(matriz_result,origin='upper')
plt.show()

img_filtered=filter_average(tam,matriz_result,img_size)
plt.imshow(img_filtered,origin='upper')
plt.show()











