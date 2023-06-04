import numpy as np
import matplotlib.pyplot as plt

def otsu_threshold(image):
   
    hist, bins = np.histogram(image.flatten(), bins=256, range=(0, 255))
    prob = hist / np.sum(hist)
    
    max_variance = 0
    optimal_threshold = 0
    
    for t in range(1, 256):
        w0 = np.sum(prob[:t])
        w1 = np.sum(prob[t:])
        mu0 = np.sum(prob[:t] * np.arange(t)) / w0
        mu1 = np.sum(prob[t:] * np.arange(t, 256)) / w1
         
        variance = w0 * w1 * ((mu0 - mu1) ** 2)
        
        if variance > max_variance:
            max_variance = variance
            optimal_threshold = t

    return optimal_threshold


image = plt.imread('Image_processing_pre_otsus_algorithm.jpg')


if len(image.shape) == 3:

    image_grayscale = np.mean(image, axis=2)
else:
    image_grayscale = image


threshold = otsu_threshold(image_grayscale)

binary_image = np.where(image_grayscale >= threshold, 255, 0)
plt.imshow(binary_image, cmap='gray')
plt.title("Image binaire avec seuil d'Otsu")
plt.axis('off')
plt.show()
