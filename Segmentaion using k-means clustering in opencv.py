import cv2
import numpy as np 
import matplotlib.pyplot as plt

image = cv2.imread('img.png')

image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)

image.shape


# convert to 2D 
pixels_values = image.reshape((-1, 3)) # -1 to convert to 2D & 3 to 3 channels rgb
# convert to Float
pixels_values = np.float32(pixels_values)


print(pixels_values.shape)


# define stopping criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER , 100, 0.95) # 100 iterations & 0.95 is accuracy

# number of clusters (K)
k = 3
_, labels, (centers) = cv2.kmeans(pixels_values, k , None , criteria, 10 , cv2.KMEANS_RANDOM_CENTERS)

# convert back to 8 bits values (original image)
centers = np.uint8(centers)

# flatten the lables array
labels = labels.flatten()

# convert all pixels to the color of the centroids
segmented_image = centers[labels.flatten()] # assigning to the saame colors to be near 



# reshape back to the original image dimension
segmented_image = segmented_image.reshape(image.shape)

# Show image
plt.imshow(segmented_image)
plt.show()

########################

#display only the cluster number 2 (turn the pixels into back)
masked_image = np.copy(image)
# convert to the shape of a vector of pixel values 
masked_image = masked_image.reshape((-1, 3))
# color (i.e cluster)  disable
cluster = 1
masked_image [labels == cluster] = [0,0,0] # black to cluster = 1
# convert back to original shape
masked_image = masked_image.reshape(image.shape)
#show the image
plt.imshow(masked_image)
plt.imshow()




cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()