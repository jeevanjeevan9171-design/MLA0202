import numpy as np
image = np.array([
    [10, 20, 30, 40, 50],
    [20, 200, 30, 200, 60],
    [30, 40, 250, 50, 70],
    [40, 200, 50, 200, 80],
    [50, 60, 70, 80, 90]
], dtype=float)
updated_image = image.copy()
for i in range(1, image.shape[0] - 1):
    for j in range(1, image.shape[1] - 1):

       
        neighbors = [
            image[i - 1, j],  
            image[i + 1, j], 
            image[i, j - 1], 
            image[i, j + 1] 
        ]

        
        updated_image[i, j] = np.mean(neighbors)

print("Original Image:")
print(image.astype(int))

print("\nUpdated Image after MRF Smoothing:")
print(updated_image.astype(int))

print("\nPixel Neighborhood Example:")
print("Each pixel is connected to its Top, Bottom, Left and Right neighbors.")
