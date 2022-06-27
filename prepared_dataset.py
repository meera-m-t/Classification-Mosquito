import cv2 
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
import matplotlib.pyplot as plt
from PIL import Image
import os
import glob
cv2.setRNGSeed(0)
# Disable OpenCL and disable multi-threading.
# cv2.ocl.setUseOpenCL(False)
# cv2.setNumThreads(1)
def segmentation(original_image):
    height, width, _ = original_image.shape
    left_margin_proportion = 0.1
    right_margin_proportion = 0.1
    up_margin_proportion = 0.1
    down_margin_proportion = 0.1

    boundary_rectangle = (
        int(width * left_margin_proportion),
        int(height * up_margin_proportion),
        int(width * (1 - right_margin_proportion)),
        int(height * (1 - down_margin_proportion)),
    )
    number_of_iterations = 15   
    img = original_image.reshape((-1, 3))  
    gmm_model = GMM(n_components= 2, covariance_type='tied').fit(img)
    gmm_label = gmm_model.predict(img)
    binarized_image = gmm_label.reshape(original_image.shape[0],original_image.shape[1])    
    mask = np.zeros((height, width), np.uint8)
    mask[:] = cv2.GC_PR_BGD
    mask[binarized_image == 1] = cv2.GC_FGD
    # Arrays used by the algorithm internally
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(
        original_image,
        mask,
        boundary_rectangle,
        background_model,
        foreground_model,
        number_of_iterations,
        cv2.GC_INIT_WITH_MASK,
    )
    grabcut_mask = np.where((mask == cv2.GC_PR_BGD) | (mask == cv2.GC_BGD), 0, 1).astype( "uint8")
    segmented_image = original_image.copy() * grabcut_mask[:, :, np.newaxis]    
    return segmented_image

def removeSmallComponents(image, threshold):
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(image, connectivity=8)
    sizes = stats[1:, -2]
    nb_components = nb_components - 1
    img2 = np.zeros((output.shape),dtype = np.uint8)
    #for every component in the image, you keep it only if it's above threshold
    for i in range(0, nb_components):
        if sizes[i] >= threshold:
            img2[output == i + 1] = 255
    return img2


def clean_image(mask, image):
    image = cv2.bitwise_and(image, image, mask=mask)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    # Find contour and sort by contour area
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    # Find bounding box and extract ROI
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        ROI = image[y:y+h, x:x+w]        
        break
    print(ROI.shape)
    return cv2.resize(ROI,(250,250))



def main():
    directory1 = 'dataset'
    directory2 = 'processed_dataset'

    for filename in os.listdir(directory1):
        print(filename)
        for file in sorted(glob.glob(f'{directory1}/{filename}/*.jpg'), key=os.path.getmtime):
            original_image = cv2.imread(file)
            seg_img = segmentation(original_image)
            gray = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY) 
            thresh = cv2.threshold(gray, 100, 255,cv2.THRESH_BINARY)[1]
            clean_img = removeSmallComponents(gray, 100)
            obj_img = clean_image(clean_img, seg_img)
            black_image = np.zeros((500, 500, 3), dtype = "uint8")
            overlay_img1 = np.ones(black_image.shape,np.uint8)*255
            rows,cols,channels = obj_img.shape
            overlay_img1[125:rows+125, 125:cols+125 ] = obj_img
            img2gray = cv2.cvtColor(overlay_img1,cv2.COLOR_BGR2GRAY)
            ret, mask = cv2.threshold(img2gray,220,55,cv2.THRESH_BINARY_INV)
            mask_inv = cv2.bitwise_not(mask)
            temp1 = cv2.bitwise_and(black_image,black_image,mask = mask_inv)
            temp2 = cv2.bitwise_and(overlay_img1,overlay_img1, mask = mask)
            result = cv2.add(temp1,temp2)
            cv2.imwrite(f'{directory2}/{filename}/{os.path.basename(file)}',result)   
            

if __name__ == "__main__":
    main()
