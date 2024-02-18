from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import cv2

frames_u=[]
f=[]

def subtract_and_clip(array, number):
    result = array - number
    result[result < 0] = 0
    return result
def log_transform(image):
    c = 65536 / np.log(1 + np.max(image))
    return (c * np.log(1 + image)).astype(np.uint16)
def read_tiff_frames(folder_path):
    
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            filepath = os.path.join(folder_path, filename)
            frame = np.array(cv2.imread(filepath, cv2.IMREAD_UNCHANGED),dtype=np.uint16)
            f.append(frame)
            temp=log_transform(subtract_and_clip(frame,second_minimum_2(frame.flatten())[0]))
            frames_u.append(np.stack((temp,) * 3, axis=-1))
    return frames_u

def second_minimum_2(arr):
    arr = arr.astype(float)
    min_val = np.min(arr)
    arr[arr == min_val] = np.inf
    second_min_val = np.min(arr)
    indices = np.where(arr == second_min_val)
    return second_min_val, indices

def save_tif_files(array, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for i, arr in enumerate(array):
        filename = f"{i:05d}.tif"
        file_path = os.path.join(output_folder, filename)
        cv2.imwrite(file_path, arr)
        print(f"Saved {file_path}")

def histeq(img):
    img = np.asarray(img)   #Convert variable type to numpy array

    h = [0]*65536                     
    for x in range(img.shape[0]):        
        for y in range(img.shape[1]):            
            i = img[x,y]                  
            h[i] = h[i]+1
    return h

def filter_images(frames_list, kernel,blue_lower,blue_upper):
    filtered_frames = []
    for frame in frames_list:
        image2=np.zeros(frame.shape,dtype=np.uint8)
        blue_mask = (frame[:, :, 0] >= blue_lower) & (frame[:, :, 0] <= blue_upper)
        image2[blue_mask] = [0, 0, 65535]
        filtered_frames.append(cv2.filter2D(image2, -1, kernel))
    return filtered_frames

def main():
    if len(sys.argv) < 3:
        print("Usage: python prepro.py [tiff_folder_path] [output_folder]")
        sys.exit(1)
    

    folder_path = sys.argv[1] # Change this to the folder containing TIFF frames
    frames_u = read_tiff_frames(folder_path)
    print(f"Read and log transformed {len(frames_u)} frames")
    colour_folder=sys.argv[2]+"/color"


    save_tif_files(frames_u, colour_folder)
    lower=np.argmax(histeq(frames_u[0][:,:,0]))+7000
    upper=2**16-1
    print(f"Mask Lower {lower} ,Upper {upper}")
    kernel=np.ones((1,1))
    mask_folder=sys.argv[2]+"/mask"
    filtered_frames = filter_images(frames_u, kernel,lower,upper)
    print(f"Number of filtered frames {len(frames_u)}.")
    save_tif_files(filtered_frames, mask_folder)

    
main()



