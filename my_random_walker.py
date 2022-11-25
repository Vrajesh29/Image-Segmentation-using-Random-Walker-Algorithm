import cv2 as cv
import numpy as np
import random
import os
from skimage.segmentation import random_walker
from skimage.data import binary_blobs
from skimage.exposure import rescale_intensity
import skimage

RESIZE = 0.5  #To resize the image for computational efficiency.
Segment_colors= [[0,255,255],[255,0,0],[0,255,0], [0,0,255]]    # The color of different segments.
#                  Yellow       Blue     Green      Red

def down_size(x):
    return int(x*RESIZE)    # Down size the image by factor of 0.5

def mouse_callback(event, x, y, flags, params):       # Notes the clicks on the image

    if event==1:
        clicks.append([x,y])
        # print(clicks)

def getVal(y,x,ar):             # check if the pixel is valid or not.
    if x<0 or y <0 or y >= ar.shape[0] or x >=ar.shape[1]:
        return np.array([-1000.0,-1000.0,-1000.0])
    else:
        return ar[y,x,:]


img_number = str(input("Enter the image number you want to segment: ")).strip()
img=cv.imread(str(img_number)+'.png')  # **Enter the image you want to segment**
Segments=int(str(input("Enter the number of segments? (Maimum 4): ")).strip())  # Enter the number of segments. Should be Less than 4.

labelledPixelsXY=[]   # To collect the pixels that we will label in a given segment.
Pixels = int(str(input("Enter the number of pixels being marked?: ")).strip())  
beta = int(str(input("Enter the value of beta: ")).strip())
# Enter the number of pixels you want to select on a given segment.

# Mark the pixels on the selected image.
for n in range(Segments):
    print("Mark pixels for segment ",n)
    cv.imshow("image",img)
    clicks=[]
    cv.setMouseCallback('image', mouse_callback)
    while True:
        if len(clicks)==Pixels:
            break
        cv.waitKey(1)
    labelledPixelsXY.append(clicks)         # Appending the labelled pixels
    clicks=[]

# print(labelledPixelsXY)

# Save the Initial marks
imgCopy=np.array(img)
for n in range(Segments):
    for i in range(len(labelledPixelsXY[n])):
        # print(imgCopy, labelledPixelsXY[n][i], 2,Segment_colors[n],3)
        cv.circle(imgCopy, (labelledPixelsXY[n][i][0], labelledPixelsXY[n][i][1]), 2,Segment_colors[n],3)


# Normalizing the image for computational efficiency
imgOriginal=np.array(img)
img=img/255.0
img=cv.resize(img, (int(img.shape[1]*RESIZE)+1, int(img.shape[0]*RESIZE)+1))
initiallyMarked=np.zeros((img.shape[0],img.shape[1]),dtype=int)
initiallyMarked.fill(-1)
segments=np.zeros((img.shape[0],img.shape[1]),dtype=int)
segments.fill(-1)
Prob=np.zeros((img.shape[0], img.shape[1],4),dtype=float)



# Generate the transition probabilites based on pixel similarity

for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        urdl=[getVal(y-1,x,img),getVal(y,x+1,img), getVal(y+1,x,img),getVal(y,x-1,img)]  # Get the 4 neighbouring pixels.
        ProbURDL=[]
        for a in range(4):
            tt=np.mean(np.abs(urdl[a]-img[y,x,:]))  # This is the 2-tree mentioned in the paper.
            tt=np.exp(-beta*np.power(tt,2))     # Beta is variable as mentioned in paper. You may choose beta=1 and proceed.
            ProbURDL.append(tt)
        ProbURDL=np.array(ProbURDL)
        normalizedProbURDL = ProbURDL / np.sum(ProbURDL)   # Normalizing the above probability.
        Prob[y,x,0]=normalizedProbURDL[0]        # Normalized and cumulative probability of the 4 neighbours.
        for a in range(1,4):
            Prob[y,x,a]=Prob[y,x,a-1]+normalizedProbURDL[a]

for s in range(Segments):
    for a in range(len(labelledPixelsXY[s])):
        initiallyMarked[down_size(labelledPixelsXY[s][a][1]), down_size(labelledPixelsXY[s][a][0])]=s
        segments[down_size(labelledPixelsXY[s][a][1]), down_size(labelledPixelsXY[s][a][0])]=s


# Random Walker Algorithm
for y in range(segments.shape[0]):
    for x in range(segments.shape[1]):
        if segments[y][x]==-1:
            yy=y
            xx=x

            while(initiallyMarked[yy,xx]==-1):
                rv = random.random()
                if Prob[yy,xx,0]>rv:
                    yy-=1
                elif Prob[yy,xx,1]>rv:
                    xx+=1
                elif Prob[yy,xx,2]>rv:
                    yy+=1
                else:
                    xx-=1
            segments[y,x]=initiallyMarked[yy,xx]


outputImg=np.array(imgOriginal)
for y in range(outputImg.shape[0]):
    for x in range(outputImg.shape[1]):        
        outputImg[y,x]=Segment_colors[segments[down_size(y),down_size(x)]]
path = "C:\\Users\\vraje\\OneDrive\\Desktop\\Assignment4\\Ouputs"
# cv.imwrite(os.path.join(path,"Input_and_Segmented_Output"+str(img_number)+".jpg"), np.concatenate((imgCopy,outputImg),axis=1))

# Now we will use built-in Random walker for comparison.
built_in_outputImg=[]
def built_in_random_walker():
    img = cv.imread(str(img_number)+'.png')
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = img/255.0
    markers = np.zeros(img.shape, dtype=np.uint)
    for s in range(Segments):
        for a in range(len(labelledPixelsXY[s])): 
            x_coord = labelledPixelsXY[s][a][0]
            y_coord = labelledPixelsXY[s][a][1]
            markers[y_coord,x_coord]=s+1
    labels = random_walker(img, markers, beta=beta, mode='bf')
    built_in_outputImg=np.array(imgOriginal)
    for y in range(built_in_outputImg.shape[0]):
        for x in range(built_in_outputImg.shape[1]):        
            built_in_outputImg[y,x]=Segment_colors[labels[y,x]-1]
    cv.imwrite(os.path.join(path,"Initial_myOutput_Builtin"+str(img_number)+".jpg"),np.concatenate((imgCopy,outputImg,built_in_outputImg),axis=1))
# The outputs folder contains the input image, image generated from out algorithm and image generated from built in function from left to right
    return built_in_outputImg
built_in_outputImg=built_in_random_walker()

def absolute_mean_error(imageA, imageB):
    imageA = np.array(imageA, dtype=np.float32)
    imageB = np.array(imageB, dtype=np.float32)
    err = np.sum(abs(imageA-imageB))
    err /= (float(imageA.shape[0] * imageA.shape[1])*3)
    return err

print(absolute_mean_error(outputImg,built_in_outputImg))
  

