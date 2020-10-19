import cv2
import numpy as np
import matplotlib.pyplot as plt
def calcGrayHist(image):
    rows,cols = image.shape
    # min=image.min()
    # max=image.max()
    grayHist = np.zeros([256],np.uint64)
    for r in range(rows):
        for c in range(cols):
            # image[r, c] = ((image[r, c] - min) * (90-30)) / (max - min)+30
            grayHist[image[r][c]] += 1  #把图像灰度值作为索引
    return(grayHist)
def otsu(image):
    rows,cols = image.shape
    #灰度直方图
    grayHist = calcGrayHist(image)
    # plt.hist(image,bins='auto')
    # plt.show()
    #归一化灰度直方图
    uniformGrayHist = grayHist/float(rows*cols)

    #计算零阶累积矩和一阶累积矩
    zeroC = np.zeros([256],np.float32)
    oneC = np.zeros([256],np.float32)
    for k in range(256):
        if k==0:
            zeroC[k] = uniformGrayHist[0]
            oneC[k] = (k)*uniformGrayHist[0]
        else:
            zeroC[k] = zeroC[k-1]+uniformGrayHist[k]
            oneC[k] = oneC[k-1]+k*uniformGrayHist[k]


    #计算类间方差
    variance = np.zeros([256],np.float32)
    for k in range(60,255,1):
        p_a = zeroC[k];
        p_b = 1-zeroC[k];
        mean=oneC[255]
        mean_a = oneC[k];
        mean_b = 1-oneC[k];
        variance[k] = p_a *((mean_a - mean)**2) + p_b * ((mean_b - mean)**2) ;

        # if zeroC[k] == 0 or zeroC[k] ==1:
        #     variance[k] = 0
        # else:
        #     variance[k]=math.pow(oneC[255]*zeroC[k]-
        #    oneC[k],2)/(zeroC[k]*(1.0-zeroC[k]))
    print(variance)
    thresh1 =np.argmax(variance)
    thresh= variance[thresh1]
    #
    threshold_Image = np.copy(image)
    threshold_Image[threshold_Image>thresh]=255
    threshold_Image[threshold_Image<=thresh]=0
    # ret, img_Otsu = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)
    return(thresh,threshold_Image)
if __name__ =='__main__':
    src = cv2.imread('MRA.pgm',cv2.IMREAD_GRAYSCALE)
    re,ra = otsu(src)
    print(re)
    print(ra)
    cv2.namedWindow('ra', 0)
    cv2.resizeWindow('ra', 138, 69);
    cv2.imshow('ra',ra)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # plt.imshow(ra,cmap='black')
    # plt.show()