from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty
import cv2
import numpy as np
import os
import string
import random
from skimage.filters import laplace ,sobel, roberts
from numpy.fft import fft2, fftshift, ifft2, ifftshift
from cv2 import *

class MyGrid(Widget):
    #infinte input

    #------------------blur-------------------------
    def blur(self):
        img = cv2.imread(self.ids.myimg.source)

        
        blur=cv2.bilateralFilter(img,d=15,sigmaColor=400,sigmaSpace=400)

        filen = random.choice(string.ascii_letters) + ".png"
        cv2.imwrite(filen,blur)
        self.ids.myimg.source = filen
        os.remove(filen)
    #------------------------------------------------

    #---------------smoothing filter-----------------
    def smoothing(self):
        img = cv2.imread(self.ids.myimg.source)

        kernel2 = np.ones((5, 5), np.float32)/25
        smooth = cv2.filter2D(src=img, ddepth=-1, kernel=kernel2)

        filen = random.choice(string.ascii_letters) + ".png"
        cv2.imwrite(filen,smooth)
        self.ids.myimg.source = filen
        os.remove(filen)
    #------------------------------------------------

    #------------------sharp-------------------------
    def sharp(self):
        img = cv2.imread(self.ids.myimg.source)

        kernel = np.array([[-1 , -1 , -1], 
                            [-1 , 9 , -1] ,
                            [-1 , -1 , -1]])
        sharp_img = cv2.filter2D(img , ddepth=-1 , kernel = kernel)

        filen = random.choice(string.ascii_letters) + ".png"
        cv2.imwrite(filen,sharp_img)
        self.ids.myimg.source = filen
        os.remove(filen)
    #------------------------------------------------

    #------------------Edges-------------------------
    def Edges(self):
        img = cv2.imread(self.ids.myimg.source)

        line_size , blur_value=7,7
        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
        blr=cv2.medianBlur(gray,blur_value)
        edges =cv2.adaptiveThreshold(blr,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,line_size,blur_value)

        filen = random.choice(string.ascii_letters) + ".png"
        cv2.imwrite(filen,edges)
        self.ids.myimg.source = filen
        os.remove(filen)
    #------------------------------------------------

    #---------------------Laplacian------------------
    # def Laplacian(self):
    #     img = cv2.imread(self.ids.myimg.source) 

    #     ImgNew  = laplace(img)

    #     filen = random.choice(string.ascii_letters) + ".png"
    #     cv2.imwrite(filen,ImgNew)
    #     self.ids.myimg.source = filen
    #     os.remove(filen)
    #------------------------------------------------

    #----------------RGB to Grayscale----------------
    def Gray(self):
        img = cv2.imread(self.ids.myimg.source) 

        gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

        filen = random.choice(string.ascii_letters) + ".png"
        cv2.imwrite(filen,gray)
        self.ids.myimg.source = filen
        os.remove(filen)
    #------------------------------------------------

    #-------Noise reduction By mean filter-----------
    def Noise_red(self):
        img = cv2.imread(self.ids.myimg.source) 

        New=img
        temb=img
        temp=np.zeros(img.shape)
        for i in range(2,img.shape[0]-2):
            for j in range(2,img.shape[1]-2):
                New[i][j]=temb[i-2:i+2, j-2:j+2].mean()
        
        filen = random.choice(string.ascii_letters) + ".png"
        cv2.imwrite(filen,New)
        self.ids.myimg.source = filen
        os.remove(filen)
    #------------------------------------------------
    def SaltAndPeper(self):
        img = cv2.imread(self.ids.myimg.source) 

        median = cv2.medianBlur(img,9)

        filen = random.choice(string.ascii_letters) + ".png"
        cv2.imwrite(filen,median)
        self.ids.myimg.source = filen
        os.remove(filen)

    #-------------------Cartoon----------------------
    def cartoon(self):
        img = cv2.imread(self.ids.myimg.source)

        def blur2(img):
            blur=cv2.bilateralFilter(img,d=15,sigmaColor=400,sigmaSpace=400)
            return blur
        def Edges2(img,line_size,blur_value):
            gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
            blr=cv2.medianBlur(gray,blur_value)
            edges =cv2.adaptiveThreshold(blr,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,line_size,blur_value)
            return edges

        cartooned = cv2.bitwise_and(blur2(img),blur2(img), mask=Edges2(img,7,7))

        filen = random.choice(string.ascii_letters) + ".png"
        cv2.imwrite(filen,cartooned)
        self.ids.myimg.source = filen
        os.remove(filen)
        #------------------------------------------------
   

    def denoise_fourier(self):
        img = cv2.imread(self.ids.myimg.source)[:,:,0]

        kernel=55
        img_freq = fft2(img)
        img_freq = fftshift(img_freq)
        freq_spectrum = 20*np.log(np.abs(img_freq))

        row, col = img.shape[0]//2, img.shape[1]//2
        fil = np.zeros(img.shape)
        fil[row-kernel:row+kernel, col-kernel:col+kernel] = 1
    
        img_clear = img_freq*fil
            
        new_img = np.abs(ifft2(ifftshift(img_clear)))

        filen = random.choice(string.ascii_letters) + ".png"
        cv2.imwrite(filen,new_img)
        self.ids.myimg.source = filen
        os.remove(filen)
    
    def selected(self, filename):
        try:
            self.ids.myimg.source = filename[0]
        except:
            pass


class MyApp(App):
    def build(self):
        #add widgets to wind ow

        return MyGrid()

if __name__ == "__main__":
    MyApp().run()
