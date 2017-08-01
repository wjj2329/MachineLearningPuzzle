import numpy as np
from PIL import Image
import scipy
class cutter:
    def __init__(self, pic, size):
        self.pic=pic
        self.size=size
        self.segments=[]
    def sliceup(self):
        img=Image.open(self.pic)
        img.load()
        self.pic=img
        shape=self.pic.size
        temp1=(shape[0]/self.size)
        temp2=(shape[1]/self.size)
        for x in range(temp1):
           for y in range(temp2):
             x1=x*self.size
             x2=(x+1)*self.size
             y1=y*self.size
             y2=(y+1)*self.size
             self.segments.append(img.crop((x1, y1, x2, y2)))
        dude=0
        for picture in self.segments:
            scipy.misc.imsave("r"+str(dude)+".png", picture)
            dude+=1
            
