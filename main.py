"""
Project: Deep visibility from Image Quality
Recreating results of paper: "Learning Local Distortion Visibility from Image Quality"
@author:Navaneeth Kamballur Kottayil
"""

from data_utils import *
from model import *
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.optimize import curve_fit
import sys
sys.path.append("D:\\Documents\\PhD\\Python Scripts\\Exact CSF\\chandlers measurement\\")
from read_masking_values import *
import datetime

from scipy import signal
rmsa = []
test = model_deep_vis(32,32,1)
im_list = []
[im, th] = read_masks()        
fin_pt = []
fin_st = []        
for i,(img,thr) in enumerate( zip(im,th) ):
    map1 = test.get_thresh(img)
    map2  = cv2.blur(map1,(21,21))
    map = cv2.resize(map2,(6,6))
    
    img2 = np.pad(img, [(0, 32), (0, 32)], mode='constant', constant_values=40)
    ip,_,_, _ =  extractpatchwithLabel( img2, 
                            img2 ,85,85,
                            score=0,subsample=1)            
    im_list += list(ip)          
    X = map.reshape(-1)
    lum = np.mean(ip, axis=(1,2,3))            
    Y = thr.reshape(-1)
    plot = 1            
    if plot:
        im_show(img, cmap='gray')
        im_show(map1,cmap='gray',title = "Predicted threshold full")
        im_show(map,cmap='gray',title = "Predicted threshold resampled")
        im_show(Y.reshape(6,6),cmap='gray',title = "Ground Truth")
        plt.scatter(X,Y)
        plt.title("SRCC :" +str(scipy.stats.pearsonr(X/np.max(X), Y/np.max(Y))[0]))        
        plt.show()

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    