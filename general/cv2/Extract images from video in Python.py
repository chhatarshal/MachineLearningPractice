#!/usr/bin/env python
# coding: utf-8

# # Here we extract images from a video using cv2 library of python

# In[1]:


# import cv2 and os
import cv2 
import os


# In[2]:


video = cv2.VideoCapture("D:\\Video\\Train.mp4")   
    # creating a folder named data 
if not os.path.exists('data'): 
    os.makedirs('data') 


# In[ ]:


# frame 
currentframe = 0
print('started')
while(True): 
    try:
        # read from frame 
        ret,frame = video.read() 
        if ret: 
            name = './data/frame' + str(currentframe) + '.jpg'
            print ('Creating...' + name) 

            # writing the extracted image  
            cv2.imwrite(name, frame)            
            # show how many frames are created 
            currentframe += 1
        else: 
            break
    except:
        print("exception")


# In[ ]:


# Release all space and windows once done 
video.release() 
cv2.destroyAllWindows() 


# In[ ]:




