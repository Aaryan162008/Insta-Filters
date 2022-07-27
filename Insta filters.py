import cv2 #bgr
import numpy as np
import dlib

img=cv2.imread("C:\\Users\\RAJAT GUPTA\\Desktop\\Aaryan\\Python Level 1\\Hritik Roshan(Bigger).jfif")
img=cv2.resize(img,(0,0),None,0.5,0.5)
imgorig=img.copy()
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
white=(255,255,255)
color=(0,0,255)

def face_detection(gray):

        detector=dlib.get_frontal_face_detector()
        faces=detector(gray)
        return faces


def landmark_detector(faces,gray):

    
    predictor=dlib.shape_predictor("C:\\Users\\RAJAT GUPTA\\Downloads\\shape_predictor_68_face_landmarks.dat")
    for face in faces:
        landmarks=predictor(gray,face)
        x1,y1=face.left(),face.top()
        x2,y2=face.right(),face.bottom()
        landmarks=predictor(gray,face)
        mypoints=[]

        for n in range(68):
        
            x=landmarks.part(n).x
            y=landmarks.part(n).y
            mypoints.append([x,y])
            mypoints_array=np.array(mypoints)
        return mypoints_array

def filter(img,points,scale=3,masked=False,cropped=True):
    if masked:
        mask=np.zeros_like(img)
        mask=cv2.fillPoly(mask,[points],white)
        img=cv2.bitwise_and(img,mask)
        return mask
    if cropped:
        box=cv2.boundingRect(points)
        x,y,w,h=box
        img_cropped=img[y:y+h,x:x+w]
        img_resized=cv2.resize(img_cropped,(0,0),None,scale,scale)
        return img_resized

    
faces=face_detection(gray)
landmark_faces=landmark_detector(faces,gray)
img_lips=filter(img,landmark_faces[48:60],3,masked=True,cropped=False)
img_color_lips=np.zeros_like(img_lips)
img_color_lips[:]=color
img_color_lips=cv2.bitwise_and(img_lips,img_color_lips)
final_image=cv2.addWeighted(img,1,img_color_lips,0.4,0)
cv2.imshow("face",final_image)


#cv2.imshow("gray",gray)
#print(faces)



#cv2.imshow("Hritik Roshan",img)

