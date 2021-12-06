import cv2
import tensorflow as tf

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
clf = tf.keras.models.load_model('fer.h5')
labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
num2emotion = {i:em for i,em in enumerate(labels)}
while True:
    _,img = cap.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.1,4)
    for (x,y,w,h) in faces:
        im = img[y:y+h,x:x+w,:]
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        im = cv2.resize(im,(48,48))
        im = im.reshape((-1,48,48,3))
        label = num2emotion[clf.predict(im)[0].argmax()]
        cv2.putText(img,label,(x+w//3,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),thickness=2)

    cv2.imshow('img',img)
    
    k=cv2.waitKey(30) & 0xff
    if k==27:
        break

cap.release()
