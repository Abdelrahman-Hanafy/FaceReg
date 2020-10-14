import face_recognition
import cv2
import os

def read_img(path) :
  img= cv2.imread(path)
  (h,w) = img.shape[:2]
  width = 500
  ratio = width/float(w)
  height = int(h*ratio)
  return cv2.resize(img,(width,height))

train_encodings = []
train_names = []
train_dir = 'train'

for file in os.listdir(train_dir):
  img = read_img(train_dir + '/' + file)
  img_encoding = face_recognition.face_encodings(img)[0]
  train_encodings.append(img_encoding)
  train_names.append(file.split('.')[0])

test_dir='test'
for file in os.listdir(test_dir):
  print("processing", file)
  img = read_img(test_dir+'/'+file)
  img_encoding = face_recognition.face_encodings(img)[0]
  results = face_recognition.compare_faces(train_encodings,img_encoding)
  

  for i in range (len(results)):
    if results[i]:
      try:
        name = train_names[i]
        (top, right, bottom, left) = face_recognition.face_locations(img)[0]
        cv2.rectangle(img,(left,top),(right,bottom),(0,0,255),2)
        cv2.putText(img, name,(left+2, top+18),  cv2.FONT_HERSHEY_PLAIN,1,(255,255,255),1)
        cv2.imshow("lalalal",img)
        cv2.waitKey(0)
      except:
        pass
      