import numpy as np
import onnxruntime
import dlib
import cv2
import glob
import os

GENDER_DICT = {0: 'male', 1: 'female'}


import os
def get_fileNames(rootdir):    
	fs = []    
	for root, dirs, files in os.walk(rootdir,topdown = True):
		for name in files: 
			_, ending = os.path.splitext(name)
			if ending == ".jpg":
					fs.append(os.path.join(root,name))   
	return fs




onnx_session = onnxruntime.InferenceSession('models-2020-11-20-14-37/best-epoch47-0.9314.onnx')
detector = dlib.get_frontal_face_detector()

#dest_path = os.path.join('style_original_images', '*/*.jpg')
jpgfiles = get_fileNames("style_original_images")#glob.glob(dest_path)

for imgfile in jpgfiles:
    img = cv2.imread(imgfile)
    face_rects = detector(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 0)
    for face_rect in face_rects:
        cv2.rectangle(img,
                    (face_rect.left(), face_rect.top()),
                    (face_rect.right(), face_rect.bottom()),
                    (255, 255, 255))
        face = img[face_rect.top():face_rect.bottom(), face_rect.left():face_rect.right(), :]
        cv2.imwrite("crop2.jpg", img);
        inputs = np.transpose(cv2.resize(face, (64, 64)), (2, 0, 1))
        inputs = np.expand_dims(inputs, 0).astype(np.float32) / 255.
        predictions = onnx_session.run(['output'], input_feed={'input': inputs})[0][0]
        gender = GENDER_DICT[int(np.argmax(predictions[:2]))]
        age = int(predictions[2])
        cv2.putText(img, 'Gender: {}, Age: {}'.format(gender, age), (face_rect.left(), face_rect.top()), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255))
        print('Image:{}, Gender: {}, Age: {}'.format(imgfile, gender, age))




#path = "style_original_images"


#cv2.imshow('', img)
#cv2.imwrite('result_3.jpg',img)
#cv2.waitKey(1)
