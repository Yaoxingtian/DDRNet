import cv2
video = '/home/yao/Documents/DDRNet.pytorch-main/videos/office_01.mp4'
savePATH = '/home/yao/Documents/DDRNet.pytorch-main/videos/office_01_images'
cap = cv2.VideoCapture(video)

j = 1
i = 1
while cap.isOpened():

    i += 1
    if i%200 == 0:
        print(i)
        ret,frame = cap.read()
        img = cv2.resize(frame,(640,480))
        cv2.imwrite('{}/office_01_{}.jpg'.format(savePATH,j),img)


        j+=1