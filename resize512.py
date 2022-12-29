import os,cv2

root = '/home/lyn/Documents/data1/'
imgDir = root + 'images'
labelDir = root + 'labels'

saveim = '/home/lyn/Documents/data1/data1_resize512/images'
savelabels = '/home/lyn/Documents/data1/data1_resize512/labels'
for im in os.listdir(imgDir):
    labelName = im.split('.')[0] + '.png'
    imPath = os.path.join(imgDir,im)
    labelPath = os.path.join(labelDir,labelName)
    image_ = cv2.imread(imPath)
    label_ = cv2.imread(labelPath,cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image_,(512,512))
    label = cv2.resize(label_, (512, 512))
    cv2.imwrite('{}/{}'.format(saveim,im),image)
    cv2.imwrite('{}/{}'.format(savelabels, labelName), label)
    # print(label.shape)
    # exit()
