import os,glob,cv2,shutil

def mouse_click(event, x, y, flags, para):
    # global one_points
    global left_points
    global right_points
    if event == cv2.EVENT_LBUTTONDOWN:  # 左边鼠标点击
        left_points.append([x, y])

    elif  event == cv2.EVENT_LBUTTONDBLCLK:
        right_points.append([x, y])
        # cv2.EVENT_LBUTTONUP


if __name__ == '__main__':

    imgDir = '/home/lyn/Documents/workspace/DDRNet.pytorch-main/run/detect/exp98'
    good = '/home/lyn/Documents/workspace/datasets/outputs/a carpet on the floor/pick/labels/'
    labelDir = '/home/lyn/Documents/workspace/DDRNet.pytorch-main/run/detect/exp99/'
    im_list = os.listdir(imgDir)
    # a = sorted(im_list)
    for i,im in enumerate(im_list):
        imName = im##.split('.')[0] + '.jpg'
        labelName = imName.split('.')[0] + '.png'
        imPath = os.path.join(imgDir,im)
        label_path = os.path.join(labelDir,labelName)
        image = cv2.imread(imPath, cv2.IMREAD_COLOR)

        # print(imPath)
        # print(image.shape)
        left_points = []
        right_points = []
        cv2.namedWindow("img")
        cv2.setMouseCallback("img", mouse_click)
        cv2.imshow('img', image)
        cv2.waitKey(0)
        print(len(left_points),left_points)
        if len(left_points) == 1:
            shutil.copy(label_path,good + imName)
            print('saved good...',good + imName)

        elif len(right_points) == 1:
            shutil.copy(imPath,labelDir + imName)
            print('saved bad...', good + imName)
        # cv2.destroyWindow(im)
        # cv2.destroyAllWindows()
