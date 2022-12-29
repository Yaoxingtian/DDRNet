import os,cv2
from collections import Counter
root = '/home/yao/Documents/hypersim/floor'
savePath = '/home/yao/Documents/hypersim/resize/labels'
for img in os.listdir(root):
    imfile = os.path.join(root,img)
    im_ = cv2.imread(imfile,cv2.IMREAD_GRAYSCALE)
    im = cv2.resize(im_,(640,480),cv2.INTER_NEAREST)
    tem = im.copy()
    im[tem > 50] = 100
    im[tem <= 50] = 1
    # k = [i for j in im for i in j]  # 将二维数组转一维数组
    # count = Counter(k)
    # print(im.shape, count)
    cv2.imwrite('{}/{}'.format(savePath,img),im)
