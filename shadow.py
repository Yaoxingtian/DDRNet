from torchvision import transforms
from RandomShadowsHighlights import RandomShadows
from PIL import Image

def shadow():
    transform = transforms.Compose([
                        # transforms.RandomHorizontalFlip(),
                        RandomShadows(p=0.8, high_ratio=(1,2), low_ratio=(0,1), left_low_ratio=(0.4,0.8),
                                     left_high_ratio=(0,0.3), right_low_ratio=(0.4,0.8), right_high_ratio=(0,0.3)),
                        # transforms.ToTensor(),
                        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                        ])
    return transform
import glob,cv2
import numpy as np
if __name__ == '__main__':

    imDir = '/home/yao/Documents/DDRNet.pytorch-main/datasets/part4'
    for img in glob.glob(imDir + '/*'):
        img = Image.open(img)
        img = shadow()(img)
        img = np.array(img)
        cv2.imshow('im',img)
        cv2.waitKey(0)
        # im  = transforms.ToPILImage(img)
        # img.show('im')

