import os,shutil

train_path = '/home/yao/Documents/guangzhou_ddr_analyse/leftImg8bit/val/all'
labelDir = '/home/yao/Documents/guangzhou_ddr_analyse/gtFine/erode'
save_path = '/home/yao/Documents/guangzhou_ddr_analyse/gtFine/all'
files_list = os.listdir(train_path)

for file in files_list:
    file_path = os.path.join(train_path,file),
    labelName = file.replace('jpg','png')
    labelPath = os.path.join(labelDir,labelName)
    savePath = os.path.join(save_path,labelName)

    shutil.copy(labelPath,savePath)
    # imgs_list = os.listdir(file_path)
    # for img in imgs_list:
    #     if img.endswith('color.png'):
    #         img_color = img
    #         img_color_path = os.path.join(train_path,file,img_color)
    #
    #         save_file = os.path.join(save_path,img_color)
    #         shutil.copy(img_color_path,save_file)
