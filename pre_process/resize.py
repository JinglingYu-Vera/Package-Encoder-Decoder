import os
import cv2
 
def resize_img(DATADIR, data_k, img_size):
    w = img_size[0]
    h = img_size[1]
    path = os.path.join(DATADIR, data_k)
    #返回path路径下所有文件的名字，以及文件夹的名字，
    img_list = os.listdir(path)
 
    for i in img_list:
        print(i)
        if i.endswith('.jpg'):
            # 调用cv2.imread读入图片，读入格式为IMREAD_COLOR
            img_array = cv2.imread((path + '/' + i), cv2.IMREAD_COLOR)
            # 调用cv2.resize函数resize图片
            new_array = cv2.resize(img_array, (w, h), interpolation=cv2.INTER_CUBIC)
            img_name = str(i)
            '''生成图片存储的目标路径'''
            save_path = path + '_new/'
            if os.path.exists(save_path):
                print(i)
                '''调用cv.2的imwrite函数保存图片'''
                save_img=save_path+img_name
                cv2.imwrite(save_img, new_array)
            else:
                os.mkdir(save_path)
                save_img = save_path + img_name
                cv2.imwrite(save_img, new_array)
 
 
if __name__ == '__main__':
    #设置图片路径
    DATADIR = "./data/"
    data_k = 'shampoo'
    '''设置目标像素大小，此处设为128 * 256'''
    img_size = [512, 512]
    resize_img(DATADIR, data_k, img_size)