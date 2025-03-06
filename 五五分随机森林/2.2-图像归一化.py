import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# 指定nii.gz文件所在的文件夹路径
dir_path = r'./images/N4'





















# 获取文件夹中所有nii.gz文件的路径列表
file_list = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.nii.gz')]

# 依次读取和归一化每个nii.gz文件
for file_path in file_list:
    # 读取nii.gz格式的图像
    img = nib.load(file_path)
    data = img.get_fdata()

    # 计算均值和方差
    x_mean = np.mean(data)
    vari = np.sqrt((np.sum((data-x_mean)**2))/(data.size))

    # 归一化
    norm = (data - x_mean) / vari

    # 保存归一化后的图像
    norm_img = nib.Nifti1Image(norm, img.affine)
    # 保存归一化后的图像
    #save_path = os.path.join(dir_path, 'normalized')
    save_path = r'./images/N4/nor'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    nib.save(norm_img, os.path.join(save_path, os.path.basename(file_path).replace('.nii.gz', '.nii.gz')))
       

    # 画出归一化前后的图像
    plt.subplot(121), plt.imshow(data[:,:,10], 'gray'), plt.title('raw')
    plt.axis('off')
    plt.subplot(122), plt.imshow(norm[:,:,10], 'gray'), plt.title('normalized')
    plt.axis('off')
    plt.show()
    # 打印归一化前后的灰度值数值
    print('raw data: mean=%.4f, std=%.4f' % (np.mean(data), np.std(data)))
    print('normalized data: mean=%.4f, std=%.4f' % (np.mean(norm), np.std(norm)))