import os
import glob
import torch
import pandas as pd
import SimpleITK as sitk
from torch.utils.data import DataLoader
from torchvision.transforms import Compose
from torchvision.models.video import r3d_18
import numpy as np

# 设置数据和特征存储路径
data_dir = r'./images'
mask_dir = r'./masks'
excel_path = r'./深度学习特征.xlsx'

# 加载预训练的3D ResNet模型
model = r3d_18(pretrained=True)

# 修改模型的输入层，使其接受单通道输入
model.stem[0] = torch.nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3), bias=False)

# 去掉最后一层全连接层
model = torch.nn.Sequential(*list(model.children())[:-1])

def extractor(img_path, mask_path, net, use_gpu):
    # 读取影像和掩膜并进行预处理
    img = sitk.ReadImage(img_path)
    mask = sitk.ReadImage(mask_path)
    img_arr = sitk.GetArrayFromImage(img)
    mask_arr = sitk.GetArrayFromImage(mask)
    img_arr = img_arr.astype('float32')
    mask_arr = mask_arr.astype('float32')
    img_arr = img_arr / 255.0
    img_arr = (img_arr - img_arr.mean()) / img_arr.std()
    mask_arr = mask_arr / mask_arr.max()
    img_arr = img_arr * mask_arr  # 将影像乘以掩膜，去除掩膜外的区域
    img_arr = img_arr[np.newaxis, np.newaxis, ...]  # 添加batch和channel维度
    img_tensor = torch.from_numpy(img_arr)
    # 如果使用GPU，则将数据和网络模型移动到GPU上
    if use_gpu:
        img_tensor = img_tensor.cuda()
        net = net.cuda()
    # 使用网络模型提取特征
    y = net(img_tensor).cpu()
    y = torch.squeeze(y)
    y = y.data.numpy()
    return y

# 遍历数据目录中的影像和掩膜文件
extensions = ['nii.gz']
img_files_list = []
mask_files_list = []

for ext in extensions:
    img_files_list.extend(glob.glob(os.path.join(data_dir, '*' + ext)))
    mask_files_list.extend(glob.glob(os.path.join(mask_dir, '*' + ext)))

# 使用os.path.basename和os.path.splitext获取不带扩展名的文件名，并存储在set中
img_files_set = {os.path.splitext(os.path.basename(f))[0] for f in img_files_list}
mask_files_set = {os.path.splitext(os.path.basename(f))[0] for f in mask_files_list}

# 找出两个目录中都存在的文件名
common_files = img_files_set.intersection(mask_files_set)

# 根据共同文件名过滤img_files_list和mask_files_list
img_files_list = [f for f in img_files_list if os.path.splitext(os.path.basename(f))[0] in common_files]
mask_files_list = [f for f in mask_files_list if os.path.splitext(os.path.basename(f))[0] in common_files]

# 提取每个影像和掩膜文件的特征并保存到DataFrame中
features = []
for img_path, mask_path in zip(img_files_list, mask_files_list):
    print("Processing image:", img_path)
    print("Processing mask:", mask_path)
    try:
        file_name = os.path.splitext(os.path.basename(img_path))[0]
        feature = extractor(img_path, mask_path, model, use_gpu=False)
        features.append([file_name, img_path, mask_path] + feature.tolist())
    except Exception as e:
        print("Error processing", img_path, ":", str(e))
        continue

columns = ['file_name', 'img_path', 'mask_path'] + ['feature_{}'.format(i) for i in range(len(features[0])-3)]
df = pd.DataFrame(features, columns=columns)

# 将DataFrame保存为Excel文件
df.to_excel(excel_path, index=False)