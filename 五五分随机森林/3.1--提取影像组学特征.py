############终极成功版##############
import radiomics
from radiomics import featureextractor
import pandas as pd
import os
import SimpleITK as sitk
# 定义特征提取器参数
settings = {}
settings['binWidth'] = int(input("请输入binWidth的值："))
settings['sigma'] = [int(i) for i in input("请输入sigma的值（多个值用空格分隔）：").split()]
settings['resampledPixelSpacing'] = [int(i) for i in input("请输入resampledPixelSpacing的值（多个值用空格分隔）：").split()]
# 将设置添加到参数字典中
params = {'binWidth': settings['binWidth'],  'sigma': settings['sigma'], 'resampledPixelSpacing': settings['resampledPixelSpacing']}
params.update(settings)
# 定义特征提取器
extractor = featureextractor.RadiomicsFeatureExtractor(**params)
image_types = {}
for imageType in ["Original", "LoG", "Wavelet"]:
    choice = input(f"是否要添加{imageType}特征，输入y/n：").lower()
    if choice == 'y':
        image_types[imageType] = {}
    
# 启用选择的图像类型
extractor.enableImageTypes(**image_types)
# 定义数据目录
dataDir = input("请输入数据目录的路径：")
# 定义结果数据框
df = pd.DataFrame()
# 遍历图像和掩膜
# 遍历图像和掩膜
for imageName in os.listdir(os.path.join(dataDir, "images")):
    for maskName in os.listdir(os.path.join(dataDir, "masks")):
        # 提取文件名
        imageFileName = os.path.splitext(imageName)[0]
        maskFileName = os.path.splitext(maskName)[0]
        # 检查文件名是否匹配
        if imageFileName == maskFileName:
            imagePath = os.path.join(dataDir, "images", imageName)
            maskPath = os.path.join(dataDir, "masks", maskName)
            # 打印正在处理的文件
            print(f"正在处理图像文件 {imageFileName} 和掩膜文件 {maskFileName}")
            # 执行特征提取操作
            try:
                featureVector = extractor.execute(imagePath, maskPath)
                # 将特征保存到数据框
                df_add = pd.DataFrame.from_dict(featureVector.values()).T
                df_add.columns = featureVector.keys()
                df_add.insert(0, 'imageName', imageFileName)
                df = pd.concat([df, df_add])
            except Exception as e:
                print(f"在处理图像文件 {imageFileName} 和掩膜文件 {maskFileName} 时遇到错误，已跳过这个文件。错误详情: {e}")
                continue

# 将结果保存到Excel文件中
result_file = os.path.join(dataDir, '影像组学特征.xlsx')
df.to_excel(result_file, index=False)
print("结果已保存到文件：", result_file)


