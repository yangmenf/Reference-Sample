#批量#######3
import os
import SimpleITK as sitk
# 输入文件夹路径
input_folder = r"./images"
# 输出文件夹路径（在输入文件夹下新建一个名为output的文件夹）
output_folder = os.path.join(input_folder, "N4")
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
# 循环处理所有nii.gz文件
for filename in os.listdir(input_folder):
    if filename.endswith(".nii.gz"):
        # 构造输入文件路径和输出文件路径
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        # 加载nii.gz文件
        input_image = sitk.ReadImage(input_path)
        # 进行偏置场校正
        input_image = sitk.Cast(input_image, sitk.sitkFloat32)
        mask_image = sitk.OtsuThreshold(input_image, 0, 1, 200)
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        output_image = corrector.Execute(input_image, mask_image)
        output_image = sitk.Cast(output_image, sitk.sitkInt16)
        # 保存输出图像
        sitk.WriteImage(output_image, output_path)