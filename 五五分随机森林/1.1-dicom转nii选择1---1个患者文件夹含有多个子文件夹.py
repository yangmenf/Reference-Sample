import os
import pydicom
import dicom2nifti
import dicom2nifti.settings as settings
from tkinter import simpledialog, Tk

settings.disable_validate_slice_increment()
path_in_patients = r"./images-dicom"
path_out_data = r"./images"

# 打印所有最底层的目录
for dirpath, dirnames, filenames in os.walk(path_in_patients):
    if not dirnames:
        print(f"{dirpath} - Number of Files: {len(filenames)}")

# 创建弹窗来获取用户输入的数字
def get_user_input():
    root = Tk()
    root.withdraw()  # 隐藏主窗口
    user_input = simpledialog.askinteger("Input", "Please enter a number:", minvalue=0)
    root.destroy()
    return user_input

threshold = get_user_input()

# 遍历每个底层目录
for dirpath, dirnames, filenames in os.walk(path_in_patients):
    # 只处理没有子目录的目录（即底层目录）
    if not dirnames:
        # 如果该目录中的文件数量大于用户输入的阈值
        if len(filenames) > threshold:
            path_parts = dirpath.split('\\')
            
            # 根据路径部分长度确定输出文件名
            if len(path_parts) >= 3:
                output_filename = path_parts[-3] + "+"+path_parts[-2] + "+"+path_parts[-1] +"+"+ str(len(filenames))
            else:
                output_filename = path_parts[-2] + "+"+ path_parts[-1] + "+"+ str(len(filenames))

            # 转换文件夹
            dicom2nifti.dicom_series_to_nifti(dirpath, os.path.join(path_out_data, output_filename + '.nii.gz'))

print("All eligible directories have been processed.")