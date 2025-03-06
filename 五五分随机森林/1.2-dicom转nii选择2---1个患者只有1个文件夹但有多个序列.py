import os
import pydicom
import dicom2nifti
import shutil
import dicom2nifti.settings as settings
settings.disable_validate_slice_increment()
base_dicom_dir = r'./images-dicom2'
base_output_dir = r'./images'
temp_dir = r'./images/temp'
















if not os.path.exists(base_output_dir):
    os.makedirs(base_output_dir)
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)
# 遍历总路径下的所有子文件夹
for subfolder in os.listdir(base_dicom_dir):
    dicom_dir = os.path.join(base_dicom_dir, subfolder)
    if not os.path.isdir(dicom_dir):
        continue

    try:
        dicom_files = [f for f in os.listdir(dicom_dir) if os.path.isfile(os.path.join(dicom_dir, f))]

        # 使用字典存储每个SeriesInstanceUID对应的文件列表
        series_files = {}

        for dicom_file in dicom_files:
            ds = pydicom.dcmread(os.path.join(dicom_dir, dicom_file))

            if "SeriesInstanceUID" in ds:
                uid = ds.SeriesInstanceUID
                if uid not in series_files:
                    series_files[uid] = []
                series_files[uid].append(os.path.join(dicom_dir, dicom_file))

        for uid, files in series_files.items():
            # 创建临时文件夹
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)

            # 复制所有文件到临时文件夹
            for file in files:
                shutil.copy(file, temp_dir)

            # 将临时文件夹中的文件转换为NIfTI
            dicom2nifti.convert_directory(temp_dir, temp_dir, compression=True, reorient=True)

            # 为输出的NIfTI文件创建文件名，包括原始文件夹名称和深度信息
            for file_name in os.listdir(temp_dir):
                if file_name.endswith('.nii.gz'):
                    new_name = f"{os.path.basename(dicom_dir)}_Depth_{len(files)}_{file_name}"
                    shutil.move(os.path.join(temp_dir, file_name), os.path.join(base_output_dir, new_name))

            print(f"Converted SeriesInstanceUID: {uid} with {len(files)} files. Output to {base_output_dir}")

            # 清空临时文件夹
            shutil.rmtree(temp_dir)
    
    except Exception as e:
        print(f"Error processing folder '{dicom_dir}': {e}")