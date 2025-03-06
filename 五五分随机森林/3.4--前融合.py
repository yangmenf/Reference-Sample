
import os
import shutil
import pandas as pd
from itertools import combinations

# ① 遍历./data目录下的所有子文件夹，并复制文件名中带有“验证集”三个字符的文件到./data目录下。
folder_path = r"./data"
files_to_delete = []  # 保存要删除的文件路径

for subdir, _, files in os.walk(folder_path):
    for file in files:
        if "验证集" in file:
            source_path = os.path.join(subdir, file)
            dest_path = os.path.join(folder_path, file)
            shutil.copy2(source_path, dest_path)
            files_to_delete.append(dest_path)  # 记录复制到./data的文件路径

# ② 对./data目录下的文件名中带有“验证集”三个字符的文件执行您提供的合并代码操作。
all_files = [f for f in os.listdir(folder_path) if ("验证集" in f) and (f.endswith('.csv') or f.endswith('.xlsx'))]

for r in range(2, len(all_files)+1):
    for subset in combinations(all_files, r):
        first_file = subset[0]
        df_combined = pd.read_excel(os.path.join(folder_path, first_file)) if first_file.endswith('.xlsx') else pd.read_csv(os.path.join(folder_path, first_file))
        
        for file in subset[1:]:
            df = pd.read_excel(os.path.join(folder_path, file)) if file.endswith('.xlsx') else pd.read_csv(os.path.join(folder_path, file))
            df_combined = pd.concat([df_combined, df.iloc[:, 2:]], axis=1)
        
        output_name = "+".join([os.path.splitext(f)[0] for f in subset]) + '.csv'
        df_combined.to_csv(os.path.join(folder_path, output_name), index=False)

# ③ 删除原先复制到./data目录下的带有“验证集”三个字符的文件
for file_path in files_to_delete:
    os.remove(file_path)

print("Files combined and cleaned up successfully!")