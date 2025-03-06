import pandas as pd
import numpy as np
import os
import glob

def split_excel(input_files, ratio):
    all_data = [pd.read_excel(file) if file.endswith('.xlsx') else pd.read_csv(file) for file in input_files] # 根据文件扩展名选择读取方式
    min_len = min([len(df) for df in all_data])  # 找出最小的DataFrame长度
    
    # 获取随机排列的索引
    index_array = np.arange(min_len)
    np.random.shuffle(index_array)

    split_point = int(min_len * ratio)  # 计算分割点

    selected_indices = index_array[:split_point]
    remaining_indices = index_array[split_point:]

    # 进行数据分割并保存
    for i, file in enumerate(input_files):
        df = all_data[i]
        df1 = df.iloc[selected_indices]
        df2 = df.iloc[remaining_indices]

        base_name = os.path.basename(file)
        base_name_without_ext = os.path.splitext(base_name)[0]
        dir_name = os.path.dirname(file)

        if file.endswith('.xlsx'):
            df1.to_excel(os.path.join(dir_name, f"{base_name_without_ext}-训练集.xlsx"), index=False)
            df2.to_excel(os.path.join(dir_name, f"{base_name_without_ext}-验证集.xlsx"), index=False)
        else:
            df1.to_csv(os.path.join(dir_name, f"{base_name_without_ext}-训练集.csv"), index=False)
            df2.to_csv(os.path.join(dir_name, f"{base_name_without_ext}-验证集.csv"), index=False)

# 获取用户输入的分割比例
ratio = float(input("请输入分割比例（例如0.7）："))

# 获取当前路径下所有的Excel和CSV文件路径
input_files = glob.glob("./data/*.xlsx") + glob.glob("./data/*.csv")

# 进行分割
split_excel(input_files, ratio)