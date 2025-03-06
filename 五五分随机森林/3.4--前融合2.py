import os
import shutil
import pandas as pd
from itertools import combinations

def common_suffix(files):
    """Find the longest common suffix amongst a list of files."""
    reversed_files = [''.join(reversed(f)) for f in files]
    reversed_suffix = common_prefix(reversed_files)
    return ''.join(reversed(reversed_suffix))

def common_prefix(strings):
    """Find the longest common prefix string amongst a list of strings."""
    if not strings:
        return ""
    shortest_str = min(strings, key=len)
    for i, char in enumerate(shortest_str):
        for other in strings:
            if other[i] != char:
                return shortest_str[:i]
    return shortest_str

folder_path = r"./data"
files_to_delete = []

for subdir, _, files in os.walk(folder_path):
    for file in files:
        if "验证集" in file:
            source_path = os.path.join(subdir, file)
            dest_path = os.path.join(folder_path, file)
            shutil.copy2(source_path, dest_path)
            files_to_delete.append(dest_path)

all_files = [f for f in os.listdir(folder_path) if ("验证集" in f) and (f.endswith('.csv') or f.endswith('.xlsx'))]
common_suffix = common_suffix(all_files)

for r in range(2, len(all_files) + 1):
    for subset in combinations(all_files, r):
        first_file = subset[0]
        df_combined = pd.read_excel(os.path.join(folder_path, first_file)) if first_file.endswith('.xlsx') else pd.read_csv(os.path.join(folder_path, first_file))
        
        for file in subset[1:]:
            df = pd.read_excel(os.path.join(folder_path, file)) if file.endswith('.xlsx') else pd.read_csv(os.path.join(folder_path, file))
            df_combined = pd.concat([df_combined, df.iloc[:, 2:]], axis=1)
        
        unique_parts = [f.split('-')[1] for f in subset]
        output_name = "+".join(unique_parts) + common_suffix + '.csv'
        df_combined.to_csv(os.path.join(folder_path, output_name), index=False)

for file_path in files_to_delete:
    os.remove(file_path)

print("Files combined and cleaned up successfully!")