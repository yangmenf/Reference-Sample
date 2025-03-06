import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# 创建特征重要性及auc值筛选子文件夹
output_folder = '特征重要性及auc值筛选/'
os.makedirs(output_folder, exist_ok=True)

# 使用os.walk遍历"data"子文件夹下的所有文件
for root, dirs, files in os.walk("data"):
    for file in files:
        if ('训练集' in file) and (file.endswith('.xlsx') or file.endswith('.csv')):
            file_path = os.path.join(root, file)

            # Load data
            file_name_without_ext = os.path.splitext(os.path.basename(file_path))[0]
            df = pd.read_excel(file_path) if file_path.endswith('.xlsx') else pd.read_csv(file_path)

            # Standardize
            X = df.iloc[:, 2:].values
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
            y = df['Group']

            # Split data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

            # Train random forest classifier
            rf = RandomForestClassifier(n_estimators=100, random_state=1)
            rf.fit(X_train, y_train)

            # Get feature importances
            importances = rf.feature_importances_

            # Sort features by importance
            sorted_idx = np.argsort(importances)[::-1]
            sorted_features = df.columns[2:][sorted_idx]

            # Initialize results
            results = []

            # Iterate over features
            for i in range(1, len(sorted_features) + 1):
                # Select features
                selected_features = sorted_features[:i]
                selected_idx = [df.columns.get_loc(f) - 2 for f in selected_features]

                # Train classifier
                rf = RandomForestClassifier(n_estimators=100, random_state=1)
                rf.fit(X_train[:, selected_idx], y_train)

                # Calculate AUC
                y_pred = rf.predict_proba(X_test[:, selected_idx])[:, 1]
                auc = roc_auc_score(y_test, y_pred)

                # Save results
                results.append((i, auc, ', '.join(selected_features)))

            # Convert results to DataFrame
            results_df = pd.DataFrame(results, columns=['n_features', 'auc', 'features'])

            # Sort results by AUC
            results_df = results_df.sort_values(by='auc', ascending=False)

            # Save results to Excel
            results_df.to_excel(output_folder + file_name_without_ext + '_auc_results.xlsx', index=False)

            # Plot AUC vs number of features
            plt.figure(figsize=(6, 6))
            sns.lineplot(x='n_features', y='auc', data=results_df, marker='o', dashes=True)
            plt.xlabel('Number of Features')
            plt.ylabel('AUC')
            plt.title('AUC vs Number of Features')
            plt.grid(True)
            plt.savefig(output_folder + file_name_without_ext + '_auc_vs_features.png', dpi=400)
            plt.savefig(output_folder + file_name_without_ext + '_auc_vs_features.pdf', dpi=400)
            plt.close()

            # Get the features of the best AUC
            best_features = results_df.iloc[0]['features'].split(', ')

            # Select the corresponding columns in the original dataframe
            df_best = df[['Group','imageName'] + best_features]

            # Save the dataframe to Excel
            df_best.to_excel(output_folder + file_name_without_ext + '_重要性+auc值筛选-特征矩阵.xlsx', index=False)