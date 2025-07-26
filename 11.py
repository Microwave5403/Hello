import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")

print(train.info())

#将房屋代码转换为str
train["MSSubClass"] = train["MSSubClass"].astype(str)
test["MSSubClass"] = test["MSSubClass"].astype(str)

drop_columns = []
columns = train.columns
for column in columns:
    if train[column].count() >1350 and train[column].count() < train.shape[0]:
        drop_columns.append(column)

drop_columns = []
columns = test.columns
for column in columns:
    if test[column].count() >1350 and test[column].count() < test.shape[0]:
        drop_columns.append(column)


train.dropna(subset=drop_columns, inplace=True)
#test.dropna(subset=drop_columns, inplace=True)
# 相反，对测试集中的缺失值进行填充
for column in drop_columns:
    if test[column].dtype == 'object':
        test[column].fillna("Unknown", inplace=True)
    else:
        test[column].fillna(test[column].median(), inplace=True)


train["LotFrontage"] = train["LotFrontage"].fillna(train["LotFrontage"].median())
train["MasVnrType"] = train["MasVnrType"].fillna("Default")
train["FireplaceQu"] = train["FireplaceQu"].fillna("Default")
train["PoolQC"] = train["PoolQC"].fillna("Default")
train["Fence"] = train["Fence"].fillna("Default")
train["MiscFeature"] = train["MiscFeature"].fillna("MiscFeature")

test["LotFrontage"] = test["LotFrontage"].fillna(test["LotFrontage"].median())
test["MasVnrType"] = test["MasVnrType"].fillna("Default")
test["FireplaceQu"] = test["FireplaceQu"].fillna("Default")
test["PoolQC"] = test["PoolQC"].fillna("Default")
test["Fence"] = test["Fence"].fillna("Default")
test["MiscFeature"] = test["MiscFeature"].fillna("MiscFeature")

train.drop(columns=["Alley"],inplace=True)
test.drop(columns=["Alley"],inplace=True)

encoded_train = train.copy()
encoded_test = test.copy()

# 对每个分类特征列进行独热编码
category_columns = train.select_dtypes(include=['object']).columns.tolist()
for column in category_columns:
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(train[[column]])

    # 对该列进行编码
    encoded_cols = encoder.transform(train[[column]])
    encoded_cols_2 = encoder.transform(test[[column]])

    # 创建新列名
    categories = encoder.categories_[0]
    column_names = [f"{column}_{category}" for category in categories]

    # 验证列名数量与编码后特征数量是否一致
    if len(column_names) != encoded_cols.shape[1]:
        print(f"警告: 列 {column} 的列名数量({len(column_names)})与编码特征数量({encoded_cols.shape[1]})不匹配")
        # 调整列名以匹配特征数
        column_names = [f"{column}_cat{i}" for i in range(encoded_cols.shape[1])]

    # 创建临时DataFrame并将编码后的列添加到结果中
    temp_df = pd.DataFrame(encoded_cols, columns=column_names, index=train.index)
    temp_df_2 = pd.DataFrame(encoded_cols_2, columns=column_names, index=test.index)


    # 添加独热编码列到结果数据集
    encoded_train = pd.concat([encoded_train, temp_df], axis=1)
    encoded_test = pd.concat([encoded_test, temp_df_2], axis=1)

    # 可选：删除原始分类列
    encoded_train.drop(columns=[column], inplace=True)
    encoded_test.drop(columns=[column], inplace=True)

X_train = encoded_train.drop(columns=["SalePrice"])
Y_train = encoded_train["SalePrice"].values
X_ask = encoded_test.copy()

scaler_X = StandardScaler()
scaler_Y = StandardScaler()
scaler_test = StandardScaler()
X_train = scaler_X.fit_transform(X_train)
Y_train = scaler_Y.fit_transform(Y_train.reshape(-1, 1)).flatten()
X_ask = scaler_X.transform(X_ask)
# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)


#print(train.info())
#print(train["MiscFeature"].unique())


from sklearn.svm import SVR

# 创建 SVM 模型
model = SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1)
model.fit(x_train, y_train)
# 进行预测
y_pred = model.predict(x_test)

#计算误差
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
y_pred = scaler_Y.inverse_transform(y_pred.reshape(-1, 1)).flatten()
y_test = scaler_Y.inverse_transform(y_test.reshape(-1, 1)).flatten()

# 计算均方误差
mse = mean_squared_error(y_test, y_pred)
print(f"均方误差 (MSE): {mse:.2f}")

# 计算均方根误差
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"均方根误差 (RMSE): {rmse:.2f}")

# 计算平均绝对误差
mae = mean_absolute_error(y_test, y_pred)
print(f"平均绝对误差 (MAE): {mae:.2f}")

# 计算 R² 分数
r2 = r2_score(y_test, y_pred)
print(f"R² 分数: {r2:.2f}")

answer = model.predict(X_ask)
answer_actual = scaler_Y.inverse_transform(answer.reshape(-1, 1)).flatten()

submission = pd.DataFrame({
    'Id': test["Id"],  # 假设ID是从1开始的索引
    'SalePrice': answer_actual
})

submission.to_csv('submission.csv', index=False)