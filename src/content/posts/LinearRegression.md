---
title: LinearRegression
published: 2025-07-12
description: First lesson in Machine Learning
tags: [Machine Learning, Python]
category: ML
draft: false
---
## 摘要
ML系列的内容来自小学期机器学习的上机实验，主要讲解代码，顺便介绍Python的库和常用方法。

## 波士顿房价预测问题
首先需要获取数据集：
```python
import os
```
引入文件处理库，里面有文件操作的方法。
```python
dirName = "./data" ## 数据集所在目录
dirName_feats = os.path.join(dirName,'housing_features.csv')
dirName_targs = os.path.join(dirName,'housing_target.csv')
dirName_names = os.path.join(dirName,'housing_names.csv')
```
`os.path.join`方法用于将目录名和文件名连接起来，形成完整的文件路径。

接下来是数据处理环节，需要用到pandas库：
```python
import pandas as pd

df_housing = pd.read_csv(dirName_feats)

df_housing.describe()

df_housing.head()
```
`pandas.read_csv`方法用于从csv文件中读取数据，返回一个DataFrame对象。

`DataFrame`对象是一个二维数据结构，可以存储多个数据列，每个数据列是一个Series对象。

`Series`对象是一个一维数据结构，用于存储一个数据列。

`DataFrame.describe`方法用于获取数据集的描述信息，返回一个DataFrame对象。

`DataFrame.head`方法用于获取数据集的前几行，返回一个DataFrame对象。

分离输入和输出数据：
```python
import numpy as np

X=np.genfromtxt(dirName_feats,delimiter=',',skip_header=1)
y=np.genfromtxt(dirName_targs,delimiter=',',skip_header=1)
names=np.genfromtxt(dirName_names, dtype='str',skip_header=1)
```
`np.genfromtxt`方法用于从csv文件中读取数据，返回一个NumPy数组。

`np.genfromtxt`方法中的参数`delimiter`用于指定数据列的分隔符，`skip_header`用于指定跳过的行数。

`np.genfromtxt`方法中的参数`dtype`用于指定数据类型，`str`表示字符串类型。

接下来是数据预处理：
```python
np.random.seed(42)

X_new = X.copy()
mask = np.random.randint(0, 2, size=X.shape).astype(bool)
X_new[mask] = np.nan
```
模拟数据缺失的场景。

处理缺失数据：
```python
from sklearn.impute import SimpleImputer

imp = SimpleImputer(strategy='mean')
X_replace_with_mean = imp.fit_transform(X_new)
```
`SimpleImputer`类用于填充缺失的数据。`strategy`参数指定填充策略，可选值有`mean`、`median`、`most_frequent`、`constant`。

`fit_transform`方法用于训练模型并填充缺失的数据。

分离训练集和测试集：
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
`train_test_split`方法用于将数据集分离为训练集和测试集。`test_size`参数指定测试集的比例，`random_state`参数指定随机数种子，保证每次分离的结果相同。

数据集的标准化：
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```
`StandardScaler`类用于对数据进行标准化处理。

`fit_transform`方法用于训练模型并转换训练集数据，`transform`方法用于转换测试集数据。

多特征线性回归：
```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)

y_lr = lr.predict(X_test)

mse_lr = mean_squared_error(y_test, y_lr)
mae_lr = mean_absolute_error(y_test, y_lr)
```
`LinearRegression`类用于线性回归模型。

`fit`方法用于训练模型，`predict`方法用于预测测试集数据。

`mean_squared_error`方法用于计算均方误差，`mean_absolute_error`方法用于计算平均绝对误差。

单特征线性回归：
```python
n_sample, n_feature = X_train.shape

mse_lr_per_feature = []
mae_lr_per_feature = []

for i in range(n_feature):
    lr = LinearRegression()
    lr.fit(np.reshape(X_train[:, i], [n_sample, 1]), y_train)
    y_lr = lr.predict(np.reshape(X_test[:, i], [X_test.shape[0], 1]))

    mse_lr_per_feature.append(np.sqrt(mean_squared_error(y_test, y_lr)))
    mae_lr_per_feature.append(mean_absolute_error(y_test, y_lr))

errors = pd.DataFrame.from_dict({'MAE': mae_lr_per_feature,
                                 'MSE': mse_lr_per_feature},
                                 orient='index', columns=names)

errors
```
`.shape`属性用于获取数组各维度的大小。

`reshape()`方法用于改变数组的形状。

`append()`方法用于向数组末尾添加元素。

`pd.DataFrame.from_dict()`方法用于将字典转换为DataFrame对象。