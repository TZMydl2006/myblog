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