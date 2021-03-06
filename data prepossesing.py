import sklearn as sk
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn import datasets
import os 
import numpy as np
#%%
#更改工作路径
os.chdir("H:\\Python learning\\dataprepossesing")
##读取数据
Video_Game= pd.read_csv('data/videogame.csv', encoding = "ISO-8859-1")
np.sum(Video_Game)
Video_Game.iloc[0:7]
#%%
#除去重复元素且排序， return_index返回对应独特元素首次出现的位置的序号 ,return_inverse在原列表中标注元素在新列表中的位置
Game_Type,th,inver_th=np.unique(Video_Game['Genre'],return_index=True,return_inverse=True)
Game_Type
#生成分类器
gle=LabelEncoder()
#将文字转为数字标签
type_labels=gle.fit_transform(Video_Game['Genre'])
'''
将数字标签转回文字标签
type_labels=gle.inverse_transform(type_labels)
'''
## enumerate 为迭代的对象添加序号，从0开始。输出对应的遍历对象和对象对应的序号
'''
 for index, label in enumerate(fle.classes_):
    print(index)
    print(label)
    '''
#遍历分类器创建字典（用于检查）
type_dic={index:label for index, label in enumerate(gle.classes_)} 
type_dic
Video_Game['TypeLabel']=type_labels

#%%
'''
One-hot Encoding
'''
from sklearn.preprocessing import OneHotEncoder
ile=OneHotEncoder()
iris=datasets.load_iris()
iris_data=pd.DataFrame(iris.data)
iris_target=np.array(iris.target).reshape(-1,1)
iris_data['type']=iris_target
gen_zero_or_one=ile.fit_transform(iris_target).toarray()
'''
如果有三种特征，也只会生成0或1
'''
#%%
#虚拟变量   避免虚拟变量陷进用drop_first
gen_dummy_features=pd.get_dummies(iris_data['type'],drop_first=True)
#与原数据合并
iris_dummy=pd.concat([iris_data,gen_dummy_features],axis=1)
#%%
iris_data[[1,2]].head()
iris_data.describe()
iris_data.sum()
#%% 多项式特征
from sklearn.preprocessing import PolynomialFeatures
pf = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
irispoly=pf.fit_transform(np.array(iris_data[1]).reshape(-1,1))
feature_names=iris.feature_names
irispoly=pd.DataFrame(irispoly,columns=[feature_names[1],feature_names[1]+'^2'])
#%% bin条
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
irispoly[feature_names[1]].hist(color='#A9C5D3')
ax.set_title(feature_names[1]+' Histogram', fontsize=12)
ax.set_xlabel(feature_names[1], fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
#%%  分位数

quantile_list = [0, .25, .5, .75, 1.]
quantiles=irispoly[feature_names[1]].quantile(quantile_list)

fig, ax = plt.subplots()

for quantile in quantiles:
    qvl = plt.axvline(quantile, color='b')
ax.legend([qvl], ['Quantiles'], fontsize=10)
irispoly[feature_names[1]].hist(color='#A9C5D3')
ax.set_title(feature_names[1]+' Histogram', fontsize=12)
ax.set_xlabel(feature_names[1], fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
quantile_label=['Q1','Q2','Q3','Q4']
irispoly['quantiles label of sepal width ']=pd.qcut(irispoly[feature_names[1]],q=quantile_list,labels=quantile_label)
#%% Z-score
from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
iris_data=pd.DataFrame(iris.data,columns=feature_names)
ss_iris=ss.fit_transform(iris_data)
#%% 归一化处理
from sklearn.preprocessing import  MinMaxScaler
mms = MinMaxScaler()
mms_iris=mms.fit_transform(iris_data)
#%%   鲁棒型估计 （xi-中位数）/四分位距离 ，排除了outlier
from sklearn.preprocessing import RobustScaler
rs=RobustScaler()
rs_iris=rs.fit_transform(iris_data)
'''用数学式表达：
quartiles = np.percentile(iris_data, (25., 75.))
iqr = quartiles[1] - quartiles[0]
rs_iris=(iris_data - np.median(iris_data)) / iqr
'''
#%%  图像特征处理
im=datasets.load_sample_images()
image=im.images
flower=image[1]
tower=image[0]
#打印图片
fig = plt.figure(1)
plt.axis('off')
plt.imshow(tower)
fig = plt.figure(2)
plt.axis('off')
plt.imshow(flower)
 # R G B
flower_r=flower.copy()
flower_r[:,:,1]=0
flower_r[:,:,2]=0 #G b为0
plt.figure(3)
plt.imshow(flower_r)
# 灰度图
plt.figure(4)
plt.subplot(2,2,1)
plt.imshow(tower,cmap="gray")
plt.subplot(2,2,2)
plt.imshow(tower)
#%%  文本预处理
