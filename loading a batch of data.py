# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 22:19:05 2020

@author: Administrator
"""
import pd
import glob,os
input_path='G:\\'
# 搜索所有 .csv后缀的文件并读取其路径
allfile=glob.glob(os.path.join(input_path,'*.csv'))
train_data_frames=[]
 #读取所有的数据， 请将要读取的列索引输入usecols
for file in allfile:
    train_data_frames.append(pd.read_csv(file,index_col='是否行索引',usecols='指定列'))