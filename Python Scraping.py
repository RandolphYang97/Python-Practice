# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 10:22:22 2020

@author: Randolph
"""

'''
urlopen的主要异常：
1.网页不存在(404 NOT FOUND)
    会抛出HTTPError ,可以直接打印(print)该对象：
    HTTP Error 404: Not Found
2. 服务器不存在
    会抛出URLError

调用不存在的标签：
    返回None对象。同时，调用None.XX会返回AttributeError错误
'''
from urllib.request import urlopen
from bs4 import BeautifulSoup
from urllib.error import HTTPError
from urllib.error import URLError
try:
    html = urlopen('https://www.pythonscraping.com/pages/page1.html')
except HTTPError as e:
    print(e)
except URLError as e:
    print('the server could not be found!')
else:
    print('it worked')
bs = BeautifulSoup(html.read(),'html.parser')
print(bs.h1)

bs = BeautifulSoup(html,'html.parser')
print(bs.h1)
bs = BeautifulSoup(html,'lxml')
print(bs)

'''
try:
    badContent = bs.nonExistingTag.anotherTag
except:
    print('Tag was not found')
else:
    if badContent == NONE :
        print('Tag was not found')
    else:
        print(badContent)
'''
def GetTitle(url):
    try:
        html = urlopen(url)
    except HTTPError as e:
        return None
    try:
        bs = BeautifulSoup(html.read(),'html.parser')
        title = bs.body.h1
    except AttributeError as e:
        return None
    return title
'''
思考代码的总体格局，让代码既可以捕捉异常又容易阅读，这是很重要的。.....通用函数(具有周密的
异常处理功能)会让快速、稳定的抓取网页变得简单易行—
                                               ————《Python网络爬虫权威指南》第一章最后一段
'''