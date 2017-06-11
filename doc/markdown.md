# Mrakdown 语法笔记

Mardown这两年备受推崇，业界给的理由是什么简洁呀、高效呀、等等... 但是我觉得这些都不是我使用Markdown的真正原因，说实在的，Markdown的语法真是少的可怜，稍微复杂效果都要使用Html（如果都使用HTML了，为啥还要整Markdown，一脸懵B）。我学习使用Markdown主要是因为把一个Word文档上传的Git仓库好不爽呀，想要看的时候还要下载下来，真是够Low的，我只想打开仓库就能看（要求不过分吧！），另一个原因就是Markdown受宠于Github，所有的开源仓库都使用Markdown做说明文档，我要是整一个Html文档，似乎画风不太一致呀（没办法，承认自己是一个庸俗的人），好了BB这么多，赶紧上干货。

## 多级标题

不得不说Markdown的这个语法设定兼职爽暴了，想要多少级标题，直接在前面增加“#”，貌似目前只支持6级标题，尽管“#”不要钱，也不要太过分呀。

>     语法

```
#      一级标题
##     二级标题
###    三级标题
####   四级标题
#####  五级标题
###### 六级标题
```

>     效果

- #      一级标题
- ##     二级标题
- ###    三级标题
- ####   四级标题
- #####  五级标题
- ###### 六级标题

## 无序列表

无序列表和有序列表也最够简单，远看这俩名字放在一起还真给了我一种错觉，咋地这玩意还会排序不成，近看才发现有序列表就是自己手动加编号（你妹呀，都怪我想多了）。

无序列表使用“*”、“+”或“-”表示，但是推荐一篇文档使用同一种表示方式（你问我为什么，额... 等我知道了再告诉你），注意标记符号与内容之间的空格（为毛浪费三个标记符号，作者你是有多土豪呀）。

>     语法

```
- 列表项
- 列表项
- 列表项
```


>     效果

- 列表项
- 列表项
- 列表项

## 有序列表

前面已经说了，这个有序列表自己加序号（1.）吧（我也懒得说了，瞬间无爱了），注意编号后面的“.”不可缺少，同时与列表项之间需要有空格。

>     语法

```
1. 列表项
2. 列表项
3. 列表项
```

>     效果

1. 列表项
2. 列表项
3. 列表项

## 表格

表格对于Markdown来说是比较累人的，说些起来并不是很方便，也不太麻烦（当我没有）。表格语法的第二行（对齐方式）神一样的存在，大家慢慢体会。

>     语法

```
|行列表 |    名字    |  年龄|
| ---- |:--------: | ---: |
| 1    | LiLei     | 2019 |
| 2    | HanMeiMei |   17 |
```

>     效果

|行列表 |    名字    |  年龄|
| ---- |:--------: | ---: |
| 1    | LiLei     | 2019 |
| 2    | HanMeiMei |   17 |

## 代码块

前文已经说了Markdown非常简陋，所以有这个功能就不错了，别指望太多了。

>     语法

```
\```
#include<iostream>
int main()
{
	std::cout << "Hello world!" << std::endl;
}
\```
（使用时去掉“\”，此处很无奈）
```

>     效果

```
#include<iostream>
int main()
{
	std::cout << "Hello world!" << std::endl;
}
```

### 图片、超链接、图片和超链接

>     语法

```
图片链接：![图片名称](https://...)

超链接方式1：[超链接](https://...)

超链接方式2：[超链接ID]: https://... (Optional title)
	  [超链接][超链接ID]

图片嵌套超链接：[![图片名称](图片地址)](超链接地址)
```

>     效果（效果有误差，正在改正，请稍等）

![Coder](https://github.com/wanghaiyang1930/wanghaiyang1930.github.io/blob/master/common/png/coder.png)

[![Build Status](https://github.com/wanghaiyang1930/wanghaiyang1930.github.io/blob/master/common/svg/build-passing-green.svg)](https://github.com/wanghaiyang1930/wanghaiyang1930.github.io)
[![GoDoc](https://github.com/wanghaiyang1930/wanghaiyang1930.github.io/blob/master/common/svg/docs-wiki-blue.svg)](https://github.com/wanghaiyang1930/wanghaiyang1930.github.io)
[![Wiki](https://github.com/wanghaiyang1930/wanghaiyang1930.github.io/blob/master/common/svg/godoc-reference-blue.svg)](https://github.com/wanghaiyang1930/wanghaiyang1930.github.io)

- [Download Binaries for different platforms](https://github.com/wanghaiyang1930/wanghaiyang1930.github.io)
- [SeaweedFS Mailing List](https://github.com/wanghaiyang1930/wanghaiyang1930.github.io)
- [Wiki Documentation](https://github.com/wanghaiyang1930/wanghaiyang1930.github.io)

## 其他

>     语法

```
黑体：（不可以有空格）
	**内容**
	__内容__

斜体：
	*内容*
	_内容_

自动链接：<https://example.com>

段落：>内容

特殊段落：>     内容（5个空格）

分隔符：***

转义字符：\
```

>     效果

**内容**

__内容__

*内容*

_内容_

>内容

>     内容

***

\# 内容

## 后记

为毛Markdown排版出来的中文这么难看呢（哭晕在厕所）。

