python

#### 安装

Python源码编译安装之前需要安装依赖，否则将导致pip3无法使用：
- yum install zlib-devel
- yum install openssl-devel
- yum install bzip2-devel
- yum install tkinter
- yum install tk-devel

#### 绘图库

Python的常用绘图库主要有两个：
- plotly      pip3 install plotly
- matplotlib  pip3 install matplotlib

> matplotlib

```
	import matplotlib.pyplot as plt
	xs = numpy.array([1,2,3,4,5,6,7,8,9,10])
	ys = numpy.array([1,2,3,4,5,6,7,8,9,10])

	plt.scatter(xs, ys)

	plt.savefig("a.png", format="png")
	plt.show()
```

#### 字符串操作

```
	str = "str1" + "str2" 连接
	n = str.index( "1" ) 查找
	cmp( str1, str2 ) 比较
	cmp( str[0:n], str2[0:n] ) 比较
	str = str.upper() 大写
	str = str.lower() 小写
	spl = str[0:-1] 切片（值操作）
	for c in str : 遍历

	list = str.split( " " ) 分割

	delimiter = ","
	list = ["a", "b", "c", "d"]
	delimiter.join( list ) 连接
```

#### 目录操作

```
	import os

	dir = "/home"
	file = "/home/file.txt"

	os.path.exists( dir )   True/False     判断目录存在
	os.path.split( dir )    ("/", "home")  切分目录
	os.path.splitext( dir ) ("/home", "")  切分后缀名
	os.makedirs( path )                    创建多层目录
	os.mkdir( path )                       创建单层目录

	os.path.exists( file )    True/False               判断文件存在
	os.path.split( file )     ("/home", "file.txt")    切分目录
	os.path.splitext( file )  ("/home/file", ".txt")   切分后缀名
	os.path.abspath( file )   /home/file.txt           绝对路径
	os.path.basename( file )  file.txt                 文件全名
	os.path.dirname( file )   /home                    存储路径

	os.path.join( "/home", "file.txt" ) /home/file.txt 生成
```

#### 字典操作

```
	for key in dict :
		print( key, dict[key] )

	for key in dirc.keys() :
		print( key, dict[key] )

	for value in dict.values() :
		print( value )

	for kv in dict.items() :
		print( kv[0], kv[1] )

	for key, value in dict.items() :
		print( key, value )

	dict.clear()
	dict.copy()
	dict.get( key, default=None)
	dict.pop( key, None )
	dict.has_key( key ) True/False
```

#### Lambda 表达式
	
```
	l = lambda x : x**2 # x: 输入参数， x**2：输出参数

	[(1, 1), (3, 3), (2, 2)].sort( key=lambda x:x[0]，reverse=True ) #按照第一项排序
```

#### 文件读写

> 行存储

```
	file = open( name, "r" )
	while True :
		line = file.readline()
		if None == line :
			break
		#Work
	file.close()

	file = open( name, "w" )
	while True :
		file.writelines( line )
	file.close()	
```
