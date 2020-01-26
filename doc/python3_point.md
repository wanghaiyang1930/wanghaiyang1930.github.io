
### 文件读写

```
f = open("file.name", encoding="utf-8")
print(f.read()) # 全部读写
f.close()
```

```
f = open("file.name", encoding="utf-8")
data = f.readline()
while data :
	print(data)
	data = f.readline()
f.close()
```

```
f = open("file.name", encoding="utf-8")
data = f.readlines()
for line in data :
	print(line.strip())
f.close()
```

```
f = open("file.name", "w")
f.write(data + '"\n")
f.flush()
f.close()
```

```
with open("file.name", encoding="utf-8") as f :
	print(f.read())
```

```
with open("file.name", encoding="utf-8") as f1,
     open("file.name", encoding="utf-8") as f2 :
     for i in f1 : 
     	j = f2.readline()
     	k = f1.readline()
     	print(i, j, k)
```

### 数据结构

#### List

elements = [1, 2, 3]

- 全局函数

```
	len(list) #列表长度
	max(list) #返回列表中元素最大值
	min(list) #返回列表中元素最小值
	list(seq) #类型转换
```

- 成员函数

```
	.append(obj) # 追加元素
	.count(obj) # 统计列表元素个数
	.extend(seq) # 扩展列表π
	.insert(index, obj) # 插入元素
	.pop(index=-1) # 移除列表中元素（默认最后一个元素）
	.remove(obj) # 移除第一个匹配值的元素
	.reverse() # 反转列表
	.sort(key=None, reverse=False) # 排序列表
	.clear() # 清空列表
	.copy() # 复制列表，深拷贝 or 浅拷贝？
```

- 遍历元素

```
for item in list:
	print(item)

for item in enumerate(list, start=0):
	print(item)

for item in iter(list):
	print(item)

for item in range(0, len(list), 1) :
	print(item)
```

#### dict

dirctory = {'Name': 'Name', 'Age': 7, 'Class': 'Second'}

- 全局函数

```
	len(dict) # 字典长度
	str(dict) # 字典字符化
	type(dict) # 字典类型
```

- 内置函数

```
	clear() # 
	copy() #
	items()
	keys()
	fromkeys()
	update(dict)
	popitem()
	pop(key[,default])
```

- 遍历元素

```
# 遍历Key
for key in dict:
	print(key)

for key in d.keys() :
	print(key)

# 遍历Value
for value in  d.values() :
	print(value)

# 遍历 Key Value
for key, value in d.items() :
	print(key, value)

# 遍历项
for item in d.items() :
	print(item) # item 为二值元组
```

### 控制语句

- if

```
if a < b :
	pass
elif a > b :
	pass
else
	pass

```

- while

```
while a < b :
	pass
else :
	pass
```

- for

```
for i in list :
	print(i)

for i in range(5) :
	print(i)

for i in range(5, 9) :
	print(i)

for i in range(5, 9, 2) :
	print(i)
```

### 字符串操作

```
	len(str)
	str.lstrip(chars)
	str.rstrip(chars)
	str.strip(chars)
	str.lower()
	str.upper()
	str.replace(old, new[,max]) # max 最大替换次数
	str.isdecimal() # 是否为十进制

	+      # 字符串拼接
	*      # 重复拼接
	[i]    # 索引访问
	[:]    # 切片， 左闭右开
	in     # 是否包含子串
	not in # 是否不包含子串
```
