#### Golang

#### Python

> 算数运算符：
```
	+ - * / // % **
```

> 其他：
```
	多行字符串：print("""
		first line,
		second line""")
```



#### Lua

|优先级|操作符|类别|示例|说明|
|:-----|:-----|:---|:---|:---|
|高|^               |二元|v^2, v^0.5, v^(-1/3)           |v平方，v平方根，v立方根倒数|算数运算符      |
|. |not #           |一元|not nil, not false, not not nil|true, true, false          |逻辑运算符和其他|
|. |* / %           |二元|                               |                           |算数运算符      |
|. |+ -             |二元|                               |                           |算数运算符      |
|. |..              |二元|s1 .. s2                       |字符串连接                 |其他            |
|. |< > <= >= ~= == |二元|                               |                           |关系运算符      |
|. |and             |二元|                               |                           |逻辑运算符      |
|低|or              |二元|                               |                           |逻辑运算符      |

> 注意：
```
	^ 和 .. 是右结合，其他都是左结合
	a % b == a - floor(a/b) 涉及浮点数操作
	    计算结果的符号永远与第二个参数相同；
	    x % 1 ：结果是x的小数部分
	    x - x % 1 : 结果是x的整数部分
	    x - x % 0.01 ：精确到小数点后两位
	关系运算符：只能对number或string进行大小比较，其他类型只能进行相等或不等比较
	first and second : first为false，则返回first，否则返回second
	first or  second : first为true，则返回first，否则返回second
	nor : 只返回ture或者false
```