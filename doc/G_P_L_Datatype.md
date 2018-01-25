
---

#### Golang

> 数据类型（8种）

|Type |Feature |Value |Initialization |
|:----|:-------|:-----|:--------------|
|boolean |          |true false            |var v bool = true                  |
|byte    |          |uint8                 |var v byte = 'c'                   |
|integer |          |int uint 8 16 32 64   |var v int64 = 12                   |
|float   |          |float32 float64       |var v float32 = 1.1                |
|complext|          |complext64 complex128 |var v complext 64 = complext(1, 2) |
|string  |immutable |"string"              |var v string = "string"            |
|rune    |          |int32                 |var v rune = 12                    |
|const   |          |                      |                                   |

> string and rune
```
	import "unicode/utf8"
	l := len(s)
	r := []rune(s)
	s := string(r)
	for range(s)
	for _, _ := range(s)
	for i, r := range(s) 
	utf8.RuneCountInString(s)
	utf8.DecodeRuneInString(s[i:])
```
	
> 扩展数据类型(8种)

|Type |Feature |Transmittal |Initialization |
|:----|:------:|:-----------|:--------------|
|array     |mutable   |value     |var v [3]int                 |
|slice     |mutable   |reference |var v []int                  |
|map       |mutable   |reference |var v map[string]int         |
|struct    |mutable   |value     |var v book                   |
|point     |
|channel   |immutable |reference |
|function  |
|interface |

> array
```
	特性：固定长度、同类元素、长度静态、自动初始化

	创建：var a [3]int
		  var a [3]int = [3]int{1, 2, 3}
		  a := [3]int{}
		  a := [3]int{1, 2}
		  a := [...]int{1, 2, 3}
		  var p *[3]int = &a

	其他：比较：== 和 != 
	      元素类型必须相同且可比较
	      内部元素逐一比较

	type Index int
	const (
		USD Index iota
		EUR
		GBP
		RMB
	)
	symbol := [...]string{USD:"$", EUR:"€", GBP:"₣", RMB:"¥"}
	fmt.Println( USD, symbol[USD] )

	a := [...]int{99:-1} //定义100个元素，第99个是-1，从索引0自动补齐到指定索引
```

> slice
```
	特性：同类元素、不可比较、因为内部包含了指针所以表现出了引用的性质

	创建：s := a[i:j] (0<=i<=j<cap(a))
		  s := []int(nil) // == nil
		  s := []int{} // != nil
		  s := make([]int, 3)
		  s := make([]int, 0, 5)
		  var s []int // == nil
		  var s []int = nil // == nil
		  var s []int = []int(nil) // == nil
		  var s []int = []int{} // != nil
		  var s []int = make([]int, 3)
		  var s []int = make([]int, 3, 5)
```

> map
```
	特性：显式创建、不可比较、键必须定义等操作

	创建：m := map[string]int{} // != nil
		  m := map[string]int{
		  	"one": 1,
		  	"two": 2,
		  }

		 
		  var m map[string]int = map[string]int{} 
		  var m map[string]int = map[string]int{
		  	"one": 1,
		  	"two": 2,
		  }

		  var m = map[string]int{} // 自动推导
		  var m = map[string]int{
		  	"one": 1,
		  	"two": 2,
		  }

		  m := make(map[string]int)  // != nil
		  var m map[string]int       // == nil
		  var m map[string]int = nil // == nil

	其他：m["key"] = 0         // 不存在：安全
		  delete("key", m)     // 不存在：安全
		  &m["key"]            // 不存在：不安全
		  v, ok := m["key"]    // 判断是否存在
		  for k := range m     // 忽略了 value
		  for k, v := range m  // 键值遍历
```

> struct
```
	特性：字段可比较则可比较、可用作map的key

	创建：type book struct {
			name string
		  }

		  b := book{}
		  b := book{ "string" }
		  b := book{ name: "string" } 

		  p := &book{}
		  p := &book{ "string" }
		  p := &book{ name: "string" }
		  p := new(book)

		  var b book
		  var b book = book{}
		  var b book = book{ "string" }
		  var b book = book{ name: "string" }
		  var b = book{ name: "string" }

		  var p *book // nil
		  var p *book = &b // not nil
		  var p = &b // auto
		  
		  b.name == p.name == (*p).name

	其他：
		字段首字符大写则导出
		zero值为结构体对象
		空结构体：b := struct{}{}
		匿名结构体字段可以访问其内部字段（方便）
		"%#v" 打印整个结构体

		type Point struct {
			x, y int
		}

		type Circle struct {
			Point
			r int
		}

		var c Circle
		c.x 等价于 c.Point.x
```

> pointer
```
	特性：可比较、可用作map的key
```

> interface
```
	特性：可比较、可用作map的key
```

> 类型转换
```
	string to []byte: b := []byte("string")
```

---

#### Python

> 数据类型（3种，其他都归于class）

|Type    |Feature |Value |Initialization |
|:----------|:-------:|:--------:|-----:|
|bollean    |         |          |             |
|number     |         |          |             |
|string     |immutable|value     |v = "string" |

> 扩展数据类型（4种，其他都归于class）

Python一切事物皆对象，在参数传递时使用引用传递。

|Type    |Feature |Value |Initialization |
|:----------|:-------:|:--------:|-----:|
|list       |mutable  |reference |v = [] or v = list() |
|slice      |mutable  |reference ||
|tuple      |immutable|          |v = () or v = tuple() |
|set        |mutable  |reference |v = set() or v = {"a", "b"} | 
|frozenset  |immutable|          |v = frozenset() or v = {"a", "b"} |
|dictionary |mutable  |reference |v = {} or v = dict() |
|range      |immutable|          |v = range( 10 )   |
|bytes      |immutable|          || 
|bytearray  |mutable  |          ||
|array      |mutable  |reference |v = array.array( 'i', [1, 2] ) |

> list
```
	构造式初始化（列表创建）：l = ["a", "b", "d", 1, 2, 3]
	构造式初始化（列表创建）：l = list(iterable)
	其他初始化：l = list( "abc" ) 等价于 ["a", "b", "c"]
				l = list( tuple )
				l = list( set )
				l = list( dict )
				l = list( dict.keys() )
				l = list( dict.values() )
				l = list( range(10) )
```

> tuple
```
	构造式初始化（列表创建）：t = ( "one", 1 )
	构造式初始化（列表创建）；t = tuple()
	其他初始化：t = tuple( "abc" ) 等价于 ("a", "b", "c")
				t = tuple( list )
	            t = tuple( set )
	            t = tuple( dict )
	            t = tuple( dict.keys() )
	            t = tuple( dict.value() )
	            t = tuple( range(10) )
```

> range
```
	构造式初始化：r = range( stop ) 
	构造式初始化：r = range( start, stop )
	构造式初始化：r = range( start, stop, step )
```

> set
```
	构造式初始化（列表创建）：s = {"one", "two", "threee", 1, 2, 3}
	结构式初始化（记录常见）：s = set()
	                          s.add( "one" )
	其他初始化：s = set( "abc" ) 等价于 {"a", "b", "c"}
				s = set( list )
	            s = set( tuple )
	            s = set( dcit ) 等价于 set( dict.keys() )
	            s = set( dict.keys() )
	            s = set( dict.values() )
	            s = set( range(10) )
	其他操作： len( set )
		       x in set
		       x not in set
		       for i in set
```

> dictionary
```
	key: immutable type
	构造式初始化（记录创建）：d = { "name":"n", "age":1 }
	                          d = dict( name="n", age=1 )
	结构式初始化（记录创建）：d = {}
	                          d = dict()
	                          d["name"]="n"
	                          d["age"]=1
	其他初始化：d = dict( zip(["one", "two", "threee"], [1, 2, 3]) )
	            d = dict( [("one", 1), ("two", 2), ("three", 3)] )
	            d = dict( {"one":1, "two":2, "three":3} )
	其他操作：len( dict )
	          x in dict
	          x not in dict

```

> bytes
```
	只支持ASCII字符串
	
	b = b"my name is 'hello', hahaha"
	b = b'my name is "hello", hahaha'
	b = b'''my name is hello'''

	b = bytes( 10 ) // all zero
	b = bytes( range(10) )

	s = "string"           // "string", b[0] == 115,只可读取，不可赋值
	b = bytes( s )         // b'string'
	sh = b.hex()           // '737472696e67'
	b2 = bytes.fromhex(sh) // b'string'
```

> bytearray
```
	ba = bytearray()
	ba = bytearray( 10 )
	ba = bytearray( range(10) ) (0<=element<<255)

	s = "string"           // "string", b[0] = 115
	ba = bytes( s )         // b'string'
	sh = ba.hex()           // '737472696e67'
	ba2 = bytearray.fromhex(sh) // b'string'
```

> 类型转换
```
	string to bytes: b = bytes( "string", encoding="utf-8" )
					 b = str.encode( "string" )
	bytes to string: s = str( b ) // "b'string'" 很奇特，不要用
	                 s = str( b, encoding="utf-8" )
	                 s = bytes.decode( b )

	list to bytes: b = bytes( l ) (0<<element<<255)
	bytes to list: l = list( b )
```

#### Lua

> 数据类型（8种）

|Type     |Feature|Value|type()|Initialization|
|:--------|:-------:|:-------:|:---------------------:|------------------------------:|
|nil      |         |non-value|type(nil)=="nil"       |                               |
|boolean  |         |         |type(true)=="true"     |v = true                       |
|number   |         |         |type(10)=="number"     |v = 0.14                       |
|string   |immutable|value    |type("")=="string"     |v = "string"                   |
|table    |mutable  |reference|type({})=="table"      |v = {}                         |
|function |         |         |type(print)=="function"|function add (v) return v+1 end|
|thread   |         |         |                       |                               |
|userdata |         |         |                       |                               |

> boolean
```
	只将 false 和 nil 视为假，其他都视为真。
```

> string 
```
	多行字符串（首个换行自动忽略）：
	lines = [[
	    first line
		second line
		third line]]

	字符串连接（空格不可忽略）：str3 = str1 .. str2
	字符串长度：len = #str
```

> table
```
	构造表达式：t = {} 
	构造式创建（列表创建）：t1 = { "a", "b", 1, 2 }
	结构体式创建（记录创建）：t2 = { name = "n", age = 1 }
	结构体式创建（记录创建）：t3 = {}; t.name = "n"; t.age = 1
	混合式创建（记录列表混合创建）: t4 = {color="blue", num=2, "a", "b", 1, 2}
	访问：print(t1[1]); print(t2["name"]);
	注意：#t1 == 4 #t2 == 0 #t3 == 0
	注意：{x=0, y=0} 等价于 {["x"]=0, ["y"]=0}
	      {"r", "g", "b"} 等价于 {[1]="r", [2]="g", [3]="b"}

```

> 类型转换
```
	string to number: num = tonumber( "10.23" )
	number to string: str = tostring( 10.23 )
```

> 数组
```
	长度操作符：#array （依赖于索引起始值：1）
	构造表达式初始化：array = {"a", "b", "c", 1, 2, 3}

```


