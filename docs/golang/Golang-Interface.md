# Interface

### 描述

- Interface 接口：是方法的集合；
- Interface 类型：接口是一种指针类型；

### 使用

- 不仅结构体可以实现接口，自定义类型也可以实现接口；
- 空接口由于没有方法，所以任何类型都实现了空接口；
- 实现一个接口，必须实现该接口的所有方法；

### 类型转换

```
var data int = 0
var i interface = nil

i = data
value, ok := x.(int)
fmt.Println(value, ok)
```

### 类型判断

```
type Data struct {

}

func Print(items ...interface{}) {
	for index, value := range items {
		switch value.(type) {
			case string:
				fmt.Printf("Type string, %d %v\n", index, value);
			case bool:
				fmt.Printf("Type bool, %d %v\n", index, value);
			case int:
				fmt.Printf("Type int, %d %v\n", index, value);
			case Data:
				fmt.Printf("Type Data, %d %v\n", index, value);
			case *Data:
				fmt.Printf("Type *Data, %d %v\n", index, value);
		}
	}
}
```