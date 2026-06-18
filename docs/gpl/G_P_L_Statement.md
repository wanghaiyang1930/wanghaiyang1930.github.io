#### Golang

> if:
```
	if expression {

	} else if expression {

	} else {

	}
```

> switch:
```
	switch value {
		case value:
			...
		case value:
			...
		default:
			...
	}

	swith {
		case value == "A":
			...
		case value == "B", value == "C":
			...
		default:
			...
	}

	var x interface{}
	swith t := x.(type) {
		case nil:
			...
		case int:
			...
		case func(int) int:
			...
		case bool, string: // or
			...
		default:
			...
	}
```

> select：
```
	select {
		case communication:
			...
		case communication:
			...
		default:
			...
	}
```

> for：
```
	for init; condition; post {
		break
	}

	for condition {
		continue
	}

	for {

	}

	for k, v := range(slice, array, string, map) {

	}
```

> 注意:
```
	select 在 for 循环之中时，在 select 中 break 并不能跳出 for 循环。
	for {
		select {
		case <- time.After( 2*time.Second):
			break
		}
	}
	这将形成一个死循环，需要使用标签来跳出循环。

	方案一：
	FOR_END：// FOR_END 和 for 之间不能有语句
	for {
		select {
		case <- time.After( 2*time.Second):
			break FOR_END：
		}
	}

	方案二：
	for {
		select {
		case <- time.After( 2*time.Second):
			goto FOR_END
		}
	}
	FOR_END：
```

***

#### python

> if:
```
	if expression :
		...
	elif expression :
		...
	else :
		...
```

> for:
```
	for v in iterator :
		...

	for v in list[:] :
		print(v)

	for i in range(0, 10, 1) :
		print(i)
```

> while:
```
	while True : # Dead loop.
		pass

	while expression :
		if exoression :
			break

		if expression :
			continue
```

***

#### Lua

> if:
```
	if expression then expression end

	if expression then
		...
	elseif expression then
		...
	else
		...
	end
```

> while:
```
	while expresion do
		...
	end
```

> repeat:
```
	repeat
		...
	until expression
```

> for:
```
	注意：var == exp 时，也要执行
	for var = init, exp, step do
		...
		break
	end

	
	注意：用于遍历数组
	for i, v in ipairs(array) do
		...
	end

	注意：用于遍历 Table
	for v in pairs(table) do
		print(v, table(v)) // key, value
	end

	for k, v in pairs(table) do
		print(k, v) // key, value
		if expression then
			break
		end
	end

	注意：break 和 return 只能是一个块的最后一条语句，之后只能是 end、else、until 之前。
```
