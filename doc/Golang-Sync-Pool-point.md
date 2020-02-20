### 对象池

1 对象池中的对象会在任意时刻被GC（准确的讲：池中对象的有效期在两个GC中间），并不会通知创建者；
2 对象池中的对象有可能被其他Goroutine取走(支持多Goroutine)；
3 不适合用于存储带有状态的对象，如：数据库链接，如果需要状态池，可以参考：go-commons-pool；

### 示例

```
package main

import "fmt"
import "sync"
import "time"

type Data struct {
	field_1 int
	field_2 int
}

func read_with_pool(pool sync.Pool, msg <-chan Data) {
	for {
		d, ok := <- msg
		if !ok {
			break
		}

		sum := d.field_1 + d.field_2
		_ = sum

		pool.Put(d)
	}
}

func write_with_pool(count int, pool sync.Pool, msg chan<- Data) {
	for i := 0; i < count; i++ {
		data := pool.Get().(*Data)
		data.field_1 = i
		data.field_2 = i + 1
		msg <- *data
	}

	close(msg)
}

func read_without_pool(msg <-chan Data) {
	for {
		d, ok := <- msg
		if !ok {
			break
		}

		sum := d.field_1 + d.field_2
		_ = sum
	}
}

func write_without_pool(count int, msg chan<- Data) {
	for i := 0; i < count; i++ {
		data := new(Data)
		data.field_1 = i
		data.field_2 = i + 1
		msg <- *data
	}

	close(msg)
}

func main() {

	data_pool := sync.Pool{New: func() interface{} {
		data := new(Data)
		return data
	}}

	count := 100000000
	wg := sync.WaitGroup{}

	msg_with_pool := make(chan Data)
	msg_without_pool := make(chan Data)

	wg.Add(1)
	go func() {
		defer wg.Done()
		start := time.Now().Unix()
		read_with_pool(data_pool,  msg_with_pool)
		end := time.Now().Unix()
		fmt.Println("read over with pool, cost:", end-start)
	} ()

	wg.Add(1)
	go func() {
		defer wg.Done()
		start := time.Now().Unix()
		write_with_pool(count, data_pool, msg_with_pool)
		end := time.Now().Unix()
		fmt.Println("write over with pool, cost:", end-start)
	} ()

	wg.Add(1)
	go func() {
		defer wg.Done()
		start := time.Now().Unix()
		read_without_pool(msg_without_pool)
		end := time.Now().Unix()
		fmt.Println("read over without pool, cost:", end-start)
	} ()

	wg.Add(1)
	go func() {
		defer wg.Done() 
		start := time.Now().Unix()
		write_without_pool(count, msg_without_pool)
		end := time.Now().Unix()
		fmt.Println("write over without pool, cost:", end-start)
	} ()

	wg.Wait();
}
```

### 测试

测试结果：https://studygolang.com/articles/21384?fr=sidebar

### 其他

golib：https://github.com/cloudflare/golibs

