## 文件读写

### io/ioutil

```
package main

import "fmt"
import "io/ioutil"

func main() {
	data, err := ioutil.ReadFile("temp.txt")
	if err != nil {
		fmt.Println("ReadFile fail, file:", "temp.txt, err:", err)
		return
	}

	str = string(data)
	fmt.Println("data:", str)

	err = ioutil.WriteFile("temp.txt", []byte(str), 0777)
	return
}
```

### os

```
func Open(name string) (*File, error)
func OpenFile(name string, flag int, perm FileMode) (file *File, err error)
    flag：os.O_RDONLY os.O_WRONLY os.O_CREATE os.O_APPEND os.O_EXCL

func (f *File) Read(b []byte) (n int, err error)
func (f *File) ReadAt(b []byte, off int64) (n int, err error)

func (f *File) Write(b []byte) (n int, err error)
func (f *File) WriteAt(b []byte, off int64) (n int, err error)
func (f *File) WriteString(s string) (n int, err error)
```

```
package main

import "fmt"
import "io/ioutil"
import "os"

func main() {
	file, ferr := os.Open("temp.txt")
	if nil != ferr {
		return
	}

	defer file.Close()

	buf := make([]byte, 1024)
	
	for { // 循环读取文件
		
		count, rerr := fp.Read(buf)
		if rerr == io.EOF {  
			break // io.EOF表示文件末尾
		}
		fmt.Println(string(buf), count)
	}
}
```

### bufio

```
package main

import (
	"bufio"
	"os"
	"fmt"
)

func main() {
	file, ferr := os.Open("temp.txt")
	if ferr != nil {
		return;
	}
	defer file.Close()

	reader := bufio.NewReader(file)

	for {
		data, rerr := reader.ReadString(byte('\n'))
		if rerr == io.EOF {
			break
		}

		if rerr != err {
			fmt.Printlnf("data:", data)
			break;
		}
	}
}
```

```
type Reader struct {
}

func NewReader(rd io.Reader) *Reader
func NewReaderSize(rd io.Reader, size int) *Reader

func (b *Reader) Buffered() int

func (b *Reader) Discard(n int) (discarded int, err error)

func (b *Reader) Peek(n int) ([]byte, error)

func (b *Reader) Read(p []byte) (n int, err error)
func (b *Reader) ReadByte() (byte, error)

func (b *Reader) ReadSlice(delim byte) (line []byte, err error)

func (b *Reader) ReadBytes(delim byte) ([]byte, error)
func (b *Reader) ReadString(delim byte) (string, error)

func (b *Reader) ReadLine() (line []byte, isPrefix bool, err error)
func (b *Reader) ReadRune() (r rune, size int, err error)
```

```
type Writer struct {
}

func NewWriter(w io.Writer) *Writer /// Default size: 4096.
func NewWriterSize(w io.Writer, size int) *Writer

func (b *Writer) Write(p []byte) (nn int, err error)
func (b *Writer) Flush() error

func (b *Writer) WriteByte(c byte) error
func (b *Writer) WriteRune(r rune) (size int, err error)
func (b *Writer) WriteString(s string) (int, error)
```