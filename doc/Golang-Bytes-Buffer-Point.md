### 创建

```
var buf bytes.Buffer

buf := new(bytes.Buffer)

func NewBuffer(buf []byte) *Buffer

func NewBufferString(s string) *Buffer
```

### NewBuffer
```
// NewBuffer creates and initializes a new Buffer using buf as its initial
// contents.  It is intended to prepare a Buffer to read existing data.  It
// can also be used to size the internal buffer for writing. To do that,
// buf should have the desired capacity but a length of zero.
//
// In most cases, new(Buffer) (or just declaring a Buffer variable) is
// sufficient to initialize a Buffer.
func NewBuffer(buf []byte) *Buffer { return &Buffer{buf: buf} }

func IntToBytes(n int) []byte {
    data := int32(n)
    
    buffer := bytes.NewBuffer([]byte{}) //创建一个内容是[]byte的slice的缓冲器(等效: bytes.NewBufferString(""))
    binary.Write(buffer, binary.BigEndian, data)

    return buffer.Bytes()
}

/// Note: 如果NewBuffer中传入一个数据，那么长度固定，当buffer满了怎么办？是否可以将问题回归到Slice对Array的解引用问题上？
```

### NewBufferString
```
// NewBufferString creates and initializes a new Buffer using string s as its
// initial contents. It is intended to prepare a buffer to read an existing
// string.
//
// In most cases, new(Buffer) (or just declaring a Buffer variable) is
// sufficient to initialize a Buffer.
func NewBufferString(s string) *Buffer {
    return &Buffer{buf: []byte(s)}
}
```

### 方法

```
// Write appends the contents of p to the buffer, growing the buffer as
// needed. The return value n is the length of p; err is always nil. If the
// buffer becomes too large, Write will panic with ErrTooLarge.
func (b *Buffer) Write(p []byte) (n int, err error) {
    b.lastRead = opInvalid
    m := b.grow(len(p))
    return copy(b.buf[m:], p), nil
}

// WriteString appends the contents of s to the buffer, growing the buffer as
// needed. The return value n is the length of s; err is always nil. If the
// buffer becomes too large, WriteString will panic with ErrTooLarge.
func (b *Buffer) WriteString(s string) (n int, err error) {
    b.lastRead = opInvalid
    m := b.grow(len(s))
    return copy(b.buf[m:], s), nil
}

// WriteByte appends the byte c to the buffer, growing the buffer as needed.
// The returned error is always nil, but is included to match bufio.Writer's
// WriteByte. If the buffer becomes too large, WriteByte will panic with
// ErrTooLarge.
func (b *Buffer) WriteByte(c byte) error {
    b.lastRead = opInvalid
    m := b.grow(1)
    b.buf[m] = c
    return nil
}

// WriteRune appends the UTF-8 encoding of Unicode code point r to the
// buffer, returning its length and an error, which is always nil but is
// included to match bufio.Writer's WriteRune. The buffer is grown as needed;
// if it becomes too large, WriteRune will panic with ErrTooLarge.
func (b *Buffer) WriteRune(r rune) (n int, err error) {
    if r < utf8.RuneSelf {
        b.WriteByte(byte(r))
        return 1, nil
    }
    n = utf8.EncodeRune(b.runeBytes[0:], r)
    b.Write(b.runeBytes[0:n])
    return n, nil
}

// Read reads the next len(p) bytes from the buffer or until the buffer
// is drained.  The return value n is the number of bytes read.  If the
// buffer has no data to return, err is io.EOF (unless len(p) is zero);
// otherwise it is nil.
func (b *Buffer) Read(p []byte) (n int, err error) {}

// ReadRune reads and returns the next UTF-8-encoded
// Unicode code point from the buffer.
// If no bytes are available, the error returned is io.EOF.
// If the bytes are an erroneous UTF-8 encoding, it
// consumes one byte and returns U+FFFD, 1.
func (b *Buffer) ReadRune() (r rune, size int, err error) {}

// ReadBytes reads until the first occurrence of delim in the input,
// returning a slice containing the data up to and including the delimiter.
// If ReadBytes encounters an error before finding a delimiter,
// it returns the data read before the error and the error itself (often io.EOF).
// ReadBytes returns err != nil if and only if the returned data does not end in
// delim.
func (b *Buffer) ReadBytes(delim byte) (line []byte, err error) {}

// ReadString reads until the first occurrence of delim in the input,
// returning a string containing the data up to and including the delimiter.
// If ReadString encounters an error before finding a delimiter,
// it returns the data read before the error and the error itself (often io.EOF).
// ReadString returns err != nil if and only if the returned data does not end
// in delim.
func (b *Buffer) ReadString(delim byte) (line string, err error) {}

// ReadFrom reads data from r until EOF and appends it to the buffer, growing
// the buffer as needed. The return value n is the number of bytes read. Any
// error except io.EOF encountered during the read is also returned. If the
// buffer becomes too large, ReadFrom will panic with ErrTooLarge.
func (b *Buffer) ReadFrom(r io.Reader) (n int64, err error) {}

// Reset resets the buffer so it has no content.(Clear)
// b.Reset() is the same as b.Truncate(0).
func (b *Buffer) Reset() { b.Truncate(0) }

// String returns the contents of the unread portion of the buffer
// as a string.  If the Buffer is a nil pointer, it returns "<nil>".
func (b *Buffer) String() string {}

```