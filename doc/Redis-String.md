# String

## Operation

- Get  key
- MGet key [key ...]

- Set  key value
- MSet key value [key value ...]

- SETNX  key value // set if not exist
- MSETNX key value [key value ...] // multi set if not exist

- SETEX  key seconds value // set if exist and update expired time(Seconds)
- PSETEX key mseconds vlaue // set if exist and update expired time(MilliSeconds)

- STRLEN key

- Append key value // if exist and is a string(???), apennd the value

- GetRange key start end
- SetRange key offset value
- SetBit key offset value

## 

### SDS Memory 内存策略

- Length <  1MB: Capacity: length * 2 + 1Byte
- Length >= 1MB: Capacity: length + 1MB + 1Byte

- Prepare alloc space;
- Lazy free space;

### SDS Advantage

- Get lenth with O(1);
- Safety for binary data;
- Resuse some function of string.h;

## Question

Q：Stirng 类型会产生很多Key、Value，这些键值存储在什么位置？什么数据结构？
A：Redis默认情况下会建立一个默认数据库，所有的 Key （String、List、Set、Hash）都存储在这个数据库中，因此String的这些键都将存储在 redisDb::dict 中（字典结构）。