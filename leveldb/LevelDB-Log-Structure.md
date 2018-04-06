#### LevelDB日志文件结构分析

LevelDB日志文件由一系列Block组成，每个Block固定32KB，文件尾最后一个Block可能不足32KB。

Block由一系列Record组成，对于长度过长的Record可能分部在多个Block中。

如果一个Block的最后空余空间小于等于6个字节时，LevelDB将丢弃这个6个字节（置0），新加入的Record将启用新的Block。官方解释为这样不合适，可以理解为当Length等于0时，整个Record最少占用7个字节，这样末尾的7个字节就不需要丢弃，如果空余6个字节，则无法满足最小Record要求。

> Record分为四种：

- FULL   == 1
- FIRST  == 2
- MIDDLE == 3
- LAST   == 4

LevelDB有可能将一条过长的用户数据切分放入多个Record中，FIRST、MIDDLE、LAST就是为切分数据而准备的。下图是Block结构图与用户数据存储示例。

![Example](https://github.com/wanghaiyang1930/wanghaiyang1930.github.io/blob/master/leveldb/image/LevelDB-Log-Struct.png)

LevelDB会将用户的每一条数据都实时的同步在日志中，目的是为防止内存数据还未来得及固化到磁盘而因外界因素导致数据丢失。但是磁盘化数据总会有效率性与实时性的冲突，每条用户数据都实时的进行了磁盘写操作。

log::Write -> WriteableFile -> PosixWriableFile，最终PosixWriableFile实现了内存与磁盘操作的衔接。其中，WriteableFile和PosixWriableFile是对底层磁盘写的抽象与封装，而log::Write则序列化了日志文件的Block与Record结构。这样的结构有助于WriteableFile和PosixWriableFile的复用，隔离了数据结构与具体的磁盘写入。
