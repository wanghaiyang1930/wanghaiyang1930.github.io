#### LevelDB SSTable Structrure

SSTable由多个Block组成，存储有序的键值对序列且文件大小限制在2M左右，结构如下图所示。

![sstable](https://github.com/wanghaiyang1930/wanghaiyang1930.github.io/blob/master/leveldb/image/LevelDB-SSTable-Structrure.png)

Data Block用作SSTable的数据存储域，所有写入SSTable中的记录（键值对）都存储在Data Block中，结构如下图所示。

![datablock](https://github.com/wanghaiyang1930/wanghaiyang1930.github.io/blob/master/leveldb/image/LevelDB-Data-Block.png)

Filter Block用作存储SSTable的过滤器，LevelDB自身实现了布隆过滤器，用户可在初始化时指定该过滤器，结构如图所示。

![filterblock](https://github.com/wanghaiyang1930/wanghaiyang1930.github.io/blob/master/leveldb/image/LevelDB-Filter-Block.png)

Meta Index Block用作存储SSTable的元数据信息，目前之存储一条记录（如果使用了过滤器），即Filer Block的元数据（Filter Block在文件中的偏移量和大小），结构如图所示。

![metadindexblock](https://github.com/wanghaiyang1930/wanghaiyang1930.github.io/blob/master/leveldb/image/LevelDB-Meta-Index-Block.png)

Data Index Block用作存储每个Data Block的元数据，Data Block的元数据中存储了该Block中最后一个Key和该Block的偏移量与大小，结构如图所示。

![indexblock](https://github.com/wanghaiyang1930/wanghaiyang1930.github.io/blob/master/leveldb/image/LevelDB-Data-Index-Block.png)

Footer共占用48 Bytes，包含了Meta Index Block和Data Index Block的元数据（Block文件中的偏移量和大小）。

在进行数据存储时，写入的每条记录（键值对）首先缓存在Data Block中，每4K数据封装一个Data Block并进行一次磁盘同步，每个Data Block写入磁盘后将该Block的索引数据缓存到Data Index Block，并且在写入每条记录的同时缓存每条记录的Key到Filter Block（如果使用过滤器），如此往复，直至文件大小达到阈值，此时文件结束。

当文件文件结束时，上述各项Block会依次追加在最后一个Data Block（<=4K）之后，由于暂时没有Meta Block，因此首先追加Filter Block（如果使用过滤器），紧接着追加Meta Index Block，再追加Index Block，最后追加Footer，结束文件。

SSTable文件中只包含一个Filter Block、Meta Index Block和Index Block，而包含多个Data Block会。

#### 写在最后

SSTable的索引信息和元数据信息都存储在文件尾，如果文件写到一半出现异常，那这个数据文件该如何恢复或处理？