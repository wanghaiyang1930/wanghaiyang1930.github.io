## Write Ahead Log(WAL)

### Write-Ahead Transaction Log(SQL Server)

SQL Server 2005 uses a write-ahead log (WAL), which guarantees that no data modifications are written to disk before the associated log record is written to disk. This maintains the ACID properties for a transaction. For more information about transactions and ACID properties, see Transactions (Database Engine).

To understand how the write-ahead log works, it is important for you to know how modified data is written to disk. SQL Server maintains a buffer cache into which it reads data pages when data must be retrieved. Data modifications are not made directly to disk, but are made to the copy of the page in the buffer cache. The modification is not written to disk until a checkpoint occurs in the database, or the modification must be written to disk so the buffer can be used to hold a new page. Writing a modified data page from the buffer cache to disk is called flushing the page. A page modified in the cache, but not yet written to disk, is called a dirty page.

At the time a modification is made to a page in the buffer, a log record is built in the log cache that records the modification. This log record must be written to disk before the associated dirty page is flushed from the buffer cache to disk. If the dirty page is flushed before the log record is written, the dirty page creates a modification on the disk that cannot be rolled back if the server fails before the log record is written to disk. SQL Server has logic that prevents a dirty page from being flushed before the associated log record is written. Log records are written to disk when the transactions are committed.

### ARIES

ARIES 是 WAL 最长用的算法 [ARIES](https://my.oschina.net/fileoptions/blog/2988622)



