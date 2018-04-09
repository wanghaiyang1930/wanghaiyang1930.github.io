#### LevelDB数据写入流程

##### 数据写入概述

DBImpl维护一个Writer的Queue，多线程写入数据时，首先将用户数据进入写入队列等待执行，获得执行时间片后，当前线程开始管理缓存数据，主要进行Mutable Memory Table与Immutable Memory Table的交换，然后重新生成Mutable Memory Table与Log文件，同时启动后台线程进行数据磁盘化。

DBImpl在当前线程环境下并没有直接将数据写入缓存（DBImpl::Write()），这其中出于多种条件限制考虑（具体条件后续分析？？？），采用批量写入缓存的策略，将数据打包成批，后续就此批量数据包写入日志与缓存。

##### 代码分析

```
Status DBImpl::Write(const WriteOptions& options, WriteBatch* my_batch) {
	// 封装需写入的数据，这样写入线程都将持有独立的Condition。方便线程间协同工作。
	// 此处使用的互斥量（Mutex）进行加锁，这种粒度的锁由此看来是可以接受的，我们在自己的项目中对
	// Mutex的使用不必过于紧张，LevelDB高性能的吞吐量尚且如此，我们也可以效仿学习。
	Writer w(&mutex_);
    w.batch = my_batch;
    w.sync = options.sync;
    w.done = false;

    MutexLock l(&mutex_);
    writers_.push_back(&w);
    // 当前线程的数据入队列后发现原队列中已经有数据，则当前线程进入等待状态，同时已经入队列的数据
    // 有可能被其他线程放入一批次中处理掉，因此线程被唤醒后需要检查当前数据状态。
    while (!w.done && &w != writers_.front()) {
      w.cv.Wait();
    }
    if (w.done) {
    return w.status;
    }

    // 管理缓存，如果Mutable Memory Table缓存满了，则交换Immutable Memory Table与Mutable Memory Table，同时
    // 启动后台磁盘化线程进行数据磁盘化。
    // May temporarily unlock and wait.
    Status status = MakeRoomForWrite(my_batch == NULL);
    uint64_t last_sequence = versions_->LastSequence();
    Writer* last_writer = &w;
    if (status.ok() && my_batch != NULL) {  // NULL batch is for compactions

    	WriteBatch* updates = BuildBatchGroup(&last_writer);
   		WriteBatchInternal::SetSequence(updates, last_sequence + 1);
    	last_sequence += WriteBatchInternal::Count(updates);

    	// Add to log and apply to memtable.  We can release the lock
    	// during this phase since &w is currently responsible for logging
    	// and protects against concurrent loggers and concurrent writes
    	// into mem_.
    	{
        	mutex_.Unlock();
        	// LevelDB的数据日志并没有使用额外的缓存，而是直接使用系统级别的文件写入缓存。
        	// 对于数据写入操作LevelDB采用这种额外的开销目的显而易见，就是防止数据丢失同
        	// 时重启后还可以恢复数据，LevelDB愿意为可靠性牺牲一点效率值得学习，同时也从
        	// 侧面给我们以启示：尽管对吞吐量需求较大的程序也可以通过每条操作写入日志来保
        	// 证可靠性，数据日志的写入损耗我们还是能够承受的。
      		status = log_->AddRecord(WriteBatchInternal::Contents(updates));
      		bool sync_error = false;
      		if (status.ok() && options.sync) {
        		status = logfile_->Sync();
        		if (!status.ok()) {
          			sync_error = true;
        		}
        	}

      		if (status.ok()) {
        		status = WriteBatchInternal::InsertInto(updates, mem_);
      		}
     	 	mutex_.Lock();
      		if (sync_error) {
        		// The state of the log file is indeterminate: the log record we
        		// just added may or may not show up when the DB is re-opened.
        		// So we force the DB into a mode where all future writes fail.
        		RecordBackgroundError(status);
      		}
    	}
    	if (updates == tmp_batch_) tmp_batch_->Clear();

    	versions_->SetLastSequence(last_sequence);
  	}

    while (true) {
    	Writer* ready = writers_.front();
    	writers_.pop_front();
    	if (ready != &w) {
      		ready->status = status;
      		ready->done = true;
      		ready->cv.Signal();
    	}
    	if (ready == last_writer) break;
  	}

  	// Notify new head of write queue
  	if (!writers_.empty()) {
    	writers_.front()->cv.Signal();
  	}

  	return status;
}
```

在写这篇分析的时候其实作者本不想贴代码，但是就这一个函数的前前后后整个逻辑看下来整整花了两天时间，这个函数的本身逻辑并不复杂，但是涉及到后台线程协同数据磁盘化，所以多线程造成代码理解上比较费时。

LevelDB在使用后台线程时没有在进程启动时直接启动后台线程，而是在前台线程需要时启动了一个永久后台线程，主要负责数据压缩然后磁盘化。LevelDB在MakeRoomForWrite()函数中选择性的启动了后台线程，这个线程并没有设置退出条件，只是在没有数据需要处理时进入Wait状态，前台线程根据具情况适当的唤醒后台线程。具体调度流程如下图所示。

![MultiThread](https://github.com/wanghaiyang1930/wanghaiyang1930.github.io/blob/master/leveldb/image/LevelDB-Compaction-BGThread.png)

##### 启示

1. 数据操作日志可以实时写入，不必过于担心影响这个系统的效率。此前在设计程序时都尽量避免磁盘操作，对磁盘操作过于敏感，通过对LevelDB的分析，发现一个偏向于写端效率的Key-Value数据库并不忌讳每次写入数据时进行日志备份，所以我们在设计系统时也不必过于敏感磁盘写入操作，此外备份日志操作也是顺序写入，这样效率其实蛮高的；
2. 多线程操作，互斥锁的使用不能泛滥但也不能避而不碰。多线程操作冲突不可避免，使用互斥锁是最直接简便的方法，LevelDB并没有忌讳互斥锁对前后台线程造成的性能损耗，在保证数据写入的正确的同时也能提供不错的性能。这一点我们可以借鉴，该加入互斥锁的时候也不能手软。
