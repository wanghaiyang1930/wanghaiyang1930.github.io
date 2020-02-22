## Raft

Raft is a protocol for implementing distributed consensus.

Raft 官网：https://raft.github.io/
Raft 动画演示：http://thesecretlivesofdata.com/raft/

### 名词：
	
- Leader：All changes to the system now go through the leader（但是这里没有明确读操作是否也只能通过 Leader 来进行）；
- Follower：所有节点在启动之初，状态都是 Follower，If followers don't hear from a leader then they can become a candidate.
- Candidate：选举过程中的中间状态，此时集群并不对外提供服务（读？写？）；
- Term：Raft 将集群被 Leader 进行管理的一段时间称作一个 Term（任期），任期的开始第一件事即：Leader 选举，直到新的 Leader 后，该任期即结束；
- Timeout：150~300ms，Follower 采用该超时时间来切换状态（转变为 Candidate），Candidate 采用该超时时间来重新发起选举；
- Log：Each change is added as an entry in the node's log（Raft 采用两段提交的进行数据同步）.

### Leader Election

#### 选举流程

1. In Raft there are two timeout settings which control elections.

2. First is the election timeout.

3. The election timeout is the amount of time a follower waits until becoming a candidate.

4. The election timeout is randomized to be between 150ms and 300ms.
解释：这里的超时是指 Follower 在没有收到 Leader 心跳消息之后，如果超时，则自身状态由 Follower 转变为 Candidate；

5. After the election timeout the follower becomes a candidate and starts a new election term...
解释：节点改变自身状态之后还会伴随着一些列的条件变更，Term 会进行自增（+1）、选票会增加一（自己选自己）；

6. ...votes for itself...
解释：Candidate 首先会为自己投一票；

7. ...and sends out Request Vote messages to other nodes.

8. If the receiving node hasn't voted yet in this term then it votes for the candidate...
解释：Follower 节点在收到选举消息之后，发现选举消息的任期（Term）比自己大，那么 Follower 会将自己的任期同步到 选举消息的任期吗？
但是一点是肯定的，在一个任期内，Follower 只会选择为第一个选举消息进行投票，如果有两个选举消息在使用同一个任期，那么 Follower 只会选择第一个选举消息。
疑问：如果短时间内第二个选举消息的任期大，是否可以理解，Follower 还会为第二个选举消息进行投票？

9. ...and the node resets its election timeout.
解释：重点：Follower 选举投票完成后重置超时时间（还是随机150～300ms吗？），这一点很重要。同时 Follower 也会更新自己的 Term，并记录自己选了哪个 Leader；

10. Once a candidate has a majority of votes it becomes leader.
解释：超过半数的选票就可以让 Candidate 成为 Leader；

11. The leader begins sending out Append Entries messages to its followers.
解释：这里涉及到日志不对其的处理，情况比较复杂；

12. These messages are sent in intervals specified by the heartbeat timeout.
解释：日志同步的消息是通过心跳消息包发送的吗？似乎可以这样理解。

13. Followers then respond to each Append Entries message.

14. This election term will continue until a follower stops receiving heartbeats and becomes a candidate.
解释：Follower 在收到 Leader 消息后，会重置 Timeout，期间 Term 会一直持续到下一个 Candidate 来进行选举。
疑问：Leader 在执行 Term 期间是没有超时的？如果日志复制不够半数，Leader 会自行进行状态变更吗（或采取其他措施）？

#### 重新选举

1. Let's stop the leader and watch a re-election happen.

2. Node A is now leader of term 2.

3. Requiring a majority of votes guarantees that only one leader can be elected per term.

4. If two nodes become candidates at the same time then a split vote can occur.

#### 选票分割

1. Let's take a look at a split vote example...

2. Two nodes both start an election for the same term...
figure-3-1.png
解释：超时时间虽然是随机，但彼此冲突的可能是绝对存在的；

3. ...and each reaches a single follower node before the other.
解释：由于到来的选举信息都是相同 Term，因此 Follower 只为第一个到来的选举信息投票（虽然投下反对票，但还是发送反馈信息）；

4. Now each candidate has 2 votes and can receive no more for this term.

5. Node D received a majority of votes in term 5 so it becomes leader.

疑问：两个 Candidate 重置了 Timeout 后，状态会不会从 Candidate 转换为 Follwer？
答疑：不会进行逆转换，Candidate 超时后会发起新一轮的选举，直至新的 Leader 产生；

疑问：这里是否出现了一个新的 Timeout，即两个 Candidate 在各自收到一个外来选票的情况下，是否启动了另一个 Timeout？
答疑：Candidate 在发出选举消息后即刻开启自己的 Timeout，这个 Timeout 与 Follower 是一致的，当选票到来不超过半数时，则持续这个 Timeout，如果收到选票超过半数，则结束当前 Timeout，进行 Leader 状态切换；

疑问：Follower 在收到选举消息后，会随着选举消息更新自己的 Term 吗？
答疑：Follower 会同步更新自己的 Term（只要比自己的 Term 大），同时还会记录自己选了谁，并且同一个 Term 的选举消息，只选第一个。

疑问：Follower 在收到选举消息投出选票后，会重置自己的 Timeout 吗？
答疑：是的，Follower 收到消息后不仅进行了相应内部数据更新，同时也重置了 Timeout（重点）。

疑问：为什么这里是刚才那两个 Candidate 其中一个苏醒了，那如果要是刚才的两个 Follower 其中一个苏醒呢？
答疑：这应该是随机的，都可以苏醒，因为 Follower 的 Term 在收到选举消息后也得到了更新，并不比 Candidate 落后（Term）。

疑问：如果两个 Candidate 没有苏醒，而是两个 Follower 其中一个苏醒，情况会怎样？
答疑：由于两个 Candidate 在上一轮选举中（平票无结果）已经将 Follower 的 Term 进行了同步（选举信息的Term大于Follower的Term），所以平票后的苏醒，所有节点都处在同一个 Term（正因为选举消息的 Term 都相同，所有产生了平票），即使是 Candidate 收到大于自己 Term 的选举信息，也不得不投票，因此选举会顺利选举出 Leader。

疑问：如果两个 Candidate 其中一苏醒，则 Term 就会自增为3，如果两个 Follower 其中一个苏醒，则 Term 就会自增为2，怎么办？
答疑：不会出现这种情况，Follower 在收到上次的选举消息后，已经将自己的 Term 与选举消息同步，即：Term 为2，即使任何一个节点苏醒，Term 并不会落后；

### 日志复制（Log Replication）

在Raft协议中，所有的日志条目都只会从 Leader 节点往 Follower 节点写入，且 Leader 节点上的日志只会增加，绝对不会删除或者覆盖。

#### 日志复制流程

1. Once we have a leader elected we need to replicate all changes to our system to all nodes.
解释：Leader 需要将所有 Follower 的日志与自己进行对其；

2. This is done by using the same Append Entries message that was used for heartbeats.
解释：Raft 不会单独提供额外消息，而是复用了心跳包进行 Append Entries message；

3. Let's walk through the process.

4. First a client sends a change to the leader.
解释：客户端写数据只能与 Leader 进行交互；

5. The change is appended to the leader's log...

6. ...then the change is sent to the followers on the next heartbeat.
解释：Follower 收到消息后，只进行日志追加；

7. An entry is committed once a majority of followers acknowledge it...
解释：Leader 此时会将自己日志中数据进行提交；

8. ...and a response is sent to the client.
解释：同时会发送提交消息给 Follower，Follower 完成提交操作；

9. Now let's send a command to increment the value by "2".

10. Our system value is now updated to "7".

疑问：Client 与 Follower 可以发生只读数据交互吗？
答疑：

疑问：如果数据处于 Log 中的 Uncommit 状态，此时读这个数据该如何处理？
答疑：

疑问：Follower 收到提交请求并提交数据失败后怎么处理？提交结束后，还要不要向 Leader 反馈信息？
答疑：Leader 如果收不到数据已经 Commited 的消息，会一直向 Follower 一致发送提交数据的请求，直至成功（岂不死循环？）；

疑问：Follower 如果能收到 Leader 的数据同步与数据提交请求，但是无法发送给我 Leader 确认信息怎么办？
答疑：同上，不过难道能个系统就停在这里了？

疑问：某些Followers可能没有成功的复制日志，Leader会无限的重试 AppendEntries RPC直到所有的Followers最终存储了所有的日志条目？
答疑：

#### 网络分区一致性

1. Raft can even stay consistent in the face of network partitions.

2. Let's add a partition to separate A & B from C, D & E.

3. Because of our partition we now have two leaders in different terms.
解释：两个 Leader：B（old）、C（new）

4. Let's add another client and try to update both leaders.

5. One client will try to set the value of node B to "3".

6. Node B cannot replicate to a majority so its log entry stays uncommitted.
解释：没有足够数量的确认消息返回，因此 Leader B 中的数据不能提交，只能返回 Client 失败；

7. The other client will try to set the value of node C to "8".

8. This will succeed because it can replicate to a majority.

9. Now let's heal the network partition.

10. Node B will see the higher election term and step down.
解释：由于新来的消息携带的 Term 更大，因此老 Leader 自己降级；

11. Both nodes A & B will roll back their uncommitted entries and match the new leader's log.

12. Our log is now consistent across our cluster.

疑问：此处的日志复制并没有将的很细致，并且日志复制还有很多不一致需要处理？
答疑：

疑问：网络恢复后，分区A的新数据可以顺利同步给分区B的节点，但是分区A的老数据如何同步给分区B的各个节点？
答疑：

### 特点

- Raft算法只有 Leader 会接受客户端的写请求；（Follower 是否对外提供读请求？Candidate 是否对外提供服务？选举期间是否对外提供读服务？）

- 如果同一时刻，所有节点都有 Follower 状态转换为 Candidate 状态，那该如何选举？答：会在此进行选举；

- Candidate 状态在再次 Timeout 期间，会给其他候选人投票吗？答：会，只要选举消息的 Term 比自己大；

### 安全

- Raft增加了如下两条限制以保证安全性：拥有最新的已提交的log entry的Follower才有资格成为Leader。
- Leader只能推进commit index来提交当前term的已经复制到大多数服务器上的日志，旧term日志的提交要等到提交当前term的日志来间接提交（log index 小于 commit index的日志被间接提交）。

### RPC

- RequestVote RPC：候选人在选举期间发起；
- AppendEntries RPC：领导人发起的一种心跳机制，复制日志也在该命令中完成；
- InstallSnapshot RPC: 领导者使用该RPC来发送快照给太落后的追随者；

### 其他链接

https://www.infoq.cn/article/coreos-analyse-etcd/
https://www.cnblogs.com/xybaby/p/10124083.html
https://blog.csdn.net/daaikuaichuan/article/details/98627822
https://zhuanlan.zhihu.com/p/32052223




