### TCP

```
/*
 * |             16 Bits           |             16 Bits           |
 * |---------------I---------------I---------------I---------------| // 32 Bits(4Bytes)
 * |        Source Port            |       Destination port        |
 * |---------------I---------------I---------------I---------------| // 32 Bits(4Bytes)
 * |                              Seq                              |
 * |---------------I---------------I---------------I---------------| // 32 Bits(4Bytes)
 * |                              Ack                              |
 * |---------------I---------------I---------------I---------------| // 32 Bits(4Bytes)
 * |  HL | Res |URG|ACK|RST|SYN|FIN|          Window Size          |
 * |---------------I---------------I---------------I---------------| // 32 Bits(4Bytes)
 * |            Checksum           |         Urgent pointer        |
 */
```

- Source Port: 2Bytes, Range: [0~65535];
- Destination Port: 2 Bytes, Range: [0~65535];
- Seq: 4 Bytes 源向目标
- Ack: 4 Bytes 只有标识为ACK为1时，该字段才生效，标示源端期望从目标端收到的下一个序列号；
- HL: 4 Bit
- Res: 4 Bit
- URG: If 1, the Urgent pointer field contains information;
- ACK: If 1, the Acknowledgment field contains information, if 0, Acknowledgment filed will be ignored;
- RST: If 1, the sender is requestiong that the connection be reset;
- SYN: If 1, he sender is requesting a syn chronization of the sequence numbers between the two nodes. This code is used when TCP requests a connection to set the initial sequence number.
- FIN: If 1, FIN-IF set to 1, the segment is the last in a sequence and the connection should be closed.

Indicates how many bytes the sender can issue to a receiver while acknowledgment for this segment is outstanding. This field performs flow control, preventing the receiver from being deluged with bytes. For example, suppose a server indicates a sliding window size of 4000 bytes. Also suppose the client has already issued 1000 bytes, 250 of which have been received and acknowledged by the server. That means that the server is still buffering 750 bytes. Therefore, the client can only issue 3250 additional bytes before it receives acknowledgment from the server for the 750 bytes.

### Checksum
谁的教验和？整个TCP帧的，还是？


Options	0-32 bits	Specifies special options, such as the maximum segment size a network can handle.
Padding	Variable	Contains filler information to ensure that the size of the TCP header is a multiple of 32 bits.
