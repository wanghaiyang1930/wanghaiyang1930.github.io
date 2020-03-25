# ACL

ACL: Access Control List.

主要作用：限制用户使用限定命令和访问限定键。

工作流程：用户连接Redis后进行身份认证，身份认证（Username、Password）后，与用户相关的权限限制便绑定在用户连接之上。

## 增加ACL的原因

- 通过访问限制提高安全性；
- 提高操作安全，减少由于误操作导致的数据损坏；

## Operation

```
> ACL LIST
1) "user default on nopass ~* +@all"
```

- the default user is configured to be active (on);
- to require no password (nopass);
- to access every possible key (~*);
- to call every possible command (+@all);

### Allow and disallow commands

- +@admin, @set, @sortedset // Add all the commands in such category to be called by the user.
- +@all // All means all the commands.

- -@<category> // Removes the commands from the list of commands the cient can call.

### Allow and disallow certain keys

~<pattern> : Add a pattern of keys that can be mentioned as part of commands.;
~* : Allows all the keys;

### Examples

```
> ACL SETUSER alice on >p1pp0 ~cached:* +get
```

- Active the user of alice;
- Authenticate with password p1pp0;
- Access key name start with the string "cached:";
- Access with only the GET commond;

## QA

Q: What happens calling ACL SETUSER multiple times?
A: What is critical to know is that every SETUSER call will NOT reset the user, but will just apply the ACL rules to the existing user.


