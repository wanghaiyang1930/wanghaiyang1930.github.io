1. 覆盖单个文件：

　　git fetch

　　git checkout origin/master -- path/to/file

2. 覆盖所有文件：

　　git fetch --all

　　git reset --hard origin/master

3. 查看Diff：

　　git diff HEAD -- <filename> （“--”前后有空格）// 查看本次改动

　　git diff HEAD^ -- <filename> （“--”前后有空格）// 查看与上以版本的改动

4. 删除分支

　　git push origin --delete name // 删除远端分支

　　git branch -d name //删除本地分支

　　git checkout -b name //创建分支

5. 回滚分支

$ git log
commit 5e88d154e63347a2abd699bab0ee15ea**
Author: ****
Date:   **

回滚本地分支：
	$ git reset --hard commit_id(commit字符)
	$ git push origin HEAD --force
	$ git reset --hard commit_id （回滚到commit_id，将commit_id之后提交的都去除)
	$ git reset --hard HEAD~3 （将最近3次的提交回滚）

回滚远程分支：
$ git checkout name
$ git pull
$ git branch name_back        //备份当前分支
$ git reset --hard commit_id  //回滚指定提交
$ git push origin : name      //删除远程分支
$ git push origin name        //使用回滚提交重新建立分支
$ git push origin : name_back //删除这个备份分支
