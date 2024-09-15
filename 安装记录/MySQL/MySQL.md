#MySQL安装和核心语法


朱俊泽

2351114


##1MySQL安装

###1.1检查

在安装之前, 先确定一下, 电脑上之前有没有安装MySQL，看看有没有这个路径。

![](https://gitee.com/tj-messi/picture/raw/master/1726372869601.png)

###1.2官网检查下载

之后进入到官方网站进行下载

![](https://gitee.com/tj-messi/picture/raw/master/1726373121934.png)

###1.3安装选择

![](https://gitee.com/tj-messi/picture/raw/master/20240915120745.png)

选择custom模式进行自定义安装

![](https://gitee.com/tj-messi/picture/raw/master/20240915120819.png)

之后选择对应的MySQL server进行下载安装

![](https://gitee.com/tj-messi/picture/raw/master/20240915121605.png)

如上的界面就是安装好了

###1.4work bench安装

安装好扩展work bench

![](https://gitee.com/tj-messi/picture/raw/master/20240915121718.png)

##2 创建与服务器的链接

###2.1 本地服务器创建
直接点击MySQL connection

![](https://gitee.com/tj-messi/picture/raw/master/1726373890914.png)

###2.2 进入本地服务器

直接点击刚刚默认的本地服务器连接进入服务器

##3 MySQL的基本使用

###3.1 创造数据库

CREATE DATABASE 数据库名字

然后快捷执行--execute

就会发现有生成好的数据库
![](https://gitee.com/tj-messi/picture/raw/master/1726378695309.png)
![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1726378778557.png)

如果再次执行同样的创造数据库就会出错，原因是数据库的名字不能一样

###3.2 指定数据库

USE 数据库名字

###3.3 创造表格

CREATE TABLE 表格名字
{

数据名字 类型....

}

典型类型：

INT 整形数据

VARCHAR() 字符串

DATE 时间

	create table eggs_record(
		num INT,
    	egg_name VARCHAR(10),
    	egg_time DATE,
	);
数据类型有空和非空两种属性，NULL和NOT NULL

然后可以设置primary key，主键，必须非空 

	create table eggs_record(
		id INT primary key,
    	egg_name VARCHAR(10) not null,
    	egg_time DATE null,
	);

###3.4 插入数据