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

INSERT INTO 数据库名字.表格名字
(列名...)
VALYES
(数值....)

	USE egg;

	create table eggs_record(
		id INT primary key auto_incrementeggs_record,
   	 	egg_name VARCHAR(10) not null,
    	sold DATE null
	);

	insert into egg.eggs_record(id,egg_name,sold)
	values(1,'鸡蛋','2020-01-01');

	insert into egg.eggs_record(id,egg_name,sold)
	values(2,'鸭蛋','2020-01-02');

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1726379882658.png)

###3.5 改变表格
####3.5.1 添加新数据
ALTER TABLE 表格名字

ADD 列名 数据类型

就可以改变表格

####3.5.2 改变旧数据
UPDATE 数据库名.表格名字
SET 值
WHERE 定位id

	alter table egg.eggs_record
	add stock int null;

	update egg.eggs_record
	set sold='2022-06-06'
	where id=2;

修改结果如下

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1726380323569.png)


###3.6 删除数据
####3.6.1删除一条记录

DELETE FROM 数据库名.表格名
WHERE 索引

	delete from egg.egg_record
	where id=1;

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1726380718183.png)

####3.6.2删除表格

DROP TABLE 数据库名.表格名

####3.6.3删除数据库

DROP DATABASE 数据库名

###3.7 查找数据

####3.7.1 查看表格全部内容

select *
from 表格名

	select *
	from eggs_record;

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1726381174493.png)

####3.7.2 查看表格某列内容

select 列名...
from 表格名

	select id,sold
	from eggs_record;

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1726381243480.png)

####3.7.3 特殊限制
#####3.7.3.1 去重

select distinct *
from 表格名

#####3.7.3.2 排序

select *
from 表格名
order by 列名 asc/desc

就是以列名的 升序/降序 排列

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1726381513098.png)

####3.7.4 过滤
select *
from 表格名
where 条件
order by 列名 asc/desc

有如下的限制内容

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1726381597424.png)

可以进行如下限制

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1726381649236.png)

####3.7.5 合并

select *
from 表格名
inner join 表格名
on 条件

用条件来取两个表格交集--inner  并集--union

保留上表格并且加入下层表格符合条件的--left

保留下表格并且加入上层表格符合条件的--right

