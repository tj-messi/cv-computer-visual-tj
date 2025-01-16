#overleaf

##创建论文项目

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/1737010066912.png)

##主内容

####申明文档类型
通常情况下，我们将以下代码置于首行，申明文档类型、文章的基础字号、纸张类型。

	\documentclass[字号,纸张类型]{文本类型}
	\documentclass{article}

文本类型：可选参数有article、book、report等，我们选择article。

字号：可选参数有9pt、10pt、11pt、12pt，默认为10pt。

纸张类型：可选参数有letter paper、a4paper、legal paper等。通常缺省选择默认值。


####导入宏包
宏包是 Latex 发行版的插件功能，通过安装宏包可以扩展或提供更多的功能。多数情况下，我们简称宏包为“包”。我们需要通过以下代码导入宏包。

	\usepackage[可选选项]{包名}

比如：

	%文档的编码选择utf8。
	\usepackage[utf8]{inputenc}
	 
	%使用graphicx包添加图片
	\usepackage{graphicx}
	 
	%设置字体为Times New Roman（英文论文的经典字体）
	\usepackage{times}
	\usepackage{mathptmx}
	 
	%数学包
	\usepackage{amsmath}

####标题，作者，日期（信息申明）

	\title{标题名称}
	\author{作者}
	\date{日期}

输入这些内容

###正文部分
####分区

正文部分即代码段 \begin{document} 到 \end{document} （蓝色框）的内容，默认代码已经给出架构，我们可以直接进行编辑。

我们可以看到正文部分的以下代码：

	\maketitle

这段代码使我们在序言部分设置的标题、作者、日期得以显示，具有重要的作用。

####环境

LaTeX中有一个重要的概念“环境”，即使用\begin{ }和\end{ }两个命令包裹代码块，使文本内容具有特殊格式或对内容进行标识。其格式如下：

	\begin{类型}
	......
	\end{类型}

####注释

latex和overleaf里面的注释用%号！

####空格
对于空格，在LaTeX中无论多少个空格（space键）都会被认为是一个空格。因而当我们需要行内键入一段空白时，需要通过其他命令实现。

	\qquad	更更大空格
	\quad	更大空格
	\+space键	大空格
	\;	中空格
	\,	小空格

####换行

对于换行，在LaTeX中单个“Enter”键并没有真正的换行效果。我们在编辑区键入“Enter”，在编辑栏可以看到文本内容被分割，但实际上并没有空格效果。

要两个enter即可

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20250116152655.png)