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

####分页

对于摘要页等特殊内容页面，我们通常令其独立成页。这就需要用到\newpage命令：

	\newpage

####缩进与行高

LaTeX很有趣，它会默认部分命令后的文本需要缩进或无需缩进。比如在以后要提到的\section命令，LaTeX默认其后的首段强制不缩进，后续段落缩进两格。因此，缩进的相关命令非常重要。

缩进的相关命令分为全局命令与局部命令，且局部命令优先级高于全局命令。

全局命令：设置后，全局文字都将采用该缩进方式，用在序言部分。
局部命令：设置后，该段文字将采用该缩进方式，用在正文部分（不）需要缩进的段首。

	%全局命令：
	\setlength{\parindent}{0em}           %段首不缩进
	\setlength{\parindent}{2em}           %段首缩进两字符
	 
	%局部命令：
	\noindent                             %取消缩进
	\indent\setlength{\parindent}{2em}    %缩进两字符

对于段落，我们还有一个重要的参数没有涉及，那就是行高。

通常情况下，我们无需另外设置行高，只需使用默认值即可。但面对特殊需求，我们可以在序言部分加入以下命令修改行高

	\renewcommand{\baselinestretch}{行距倍数}

####字体
字体：实际使用中我们更多是根据需求导入相关的字体包，以最为经典的Times New Roman字体为例，我们直接在序言区拷贝相关字体的LaTeX代码即可。

	%设置字体为Times New Roman
	\usepackage{times}
	 
	%主体中正文和数学公式都将以 Times 字体排版，其他仍以默认字体排版
	\usepackage{mathptmx}

字号：当我们在 \docunment{article} 选定基础字号后，就无需再关注全文的字体大小。（后续在章节部分也会讲到）
加粗：LaTeX中的粗体文本使用 \textbf{...} 命令编写。
斜体：LaTeX 中的斜体文本使用 \textit{...} 命令编写。
下划线：LaTeX 中的下划线文本使用 \underline{...}命令编写。

	%粗体
	\textbf{...} 
	 
	%斜体
	\textit{...} 
	 
	%下划线
	\underline{...}

####对齐方式

无论是文本还是图片，都要面对“对齐”的审判。

我们常见的对齐方式有两种，一种是添加环境，一种是段前添加对齐命令。

添加环境的对齐方式更适合文本，其相关命令如下：

	
	\begin{对齐方式}
	......
	\end{对齐方式}
	
	对齐方式：包括center、flushleft、flushleft三种。

####分块

论文写作过程中，我们需要清楚地告诉编辑器现在写的内容属于论文的哪一部分。是摘要？还是第一章“背景”？还是 第三章“模型建立”……

这时候就需要我们编写相关环境代码。
此外，章节经常会存在层次关系。比如，“第三章第一节”中，相比“第一节”，第三章应该是更高一级的概念，字号应该更大，且能够包含很多小节的内容。

这时候就需要我们编写相关层级代码

通过建立document环境可以告诉编辑器，现在是正文部分：

	\begin{document}
	......
	\end{document}

通过建立abstract环境可以告诉编辑器，现在是摘要部分：

	\begin{abstract}
	......
	\end{abstract}

LaTeX通常将论文分为三个层级，通常是部分、子部分、子子部分。对应的命令为： \section{}、\subsection{}、\subsubsection{}，括号内为该部分的名称。

LaTeX会自动根据层级关系为你适配内容的对应字号大小，父章节会比子章节字号大一些。总之，当我们通过命令申明内容对应的论文部分后，层级关系就会一目了然。


	%部分
	\section{章节名称} 
	 
	%子部分
	\subsection{子章节名称} 
	 
	%子子部分
	\subsubsection{子子章节名称}

####图片

图片是论文中不可或缺的一部分。

首先，我们需要在操作栏创建新的文件夹（New Folder），并对其进行命名，以images为例

先创建文件夹

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20250116154648.png)

通过拖拽或选择将目标图片进行上传，成功后可以看到images文件夹中已经有了我们的图片：（我选择的是名为scenery的jpg格式的图片）

通过导入graphicx宏包可以完成添加图片功能，通过设置图片路径可以使得系统定位到图片所在的位置。如果是images文件夹，则图片位置为\graphicspath{ {images/} }，其余同理。

	%导入与图片相关的宏包
	\usepackage{graphicx}
	 
	%设置图片路径
	\graphicspath{图片位置}

然后导入图片

	\includegraphics[宽度,高度]{图片名称}

指定长度：[width=4cm,height=5cm]
指定比例：[width=0.8\textwidth,height=0.5\textwidth]

![](https://cdn.jsdelivr.net/gh/tj-messi/picture/20250116154910.png)