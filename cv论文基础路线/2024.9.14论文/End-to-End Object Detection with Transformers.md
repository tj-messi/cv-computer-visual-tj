#End-to-End Object Detection with Transformers

##abstract
在其他方式进行目标检测的时候会在最后一步使用到NMS来取出冗余的框，这会导致模型调参不好调，部署困难。

而transformer实现了端到端的全局建模，直接呈现了一个集合的预测问题