#LSTM-based encoder-decoder for multi-sensor anomaly detection（基于lstm的多传感器异常检测编码器）

---
##综述
本文展示了一种基于长短期记忆网络的编码器-解码器的异常检测方案（EncDec-AD)

该方案学习重建“正常”时间序列行为，使用重建误差来检测异常。

我们证明了EncDecAD具有鲁棒性，可以从可预测、不可预测、周期、非周期和准周期时间序列中检测异常。

---
##introduction
LSTM网络(Hochreiter & Schmidhuber, 1997)是一种循环模型，已用于许多序列学习任务，如手写识别、语音识别和情感分析。

提出了一种基于lstm的多传感器时间序列异常检测编码器(EncDecAD)方案。编码器学习输入时间序列的向量表示，解码器使用该表示来重建时间序列。