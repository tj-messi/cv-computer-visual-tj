#Literature Review of RNN-Time-series-Anomaly-Detection paper

##Intruduction
Time-series anomaly detection has been an active area of research in recent years, with applications spanning diverse domains such as finance, healthcare, transportation, and industrial monitoring. With the proliferation of data-driven systems, the need for effective anomaly detection techniques has become increasingly important.

##Situation
Previous studies have explored various machine learning approaches for time-series anomaly detection, including traditional statistical methods[1], distance-based algorithms, density-based models[2], and more recently, deep learning techniques. Among these, Recurrent Neural Networks (RNNs) have shown promising results due to their ability to capture temporal dependencies in time-series data.

Several researchers have investigated the use of RNNs for anomaly detection. Malhotra et al. (2015, 2016) demonstrated the effectiveness of Long Short-Term Memory (LSTM) networks, a type of RNN, in detecting anomalies in time-series data. Their work focused on using LSTM networks for multi-step prediction and anomaly scoring, where anomalies were identified as deviations from the predicted values.

In addition to LSTM networks, other variants of RNNs have also been explored for anomaly detection. For instance, the proposed RNN-based model in this paper (Park, 2018) utilizes a stacked RNN architecture to recursively predict future values of time-series data. This model then calculates anomaly scores based on the deviation between the predicted and actual values, using a multivariate Gaussian distribution model.

While these approaches have shown promising results, they also have limitations. For instance, traditional RNNs can suffer from vanishing or exploding gradient problems, which can impact their performance over long time-series data. To address this issue, more recent studies have explored the use of advanced RNN variants, such as Gated Recurrent Units (GRUs), which have been found to be more robust to these issues.

Overall, the literature on time-series anomaly detection using RNNs is rich and diverse, with many promising approaches and techniques being proposed. However, there is still a need for further research to develop more accurate and efficient models that can effectively detect anomalies in real-world time-series data. The proposed RNN-based model in this paper aims to contribute to this field by offering a two-stage strategy comprising time-series prediction and anomaly score calculation.

##
[1]曹晨曦,田友琳,张昱堃,等.基于统计方法的异常点检测在时间序列数据上的应用[J].合肥工业大学学报(自然科学版),2018,41(09):1284-1288.

[2]孙梅玉.基于距离和密度的时间序列异常检测方法研究[J].计算机工程与应用,2012,48(20):11-17+22.
