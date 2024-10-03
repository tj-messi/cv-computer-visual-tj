# RNN-Time-series-Anomaly-Detection

##ABSTACT

In this paper, we have implemented time series anomaly detection based on RNN neural networks within PyTorch.The proposed model employs a two-stage strategy consisting of time-series prediction and anomaly score calculation. Firstly, a stacked RNN model is trained on a time-series dataset that does not contain anomalies. This model is then used to recursively predict future values in the time-series. Subsequently, in the second stage, anomaly scores are computed using a multivariate Gaussian distribution model fitted on the predicted time-series. The model is evaluated on multiple datasets including NYC taxi passenger count, electrocardiograms (ECGs), 2D gesture videos, respiration data, space shuttle valve time-series, and power demand data. Experimental results demonstrate the effectiveness of the proposed model in detecting anomalies in these time-series datasets. 

The proposed model offers a robust solution for detecting anomalies in time-series data, particularly in domains where real-time monitoring and rapid response are crucial. It is applicable to various real-world scenarios, including but not limited to healthcare, transportation, and industrial settings. Future work includes optimizing the model
