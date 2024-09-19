import numpy as np
import pandas as pd
import tushare as ts


import matplotlib.pyplot as plt
import paddle
import paddle.nn as nn
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error


# 窗口划分
def split_windows(data, size):
    X = []
    Y = []
    # X作为数据，Y作为标签
    # 滑动窗口，步长为1，构造窗口化数据，每一个窗口的数据标签是窗口末端的close值（收盘价格）
    for i in range(len(data) - size):
        X.append(data[i:i + size, :])
        Y.append(data[i + size, 2])
    return np.array(X), np.array(Y)

window_size = 7
fea_num = 5
batch_size = 32
class CNN_LSTM(nn.Layer):
    def __init__(self, window_size, fea_num):
        super().__init__()
        self.window_size = window_size
        self.fea_num = fea_num
        self.conv1 = nn.Conv2D(in_channels=1, out_channels=64, stride=1, kernel_size=3, padding='same')
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2D(kernel_size=2, stride=1, padding='same')
        self.dropout = nn.Dropout2D(0.3)

        self.lstm1 = nn.LSTM(input_size=64*fea_num, hidden_size=128, num_layers=1, time_major=False)
        self.lstm2 = nn.LSTM(input_size=128, hidden_size=64, num_layers=1, time_major=False)
        self.fc = nn.Linear(in_features=64, out_features=32)
        self.relu2 = nn.ReLU()
        self.head = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        x = x.reshape([x.shape[0], 1, self.window_size, self.fea_num])
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.dropout(x)

        x = x.reshape([x.shape[0], self.window_size, -1])
        x, (h, c) = self.lstm1(x)
        x, (h,c) = self.lstm2(x)
        x = x[:,-1,:] # 最后一个LSTM只要窗口中最后一个特征的输出
        x = self.fc(x)
        x = self.relu2(x)
        x = self.head(x)

        return x

def process(data, bs):
    l = len(data)
    tmp = []
    for i in range(0, l, bs):
        if i + bs > l:
            return np.array(tmp)
            # tmp.append(data[i:].tolist())
        else:
            tmp.append(data[i:i + bs].tolist())
    # 确保所有子列表长度一致，或者不进行数组转换
    tmp = np.array(tmp)
    return tmp

if __name__ == '__main__':
    # 获取中国平安三年内K线数据
    pro = ts.pro_api('79b210fc24a8301fada373f3926eaba381415380c2a50dbfe44fc47b')

    df = pro.daily(ts_code='000908.SZ', start_date='20200919', end_date='20240919')
    df.index = pd.to_datetime(df.trade_date)
    data = df[['open', 'high', 'close', 'low', 'pct_chg', 'vol']]
    # 获取到的数据是时间逆序的，这里将其翻转并重置索引
    data = data[::-1].reindex()
    data.to_csv('./data.csv')

    df = pd.read_csv('./data.csv', usecols=['open', 'high', 'close', 'low', 'vol'])
    all_data = df.values
    train_len = 500
    train_data = all_data[:train_len, :]
    test_data = all_data[train_len:, :]

    plt.figure(figsize=(12, 8))
    # 数据可视化
    plt.plot(np.arange(train_data.shape[0]), train_data[:, 2], label='train data')
    plt.plot(np.arange(train_data.shape[0], train_data.shape[0] + test_data.shape[0]), test_data[:, 2],
             label='test data')
    plt.legend()
    # plt.show()

    # normalizatioin processing
    scaler = preprocessing.MinMaxScaler()
    scaled_train_data = scaler.fit_transform(train_data)
    # 使用训练集的最值对测试集归一化，保证训练集和测试集的分布一致性
    scaled_test_data = scaler.transform(test_data)
    # 训练集测试集划分
    window_size = 7
    train_X, train_Y = split_windows(scaled_train_data, size=window_size)
    test_X, test_Y = split_windows(scaled_test_data, size=window_size)
    print('train shape', train_X.shape, train_Y.shape)
    print('test shape', test_X.shape, test_Y.shape)

    model = CNN_LSTM(window_size, fea_num)
    paddle.summary(model, (99, 7, 5))

    # 定义超参数
    base_lr = 0.005
    BATCH_SIZE = 24
    EPOCH = 200
    lr_schedual = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=base_lr, T_max=EPOCH, verbose=True)
    loss_fn = nn.MSELoss()
    metric = paddle.metric.Accuracy()
    opt = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=lr_schedual, beta1=0.9, beta2=0.999)

    # 处理数据集
    train_X = process(train_X, 32)
    train_Y = process(train_Y, 32)
    print(train_X.shape, train_Y.shape)

    # 模型训练
    for epoch in range(EPOCH):
        model.train()
        loss_train = 0
        for batch_id, data in enumerate(train_X):
            label = train_Y[batch_id]
            data = paddle.to_tensor(data, dtype='float32')
            label = paddle.to_tensor(label, dtype='float32')
            label = label.reshape([label.shape[0], 1])
            y = model(data)

            loss = loss_fn(y, label)
            opt.clear_grad()
            loss.backward()
            opt.step()
            loss_train += loss.item()
        print("[TRAIN] ========epoch : {},  loss: {:.4f}==========".format(epoch + 1, loss_train))
        lr_schedual.step()

    # 保存模型参数
    paddle.save(model.state_dict(), 'work/cnn_lstm_ep200_lr0.005.params')
    paddle.save(lr_schedual.state_dict(), 'work/cnn_lstm_ep200_lr0.005.pdopts')

    # 加载模型
    model = CNN_LSTM(window_size, fea_num)
    model_dict = paddle.load('work/cnn_lstm_ep200_lr0.005.params')
    model.load_dict(model_dict)

    test_X = paddle.to_tensor(test_X, dtype='float32')
    prediction = model(test_X)
    prediction = prediction.cpu().numpy()
    prediction = prediction.reshape(prediction.shape[0], )
    # 反归一化
    scaled_prediction = prediction * (scaler.data_max_[2] - scaler.data_min_[2]) + scaler.data_min_[2]
    scaled_true = test_Y * (scaler.data_max_[2] - scaler.data_min_[2]) + scaler.data_min_[2]
    # 画图
    plt.plot(range(len(scaled_true)), scaled_true, label='true')
    plt.plot(range(len(scaled_prediction)), scaled_prediction, label='prediction', marker='*')
    plt.legend()
    plt.show()
    print('RMSE', np.sqrt(mean_squared_error(scaled_prediction, scaled_true)))
