import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from getData import GetDataSet


class client(object):
    """dev:device 选择用于训练的设备"""
    def __init__(self, trainDataSet, dev):
        self.train_ds = trainDataSet
        self.dev = dev
        self.train_dl = None
        self.local_parameters = None

    """
    localEpoch:本地迭代次数
    localBatchSize:本地批处理大小
    Net:构建的训练网络,CNN or 2NN
    lossFun:模型训练损失函数
    opti:优化函数，包括梯度更新方法和学习率
    global_parameters:全局参数
    """
    def localUpdate(self, localEpoch, localBatchSize, Net, lossFun, opti, global_parameters):
        Net.load_state_dict(global_parameters, strict=True)  # 预训练权重加载
        self.train_dl = DataLoader(self.train_ds, batch_size=localBatchSize, shuffle=True)
        for epoch in range(localEpoch):
            for data, label in self.train_dl:
                data, label = data.to(self.dev), label.to(self.dev)  # 选择用于训练的gpu
                preds = Net(data)
                loss = lossFun(preds, label)
                loss.backward()  # 反向传播计算梯度
                opti.step()  # 梯度下降更新参数值
                opti.zero_grad()  # 梯度归0

        return Net.state_dict()  # 返回本地训练模型权重

    def local_val(self):
        pass


class ClientsGroup(object):
    def __init__(self, dataSetName, isIID, numOfClients, dev):
        self.data_set_name = dataSetName
        self.is_iid = isIID
        self.num_of_clients = numOfClients
        self.dev = dev
        self.clients_set = {}

        self.test_data_loader = None

        self.dataSetBalanceAllocation()

    def dataSetBalanceAllocation(self):
        mnistDataSet = GetDataSet(self.data_set_name, self.is_iid)

        test_data = torch.tensor(mnistDataSet.test_data)
        test_label = torch.argmax(torch.tensor(mnistDataSet.test_label), dim=1)
        self.test_data_loader = DataLoader(TensorDataset( test_data, test_label), batch_size=100, shuffle=False)

        train_data = mnistDataSet.train_data
        train_label = mnistDataSet.train_label

        shard_size = mnistDataSet.train_data_size // self.num_of_clients // 2  # 先做除法然后向下取整，为什么要除2分两个分区？
        shards_id = np.random.permutation(mnistDataSet.train_data_size // shard_size)  # 返回一个指定大小的随机排序序列
        for i in range(self.num_of_clients):
            shards_id1 = shards_id[i * 2]
            shards_id2 = shards_id[i * 2 + 1]
            data_shards1 = train_data[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]  # 寻找分区1的数据
            data_shards2 = train_data[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]  # 寻找分区2的数据
            label_shards1 = train_label[shards_id1 * shard_size: shards_id1 * shard_size + shard_size]  # 寻找分区1的标签
            label_shards2 = train_label[shards_id2 * shard_size: shards_id2 * shard_size + shard_size]  # 寻找分区2的标签
            local_data, local_label = np.vstack((data_shards1, data_shards2)), np.vstack((label_shards1, label_shards2))  # 组合两个分区
            local_label = np.argmax(local_label, axis=1)  # 将one-hot转化为10进制标签
            someone = client(TensorDataset(torch.tensor(local_data), torch.tensor(local_label)), self.dev)
            self.clients_set['client{}'.format(i)] = someone

if __name__=="__main__":
    MyClients = ClientsGroup('mnist', True, 100, 1)
    # torch.set_printoptions(threshold=np.inf)
    print(MyClients.clients_set['client10'].train_ds[0:100])
    print(MyClients.clients_set['client11'].train_ds[400:500])


