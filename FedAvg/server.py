import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from Models import Mnist_2NN, Mnist_CNN
from clients import ClientsGroup, client
from torch.utils.tensorboard import SummaryWriter
import encrypt
from datetime import datetime


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, description="FedAvg")  # 命令行解析器
parser.add_argument('-g', '--gpu', type=str, default='0', help='gpu id to use(e.g. 0,1,2,3)')
parser.add_argument('-nc', '--num_of_clients', type=int, default=100, help='numer of the clients')
parser.add_argument('-cf', '--cfraction', type=float, default=0.1, help='C fraction, 0 means 1 client, 1 means total clients')
parser.add_argument('-E', '--epoch', type=int, default=20, help='local train epoch')
parser.add_argument('-B', '--batchsize', type=int, default=10, help='local train batch size')
parser.add_argument('-mn', '--model_name', type=str, default='mnist_cnn', help='the model to train')
parser.add_argument('-lr', "--learning_rate", type=float, default=0.01, help="learning rate, \
                    use value from origin paper as default")
parser.add_argument('-vf', "--val_freq", type=int, default=5, help="model validation frequency(of communications)：全局更新参数频率")
parser.add_argument('-sf', '--save_freq', type=int, default=100, help='global model save frequency(of communication)')
parser.add_argument('-ncomm', '--num_comm', type=int, default=1000, help='number of communications')
parser.add_argument('-sp', '--save_path', type=str, default='./checkpoints', help='the saving path of checkpoints')
parser.add_argument('-iid', '--IID', type=int, default=0, help='the way to allocate data to clients')


def test_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)


if __name__=="__main__":
    """tensorboard配置"""
    status = 0  # 是否加DP 默认不加
    TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())  # 时间戳

    args = parser.parse_args()
    args = args.__dict__   # 转化为字典对象

    test_mkdir(args['save_path'])

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    net = None
    if args['model_name'] == 'mnist_2nn':
        net = Mnist_2NN()
    elif args['model_name'] == 'mnist_cnn':
        net = Mnist_CNN()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        net = torch.nn.DataParallel(net)
    net = net.to(dev)

    loss_func = F.cross_entropy  # 交叉熵损失函数
    opti = optim.SGD(net.parameters(), lr=args['learning_rate'])  # 随机梯度下降更新梯度

    myClients = ClientsGroup('mnist', args['IID'], args['num_of_clients'], dev)
    testDataLoader = myClients.test_data_loader

    num_in_comm = int(max(args['num_of_clients'] * args['cfraction'], 1))

    global_parameters = {}
    for key, var in net.state_dict().items():
        global_parameters[key] = var.clone()

    accuracy = []
    for i in range(args['num_comm']):  # 通信次数
        print("communicate round {}".format(i+1))

        order = np.random.permutation(args['num_of_clients'])  # 产生随机序列以随机选取客户端
        clients_in_comm = ['client{}'.format(i) for i in order[0:num_in_comm]]  # 选择本地训练的客户端

        sum_parameters = None
        for client in tqdm(clients_in_comm):  # tqdm:进度条
            local_parameters = myClients.clients_set[client].localUpdate(args['epoch'], args['batchsize'], net,
                                                                         loss_func, opti, global_parameters)
            """DP噪声添加"""
            for key, var in local_parameters.items():
                status = 1
                local_parameters[key] = encrypt.laplace_mech(var, 1, 3)  # 加DP噪声

            if sum_parameters is None:
                sum_parameters = {}
                for key, var in local_parameters.items():
                    sum_parameters[key] = var.clone()  # 不加密就只保留var.clone()
            else:
                for key in sum_parameters:
                    sum_parameters[key] = sum_parameters[key] + local_parameters[key]  # 不加密只保留local_parameters[key]
        for key in global_parameters:
            global_parameters[key] = (sum_parameters[key] / num_in_comm)  #不加密删除torch.tensor

        with torch.no_grad():  # 防止tensor自动求导
            if (i + 1) % args['val_freq'] == 0:
                net.load_state_dict(global_parameters, strict=True)  # 全局更新参数
                sum_accu = 0
                num = 0
                for data, label in testDataLoader:
                    data, label = data.to(dev), label.to(dev)
                    preds = net(data)  # cnn预测
                    preds = torch.argmax(preds, dim=1)  # one-hot转换为1进制标签
                    sum_accu += (preds == label).float().mean()
                    num += 1
                print('accuracy: {}'.format(sum_accu / num))
                if status == 0:
                    log_dir = './log/NoDP/' + TIMESTAMP
                    writer_NoDP = SummaryWriter(log_dir)
                    writer_NoDP.add_scalar('accuracy/FLcnn', sum_accu/num, i)
                else:
                    log_dir = './log/DP/' + TIMESTAMP
                    writer_DP = SummaryWriter(log_dir)
                    writer_DP.add_scalar('accuracy/FLDPcnn', sum_accu/num, i)
        if (i + 1) % args['save_freq'] == 0:
            torch.save(net, os.path.join(args['save_path'],
                                         '{}_num_comm{}_E{}_B{}_lr{}_num_clients{}_cf{}'.format(args['model_name'],
                                                                                                i, args['epoch'],
                                                                                                args['batchsize'],
                                                                                                args['learning_rate'],
                                                                                                args['num_of_clients'],
                                                                                                args['cfraction'])))





