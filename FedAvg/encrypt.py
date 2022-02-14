import binascii

import numpy as np
import torch
import rsa
import pyunit_prime as pp
import math
import gmpy2
import libnum
from phe import paillier


# 计算拉普拉斯噪声
def laplace_noisy(sensitivety, epsilon):
    n_value = np.random.laplace(0, sensitivety / epsilon, 1)
    return n_value


# 计算基于拉普拉斯加噪的混淆值 运行的也太慢了
def laplace_mech(data, sensitivety, epsilon):
    dim = data.ndim
    if dim == 1:
        dim0 = data.shape[0]
        noisy = np.zeros(dim0)
        for i in range(dim0):
            noisy[i] = 1 + laplace_noisy(sensitivety, epsilon)
            # noisy[i] = laplace_noisy(sensitivety, epsilon)
    elif dim == 2:
        dim0, dim1 = data.shape[0], data.shape[1]
        noisy = np.zeros((dim0, dim1))
        for i in range(dim0):
            for j in range(dim1):
                noisy[i][j] = 1 + laplace_noisy(sensitivety, epsilon)
                # noisy[i][j] = laplace_noisy(sensitivety, epsilon)
    elif dim == 3:
        dim0, dim1, dim2 = data.shape[0], data.shape[1], data.shape[2]
        noisy = np.zeros((dim0, dim1, dim2))
        for i in range(dim0):
            for j in range(dim1):
                for k in range(dim2):
                    noisy[i][j][k] = 1 + laplace_noisy(sensitivety, epsilon)
                    # noisy[i][j][k] = laplace_noisy(sensitivety, epsilon)
    else:
        dim0, dim1, dim2, dim3 = data.shape[0], data.shape[1], data.shape[2], data.shape[3]
        noisy = np.zeros((dim0, dim1, dim2, dim3))
        for i in range(dim0):
            for j in range(dim1):
                for k in range(dim2):
                    for p in range(dim3):
                        noisy[i][j][k][p] = 1 + laplace_noisy(sensitivety, epsilon)
                        # noisy[i][j][k][p] = laplace_noisy(sensitivety, epsilon)
    return torch.mul(data, torch.from_numpy(noisy).cuda())
    # return torch.add(data, torch.from_numpy(noisy).cuda())


# def exgcd(a, b):
#     if b == 0:
#         return 1, 0, a
#     else:
#         x, y, q = exgcd(b, a % b)
#         x, y = y, (x - (a // b) * y)
#         return x, y, q
#
#
# # 扩展欧几里得求逆元
# def ModReverse(a, p):
#     x, y, q = exgcd(a, p)
#     if q != 1:
#         raise Exception("No solution.")
#     else:
#         return (x + p) % p  # 防止负数

"""
class Encryption(object):
    def __init__(self, security_parameter):
        self.security_parameter = security_parameter
        self.p = pp.get_large_prime_length(self.security_parameter)
        self.q = pp.get_large_prime_length(self.security_parameter)
        self.p_inverse = gmpy2.invert(self.p, self.q)
        self.q_inverse = gmpy2.invert(self.q, self.p)
        self.N = self.p * self.q

    def KGen(self, security_parameter):
        self.p = pp.get_large_prime_length(self.security_parameter)
        self.q = pp.get_large_prime_length(self.security_parameter)
        self.N = self.p * self.q
        self.p_inverse = gmpy2.invert(self.p, self.q)
        self.q_inverse = gmpy2.invert(self.q, self.p)

    def encrypt(self, data):

        str_int = libnum.s2n(str(data))  # 字符串转整数
        print(str_int)
        omegap = str_int % self.p
        omegaq = str_int % self.q
        omega = (self.q_inverse * self.q * omegap ^ (self.p) + self.p_inverse * self.p * omegap ^ (self.q)) % self.N
        print(omega)
        return omega

    def decrypt(self, data):
        omegap = data % self.p
        omegaq = data % self.q
        omega = (self.q_inverse * self.q * omegap  + self.p_inverse*self.p*omegaq) % self.N
        print(omega)
        int_str = libnum.n2s(int(omega))  # 整数转字符串并还原数据
        # int_str = binascii.unhexlify(hex(omega)[2:]).decode('utf-8')  # 十进制转bytes还原数据
        return int_str
"""

""" paillier同态加密 """
class Encryption(object):
    def __init__(self, security_parameter):
        self.public_key, self.private_key = paillier.generate_paillier_keypair()

    def encrypt(self, data):
        return [self.public_key.encrypt(x) for x in data]

    def decrypt(self, data):
        return [self.private_key.decrypt(x) for x in data]


if __name__ == "__main__":
    # myen = Encryption(20)
    # print(myen.p)
    # print(myen.q)
    # print(myen.N)
    # a = [0.12, 0.32, 0.085, 0.745]
    # b = [0.12, 0.32, 0.085, 0.745]
    # c = [0.24, 0.64, 0.17, 1.49]
    # a = 0.12
    # b = 0.12
    # c = 0.24
    # add = myen.encrypt(a) + myen.encrypt(b)
    # str = myen.decrypt(add)
    # print(str)

    # generate a public key and private key pair
    public_key, private_key = paillier.generate_paillier_keypair()

    # start encrypting numbers
    secret_number_list = [3.141592653, 300, -4.6e-12]
    b = [3.141592653, 300, -4.6e-12]
    encrypted_number_list = [public_key.encrypt(x) for x in secret_number_list]
    print(encrypted_number_list)
    en_b = [public_key.encrypt(x) for x in b]
    encrypted_all_number_list = map(lambda a, b: a + b, encrypted_number_list, en_b)
    # decrypt encrypted number and print
    print([private_key.decrypt(x) for x in encrypted_all_number_list])
