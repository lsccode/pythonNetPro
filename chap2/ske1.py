import numpy
import scipy.special
import matplotlib.pyplot


# 定义神经网络类
class neuralNetwork:

    # 初始化函数，设定输入层节点，隐藏层节点和输出层节点数量
    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate):
        # 依次设置输入，隐藏，输出节点的值
        self.iNodes = inputNodes
        self.hNodes = hiddenNodes
        self.oNodes = outputNodes

        # 设置学习率
        self.lr = learningRate

        # 设置神经网络的权重矩阵
        # rand(行数,列数)，权重矩阵中，行数：后者决定，列数：前者决定，如果计算inputNodes和hiddenNodes的权重矩阵，那么i为前者，h为后者

        # 以下是简单的随机初始化权重矩阵
        #self.weights_ih = numpy.random.rand(self.hNodes,self.iNodes) - 0.5
        #self.weights_ho = numpy.random.rand(self.oNodes,self.oNodes) - 0.5

        # 以下是正态分布的随机初始化权重矩阵
        self.weights_ih = numpy.random.normal(0.0, pow(self.hNodes, -0.5), (self.hNodes, self.iNodes))
        self.weights_ho = numpy.random.normal(0.0, pow(self.oNodes, -0.5), (self.oNodes, self.hNodes))

        # 定义激活函数（S函数）
        self.activation_function = lambda x: scipy.special.expit(x)
        pass

    # 网络训练函数，学习给定训练集样本后，优化权重
    def train(self, inputs_list, targets_list):
        # 将输入数据的list和目标数据的list转换为矩阵
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # hidden层的输入数据
        hidden_inputs = numpy.dot(self.weights_ih, inputs)
        # hidden层的输出数据
        hidden_outputs = self.activation_function(hidden_inputs)
        # 最后输出层的输入数据
        final_inputs = numpy.dot(self.weights_ho, hidden_outputs)
        # 最后输出层的输出数据
        final_outputs = self.activation_function(final_inputs)

        # 计算输出结果的误差，（目标数据-实际输出数据）
        output_errors = targets - final_outputs

        # 将输出结果的误差反向传播到hidden层，weights_hidden_output(反转)×errors_output
        hidden_errors = numpy.dot(self.weights_ho.T, output_errors)

        # 这样就会得到2个误差矩阵(hidden_errors和output_errors),2个权重矩阵(weights_ih和weights_ho)，然后根据公式计算每个权重需要调节的量
        # 调节量公式： W_j_k =（学习率 × E_k * sigmoid(O_k) * (1-sigmoid(O_k)) ） * O_j(矩阵反转)
        self.weights_ho += self.lr * numpy.dot((output_errors * final_outputs * (1 - final_outputs)),
                                               numpy.transpose(hidden_outputs))
        self.weights_ih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1 - hidden_outputs)),
                                               numpy.transpose(inputs))

        pass

    # 查询函数，给定输入，从输出节点给出答案
    def query(self, inputs_list):
        # 将传入的list转换成矩阵
        inputs = numpy.array(inputs_list, ndmin=2).T

        # hidden层的输入数据
        hidden_inputs = numpy.dot(self.weights_ih, inputs)
        # hidden层的输出数据
        hidden_outputs = self.activation_function(hidden_inputs)
        # 最后输出层的输入数据
        final_inputs = numpy.dot(self.weights_ho, hidden_outputs)
        # 最后输出层的输出数据
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

# number of input, hidden and output nodes
input_nodes = 3
hidden_nodes = 3
output_nodes = 3
# learning rate is 0. 5
learning_rate = 0.3
# create instance of neural network
n = neuralNetwork(input_nodes,hidden_nodes, output_nodes,learning_rate)

output = n.query([1.0, 0.5, -1.5])
print("output is", output)

#numpy.random.rand(3,3)