import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import xlrd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dif_sig(x):
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(h_target, h_output):
    return ((h_target - h_output) ** 2).mean()


class BPNetwork:
    def __init__(self):
        # 输入层到隐含层的第一个神经元的权重和偏置
        self.w11 = np.random.normal()
        self.w21 = np.random.normal()
        self.w31 = np.random.normal()
        self.w41 = np.random.normal()
        self.w51 = np.random.normal()
        self.b1 = np.random.normal()
        # 输入层到隐含层的第二个神经元的权重和偏置
        self.w12 = np.random.normal()
        self.w22 = np.random.normal()
        self.w32 = np.random.normal()
        self.w42 = np.random.normal()
        self.w52 = np.random.normal()
        self.b2 = np.random.normal()
        # 输入层到隐含层的第三个神经元的权重和偏置
        self.w13 = np.random.normal()
        self.w23 = np.random.normal()
        self.w33 = np.random.normal()
        self.w43 = np.random.normal()
        self.w53 = np.random.normal()
        self.b3 = np.random.normal()
        # 隐含层到输出层的权重和偏置
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.b4 = np.random.normal()

    def feedforward(self, x):
        # x是输入值向量,拥有5个元素的向量组
        a1 = sigmoid(self.w11 * x[0] + self.w21 * x[1] + self.w31 * x[2] + self.w41 * x[3] + self.w51 * x[4] + self.b1)
        a2 = sigmoid(self.w12 * x[0] + self.w22 * x[1] + self.w32 * x[2] + self.w42 * x[3] + self.w52 * x[4] + self.b2)
        a3 = sigmoid(self.w13 * x[0] + self.w23 * x[1] + self.w33 * x[2] + self.w43 * x[3] + self.w53 * x[4] + self.b3)
        h_output = sigmoid(self.w1 * a1 + self.w2 * a2 + self.w3 * a3 + self.b4)
        return h_output

    def train(self, data, h_target, record):
        learning_rate = 0.05
        epochs = 400000

        for epoch in range(epochs):
            for x, h_ture in zip(data, h_target):
                sum_a1 = self.w11 * x[0] + self.w21 * x[1] + self.w31 * x[2] + self.w41 * x[3] + self.w51 * x[4] \
                         + self.b1
                a1 = sigmoid(sum_a1)

                sum_a2 = self.w12 * x[0] + self.w22 * x[1] + self.w32 * x[2] + self.w42 * x[3] + self.w52 * x[4] \
                       + self.b2
                a2 = sigmoid(sum_a2)

                sum_a3 = self.w13 * x[0] + self.w23 * x[1] + self.w33 * x[2] + self.w43 * x[3] + self.w53 * x[4] \
                         + self.b3
                a3 = sigmoid(sum_a3)

                sum_h = self.w1 * a1 + self.w2 * a2 + self.w3 * a3 + self.b4
                h_output = sigmoid(sum_h)

                # 计算偏微分
                # 将d_L_d_output 表示为 partial L /partial output
                d_L_d_output = -2 * (h_ture - h_output)

                # h神经元
                d_output_d_w1 = a1 * dif_sig(sum_h)
                d_output_d_w2 = a2 * dif_sig(sum_h)
                d_output_d_w3 = a3 * dif_sig(sum_h)
                d_output_d_b4 = dif_sig(sum_h)

                d_output_d_a1 = self.w1 * dif_sig(sum_h)
                d_output_d_a2 = self.w2 * dif_sig(sum_h)
                d_output_d_a3 = self.w3 * dif_sig(sum_h)

                # a1神经元
                d_a1_d_w11 = x[0] * dif_sig(sum_a1)
                d_a1_d_w21 = x[1] * dif_sig(sum_a1)
                d_a1_d_w31 = x[2] * dif_sig(sum_a1)
                d_a1_d_w41 = x[3] * dif_sig(sum_a1)
                d_a1_d_w51 = x[4] * dif_sig(sum_a1)
                d_a1_d_b1 = dif_sig(sum_a1)

                # a2神经元
                d_a2_d_w12 = x[0] * dif_sig(sum_a2)
                d_a2_d_w22 = x[1] * dif_sig(sum_a2)
                d_a2_d_w32 = x[2] * dif_sig(sum_a2)
                d_a2_d_w42 = x[3] * dif_sig(sum_a2)
                d_a2_d_w52 = x[4] * dif_sig(sum_a2)
                d_a2_d_b2 = dif_sig(sum_a2)

                # a3神经元
                d_a3_d_w13 = x[0] * dif_sig(sum_a3)
                d_a3_d_w23 = x[1] * dif_sig(sum_a3)
                d_a3_d_w33 = x[2] * dif_sig(sum_a3)
                d_a3_d_w43 = x[3] * dif_sig(sum_a3)
                d_a3_d_w53 = x[4] * dif_sig(sum_a3)
                d_a3_d_b3 = dif_sig(sum_a3)

                # 更新权重和偏置
                # a1神经元
                self.w11 -= learning_rate * d_L_d_output * d_output_d_a1 * d_a1_d_w11
                self.w21 -= learning_rate * d_L_d_output * d_output_d_a1 * d_a1_d_w21
                self.w31 -= learning_rate * d_L_d_output * d_output_d_a1 * d_a1_d_w31
                self.w41 -= learning_rate * d_L_d_output * d_output_d_a1 * d_a1_d_w41
                self.w51 -= learning_rate * d_L_d_output * d_output_d_a1 * d_a1_d_w51
                self.b1 -= learning_rate * d_L_d_output * d_output_d_a1 * d_a1_d_b1

                # a2神经元
                self.w12 -= learning_rate * d_L_d_output * d_output_d_a2 * d_a2_d_w12
                self.w22 -= learning_rate * d_L_d_output * d_output_d_a2 * d_a2_d_w22
                self.w32 -= learning_rate * d_L_d_output * d_output_d_a2 * d_a2_d_w32
                self.w42 -= learning_rate * d_L_d_output * d_output_d_a2 * d_a2_d_w42
                self.w52 -= learning_rate * d_L_d_output * d_output_d_a2 * d_a2_d_w52
                self.b2 -= learning_rate * d_L_d_output * d_output_d_a2 * d_a2_d_b2

                # a3神经元
                self.w13 -= learning_rate * d_L_d_output * d_output_d_a3 * d_a3_d_w13
                self.w23 -= learning_rate * d_L_d_output * d_output_d_a3 * d_a3_d_w23
                self.w33 -= learning_rate * d_L_d_output * d_output_d_a3 * d_a3_d_w33
                self.w43 -= learning_rate * d_L_d_output * d_output_d_a3 * d_a3_d_w43
                self.w53 -= learning_rate * d_L_d_output * d_output_d_a3 * d_a3_d_w53
                self.b3 -= learning_rate * d_L_d_output * d_output_d_a3 * d_a3_d_b3

                # h神经元
                self.w1 -= learning_rate * d_L_d_output * d_output_d_w1
                self.w2 -= learning_rate * d_L_d_output * d_output_d_w2
                self.w3 -= learning_rate * d_L_d_output * d_output_d_w3
                self.b4 -= learning_rate * d_L_d_output * d_output_d_b4

            if epoch % 1000 == 0:
                # 按列运算
                h_pre = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(h_target, h_pre)
                record.append(loss)
                print("Epoch %d loss : %.4f" % (epoch, loss))


# 从excel文件中读取数据集
data = xlrd.open_workbook('DataSets.xls')
# 根据分项表格读取4个表格的数据,分为健康、患者训练集和健康、患者测试集
sheet_train_healthy = data.sheets()[0]
sheet_train_unhealthy = data.sheets()[1]
sheet_test_healthy = data.sheets()[2]
sheet_test_unhealthy = data.sheets()[3]


# 定义获取数据集函数
def get_dataset(feature_list, label_list, sheet):
    # 提取该表格的行数,以便将其存入列表当中
    row = sheet.nrows
    for i in range(row):
        # 获取当前行的所有数据值并返回列表temp,由于temp的最后一列元素是标签,
        # 不属于输入值,故舍弃删除,feature_list是输入元素的属性,label_list是标签
        temp = sheet.row_values(i, start_colx=0, end_colx=None)
        del temp[-1]
        feature_list.append(temp)
        label_list.append(sheet.cell_value(i, 5))


# 数据归一化,使其更好收敛
def normal(lists):
    X = np.array(lists)
    min_max_scaler = preprocessing.MinMaxScaler()
    X_minMax = min_max_scaler.fit_transform(X)
    normal_list = np.matrix.tolist(X_minMax)
    return normal_list


# 从四组sheet中提取数据集
# 获取健康人群训练集属性及其标签
train_healthy_features = []
train_healthy_label = []
get_dataset(train_healthy_features, train_healthy_label, sheet_train_healthy)

# 获取患者人群训练集及其标签
train_unhealthy_features = []
train_unhealthy_label = []
get_dataset(train_unhealthy_features, train_unhealthy_label, sheet_train_unhealthy)

# 获取健康人群测试集属性及其标签
test_healthy_features = []
test_healthy_label = []
get_dataset(test_healthy_features, test_healthy_label, sheet_test_healthy)

# 获取患者人群测试集属性及其标签
test_unhealthy_features = []
test_unhealthy_label = []
get_dataset(test_unhealthy_features, test_unhealthy_label, sheet_test_unhealthy)

# 数据拼接
train_data = train_healthy_features + train_unhealthy_features
train_label = train_healthy_label + train_unhealthy_label
# 对数据集进行归一化处理
train_data = normal(train_data)

train_data = np.array(train_data)
train_label = np.array(train_label)


record = []
network = BPNetwork()
# 训练神经网络
network.train(train_data, train_label, record)


# 炼丹之旅---九品炼药师+23种异火还炼不爆你？？？
test_data = test_healthy_features + test_unhealthy_features
test_label = test_healthy_label + test_unhealthy_label
test_data = normal(test_data)

accuracy = 0
correct_num = 0
for j in range(len(test_data)):
    array = np.array(test_data[j])
    predict = network.feedforward(array)
    print("number: %d ,predict: %.5f" % ((j + 1), predict))
    if predict > 0.5:
        pre_label = 1.0
    else:
        pre_label = 0.0
    if pre_label == test_label[j]:
        correct_num += 1

accuracy = correct_num / len(test_data)
print("一共测试 %d 组数据,正确率为：%.4f%%" % (len(test_data), accuracy * 100))
y = np.linspace(0, 400000, 400)
plt.plot(y, record)
plt.show()