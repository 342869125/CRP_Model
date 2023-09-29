import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import numpy as np

import torch.nn as nn

import sys
sys.path.append('D:\\DilateMaster')#添加根目录

from data.synthetic_dataset import create_synthetic_dataset,create_synthetic_dataset_test, create_synthetic_dataset_without_choose,create_synthetic_dataset_circle

import warnings; warnings.simplefilter('ignore')

from Generator_model.Generator import Generate_reputation,Generate_reputation_1#,Generate_non_reputation
import torch.optim as optim
import matplotlib.pyplot as plt
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def plot_synthetic_data(X_input, X_target, breakpoints, sample_index):
    input_series = X_input[sample_index]
    output_series = X_target[sample_index]
    breakpoint = breakpoints[sample_index]

    plt.figure(figsize=(12, 5))
    plt.plot(range(len(input_series)), input_series, label="Input Series")
    plt.plot(range(len(input_series), len(input_series) + len(output_series)), output_series, label="Output Series")
    plt.axvline(x=len(input_series), color='r', linestyle='--', label='Input-Output Boundary')
    #plt.axvline(x=breakpoint, color='g', linestyle='--', label='Breakpoint')
    plt.legend()
    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.title("Synthetic Time Series Data")
    plt.show()

# Generate synthetic dataset
N = 1000#总的数据集的个数，文章中是五百个用于训练，五百个用于验证，五百个用于测试
N_input = 40#输入的步长总共有40个步长，前20个是输入即用0-19是用来预测（20-39），并与真实的20-39作比
N_output = 40#预测输出的步长
sigma = 0.01#方差为 0.01 的加性高斯白噪声破坏
gamma = 0.01
parameter,X_train_input, X_train_target, X_test_input, X_test_target, train_breakpoints, test_breakpoints = create_synthetic_dataset_circle(N, N_input, N_output, sigma)
# Plot a sample from the dataset
sample_index = 0
plot_synthetic_data(X_train_input, X_train_target, train_breakpoints, sample_index)
np.savetxt("./Deilate_train_input_N500_17.txt", X_train_input, fmt='%f')
np.savetxt("./Deilate_train_traget_N500_17.txt", X_train_target, fmt='%f ')
np.savetxt("./Deilate_test_input_N500_17.txt", X_test_input, fmt='%f')
np.savetxt("./Deilate_test_traget_N500_17.txt", X_test_target, fmt='%f')
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, padding=dilation * (kernel_size - 1),
                                dilation=dilation)
    def forward(self, x):
        x = self.conv1d(x)
        return x[:, :, :-self.conv1d.padding[0]]
# 定义TCN模型
class TCN(nn.Module):
    def __init__(self, num_layers, in_channels, out_channels, kernel_size, hidden_channels):
        super(TCN, self).__init__()
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            dilation = 2 ** i
            in_channels = in_channels if i == 0 else hidden_channels
            self.layers.append(CausalConv1d(in_channels, hidden_channels, kernel_size, dilation))
            self.layers.append(nn.ReLU())

        self.final_layer = nn.Conv1d(hidden_channels, out_channels, 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.final_layer(x)
        return x

num_layers = 3
in_channels = 1
out_channels = 1
kernel_size = 2
hidden_channels = 16
learning_rate = 0.001

X_train_input = torch.from_numpy(X_train_input).unsqueeze(1).float()
X_train_target = torch.from_numpy(X_train_target).unsqueeze(1).float()
X_test_input = torch.from_numpy(X_test_input).unsqueeze(1).float()
X_train_input=X_train_input.to(device)
X_train_target=X_train_target.to(device)
X_test_input=X_test_input.to(device)
criterion=nn.MSELoss()
model = TCN(num_layers, in_channels, out_channels, kernel_size, hidden_channels).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
####通过TCN得到的output_是因果因子S
for epoch in range(250):
    # 前向传播
    output_ = model(X_train_input)
    # 计算损失
    loss = criterion(output_, X_train_target)
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    optimizer.step()
    model.eval()
    with torch.no_grad():
        output_val = model(X_test_input)
        # loss_val = criterion(output_val, A3)
    print(f'Epoch: {epoch + 1}, Loss: {loss.item():.4f}')
output_=output_.squeeze()
output_val=output_val.squeeze()
X_train_target=X_train_target.squeeze()
output_=output_.to('cpu')
X_train_target=X_train_target.to('cpu')
output_val=output_val.to('cpu')
output_=output_.detach().numpy()
X_train_target=X_train_target.numpy()
output_val=output_val.detach().numpy()
X_train_input_G_1,Y_train_input_G_1 = Generate_reputation_1(output_, X_train_target,output_,parameter[:len(output_)])#第一个位置和第二个位置被放进带事件注意力的transformer中
X_test_input_G_1,Y_test_input_G_1 = Generate_reputation_1(output_, X_train_target,output_val,parameter[:len(output_val)])
np.savetxt("./Deilate_train_input_N500_X_17_C.txt", X_train_input_G_1, fmt='%f')
np.savetxt("./Deilate_test_input_N500_X_17_C.txt", X_test_input_G_1, fmt='%f')
np.savetxt("./Deilate_train_input_N500_Y_17_C.txt", Y_train_input_G_1, fmt='%f')#应该只有三个C和I(train:X and Y；test:X)而不是四个C和I
np.savetxt("./Deilate_test_input_N500_Y_17_C.txt", Y_test_input_G_1, fmt='%f')
X_train_input_G_0,Y_train_input_G_0 = Generate_reputation(output_, X_train_target,output_,parameter[:len(output_)])
X_test_input_G_0,Y_test_input_G_0 = Generate_reputation(output_, X_train_target,output_val,parameter[:len(output_val)])
np.savetxt("./Deilate_train_input_N500_X_17_I.txt", X_train_input_G_0, fmt='%f')
np.savetxt("./Deilate_test_input_N500_X_17_I.txt", X_test_input_G_0, fmt='%f')
np.savetxt("./Deilate_train_input_N500_Y_17_I.txt", Y_train_input_G_0, fmt='%f')
np.savetxt("./Deilate_test_input_N500_Y_17_I.txt", Y_test_input_G_0, fmt='%f')
print('........................................................')

parameter_v1,X_train_input_v1, X_train_target_v1, X_test_input_v1, X_test_target_v1, train_breakpoints_v1, test_breakpoints_v1 = create_synthetic_dataset_test(N, N_input, N_output, sigma)
np.savetxt("./Deilate_train_input_N500_18.txt", X_train_input_v1, fmt='%f')
np.savetxt("./Deilate_train_traget_N500_18.txt", X_train_target_v1, fmt='%f ')
np.savetxt("./Deilate_test_input_N500_18.txt", X_test_input_v1, fmt='%f')
np.savetxt("./Deilate_test_traget_N500_18.txt", X_test_target_v1, fmt='%f')

criterion_unfact=nn.MSELoss()
model_unfact = TCN(num_layers, in_channels, out_channels, kernel_size, hidden_channels)
optimizer_unfact = optim.Adam(model.parameters(), lr=learning_rate)
X_train_input_v1=torch.from_numpy(X_train_input_v1).unsqueeze(1).float()
X_train_target_v1=torch.from_numpy(X_train_target_v1).unsqueeze(1).float()
X_test_input_v1=torch.from_numpy(X_test_input_v1).unsqueeze(1).float()
for epoch in range(250):
    # 前向传播
    output_v1 = model_unfact(X_train_input_v1)
    # 计算损失
    loss_v1= criterion_unfact(output_v1, X_train_target_v1)
    # 反向传播和优化
    optimizer_unfact.zero_grad()
    loss_v1.backward(retain_graph=True)
    optimizer_unfact.step()
    model_unfact.eval()
    with torch.no_grad():
        output_val_v1 = model_unfact(X_test_input_v1)
        # loss_val = criterion(output_val, A3)
    print(f'Epoch: {epoch + 1}, Loss: {loss.item():.4f}')

output_v1=output_v1.squeeze()
output_val_v1=output_val_v1.squeeze()
X_train_target_v1=X_train_target_v1.squeeze()
output_v1=output_v1.detach().numpy()
X_train_target_v1=X_train_target_v1.numpy()
output_val_v1=output_val_v1.detach().numpy()
X_train_input_G_1_v1,Y_train_input_G_1_v1 = Generate_reputation_1(output_v1, X_train_target_v1,output_v1,parameter_v1[:len(X_train_input_v1)])
X_test_input_G_1_v1,Y_test_input_G_1_v1 = Generate_reputation_1(output_v1, X_train_target_v1,output_val_v1,parameter_v1[:len(X_train_input_v1)])
np.savetxt("./Deilate_train_input_N500_X_18_C.txt", X_train_input_G_1_v1, fmt='%f')
np.savetxt("./Deilate_test_input_N500_X_18_C.txt", X_test_input_G_1_v1, fmt='%f')
np.savetxt("./Deilate_train_input_N500_Y_18_C.txt", Y_train_input_G_1_v1, fmt='%f')
np.savetxt("./Deilate_test_input_N500_Y_18_C.txt", Y_test_input_G_1_v1, fmt='%f')
X_train_input_G_0_v1,Y_train_input_G_0_v1 = Generate_reputation(output_v1, X_train_target_v1,output_v1,parameter_v1[:len(X_train_input_v1)])
X_test_input_G_0_v1,Y_test_input_G_0_v1 = Generate_reputation(output_v1, X_train_target_v1,output_val_v1,parameter_v1[:len(X_train_input_v1)])
np.savetxt("./Deilate_train_input_N500_X_18_I.txt", X_train_input_G_0_v1, fmt='%f')
np.savetxt("./Deilate_test_input_N500_X_18_I.txt", X_test_input_G_0_v1, fmt='%f')
np.savetxt("./Deilate_train_input_N500_Y_18_I.txt", Y_train_input_G_0_v1, fmt='%f')
np.savetxt("./Deilate_test_input_N500_Y_18_I.txt", Y_test_input_G_0_v1, fmt='%f')
# X_train_input_G_1 = Generate_non_reputation(X_train_input, X_train_target,parameter[len(X_train_input):])
# X_test_input_G_1 = Generate_non_reputation(X_test_input, X_test_target,parameter[len(X_train_input):])
# np.savetxt("./Deilate_train_input_500_G_15_1.txt", X_train_input_G_1, fmt='%f')
# np.savetxt("./Deilate_test_input_500_G_15_1.txt", X_test_input_G_1, fmt='%f')