import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import torch
print(torch.__version__)  #注意是双下划线
import torch
import torch.nn as nn
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
train_X_C = np.loadtxt("./Sys_data/Deilate_train_input_N500_X_17_C.txt", dtype=np.float32)
train_X_I = np.loadtxt("./Sys_data/Deilate_train_input_N500_X_17_I.txt", dtype=np.float32)
train_X = np.loadtxt("./Sys_data/Deilate_train_input_N500_17.txt", dtype=np.float32)
train_Y_C = np.loadtxt("./Sys_data/Deilate_train_input_N500_Y_17_C.txt", dtype=np.float32)
train_Y_I = np.loadtxt("./Sys_data/Deilate_train_input_N500_Y_17_I.txt", dtype=np.float32)
train_Y = np.loadtxt("./Sys_data/Deilate_train_traget_N500_17.txt", dtype=np.float32)
test_X_C = np.loadtxt("./Sys_data/Deilate_test_input_N500_X_17_C.txt", dtype=np.float32)
test_X_I = np.loadtxt("./Sys_data/Deilate_test_input_N500_Y_17_I.txt", dtype=np.float32)
test_X = np.loadtxt("./Sys_data/Deilate_test_input_N500_17.txt", dtype=np.float32)
test_Y = np.loadtxt("./Sys_data/Deilate_test_traget_N500_17.txt", dtype=np.float32)
####引入数据
train_Y_C = torch.FloatTensor(train_Y_C)
train_Y_I = torch.FloatTensor(train_Y_I)
test_Y = torch.FloatTensor(test_Y)  # 假设 test_Y 是 500*20 的测试集

# Step 2: 定义Transformer模型


#####训练好了从train_Y_I、train_Y_C到train_Y的MLP
test_X_C = torch.FloatTensor(test_X_C)
train_X_C = torch.FloatTensor(train_X_C)
# loss_function_MLP_C = nn.MSELoss()
#
# input_size = train_X_C.shape[1]
# output_size =train_Y_C.shape[1]
# hidden_size = 64
# mlp_model_C = MLP(input_size, hidden_size, output_size)
# optimizer = optim.Adam(mlp_model.parameters(), lr=0.001)
# num_epochs = 300
# for epoch in range(num_epochs):
#    # 前向传播
#    inputs = train_X_C
#    outputs = mlp_model_C(inputs)
#
#    # 计算损失函数
#    loss = loss_function_MLP_C(outputs, train_Y_C)
#
#    # 反向传播及优化
#    optimizer.zero_grad()
#    loss.backward()
#    optimizer.step()
#
#    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
# torch.save(mlp_model.state_dict(), 'trained_mlp_model.pth')

# Step 2: 定义VAE模型#ML
# class VAE(nn.Module):
#     def __init__(self, input_size, hidden_size, latent_size):
#         super(VAE, self).__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(input_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, latent_size * 2)  # 2 * latent_size for mean and log_variance
#         )
#         self.decoder = nn.Sequential(
#             nn.Linear(latent_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, input_size),
#             nn.Sigmoid()  # Sigmoid activation for output between 0 and 1
#         )
#         self.latent_size = latent_size
#
#     def reparameterize(self, mean, log_variance):
#         std = torch.exp(0.5 * log_variance)
#         epsilon = torch.randn_like(std)
#         return mean + epsilon * std
#
#     def forward(self, x):
#         latent_params = self.encoder(x)
#         mean, log_variance = torch.split(latent_params, self.latent_size, dim=1)
#         z = self.reparameterize(mean, log_variance)
#         reconstruction = self.decoder(z)
#         return reconstruction, mean, log_variance
#
# # Step 3: 定义损失函数（重构损失 + KL散度）
# def loss_function(reconstruction, target, mean, log_variance):
#     # 重构损失（MSE损失）
#     reconstruction_loss = nn.MSELoss()(reconstruction, target)
#
#     # KL散度
#     kl_divergence = -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp())
#
#     # ELBO = 重构损失 + KL散度
#     elbo = reconstruction_loss + kl_divergence
#     return elbo
# # Step 4: 使用训练好的VAE模型对train_X_C进行重构，得到train_output_C
# def get_train_output_C(vae_model, train_X):
#     with torch.no_grad():
#         train_output_C, _, _ = vae_model(train_X)
#     return train_output_C
# # Step 5: 训练VAE模型并得到train_output_C
# def train_vae(train_X, num_epochs=200, batch_size=32):
#     input_size = train_X.shape[1]
#     hidden_size = 64
#     latent_size = 10
#     # 转换数据为PyTorch DataLoader
#     train_data = torch.utils.data.TensorDataset(train_X)
#     train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
#     # 创建VAE模型实例
#     vae_model = VAE(input_size, hidden_size, latent_size)
#     # 优化器
#     optimizer = optim.Adam(vae_model.parameters(), lr=0.001)
#     Loss = []
#     for epoch in range(num_epochs):
#         total_loss = 0
#         for batch in train_loader:
#             x = batch[0]
#             optimizer.zero_grad()
#             reconstruction, mean, log_variance = vae_model(x)
#             loss = loss_function(reconstruction, x, mean, log_variance)
#             loss.backward()
#             optimizer.step()
#             total_loss += loss.item()
#         print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
#         Loss.append(loss.item())
#     plt.title('Train Loss of VAE ', fontdict={"family": "Times New Roman", "size": 18})
#     plt.plot(Loss, label='Train Loss')
#     plt.xlabel('Epoch:round', fontdict={"family": "Times New Roman", "size": 13})
#     plt.ylabel('Loss', fontdict={"family": "Times New Roman", "size": 14})
#     plt.xticks(fontsize=13)
#     plt.yticks(fontsize=13)
#     plt.legend(fontsize=16)
#     plt.show()
#     return vae_model
# # 训练VAE模型
# trained_vae_model = train_vae(train_X_C)
# # 获取train_output_C
# train_output_C = get_train_output_C(trained_vae_model, train_X_C)
# # 在验证集上进行验证
# with torch.no_grad():
#     test_output_C, mean, log_variance = trained_vae_model(test_X_C)
# # 计算训练集上的损失
# train_loss = loss_function(train_output_C, train_X_C, mean, log_variance)
# print("Train Loss:", train_loss.item())
# # 计算验证集上的损失
# test_loss = loss_function(test_output_C, test_X_C, mean, log_variance)
# print("Test Loss:", test_loss.item())


class MLP(torch.nn.Module):

    def __init__(self, num_i, num_h, num_o):
        super(MLP, self).__init__()

        self.linear1 = torch.nn.Linear(num_i, num_h)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(num_h, num_h)  # 2个隐层
        self.relu2 = torch.nn.ReLU()
        self.linear3 = torch.nn.Linear(num_h, num_o)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        return x
loss_function_MLP_C = nn.MSELoss()

input_size = train_X_C.shape[1]
output_size =train_Y_C.shape[1]
hidden_size = 64
mlp_model_C = MLP(input_size, hidden_size, output_size)
optimizer = optim.Adam(mlp_model_C.parameters(), lr=0.001)
num_epochs = 300
for epoch in range(num_epochs):
   # 前向传播
   inputs = train_X_C
   outputs = mlp_model_C(inputs)

   # 计算损失函数
   loss = loss_function_MLP_C(outputs, train_Y_C)

   # 反向传播及优化
   optimizer.zero_grad()
   loss.backward()
   optimizer.step()

   print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
torch.save(mlp_model_C.state_dict(), 'trained_mlp_model.pth')
MLP_model = MLP(input_size, hidden_size, output_size)
MLP_model.load_state_dict(torch.load("trained_mlp_model.pth"))
MLP_model.eval()
with torch.no_grad():
    test_output_C = MLP_model(test_X_C)
####完成了C部分的转移,输出为test_ouput_C
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class CustomQuantileLoss(nn.Module):
    def __init__(self, q1=0.25, q2=0.6, threshold=0.001):
        super().__init__()
        self.q1 = q1
        self.q2 = q2
        self.threshold = threshold

    def forward(self, preds, target):
        errors = target - preds
        abs_errors = torch.abs(errors)

        mask_small = abs_errors <= self.threshold
        mask_large = abs_errors > self.threshold

        zero_tensor = torch.zeros_like(errors)
        quantile_small = self.q1 * torch.where(errors[mask_small] > 0, errors[mask_small], zero_tensor[mask_small]) + (1 - self.q1) * torch.where(errors[mask_small] < 0, -errors[mask_small], zero_tensor[mask_small])
        quantile_large = self.q2 * torch.where(errors[mask_large] > 0, errors[mask_large], zero_tensor[mask_large]) + (1 - self.q2) * torch.where(errors[mask_large] < 0, -errors[mask_large], zero_tensor[mask_large])

        loss = torch.cat([quantile_small, quantile_large]).mean()

        return loss
quantile_loss = CustomQuantileLoss()
class ProductUnit(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ProductUnit, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        original_shape = x.shape
        x = x.view(-1, original_shape[-1])
        out = self.linear(x)
        out = out.view(*original_shape[:-1], -1)
        #out = torch.prod(out, dim=-1, keepdim=True)  # 修改为dim=1，直接相乘会变为0
        out = torch.logsumexp(out, dim=-1, keepdim=True) # 取对数、相加
        out = torch.exp(out)
        x = x.view(-1, original_shape[-1])
        out = self.linear(x)
        out = out.view(*original_shape[:-1], -1)
        out = self.activation(out)

        return out

class PUNs_Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PUNs_Net, self).__init__()
        self.product_unit1 = ProductUnit(input_dim, hidden_dim)
        self.product_unit2 = ProductUnit(hidden_dim, output_dim)
    def forward(self, x):
        out = self.product_unit1(x)
        out = self.product_unit2(out)
        return out
class CCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device):
        super(CCN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.encoder = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.decoder = nn.LSTM(output_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.device = device
    def forward(self, x, y):
        model = PUNs_Net(1,32,1).to(device)#这里的input_dim和out_put_dim都是每个样本点的维度数
        x = model(x).to(device)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        encoder_out, _ = self.encoder(x, (h0, c0))
        h0 = torch.zeros(self.num_layers, y.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, y.size(0), self.hidden_size).to(self.device)
        decoder_out, _ = self.decoder(y, (h0, c0))
        out = self.fc(decoder_out)
        return out
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, KFold
from scipy.stats import ks_2samp

def normalize_data_by_subarray(data):
    min_values = np.min(data, axis=1, keepdims=True)+100
    max_values = np.max(data, axis=1, keepdims=True)-100
    normalized_data = (data - min_values) / (max_values - min_values)
    return normalized_data, min_values, max_values

# 从归一化的结果还原数据函数
def denormalize_data_by_subarray(normalized_data, min_values, max_values):
    original_data = normalized_data * (max_values - min_values) + min_values
    return original_data
def ks_test(array1, array2):
    ks_stats = []
    for i in range(array1.shape[1]):
        ks_stat, _ = ks_2samp(array1[:, i], array2[:, i])
        ks_stats.append(ks_stat)
    return np.mean(ks_stats)
class MSLELoss(nn.Module):
    def __init__(self):
        super(MSLELoss, self).__init__()

    def forward(self, pred, true):
        return torch.mean((torch.log1p(pred) - torch.log1p(true)) ** 2)
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))
def adversarial_validation(array1, array2):
    labels = np.concatenate([np.ones(len(array1)), np.zeros(len(array2))])
    data = np.concatenate((array1, array2), axis=0)

    model = GradientBoostingClassifier(random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, data, labels, cv=cv)

    return np.mean(scores)
# 实例化Seq2Seq模型
criterion_mse = nn.MSELoss()
criterion_smooth_l1 = nn.SmoothL1Loss()
criterion_mae = nn.L1Loss()
criterion_msle = MSLELoss()
quantile_loss = CustomQuantileLoss()

# 定义输入数据
train_X_I = np.loadtxt("./Sys_data/Deilate_train_input_N500_X_17_I.txt", dtype=np.float32)
train_X = np.loadtxt("./Sys_data/Deilate_train_input_N500_17.txt", dtype=np.float32)
# MAX_train_X_I=np.max(train_X_I)#归一化因子
# train_X_I=train_X_I/MAX_train_X_I
train_X_I_tensor = torch.FloatTensor(train_X_I).to(device)
train_X_I_tensor=train_X_I_tensor.unsqueeze(2)
train_Y_I = np.loadtxt("./Sys_data/Deilate_train_input_N500_Y_17_I.txt", dtype=np.float32)
# MAX_train_Y_I=np.max(train_Y_I)
# train_Y_I=train_Y_I/MAX_train_Y_I
train_Y_I_tensor = torch.FloatTensor(train_Y_I).to(device)
train_Y_I_tensor=train_Y_I_tensor.unsqueeze(2)
test_X_I = np.loadtxt("./Sys_data/Deilate_test_input_N500_X_17_I.txt", dtype=np.float32)
test_X=np.loadtxt("./Sys_data/Deilate_test_input_N500_17.txt", dtype=np.float32)
# MAX_test_X_I =np.max(test_X_I )
# test_X_I =test_X_I /MAX_test_X_I
test_X_I_tensor = torch.FloatTensor(test_X_I).to(device)
test_X_I_tensor =test_X_I_tensor.unsqueeze(2)
# test_Y = np.loadtxt("./Sys_data/Deilate_test_traget_N500_17.txt.txt", dtype=np.float32)#测试集应该是待预测的I
# MAX_test_target=np.max(X_test_target)
# X_test_target=X_test_target/MAX_test_target
# X_test_target_tensor = torch.from_numpy(X_test_target).to(device)
# X_test_target_tensor = X_test_target_tensor.unsqueeze(2)

# Comparing (500, 40) arrays

ks_score_40 = ks_test(train_X_I, test_X_I)
print("KS score for (500, 40) arrays:", ks_score_40)
def calculate_mae(actual, predicted):
    return np.mean(np.abs((actual - predicted) ))

def calculate_rmse(actual, predicted):
    return np.sqrt(np.mean((actual - predicted) ** 2))
# 定义验证输入数据
# 定义验证目标数据
Num2=int(1000)
Num1=int(1000/2)
train_X_I_tensor_pre=train_X_I_tensor[Num1:Num2]
train_X_I_tensor_his=train_X_I_tensor[0:Num1]
train_Y_I_tensor_his=train_Y_I_tensor[0:Num1]
train_Y_I_tensor_pre=train_Y_I_tensor[Num1:Num2]
test_X_I_tensor_pre=test_X_I_tensor[Num1:Num2]
test_X_I_tensor_his=test_X_I_tensor[0:Num1]
# X_test_target_tensor_his=X_test_target_tensor[0:Num1]
# X_test_target_tensor_pre=X_test_target_tensor[Num1:Num2]
num_epochs = 200
loss_functions = {
    "MSE": criterion_mse,
    # "Smooth_L1": criterion_smooth_l1,
    # "MAE": criterion_mae,
    # "MSLE": criterion_msle,
    # "Quantile": quantile_loss
}
train_loss_history = {name: [] for name in loss_functions.keys()}
# val_loss_history = {name: [] for name in loss_functions.keys()}
for loss_name, criterion in loss_functions.items():
    print(f"Training with {loss_name} loss function")
    ccn = CCN(input_size=1, hidden_size=1028, num_layers=2, output_size=1, device=device)
    optimizer = torch.optim.Adam(ccn.parameters())
    ccn.to(device)
    for epoch in range(num_epochs):
        # 训练阶段
        ccn.train()
        optimizer.zero_grad()
        outputs = ccn(train_X_I_tensor_pre, train_Y_I_tensor_his)#相当于是把test的现代数据，和Y的历史数据带入其中
        loss = criterion(outputs, train_Y_I_tensor_pre)
        loss.backward()
        optimizer.step()
        train_loss_history[loss_name].append(loss.item())
        # 验证阶段
        # ccn.eval()
        # with torch.no_grad():
        #     val_outputs = ccn(test_X_I_tensor_pre, X_test_target_tensor_his)
        #     val_loss = criterion(val_outputs, X_test_target_tensor_pre)
        #     val_loss_history[loss_name].append(val_loss.item())
        print(f"Epoch [{epoch + 1}/{num_epochs}], {loss_name} Train Loss: {loss.item():.4f}")
    plt.title('Train Loss of CNN ', fontdict={"family": "Times New Roman", "size": 18})
    plt.plot(train_loss_history[loss_name], label='Train Loss')
    plt.xlabel('Epoch:round', fontdict={"family": "Times New Roman", "size": 13})
    plt.ylabel('Loss', fontdict={"family": "Times New Roman", "size": 14})
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=16)
    plt.show()
torch.save(ccn.state_dict(), 'CCN_model.th')
model = CCN(input_size=1, hidden_size=1028, num_layers=2, output_size=1, device=device)
model.load_state_dict(torch.load("CCN_model.th"))
model.to(device)
model.eval()
with torch.no_grad():
    test_outputs_I = model( test_X_I_tensor_pre,train_Y_I_tensor_pre)
####完成从Train_X_I到Test_X_I


class TransformerModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_heads, num_layers):
        super(TransformerModel, self).__init__()
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_size, nhead=num_heads),
            num_layers=num_layers
        )
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x_c, x_i):
        x = torch.cat((x_c, x_i), dim=1)  # 拼接 x_c 和 x_i
        x = x.unsqueeze(0)  # 添加 batch 维度，变成 1*batch_size*input_size
        x = self.transformer(x)  # Transformer 编码
        x = x.squeeze(0)  # 移除 batch 维度，变成 batch_size*input_size
        output = self.fc(x)  # 全连接层，输出output_Y
        return output


class CustomQuantileLoss(nn.Module):
    def __init__(self, q1=0.06, q2=0.6, threshold=0.001):
        super().__init__()
        self.q1 = q1
        self.q2 = q2
        self.threshold = threshold

    def forward(self, preds, target):
        errors = target - preds
        abs_errors = torch.abs(errors)

        mask_small = abs_errors <= self.threshold
        mask_large = abs_errors > self.threshold

        zero_tensor = torch.zeros_like(errors)
        quantile_small = self.q1 * torch.where(errors[mask_small] > 0, errors[mask_small], zero_tensor[mask_small]) + (1 - self.q1) * torch.where(errors[mask_small] < 0, -errors[mask_small], zero_tensor[mask_small])
        quantile_large = self.q2 * torch.where(errors[mask_large] > 0, errors[mask_large], zero_tensor[mask_large]) + (1 - self.q2) * torch.where(errors[mask_large] < 0, -errors[mask_large], zero_tensor[mask_large])

        loss = torch.cat([quantile_small, quantile_large]).mean()

        return loss

# Step 3: 定义损失函数（MSE损失）
loss_function = CustomQuantileLoss()

# Step 4: 训练Transformer模型
def train_transformer(train_Y_C, train_Y_I, train_Y, num_epochs=400, batch_size=32):
    input_size = train_Y_C.shape[1] + train_Y_I.shape[1]
    output_size = test_Y.shape[1]
    hidden_size = 64
    num_heads = 4
    num_layers = 2
    train_Y=torch.tensor(train_Y,dtype=torch.float)
    # 转换数据为PyTorch DataLoader
    train_data = torch.utils.data.TensorDataset(train_Y_C, train_Y_I, train_Y)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # 创建Transformer模型实例
    transformer_model = TransformerModel(input_size, output_size, hidden_size, num_heads, num_layers).to(device)

    # 优化器
    optimizer = optim.Adam(transformer_model.parameters(), lr=0.001)
    transformer_loss=[]
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_c, batch_i,batch_y in train_loader:
            optimizer.zero_grad()
            output = transformer_model(batch_c, batch_i)
            loss = loss_function(output, batch_y)  # 使用batch_c作为训练集上的目标
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            transformer_loss.append(loss.item())
        average_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}")
    plt.title('Train Loss of Transformer ', fontdict={"family": "Times New Roman", "size": 18})
    plt.plot(transformer_loss, label='Train Loss')
    plt.xlabel('Epoch:round', fontdict={"family": "Times New Roman", "size": 13})
    plt.ylabel('Loss', fontdict={"family": "Times New Roman", "size": 14})
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    plt.legend(fontsize=16)
    plt.show()
    return transformer_model
test_outputs_I=test_outputs_I.squeeze()
# 训练Transformer模型:从train_Y_C,train_Y_I到train_Y
trained_transformer_model = train_transformer(test_output_C[0:250][:].to(device), test_outputs_I[0:250][:].to(device), test_Y[0:250][:].to(device)).to(device)

# 在训练集上预测
with torch.no_grad():
    output_Y = trained_transformer_model(test_output_C[250:500][:].to(device), test_outputs_I[250:500][:].to(device)).to(device)

#已经变torch了 train_Y_C = torch.FloatTensor(train_Y_C)
# 计算output_Y和test_Y之间的损失
loss = loss_function(output_Y, test_Y[250:500][:].to(device))
print("Loss:", loss.item())
torch.save(trained_transformer_model.state_dict(), 'transformer_model.th')

test_outputs_I=test_outputs_I.to(device)
test_output_C=test_output_C[Num1:Num2].to(device)
inputs_test = torch.cat((test_output_C,test_outputs_I), dim=1)
input_size = train_Y_C.shape[1] + train_Y_I.shape[1]
output_size = test_Y.shape[1]
hidden_size = 64
num_heads = 4
num_layers = 2
transformer_model = TransformerModel(input_size, output_size, hidden_size, num_heads, num_layers)
transformer_model.load_state_dict(torch.load("transformer_model.th"))
transformer_model.to(device)
with torch.no_grad():
    test_Y_outputs = transformer_model(test_output_C,test_outputs_I)
transformer_model.eval()
# mlp_model = MLP(input_size, hidden_size, output_size)
#
# mlp_model.load_state_dict(torch.load("trained_mlp_model.pth"))
# mlp_model.to(device)

test_Y_outputs.to(device)
test_Y=test_Y[Num1:Num2]
# test_Y,test_Y_outputs
test_Y_np = test_Y.cpu().numpy()
test_Y_outputs_np = test_Y_outputs.cpu().numpy()

# 计算 MSE
mse = np.mean((test_Y_np - test_Y_outputs_np)**2)

# 计算 F1 分数
def binary_f1_score(y_true, y_pred):
    y_pred_binary = (y_pred > 0.5).astype(int)
    tp = np.sum(y_true * y_pred_binary)  # True Positives
    fp = np.sum((1 - y_true) * y_pred_binary)  # False Positives
    fn = np.sum(y_true * (1 - y_pred_binary))  # False Negatives
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return f1

f1 = binary_f1_score(test_Y_np, test_Y_outputs_np)

# 计算 MAE
mae = np.mean(np.abs(test_Y_np - test_Y_outputs_np))

# 计算 RMSE
rmse = np.sqrt(np.mean((test_Y_np - test_Y_outputs_np)**2))

print("MSE:", mse)
print("F1:", f1)
print("MAE:", mae)
print("RMSE:", rmse)
test_Y_np_heat=np.concatenate((test_X[900:1000],test_Y_np[400:500]), axis=1)

plt.subplot(1, 2, 1)
plt.imshow(test_Y_np_heat, cmap='viridis', aspect='auto')
plt.title('test_Y Heatmap')
plt.xlabel('Feature')
plt.ylabel('Sample')

# 绘制 test_Y_outputs_np 的热度图
test_Y_outputs_np_heat=np.concatenate((test_X[900:1000],test_Y_outputs_np[400:500]), axis=1)
plt.subplot(1, 2, 2)
plt.imshow(test_Y_outputs_np_heat, cmap='viridis', aspect='auto')
plt.title('test_outputs Heatmap')
plt.xlabel('Feature')
plt.ylabel('Sample')

plt.tight_layout()  # 调整子图布局，防止重叠
plt.show()

x = np.arange(0, test_Y_np_heat.shape[1], 1)  # 列数
y = np.arange(0,test_Y_outputs_np_heat.shape[0], 1)  # 行数
x, y = np.meshgrid(x, y)

# 绘制 test_Y_np 的 3D 图
fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(x, y, test_Y_np_heat, cmap='viridis')
ax1.set_title('test_Y 3D Plot')
ax1.set_xlabel('Feature')
ax1.set_ylabel('Sample')
ax1.set_zlabel('Value')

# 绘制 test_Y_outputs_np 的 3D 图
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(x, y, test_Y_outputs_np_heat, cmap='viridis')
ax2.set_title('test_outputs 3D Plot')
ax2.set_xlabel('Feature')
ax2.set_ylabel('Sample')
ax2.set_zlabel('Value')

plt.show()

XX = np.arange(0, test_Y_outputs_np_heat.shape[1], 1)
plt.plot(XX,test_Y_np_heat[99][:],label="True")
plt.plot(XX,test_Y_outputs_np_heat[99][:],label="Prediciton")
plt.xlabel('time')
plt.ylabel('value')
plt.title('Specific sample')
plt.legend()
plt.show()

XX = np.arange(0, test_Y_outputs_np_heat.shape[1], 1)
plt.plot(XX,test_Y_np_heat[80][:],label="True")
plt.plot(XX,test_Y_outputs_np_heat[80][:],label="Prediciton")
plt.xlabel('time')
plt.ylabel('value')
plt.title('Specific sample')
plt.legend()
plt.show()
######################反事实实验
train_Y_I_unfact = np.loadtxt("./Sys_data/Deilate_train_input_N500_Y_18_I.txt", dtype=np.float32)
test_X_C_unfact = np.loadtxt("./Sys_data/Deilate_test_input_N500_X_18_C.txt", dtype=np.float32)
test_X_I_unfact = np.loadtxt("./Sys_data/Deilate_test_input_N500_Y_18_I.txt", dtype=np.float32)
test_X_unfact = np.loadtxt("./Sys_data/Deilate_test_input_N500_18.txt", dtype=np.float32)
test_Y_unfact = np.loadtxt("./Sys_data/Deilate_test_traget_N500_18.txt", dtype=np.float32)
train_Y_I_unfact = torch.FloatTensor(train_Y_I_unfact)
test_X_C_unfact = torch.FloatTensor(test_X_C_unfact)
test_X_I_unfact = torch.FloatTensor(test_X_I_unfact)
test_X_unfact = torch.FloatTensor(test_X_unfact)
test_Y_unfact = torch.FloatTensor(test_Y_unfact)
test_X_I_tensor_pre_unfact=test_X_I_unfact[Num1:Num2].unsqueeze(2)
train_Y_I_tensor_pre_unfact=train_Y_I_unfact[Num1:Num2].unsqueeze(2)

input_size = train_X_C.shape[1]
output_size =train_Y_C.shape[1]
hidden_size = 64
MLP_model_unfact = MLP(input_size, hidden_size, output_size)
MLP_model_unfact.load_state_dict(torch.load("trained_mlp_model.pth"))
MLP_model_unfact.eval()
with torch.no_grad():
    test_output_C_unfact = MLP_model_unfact(test_X_C_unfact)

model_unfact = CCN(input_size=1, hidden_size=1028, num_layers=2, output_size=1, device=device)
model_unfact.load_state_dict(torch.load("CCN_model.th"))
model_unfact.to(device)
model_unfact.eval()
with torch.no_grad():
    test_outputs_I_unfact = model_unfact( test_X_I_tensor_pre_unfact.to(device),train_Y_I_tensor_pre_unfact.to(device)).to(device)
test_outputs_I_unfact =test_outputs_I_unfact.squeeze()

input_size = train_Y_C.shape[1] + train_Y_I.shape[1]
output_size = test_Y.shape[1]
hidden_size = 64
num_heads = 4
num_layers = 2
transformer_model_unfact = TransformerModel(input_size, output_size, hidden_size, num_heads, num_layers)
transformer_model_unfact .load_state_dict(torch.load("transformer_model.th"))
transformer_model_unfact .to(device)
transformer_model_unfact .eval()
with torch.no_grad():
    test_Y_outputs_unfact  = transformer_model_unfact (test_output_C_unfact[Num1:Num2][:].to(device),test_outputs_I_unfact.to(device)).to(device)
test_Y_outputs_unfact.to(device)
test_Y_unfact=test_Y_unfact[Num1:Num2]

test_Y_np_unfact = test_Y_unfact.cpu().numpy()
test_Y_outputs_np_unfact = test_Y_outputs_unfact.cpu().numpy()



MSE = np.mean((test_Y_np_unfact - test_Y_outputs_np_unfact)**2)
# 计算 MAE
MAE = np.mean(np.abs(test_Y_np_unfact - test_Y_outputs_np_unfact))
# 计算 RMSE
RMSE = np.sqrt(np.mean((test_Y_np_unfact - test_Y_outputs_np_unfact)**2))
print("MSE_Unfact:", MSE)
print("MAE_Unfact:", MAE)
print("RMSE_Unfact:", RMSE)
test_Y_np_heat_unfact=np.concatenate((test_X[500:1000],test_Y_np), axis=1)
plt.subplot(1, 2, 1)
plt.imshow(test_Y_np_heat, cmap='viridis', aspect='auto')
plt.title('test_Y_unfact Heatmap')
plt.xlabel('Time')
plt.ylabel('Sample')

# 绘制 test_Y_outputs_np 的热度图
test_Y_outputs_np_heat=np.concatenate((test_X[500:1000],test_Y_outputs_np), axis=1)
plt.subplot(1, 2, 2)
plt.imshow(test_Y_outputs_np_heat, cmap='viridis', aspect='auto')
plt.title('test_outputs_unfact Heatmap')
plt.xlabel('Time')
plt.ylabel('Sample')

plt.tight_layout()  # 调整子图布局，防止重叠
plt.show()