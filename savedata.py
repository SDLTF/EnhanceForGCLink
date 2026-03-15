import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# 假设你已经定义了模型和数据加载函数
from GCLink_main_SimSiam import build_model, load_data, DEVICE
print("111")
# 加载数据
data, gene_names = load_data()
features = data.x.to(DEVICE)

# 构建模型
model = build_model(data).to(DEVICE)

# 设置优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.BCEWithLogitsLoss()

# 训练过程
def train(model, optimizer, loss_fn, features, epochs=50):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # 计算模型的预测结果
        # 假设我们有 `train_x`, `train_y` 作为训练数据
        pred = model(features)  # 你可能需要根据具体的模型调整这里的调用
        
        # 计算损失
        loss = loss_fn(pred, features)  # 这里的目标可以是你的标签数据（基因表达，或者其他）
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

    # 保存模型
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item(),
    }, "model_checkpoint.ckpt")  # 保存检查点

# 训练模型并保存

print("1. savedata.py 开始执行")

print("2. 准备导入 GCLink_main_SimSiam")
from GCLink_main_SimSiam import build_model, load_data, DEVICE
print("3. 导入完成")

train(model, optimizer, loss_fn, features)
print("4. 程序结束")

