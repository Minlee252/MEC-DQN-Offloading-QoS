import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()

        # ==========================================================
        # [1] THIẾT LẬP KIẾN TRÚC MẠNG NƠ-RON (HIDDEN LAYERS)
        # - Cấu hình cho 11 thiết bị: Sử dụng 1024 nơ-ron
        # - Cấu hình cho 9 thiết bị: Sử dụng 512 nơ-ron
        # - Cấu hình cho 7 thiết bị: Sử dụng 256 nơ-ron
        # - Cấu hình cho 1, 3, 5 thiết bị: Sử dụng 128 nơ-ron
        # ==========================================================
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DoubleDQNAgent:
    # ==========================================================
    # [2] THIẾT LẬP KÍCH THƯỚC KHÔNG GIAN TRẠNG THÁI VÀ HÀNH ĐỘNG
    # Các giá trị mặc định tương ứng với cấu hình hệ thống:
    # - 11 thiết bị: state_size=33, action_size=2048
    # - 9 thiết bị: state_size=27, action_size=512
    # - 7 thiết bị: state_size=21, action_size=128
    # - 5 thiết bị: state_size=15, action_size=32
    # - 3 thiết bị: state_size=9, action_size=8
    # - 1 thiết bị: state_size=3, action_size=2
    # ==========================================================
    def __init__(self, state_size=9, action_size=8):
        self.state_size = state_size
        self.action_size = action_size
        self.tau = 0.005

        # Thiết lập tốc độ học (Learning Rate) dựa trên quy mô hệ thống
        # (Lưu ý: Sử dụng lr = 0.0005 cho kịch bản từ 7 đến 11 thiết bị để đảm bảo hội tụ ổn định)
        self.lr = 0.001
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.001

        # ==========================================================
        # [3] THIẾT LẬP HỆ SỐ SUY GIẢM KHÁM PHÁ (EPSILON DECAY)
        # - 11 thiết bị: 0.9999
        # - 9 thiết bị: 0.9998
        # - 7 thiết bị: 0.99985
        # - 5 thiết bị: 0.996
        # - 3 thiết bị: 0.99
        # - 1 thiết bị: 0.95
        # ==========================================================
        self.epsilon_decay = 0.99

        # ==========================================================
        # [4] THIẾT LẬP KÍCH THƯỚC BỘ NHỚ VÀ LÔ DỮ LIỆU HUẤN LUYỆN
        # - 11 thiết bị: batch_size = 256, maxlen = 50000
        # - 9 thiết bị: batch_size = 128, maxlen = 30000
        # - 7 thiết bị: batch_size = 64, maxlen = 15000
        # - 1, 3, 5 thiết bị: batch_size = 64, maxlen = 10000
        # ==========================================================
        self.batch_size = 64
        self.memory = deque(maxlen=10000)

        # ==========================================================
        # KHỞI TẠO MÔI TRƯỜNG TÍNH TOÁN ĐỘNG (GPU/CPU)
        # ==========================================================
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[+] Trí não AI đang được khởi động trên: {self.device.type.upper()}")

        # Chuyển cấu trúc mạng Nơ-ron sang thiết bị tính toán
        self.q_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network = QNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.5)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        # Chuyển đổi trạng thái đầu vào thành Tensor và đưa vào thiết bị tính toán
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state)

        # Trích xuất chỉ số hành động tối ưu và chuyển kết quả về CPU
        return np.argmax(q_values.cpu().numpy())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)

        # Chuyển đổi lô dữ liệu kinh nghiệm sang Tensor để xử lý song song
        states = torch.FloatTensor(np.array([t[0] for t in minibatch])).to(self.device)
        actions = torch.LongTensor(np.array([t[1] for t in minibatch])).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array([t[2] for t in minibatch])).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([t[3] for t in minibatch])).to(self.device)
        dones = torch.FloatTensor(np.array([t[4] for t in minibatch])).unsqueeze(1).to(self.device)

        best_actions = self.q_network(next_states).argmax(1).unsqueeze(1)
        next_q_values = self.target_network(next_states).gather(1, best_actions)
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        current_q_values = self.q_network(states).gather(1, actions)

        loss = F.mse_loss(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay