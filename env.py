import numpy as np

class MEC_Environment:
    # Khởi tạo môi trường với số lượng thiết bị và mảng trọng số ưu tiên
    def __init__(self, num_users, weights=None):
        self.num_users = num_users

        # =======================================================
        # KHỞI TẠO TRỌNG SỐ ƯU TIÊN (QoS)
        # =======================================================
        if weights is not None:
            # Kiểm tra tính hợp lệ của mảng trọng số đầu vào
            assert len(weights) == self.num_users, f"Lỗi: Có {self.num_users} thiết bị nhưng nhận được {len(weights)} trọng số!"
            self.weights = weights
        else:
            # Mặc định: Phân bổ đều trọng số nếu không được chỉ định
            self.weights = [1.0 / self.num_users] * self.num_users

        # =======================================================
        # THIẾT LẬP THAM SỐ VẬT LÝ HỆ THỐNG
        # =======================================================
        self.B = 2.0e6         # Băng thông hệ thống (2 MHz)
        self.f_edge = 6.0e9    # Khả năng xử lý CPU của Trạm Edge (6 GHz)
        self.P_tx = 0.1        # Công suất truyền dẫn
        self.N0 = 1e-13        # Mật độ phổ công suất nhiễu trắng
        self.C = 1000          # Số chu kỳ CPU cần thiết để xử lý 1 bit
        self.f_loc = 1.5e9     # Khả năng xử lý CPU nội bộ của thiết bị (1.5 GHz)
        self.kappa = 2e-29     # Hệ số tiêu thụ năng lượng khi xử lý Local

        self.alpha = 0.5       # Hệ số đánh đổi của độ trễ
        self.beta = 0.5        # Hệ số đánh đổi của năng lượng
        self.max_battery = 100.0
        self.max_steps = 50
        self.current_step = 0

        # Mở rộng không gian trạng thái và hành động tự động theo số User
        self.state_dim = 3 * self.num_users
        self.action_dim = 2 ** self.num_users

        self.batteries = np.full(self.num_users, self.max_battery)

    def reset(self):
        # Khôi phục trạng thái môi trường về ban đầu
        self.batteries = np.full(self.num_users, self.max_battery)
        self.current_step = 0

        # Khởi tạo ngẫu nhiên kích thước tác vụ (2 - 6 Mbits) và trạng thái kênh truyền
        self.tasks = np.random.uniform(2e6, 6e6, self.num_users)
        self.channels = np.random.uniform(1e-8, 1e-7, self.num_users)
        return self._get_normalized_state()

    def _get_normalized_state(self):
        # Chuẩn hóa không gian trạng thái đầu vào cho mạng Nơ-ron
        state = []
        for i in range(self.num_users):
            state.extend([
                self.tasks[i] / 6e6,
                self.channels[i] / 1e-7,
                self.batteries[i] / 100.0
            ])
        return np.array(state, dtype=np.float32)

    def step(self, action_index):
        # Thực thi hành động và trả về trạng thái mới, phần thưởng
        self.current_step += 1

        # Chuyển đổi chỉ số Action thành mảng nhị phân tương ứng với từng thiết bị
        binary_action = format(action_index, f'0{self.num_users}b')
        actions = [int(a) for a in binary_action]

        # Xác định số lượng thiết bị dỡ tải để phân chia tài nguyên Edge
        num_offloaders = sum(actions)
        shared_B = self.B / num_offloaders if num_offloaders > 0 else self.B
        shared_f_edge = self.f_edge / num_offloaders if num_offloaders > 0 else self.f_edge

        total_latency = 0.0
        total_energy = 0.0
        total_weighted_cost = 0.0  # Tổng chi phí hệ thống sau khi áp dụng QoS
        reward = 0.0

        for i in range(self.num_users):
            D = self.tasks[i]
            h = self.channels[i]
            action = actions[i]

            # Xử lý nội bộ (Local)
            if action == 0:
                latency = (D * self.C) / self.f_loc
                energy = self.kappa * D * self.C * (self.f_loc ** 2)
            # Dỡ tải lên máy chủ biên (Edge)
            else:
                SNR = (self.P_tx * h) / self.N0
                R = shared_B * np.log2(1 + SNR)
                t_tx = D / R
                t_edge = (D * self.C) / shared_f_edge
                latency = t_tx + t_edge
                energy = self.P_tx * t_tx

            # Ghi nhận chỉ số để giám sát
            total_latency += latency
            total_energy += energy

            # =======================================================
            # TÍNH TOÁN CHI PHÍ VÀ ÁP DỤNG TRỌNG SỐ ƯU TIÊN (QoS)
            # =======================================================
            individual_cost = self.alpha * latency + self.beta * (energy * 10)
            total_weighted_cost += self.weights[i] * individual_cost

            # Cập nhật trạng thái pin của thiết bị
            self.batteries[i] -= (energy * 0.1)

            # Ràng buộc năng lượng: Xử phạt nếu thiết bị cạn pin
            if self.batteries[i] <= 0:
                self.batteries[i] = 0
                reward -= 50

        # Hàm phần thưởng (Reward) ngược chiều với tổng chi phí mục tiêu
        reward -= total_weighted_cost

        done = self.current_step >= self.max_steps or np.any(self.batteries <= 0)

        # Cập nhật thông số môi trường cho bước tiếp theo
        self.tasks = np.random.uniform(2e6, 6e6, self.num_users)
        self.channels = np.random.uniform(1e-8, 1e-7, self.num_users)

        info = {'latency': total_latency, 'energy': total_energy}
        return self._get_normalized_state(), reward, done, info