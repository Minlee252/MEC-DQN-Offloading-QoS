import warnings
import numpy as np
from env import MEC_Environment
from agent import DoubleDQNAgent
from utils import plot_learning_curve, plot_comparison_bar

# Loại bỏ các thông báo cảnh báo từ hệ thống để tối ưu hóa hiển thị log
warnings.filterwarnings("ignore")

def train_and_evaluate(num_users, episodes, weights=None, test_episodes=30):
    # Hàm điều phối quá trình huấn luyện Agent và đánh giá so sánh hiệu năng
    print(f"\n{'='*60}")
    print(f"--- TIEN TRINH: {num_users} THIET BI | {episodes} TAP HUAN LUYEN ---")
    if weights:
        print(f"--- TRONG SO UU TIEN (QoS): {weights} ---")
    print(f"{'='*60}")

    # Khởi tạo môi trường mạng dựa trên số lượng thiết bị và trọng số QoS đã thiết lập
    env = MEC_Environment(num_users=num_users, weights=weights)

    # Khởi tạo Agent Double DQN với kích thước không gian trạng thái và hành động tương ứng
    agent = DoubleDQNAgent(state_size=env.state_dim, action_size=env.action_dim)

    rewards_history = []

    # ==========================================================
    # GIAI ĐOẠN HUẤN LUYỆN (TRAINING PHASE)
    # ==========================================================
    for e in range(episodes):
        state, total_reward, done = env.reset(), 0, False
        while not done:
            # Agent thực hiện lựa chọn hành động dựa trên chiến lược thám hiểm Epsilon-greedy
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            # Lưu trữ dữ liệu trải nghiệm vào bộ nhớ đệm và thực thi thuật toán học tập
            agent.remember(state, action, reward, next_state, done)
            agent.learn()

            state = next_state
            total_reward += reward

        # Điều chỉnh tốc độ học (Learning Rate) thông qua Scheduler sau mỗi tập huấn luyện
        if len(agent.memory) >= agent.batch_size:
            agent.scheduler.step()

        rewards_history.append(total_reward)

        # Hiển thị thông số tiến độ huấn luyện định kỳ sau mỗi 500 tập
        if e % 500 == 0:
            print(f"Tap {e}/{episodes} | Reward: {total_reward:.2f} | Epsilon: {agent.epsilon:.3f}")

    # ==========================================================
    # GIAI ĐOẠN ĐÁNH GIÁ HIỆU NĂNG (EVALUATION PHASE)
    # So sánh kết quả của AI với các phương pháp Baseline (Local/Edge)
    # ==========================================================
    print(f"\n--- ĐANG ĐÁNH GIÁ HIỆU NĂNG (Sau {test_episodes} lần chạy thử) ---")
    results = {'AI': [], 'Local': [], 'Edge': []}

    # Xác định chỉ số hành động tương ứng với phương án dỡ tải toàn phần (All-Edge)
    all_edge_action = (2 ** num_users) - 1

    for _ in range(test_episodes):
        # 1. Đánh giá chiến thuật của Agent AI (Double DQN)
        s = env.reset(); r_ai = 0; d = False
        while not d:
            # Chế độ kiểm thử: AI luôn chọn hành động mang lại giá trị Q-value cao nhất
            action = agent.act(s)
            s, r, d, _ = env.step(action); r_ai += r
        results['AI'].append(r_ai)

        # 2. Đánh giá phương án xử lý nội bộ toàn phần (All-Local)
        s = env.reset(); r_local = 0; d = False
        while not d:
            s, r, d, _ = env.step(0); r_local += r
        results['Local'].append(r_local)

        # 3. Đánh giá phương án dỡ tải biên toàn phần (All-Edge)
        s = env.reset(); r_edge = 0; d = False
        while not d:
            s, r, d, _ = env.step(all_edge_action); r_edge += r
        results['Edge'].append(r_edge)

    # Tổng hợp giá trị trung bình từ các tập đánh giá
    avg_results = {k: np.mean(v) for k, v in results.items()}
    return rewards_history, avg_results


if __name__ == '__main__':
    # ==========================================================
    # CẤU HÌNH THAM SỐ HỆ THỐNG VÀ KỊCH BẢN THỬ NGHIỆM
    # ==========================================================
    NUM_USERS = 3       # Thiết lập số lượng thiết bị (Tùy chọn: 1, 3, 5, 7, 11)
    EPISODES = 3000     # Tổng số tập huấn luyện (Tùy chỉnh theo quy mô kịch bản)

    # Thiết lập ma trận trọng số ưu tiên QoS (QoS Priority Weights)
    if NUM_USERS == 3:
        # Cấu hình đồng nhất cho 3 thiết bị
        USER_WEIGHTS = [0.34, 0.33, 0.33]
    elif NUM_USERS == 5:
        # Thiết lập thiết bị 1 là đối tượng ưu tiên cao (60%)
        USER_WEIGHTS = [0.6, 0.1, 0.1, 0.1, 0.1]
    elif NUM_USERS == 7:
        # Phân cấp ưu tiên cho 2 thiết bị trọng yếu
        USER_WEIGHTS = [0.5, 0.2, 0.06, 0.06, 0.06, 0.06, 0.06]
    elif NUM_USERS == 11:
        # Kịch bản mở rộng: Phân cấp ưu tiên cho nhóm thiết bị VIP
        USER_WEIGHTS = [0.4, 0.2, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
    else:
        # Mặc định phân bổ trọng số đều nếu không thuộc các cấu hình trên
        USER_WEIGHTS = None

    # Thực thi tiến trình huấn luyện và thu thập dữ liệu hiệu năng
    history, eval_results = train_and_evaluate(
        num_users=NUM_USERS,
        episodes=EPISODES,
        weights=USER_WEIGHTS
    )

    print("\n[+] HOÀN TẤT! ĐANG XUẤT BẢN ĐỒ THỊ TRỰC QUAN...")

    # Thiết lập tên tệp tin lưu trữ để quản lý các kịch bản khác nhau
    curve_name = f'hoi_tu_{NUM_USERS}_users_QoS.png'
    bar_name = f'so_sanh_{NUM_USERS}_users_QoS.png'

    # Khởi tạo quy trình vẽ đồ thị từ thư viện utils.py
    plot_learning_curve(history, num_users=NUM_USERS, filename=curve_name)
    plot_comparison_bar(eval_results, num_users=NUM_USERS, filename=bar_name)