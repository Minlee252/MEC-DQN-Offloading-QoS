import matplotlib.pyplot as plt
import numpy as np

# Thiết lập font chữ mặc định cho thư viện matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'

def plot_learning_curve(rewards, num_users, filename="learning_curve.png"):
    # ==========================================================
    # HÀM TRỰC QUAN HÓA QUÁ TRÌNH HỘI TỤ
    # Biểu diễn đồ thị thay đổi phần thưởng (reward) theo số tập huấn luyện.
    # ==========================================================
    plt.figure(figsize=(10, 6))

    # 1. Trực quan hóa dữ liệu điểm thưởng thô từng tập
    plt.plot(rewards, color='#4169E1', alpha=0.3, label='Diem tho tung tap (Raw)')

    # 2. Tính toán linh hoạt kích thước cửa sổ trượt (Window size) cho đường trung bình
    total_episodes = len(rewards)
    if total_episodes <= 500:
        window = 20
    elif total_episodes <= 1500:
        window = 50
    else:
        window = 250

    # 3. Tính toán và vẽ đường trung bình trượt (Moving Average - MA)
    if total_episodes >= window:
        moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        x_axis = range(window - 1, total_episodes)
        plt.plot(x_axis, moving_avg, color='#0000CD', linewidth=2.5, label=f'Duong xu huong (MA {window})')

    # 4. Định dạng và chú thích đồ thị
    plt.title(f'Bieu do Hoi tu cua AI (Double DQN - {num_users} Thiet bi)', fontsize=14, fontweight='bold')
    plt.xlabel('So tap huan luyen (Episodes)', fontsize=12)
    plt.ylabel('Tong diem thuong he thong (Reward)', fontsize=12)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=11)

    # 5. Xuất tệp tin hình ảnh và hiển thị
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"\n[+] Bieu do hoi tu da duoc luu tai: '{filename}'")


def plot_comparison_bar(eval_results, num_users, filename="so_sanh_cot.png"):
    # ==========================================================
    # HÀM TRỰC QUAN HÓA SO SÁNH HIỆU NĂNG
    # Biểu diễn bằng đồ thị cột để đánh giá 3 phương pháp: AI, All-Local và All-Edge.
    # ==========================================================
    strategies = ['AI (Double DQN)', 'All-Local', 'All-Edge']
    scores = [eval_results['AI'], eval_results['Local'], eval_results['Edge']]
    colors = ['#2ecc71', '#e74c3c', '#3498db'] # Phân bổ màu sắc: AI (Xanh lá), Local (Đỏ), Edge (Xanh dương)

    plt.figure(figsize=(9, 6))
    bars = plt.bar(strategies, scores, color=colors, width=0.6)

    # Hiển thị giá trị cụ thể tại vị trí tương ứng của từng cột
    for bar in bars:
        yval = bar.get_height()
        # Căn chỉnh vị trí nhãn dữ liệu tránh trùng lặp
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}',
                 ha='center', va='bottom' if yval > 0 else 'top',
                 fontweight='bold', color='black')

    plt.title(f'SO SANH HIEU NANG - KICH BAN {num_users} THIET BI', fontsize=14, fontweight='bold')
    plt.ylabel('Diem thuong trung binh (Average Reward)', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Xuất kết luận tự động hóa dựa trên kết quả trung bình cao nhất
    best_score = max(scores)
    if best_score == scores[0]:
        conclusion = f"==> KET LUAN: AI CHIEN THANG O KICH BAN {num_users} MAY!"
    elif best_score == scores[2]:
        conclusion = f"==> KET LUAN: ALL-EDGE VAN TOI UU NHAT O KICH BAN {num_users} MAY."
    else:
        conclusion = "==> KET LUAN: ALL-LOCAL TOT NHAT."

    plt.figtext(0.5, 0.02, conclusion, ha='center', fontsize=12, fontweight='bold', color='darkred')

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15) # Dành không gian hiển thị cho nhãn kết luận
    plt.savefig(filename, dpi=300)
    plt.show()
    print(f"\n[+] Bieu do cot da duoc luu tai: '{filename}'")