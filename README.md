# MEC-DQN-Offloading-QoS
**Tối ưu hóa dỡ tải tính toán trong mạng Mobile Edge Computing bằng thuật toán Học tăng cường sâu (Double DQN)**

Đề tài này nghiên cứu và mô phỏng bài toán dỡ tải tính toán (Computation Offloading) trong môi trường mạng MEC. Mục tiêu là tối ưu hóa hàm chi phí bao gồm độ trễ (Latency) và năng lượng tiêu thụ (Energy Consumption), đồng thời đảm bảo chất lượng dịch vụ (QoS) thông qua cơ chế trọng số ưu tiên.

## Đặc điểm hệ thống
* **Thuật toán Double DQN (DDQN):** Khắc phục hiện tượng đánh giá quá mức (Overestimation) của DQN truyền thống, cải thiện độ ổn định trong quá trình hội tụ.
* **Cơ chế QoS:** Hỗ trợ thiết lập trọng số ưu tiên (Priority Weights) cho từng thiết bị trong hàm mục tiêu tối ưu.
* **Môi trường tính toán:** Tích hợp tùy chọn xử lý trên GPU/CUDA để tăng hiệu suất huấn luyện.
* **Khả năng mở rộng:** Cho phép tùy chỉnh cấu hình mô phỏng từ 1 đến 11 thiết bị (không gian hành động tối đa 2048 tổ hợp).

## Cấu trúc mã nguồn
Mã nguồn được chia thành 4 module chính:
* `env.py`: Môi trường mô phỏng các thông số hệ thống (Băng thông, xung nhịp CPU, suy hao kênh truyền, mức tiêu thụ năng lượng, cấu hình QoS).
* `agent.py`: Định nghĩa kiến trúc mạng Nơ-ron và luồng thực thi thuật toán Double DQN.
* `main.py`: Tập lệnh thực thi chính, quản lý tham số kịch bản và điều phối quá trình huấn luyện.
* `utils.py`: Các hàm hỗ trợ trực quan hóa dữ liệu và xuất biểu đồ kết quả (Matplotlib).

## Các kịch bản thực nghiệm
Hệ thống được thiết kế để chạy 3 kịch bản chính:
1. **Kịch bản cơ sở (3 thiết bị):** Đánh giá khả năng hội tụ của mô hình trong điều kiện tài nguyên đáp ứng đủ.
2. **Kịch bản giới hạn tài nguyên (7 thiết bị):** Kiểm chứng hiệu quả của cơ chế phân bổ dựa trên trọng số QoS khi hệ thống bị quá tải.
3. **Kịch bản mở rộng (11 thiết bị):** Đánh giá tính ổn định và khả năng mở rộng (Scalability) của thuật toán với không gian trạng thái lớn.

## Hướng dẫn sử dụng
1. **Yêu cầu môi trường:** Python 3.8+, PyTorch, NumPy, Matplotlib.
2. **Cài đặt thư viện:** `pip install torch numpy matplotlib`
3. **Thực thi mô phỏng:** `python main.py`

## Kết quả xuất ra
Sau khi hoàn tất quá trình huấn luyện và đánh giá, chương trình sẽ tự động lưu các kết quả tại thư mục gốc, bao gồm:
* Biểu đồ đường cong hội tụ (Learning Curve).
* Biểu đồ cột so sánh hiệu năng (Comparison Bar Chart) giữa DDQN, All-Local và All-Edge.
