# Ứng Dụng Tối Ưu Hóa Danh Mục Đầu Tư (LSTM-GRU)

Ứng dụng web được xây dựng trên nền tảng **Streamlit** để tối ưu hóa danh mục đầu tư chứng khoán Việt Nam sử dụng mô hình học sâu lai LSTM-GRU dựa trên tỷ số Sharpe. Ứng dụng tải dữ liệu thời gian thực từ thư viện `vnstock` và so sánh hiệu quả của mô hình học sâu với các chiến lược phân bổ truyền thống như Phân bổ đều (Equal-Weight) và Phân bổ 80-20.

---

## 📁 Cấu Trúc Thư Mục Dự Án

*   `app.py`: Tệp mã nguồn chính chạy ứng dụng Streamlit.
*   `industry_tickers.py`: Danh sách các mã cổ phiếu theo nhóm ngành tại thị trường Việt Nam.
*   `requirements.txt`: Danh sách các thư viện Python cần thiết để chạy dự án.
*   `README.md`: Hướng dẫn cài đặt và sử dụng ứng dụng.

---

## 🛠️ Hướng Dẫn Cài Đặt và Chạy Cục Bộ (Local)

### Bước 1: Chuẩn bị môi trường Python
Đảm bảo máy tính của bạn đã cài đặt Python (phiên bản khuyên dùng là 3.9 - 3.12).

### Bước 2: Tạo và kích hoạt môi trường ảo
Mở terminal tại thư mục dự án và chạy các lệnh sau:

*   **Trên Windows (PowerShell/CMD):**
    ```powershell
    python -m venv .venv
    .\.venv\Scripts\activate
    ```
*   **Trên macOS / Linux:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

### Bước 3: Cài đặt các thư viện phụ thuộc
Cài đặt tất cả các thư viện cần thiết đã được khai báo trong `requirements.txt`:
```bash
pip install -r requirements.txt
```

### Bước 4: Chạy ứng dụng Streamlit
Khởi chạy ứng dụng web trên trình duyệt cục bộ:
```bash
streamlit run app.py
```
Sau khi chạy lệnh, Streamlit sẽ tự động mở một tab trên trình duyệt của bạn (thường tại địa chỉ `http://localhost:8501`).

---

## 🚀 Hướng Dẫn Deploy Lên Streamlit Community Cloud (Miễn Phí)

Streamlit Cloud cho phép bạn deploy ứng dụng của mình trực tiếp từ kho lưu trữ GitHub hoàn toàn miễn phí.

### Bước 1: Đẩy mã nguồn lên GitHub
1. Tạo một kho lưu trữ (repository) mới trên tài khoản GitHub của bạn (ở chế độ **Public**).
2. Đẩy các tệp tin trong dự án lên GitHub:
    *   `app.py`
    *   `industry_tickers.py`
    *   `requirements.txt`
    *   `README.md`
    *(Lưu ý: Không đẩy thư mục `.venv` lên GitHub. Bạn nên tạo tệp `.gitignore` và thêm dòng `.venv/` vào đó).*

### Bước 2: Deploy ứng dụng trên Streamlit Cloud
1. Truy cập vào trang web [share.streamlit.io](https://share.streamlit.io/) và đăng nhập bằng tài khoản GitHub của bạn.
2. Nhấp vào nút **"New app"** ở góc trên cùng bên phải.
3. Cấu hình các thông tin ứng dụng:
    *   **Repository**: Chọn tên kho lưu trữ GitHub bạn vừa tạo.
    *   **Branch**: Chọn nhánh chứa mã nguồn (thường là `main` hoặc `master`).
    *   **Main file path**: Điền tên tệp chạy chính là `app.py`.
4. Nhấp vào nút **"Deploy!"**.
5. **Cấu hình phiên bản Python thích hợp**: 
    *   Sau khi bấm Deploy hoặc trong phần cấu hình nâng cao (**Advanced Settings**), vui lòng chọn phiên bản Python là **3.11** hoặc **3.12**. 
    *   *Lưu ý:* Nếu chạy trên Python 3.14 (phiên bản mặc định mới), Streamlit sẽ báo lỗi cài đặt dependencies do thư viện `tensorflow` chưa hỗ trợ Python 3.14. Ứng dụng đã được lập trình để tự động bỏ qua cài đặt tensorflow trên các phiên bản Python mới hơn và hiển thị cảnh báo hướng dẫn này. 

Streamlit Cloud sẽ tự động khởi tạo máy chủ, cài đặt các thư viện trong `requirements.txt` và chạy ứng dụng của bạn. Quá trình này có thể mất từ 1-3 phút trong lần đầu tiên. Sau khi hoàn tất, bạn sẽ nhận được một đường link URL công khai để chia sẻ ứng dụng của mình với mọi người!

---

## 💡 Các Tính Năng Chính Của Ứng Dụng

1.  **Lựa chọn Nhóm ngành linh hoạt:** Tải dữ liệu các mã cổ phiếu thời gian thực theo nhóm ngành lựa chọn từ `vnstock`.
2.  **Bộ lọc Sharpe Tối ưu:** Tự động lọc ra top 10 cổ phiếu có tỷ lệ Sharpe tốt nhất trong nhóm ngành để đưa vào huấn luyện mô hình.
3.  **Tùy chỉnh Tham số:** Cho phép thay đổi số lượng epochs, batch size, risk-free rate, và số lượng hạt giống huấn luyện (seeds) trực tiếp trên sidebar.
4.  **Trực quan hóa Sinh động:** Các biểu đồ tương tác cao bằng Plotly giúp dễ dàng theo dõi biến động giá, phân bổ tỷ trọng đầu tư và hiệu suất tích lũy của các chiến lược.
5.  **Bảng So sánh Hiệu suất:** Đánh giá chi tiết 3 chiến lược dựa trên các chỉ số tài chính tiêu chuẩn bao gồm: Lợi nhuận trung bình, Độ lệch chuẩn, và Hệ số Sharpe.
