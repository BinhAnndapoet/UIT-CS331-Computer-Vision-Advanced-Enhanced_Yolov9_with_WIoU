import cv2
import os

# --- CẤU HÌNH ---
TARGET_FILE = '000000001296.jpg'             # Tên ảnh bạn chọn
CLEAN_DIR = 'data/coco5k_clean'              # Thư mục gốc
NOISY_DIR = 'data/coco5k_clean_noisy_single' # Thư mục nhiễu (đã tạo ở bước trước)
OUTPUT_FILE = f'compare_{TARGET_FILE}'       # Tên ảnh kết quả

def yolo_to_pixel(x, y, w, h, img_w, img_h):
    # Chuyển đổi YOLO (0-1) -> Pixel (x1, y1, x2, y2)
    x1 = int((x - w / 2) * img_w)
    y1 = int((y - h / 2) * img_h)
    x2 = int((x + w / 2) * img_w)
    y2 = int((y + h / 2) * img_h)
    return x1, y1, x2, y2

def visualize_specific_structured():
    print(f"--- Đang tìm kiếm ảnh: {TARGET_FILE} ---")

    # 1. Tìm xem ảnh nằm trong 'train' hay 'val'
    found_split = None
    img_path = None
    
    # Duyệt qua các sub-folder phổ biến
    for split in ['train', 'val', 'test']:
        possible_path = os.path.join(CLEAN_DIR, 'images', split, TARGET_FILE)
        if os.path.exists(possible_path):
            found_split = split
            img_path = possible_path
            break
    
    if not found_split:
        print(f"❌ Lỗi: Không tìm thấy ảnh {TARGET_FILE} trong thư mục images/train hoặc images/val của {CLEAN_DIR}")
        return

    print(f"✅ Đã tìm thấy ảnh trong tập: '{found_split}'")

    # 2. Xác định đường dẫn file Label tương ứng
    label_name = TARGET_FILE.replace('.jpg', '.txt').replace('.png', '.txt')
    
    # Label Gốc
    clean_lbl_path = os.path.join(CLEAN_DIR, 'labels', found_split, label_name)
    # Label Nhiễu (Cấu trúc tương tự)
    noisy_lbl_path = os.path.join(NOISY_DIR, 'labels', found_split, label_name)

    # Kiểm tra tồn tại
    if not os.path.exists(clean_lbl_path):
        print(f"❌ Lỗi: Không tìm thấy file nhãn gốc tại {clean_lbl_path}")
        return
    if not os.path.exists(noisy_lbl_path):
        print(f"❌ Lỗi: Chưa có file nhãn nhiễu tại {noisy_lbl_path}. Bạn đã chạy script tạo nhiễu chưa?")
        return

    # 3. Đọc ảnh và nhãn
    img = cv2.imread(img_path)
    if img is None:
        print("❌ Lỗi: Không đọc được nội dung ảnh.")
        return
    h_img, w_img, _ = img.shape

    with open(clean_lbl_path, 'r') as f:
        clean_lines = f.readlines()
    with open(noisy_lbl_path, 'r') as f:
        noisy_lines = f.readlines()

    # 4. Vẽ Box Gốc (Màu XANH LÁ - Green)
    for line in clean_lines:
        parts = [float(x) for x in line.strip().split()]
        if len(parts) < 5: continue
        bbox = parts[1:]
        x1, y1, x2, y2 = yolo_to_pixel(*bbox, w_img, h_img)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # 5. Vẽ Box Nhiễu (Màu ĐỎ - Red)
    count_diff = 0
    for line in noisy_lines:
        parts = [float(x) for x in line.strip().split()]
        if len(parts) < 5: continue
        bbox = parts[1:]
        x1, y1, x2, y2 = yolo_to_pixel(*bbox, w_img, h_img)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # 6. Thêm chú thích
    cv2.putText(img, "Green: Original (Clean)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(img, "Red: Noisy (Outlier)", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    # 7. Lưu kết quả
    cv2.imwrite(OUTPUT_FILE, img)
    print(f"✅ Xong! Ảnh so sánh đã lưu tại: {os.path.abspath(OUTPUT_FILE)}")
    print("👉 Hãy mở ảnh này lên để thấy sự khác biệt của box màu Đỏ!")

if __name__ == "__main__":
    visualize_specific_structured()