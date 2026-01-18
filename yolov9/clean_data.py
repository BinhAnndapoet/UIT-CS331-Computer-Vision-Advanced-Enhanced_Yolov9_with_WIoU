import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm  # Thư viện hiển thị thanh tiến trình (cần pip install tqdm nếu chưa có)

# ================= CẤU HÌNH =================
ROOT_DIR = Path('data/coco5k_clean')
IMAGES_DIR = ROOT_DIR / 'images'
LABELS_DIR = ROOT_DIR / 'labels'

VAL_RATIO = 0.2  # 20% cho Validation, 80% cho Train
SEED = 42        # Giữ cố định để lần sau chạy lại vẫn chia y hệt (Reproducibility)
# ============================================

def split_dataset():
    # 1. Kiểm tra thư mục gốc
    if not IMAGES_DIR.exists() or not LABELS_DIR.exists():
        print("❌ Lỗi: Không tìm thấy thư mục images hoặc labels.")
        return

    # 2. Lấy danh sách file (chỉ lấy tên, không đuôi)
    # Lưu ý: Chỉ quét file ở root folder images/, bỏ qua các subfolder nếu đã chạy rồi
    files = [f.stem for f in IMAGES_DIR.iterdir() if f.is_file() and not f.name.startswith('.')]
    
    if len(files) == 0:
        print("❌ Không tìm thấy file ảnh nào để chia (hoặc file đã nằm trong subfolder).")
        return

    # 3. Xáo trộn ngẫu nhiên
    random.seed(SEED)
    random.shuffle(files)

    # 4. Tính toán số lượng
    val_count = int(len(files) * VAL_RATIO)
    train_count = len(files) - val_count

    print(f"📊 Tổng file: {len(files)}")
    print(f"➡️ Train: {train_count} | Val: {val_count}")

    # 5. Định nghĩa đường dẫn đích
    dirs = {
        'train': {
            'images': IMAGES_DIR / 'train',
            'labels': LABELS_DIR / 'train'
        },
        'val': {
            'images': IMAGES_DIR / 'val',
            'labels': LABELS_DIR / 'val'
        }
    }

    # Tạo thư mục đích
    for split in dirs:
        dirs[split]['images'].mkdir(parents=True, exist_ok=True)
        dirs[split]['labels'].mkdir(parents=True, exist_ok=True)

    # 6. Di chuyển file (Move)
    # Danh sách file cho từng tập
    splits = {
        'val': files[:val_count],
        'train': files[val_count:]
    }

    print("\n🚀 Đang di chuyển file...")
    
    for split_name, split_files in splits.items():
        dest_img_dir = dirs[split_name]['images']
        dest_lbl_dir = dirs[split_name]['labels']
        
        # Dùng tqdm để hiện thanh loading cho chuyên nghiệp
        for name in tqdm(split_files, desc=f"Processing {split_name.upper()}"):
            # Đường dẫn file gốc
            # Cần tìm đúng đuôi file ảnh (jpg, png, jpeg...)
            # Cách tìm nhanh: quét file bắt đầu bằng tên đó trong folder gốc
            src_img_candidates = list(IMAGES_DIR.glob(f"{name}.*"))
            src_lbl = LABELS_DIR / f"{name}.txt"
            
            if src_img_candidates and src_lbl.exists():
                src_img = src_img_candidates[0] # Lấy file ảnh đầu tiên khớp tên
                
                # Di chuyển (Move)
                shutil.move(str(src_img), str(dest_img_dir / src_img.name))
                shutil.move(str(src_lbl), str(dest_lbl_dir / src_lbl.name))
            else:
                print(f"⚠️ Lỗi file: {name} (Có thể thiếu cặp ảnh/nhãn)")

    print("\n✅ Hoàn tất! Cấu trúc thư mục mới:")
    print(f"   {IMAGES_DIR}/train")
    print(f"   {IMAGES_DIR}/val")
    print(f"   {LABELS_DIR}/train")
    print(f"   {LABELS_DIR}/val")

if __name__ == "__main__":
    # Cài tqdm nếu chưa có: pip install tqdm
    try:
        from tqdm import tqdm
    except ImportError:
        print("⚠️ Chưa cài tqdm. Đang chạy chế độ basic...")
        # Mock tqdm nếu user chưa cài
        def tqdm(iterator, desc=""):
            print(f"-- {desc} --")
            return iterator
            
    split_dataset()