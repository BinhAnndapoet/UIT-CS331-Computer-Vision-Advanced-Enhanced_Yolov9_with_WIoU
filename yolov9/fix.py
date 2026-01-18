import os
import shutil
from pathlib import Path

# Đường dẫn gốc tới dataset VisDrone
base_dir = Path('data/visdrone')

# Danh sách 3 tập con
splits = ['VisDrone2019-DET-train', 'VisDrone2019-DET-val', 'VisDrone2019-DET-test-dev']

print("--- BẮT ĐẦU SỬA CẤU TRÚC VISDRONE ---")

for split in splits:
    split_path = base_dir / split
    
    # Đường dẫn cũ (annotations) và mới (labels)
    old_labels_dir = split_path / 'annotations'
    new_labels_dir = split_path / 'labels'
    
    if not split_path.exists():
        print(f"⚠️  Không tìm thấy: {split_path}")
        continue

    # 1. Đổi tên folder 'annotations' -> 'labels'
    if old_labels_dir.exists():
        if not new_labels_dir.exists():
            print(f"✅ {split}: Đổi tên 'annotations' -> 'labels'")
            os.rename(old_labels_dir, new_labels_dir)
        else:
            # Nếu folder labels đã lỡ tạo rồi nhưng rỗng, thì gộp vào
            print(f"⚠️ {split}: Folder 'labels' đã tồn tại. Đang di chuyển file...")
            for f in old_labels_dir.glob('*.txt'):
                shutil.move(str(f), str(new_labels_dir))
            # Xóa folder cũ nếu rỗng
            try:
                old_labels_dir.rmdir()
            except:
                pass
    elif new_labels_dir.exists():
        print(f"🆗 {split}: Đã có folder 'labels' (Đúng chuẩn).")
    else:
        print(f"❌ {split}: Không tìm thấy folder nhãn nào!")

print("\n--- HOÀN TẤT! SẴN SÀNG TRAIN ---")
# Kiểm tra lại
os.system(f"ls -R {base_dir} | grep ':$\\|txt$' | head -n 10")