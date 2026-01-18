import os
from pathlib import Path

# Cấu hình đường dẫn
base_path = Path('data/coco5k_clean')
images_path = base_path / 'images'
labels_path = base_path / 'labels'

# Lấy tập hợp tên file (chỉ lấy tên, bỏ đuôi mở rộng)
image_files = {f.stem for f in images_path.iterdir() if f.is_file() and not f.name.startswith('.')}
label_files = {f.stem for f in labels_path.iterdir() if f.is_file() and not f.name.startswith('.')}

# --- CƠ CHẾ KIỂM TRA KHỚP (INTERSECTION) ---
matched_files = image_files.intersection(label_files) # Phép giao
# -------------------------------------------

print(f"📂 Tổng file trong folder Images: {len(image_files)}")
print(f"📂 Tổng file trong folder Labels: {len(label_files)}")
print("-" * 30)
print(f"✅ SỐ LƯỢNG CẶP KHỚP NHAU (VALID PAIRS): {len(matched_files)}")
print("-" * 30)

# Kiểm tra chi tiết
if len(matched_files) == len(image_files) == len(label_files):
    print("Tuyệt vời. Dữ liệu đồng bộ 100%.")
else:
    print(f"❌ Dữ liệu không đồng bộ.")
    print(f"Model sẽ chỉ train trên {len(matched_files)} mẫu này.")
    
    # Chỉ ra cụ thể
    img_only = image_files - label_files
    lbl_only = label_files - image_files
    
    if img_only: print(f"-> Có {len(img_only)} ảnh thừa (không có nhãn).")
    if lbl_only: print(f"-> Có {len(lbl_only)} nhãn thừa (không có ảnh).")