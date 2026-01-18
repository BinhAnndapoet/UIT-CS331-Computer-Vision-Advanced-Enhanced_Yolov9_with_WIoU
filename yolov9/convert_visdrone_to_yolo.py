import os
from PIL import Image
from tqdm import tqdm
from pathlib import Path

# Cấu hình đường dẫn
base_dir = Path('data/visdrone')
splits = ['VisDrone2019-DET-train', 'VisDrone2019-DET-val', 'VisDrone2019-DET-test-dev']

def convert_box(size, box):
    # Chuyển từ (x_min, y_min, w, h) sang (x_center, y_center, w, h) chuẩn hóa
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = box[0] + box[2] / 2.0
    y = box[1] + box[3] / 2.0
    w = box[2]
    h = box[3]
    
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)

print("--- BẮT ĐẦU CHUYỂN ĐỔI VISDRONE SANG YOLO ---")

for split in splits:
    split_dir = base_dir / split
    images_dir = split_dir / 'images'
    # Folder chứa nhãn gốc (đang bị nhầm là labels, ta sẽ đọc từ đây)
    labels_dir = split_dir / 'labels' 
    
    if not labels_dir.exists():
        print(f"Bỏ qua {split} vì không thấy folder labels")
        continue

    print(f"Đang xử lý {split}...")
    
    # Lấy danh sách file
    label_files = list(labels_dir.glob('*.txt'))
    
    for label_file in tqdm(label_files):
        image_file = images_dir / (label_file.stem + '.jpg')
        
        # VisDrone đôi khi thiếu ảnh hoặc lệch tên, kiểm tra tồn tại
        if not image_file.exists():
            continue
            
        # 1. Đọc kích thước ảnh để chuẩn hóa
        with Image.open(image_file) as img:
            w_img, h_img = img.size

        # 2. Đọc file nhãn cũ
        with open(label_file, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split(',')
            # VisDrone format: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
            if len(parts) < 6: continue
            
            try:
                cls_id = int(parts[5])
                # VisDrone class mapping:
                # 0:Ignore(bỏ), 1:Pedestrian, 2:People, 3:Bicycle, 4:Car, 5:Van, 
                # 6:Truck, 7:Tricycle, 8:Awning-tricycle, 9:Bus, 10:Motor, 11:Others(bỏ)
                
                # Chỉ lấy các class từ 1 đến 10
                if cls_id == 0 or cls_id == 11:
                    continue
                
                # YOLO class phải bắt đầu từ 0. Vậy map: 1->0, 2->1, ..., 10->9
                new_cls_id = cls_id - 1
                
                # Lấy tọa độ
                box = (float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]))
                
                # Convert sang YOLO format
                bb = convert_box((w_img, h_img), box)
                
                # Ghi dòng mới: class x y w h
                new_line = f"{new_cls_id} {bb[0]:.6f} {bb[1]:.6f} {bb[2]:.6f} {bb[3]:.6f}\n"
                new_lines.append(new_line)
                
            except ValueError:
                continue

        # 3. Ghi đè lại nội dung chuẩn YOLO vào file cũ
        with open(label_file, 'w') as f:
            f.writelines(new_lines)

print("--- CHUYỂN ĐỔI HOÀN TẤT ---")