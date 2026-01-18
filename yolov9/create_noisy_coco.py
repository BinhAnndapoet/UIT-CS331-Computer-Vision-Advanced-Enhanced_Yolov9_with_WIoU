import os
import shutil
import random
import yaml
from tqdm import tqdm

def create_single_noisy_custom_dataset(
    source_dir='data/coco5k_clean', 
    target_dir='data/coco5k_clean_noisy_single', 
    config_file='data/coco5k_clean.yaml',
    noise_level=0.05
):
    print(f"--- Bắt đầu tạo dữ liệu nhiễu từ: {source_dir} ---")
    
    # 1. Xóa thư mục đích nếu đã tồn tại để làm mới
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    
    # 2. Copy toàn bộ thư mục images (Giữ nguyên ảnh gốc)
    print(f"-> Đang copy thư mục images...")
    shutil.copytree(f'{source_dir}/images', f'{target_dir}/images')
    
    # 3. Xử lý thư mục labels (Giữ nguyên cấu trúc thư mục con nếu có)
    source_labels = f'{source_dir}/labels'
    target_labels = f'{target_dir}/labels'
    
    # Đếm tổng số file để hiển thị thanh tiến trình
    total_files = sum([len(files) for r, d, files in os.walk(source_labels) if any(f.endswith('.txt') for f in files)])
    
    print(f"-> Đang tạo nhiễu cho {total_files} file labels (1 object/ảnh)...")
    
    # Dùng os.walk để duyệt đệ quy (phòng trường hợp bên trong labels có chia thư mục train/val)
    pbar = tqdm(total=total_files)
    
    for root, dirs, files in os.walk(source_labels):
        # Tạo thư mục tương ứng bên đích
        relative_path = os.path.relpath(root, source_labels)
        target_root = os.path.join(target_labels, relative_path)
        os.makedirs(target_root, exist_ok=True)
        
        for file_name in files:
            if not file_name.endswith('.txt'):
                continue
                
            pbar.update(1)
            
            # Đọc file label gốc
            src_file_path = os.path.join(root, file_name)
            tgt_file_path = os.path.join(target_root, file_name)
            
            with open(src_file_path, 'r') as f:
                lines = f.readlines()
            
            new_lines = []
            
            if not lines:
                # File rỗng thì copy rỗng
                with open(tgt_file_path, 'w') as f:
                    pass
                continue

            # --- LOGIC CHỌN 1 OBJECT ĐỂ LÀM NHIỄU ---
            idx_to_noise = random.randint(0, len(lines) - 1)
            
            for i, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5: continue # Bỏ qua dòng lỗi

                # Nếu là dòng được chọn -> Làm nhiễu
                if i == idx_to_noise:
                    cls = parts[0]
                    coords = [float(x) for x in parts[1:]]
                    
                    # Tạo nhiễu ngẫu nhiên (-5% đến 5%)
                    noise = [random.uniform(-noise_level, noise_level) for _ in range(4)]
                    
                    # Cộng nhiễu và giới hạn trong khung [0.001, 0.999]
                    new_coords = [
                        min(max(coords[j] + noise[j], 0.001), 0.999) for j in range(4)
                    ]
                    
                    new_line = f"{cls} " + " ".join([f"{x:.6f}" for x in new_coords]) + "\n"
                    new_lines.append(new_line)
                else:
                    # Các dòng khác giữ nguyên
                    new_lines.append(line)
            
            # Ghi file mới
            with open(tgt_file_path, 'w') as f:
                f.writelines(new_lines)
                
    pbar.close()

    # 4. Xử lý file YAML
    if os.path.exists(config_file):
        print(f"-> Đang tạo file config mới...")
        with open(config_file, 'r') as f:
            data = yaml.safe_load(f)
        
        # Cập nhật đường dẫn path
        # Lưu ý: YOLOv9/v5 thường yêu cầu absolute path hoặc path tương đối chuẩn
        # Ở đây ta trỏ path về thư mục mới
        data['path'] = os.path.abspath(target_dir) 
        
        # Tạo tên file yaml mới
        new_config_name = f'{target_dir}.yaml'.replace('data/', '') # ex: coco5k_clean_noisy_single.yaml
        new_config_path = os.path.join('data', new_config_name)
        
        with open(new_config_path, 'w') as f:
            yaml.dump(data, f)
            
        print(f"✅ Hoàn tất! Dataset mới tại: {target_dir}")
        print(f"✅ File config mới tại: {new_config_path}")
    else:
        print(f"⚠️ Không tìm thấy file config {config_file}, vui lòng tạo thủ công.")

# --- CHẠY HÀM ---
# Đảm bảo bạn đang đứng ở thư mục gốc (yolov9) khi chạy lệnh này
create_single_noisy_custom_dataset()