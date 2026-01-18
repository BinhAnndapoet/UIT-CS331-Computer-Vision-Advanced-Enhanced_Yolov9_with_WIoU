import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import os
import sys
from pathlib import Path

# =====================================================
# CẤU HÌNH ĐƯỜNG DẪN TUYỆT ĐỐI (YOLOv9 ROOT)
# =====================================================
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
os.chdir(ROOT)

# =====================================================
# IMPORT MODULES YOLOv9
# =====================================================
try:
    from models.common import DetectMultiBackend
    from utils.general import (
        check_img_size,
        non_max_suppression,
        scale_boxes
    )
    from utils.plots import Annotator, colors
    from utils.torch_utils import select_device
    from utils.augmentations import letterbox
except ImportError as e:
    st.error(f"Lỗi import YOLOv9: {e}")
    st.stop()

# =====================================================
# LOAD MODEL (CACHE)
# =====================================================
@st.cache_resource(show_spinner=False)
def load_model(weights_path):
    device = select_device('')
    model = DetectMultiBackend(
        weights_path,
        device=device,
        dnn=False,
        data=None,
        fp16=False
    )
    model.eval()
    return model

# =====================================================
# INFERENCE
# =====================================================
def run_inference(model, image, conf_thres, iou_thres, img_size=640):
    img0 = np.array(image)

    # Resize + padding
    img = letterbox(
        img0,
        img_size,
        stride=model.stride,
        auto=True
    )[0]

    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(model.device)
    img = img.float() / 255.0
    if img.ndim == 3:
        img = img.unsqueeze(0)

    # Inference
    pred = model(img, augment=False, visualize=False)

    # NMS
    pred = non_max_suppression(
        pred,
        conf_thres,
        iou_thres,
        classes=None,
        agnostic=False,
        max_det=1000
    )

    det = pred[0]
    img_result = img0.copy()

    if len(det):
        det[:, :4] = scale_boxes(
            img.shape[2:],
            det[:, :4],
            img0.shape
        ).round()

        annotator = Annotator(img_result, line_width=3)

        for *xyxy, conf, cls in det:
            label = f"{model.names[int(cls)]} {conf:.2f}"
            annotator.box_label(
                xyxy,
                label,
                color=colors(int(cls), True)
            )

        img_result = annotator.result()

    return img_result

# =====================================================
# STREAMLIT UI
# =====================================================
st.set_page_config(
    page_title="YOLOv9 WIoU Demo",
    layout="wide"
)

st.title("🔍 YOLOv9 – CIoU vs WIoU Demo")
st.markdown("---")

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.header("⚙️ Cấu hình")

model_options = {
    "CIoU coco": "runs/train/yolov9_coco5k_ciou/weights/best.pt",
    "WIoU v3 coco": "runs/train/yolov9_coco5k_wiou_v3/weights/best.pt",
    "WIoU v2 coco": "runs/train/yolov9_coco5k_wiou_v2/weights/best.pt",
    "WIoU v1 coco": "runs/train/yolov9_coco5k_wiou_v1/weights/best.pt",
    "CIoU visdrone": "runs/train/yolov9_visdrone_ciou/weights/best.pt",
    "WIoU v3 visdrone": "runs/train/yolov9_visdrone_wiou_v3/weights/best.pt",
    "WIoU v2 visdrone": "runs/train/yolov9_visdrone_wiou_v2/weights/best.pt",
}

selected_model_name = st.sidebar.selectbox(
    "Chọn Model",
    list(model_options.keys())
)
weights_path = model_options[selected_model_name]

if not os.path.exists(weights_path):
    st.sidebar.warning(f"⚠️ Không tìm thấy: {weights_path}")

conf_thres = st.sidebar.slider(
    "Confidence Threshold",
    0.0, 1.0, 0.25, 0.05
)
iou_thres = st.sidebar.slider(
    "IoU Threshold (NMS)",
    0.0, 1.0, 0.45, 0.05
)
img_size = st.sidebar.number_input(
    "Input Image Size",
    value=640,
    step=32
)

# =====================================================
# MAIN
# =====================================================
col1, col2 = st.columns(2)

with col1:
    st.subheader("1️⃣ Tải ảnh")
    uploaded_file = st.file_uploader(
        "Chọn ảnh",
        type=["jpg", "jpeg", "png"]
    )

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    with col1:
        st.image(
            image,
            caption="Ảnh gốc",
            use_container_width=True
        )

    if st.sidebar.button("🚀 Chạy Detect", type="primary"):
        if os.path.exists(weights_path):
            with st.spinner("Đang xử lý..."):
                try:
                    model = load_model(weights_path)
                    img_size = check_img_size(img_size, s=model.stride)
                    result_img = run_inference(
                        model,
                        image,
                        conf_thres,
                        iou_thres,
                        img_size
                    )

                    with col2:
                        st.subheader("2️⃣ Kết quả")
                        st.image(
                            result_img,
                            caption=selected_model_name,
                            use_container_width=True
                        )
                        st.success("Hoàn tất")
                except Exception as e:
                    st.error(f"Lỗi: {e}")
        else:
            st.error("Không tìm thấy file weights")
else:
    with col2:
        st.info("👈 Upload ảnh để bắt đầu")

# =====================================================
# FOOTER
# =====================================================
st.markdown("---")
st.caption(
    "Đồ án Computer Vision Nâng cao – YOLOv9 & WIoU"
)
