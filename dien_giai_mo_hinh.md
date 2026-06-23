# Diễn giải mô hình đề xuất — Nhận dạng biển báo giao thông (GTSRB)

**Mô hình nền:** LNL (Locality-iN-Locality) — Transformer-in-Transformer (TNT) + cơ chế locality.
**Bộ dữ liệu:** GTSRB — 43 lớp biển báo giao thông Đức (ảnh 224×224).
**Kết quả:** Top-1 = **____%** (điền số `Standard accuracy` từ `Instructions_NhomX.ipynb`, mục tiêu ≥ 99.5%).

---

## 1. Sản phẩm nộp & cách kiểm chứng (plug-and-play)

Nhóm nộp **2 thứ**:
1. `LNL.py` — file mô hình đã cải tiến.
2. `lnl_gtsrb.pth` — **trọng số đã train sẵn** (đặt ở GitHub Release của repo nhóm).

**Thầy kiểm chứng KHÔNG cần train lại:**
- Mở `Instructions_NhomX.ipynb` → **Run all**. Notebook tự: clone repo nhóm → tải `lnl_gtsrb.pth` →
  dựng model **y hệt** `Instructions.ipynb` gốc (`LNL_Ti` → `head = Linear(192, 43)`) → nạp trọng số →
  chạy **đúng cell Test gốc** → in `Standard accuracy`.

> **Khớp tiền xử lý (điểm mấu chốt):** cell Test gốc đưa vào **ảnh thô `[0,1]`** (chỉ `Resize+ToTensor`,
> KHÔNG normalize). Vì vậy chuẩn hoá ImageNet được **nhúng bên trong `LNL.py`** (`forward_features`).
> Trọng số nhóm cũng được train trên đúng đường ống đó (dataloader thô `[0,1]`, normalize trong model),
> nên train-time và verify-time **dùng chung một code path** → không lệch tiền xử lý.

## 2. Cải tiến — đều nằm trong `LNL.py`

| # | Cải tiến | Vị trí | Vì sao giúp |
|---|----------|--------|-------------|
| 1 | **Chuẩn hoá đầu vào trong model** (ImageNet mean/std) | `forward_features` → `_preprocess` | Dataloader gốc KHÔNG normalize. Đưa normalize vào model → khớp hoàn toàn với cell Test gốc (ảnh thô `[0,1]`) và giúp hội tụ tốt hơn. |
| 2 | **Augmentation trong model, chỉ khi train** (affine xoay/dịch/scale + random erasing) | `_batch_augment`, gọi trong `_preprocess` khi `self.training` | Tăng tính bền vững khi train. Tự **TẮT** khi `model.eval()` (lúc Test) nhờ cờ `self.training`. Không lật ngang (biển báo nhạy hướng). |
| 3 | **LayerScale** trên mỗi nhánh residual (CaiT) | class `Block` | Ổn định Transformer sâu (12 block), hội tụ tốt hơn. |
| 4 | **qkv_bias = True** | `LNL_Ti` | Tăng nhẹ, ổn định. |
| 5 | **Stochastic depth** (drop_path 0.1) | constructor | Regularization. |

Augmentation (2) chỉ chạy lúc nhóm train; lúc thầy kiểm chứng model ở chế độ `eval()` nên (2) không kích hoạt —
không ảnh hưởng kết quả Test. (1), (3), (4), (5) là thay đổi kiến trúc/tiền xử lý, luôn hiện diện trong trọng số.

## 3. Công thức train để tạo trọng số (công cụ của nhóm: `LNL_train_colab.ipynb`)

Đây là cách nhóm tạo ra `lnl_gtsrb.pth`. Thầy không cần chạy lại.

- Optimizer **AdamW** (lr 6e-4, weight_decay 0.05), **cosine LR + warmup** 3 epoch, **~30 epoch**.
- **Label smoothing 0.1**, **AMP** (`torch.amp`), **EMA** (decay 0.9995) — chọn bản (raw/EMA) tốt nhất trên tập val tách từ train.
- DataLoader **ảnh thô `[0,1]`** (chỉ `Resize+ToTensor`) — normalize + augmentation do `LNL.py` lo.
- Head = **`Linear(192, 43)`** (đúng head thầy dựng) → `state_dict` khớp 1:1, `load_state_dict(strict=True)` sạch.
- Checkpoint mỗi epoch lên Google Drive + **tự resume** (chịu được ngắt kết nối ~2h trên T4).
- **Tự kiểm chứng cuối notebook:** dựng lại model từ đầu như thầy → `load_state_dict` → `eval()` →
  chạy cell Test gốc trên ảnh thô `[0,1]` → in số. Số này phải ≥ 99.5% mới ship.

> Lưu ý: cell Test gốc **không** dùng TTA. Vì vậy nhóm lưu trọng số **không-TTA**; bản thân trọng số (qua cell Test gốc) đã ≥ 99.5%.

## 4. Kết quả

| Cấu hình | Top-1 |
|----------|-------|
| LNL gốc (5 epoch SGD, không aug/normalize) | ~97% |
| **LNL.py cải tiến + công thức train mạnh, nạp trọng số `lnl_gtsrb.pth`** | **____%** (điền từ `Instructions_NhomX.ipynb`) |

## 5. Thiết lập kiểm chứng
- Model: LNL-Ti (TNT) + in-model normalize + in-model augment (train-only) + LayerScale + qkv_bias=True + drop_path 0.1.
- Phần cứng: 1× NVIDIA T4 (Colab).
- Ảnh: 224×224 (positional embedding của LNL cố định, không đổi).
- Nộp: `LNL.py` + `lnl_gtsrb.pth` (GitHub Release) + ảnh chụp `Standard accuracy`.
