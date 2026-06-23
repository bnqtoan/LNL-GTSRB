# LNL-GTSRB — Improved (Nhóm X, Deep Learning giữa kỳ)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bnqtoan/LNL-GTSRB-NhomX/blob/main/Instructions_NhomX.ipynb)

Bài giữa kỳ: cải tiến mô hình nhận dạng biển báo giao thông **GTSRB** dựa trên
**LNL (Locality-iN-Locality)** — Transformer-in-Transformer + cơ chế locality.
Mục tiêu chấm điểm: **Top-1 ≥ 99.5%**.

> **Academic fork.** Mã gốc thuộc về **Omid Nejati Manzari** (xem Citation cuối trang).
> Repo này chỉ dùng cho mục đích học tập; phần cải tiến nằm trong `LNL.py`.

## Kiểm chứng độ chính xác — KHÔNG cần train lại (plug-and-play)

Bấm **Open In Colab** ở trên (mở `Instructions_NhomX.ipynb`) → Runtime → **GPU (T4)** → **Run all**.

Notebook tự động:
1. `git clone` repo này (đã chứa `LNL.py` cải tiến);
2. tải trọng số đã train sẵn **`lnl_gtsrb.pth`** từ [GitHub Release](../../releases/latest);
3. dựng model **y hệt** `Instructions.ipynb` gốc: `LNL_Ti` → `head = Linear(192, 43)` → `.cuda()`;
4. nạp trọng số rồi chạy **đúng cell Test gốc** (ảnh thô `[0,1]`) → in **`Standard accuracy`**.

Số `Standard accuracy` ở mục 6 là kết quả Top-1.

> **Vì sao plug-and-play đúng:** cell Test gốc đưa vào ảnh thô `[0,1]` (chỉ `Resize+ToTensor`).
> Chuẩn hoá ImageNet được nhúng **bên trong** `LNL.py` (`forward_features`), và trọng số cũng được
> train trên đúng đường ống đó → train-time và verify-time dùng chung code path, không lệch tiền xử lý.

## Cải tiến (đều nằm trong `LNL.py`)

1. **Chuẩn hoá đầu vào trong model** (ImageNet mean/std) — để khớp cell Test gốc (ảnh thô `[0,1]`) và hội tụ tốt hơn.
2. **Augmentation trong model** (affine + random erasing), chỉ bật khi `self.training` (tự tắt lúc test). Không lật ngang vì biển báo nhạy hướng.
3. **LayerScale** trên mỗi nhánh residual (ổn định Transformer sâu — CaiT).
4. **qkv_bias = True**.
5. **Stochastic depth** (drop_path = 0.1).

## Deliverables

| File | Mô tả |
|------|-------|
| `LNL.py` | Model đã cải tiến (plug-and-play với `Instructions.ipynb` gốc) |
| `lnl_gtsrb.pth` | **Trọng số đã train sẵn** để kiểm chứng — tải ở **[Releases](../../releases/latest)** |
| `Instructions_NhomX.ipynb` | Notebook kiểm chứng: clone → tải `.pth` → nạp → chạy cell Test gốc |
| `LNL_train_colab.ipynb` | (Công cụ của nhóm) notebook train tạo ra `lnl_gtsrb.pth` — thầy không cần chạy |
| `dien_giai_mo_hinh.md` | Diễn giải ngắn gọn mô hình & cách kiểm chứng |

## Train lại (tuỳ chọn — tạo lại trọng số)

Mở `LNL_train_colab.ipynb` trên Colab (T4) → Run all. Notebook dùng AdamW + cosine+warmup + label
smoothing + EMA (~30 epoch), DataLoader ảnh thô `[0,1]` (normalize/aug do `LNL.py` lo), checkpoint +
auto-resume lên Google Drive, và tự kiểm chứng plug-and-play cuối notebook.

---

## Original work / Citation

This repository is an academic fork of **"Locality iN Locality"** by Omid Nejati Manzari et al.
Original repo: https://github.com/Omid-Nejati/Locality-iN-Locality · Paper: https://arxiv.org/abs/2301.11553

```
@article{manzari2023robust,
  title={Robust transformer with locality inductive bias and feature normalization},
  author={Manzari, Omid Nejati and Kashiani, Hossein and Dehkordi, Hojat Asgarian and Shokouhi, Shahriar B},
  journal={Engineering Science and Technology, an International Journal},
  volume={38}, pages={101320}, year={2023}, publisher={Elsevier}
}
```
