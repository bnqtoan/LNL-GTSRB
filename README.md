# LNL-GTSRB — Improved (Nhóm X, Deep Learning giữa kỳ)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bnqtoan/LNL-GTSRB-NhomX/blob/main/Instructions_NhomX.ipynb)

Bài giữa kỳ: cải tiến mô hình nhận dạng biển báo giao thông **GTSRB** dựa trên
**LNL (Locality-iN-Locality)** — Transformer-in-Transformer + cơ chế locality.

> **Academic fork.** Mã gốc thuộc về **Omid Nejati Manzari** (xem Citation cuối trang).
> Repo này chỉ dùng cho mục đích học tập; phần cải tiến nằm trong `LNL.py`.

## Cách chạy (plug-and-play)

Bấm nút **Open In Colab** ở trên → Runtime → GPU (T4) → Run all.
Notebook tự `git clone` repo này (đã chứa `LNL.py` cải tiến) rồi chạy nguyên luồng gốc.

## Cải tiến (đều nằm trong `LNL.py`)

Mọi thay đổi nằm bên trong file model, tự kích hoạt khi chạy notebook:

1. **Chuẩn hoá đầu vào trong model** (ImageNet mean/std) — dataloader gốc không normalize; đưa vào `forward` giúp hội tụ tốt hơn và để pretrained weights plug-and-play với cell Test gốc (ảnh thô [0,1]).
2. **Augmentation trong model** (affine + random erasing), chỉ bật khi `self.training` (tự tắt lúc test). Không lật ngang vì biển báo nhạy hướng.
3. **LayerScale** trên mỗi nhánh residual (ổn định Transformer sâu — CaiT).
4. **qkv_bias = True**.
5. **Stochastic depth** (drop_path = 0.1).

## Deliverables (giữa kỳ)

| File | Mô tả |
|------|-------|
| `LNL.py` | Model đã cải tiến (plug-and-play với Instructions.ipynb gốc) |
| `Instructions_NhomX.ipynb` | Notebook có thêm bước **lưu** (`torch.save`) và **nạp** (`load_state_dict`) trọng số |
| `lnl_gtsrb.pth` | Pretrained model để kiểm chứng (tạo ra khi chạy cell "Lưu mô hình") |

## Kiểm chứng không cần train lại

Build model (cell 20–23) → chạy cell **"Nạp pretrained"** (`model.load_state_dict('lnl_gtsrb.pth')`) → chạy cell **Test**.

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
