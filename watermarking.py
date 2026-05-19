import argparse
import io
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# ─────────────────────────────────────────────
# 1. WATERMARK GENERATION
# ─────────────────────────────────────────────

def generate_binary_watermark(size=(32, 32)):
    """Binary checkerboard watermark."""
    wm = np.zeros(size, dtype=np.float32)
    for i in range(size[0]):
        for j in range(size[1]):
            if (i + j) % 2 == 0:
                wm[i, j] = 1.0
    return wm


def generate_random_watermark(size=(32, 32), seed=42):
    """Pseudo-random binary watermark (seed = secret key)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=size).astype(np.float32)


# ─────────────────────────────────────────────
# 2. DCT EMBEDDING
# ─────────────────────────────────────────────

# Mid-frequency zigzag positions in an 8x8 DCT block
MID_FREQ_POSITIONS = [(3, 2), (2, 3), (4, 1), (1, 4), (3, 3), (4, 2), (2, 4)]


def embed_dct(image_bgr, watermark, alpha=20):
    img = image_bgr.copy()
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    Y = ycrcb[:, :, 0]

    H, W = Y.shape
    wm_bits = watermark.flatten()
    num_bits = len(wm_bits)

    bit_idx = 0
    blocks_used = 0

    for row in range(0, H - 7, 8):
        for col in range(0, W - 7, 8):
            if bit_idx >= num_bits:
                break
            block = Y[row:row+8, col:col+8]
            dct_block = cv2.dct(block)

            pos = MID_FREQ_POSITIONS[bit_idx % len(MID_FREQ_POSITIONS)]
            bit = wm_bits[bit_idx]

            if bit == 1:
                dct_block[pos] += alpha
            else:
                dct_block[pos] -= alpha

            Y[row:row+8, col:col+8] = cv2.idct(dct_block)
            bit_idx += 1
            blocks_used += 1

    ycrcb[:, :, 0] = np.clip(Y, 0, 255)
    watermarked_bgr = cv2.cvtColor(ycrcb.astype(np.uint8), cv2.COLOR_YCrCb2BGR)
    return watermarked_bgr, wm_bits, blocks_used


# ─────────────────────────────────────────────
# 3. JPEG COMPRESSION
# ─────────────────────────────────────────────

def jpeg_compress(image_bgr, quality):
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    decompressed = np.array(Image.open(buffer))
    return cv2.cvtColor(decompressed, cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────
# 4. DCT EXTRACTION
# ─────────────────────────────────────────────

def extract_dct(original_bgr, compressed_bgr, wm_shape, alpha=20):
    """
    Extract watermark from DCT coefficients by comparing original
    and compressed images. Decision rule:
        bit = 1  if  coeff_compressed > coeff_original
        bit = 0  otherwise
    """
    def get_Y(bgr):
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb).astype(np.float32)[:, :, 0]

    Y_orig = get_Y(original_bgr)
    Y_comp = get_Y(compressed_bgr)

    H, W = Y_orig.shape
    num_bits = wm_shape[0] * wm_shape[1]
    extracted_bits = []
    bit_idx = 0

    for row in range(0, H - 7, 8):
        for col in range(0, W - 7, 8):
            if bit_idx >= num_bits:
                break
            dct_orig = cv2.dct(Y_orig[row:row+8, col:col+8])
            dct_comp = cv2.dct(Y_comp[row:row+8, col:col+8])

            pos = MID_FREQ_POSITIONS[bit_idx % len(MID_FREQ_POSITIONS)]
            diff = dct_comp[pos] - dct_orig[pos]
            extracted_bits.append(1.0 if diff > 0 else 0.0)
            bit_idx += 1

    extracted_bits = np.array(extracted_bits, dtype=np.float32)
    if len(extracted_bits) < num_bits:
        extracted_bits = np.pad(extracted_bits, (0, num_bits - len(extracted_bits)))
    return extracted_bits.reshape(wm_shape)


# ─────────────────────────────────────────────
# 5. METRICS
# ─────────────────────────────────────────────

def bit_error_rate(original_wm, extracted_wm):
    return float(np.sum(original_wm != extracted_wm) / original_wm.size)


def normalized_correlation(original_wm, extracted_wm):
    o = 2.0 * original_wm.flatten() - 1.0
    e = 2.0 * extracted_wm.flatten() - 1.0
    return float(np.dot(o, e) / (np.linalg.norm(o) * np.linalg.norm(e) + 1e-12))


def psnr(img1, img2):
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    return float('inf') if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


# ─────────────────────────────────────────────
# 6. MAIN EVALUATION PIPELINE
# ─────────────────────────────────────────────

def evaluate(image_path, watermark_type="random", alpha=20, seed=42):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    print(f"[INFO] Loaded image: {image_path}  shape={image.shape}")

    H_img, W_img = image.shape[:2]
    wm_size = (H_img // 8, W_img // 8)
    if watermark_type == "binary":
        watermark = generate_binary_watermark(wm_size)
        print("[INFO] Using binary (checkerboard) watermark")
    else:
        watermark = generate_random_watermark(wm_size, seed=seed)
        print(f"[INFO] Using random watermark (seed={seed})")

    wm_image, wm_bits, blocks_used = embed_dct(image, watermark, alpha=alpha)
    img_psnr = psnr(wm_image, image)
    quality_label = 'excellent' if img_psnr > 40 else 'good' if img_psnr > 35 else 'acceptable'
    print(f"[INFO] Watermark embedded in {blocks_used} DCT blocks.")
    print(f"[INFO] PSNR (original vs watermarked) = {img_psnr:.2f} dB ({quality_label})")
    print(f"[INFO] PSNR > 40 dB means the watermark is visually imperceptible.")

    quality_factors = list(range(10, 101, 5))
    ber_list, nc_list = [], []

    print(f"\n{'QF':>4} | {'BER':>8} | {'NC':>8} | {'Extractable?':>12}")
    print("-" * 42)

    for qf in quality_factors:
        compressed = jpeg_compress(wm_image, quality=qf)
        extracted  = extract_dct(image, compressed, wm_size, alpha)
        ber        = bit_error_rate(watermark, extracted)
        nc         = normalized_correlation(watermark, extracted)
        ber_list.append(ber)
        nc_list.append(nc)
        extractable = "YES" if ber < 0.15 else "NO"
        print(f"  {qf:>3} | {ber:>8.4f} | {nc:>8.4f} | {extractable:>12}")

    cannot_extract = [qf for qf, ber in zip(quality_factors, ber_list) if ber >= 0.15]
    can_extract    = [qf for qf, ber in zip(quality_factors, ber_list) if ber < 0.15]

    print()
    if can_extract:
        print(f"[RESULT] Watermark CAN be extracted at QF: {can_extract}")
    if cannot_extract:
        print(f"[RESULT] Watermark CANNOT be extracted at QF: {cannot_extract}")
        print(f"[RESULT] => At QF <= {max(cannot_extract)}, JPEG compression destroys the watermark.")

    out_dir = "watermark_output"
    os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(os.path.join(out_dir, "original.png"), image)
    cv2.imwrite(os.path.join(out_dir, "watermarked.png"), wm_image)
    cv2.imwrite(os.path.join(out_dir, "watermark.png"), (watermark * 255).astype(np.uint8))

    for qf in [10, 30, 50, 70, 90]:
        comp = jpeg_compress(wm_image, quality=qf)
        cv2.imwrite(os.path.join(out_dir, f"compressed_qf{qf}.jpg"), comp)

    _save_plots(quality_factors, ber_list, nc_list, out_dir)
    _save_visual_summary(image, wm_image, watermark, wm_size, alpha, out_dir)

    print(f"\n[INFO] All outputs saved in: {out_dir}/")
    return quality_factors, ber_list, nc_list


# ─────────────────────────────────────────────
# 7. PLOTS & VISUAL SUMMARY
# ─────────────────────────────────────────────

def _save_plots(qf_list, ber_list, nc_list, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(qf_list, ber_list, "r-o", linewidth=2, markersize=5)
    axes[0].axhline(y=0.15, color="k", linestyle="--", label="BER threshold (0.15)")
    axes[0].fill_between(qf_list, ber_list, 0.15,
                         where=[b >= 0.15 for b in ber_list],
                         alpha=0.2, color="red", label="Cannot extract")
    axes[0].fill_between(qf_list, ber_list, 0.15,
                         where=[b < 0.15 for b in ber_list],
                         alpha=0.2, color="green", label="Can extract")
    axes[0].set_xlabel("JPEG Quality Factor (QF)")
    axes[0].set_ylabel("Bit Error Rate (BER)")
    axes[0].set_title("BER vs. JPEG Quality Factor\n(DCT Watermarking)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(qf_list, nc_list, "b-o", linewidth=2, markersize=5)
    axes[1].axhline(y=0.7, color="k", linestyle="--", label="NC threshold (0.7)")
    axes[1].fill_between(qf_list, nc_list, 0.7,
                         where=[n >= 0.7 for n in nc_list],
                         alpha=0.2, color="green", label="Can extract")
    axes[1].set_xlabel("JPEG Quality Factor (QF)")
    axes[1].set_ylabel("Normalized Correlation (NC)")
    axes[1].set_title("NC vs. JPEG Quality Factor\n(DCT Watermarking)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "robustness_plot.png"), dpi=150)
    plt.close()
    print("[INFO] Robustness plot saved.")


def _save_visual_summary(original, wm_image, watermark, wm_size, alpha, out_dir):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    def bgr2rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    axes[0, 0].imshow(bgr2rgb(original));      axes[0, 0].set_title("Original");             axes[0, 0].axis("off")
    axes[0, 1].imshow(bgr2rgb(wm_image));      axes[0, 1].set_title("Watermarked (DCT)");    axes[0, 1].axis("off")
    axes[0, 2].imshow(watermark, cmap="gray"); axes[0, 2].set_title("Watermark Pattern");    axes[0, 2].axis("off")

    diff = cv2.absdiff(wm_image, original).astype(np.float32)
    diff_vis = np.clip(diff * 15, 0, 255).astype(np.uint8)
    axes[0, 3].imshow(bgr2rgb(diff_vis));      axes[0, 3].set_title("Difference x15\n(should look like noise)"); axes[0, 3].axis("off")

    for idx, qf in enumerate([10, 30, 70, 95]):
        compressed = jpeg_compress(wm_image, quality=qf)
        extracted  = extract_dct(original, compressed, wm_size, alpha)
        ber        = bit_error_rate(watermark, extracted)
        status = "Extractable" if ber < 0.15 else "Destroyed"
        axes[1, idx].imshow(extracted, cmap="gray")
        axes[1, idx].set_title(f"Extracted @ QF={qf}\nBER={ber:.3f}  [{status}]")
        axes[1, idx].axis("off")

    plt.suptitle("DCT Watermarking - Invisibility & JPEG Robustness Evaluation", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "visual_summary.png"), dpi=150)
    plt.close()
    print("[INFO] Visual summary saved.")



# ─────────────────────────────────────────────
# 8. ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DCT-Based Digital Watermarking Evaluator")
    parser.add_argument("--image", type=str, default=None,
                        help="Path to your face photo. Omit for synthetic demo.")
    parser.add_argument("--watermark_type", type=str, default="random",
                        choices=["binary", "random"],
                        help="'binary' (checkerboard) or 'random' (PN sequence)")
    parser.add_argument("--alpha", type=int, default=20,
                        help="Embedding strength (default: 20). Higher = more robust, slightly less invisible.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed / secret key for watermark generation")
    args = parser.parse_args()

    if args.image is None or not os.path.exists(str(args.image)):
        print("[INFO] No image provided. Using synthetic demo face...")
        demo_path = "demo_face.png"
        cv2.imwrite(demo_path, create_demo_face())
        image_path = demo_path
    else:
        image_path = args.image

    evaluate(image_path,
             watermark_type=args.watermark_type,
             alpha=args.alpha,
             seed=args.seed)
