# ensemble_CV_fixed.py
import os
import zipfile
import numpy as np
import nibabel as nib
from pathlib import Path
import tempfile
from tqdm import tqdm
import argparse



N_Zip = 8

def extract_zip(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def get_patient_paths(fold_dirs):
    patient_dict = {}
    for fold_idx, fold_dir in enumerate(fold_dirs):
        for file in os.listdir(fold_dir):
            if file.endswith(".nii.gz") and file.startswith("patient"):
                pid = file.split('.')[0]
                if pid not in patient_dict:
                    patient_dict[pid] = [None] * len(fold_dirs)
                patient_dict[pid][fold_idx] = os.path.join(fold_dir, file)
    return patient_dict

def load_nii_gz(path):
    img = nib.load(path)
    data = img.get_fdata()
    return data, img.affine, img.header

def save_nii_gz(data, affine, header, save_path):
    img = nib.Nifti1Image(data, affine, header=header)
    nib.save(img, save_path)

def is_probability_map(arr, num_classes=4):
    """判斷是否為概率圖（float + 接近 0~1 + 通道數 == num_classes）"""
    if not np.issubdtype(arr.dtype, np.floating):
        return False
    if arr.ndim != 4:
        return False
    if arr.shape[0] != num_classes:
        return False
    # 檢查值是否在 [0,1] 且每 voxel 總和 ≈1
    sums = arr.sum(axis=0)
    return np.allclose(sums, 1.0, atol=1e-2)

def mask_to_prob(mask, num_classes):
    """整數 mask → one-hot 概率圖 (C, H, W, D)"""
    mask = mask.astype(np.int64)
    prob = np.eye(num_classes)[mask]           # (H, W, D, C)
    prob = np.moveaxis(prob, -1, 0)            # (C, H, W, D)
    return prob

def ensemble_volume(prob_maps, weights):
    weights = np.array(weights) / np.sum(weights)
    fused_prob = sum(w * p for w, p in zip(weights, prob_maps))
    fused_mask = np.argmax(fused_prob, axis=0).astype(np.uint8)
    return fused_mask

def main():
    parser = argparse.ArgumentParser(description="K-Fold Ensemble for 3D CT Segmentation (Fixed)")
    parser.add_argument("--zip_files", nargs=N_Zip, required=True,
                        help=f"{N_Zip} 個 predict-main_xxx.zip 檔案")
    parser.add_argument("--weights", type=float, nargs=N_Zip, default=[1,1,1,1,1],
                        help=f"{N_Zip} 個 Fold 權重")
    parser.add_argument("--output_zip", type=str, default="ensemble_final.zip")
    parser.add_argument("--num_classes", type=int, default=4,
                        help="分割類別數（預設 4：背景+LV+RV+MYO）")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        fold_dirs = []

        print(f"正在解壓 {N_Zip} 個 zip 檔...")
        for i, zip_path in enumerate(args.zip_files):
            fold_dir = tmp_path / f"fold_{i}"
            fold_dir.mkdir(exist_ok=True)
            extract_zip(zip_path, fold_dir)
            fold_dirs.append(fold_dir)
            print(f"  Fold {i+1}: {Path(zip_path).name} → {fold_dir.name}")

        patient_dict = get_patient_paths(fold_dirs)
        patient_ids = sorted(patient_dict.keys())
        print(f"找到 {len(patient_ids)} 位病人，開始融合...")

        output_dir = tmp_path / "ensemble_output"
        output_dir.mkdir()

        for pid in tqdm(patient_ids, desc="Ensemble"):
            paths = patient_dict[pid]
            if None in paths:
                print(f"警告: {pid} 缺少 Fold，跳過")
                continue

            # 載入第一個檔案判斷類型
            first_data, affine, header = load_nii_gz(paths[0])
            is_prob = is_probability_map(first_data, args.num_classes)

            prob_maps = []
            for path in paths:
                data, aff, hdr = load_nii_gz(path)
                if affine is None:
                    affine, header = aff, hdr

                if is_prob:
                    # 已是 (C, H, W, D) 概率圖
                    prob_maps.append(data)
                else:
                    # 是整數 mask → 轉 one-hot
                    prob_maps.append(mask_to_prob(data, args.num_classes))

            # 融合
            fused_mask = ensemble_volume(prob_maps, args.weights)
            save_path = output_dir / f"{pid}.nii.gz"
            save_nii_gz(fused_mask, affine, header, save_path)

        # 壓縮輸出
        print(f"正在產生 {args.output_zip}...")
        with zipfile.ZipFile(args.output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file in output_dir.glob("*.nii.gz"):
                zipf.write(file, arcname=file.name)

    print(f"Ensemble 完成！輸出：{args.output_zip}")

if __name__ == "__main__":
    main()
