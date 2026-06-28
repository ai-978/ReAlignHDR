import argparse
import os
import glob
import shutil
import numpy as np
import torch
from PIL import Image
from sam3.model_builder import build_sam3_video_predictor


# ================= 辅助函数：生成随机调色板 (用于PNG预览) =================
def get_palette(num_cls=256):
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    palette[0:3] = [0, 0, 0] 
    return palette

def convert_and_prepare_frames(src_dir, temp_dir):
    """TIF -> TIFF 转换与排序"""
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    # 支持 tif 和 tiff
    tif_files = sorted(
        glob.glob(os.path.join(src_dir, "*.tif")) + 
        glob.glob(os.path.join(src_dir, "*.tiff"))
    )

    if not tif_files:
        return []

    print(f"  [预处理] 转换 {len(tif_files)} 帧...")
    prepared_files_map = [] 

    for idx, file_path in enumerate(tif_files):
        try:
            img = Image.open(file_path).convert("RGB")
            temp_filename = f"{idx:05d}.tiff"
            temp_path = os.path.join(temp_dir, temp_filename)
            img.save(temp_path, format='TIFF', compression='tiff_lzw')
            original_name = os.path.basename(file_path)
            prepared_files_map.append(original_name)
        except Exception as e:
            print(f"    转换失败: {e}")

    return prepared_files_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", default="/home/waas/buchong", help="输入根目录 (存放tif)")
    parser.add_argument("--scene_dir", default=None, help="单个场景目录；设置后只处理该场景")
    parser.add_argument("--output_root", default="/home/waas/output_bat", help="输出根目录 (存放预览png)")
    # 用逗号分割的多个文本类别，比如 "person, book"
    parser.add_argument(
        "--prompt",
        default="person, book",
        help="要依次分割的类别，逗号分隔，例如: 'person, book'"
    )
    args = parser.parse_args()

    # 解析为列表并去掉空格
    target_prompts = [p.strip() for p in args.prompt.split(",") if p.strip()]

    print("将依次分割的类别：", target_prompts)

    print("正在初始化 SAM3 Video Predictor...")
    gpus = [0] if torch.cuda.is_available() else []
    predictor = build_sam3_video_predictor(gpus_to_use=gpus)
    
    color_palette = get_palette(256)

    if args.scene_dir:
        scene_path = os.path.abspath(args.scene_dir)
        if not os.path.isdir(scene_path):
            raise FileNotFoundError(f"场景目录不存在: {scene_path}")
        scenes = [(os.path.basename(os.path.normpath(scene_path)), scene_path)]
    else:
        scene_dirs = sorted([
            d for d in os.listdir(args.input_root)
            if os.path.isdir(os.path.join(args.input_root, d))
        ])
        scenes = [(scene_name, os.path.join(args.input_root, scene_name)) for scene_name in scene_dirs]

    for scene_name, scene_input_path in scenes:
        print(f"\n=== 处理场景: {scene_name} ===")
        
        # 定义路径
        scene_output_path = os.path.join(args.output_root, scene_name) # png预览目录
        temp_frames_dir = os.path.join(args.output_root, "temp_frames_cache", scene_name)
        
        # 1. 转换图片
        original_filenames = convert_and_prepare_frames(scene_input_path, temp_frames_dir)
        if not original_filenames:
            print("  无tif文件，跳过。")
            continue
        
        num_frames = len(original_filenames)
        mid_frame_idx = num_frames // 2

        # ✅ 新增：读取第一帧的尺寸，用来生成全0图
        first_frame_path = os.path.join(temp_frames_dir, f"{0:05d}.tiff")
        with Image.open(first_frame_path) as im:
            first_H, first_W = im.height, im.width

        # 用于汇总不同类别的分割结果：
        # key: frame_index, value: (H, W) uint8 的 0/1 二值图
        # 0 表示背景，1 表示任意类别、任意实例的前景
        combined_binary_masks = {}

        try:
            # 依次对 person / book / phone 等类别进行分割
            for cat_idx, text_prompt in enumerate(target_prompts):
                print(f"\n  >>> 类别 {cat_idx+1}/{len(target_prompts)}: '{text_prompt}'")

                # 为当前类别启动一个新的 session
                response = predictor.handle_request(
                    dict(type="start_session", resource_path=temp_frames_dir)
                )
                sid = response["session_id"]

                print(f"  应用提示词 '{text_prompt}' (Anchor Frame: {mid_frame_idx})")
                predictor.handle_request(
                    dict(
                        type="add_prompt", 
                        session_id=sid, 
                        frame_index=mid_frame_idx, 
                        text=text_prompt
                    )
                )

                print("  正在进行视频传播并收集该类别的实例...")

                for msg in predictor.handle_stream_request(
                    dict(
                        type="propagate_in_video",
                        session_id=sid,
                        propagation_direction="both",
                        start_frame_index=mid_frame_idx,
                        max_frame_num_to_track=None,
                    )
                ):
                    fi = msg["frame_index"]
                    out = msg["outputs"]
                    
                    masks = out.get("out_binary_masks")  # (N, H, W)
                    obj_ids = out.get("out_obj_ids")     # 长度 N，对实例编号

                    if masks is None or masks.size == 0:
                        continue

                    # 二值化（>0 为前景）
                    masks_bin = (masks > 0)

                    N, H, W = masks_bin.shape

                    # 初始化该帧的汇总二值图
                    if fi not in combined_binary_masks:
                        combined_binary_masks[fi] = np.zeros((H, W), dtype=np.uint8)

                    if not obj_ids:
                        obj_ids = list(range(1, N + 1))

                    # 对当前类别的每个实例，统一并入前景=1
                    for i in range(N):
                        mask_layer = masks_bin[i]
                        # 将该实例的前景位置写入到总二值图中
                        combined_binary_masks[fi][mask_layer > 0] = 1

                # 关闭当前类别的 session
                try:
                    predictor.handle_request(dict(type="close_session", session_id=sid))
                except:
                    predictor.handle_request(dict(type="end_session", session_id=sid))

            # ===== 所有类别都跑完，此时 combined_binary_masks 里已经包含了所有前景 =====
            # 确保输出目录存在
            os.makedirs(scene_output_path, exist_ok=True)

            print("\n  汇总并保存每一帧的 0/1 mask 结果（所有类别与实例合并为前景）...")
            for fi in range(num_frames):
                if fi not in combined_binary_masks:
                    # 没有任何前景，则是全0图
                    # 这里可以用第一帧的尺寸来创建，或者跳过保存
                    # 为简单起见，如果完全没有结果就跳过这帧
                    combined_binary_masks[fi] = np.zeros((first_H, first_W), dtype=np.uint8)

                binary_mask_array = combined_binary_masks[fi]  # (H, W), 值只有 0/1

                orig_name = original_filenames[fi]
                name_no_ext = os.path.splitext(orig_name)[0]

                # 1. 保存 .npy 到原始输入目录 (Source Directory)
                npy_name = f"{name_no_ext}.npy"
                npy_path = os.path.join(scene_input_path, npy_name)
                np.save(npy_path, binary_mask_array)

                # 2. 保存 .png 到输出预览目录 (Output Directory)
                png_name = f"{name_no_ext}_mask.png"
                png_path = os.path.join(scene_output_path, png_name)

                out_img = Image.fromarray(binary_mask_array, mode='P')
                out_img.putpalette(color_palette)
                out_img.save(png_path)

                print(f"    帧 {fi}: 合并后的 0/1 mask 已保存 -> {npy_path}")

        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            if os.path.exists(temp_frames_dir):
                shutil.rmtree(temp_frames_dir)

    # 清理
    temp_root = os.path.join(args.output_root, "temp_frames_cache")
    if os.path.exists(temp_root):
        shutil.rmtree(temp_root)
    print("\n程序执行完毕。")

if __name__ == "__main__":
    main()
