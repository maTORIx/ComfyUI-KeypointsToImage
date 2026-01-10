import torch
import numpy as np
import cv2
import json


class KeypointsToImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "keypoints_text": ("STRING", {"multiline": True, "default": ""}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096}),
                "line_thickness": ("INT", {"default": 4, "min": 1, "max": 20}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_image"
    CATEGORY = "CustomNodes/Pose"

    def generate_image(self, keypoints_text, width, height, line_thickness):
        # 1. JSONテキストをパース
        try:
            data = json.loads(keypoints_text)
        except Exception as e:
            print(f"Error parsing JSON: {e}")
            # エラー時は真っ黒な画像を返す
            return (torch.zeros((1, height, width, 3), dtype=torch.float32),)

        # 黒いキャンバスの作成 (RGB)
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        # OpenPoseの標準的な接続（18点モデル）
        # [点A, 点B, 色(R, G, B)]
        LIMBS = [
            [1, 2, [255, 0, 0]],
            [1, 5, [255, 85, 0]],
            [2, 3, [255, 170, 0]],
            [3, 4, [255, 255, 0]],
            [5, 6, [170, 255, 0]],
            [6, 7, [85, 255, 0]],
            [1, 8, [0, 255, 0]],
            [8, 9, [0, 255, 85]],
            [9, 10, [0, 255, 170]],
            [1, 11, [0, 255, 255]],
            [11, 12, [0, 170, 255]],
            [12, 13, [0, 85, 255]],
            [1, 0, [0, 0, 255]],
            [0, 14, [85, 0, 255]],
            [14, 16, [170, 0, 255]],
            [0, 15, [255, 0, 255]],
            [15, 17, [255, 0, 170]],
        ]

        # data['people'] がリストであることを想定
        people = data.get("people", [])

        for person in people:
            pose_keypoints = person.get("pose_keypoints_2d", [])

            # キーポイントを (x, y, confidence) のリストに変換
            points = []
            for i in range(0, len(pose_keypoints), 3):
                points.append(
                    (
                        (
                            int(pose_keypoints[i] * width)
                            if pose_keypoints[i] <= 1.0
                            else int(pose_keypoints[i])
                        ),
                        (
                            int(pose_keypoints[i + 1] * height)
                            if pose_keypoints[i + 1] <= 1.0
                            else int(pose_keypoints[i + 1])
                        ),
                        pose_keypoints[i + 2],
                    )
                )

            # 枝（Limb）の描画
            for limb in LIMBS:
                index_a, index_b, color = limb
                if index_a < len(points) and index_b < len(points):
                    p1, p2 = points[index_a], points[index_b]
                    # confidence > 0 の場合のみ描画
                    if p1[2] > 0 and p2[2] > 0:
                        cv2.line(
                            canvas,
                            (p1[0], p1[1]),
                            (p2[0], p2[1]),
                            color,
                            line_thickness,
                        )

            # 関節（Joint）の描画
            for p in points:
                if p[2] > 0:
                    cv2.circle(
                        canvas, (p[0], p[1]), line_thickness, (255, 255, 255), -1
                    )

        # Numpy (H, W, C) -> Tensor (1, H, W, C) への変換
        canvas_float = canvas.astype(np.float32) / 255.0
        tensor_output = torch.from_numpy(canvas_float).unsqueeze(0)

        return (tensor_output,)


# ComfyUIへの登録用辞書
NODE_CLASS_MAPPINGS = {"KeypointsToImage": KeypointsToImage}

NODE_DISPLAY_NAME_MAPPINGS = {"KeypointsToImage": "Keypoints to OpenPose Image"}
