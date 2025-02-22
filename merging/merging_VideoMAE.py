import os
import sys
import numpy as np
import re

# 현재 스크립트 위치 기준으로 ucf_option 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "ucf_option")))

# 관련 옵션 가져오기
from ucf_option_VideoMAEv2_merging import args

def extract_number_range(filename):
    """ 
    파일명에서 숫자 범위를 추출하는 함수.
    예: "Arrest001_x264_visual_feature_1-16.npy" → (1, 16)
    """
    match = re.search(r"(\d+)-(\d+)", filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return (float('inf'), float('inf'))  # 잘못된 경우 뒤로 보내기

def merge_all_features(feature_dir, save_dir):
    """
    이미 16프레임 단위로 병합된 ViT 피처들을 하나로 합쳐서 저장하는 함수.

    Args:
        feature_dir (str): ViT 피처가 저장된 경로 (segment feature)
        save_dir (str): 병합된 feature를 저장할 경로
    """

    # 클래스별 폴더를 순회
    for class_folder in os.listdir(feature_dir):
        if class_folder == "Train_normal":  
            continue  # "Train_normal" 폴더는 무시 (Train_normal은 여러 계정에 나눠져 있어 따로 처리하려고 예외)
        
        class_path = os.path.join(feature_dir, class_folder)
        if not os.path.isdir(class_path):
            continue

        save_class_path = os.path.join(save_dir, class_folder)
        os.makedirs(save_class_path, exist_ok=True)

        print(f"[INFO] Processing class: {class_folder}...")  # 클래스 시작 알림

        # 영상별 폴더를 순회
        for video_folder in os.listdir(class_path):
            video_path = os.path.join(class_path, video_folder)
            if not os.path.isdir(video_path):
                continue  # 폴더가 아니면 무시

            save_video_path = os.path.join(save_class_path, f"{video_folder}_visual_feature.npy")

            print(f"[INFO] Processing video: {video_folder}...")  # 영상 시작 알림

            # 해당 영상 폴더 내의 feature 파일들을 가져옴 (이제 1-16, 17-32 같은 파일명을 고려)
            feature_files = sorted(
                [f for f in os.listdir(video_path) if f.endswith('.npy')],
                key=lambda x: extract_number_range(x)  # 숫자 범위를 기준으로 정렬
            )

            if not feature_files:
                print(f"[WARNING] No feature files found in {video_folder}. Skipping...")  # 파일이 없으면 경고
                continue  # 빈 폴더라면 건너뛰기

            try:
                # 모든 segment feature 로드 후 병합
                features = [np.load(os.path.join(video_path, f)) for f in feature_files]
                merged_features = np.vstack(features)  # (total_frames, feature_dim)
            except Exception as e:
                print(f"[ERROR] Failed to load features for {video_folder}: {e}")  # 오류 처리
                continue  # 오류가 발생한 경우 해당 영상 건너뛰기

            # 병합된 feature 저장
            np.save(save_video_path, merged_features)
            print(f"[INFO] Saved merged features for video: {video_folder}")  # 작업 완료 알림

        print(f"[INFO] Finished processing class: {class_folder}")  # 클래스 처리 완료 알림

if __name__ == "__main__":

    merge_all_features(
        feature_dir=os.path.join(args.feature_frame_save_path, args.backbone_folder, "visual_features"),
        save_dir=os.path.join(args.feature_segment_save_path, "VideoMAE")
    )