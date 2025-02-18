import os
import sys
import numpy as np

# 현재 스크립트 위치 기준으로 ucf_option 경로 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "ucf_option")))

# 관련 옵션 가져오기
from ucf_option_VIT_16_merging import args

def merge_features(feature_dir, save_dir, num_segments, segment_handling):
    """
    ViT 프레임별 feature를 num_segments 프레임 단위로 묶어서 평균내어 저장하는 함수.

    Args:
        feature_dir (str): ViT 피처가 저장된 경로 (frame feature)
        save_dir (str): 변환된 segment feature를 저장할 경로
        num_segments (int): 몇 개의 프레임을 하나의 segment로 묶을지 지정
        segment_handling (str): 'padding' 또는 'drop' 처리 방식 지정
    """

    # 클래스별 폴더를 순회
    for class_folder in os.listdir(feature_dir):
        if class_folder == "Train_normal":  
            continue  # "Train_normal" 폴더는 무시
        
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

            save_video_path = os.path.join(save_class_path, f"{video_folder}_visual_feature_all_segments.npy")

            print(f"[INFO] Processing video: {video_folder}...")  # 영상 시작 알림

            # 해당 영상 폴더 내의 feature 파일들을 가져옴
            feature_files = sorted([f for f in os.listdir(video_path) if f.endswith('.npy')], 
                                   key=lambda x: int(x.split('_')[-1].split('.')[0]))  # 숫자 기준 정렬

            if not feature_files:
                print(f"[WARNING] No feature files found in {video_folder}. Skipping...")  # 파일이 없으면 경고
                continue  # 빈 폴더라면 건너뛰기

            try:
                features = [np.load(os.path.join(video_path, f)) for f in feature_files]
                features = np.stack(features)  # (num_frames, feature_dim)
            except Exception as e:
                print(f"[ERROR] Failed to load features for {video_folder}: {e}")  # 오류 처리
                continue  # 오류가 발생한 경우 해당 영상 건너뛰기

            num_frames = features.shape[0]
            feature_dim = features.shape[1]

            # 영상별 모든 segment feature들을 모을 리스트
            all_segment_features = []

            # segment 단위로 feature를 저장
            for start in range(0, num_frames, num_segments):
                end = start + num_segments
                segment_features = features[start:end]

                # 남은 프레임이 num_segments보다 작을 경우 처리 방식
                if segment_features.shape[0] < num_segments:
                    if segment_handling == "drop":
                        continue  # 남은 프레임 무시 (drop 방식)
                    elif segment_handling == "padding":
                        pad_size = num_segments - segment_features.shape[0]
                        padding = np.zeros((pad_size, feature_dim))
                        segment_features = np.vstack([segment_features, padding])

                # 각 segment 내의 feature를 평균내기
                segment_features_mean = np.mean(segment_features, axis=0)

                # segment feature들을 리스트에 추가
                all_segment_features.append(segment_features_mean)

            if all_segment_features:
                # 영상별로 모든 segment features를 저장
                all_segment_features = np.stack(all_segment_features)  # (num_segments, feature_dim)
                np.save(save_video_path, all_segment_features)
                print(f"[INFO] Saved segment features for video: {video_folder}")  # 작업 완료 알림

        print(f"[INFO] Finished processing class: {class_folder}")  # 클래스 처리 완료 알림

if __name__ == "__main__":

    merge_features(
        feature_dir=os.path.join(args.feature_frame_save_path, args.backbone_folder, "visual_features"),
        save_dir=os.path.join(args.feature_segment_save_path, args.backbone_folder),
        num_segments=args.num_segments,
        segment_handling=args.segment_handling
    )