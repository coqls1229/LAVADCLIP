import argparse
import os

parser = argparse.ArgumentParser(description='WSVAD - MIL Loss with Transformer')

# 학습 하이퍼파라미터
parser.add_argument('--seed', default=234, type=int)
parser.add_argument('--embed-dim', default=512, type=int)
parser.add_argument('--visual-length', default=256, type=int)
parser.add_argument('--visual-width', default=512, type=int)
parser.add_argument('--num-segments', default=16, type=int)
parser.add_argument('--batch-size', default=32, type=int)
parser.add_argument('--lr', default=2e-5, type=float)
parser.add_argument('--max-epoch', default=15, type=int)
parser.add_argument('--save-interval', default=5, type=int)

# Positional Encoding 최대 길이 설정
parser.add_argument('--max-len', default=10000, type=int)  # 🚀 10000으로 설정하여 긴 시퀀스 대응

# 데이터셋 및 모델 경로
base_path = "/data/coqls1229/repos/VAD"

parser.add_argument('--video-path', default=os.path.join(base_path, "data/Videos"), type=str)
parser.add_argument('--frame-path', default=os.path.join(base_path, "/data/jhlee39/workspace/repos/Data/ucf_frame/Training-Normal-Videos"), type=str)

parser.add_argument('--backbone', default='ViT-B/16', type=str, choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'VideoMAE V2', 'InternVideo', 'MVD'])
parser.add_argument('--backbone-folder', default='ViT-B-16', type=str, choices=['ViT-B-32', 'ViT-B-16', 'ViT-L-14', 'VideoMAE V2', 'InternVideo', 'MVD'])
parser.add_argument('--segment-handling', default='padding', type=str, choices=['padding', 'drop'])

parser.add_argument('--feature-frame-save-path', default=os.path.join(base_path, "data/CLIP_feature/frame_feature"), type=str)
parser.add_argument('--feature-segment-save-path', default=os.path.join(base_path, "data/CLIP_feature/segment_feature"), type=str)
parser.add_argument('--feature-LLM-save-path', default=os.path.join(base_path, "data/CLIP_LLM_feature"), type=str)

parser.add_argument('--train-list', default=os.path.join(base_path, "list/Anomaly_Train.txt"), type=str)
parser.add_argument('--test-list', default=os.path.join(base_path, "list/Anomaly_Test.txt"), type=str)
parser.add_argument('--temporal-annotation', default=os.path.join(base_path, "list/Temporal_Anomaly_Annotation.txt"), type=str)
parser.add_argument('--visual-feature-path', default=os.path.join(base_path, "data/features/visual_features"), type=str)
parser.add_argument('--text-feature-path', default=os.path.join(base_path, "data/features/text_features"), type=str)
parser.add_argument('--model-save-path', default=os.path.join(base_path, "model"), type=str)

args = parser.parse_args()

