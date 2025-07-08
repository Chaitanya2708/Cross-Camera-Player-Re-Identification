import argparse
import os
import cv2
import numpy as np
from torchvision import models, transforms
import torch
from torch import nn
from tqdm import tqdm
import csv

def extract_features_from_folder(folder, model, transform, device):
    features = []
    filenames = []

    for file in tqdm(os.listdir(folder), desc=f"Extracting features from {folder}"):
        if file.endswith(('.jpg', '.png')):
            image_path = os.path.join(folder, file)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                feature = model(image).squeeze().cpu().numpy()
                features.append(feature)
                filenames.append(file)
    return np.array(features), filenames

def match_players(features1, features2, filenames1, filenames2, output_dir, top_k=1):
    os.makedirs(output_dir, exist_ok=True)
    matched_pairs = []

    for idx1, feature1 in enumerate(features1):
        dists = np.linalg.norm(features2 - feature1, axis=1)
        topk_indices = np.argsort(dists)[:top_k]

        for idx2 in topk_indices:
            matched_pairs.append((filenames1[idx1], filenames2[idx2], dists[idx2]))

            # Save side-by-side image for matched pair
            img1 = cv2.imread(os.path.join(args.det1, filenames1[idx1]))
            img2 = cv2.imread(os.path.join(args.det2, filenames2[idx2]))
            if img1 is not None and img2 is not None:
                concat = cv2.hconcat([img1, img2])
                match_name = f"{filenames1[idx1].split('.')[0]}_{filenames2[idx2].split('.')[0]}.jpg"
                cv2.imwrite(os.path.join(output_dir, match_name), concat)

    return matched_pairs

def save_matches_csv(matches, output_dir):
    csv_path = os.path.join(output_dir, "matched_players.csv")
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['det1_image', 'det2_image', 'distance'])
        for row in matches:
            writer.writerow(row)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--det1', required=True, help="Path to first detection folder")
    parser.add_argument('--det2', required=True, help="Path to second detection folder")
    parser.add_argument('--output_dir', required=True, help="Path to save output matches")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Use ResNet18 without classification head
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    resnet.fc = nn.Identity()
    resnet.eval().to(device)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    print(f"[INFO] Extracting features from {args.det1} and {args.det2}...")
    features1, filenames1 = extract_features_from_folder(args.det1, resnet, transform, device)
    features2, filenames2 = extract_features_from_folder(args.det2, resnet, transform, device)

    print("[INFO] Matching players...")
    matched_pairs = match_players(features1, features2, filenames1, filenames2, args.output_dir)

    print(f"[INFO] Found {len(matched_pairs)} matched player pairs.")
    save_matches_csv(matched_pairs, args.output_dir)
    print(f"[INFO] Matching complete. Results saved to '{args.output_dir}'")
