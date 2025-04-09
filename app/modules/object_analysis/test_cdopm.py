import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable as V
import torchvision.models as models
from torch.nn import functional as F
import os
import json
import numpy as np
from tqdm import tqdm  # Progress bar

from model import Object_CDOPM_ResNet18, Object_CDOPM_ResNet50, Object_IOM, Fusion_CDOPM_ResNet50, Fusion_CIOM, Fusion_CDOPM_ResNet18
from arguments import ArgumentsParse
from dataset import DatasetSelection


# Global Variables
global args
best_prec1 = 0

# Function to Extract Features
def my_forward(model, x):
    mo = nn.Sequential(*list(model.children())[:-1])  # Remove last layer
    feature = mo(x)
    feature = feature.view(x.size(0), -1)
    return feature


# Main Execution
if __name__ == '__main__':
    args = ArgumentsParse.test_argsParser()

    # Dataset Selection
    dataset_selection = DatasetSelection(args.dataset)
    one_hot, data_dir, classes = dataset_selection.datasetSelection()
    discriminative_matrix = dataset_selection.discriminative_matrix_estimation()

    print(f"\n🚀 Using dataset: {args.dataset}")
    print(f"🛠 Model will classify {args.num_classes} classes\n")

    # Model Selection
    if args.om_type == 'ciom_resnet50':
        object_idt = Object_IOM()
        classifier = Fusion_CIOM(args.num_classes)
    elif args.om_type == 'cdopm_resnet18':
        object_idt = Object_CDOPM_ResNet18()
        classifier = Fusion_CDOPM_ResNet18(args.num_classes)
    elif args.om_type == 'cdopm_resnet50':
        object_idt = Object_CDOPM_ResNet50()
        classifier = Fusion_CDOPM_ResNet50(args.num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    object_idt.to(device)
    classifier.to(device)

    # Load Pretrained Model
    model_path = args.pretrained
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file '{model_path}' not found.")

    arch = 'resnet50' if args.om_type == 'cdopm_resnet50' else 'resnet18'
    model = models.__dict__[arch](pretrained=False)  # Initialize without pretrained weights
    checkpoint = torch.load(model_path, map_location=device)

    print(f"✅ Loaded model from {model_path}. Best Accuracy: {checkpoint.get('best_prec1', 'N/A')}\n")
    print(f"📌 Checkpoint keys: {checkpoint.keys()}\n")

    # Load Model Weights Based on Available Keys
    if 'model_state_dict' in checkpoint:
        model_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
    elif 'state_dict' in checkpoint:
        model_state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
    else:
        raise KeyError("❌ Neither 'model_state_dict' nor 'state_dict' found in the checkpoint file.")

    # Remove fc layer weights if mismatch
    model_state_dict.pop("fc.weight", None)
    model_state_dict.pop("fc.bias", None)

    # Load the modified state_dict
    model.load_state_dict(model_state_dict, strict=False)

    # Replace the fc layer with the correct number of output classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, args.num_classes)

    # Load Object & Classifier Weights
    obj_state_dict = checkpoint.get('obj_state_dict', {})
    classifier_state_dict = checkpoint.get('classifier_state_dict', {})

    object_idt.load_state_dict(obj_state_dict, strict=False)
    classifier.load_state_dict(classifier_state_dict, strict=False)

    model.to(device)
    model.eval()
    object_idt.eval()
    classifier.eval()

    # Load JSON Data Instead of Images
    json_file = f"F:/Anfa Backup/E/IBA_8th_sm/FYP/brom_14/borm/object_information/150obj_7classes_SUN.json"
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"❌ JSON file '{json_file}' not found. Ensure the dataset is available.")

    with open(json_file, 'r') as f:
        json_data = json.load(f)

    # Load classes from categories_Places365_7.txt
    class_file = "F:/Anfa Backup/E/IBA_8th_sm/FYP/brom_14/borm/object_information/categories_Places365_7.txt"
    with open(class_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

    # Classes to exclude from testing
    excluded_classes = {"playroom", "garage-indoor"}

    # Dictionary to store per-class accuracy
    class_correct = {class_name: 0 for class_name in classes}
    class_total = {class_name: 0 for class_name in classes}

    # Initialize progress bar
    progress_bar = tqdm(json_data.items(), desc="🔄 Processing Images", unit="image")

    for img_path, label in progress_bar:
        # Extract class name from one-hot label
        class_idx = label.index(1)  # Find the index where the one-hot vector has '1'
        class_name = classes[class_idx]

        # Skip excluded classes
        if class_name.lower() in excluded_classes:
            print(f"🛑 Skipping {class_name} image: {img_path}")
            continue  # Skip this sample

        class_total[class_name] += 1  # Count total images per class

        # Convert object vector into NumPy array
        row = np.array(label).reshape(1, -1)  # Convert to 2D array
        object_pair_matrix = np.dot(row.T, row)  # Generate object-pair matrix
        obj_hot_vector = object_pair_matrix.reshape(22500).tolist()  # Flatten to 22500 dimensions

        # Convert to tensor
        t = torch.autograd.Variable(torch.FloatTensor(obj_hot_vector)).to(device)

        # Run through model
        output_idt = object_idt(t).unsqueeze(0)
        logit = classifier(output_idt, output_idt)
        h_x = F.softmax(logit, 1).data.squeeze()
        probs, idx = h_x.sort(0, True)

        predicted_class = classes[idx[0]]  # Get predicted class name

        # Debugging: Print actual vs predicted class
        print(f"🧐 Actual: {class_name}, Predicted: {predicted_class}")

        # Check if prediction is correct
        if predicted_class == class_name:
            class_correct[class_name] += 1  # Track correct predictions

        # Update progress bar with accuracy
        overall_correct = sum(class_correct.values())
        overall_total = sum(class_total.values())
        current_accuracy = (100 * overall_correct / overall_total) if overall_total > 0 else 0
        progress_bar.set_postfix(accuracy=f"{current_accuracy:.2f}%")

    # Compute per-class accuracy
    print("\n📊 **Per-Class Accuracy:**")
    for class_name in classes:
        if class_total[class_name] > 0:
            class_acc = 100 * class_correct[class_name] / class_total[class_name]
            print(f"✅ {class_name}: {class_acc:.2f}% ({class_correct[class_name]}/{class_total[class_name]})")
        else:
            print(f"❌ {class_name}: No samples found.")

    # Compute final accuracy (Excluding "playroom" & "garage indoor")
    avg_accuracy = (100 * sum(class_correct.values()) / sum(class_total.values())) if sum(class_total.values()) > 0 else 0
    print(f"\n🎯 Final Average Test Accuracy: {avg_accuracy:.2f}%\n")
