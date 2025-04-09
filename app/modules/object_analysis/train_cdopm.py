import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import time
from tqdm import tqdm
from dataset import DatasetSelection
from model import Object_CDOPM_ResNet18, Object_CDOPM_ResNet50, Fusion_CDOPM_ResNet50, Fusion_CDOPM_ResNet18
from arguments import ArgumentsParse

# ✅ Global Variables
best_prec1 = 0

def main():
    global args, best_prec1
    args = ArgumentsParse.argsParser()  # ✅ Fixed Argument Parsing
    print(args)

    # ✅ Load dataset (Using JSON, No Images)
    dataset_selection = DatasetSelection(args.dataset)
    train_data, data_dir, classes = dataset_selection.datasetSelection()  # Correct unpacking
    val_data, _, _ = dataset_selection.datasetSelection()  # Unpack only necessary values
    discriminative_matrix = dataset_selection.discriminative_matrix_estimation()

    print(f"✅ Training on {len(train_data)} samples | Validating on {len(val_data)} samples")

    # ✅ Model Selection
    if args.om_type == 'cdopm_resnet18':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        object_idt = Object_CDOPM_ResNet18()
        classifier = Fusion_CDOPM_ResNet18(args.num_classes)
        model_name = "cdopm_resnet18_7.pth.tar"
    elif args.om_type == 'cdopm_resnet50':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        object_idt = Object_CDOPM_ResNet50()
        classifier = Fusion_CDOPM_ResNet50(args.num_classes)
        model_name = "cdopm_resnet50_7.pth.tar"
    else:
        raise ValueError(f"Unknown model type '{args.om_type}'.")

    # Modify the final layer to match the number of classes (7)
    num_features = model.fc.in_features  # Get the number of input features to the final layer
    model.fc = nn.Linear(num_features, args.num_classes)  # Change the output to match the number of classes

    model, object_idt, classifier = model.cuda(), object_idt.cuda(), classifier.cuda()

    # ✅ Define loss function & optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(
        list(object_idt.parameters()) + list(classifier.parameters()),
        args.lr, momentum=args.momentum, weight_decay=0.0005  # Added weight decay for L2 regularization
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # ✅ Checkpoint paths
    latest_model_path = f"models/{model_name}"
    best_model_path = f"models/{model_name.replace('.pth.tar', '_best.pth.tar')}"

    # Training loop
    for epoch in range(args.start_epoch, args.epochs):
        scheduler.step()  # Step the scheduler to adjust learning rate

        print(f"\n🔹 Epoch [{epoch + 1}/{args.epochs}] Training...")
        train_loss = train(train_data, model, object_idt, classifier, criterion, optimizer, epoch, discriminative_matrix)

        print(f"\n✅ Epoch [{epoch + 1}/{args.epochs}] Validating...")
        prec1, predictions, targets = validate(val_data, model, object_idt, classifier, criterion, discriminative_matrix)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'obj_state_dict': object_idt.state_dict(),
            'classifier_state_dict': classifier.state_dict(),
            'best_prec1': best_prec1,
        }, is_best, latest_model_path, best_model_path)

        print(f"\n📊 Epoch {epoch + 1} Summary: Train Loss = {train_loss:.4f} | Val Accuracy = {prec1:.2f}% | Best Accuracy = {best_prec1:.2f}%\n")

def train(train_data, model, object_idt, classifier, criterion, optimizer, epoch, discriminative_matrix):
    model.train()
    object_idt.train()
    classifier.train()

    total_loss = 0
    num_samples = len(train_data[0])  # Correctly use the number of images
    start_time = time.time()

    for i, (img_path, obj_vector) in enumerate(tqdm(zip(train_data[0], train_data[1]), desc="🏋️ Training", unit="batch")):  # Use zip
        # Debugging: Check type and contents of obj_vector
        print(f"Type of obj_vector: {type(obj_vector)}")
        print(f"Object vector: {obj_vector}")

        # Ensure obj_vector is a list of numeric values and not a string
        if isinstance(obj_vector, list) and all(isinstance(x, (int, float)) for x in obj_vector):
            obj_tensor = process_object_vector(obj_vector, discriminative_matrix)
        else:
            print(f"Skipping invalid object vector: {obj_vector}")
            continue

        # Simulated target (extract actual class from img_path)
        target_class = get_class_from_path(img_path)
        target = torch.tensor([target_class], dtype=torch.long).cuda()

        output_idt = object_idt(obj_tensor)
        output = classifier(output_idt, output_idt)  # Pass same tensor twice (since no image features)

        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    epoch_time = time.time() - start_time
    print(f"  ⏳ Epoch Training Time: {epoch_time:.2f}s")

    return total_loss / num_samples

def validate(val_data, model, object_idt, classifier, criterion, discriminative_matrix):
    model.eval()
    object_idt.eval()
    classifier.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for img_path, obj_vector in tqdm(zip(val_data[0], val_data[1]), desc="🔍 Validating", unit="batch"):  # Use zip
            obj_tensor = process_object_vector(obj_vector, discriminative_matrix)

            # Simulated target (extract actual class from img_path)
            target_class = get_class_from_path(img_path)
            target = torch.tensor([target_class], dtype=torch.long).cuda()

            output_idt = object_idt(obj_tensor)
            output = classifier(output_idt, output_idt)

            _, predicted = torch.max(output, 1)
            total += 1
            correct += (predicted == target).sum().item()

    return 100 * correct / total

def process_object_vector(obj_vector, discriminative_matrix):
    """ Convert object vector to tensor with discriminative matrix applied. """
    try:
        # Ensure obj_vector is a numeric type and reshape it
        print(f"Object vector: {obj_vector}")  # Debugging
        row = np.array(obj_vector, dtype=np.float32).reshape(1, -1)
    except ValueError as e:
        print(f"ValueError processing obj_vector: {e}")
        raise

    # Ensure discriminative_matrix is of the correct type (float32)
    discriminative_matrix = discriminative_matrix.astype(np.float32)

    # Perform dot product and apply discriminative matrix
    object_discriminative_matrix = np.dot(row.T, row) * discriminative_matrix

    # Flatten the matrix and convert it to a tensor
    obj_tensor = torch.tensor(object_discriminative_matrix.flatten(), dtype=torch.float32).unsqueeze(0).cuda()
    return obj_tensor

def get_class_from_path(img_path):
    """ Extracts class label from image path. """
    # Debugging: Print the img_path to check its structure
    print(f"Extracting class from path: {img_path}")
    class_map = {
        "bathroom": 0,
        "bedroom": 1,
        "corridor": 2,
        "dining_room": 3,
        "kitchen": 4,
        "living_room": 5,
        "office": 6
    }
    try:
        class_name = img_path.split("/")[-2]  # Extract folder name from the image path
    except IndexError as e:
        print(f"IndexError extracting class from path: {e}")
        class_name = "unknown"  # Default to "unknown" if the split fails

    return class_map.get(class_name, 0)  # Default to 0 if class not found

def save_checkpoint(state, is_best, latest_model_path, best_model_path):
    os.makedirs(os.path.dirname(latest_model_path), exist_ok=True)
    torch.save(state, latest_model_path)
    if is_best:
        torch.save(state, best_model_path)

if __name__ == '__main__':
    main()
