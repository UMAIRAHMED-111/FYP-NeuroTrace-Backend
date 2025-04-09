import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os
from ultralytics import YOLO  # Import YOLO for object detection
from model import Object_CDOPM_ResNet18, Object_CDOPM_ResNet50, Fusion_CDOPM_ResNet50, Fusion_CDOPM_ResNet18
from arguments import ArgumentsParse


class Inference:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.yolo_model = YOLO("yolov8n.pt")  # Initialize YOLO model here

    def load_classes(self, classes_file):
        """Loads class names from a text file."""
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes

    def load_model(self, model_type, model_path):
        """Loads the model from the checkpoint."""
        print(f"Loading model {model_type} from {model_path}...")

        # Use pretrained model for feature extraction (ResNet18)
        if model_type == 'cdopm_resnet50':
            object_idt = Object_CDOPM_ResNet50().to(self.device)
            classifier = Fusion_CDOPM_ResNet50(365).to(self.device)  # Updated class number to 365
            model = models.resnet50(num_classes=365).to(self.device)  # Pretrained on 365 classes
        elif model_type == 'cdopm_resnet18':
            object_idt = Object_CDOPM_ResNet18().to(self.device)
            classifier = Fusion_CDOPM_ResNet18(365).to(self.device)  # Updated class number to 365
            model = models.resnet18(num_classes=365).to(self.device)  # Pretrained on 365 classes
        else:
            raise ValueError("Invalid model type. Choose 'cdopm_resnet50' or 'cdopm_resnet18'.")

        checkpoint = torch.load(model_path, map_location=self.device)

        # Load weights for the layers matching the model (except the fc layer)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()})

        # Replace the fully connected (fc) layer with a new one matching the 365 classes (for scene classification)
        num_features = model.fc.in_features
        model.fc = torch.nn.Linear(num_features, 365).to(self.device)  # Modify for 365 output classes (indoor scenes)

        # Load other layers' state dicts
        object_idt.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['obj_state_dict'].items()})
        classifier.load_state_dict(
            {k.replace('module.', ''): v for k, v in checkpoint['classifier_state_dict'].items()})

        model.eval()
        object_idt.eval()
        classifier.eval()
        return model, object_idt, classifier

    def detect_objects(self, image_path):
        """Runs YOLOv8 on the image to detect objects."""
        results = self.yolo_model(image_path)  # Run YOLO on the image
        detected_objects = set()

        for result in results:
            for box in result.boxes:
                obj_id = int(box.cls.item())
                detected_objects.add(self.yolo_model.names[obj_id])  # Get object name

        return list(detected_objects)

    def predict(self, model, classes, image_path):
        """Runs inference on a single image and detects objects."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # **Step 1: Detect Objects in Image using YOLO**
        detected_objects = self.detect_objects(image_path)
        print(f"Detected Objects: {detected_objects}")

        # **Step 2: Process Image for Scene Recognition**
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        input_tensor = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # **Step 3: Scene Recognition**
            output_conv = model(input_tensor)
            h_x = F.softmax(output_conv, dim=1).squeeze()
            probs, idx = h_x.sort(0, True)

            # Get the predicted label using the class index
            predicted_label = classes[idx[0].item()]

        print(f"Predicted Scene: {predicted_label} (Confidence: {probs[0].item():.4f})")
        return predicted_label, probs[0].item(), detected_objects


if __name__ == '__main__':
    # File paths
    model_path = 'F:/Anfa Backup/E/IBA_8th_sm/FYP/brom_14/borm/models/cdopm_resnet18_best.pth.tar'
    classes_file = 'object_information/categories_Places365_365.txt'  # Ensure this file has 365 classes
    image_path = 'F:/Anfa Backup/E/IBA_8th_sm/FYP/brom/borm/assests/tvtable.jpg'

    # Initialize Inference Class
    tester = Inference()

    # Load the classes and model
    classes = tester.load_classes(classes_file)
    model, object_idt, classifier = tester.load_model(model_type='cdopm_resnet18', model_path=model_path)

    # Perform prediction
    predicted_label, confidence, detected_objects = tester.predict(model, classes, image_path)

    # Display results
    print(f"Predicted label: {predicted_label}, Confidence: {confidence}")
    print(f"Detected objects: {detected_objects}")
