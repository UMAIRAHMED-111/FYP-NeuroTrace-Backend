import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import os
from ultralytics import YOLO  # Import YOLO for object detection
from model import Object_CDOPM_ResNet18, Object_CDOPM_ResNet50, Fusion_CDOPM_ResNet50, Fusion_CDOPM_ResNet18
from arguments import ArgumentsParse


class Inference:
    def __init__(self, model_type='cdopm_resnet50', model_path='models/cdopm_resnet50_best.pth.tar',
                 classes_file='object_information/categories_Places365_14.txt',
                 image_path='F:\\Anfa Backup\\E\\IBA_8th_sm\\FYP\\brom\\borm\\assests\\self_lounge.jpg'):
        self.model_type = model_type
        self.model_path = model_path
        self.image_path = image_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = self.load_classes(classes_file)
        self.model, self.object_idt, self.classifier = self.load_model()
        self.yolo_model = YOLO("yolov8l.pt")

        # Image transformation (same as used in test_resnet+OR.py)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load_classes(self, classes_file):
        """Loads class names from a text file."""
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines()]
        return classes

    def load_model(self):
        print(f"Loading model {self.model_type} from {self.model_path}...")
        if self.model_type == 'cdopm_resnet50':
            object_idt = Object_CDOPM_ResNet50().to(self.device)
            classifier = Fusion_CDOPM_ResNet50(14).to(self.device)  # Updated class number
            model = models.resnet50(num_classes=14).to(self.device)  # Updated class number
        elif self.model_type == 'cdopm_resnet18':
            object_idt = Object_CDOPM_ResNet18().to(self.device)
            classifier = Fusion_CDOPM_ResNet18(14).to(self.device)  # Updated class number
            model = models.resnet18(num_classes=14).to(self.device)  # Updated class number
        else:
            raise ValueError("Invalid model type. Choose 'cdopm_resnet50' or 'cdopm_resnet18'.")

        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()})
        object_idt.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['obj_state_dict'].items()})
        classifier.load_state_dict(
            {k.replace('module.', ''): v for k, v in checkpoint['classifier_state_dict'].items()})

        model.eval()
        object_idt.eval()
        classifier.eval()
        return model, object_idt, classifier

    def detect_objects(self):
        """Runs YOLOv8 on the image to detect objects."""
        results = self.yolo_model(self.image_path)  # Run YOLO on the image
        detected_objects = set()

        for result in results:
            for box in result.boxes:
                obj_id = int(box.cls.item())
                detected_objects.add(self.yolo_model.names[obj_id])  # Get object name

        return list(detected_objects)

    def predict(self):
        """Runs inference on a single image and detects objects."""
        if not os.path.exists(self.image_path):
            raise FileNotFoundError(f"Image not found: {self.image_path}")

        # **Step 1: Detect Objects in Image using YOLO**
        detected_objects = self.detect_objects()
        print(f" Detected Objects: {detected_objects}")

        # **Step 2: Convert Objects to One-Hot Vector**
        object_vector = [1 if obj in detected_objects else 0 for obj in self.classes]

        # **Step 3: Process Image for Scene Recognition**
        image = Image.open(self.image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            # **Step 4: Scene Recognition**
            output_conv = self.model(input_tensor)
            h_x = F.softmax(output_conv, dim=1).squeeze()
            probs, idx = h_x.sort(0, True)
            predicted_label = self.classes[idx[0].item()]

        print(f" Predicted Scene: {predicted_label} (Confidence: {probs[0].item():.4f})")
        return predicted_label, probs[0].item(), detected_objects


if __name__ == '__main__':
    tester = Inference()
    tester.predict()
