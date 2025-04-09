import os
import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from collections import defaultdict, Counter
from ultralytics import YOLO

from app.modules.object_analysis.model import (
    Object_CDOPM_ResNet18,
    Object_CDOPM_ResNet50,
    Fusion_CDOPM_ResNet50,
    Fusion_CDOPM_ResNet18,
)

class FramePipeline:
    def __init__(self, model_type='cdopm_resnet50',
                 model_path=None,
                 classes_file=None,
                 frames_dir=None):

        # Get base directory of this file (app/modules/object_analysis)
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # Paths resolved relative to this file
        if classes_file is None:
            classes_file = os.path.join(base_dir, "object_information", "categories_Places365_14.txt")

        if model_path is None:
            model_path = os.path.join(base_dir, "models", "cdopm_resnet50_best.pth.tar")
            
        if frames_dir is None:
            # Get project root (two levels up from this file)
            root_dir = os.path.abspath(os.path.join(base_dir, "..", "..", ".."))
            frames_dir = os.path.join(root_dir, "assets", "frames")

        self.model_type = model_type
        self.model_path = model_path
        self.frames_dir = frames_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.classes = self.load_classes(classes_file)
        self.model, self.object_idt, self.classifier = self.load_model()
        self.yolo_model = YOLO("yolov8l.pt")

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.object_counter = defaultdict(int)
        self.scene_votes = []

    def load_classes(self, classes_file):
        with open(classes_file, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def load_model(self):
        print(f"Loading model: {self.model_type} from {self.model_path}")
        if self.model_type == 'cdopm_resnet50':
            object_idt = Object_CDOPM_ResNet50().to(self.device)
            classifier = Fusion_CDOPM_ResNet50(14).to(self.device)
            model = models.resnet50(num_classes=14).to(self.device)
        elif self.model_type == 'cdopm_resnet18':
            object_idt = Object_CDOPM_ResNet18().to(self.device)
            classifier = Fusion_CDOPM_ResNet18(14).to(self.device)
            model = models.resnet18(num_classes=14).to(self.device)
        else:
            raise ValueError("Invalid model type.")

        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()})
        object_idt.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['obj_state_dict'].items()})
        classifier.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['classifier_state_dict'].items()})

        model.eval()
        object_idt.eval()
        classifier.eval()
        return model, object_idt, classifier

    def detect_objects(self, image_path):
        results = self.yolo_model(image_path)
        detected = []  # No set used here

        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls.item())
                class_name = self.yolo_model.names[cls_id]
                detected.append(class_name)

        return detected

    def predict_scene(self, image_path):
        image = Image.open(image_path).convert('RGB')
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output_conv = self.model(tensor)
            h_x = F.softmax(output_conv, dim=1).squeeze()
            probs, idx = h_x.sort(0, True)
            predicted_label = self.classes[idx[0].item()]
            return predicted_label

    def process_frames(self):
        if not os.path.exists(self.frames_dir):
            raise FileNotFoundError(f"Frames folder not found: {self.frames_dir}")

        frame_files = [f for f in os.listdir(self.frames_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        if not frame_files:
            print("No frames found in the folder.")
            return

        for frame_file in frame_files:
            frame_path = os.path.join(self.frames_dir, frame_file)
            print(f"\n🔍 Processing: {frame_file}")

            detected = self.detect_objects(frame_path)
            for obj in detected:
                self.object_counter[obj] += 1

            print(f"🧠 Detected Objects in This Frame:")
            for obj in detected:
                print(f"   - {obj}")

            scene = self.predict_scene(frame_path)
            self.scene_votes.append(scene)
            print(f"🏞️ Predicted Scene: {scene}")

        # Final results
        print("\n🎯 ===== FINAL SUMMARY =====")
        print("✅ Object Detection Dictionary:")
        for obj, count in sorted(self.object_counter.items()):
            print(f"   - {obj}: {count} time(s)")

        final_scene = Counter(self.scene_votes).most_common(1)[0][0]
        print(f"\n🏆 Final Scene Prediction (Majority Vote): {final_scene}")

        # Clean up
        print("\n🧹 Deleting all processed frames...")
        for f in frame_files:
            os.remove(os.path.join(self.frames_dir, f))
        print("✅ Cleanup complete.")

        return dict(self.object_counter), final_scene


if __name__ == '__main__':
    pipeline = FramePipeline()
    pipeline.process_frames()
