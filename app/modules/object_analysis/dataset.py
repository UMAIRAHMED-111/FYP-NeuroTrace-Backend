import json
import numpy as np
import os
import torch
from torchvision import transforms

class DatasetSelection:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.data_dir = r"F:\Anfa Backup\E\IBA_8th_sm\FYP\brom_14\borm\object_information"  # Adjust base directory
        self.cls_num = None  # Ensure cls_num is always defined

        # Set dataset parameters
        if self.dataset_name in ['Places365-7', 'sun']:
            self.cls_num = 7
        elif self.dataset_name == 'Places365-14':
            self.cls_num = 14
        else:
            raise ValueError("❌ Error: Dataset not recognized")

    def load_json(self, filename):
        """Loads a JSON file from the dataset directory."""
        json_path = os.path.join(self.data_dir, filename)
        with open(json_path, "r", encoding="utf-8") as json_file:  # Encoding fix for Windows
            return json.load(json_file)

    def discriminative_matrix_estimation(self):
        """Computes the discriminative matrix for the dataset."""
        if self.dataset_name in ['Places365-7', 'sun']:
            obj_result_file = '150obj_result_Places365_7.npy'
            obj_count_file = '150obj_number_Places365_7.npy'
        elif self.dataset_name == 'Places365-14':
            obj_result_file = '150obj_result_Places365_14.npy'
            obj_count_file = '150obj_number_Places365_14.npy'
        else:
            raise ValueError("❌ Error: Dataset not recognized")

        self.num_sp = np.load(os.path.join(self.data_dir, obj_result_file))
        self.num_total = np.load(os.path.join(self.data_dir, obj_count_file))
        self.obj_num = 150

        matrix_p_o_c = np.zeros((self.cls_num, self.obj_num, self.obj_num))

        for i in range(self.cls_num):
            p_o_c = self.num_sp[i] / self.num_total[i]
            p_o_c = p_o_c.reshape(1, -1)
            matrix_p_o_c[i] = np.dot(p_o_c.T, p_o_c)

        discriminative_matrix = np.zeros((self.obj_num, self.obj_num))
        for i in range(self.obj_num):
            for j in range(self.obj_num):
                values = [matrix_p_o_c[k, i, j] / self.cls_num for k in range(self.cls_num)]
                discriminative_matrix[i, j] = np.std(values)

        return discriminative_matrix

    def datasetSelection(self):
        """Loads the dataset JSON file and separates data into image paths and labels."""
        print("🔹 Loading dataset...")

        # Load JSON file containing paths and one-hot encoded labels
        full_data = self.load_json(f'150obj_Places365_{self.cls_num}.json')

        print(f"✅ Dataset loaded successfully!")
        print(f"📊 Total images: {len(full_data)}")

        # Load valid class names from category file
        category_file = os.path.join(self.data_dir, f'categories_Places365_7.txt')
        valid_categories = {}

        with open(category_file, "r") as f:
            for line in f:
                path, idx = line.strip().split(" ")
                category_name = path[3:]  # Remove leading "/b/", "/c/" etc.
                valid_categories[category_name] = int(idx)

        # **Filter dataset to include only selected categories**
        excluded_classes = {"playroom", "garage-indoor"}

        filtered_data = {
            k: v for k, v in full_data.items()
            if k.split("/")[-2] in valid_categories and k.split("/")[-2] not in excluded_classes
        }

        # Store filtered class names
        classes = tuple(valid_categories.keys())

        print(f"✅ Filtered dataset to only {len(classes)} categories (Excluding Playroom & Garage Indoor): {classes}")

        # Convert JSON paths to correct format for Windows
        image_paths = [os.path.join(self.data_dir, path.lstrip('/data')) for path in filtered_data.keys()]
        labels = list(filtered_data.values())

        return image_paths, labels, self.data_dir  # Return only the necessary three variables
