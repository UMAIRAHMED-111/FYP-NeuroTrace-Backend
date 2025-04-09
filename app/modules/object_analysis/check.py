import torch

# Path to your model checkpoint
model_path = ('F:/Anfa Backup/E/IBA_8th_sm/FYP/brom_14/borm/models/resnet50_best_res50.pth.tar'
              '')

# Load the checkpoint
checkpoint = torch.load(model_path)

# Print all keys in the checkpoint
print("Keys in the checkpoint:")
for key in checkpoint.keys():
    print(f"- {key}")

# Optionally, print the state_dict keys (model weights) if you want to inspect those as well
if 'model_state_dict' in checkpoint:
    print("\nKeys in model_state_dict:")
    for key in checkpoint['model_state_dict'].keys():
        print(f"- {key}")
