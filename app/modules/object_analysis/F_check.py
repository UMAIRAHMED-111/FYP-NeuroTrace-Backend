import torch

# Path to your model checkpoint
model_path = 'F:/Anfa Backup/E/IBA_8th_sm/FYP/brom_14/borm/models/resnet50_best_res50.pth.tar'

# Load the checkpoint
checkpoint = torch.load(model_path)

# Get best accuracy from the checkpoint (if available)
best_accuracy = checkpoint.get('best_prec1', 'N/A')

# Print the best accuracy
print(f"Best Accuracy (best_prec1): {best_accuracy}")
