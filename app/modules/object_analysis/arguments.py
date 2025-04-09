import argparse
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

# Constants
BATCH_SIZE = 128
EPOCHS = 40
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
NUM_CLASSES = 7  # 🔥 Set to 7 since you only need 7 classes
DIS_SCALE = 1.0


class ArgumentsParse:
    """ Class for parsing training and testing arguments. """

    @staticmethod
    def argsParser():
        """ Parse training arguments. """
        parser = argparse.ArgumentParser(description='Training of CIOM model')

        parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50',
                            help='Model architecture: ' +
                                 ' | '.join(model_names) +
                                 ' (default: resnet50)')
        parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                            help='Number of data loading workers (default: 4)')
        parser.add_argument('--epochs', default=EPOCHS, type=int, metavar='N',
                            help='Number of total epochs to run')
        parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                            help='Manual epoch number (useful on restarts)')
        parser.add_argument('-b', '--batch-size', default=BATCH_SIZE, type=int,
                            metavar='N', help='Mini-batch size (default: 128)')
        parser.add_argument('--lr', '--learning-rate', default=LEARNING_RATE, type=float,
                            metavar='LR', help='Initial learning rate')
        parser.add_argument('--momentum', default=MOMENTUM, type=float, metavar='M',
                            help='Momentum')
        parser.add_argument('--weight-decay', '--wd', default=WEIGHT_DECAY, type=float,
                            metavar='W', help='Weight decay (default: 1e-4)')
        parser.add_argument('--print-freq', '-p', default=500, type=int,
                            metavar='N', help='Print frequency (default: 10)')
        parser.add_argument('--resume', default='', type=str, metavar='PATH',
                            help='Path to latest checkpoint (default: none)')
        parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                            help='Evaluate model on validation set')
        parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                            help='Path to pretrained model')
        parser.add_argument('--num_classes', default=NUM_CLASSES, type=int,  # 🔥 Set default to 7
                            help='Number of classes in the model')
        parser.add_argument('--om_cls_num', default=150, type=int,
                            help='Number of object classes')
        parser.add_argument('--dataset', default='Places365-7', type=str,  # 🔥 Set default to `Places365-7`
                            help='Choose the dataset used for training')
        parser.add_argument('--om_type', default='copm_resnet50', type=str,
                            help='Choose the type of object model')
        parser.add_argument('--DIS_SCALE', default=DIS_SCALE, type=float,
                            help='Choose the scale of discriminative matrix')

        return parser.parse_args()

    @staticmethod
    def test_argsParser():
        """ Parse testing arguments. """
        parser = argparse.ArgumentParser(description='Testing of CIOM model')

        parser.add_argument('--dataset', default='Places365-7', type=str,  # 🔥 Set default to `Places365-7`
                            help='Choose the dataset used for testing')
        parser.add_argument('--num_classes', default=NUM_CLASSES, type=int,  # 🔥 Ensure it uses 7 classes
                            help='Number of classes in the model')
        parser.add_argument('--om_type', default='copm_resnet50', type=str,
                            help='Choose the type of object model')
        parser.add_argument('--DIS_SCALE', default=DIS_SCALE, type=float,
                            help='Choose the scale of discriminative matrix')
        parser.add_argument('--pretrained', default='', type=str, metavar='PATH',
                            help='Path to pretrained model')

        return parser.parse_args()
