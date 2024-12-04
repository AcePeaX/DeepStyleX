import argparse
import os
import sys

dirname = os.path.abspath(os.path.join(__file__, "..", "..", "lib"))
sys.path.append(dirname)


def get_train_args(parser):
    """Define arguments for the 'train' command."""
    parser.add_argument('--epochs', type=int, required=True, help="Number of training epochs")
    parser.add_argument('--style-path', type=str, required=True, help="The path of the style image")
    parser.add_argument('--log-interval', type=int, default=None, help="Log training progress every N images (default: None)")
    parser.add_argument('--checkpoint-interval', type=int, default=None, help="Save model checkpoint every N images (default: None)")
    parser.add_argument('--checkpoint-dir', type=str, default=None, help="Directory to save checkpoints (required if --checkpoint-interval is set)")
    parser.add_argument('--dataset', type=str, required=True, help="Path to the dataset folder")
    parser.add_argument('--style-weight', type=float, default=100000, help="Weight for style loss")
    parser.add_argument('--model-name', type=str, default=None, help="Name of the model being trained")
    parser.add_argument('--resume-path', type=str, default=None, help="Path to resume training from a saved model")
    parser.add_argument('--stored-optimizer-path', type=str, default=None, help="Path to a stored optimizer state to resume training")
    parser.add_argument('--save-optimizer', action='store_true', help="Save optimizer state along with the model")
    parser.add_argument('--save-optimizer-checkpoint', action='store_true', help="Save optimizer state in checkpoints")
    parser.add_argument('--output-path', type=str, required=True, help="Path to save the final trained model")
    parser.add_argument('--cuda', default=None, action='store_true', help="Use CUDA (GPU) for training")
    parser.add_argument('--cpu', action='store_false', dest="cuda", help="Use CPU for training")
    parser.add_argument('--mps', action='store_true', help="Use Metal Performance Shaders (MPS) on macOS")
    return parser


def get_eval_args(parser):
    """Define arguments for the 'eval' command."""
    parser.add_argument('--model-path', type=str, required=True, help="Path to the trained model for evaluation")
    parser.add_argument('--input-image', type=str, required=True, help="Path to the input image")
    parser.add_argument('--output-image', type=str, required=True, help="Path to save the stylized image")
    parser.add_argument('--cuda', action='store_true', help="Use CUDA (GPU) for evaluation")
    parser.add_argument('--mps', action='store_true', help="Use Metal Performance Shaders (MPS) on macOS")
    return parser


def parse_arguments():
    """Parse arguments for 'train' and 'eval' commands."""
    parser = argparse.ArgumentParser(description="DeepStyleX Training and Evaluation")

    # Subparsers for train and eval
    subparsers = parser.add_subparsers(dest='command', required=True, help="Sub-command to run")

    # Add sub-parser for 'train'
    train_parser = subparsers.add_parser('train', help="Train the model")
    get_train_args(train_parser)

    # Add sub-parser for 'eval'
    eval_parser = subparsers.add_parser('eval', help="Evaluate the model")
    get_eval_args(eval_parser)

    return parser.parse_args()


def train(args):
    """Training logic."""
    print("Importing...")
    from DeepStyleX import DeepStyleX
    from vgg import VGGFeatures
    from utils import preprocess, deprocess, normalize, gram_matrix, resize_image_with_max_resolution, normalize_batch
    from PIL import Image

    style_image = Image.open(args.style_path)
    print(style_image)

    print(args)
    device = "cuda" if args.cuda else "mps" if args.mps else "cpu"
    print(f"Using device: {device}")


def evaluate(args):
    """Evaluation logic."""
    print("Evaluating...")
    print(f"Model path: {args.model_path}")
    print(f"Input image: {args.input_image}")
    print(f"Output image: {args.output_image}")

    device = "cuda" if args.cuda else "mps" if args.mps else "cpu"
    print(f"Using device: {device}")


def main():
    args = parse_arguments()

    # Dispatch based on the command
    if args.command == 'train':
        # Validate train-specific arguments
        if args.checkpoint_interval is not None and args.checkpoint_dir is None:
            sys.exit("Error: --checkpoint-dir is required when --checkpoint-interval is set")
        if args.save_optimizer_checkpoint and not args.save_optimizer:
            sys.exit("Error: --save-optimizer-checkpoint requires --save-optimizer to be set")
        train(args)

    elif args.command == 'eval':
        # Validate eval-specific arguments
        if not os.path.isfile(args.model_path):
            sys.exit(f"Error: Model path '{args.model_path}' does not exist.")
        if not os.path.isfile(args.input_image):
            sys.exit(f"Error: Input image '{args.input_image}' does not exist.")
        evaluate(args)


if __name__ == "__main__":
    main()
