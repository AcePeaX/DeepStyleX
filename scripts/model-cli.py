import argparse
import os
import sys
import numpy as np

dirname = os.path.abspath(os.path.join(__file__, "..", "..", "lib"))
sys.path.append(dirname)


def get_train_args(parser):
    """Define arguments for the 'train' command."""
    parser.add_argument('--epochs', type=int, required=True, help="Number of training epochs")
    parser.add_argument('--style-path', type=str, required=True, help="The path of the style image")
    #parser.add_argument('--log-interval', type=int, default=None, help="Log training progress every N images (default: None)")
    parser.add_argument('--checkpoint-interval', type=int, default=None, help="Save model checkpoint every N images (default: None)")
    parser.add_argument('--checkpoint-dir', type=str, default=None, help="Directory to save checkpoints (required if --checkpoint-interval is set)")
    parser.add_argument('--dataset', type=str, required=True, help="Path to the dataset folder")
    parser.add_argument('--batch-size', type=int, default=4, help="The batch size for training")
    parser.add_argument('--style-weight', type=float, default=500000, help="Weight for style loss")
    parser.add_argument('--batch-norm', action='store_true', default=False, help="Use BatchNorm instead of InstanceNorm")
    parser.add_argument('--model-name', type=str, default=None, help="Name of the model being trained")
    parser.add_argument('--resume-path', type=str, default=None, help="Path to resume training from a saved model")
    parser.add_argument('--stored-optimizer-path', type=str, default=None, help="Path to a stored optimizer state to resume training")
    parser.add_argument('--save-optimizer', action='store_true', help="Save optimizer state along with the model")
    parser.add_argument('--save-optimizer-checkpoint', action='store_true', help="Save optimizer state in checkpoints")
    parser.add_argument('--output-path', type=str, required=True, help="Path to save the final trained model")
    parser.add_argument('--seed', type=int, default=42, help="Path to save the final trained model")
    parser.add_argument('--lr', type=float, default=8e-4, help="The learning rate of the model")
    parser.add_argument('--cuda', default=None, action='store_true', help="Use CUDA (GPU) for training")
    parser.add_argument('--cpu', action='store_false', dest="cuda", help="Use CPU for training")
    parser.add_argument('--mps', action='store_true', help="Use Metal Performance Shaders (MPS) on macOS")
    
    parser.add_argument('--no-relu1', action='store_false', default=True, help="Not use the smallest features in style image")
    parser.add_argument('--no-relu2', action='store_false', default=True, help="Not use the smaller features in style image")
    parser.add_argument('--no-relu3', action='store_false', default=True, help="Not use the medium features in style image")
    parser.add_argument('--no-relu4', action='store_false', default=True, help="Not use the larger features in style image")
    parser.add_argument('--no-relu5', action='store_false', default=True, help="Not use the largest features in style image")

    return parser


def get_eval_args(parser):
    """Define arguments for the 'eval' command."""
    parser.add_argument('--model-path', type=str, required=True, help="Path to the trained model for evaluation")
    parser.add_argument('--input-image', type=str, required=True, help="Path to the input image")
    parser.add_argument('--output-image', type=str, required=True, help="Path to save the stylized image")
    parser.add_argument('--cuda', default=None, action='store_true', help="Use CUDA (GPU) for testing")
    parser.add_argument('--cpu', action='store_false', dest="cuda", help="Use CPU for testing")
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
    import torch
    from DeepStyleX import DeepStyleX
    from vgg import VGGFeatures
    from data import ImageFolderDataset
    from utils import preprocess, normalize, gram_matrix
    from PIL import Image
    from modeltools import loss_function, append_metadata
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    vgg_layers = []
    if args.no_relu1:
        vgg_layers.append('relu1_2')
    if args.no_relu2:
        vgg_layers.append('relu2_2')
    if args.no_relu3:
        vgg_layers.append('relu3_3')
    if args.no_relu4:
        vgg_layers.append('relu4_3')
    if args.no_relu5:
        vgg_layers.append('relu5_3')

    # Set the device
    if args.cuda==None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if args.cuda else "cpu")


    # Making the checkpoints and save directories
    save_directory = os.path.dirname(args.output_path)
    if not os.path.exists(save_directory):
        # Create the folder (and any necessary intermediate directories)
        os.makedirs(save_directory)
    if args.checkpoint_dir!=None:
        if not os.path.exists(args.checkpoint_dir):
            os.makedirs(args.checkpoint_dir)

    style_image = Image.open(args.style_path)

    preprocess_device = lambda x: preprocess(x, device=device)
    images_dataset = ImageFolderDataset(args.dataset, transform=preprocess_device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    optimizer = None

    if args.resume_path!=None:
        if os.path.exists(args.resume_path):
            model, optimizer = DeepStyleX.load(args.resume_path)
        else:
            print("Model not found in:",args.resume_path)
            answer = input("Do you want to continue with a fresh model? (yes) : ")
            if answer.lower()=='yes':
                model = DeepStyleX(batch_norm=args.batch_norm)
            else:
                print("Exiting.")
                return
    else:
        model = DeepStyleX(batch_norm=args.batch_norm)  # Here we choose to not use batch norm, instead we use Instance norm
    model.to(device)

    vgg_model = VGGFeatures()
    vgg_model.to(device)

    batch_size = args.batch_size

    feature_style = vgg_model(normalize(preprocess(style_image, device=device,resize=False)))
    
    dataloader = DataLoader(images_dataset, batch_size=batch_size, shuffle=True)


    criterion = torch.nn.MSELoss()
    if optimizer==None:
        optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)
    model.train()

    style_gram_features = dict()
    for feature in feature_style.keys():
        style_gram_features[feature] = gram_matrix(feature_style[feature])

    batch_len = int(len(images_dataset)/batch_size)

    content_weight = 1
    style_weight = args.style_weight

    batch_count = 0

    checkpoint_basename = "checkpoint"
    if args.model_name!=None:
        checkpoint_basename+="_"+args.model_name

    for epoch in range(args.epochs):
        pbar = tqdm(enumerate(dataloader), total=batch_len)
        for j, batch_images in pbar:
            optimizer.zero_grad()
            output = model(batch_images)
            with torch.no_grad():
                original_features = vgg_model(normalize(batch_images))
            output_features = vgg_model(normalize(output))

            style_gram_features_batch = dict()
            for feature in feature_style.keys():
                style_gram_features_batch[feature] = style_gram_features[feature].repeat(len(batch_images), 1, 1)
            
            loss = loss_function(output_features, original_features, style_gram_features_batch, criterion, content_weight=content_weight, style_weight=style_weight, vgg_layers=vgg_layers)

            loss.backward()
            optimizer.step()
            pbar.set_description(str(epoch+1)+' -> '+str(j+1)+' : '+'{0:.8f}'.format(loss.item()))

            batch_count+=1
            
            if args.checkpoint_interval!=None:
                if batch_count%args.checkpoint_interval==0:
                    model.save(os.path.join(args.checkpoint_dir,append_metadata(checkpoint_basename, epoch, j+1))+".pth")
    
    if args.save_optimizer:
        model.save(args.output_path, optimizer=optimizer, style_image=style_image)
    else:
        model.save(args.output_path, style_image=style_image)


def evaluate(args):
    """Evaluation logic."""
    print("Importing...")

    import torch
    from DeepStyleX import DeepStyleX
    from utils import preprocess, deprocess
    from PIL import Image

    print("Styling...")

    # Set the device
    if args.cuda==None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if args.cuda else "cpu")

    input_image = Image.open(args.input_image)
    input_image = preprocess(input_image, resize=False).unsqueeze(0)

    with torch.no_grad():
        model, _ = DeepStyleX.load(args.model_path, map_location=device)
        model.to(device)
        model.eval()
        output = model(input_image)

    print("Saving...")

    output_img = deprocess(output)

    # Making the output directory
    save_directory = os.path.dirname(args.output_image)
    if not os.path.exists(save_directory):
        # Create the folder (and any necessary intermediate directories)
        os.makedirs(save_directory)
    output_img.save(args.output_image)


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
