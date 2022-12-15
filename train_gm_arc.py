import argparse
import datetime
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

from lib.datasets.utils import get_train_and_val_datasets
from lib.models.my_arcface import MyArcFace
from lib.utils import seed_worker


# To parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Modernized GestaltMatcher')

    # Run parameters
    parser.add_argument('--session', type=int, dest='session',
                        help='session used to distinguish model tests.')
    parser.add_argument('--batch_size', type=int, default=128, metavar='BN',
                        help='input batch size for training (default: 280)')
    parser.add_argument('--epochs', type=int, default=50, metavar='E',
                        help='number of epochs to train (default: 500)')
    parser.add_argument('--lr', type=float, default=5e-3, metavar='LR',  # lr=1e-3
                        help='learning rate (default: 0.005)')
    parser.add_argument('--seed', type=int, default=11, metavar='S',
                        help='random seed (default: 11)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--log_interval', type=int, default=1000,
                        help='how many images (not batches) to wait before logging training status')
    parser.add_argument('--val_interval', type=int, default=10000,
                        help='how many images (not batches) to wait before validation is evaluated (and optimizer is stepped).')
    parser.add_argument('--use_tensorboard', action='store_true', default=False,
                        help='Use tensorboard for logging')

    # Model parameters
    parser.add_argument('--model_type', default='glint360k_r50', dest='model_type',
                        help='model backend to use')
    parser.add_argument('--in_channels', default=3, dest='in_channels', type=int,
                        help='number of color channels of the images used as input (default: 1)')
    parser.add_argument('--img_size', default=112, dest='img_size', type=int,
                        help='input image size of the model (default: 100)')
    parser.add_argument('--unfreeze', action='store_false', default=True, dest='freeze',
                        help='flag to set if you want to unfreeze the base model weights.')

    # Dataset parameters
    parser.add_argument('--dataset', default='gmdb', dest='dataset',
                        help='which dataset to use. (Options: "casia", "gmdb")')
    parser.add_argument('--dataset_type', default='', dest='dataset_type',
                        help='type of the dataset to use, e.g. normal (="") or augmented(="aug") (default="")')
    parser.add_argument('--dataset_version', default='v1.0.3', dest='dataset_version', type=str,
                        help='version of the dataset to use (default="v1.0.3")')
    parser.add_argument('--lookup_table', default='', dest='lookup_table_path',
                        help='lookup table path, use if you want to load path instead of generation a lookup table (default = "")')

    # File locations
    parser.add_argument('--data_dir', default='C:/Users/Alexander/Documents/data', dest='data_dir',
                        help='Location of the data directory (not dataset). (default = home pc)')
    parser.add_argument('--weight_dir', default='saved_models', dest='weight_dir',
                        help='Location of the model weights directory. (default = "saved_models")')

    # running on my local machine means different path types, and num_workers
    parser.add_argument('--local', action='store_true', default=False,
                        help='Running on local machine, fewer num_workers')

    #
    parser.add_argument('--paper_model', default='None', dest='paper_model', type=str,
                        help='Use when reproducing paper models a) r50-mix, or b) r100')

    return parser.parse_args()


# Training loop
def train(args, model, device, train_loader, optimizer, epochs=-1, val_loader=None, scheduler=None):
    model.train()

    # Time measurements
    tick = datetime.datetime.now()

    # Tensorboard Writer
    if args.use_tensorboard:
        writer = SummaryWriter(
            comment=f"s{args.session}_{args.model_type}_512d_{args.dataset}_{args.dataset_type}"
                    f"_{args.dataset_version}_bs{args.batch_size}_size{args.img_size}_channels{args.in_channels}")
    global_step = 0

    if epochs == -1:
        epochs = args.epochs

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.
        for batch_idx, (data, target) in enumerate(train_loader):

            data = data.to(device, dtype=torch.float32)
            target = target.to(device, dtype=torch.int64).unsqueeze(1)

            pred, pred_rep = model(data)
            loss = F.cross_entropy(pred, target.view(-1), weight=args.ce_weights)
            loss.backward()

            ## Clipping gradients here, if we get exploding gradients we should revise...
            # nn.utils.clip_grad_value_(model.parameters(), 0.1)

            optimizer.step()
            optimizer.zero_grad()

            del pred, pred_rep, data, target

            epoch_loss += loss.item()
            if (batch_idx + 1) % args.log_interval == 0:
                tock = datetime.datetime.now()
                print('[{}] Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t(Elapsed time {:.1f}s)'.format(
                    tock.strftime("%H:%M:%S"), epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                                                      100. * batch_idx / len(train_loader), loss.item(),
                    (tock - tick).total_seconds()))
                tick = tock

                if args.use_tensorboard:
                    writer.add_scalar('Train/ce_loss', loss.item(), global_step)

            del loss
            if val_loader:
                if (batch_idx + 1) % args.val_interval == 0:
                    avg_val_loss, t_acc, t5_acc, ma_t1_acc, ma_t5_acc = validate(model, device, val_loader, args)

                    tick = datetime.datetime.now()

                    if args.use_tensorboard:
                        writer.add_scalar('Val/ce_loss', avg_val_loss, global_step)
                        writer.add_scalar('Val/top_acc', t_acc, global_step)
                        writer.add_scalar('Val/top_5_acc', t5_acc, global_step)
                        writer.add_scalar('Val/top_1_mean_acc', ma_t1_acc, global_step)
                        writer.add_scalar('Val/top_5_mean_acc', ma_t5_acc, global_step)

                    if scheduler:
                        scheduler.step(ma_t5_acc)
                        #scheduler.step(avg_val_loss)

            global_step += args.batch_size

        # Epoch is completed
        print(f"Overall average training loss: {epoch_loss / len(train_loader):.6f}")
        if args.use_tensorboard:
            writer.add_scalar('Train/ce_loss', epoch_loss / len(train_loader), global_step)

        # Plot the performance on the validation set
        avg_val_loss, t_acc, t5_acc, ma_t1_acc, ma_t5_acc = validate(model, device, val_loader, args)
        if args.use_tensorboard:
            writer.add_scalar('Val/ce_loss', avg_val_loss, global_step)
            writer.add_scalar('Val/top_acc', t_acc, global_step)
            writer.add_scalar('Val/top_5_acc', t5_acc, global_step)
            writer.add_scalar('Val/top_1_mean_acc', ma_t1_acc, global_step)
            writer.add_scalar('Val/top_5_mean_acc', ma_t5_acc, global_step)

        if scheduler:
            scheduler.step(ma_t5_acc)
            #scheduler.step(avg_val_loss)

        # Save model
        print(
            f"Saving model in: "
            f"s{args.session}_{args.model_type}_512d_{args.dataset}_{args.dataset_type}_{args.dataset_version}"
            f"_bs{args.batch_size}_size{args.img_size}_channels{args.in_channels}_e{epoch}.pt")
        torch.save(
            model.state_dict(),
            os.path.join(args.weight_dir,f"s{args.session}_{args.model_type}_512d_{args.dataset}_{args.dataset_type}"
            f"_{args.dataset_version}_bs{args.batch_size}_size{args.img_size}_channels{args.in_channels}_e{epoch}.pt"))

    if args.use_tensorboard:
        writer.flush()
        writer.close()


# Validation loop
def validate(model, device, val_loader, args, out=False):
    model.eval()
    val_ce_loss = 0.
    top_acc = 0.
    top_5_acc = 0.

    pred_per_class = [[] for _ in range(args.num_classes)]

    tick = datetime.datetime.now()
    val_size = 0
    with torch.no_grad():
        diag = torch.eye(args.val_bs, device=device)
        for idx, (data, target) in enumerate(val_loader):
            data = data.to(device, dtype=torch.float32)
            target = target.to(device, dtype=torch.int64).unsqueeze(1)

            pred, pred_rep = model(data)
            pred, pred_rep = pred.detach(), pred_rep.detach()
            val_ce_loss += F.cross_entropy(pred, target.view(-1), weight=args.ce_weights, reduction='sum').item()

            if out:
                for i in range(args.val_bs):
                    print(f"{target[i].item()},{pred[i].tolist()}")

            # some times the last batch might not be the same size
            bs = len(data)
            if bs != args.val_bs:
                diag = torch.eye(len(data), device=device)

            # extra stats
            max_pred, max_idx = torch.max(pred, dim=-1)
            top_pred, top_idx = torch.topk(pred, k=5, dim=-1)
            top_acc += torch.sum((target == max_idx) * diag).item()
            top_5_acc += np.sum([target[i] in top_idx[i] for i in range(bs)]).item()  # ... yep, quite ugly

            # TODO: support bs > 1
            if bs == 1:
                # append a ranked list of predictions to the correct class
                pred_per_class[target[0]].append(top_idx[0].cpu().numpy())

            val_size += bs

    top_acc = torch.true_divide(top_acc, val_size).item()
    top_5_acc = torch.true_divide(top_5_acc, val_size).item()

    # calculate the mean of the average performance per class
    mean_average_top_1 = np.mean([np.mean([(class_idx == prediction[0]) for prediction in class_pred_list])
                                  for class_idx, class_pred_list in enumerate(pred_per_class)])
    mean_average_top_5 = np.mean([np.mean([(class_idx in prediction) for prediction in class_pred_list])
                                  for class_idx, class_pred_list in enumerate(pred_per_class)])

    model.train()

    print(f"Average BCE Loss ({val_ce_loss / val_size}) during validation")
    print(f"\tTop-1 accuracy: {top_acc}, Top-5 accuracy: {top_5_acc}")
    print(f"\tMean Top-1 accuracy: {mean_average_top_1}, Mean top-5 accuracy: {mean_average_top_5}")
    print(f"Elapsed time during validation: {(datetime.datetime.now() - tick).total_seconds():.1f}s")

    return val_ce_loss / val_size, top_acc, top_5_acc, mean_average_top_1, mean_average_top_5


def main():
    # Training settings
    args = parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using {'GPU.' if use_cuda else 'CPU, as was explicitly requested, or as GPU is not available.'}")

    # Seed everything
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # When using GPU we need to set extra seeds
    if use_cuda:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
    # torch.set_deterministic(True)

    # If we want to reproduce paper results, by giving command-line argument `args.paper_model` == 'a' or 'b',
    # we replace some other parameters
    f_wd = c_wd = lr = c_lr = -1
    if args.paper_model == 'a':    # r50 mix
        args.model_type = 'glint360k_r50'
        args.batch_size = 64
        f_wd = 5e-5
        c_wd = 5e-4
        lr = 5e-4
        c_lr = 1e-3
    elif args.paper_model == 'b':  # r100
        args.model_type = 'glint360k_r100'
        args.batch_size = 128
        f_wd = 0.
        c_wd = 5e-4
        lr = 1e-3
        c_lr = 1e-3

    # Dataset and dataloaders
    kwargs = {}
    if use_cuda:
        kwargs.update({'num_workers': (0 if args.local else 16), 'pin_memory': True})

    dataset_train = dataset_val = None
    lookup_table = None
    distr = None

    # Create and get the training and validation datasets
    dataset_train, dataset_val = get_train_and_val_datasets(args.dataset, args.dataset_type, args.dataset_version,
                                                            args.img_size, args.in_channels, args.data_dir,
                                                            img_postfix='_rot_aligned')

    # Get the number of classes from the dataset
    args.num_classes = dataset_train.get_num_classes()

    # Try and get the dataset distribution and lookup table
    try:
        distr = dataset_train.get_distribution()
        print(f"Training dataset size: {sum(distr)}, with {len(distr)} classes and distribution: {distr}")
        print(f"Validation dataset size: {sum(dataset_val.get_distribution())}, "
              f"with distribution: {dataset_val.get_distribution()}")

        lookup_table = dataset_train.get_lookup_table()
    except:
        print("An error occurred while getting the dataset distribution and lookup table. Likely the dataset does not "
              "have an implementation for these functions.")

    # Write lookup table to file if we generated a lookup table for GMDB (i.e. when we don't supply one)
    if (lookup_table is not None) and (args.dataset != 'casia') and (args.lookup_table_path == ''):
        f = open(f"lookup_table_{args.dataset}_{args.dataset_version}.txt", "w+")
        f.write("index_id to disorder_id\n")
        f.write(f"{lookup_table}")
        f.flush()
        f.close()

    # Set validation batch size to 1
    args.val_bs = 1

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(dataset_train, **kwargs, shuffle=True, batch_size=args.batch_size,
                                               worker_init_fn=seed_worker, drop_last=True)
    val_loader = torch.utils.data.DataLoader(dataset_val, pin_memory=True, num_workers=12, shuffle=False,
                                             drop_last=False,
                                             worker_init_fn=seed_worker,
                                             batch_size=args.val_bs)

    # Attempt to deal with data imbalance: inverse frequency divided by lowest frequency class (0.5 < class_weight <= 1)
    if distr is not None:
        args.ce_weights = (torch.tensor([(sum(distr) / freq) / (sum(distr) / min(distr)) for freq in distr]).float()
                           .to(device)) * 0.5 + 0.5
    else:
        args.ce_weights = None
    print(f"Weighted cross entropy weights: {args.ce_weights}")

    # Create model
    model = MyArcFace(args.num_classes, dataset_base=os.path.join(args.weight_dir, f'{args.model_type}.onnx'),
                      device=device, freeze=True).to(device)
    print(f"Created {'frozen ' if args.freeze else ''}{args.model_type} model with {args.in_channels} in channel"
          f"{'s' if args.in_channels > 1 else ''}, 512d feature dimensionality and {args.num_classes} classes")

    # Set log intervals
    args.log_interval = args.log_interval // args.batch_size
    args.val_interval = args.val_interval // args.batch_size

    ## Continue training/testing:
    # model.load_state_dict(torch.load(f"saved_models/<saved weights>.pt", map_location=device))

    ## Init optimizer
    # We seperate the optimizer for cnn-base and classifier
    optimizer = optim.Adam([
        {'params': model.base.parameters()},
        {'params': model.features.parameters(), 'weight_decay': f_wd if f_wd != -1 else 5e-5},
        {'params': model.classifier.parameters(),
         'weight_decay': c_wd if c_wd != -1 else 5e-4, 'lr': c_lr if c_lr != -1 else 1e-3
         }
    ], lr=lr if lr != -1 else args.lr, weight_decay=0.)

    # Init scheduler
    scheduler = lr_sched.ReduceLROnPlateau(optimizer, factor=0.5, verbose=True, min_lr=1e-5, mode="max", patience=5, threshold=5e-4)

    ## Call explicit model weight initialization (only do this for the base task, if at all)
    # model.init_layer_weights()

    # Run training loop
    train(args, model, device, train_loader, optimizer, val_loader=val_loader, scheduler=scheduler)

    ## Run final validation step with extra output
    # validate(model, device, val_loader, args, out=True)

    # Save entire model
    torch.save(model, os.path.join(args.weight_dir, f"s{args.session}_{args.model_type}_512d_{args.dataset}"
                                                    f"_{args.dataset_type}_{args.dataset_version}_bs{args.batch_size}"
                                                    f"_size{args.img_size}_channels{args.in_channels}_last_model.pth"))


if __name__ == '__main__':
    main()
