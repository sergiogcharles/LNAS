# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import copy
import logging
import time
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import datasets
import utils
from model import CNN
from nni.nas.pytorch.fixed import apply_fixed_architecture
from nni.nas.pytorch.utils import AverageMeter

from nni.compression.pytorch.utils.counter import count_flops_params
from nni.algorithms.compression.pytorch.pruning import LotteryTicketPruner
from nni.compression.pytorch import apply_compression_results, ModelSpeedup
import time
import os

# logger = logging.getLogger('nni')
logger = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# writer = SummaryWriter(args.exp)
writer= None


def train(config, train_loader, model, optimizer, criterion, epoch, prune_iter):
    top1 = AverageMeter("top1")
    top5 = AverageMeter("top5")
    losses = AverageMeter("losses")

    cur_step = epoch * len(train_loader)
    cur_lr = optimizer.param_groups[0]["lr"]
    logger.info("Epoch %d LR %.6f", epoch, cur_lr)
    writer.add_scalar("lr", cur_lr, global_step=cur_step)

    model.train()

    for step, (x, y) in enumerate(train_loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        bs = x.size(0)

        optimizer.zero_grad()
        logits, aux_logits = model(x)
        loss = criterion(logits, y)
        if config.aux_weight > 0.:
            loss += config.aux_weight * criterion(aux_logits, y)
        loss.backward()
        # gradient clipping
        nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        optimizer.step()

        accuracy = utils.accuracy(logits, y, topk=(1, 5))
        losses.update(loss.item(), bs)
        top1.update(accuracy["acc1"], bs)
        top5.update(accuracy["acc5"], bs)
        writer.add_scalar("loss/train", loss.item(), global_step=cur_step)
        writer.add_scalar("acc1/train", accuracy["acc1"], global_step=cur_step)
        writer.add_scalar("acc5/train", accuracy["acc5"], global_step=cur_step)

        if step % config.log_frequency == 0 or step == len(train_loader) - 1:
            logger.info(
                "Train: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                    epoch + 1, config.epochs, step, len(train_loader) - 1, losses=losses,
                    top1=top1, top5=top5))

        cur_step += 1

    logger.info("Train: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, config.epochs, top1.avg))

    # Write train loss to file
    train_loss_filename = 'exp' + config.exp + '_sparsity_' + str(config.sparsity) + "/train_loss.txt"
    os.makedirs(os.path.dirname(train_loss_filename), exist_ok=True)
    with open(train_loss_filename, "a") as f:
        f.write(f'Train loss @ pruning iter {prune_iter} @ epoch {epoch}: {losses.avg}\n')

    # Write train accuracy to file
    train_accuracy_filename = 'exp' + config.exp + '_sparsity_' + str(config.sparsity) + "/train_accuracy.txt"
    os.makedirs(os.path.dirname(train_accuracy_filename), exist_ok=True)
    with open(train_accuracy_filename, "a") as f:
        f.write(f'Train top-1 accuracy @ pruning iter {prune_iter} @ epoch {epoch}: {top1.avg}\n')


def validate(config, valid_loader, model, criterion, epoch, cur_step, prune_iter):
    top1 = AverageMeter("top1")
    top5 = AverageMeter("top5")
    losses = AverageMeter("losses")

    model.eval()

    with torch.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            bs = X.size(0)

            logits = model(X)
            loss = criterion(logits, y)

            accuracy = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), bs)
            top1.update(accuracy["acc1"], bs)
            top5.update(accuracy["acc5"], bs)

            if step % config.log_frequency == 0 or step == len(valid_loader) - 1:
                logger.info(
                    "Valid: [{:3d}/{}] Step {:03d}/{:03d} Loss {losses.avg:.3f} "
                    "Prec@(1,5) ({top1.avg:.1%}, {top5.avg:.1%})".format(
                        epoch + 1, config.epochs, step, len(valid_loader) - 1, losses=losses,
                        top1=top1, top5=top5))

    writer.add_scalar("loss/test", losses.avg, global_step=cur_step)
    writer.add_scalar("acc1/test", top1.avg, global_step=cur_step)
    writer.add_scalar("acc5/test", top5.avg, global_step=cur_step)

    logger.info("Valid: [{:3d}/{}] Final Prec@1 {:.4%}".format(epoch + 1, config.epochs, top1.avg))

    # Write val loss to file
    val_loss_filename = 'exp' + config.exp + '_sparsity_' + str(config.sparsity) + "/val_loss.txt"
    os.makedirs(os.path.dirname(val_loss_filename), exist_ok=True)
    with open(val_loss_filename, "a") as f:
        f.write(f'Val loss @ pruning iter {prune_iter} @ epoch {epoch}: {losses.avg}\n')

    # Write val accuracy to file
    val_accuracy_filename = 'exp' + config.exp + '_sparsity_' + str(config.sparsity) + "/val_accuracy.txt"
    os.makedirs(os.path.dirname(val_accuracy_filename), exist_ok=True)
    with open(val_accuracy_filename, "a") as f:
        f.write(f'Val top-1 accuracy @ pruning iter {prune_iter} @ epoch {epoch}: {top1.avg}\n')

    return top1.avg


# def test(args, model, device, criterion, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             output = model(data)
#             test_loss += criterion(output, target).item()
#             pred = output.argmax(dim=1, keepdim=True)
#             correct += pred.eq(target.view_as(pred)).sum().item()
#     test_loss /= len(test_loader.dataset)
#     acc = 100 * correct / len(test_loader.dataset)

#     print('Test Loss: {}  Accuracy: {}%\n'.format(
#         test_loss, acc))
#     return acc

if __name__ == "__main__":
    parser = ArgumentParser("darts")
    parser.add_argument("--layers", default=20, type=int)
    parser.add_argument("--batch-size", default=96, type=int)
    parser.add_argument("--log-frequency", default=10, type=int)
    parser.add_argument("--epochs", default=2, type=int)
    parser.add_argument("--aux-weight", default=0.4, type=float)
    parser.add_argument("--drop-path-prob", default=0.2, type=float)
    parser.add_argument("--workers", default=2)
    parser.add_argument("--grad-clip", default=5., type=float)
    parser.add_argument("--arc-checkpoint", default="./checkpoints/epoch_19.json")
    parser.add_argument("--sparsity", type=float, required=True)
    parser.add_argument("--exp", type=str, required=True)

    args = parser.parse_args()
    dataset_train, dataset_valid = datasets.get_dataset("cifar10", cutout_length=16)

    model = CNN(32, 3, 36, 10, args.layers, auxiliary=True)
    apply_fixed_architecture(model, args.arc_checkpoint)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    criterion.to(device)

    optimizer = torch.optim.SGD(model.parameters(), 0.025, momentum=0.9, weight_decay=3.0E-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=1E-6)

    train_loader = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(dataset_valid,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=args.workers,
                                               pin_memory=True)

    logger = logging.getLogger(os.path.join('logging', args.exp))
    writer = SummaryWriter(os.path.join('summary', args.exp))
    
    config_list = [{
        'prune_iterations': 10,
        'sparsity': args.sparsity,
        'op_types': ['default']
    }]

    model_path = os.path.join('./experiment_data', 'pruned_{}_{}_{}.pth'.format(
        'darts', 'cifar10', 'lottery'))
    mask_path = os.path.join('./experiment_data', 'mask_{}_{}_{}.pth'.format(
        'darts', 'cifar10', 'lottery'))

    pruner = LotteryTicketPruner(model, config_list, optimizer)
    pruner.compress()

    best_state_dict = None
    best_top1 = 0.

    for i in pruner.get_prune_iterations():
      for epoch in range(args.epochs):
          drop_prob = args.drop_path_prob * epoch / args.epochs
          model.drop_path_prob(drop_prob)

          # training
          train(args, train_loader, model, optimizer, criterion, epoch, i)

          # validation
          cur_step = (epoch + 1) * len(train_loader)
          top1 = validate(args, valid_loader, model, criterion, epoch, cur_step, i)
          # best_top1 = max(best_top1, top1)

          # Write top1 to txt file
          # Write accuracy at epoch
        #   accuracies_filename = 'exp' + args.exp + '_sparsity_' + str(args.sparsity) + "/accuracies.txt"
        #   os.makedirs(os.path.dirname(accuracies_filename), exist_ok=True)
        #   with open(accuracies_filename, "a") as f:
        #       f.write('Prune iteration: ' + str(i) + ', ' + 'Epoch: ' + str(epoch) + ', ' + 'top 1: ' + str(top1) + '\n')

          if top1 > best_top1:
            best_top1 = top1
            best_state_dict = copy.deepcopy(model.state_dict())
            # Export the best model, 'model_path' stores state_dict of the pruned model,
            # mask_path stores mask_dict of the pruned model
            pruner.export_model(model_path=model_path, mask_path=mask_path)

          lr_scheduler.step()
      print('prune iteration: {0}, Prec@1: {1}'.format(i, top1))

    logger.info("Final best Prec@1 = {:.4%}".format(best_top1))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()

    apply_compression_results(model, mask_path, device)

    dummy_input = torch.randn([1, 3, 32, 32]).to(device)

    # test model speed
    start = time.time()
    for _ in range(32):
        use_mask_out = model(dummy_input)
    print('elapsed time when use mask: ', time.time() - start)

    flops, params, results = count_flops_params(model, dummy_input)
    print(f"FLOPs: {flops}, params: {params}")

    # Perform speed up
    m_speedup = ModelSpeedup(model, dummy_input, mask_path, device)
    m_speedup.speedup_model()

    flops, params, results = count_flops_params(model, dummy_input)
    print(f"FLOPs: {flops}, params: {params} when using speedup")

    start = time.time()
    for _ in range(32):
        use_speedup_out = model(dummy_input)
    print('elapsed time when use speedup: ', time.time() - start)

    top1 = validate(args, valid_loader, model, criterion, epoch, 100, 100)
    print(f'Top 1 after compression: {top1}')

    # Write stats to txt files 
    cwd = os.getcwd()

    # Write best top1 accuracy
    best_accuracy_filename = 'exp' + args.exp + '_sparsity_' + str(args.sparsity) + "/best_accuracy.txt"
    os.makedirs(os.path.dirname(best_accuracy_filename), exist_ok=True)
    with open(best_accuracy_filename, "w") as f:
        f.write('Best top 1: ' + str(best_top1))

    # Write params 
    params_filename = 'exp' + args.exp + '_sparsity_' + str(args.sparsity) + "/params.txt"
    os.makedirs(os.path.dirname(params_filename), exist_ok=True)
    with open(params_filename, "w") as f:
        f.write('Total params: ' + str(params) + '\n')
        f.write(f'Equivalent to: {params/1e6:.3f}M')

    # Write flops
    flops_filename = 'exp' + args.exp + '_sparsity_' + str(args.sparsity) + "/flops.txt"
    os.makedirs(os.path.dirname(flops_filename), exist_ok=True)
    with open(flops_filename, "w") as f:
        f.write('Total flops: ' + str(flops) + '\n')
        f.write(f'Equivalent to: {flops/1e6:.3f}M')