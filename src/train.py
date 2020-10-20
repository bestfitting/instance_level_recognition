# coding: utf-8
import argparse
from timeit import default_timer as timer

from torch.nn import DataParallel
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from apex.parallel import DistributedDataParallel
from apex.parallel import convert_syncbn_model

from config.config import *
from layers.loss import *
from layers.scheduler import *
from utilities.augment_util import *
from layers.tool import Logger
from networks.imageclsnet import init_network
from dataset.landmark_dataset import RetrievalDataset, image_collate
from utilities.metric_util import generate_score_by_model

parser = argparse.ArgumentParser(description='PyTorch Classification')
parser.add_argument('--out_dir', type=str, help='destination where trained network should be saved')
parser.add_argument('--gpu_id', default='0', type=str, help='gpu id used for training (default: 0)')
parser.add_argument('--arch', default='class_efficientnet_b7_gem_fc_arcface2_1head', type=str,
                    help='model architecture (default: class_efficientnet_b7_gem_fc_arcface2_1head)')
parser.add_argument('--num_classes', default=81313, type=int, help='number of classes (default: 81313)')
parser.add_argument('--in_channels', default=3, type=int, help='in channels (default: 3)')
parser.add_argument('--distributed', default=1, type=int, help='distributed train (default: 1)')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--loss', default='LabelSmoothingLossV1', type=str, help='loss function (default: LabelSmoothingLossV1)')
parser.add_argument('--scheduler', default='SGD', type=str, help='scheduler name (default: SGD)')
parser.add_argument('--epochs', default=7, type=int, help='number epochs to train (default: 7)')
parser.add_argument('--resume', default=None, type=str, help='name of the latest checkpoint (default: None)')
parser.add_argument('--aug_version', default='1', type=str, help='argument version (default: 1)')
parser.add_argument('--split_type', default='v2c', type=str)
parser.add_argument('--batch_size', default=7, type=int, help='train mini-batch size (default: 7)')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers (default: 4)')
parser.add_argument('--preprocessing', type=int, default=1)
parser.add_argument('--save_every_epoch', type=float, default=0.1)
parser.add_argument('--img_size', default=448, type=int, help='image size (default: 448)')
parser.add_argument('--eval_img_size', default=512, type=int, help='image size (default: 512)')
parser.add_argument('--model_file', default=None, type=str, help='fine tune with model file (default: None)')
args = parser.parse_args()

def main():
  args.can_print = (args.distributed and args.local_rank == 0) or (not args.distributed)

  log_out_dir = f'{RESULT_DIR}/logs/{args.out_dir}'
  os.makedirs(log_out_dir, exist_ok=True)
  if args.can_print:
    log = Logger()
    log.open(f'{log_out_dir}/log.train.txt', mode='a')
  else:
    log = None

  model_out_dir = f'{RESULT_DIR}/models/{args.out_dir}'
  if args.can_print:
    log.write(f'>> Creating directory if it does not exist:\n>> {model_out_dir}\n')
  os.makedirs(model_out_dir, exist_ok=True)

  # set cuda visible device
  os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

  # set random seeds
  torch.manual_seed(0)
  torch.cuda.manual_seed_all(0)
  np.random.seed(0)

  model_params = {}
  model_params['architecture'] = args.arch
  model_params['num_classes'] = args.num_classes
  model_params['in_channels'] = args.in_channels
  model_params['can_print'] = args.can_print
  model = init_network(model_params)

  # move network to gpu
  if args.distributed:
    dist.init_process_group(backend='nccl', init_method='env://')
    model = convert_syncbn_model(model)
  model.cuda()
  if args.distributed:
    model = DistributedDataParallel(model, delay_allreduce=True)
  else:
    model = DataParallel(model)

  # define loss function (criterion)
  try:
    criterion = eval(args.loss)().cuda()
  except:
    raise RuntimeError(f'Loss {args.loss} not available!')

  start_epoch = 0
  best_score = 0
  best_epoch = 0

  # define scheduler
  try:
    scheduler = eval(args.scheduler)(model)
  except:
    raise RuntimeError(f'Scheduler {args.scheduler} not available!')

  # optionally resume from a checkpoint
  reset_epoch = True
  pretrained_file = None
  if args.model_file:
    reset_epoch = True
    pretrained_file = args.model_file
  if args.resume:
    reset_epoch = False
    pretrained_file = f'{model_out_dir}/{args.resume}'
  if pretrained_file and os.path.isfile(pretrained_file):
    # load checkpoint weights and update model and optimizer
    if args.can_print:
      log.write(f'>> Loading checkpoint:\n>> {pretrained_file}\n')

    checkpoint = torch.load(pretrained_file)
    if not reset_epoch:
      start_epoch = checkpoint['epoch']
      best_epoch = checkpoint['best_epoch']
      best_score = checkpoint['best_score']
    model.module.load_state_dict(checkpoint['state_dict'])
    if args.can_print:
      if reset_epoch:
        log.write(f'>>>> loaded checkpoint:\n>>>> {pretrained_file}\n')
      else:
        log.write(f'>>>> loaded checkpoint:\n>>>> {pretrained_file} (epoch {checkpoint["epoch"]:.2f})\n')
  else:
    if args.can_print:
      log.write(f'>> No checkpoint found at {pretrained_file}\n')

  # Data loading code
  train_transform = eval(f'train_multi_augment{args.aug_version}')
  train_split_file = f'{DATA_DIR}/split/{args.split_type}/random_train_cv0.csv'
  valid_split_file = f'{DATA_DIR}/split/{args.split_type}/random_valid_cv0.csv'
  train_dataset = RetrievalDataset(
    args,
    train_split_file,
    transform=train_transform,
    data_type='train',
  )
  valid_dataset = RetrievalDataset(
    args,
    valid_split_file,
    transform=None,
    data_type='valid',
  )
  if args.distributed:
    train_sampler = dist.DistributedSampler(train_dataset)
    valid_sampler = dist.DistributedSampler(valid_dataset)
  else:
    train_sampler = RandomSampler(train_dataset)
    valid_sampler = SequentialSampler(valid_dataset)
  train_loader = DataLoader(
    train_dataset,
    sampler=train_sampler,
    batch_size=args.batch_size,
    drop_last=True,
    num_workers=args.workers,
    pin_memory=True,
    collate_fn=image_collate,
  )
  valid_loader = DataLoader(
    valid_dataset,
    sampler=valid_sampler,
    batch_size=args.batch_size,
    drop_last=False,
    num_workers=args.workers,
    pin_memory=True,
    collate_fn=image_collate,
  )

  train(args, train_loader, valid_loader, model, criterion, scheduler, log, best_epoch, best_score, start_epoch, model_out_dir)

def train_generator(dataloader):
  while True:
    for it, (images, labels) in enumerate(dataloader, 0):
      if it == len(dataloader)-1:
        dataloader.dataset.on_epoch_end()
      yield images, labels

def reduce_tensor(tensor: torch.Tensor):
  rt = tensor.clone()
  dist.all_reduce(rt, op=dist.reduce_op.SUM)
  rt /= dist.get_world_size()
  return rt

def train(args, train_loader, valid_loader, model, criterion, scheduler, log, best_epoch, best_score, start_epoch, model_out_dir):
  if args.can_print:
    log.write('** start training here! **\n')
    log.write('\n')
    log.write('epoch   iter    rate    |   train_loss/acc    |   valid_loss/acc/map100    |  best_epoch/score |   min (valid_min)  \n')
    log.write('--------------------------------------------------------------------------------------------------------------------\n')

  model.train()
  last_cpt_epoch = start_epoch
  epoch = start_epoch
  max_iters = args.epochs * len(train_loader)
  global_step = int(start_epoch * len(train_loader))
  smooth_loss = 0.0
  smooth_acc = 0.0
  train_num = 0
  start = timer()
  for it, iter_data in enumerate(train_generator(train_loader)):
    if global_step >= max_iters:
      break

    optimizer, rate = scheduler.schedule(epoch, args.epochs, best_epoch=best_epoch)
    images, labels = iter_data
    images = Variable(images.cuda())
    labels = Variable(labels.cuda())
    outputs = model(images, label=labels)
    loss = criterion(outputs, labels, epoch=epoch)
    if type(outputs) == tuple:
      logits = outputs[0]
    else:
      logits = outputs
    probs = F.softmax(logits).data

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_acc = (probs.argmax(dim=1) == labels).float().mean()
    if args.distributed:
      train_loss = reduce_tensor(loss.data)
      train_acc = reduce_tensor(train_acc)
    else:
      train_loss = loss
    if smooth_loss == 0.0:
      smooth_loss = train_loss.item()
      smooth_acc = train_acc.item()
    smooth_loss = 0.99 * smooth_loss + 0.01 * train_loss.item()
    smooth_acc = 0.99 * smooth_acc + 0.01 * train_acc.item()
    global_step += 1
    if args.distributed:
      train_num += (args.batch_size * dist.get_world_size())
    else:
      train_num += args.batch_size
    epoch = start_epoch + train_num / len(train_loader.dataset)

    if args.can_print:
      print('\r%5.2f %6d  %0.6f  |   %6.4f   %6.4f   | ... ' % (epoch, global_step, rate, smooth_loss, smooth_acc), end='', flush=True)

    if int(epoch * 1000) > 0 and int(epoch * 1000) % int(args.save_every_epoch * 1000) == 0 \
      and epoch - last_cpt_epoch > args.save_every_epoch * 0.1:
      need_eval = True
      last_cpt_epoch = epoch
    else:
      need_eval = False

    model_file = f'{model_out_dir}/{epoch:.2f}.pth'
    if need_eval and args.can_print:
      save_model(model, model_file, best_score, best_epoch, epoch)

    if need_eval:
      model.eval()
      valid_start = timer()
      with torch.no_grad():
        valid_loss, valid_acc = validate(args, model, epoch, valid_loader, criterion)
        if args.can_print:
          valid_map100 = generate_score_by_model(
            model, img_size=(args.eval_img_size, args.eval_img_size),
            batch_size=1, preprocessing=args.preprocessing
          )
        else:
          valid_map100 = 0
      valid_end = timer()
      valid_run_time = (valid_end - valid_start) / 60.

      if valid_map100 > best_score:
        best_score = valid_map100
        best_epoch = epoch
        final_model_file = f'{model_out_dir}/final.pth'
        save_model(model, final_model_file, best_score, best_epoch, epoch)
        save_model(model, model_file, best_score, best_epoch, epoch)

      end = timer()
      time = (end - start) / 60
      start = timer()

      if args.can_print:
        print('\r', end='', flush=True)
        log.write(
          '%5.2f %6d  %0.6f  |   %6.4f   %6.4f   |  %6.4f   %6.4f   %6.4f  |   %5.2f   %6.4f  | %3.1f min (%3.1f min) \n' % \
          (epoch, global_step, rate, smooth_loss, smooth_acc, valid_loss, valid_acc, valid_map100, best_epoch, best_score, time, valid_run_time)
        )
      model.train()

def validate(args, model, epoch, valid_loader, criterion):
  valid_num = 0
  valid_loss = 0
  valid_acc = 0
  for it, iter_data in enumerate(valid_loader, 0):
    images, labels = iter_data

    images = Variable(images.cuda())
    labels = Variable(labels.cuda())

    outputs = model(images, label=labels)
    loss = criterion(outputs, labels, epoch=epoch)
    if type(outputs) == tuple:
      logits = outputs[0]
    else:
      logits = outputs
    probs = F.softmax(logits).data
    batch_size = len(images)
    valid_acc_batch = (probs.argmax(dim=1) == labels).float().mean()
    if args.distributed:
      valid_loss_batch = reduce_tensor(loss.data)
      valid_acc_batch = reduce_tensor(valid_acc_batch)
    else:
      valid_loss_batch = loss

    valid_num += batch_size
    valid_loss += batch_size * valid_loss_batch.item()
    valid_acc += batch_size * valid_acc_batch.item()

  valid_loss = valid_loss / valid_num
  valid_acc = valid_acc / valid_num
  return valid_loss, valid_acc

def save_model(model, model_file, best_score, best_epoch, epoch):
  if type(model) == DataParallel or type(model) == DistributedDataParallel:
    state_dict = model.module.state_dict()
  else:
    state_dict = model.state_dict()
  state_dict_to_save=dict()
  for key in state_dict.keys():
    state_dict_to_save[key] = state_dict[key].cpu()
  torch.save({
    'best_score': best_score,
    'state_dict': state_dict_to_save,
    'best_epoch': best_epoch,
    'epoch': epoch,
  }, model_file)

if __name__ == '__main__':
  main()
