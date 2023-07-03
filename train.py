
import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.utils.data
import os
import shutil
import numpy as np
from loss import Norm1
import logging
from tensorboardX import SummaryWriter

from FullNet import FullNet, FCN_pooling
import utils
from data_folder import DataFolder
from options import Options
from loss import LossVariance
from attention_unet import U_Net,AttU_Net

def main():
    global opt, best_loss, num_iter, tb_writer, logger, logger_results


    opt = Options(isTrain=True)
    opt.parse()
    opt.save_options()

    tb_writer = SummaryWriter('{:s}/tb_logs'.format(opt.train['save_dir']))
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in opt.train['gpu'])

    # set up logger
    logger, logger_results = setup_logging(opt)
    opt.print_options(logger)

    # ----- create model ----- #
    if opt.model['name']=='FullNet':
        model = FullNet(opt.model['in_c'], opt.model['out_c'], n_layers=opt.model['n_layers'],
                        growth_rate=opt.model['growth_rate'], drop_rate=opt.model['drop_rate'],
                        dilations=opt.model['dilations'], is_hybrid=opt.model['is_hybrid'],
                        compress_ratio=opt.model['compress_ratio'], layer_type=opt.model['layer_type'])
    elif opt.model['name']=='Unet':
        model = U_Net(opt.model['in_c'], opt.model['out_c'])
    elif opt.model['name'] == 'Att_Unet':
        model = AttU_Net(opt.model['in_c'], opt.model['out_c'])
    elif opt.model['name']=='FCN_pooling':
        model = FCN_pooling(opt.model['in_c'], opt.model['out_c'], n_layers=opt.model['n_layers'],
                        growth_rate=opt.model['growth_rate'], drop_rate=opt.model['drop_rate'],
                        dilations=opt.model['dilations'],
                        compress_ratio=opt.model['compress_ratio'], layer_type=opt.model['layer_type'])
    else:
        raise ('model must in FullNet and Unet')
    model = nn.DataParallel(model)
    model = model.cuda()
    torch.backends.cudnn.benchmark = True

    # ----- define optimizer ----- #
    optimizer = torch.optim.Adam(model.parameters(), opt.train['lr'], betas=(0.9, 0.99),
                                 weight_decay=opt.train['weight_decay'])

    # ----- define criterion ----- #
    criterion = torch.nn.NLLLoss(reduction='none').cuda()
    nuclei_criterion=torch.nn.L1Loss(reduction='none').cuda()
    Norm_criterion=Norm1().cuda()
    hinge_criterion=torch.nn.HingeEmbeddingLoss()
    # criterion = torch.nn.CrossEntropyLoss(reduction='none').cuda()
    # criterion=BEC_Jaccard_Loss()
    # criterion=FocalLoss(gamma=2,alpha=[5,5,1])

    if opt.train['alpha'] > 0:
        logger.info('=> Using variance term in loss...')
        global criterion_var
        criterion_var = LossVariance()
    # ----- load data ----- #
    dsets = {}
    for x in ['train', 'val']:
        img_dir = opt.train['data_dir']+'/{:s}/h5'.format(x)
        if x=='train':
            dsets[x] = DataFolder( data_dir=img_dir,image_size=opt.train['input_size'],
                                resize_ratio_list= np.arange(0.9, 1.0,1.2),
                                is_training=True)
        else:
            dsets[x] = DataFolder(data_dir=img_dir,image_size=opt.train['input_size'],
                                  resize_ratio_list=np.arange(0.9, 1.0, 1.2),
                                   is_training=False)
    train_loader = DataLoader(dsets['train'], batch_size=opt.train['batch_size'], shuffle=True,
                              num_workers=opt.train['workers'])


    # ----- optionally load from a checkpoint for validation or resuming training ----- #
    if opt.train['checkpoint']:
        if os.path.isfile(opt.train['checkpoint']):
            logger.info("=> loading checkpoint '{}'".format(opt.train['checkpoint']))
            checkpoint = torch.load(opt.train['checkpoint'])
            opt.train['start_epoch'] = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            logger.info("=> loaded checkpoint '{}' (epoch {})"
                        .format(opt.train['checkpoint'], checkpoint['epoch']))
        else:
            logger.info("=> no checkpoint found at '{}'".format(opt.train['checkpoint']))

    # ----- training and validation ----- #
    for epoch in range(opt.train['start_epoch'], opt.train['num_epochs']):
        # train for one epoch or len(train_loader) iterations
        if epoch%opt.train['decay_epoch']==0:
            for p in optimizer.param_groups:
                p['lr']*=0.1
        logger.info('Epoch: [{:d}/{:d}]'.format(epoch+1, opt.train['num_epochs']))

        train_loss= train(train_loader, model, optimizer, criterion, nuclei_criterion,Norm_criterion,hinge_criterion,epoch)

        #loss.item(), pixel_accu, iou
        # evaluate on validation set


        cp_flag = (epoch+1) % opt.train['checkpoint_freq'] == 0

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, epoch, opt.train['save_dir'], cp_flag)
        # save the training results to txt files
        # logger_results.info('{:d}\t{:.4f}\t{:.4f}'
        #                     .format(epoch+1, train_loss,val_loss))
        # tensorboard logs
        # tb_writer.add_scalars('epoch_losses',
        #                       {'train_loss': train_loss,  'val_loss': val_loss}, epoch)

    tb_writer.close()

def train(train_loader, model, optimizer, criterion,nuclei_criterion,Norm_criterion,hinge_criterion, epoch):
    # list to store the average loss and iou for this epoch
    results = utils.AverageMeter(5)

    # switch to train mode
    model.train()

    for i, sample in enumerate(train_loader):
        # if i >2:continue
        input, target,point = sample
        point_var=point.cuda()
        input_var = input.cuda()
        target_var = target.cuda()
        # compute output
        d_cell,d_membrane,output = model(x=input_var)
        log_prob_maps = F.log_softmax(output, dim=1)
        #membrane_loss
        #pass
        #nuclei_loss
        n_loss=nuclei_criterion(d_cell,point_var)
        n_loss=torch.mean(n_loss)
        target_var_arg = torch.argmax(target_var, dim=1)
        loss_pred = criterion(log_prob_maps, target_var_arg)
        loss_pred=torch.mean(loss_pred)
        loss_norm=Norm_criterion(d_membrane)
        hinge_target=-1.0*torch.ones_like(d_membrane)
        hinge_loss=hinge_criterion(d_membrane,hinge_target)
        loss =n_loss+loss_pred+loss_norm+hinge_loss
        result = [loss.item(),loss_pred.item(),n_loss.item(),loss_norm.item(),hinge_loss.item()]
        results.update(result, input.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # return results.avg
        del input_var, output, target_var, loss

        if i % opt.train['log_interval'] == 0:
            logger.info('\tIteration: [{:d}/{:d}]'
                        '\tLoss {r[0]:.4f}'
                        '\tpred Loss {r[1]:.4f}'
                        '\tNuclei Loss {r[2]:.4f}'
                        '\tNorm Loss {r[3]:.4f}'
                        '\thinge Loss {r[4]:.4f}'
                        .format(i, len(train_loader), r=results.avg))

    logger.info('\t=> Train Avg: Loss {r[0]:.4f}'
               .format(epoch, opt.train['num_epochs'], r=results.avg))

    return results.avg





def save_checkpoint(state, epoch, save_dir, cp_flag):
    cp_dir = '{:s}/checkpoints'.format(save_dir)
    if not os.path.exists(cp_dir):
        os.mkdir(cp_dir)
    filename = '{:s}/checkpoint.pth.tar'.format(cp_dir)
    torch.save(state, filename)
    if cp_flag:
        shutil.copyfile(filename, '{:s}/checkpoint_{:d}.pth.tar'.format(cp_dir, epoch+1))


def setup_logging(opt):
    mode = 'a' if opt.train['checkpoint'] else 'w'

    # create logger for training information
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.DEBUG)
    # create console handler and file handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    file_handler = logging.FileHandler('{:s}/train.log'.format(opt.train['save_dir']), mode=mode)
    file_handler.setLevel(logging.DEBUG)
    # create formatter
    formatter = logging.Formatter('%(asctime)s\t%(message)s', datefmt='%Y-%m-%d %I:%M')
    # add formatter to handlers
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    # add handlers to logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # create logger for epoch results
    logger_results = logging.getLogger('results')
    logger_results.setLevel(logging.DEBUG)
    file_handler2 = logging.FileHandler('{:s}/epoch_results.txt'.format(opt.train['save_dir']), mode=mode)
    file_handler2.setFormatter(logging.Formatter('%(message)s'))
    logger_results.addHandler(file_handler2)

    logger.info('***** Training starts *****')
    logger.info('save directory: {:s}'.format(opt.train['save_dir']))
    if mode == 'w':
        logger_results.info('epoch\ttrain_loss\ttrain_loss_CE\ttrain_loss_var\ttrain_acc\ttrain_iou\t'
                            'val_loss\tval_acc\tval_iou')

    return logger, logger_results


if __name__ == '__main__':
    main()
