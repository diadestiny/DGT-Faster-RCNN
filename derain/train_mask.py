import datetime
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '2'
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

import triple_transforms
from nets import depth_predciton, basic, basic_NL, DGNLNet, MyNet, DDGN_Depth
from config import dataset_dir, dataset_dir
from dataset3 import ImageFolder
from misc import AvgMeter, check_mkdir
from torch.utils.tensorboard import SummaryWriter
# torch.cuda.set_device(1)
cudnn.benchmark = True

ckpt_path = './ckpt'
exp_name = 'new_kitti_mask'

args = {
    'iter_num': 40000,
    'train_batch_size': 8,
    'last_iter': 0,
    'lr': 5e-4,  # 5e-4
    'lr_decay': 0.9,
    'weight_decay': 0,
    'momentum': 0.9,
    'resume_snapshot': '',
    'val_freq': 500,
    'img_size_h': 512,
    'img_size_w': 1024,
    # 'crop_size': 512,
    'snapshot_epochs': 500
}

transform = transforms.Compose([
    transforms.ToTensor()
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

to_pil = transforms.ToPILImage()

triple_transform = triple_transforms.Compose([
    triple_transforms.Resize((args['img_size_h'], args['img_size_w'])),
    # triple_transforms.RandomCrop(args['crop_size']),
    triple_transforms.RandomHorizontallyFlip()
])

train_set = ImageFolder(dataset_dir, transform=transform, target_transform=transform,
                        triple_transform=triple_transform, is_train=True, train_str="mask")
train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=8, shuffle=True)
test1_set = ImageFolder(dataset_dir, transform=transform, target_transform=transform, is_train=False, train_str="mask")
test1_loader = DataLoader(test1_set, batch_size=8)

criterion = nn.L1Loss()
criterion_depth = nn.L1Loss()
log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')


def main():
    net = DDGN_Depth().cuda().train()
    tb_writer = SummaryWriter(log_dir="runs/mask_experiment")
    # init_img = torch.zeros((1, 3, args['img_size_h'], args['img_size_w']))
    # init_img = init_img.cuda()
    # tb_writer.add_graph(net, init_img)
    optimizer = optim.Adam([
        {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias' and param.requires_grad],
         'lr': 2 * args['lr']},
        {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias' and param.requires_grad],
         'lr': args['lr'], 'weight_decay': args['weight_decay']}
    ])

    if len(args['resume_snapshot']) > 0:
        print('training resumes from \'%s\'' % args['resume_snapshot'])
        net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['resume_snapshot'] + '.pth')))
        optimizer.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['resume_snapshot'] + '_optim.pth')))
        optimizer.param_groups[0]['lr'] = 2 * args['lr']
        optimizer.param_groups[1]['lr'] = args['lr']

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')
    train(net, optimizer,tb_writer)


def train(net, optimizer,tb_writer):
    curr_iter = args['last_iter']
    while True:
        train_loss_record = AvgMeter()
        train_net_loss_record = AvgMeter()
        train_depth_loss_record = AvgMeter()

        k = 0
        for i, data in enumerate(train_loader):
            optimizer.param_groups[0]['lr'] = 2 * args['lr'] * (1 - float(curr_iter) / args['iter_num']) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['lr'] * (1 - float(curr_iter) / args['iter_num']) ** args['lr_decay']

            inputs, gts, dps = data
            batch_size = inputs.size(0)
            inputs = Variable(inputs).cuda()
            gts = Variable(gts).cuda()
            dps = Variable(dps).cuda()
            mask_dps = ((gts-inputs)[:,0,:,:]>0).type(torch.cuda.FloatTensor)
            mask_dps = mask_dps.unsqueeze(1)
            # mask_dps = dps[:, 1, :, :].unsqueeze(1)

            optimizer.zero_grad()
            result = net(inputs,mask_dps)
            loss_net = criterion(result, gts)
            # loss_depth = criterion_depth(depth_pred, dps)
            loss = loss_net
            loss.backward()
            optimizer.step()

            # for n, p in net.named_parameters():
            #     if n[-5:] == 'alpha':
            #         print(p.grad.data)
            #         print(p.data)

            train_loss_record.update(loss.data, batch_size)
            train_net_loss_record.update(loss_net.data, batch_size)
            # train_depth_loss_record.update(loss_depth.data, batch_size)

            curr_iter += 1

            log = '[iter %d], [train loss %.5f], [lr %.13f], [loss_net %.5f], [loss_mask %.5f]' % \
                  (curr_iter, train_loss_record.avg, optimizer.param_groups[1]['lr'],
                   train_net_loss_record.avg, train_depth_loss_record.avg)

            tb_writer.add_scalar("train_loss", train_loss_record.avg, curr_iter)
            tb_writer.add_scalar("lr", optimizer.param_groups[1]['lr'], curr_iter)

            # print(curr_iter)
            if curr_iter % 10 == 0:
                print(i, log)
                open(log_path, 'a').write(log + '\n')

            if (curr_iter + 1) % args['val_freq'] == 0:
                validate(net, curr_iter, optimizer, tb_writer)

            if (curr_iter + 1) % args['snapshot_epochs'] == 0:
                # print("save_model")
                # torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, ('%d.pth' % (curr_iter + 1))))
                # torch.save(optimizer.state_dict(),
                #            os.path.join(ckpt_path, exp_name, ('%d_optim.pth' % (curr_iter + 1))))
                pass

            if curr_iter > args['iter_num']:
                return


def validate(net, curr_iter, optimizer, tb_writer):
    print('validating...')
    net.eval()

    loss_record1, loss_record2 = AvgMeter(), AvgMeter()
    iter_num1 = len(test1_loader)

    with torch.no_grad():
        for i, data in enumerate(test1_loader):
            inputs, gts, dps = data
            inputs = Variable(inputs).cuda()
            gts = Variable(gts).cuda()
            dps = Variable(dps).cuda()

            mask_dps = ((gts - inputs)[:, 0, :, :] > 0).type(torch.cuda.FloatTensor)
            mask_dps = mask_dps.unsqueeze(1)
            res = net(inputs,mask_dps)

            loss = criterion(res, gts)
            loss_record1.update(loss.data, inputs.size(0))
            if i % 100 == 0:
                print('processed test1 %d / %d' % (i + 1, iter_num1))

    snapshot_name = 'iter_%d_loss1_%.5f_loss2_%.5f_lr_%.6f' % (curr_iter + 1, loss_record1.avg, loss_record2.avg,
                                                               optimizer.param_groups[1]['lr'])
    val_log = '[validate]: [iter %d], [loss1 %.5f], [loss2 %.5f]' % (curr_iter + 1, loss_record1.avg, loss_record2.avg)
    print(val_log)
    tb_writer.add_scalar("val_loss", loss_record1.avg, curr_iter + 1)

    torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, snapshot_name + '.pth'))
    torch.save(optimizer.state_dict(), os.path.join(ckpt_path, exp_name, snapshot_name + '_optim.pth'))

    net.train()


if __name__ == '__main__':
    main()
