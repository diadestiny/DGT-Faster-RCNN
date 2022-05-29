import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from derain.nets import DDGN_Depth_CFT, DDGN_Depth_CFT_Pred
from network_files.faster_rcnn_framework import Derain_FasterRCNN
import datetime
import torch

import transforms
from network_files import FasterRCNN, FastRCNNPredictor
from backbone import resnet50_fpn_backbone
from data_utils.dataset import VOCDataSet
from train_utils import GroupedBatchSampler, create_aspect_ratio_groups
from train_utils import train_eval_utils as utils


def create_model(num_classes):
    # 注意，这里的backbone默认使用的是FrozenBatchNorm2d，即不会去更新bn参数
    # 目的是为了防止batch_size太小导致效果更差(如果显存很小，建议使用默认的FrozenBatchNorm2d)
    # 如果GPU显存很大可以设置比较大的batch_size就可以将norm_layer设置为普通的BatchNorm2d
    # trainable_layers包括['layer4', 'layer3', 'layer2', 'layer1', 'conv1']， 5代表全部训练
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d,
                                     trainable_layers=3)
    # 训练自己数据集时不要修改这里的91，修改的是传入的num_classes参数
    model = FasterRCNN(backbone=backbone, num_classes=91)
    # 载入预训练模型权重
    # https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
    weights_dict = torch.load("./models/fasterrcnn_resnet50_fpn_coco.pth", map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(weights_dict, strict=False)
    if len(missing_keys) != 0 or len(unexpected_keys) != 0:
        print("missing_keys: ", missing_keys)
        print("unexpected_keys: ", unexpected_keys)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def main(parser_data):
    device = torch.device(parser_data.device if torch.cuda.is_available() else "cpu")
    print("Using {} device training.".format(device.type))

    # 用来保存coco_info的文件
    results_file = "results-lr0.005-keepon-kitti-depth-{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    data_transform = {
        "train": transforms.Compose([transforms.ToTensor(),
                                     transforms.RandomHorizontalFlip(0.5)]),
        "val": transforms.Compose([transforms.ToTensor()])
    }

    VOC_root = parser_data.data_path
    # check voc root
    if os.path.exists(os.path.join(VOC_root, "VOCdevkit")) is False:
        raise FileNotFoundError("VOCdevkit dose not in path:'{}'.".format(VOC_root))

    # load train data set
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> train.txt
    train_dataset = VOCDataSet(VOC_root, "2012", data_transform["train"], "train.txt",dataset_name=args.dataset)
    train_sampler = None

    # 是否按图片相似高宽比采样图片组成batch
    # 使用的话能够减小训练时所需GPU显存，默认使用
    if args.aspect_ratio_group_factor >= 0:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        # 统计所有图像高宽比例在bins区间中的位置索引
        group_ids = create_aspect_ratio_groups(train_dataset, k=args.aspect_ratio_group_factor)
        # 每个batch图片从同一高宽比例区间中取
        train_batch_sampler = GroupedBatchSampler(train_sampler, group_ids, args.batch_size)

    # 注意这里的collate_fn是自定义的，因为读取的数据包括image和targets，不能直接使用默认的方法合成batch
    batch_size = parser_data.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using %g dataloader workers' % nw)
    if train_sampler:
        # 如果按照图片高宽比采样图片，dataloader中需要使用batch_sampler
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_sampler=train_batch_sampler,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)
    else:
        train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                        batch_size=batch_size,
                                                        shuffle=True,
                                                        pin_memory=True,
                                                        num_workers=nw,
                                                        collate_fn=train_dataset.collate_fn)

    # load validation data set
    # VOCdevkit -> VOC2012 -> ImageSets -> Main -> val.txt
    val_dataset = VOCDataSet(VOC_root, "2012", data_transform["val"], "val.txt",dataset_name=args.dataset)
    val_data_set_loader = torch.utils.data.DataLoader(val_dataset,
                                                      batch_size=8,
                                                      shuffle=False,
                                                      pin_memory=True,
                                                      num_workers=nw,
                                                      collate_fn=val_dataset.collate_fn)

    # create models num_classes equal background + 20 classes
    faster_rcnn_model = create_model(num_classes=parser_data.num_classes + 1)
    de_rain_model = DDGN_Depth_CFT_Pred()
    de_rain_model.load_state_dict(torch.load("derain/ckpt/kitti_depth_cft_pred/iter_40000_loss1_0.01297_loss2_0.00000_lr_0.000000.pth"))
    myModel = Derain_FasterRCNN(FasterRCNN=faster_rcnn_model,DerainNet=de_rain_model)
    myModel.to(device)
    # define optimizer
    params = [p for p in myModel.parameters() if p.requires_grad]

    # 0.005 / 0.0001 / 0.000001
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)

    # learning rate scheduler step_size=3
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.33)

    # 如果指定了上次训练保存的权重文件地址，则接着上次结果接着训练
    if parser_data.resume != "":
        checkpoint = torch.load(parser_data.resume, map_location='cpu')
        myModel.FasterRCNN.load_state_dict(checkpoint['model'],strict=False)

        # weights_dict = torch.load("./models/fasterrcnn_resnet50_fpn_coco.pth", map_location='cpu')

        # depth_encoder_checkpoint = torch.load("./models/mobilenet_v2.pth", map_location='cpu')
        # faster_rcnn_model.depth_encoder.load_state_dict(depth_encoder_checkpoint,strict=True)
        # faster_rcnn_model.load_state_dict(checkpoint['model'],strict=False)

        depth_params = myModel.FasterRCNN.depth_encoder.state_dict()
        pretrained_dict = {k.replace('backbone.',''):v for k,v in checkpoint.items() if k.replace('backbone.','') in depth_params.keys()}
        # # print(pretrained_dict.keys())
        depth_params.update(pretrained_dict)
        myModel.FasterRCNN.depth_encoder.load_state_dict(depth_params,strict=True)

        # loaded_dict_enc = torch.load("./models/mono+stereo_640x192/encoder.pth", map_location=device)
        # filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in depth_params}
        # faster_rcnn_model.depth_encoder.load_state_dict(filtered_dict_enc)
        # faster_rcnn_model.depth_encoder.to(device)

        # optimizer.load_state_dict(checkpoint['optimizer'])
        # lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        # parser_data.start_epoch = checkpoint['epoch'] + 1
        print("the training process from epoch{}...".format(parser_data.start_epoch))

    train_loss = []
    learning_rate = []
    val_map = []
    for epoch in range(parser_data.start_epoch, parser_data.epochs):
        # train for one epoch, printing every 10 iterations
        # coco_info = utils.evaluate(myModel, val_data_set_loader, device=device)
        mean_loss, lr = utils.train_one_epoch(myModel, optimizer, train_data_loader,
                                              device=device, epoch=epoch,
                                              print_freq=50, warmup=True)
        train_loss.append(mean_loss.item())
        learning_rate.append(lr)

        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        coco_info = utils.evaluate(myModel, val_data_set_loader, device=device)

        # write into txt
        with open(results_file, "a") as f:
            # 写入的数据包括coco指标还有loss和learning rate
            result_info = [str(round(i, 4)) for i in coco_info + [mean_loss.item()]] + [str(round(lr, 6))]
            txt = "epoch:{} {}".format(epoch, '  '.join(result_info))
            f.write(txt + "\n")
        val_map.append(coco_info[1])  # pascal mAP

        # save weights
        save_files = {
            'model': faster_rcnn_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch}
        torch.save(save_files, "./save_weights/lr0.005-keepon-kitti-depth-models-{}.pth".format(epoch))
        # torch.save(de_rain_model.state_dict(), "./save_weights/de_rain-{}.pth".format(epoch))
        # torch.save(myModel.fea_similar_conv.state_dict(), "./save_weights/fea_similar_conv-{}.pth".format(epoch))
    # plot loss and lr curve
    if len(train_loss) != 0 and len(learning_rate) != 0:
        from plot_curve import plot_loss_and_lr
        plot_loss_and_lr(train_loss, learning_rate)

    # plot mAP curve
    if len(val_map) != 0:
        from plot_curve import plot_map
        plot_map(val_map)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)

    # 训练设备类型
    parser.add_argument('--device', default='cuda:0', help='device')
    # 训练数据集的根目录(VOCdevkit)
    parser.add_argument('--data-path', default='../', help='dataset')
    # 检测目标类别数(不包含背景)
    parser.add_argument('--dataset', default='kitti', type=str)
    parser.add_argument('--num-classes', default=7, type=int, help='num_classes')
    # 文件保存地址
    parser.add_argument('--output-dir', default='./save_weights', help='path where to save')
    # 若需要接着上次训练，则指定上次训练保存权重文件地址 kitti:resNetFpn-models-41.pth  kitti_20211222_39.pth
    parser.add_argument('--resume', default='./models/kitti-raw-models-2.pth', type=str, help='resume from checkpoint')
    # ./ old_models / 20211222_39_new_train.pth
    # save_weights
    # 指定接着从哪个epoch数开始训练
    parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
    # 训练的总epoch数
    parser.add_argument('--epochs', default=10, type=int, metavar='N',
                        help='number of total epochs to run')
    # 训练的batch size
    parser.add_argument('--batch_size', default=8, type=int, metavar='N',
                        help='batch size when training.')
    parser.add_argument('--aspect-ratio-group-factor', default=3, type=int)

    args = parser.parse_args()
    print(args)

    # 检查保存权重文件夹是否存在，不存在则创建
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)