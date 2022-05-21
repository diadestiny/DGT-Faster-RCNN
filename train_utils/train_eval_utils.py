import json
import math
import sys
import time

import torch

from validation import summarize
from .coco_utils import get_coco_api_from_dataset
from .coco_eval import CocoEvaluator
import train_utils.distributed_utils as utils


def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq=50, warmup=False):
    model.to(device)
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0 and warmup is True:  # 当训练第一轮（epoch=0）时，启用warmup训练方式，可理解为热身训练
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    mloss = torch.zeros(1).to(device)  # mean losses
    print(optimizer.param_groups[0]["lr"])
    enable_amp = True if "cuda" in device.type else False
    for i, [images, targets,rains,depths] in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # images = list(image.to(device) for image in images)
        rains = list(rain.to(device) for rain in rains)
        # depths = list(depth.to(device) for depth in depths)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 混合精度训练上下文管理器，如果在CPU环境中不起任何作用
        with torch.cuda.amp.autocast(enabled=enable_amp):
            loss_dict = model(rains=rains,targets=targets)
            # loss_dict = model(images, targets=targets)
            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purpose
            loss_dict_reduced = utils.reduce_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())

            loss_value = losses_reduced.item()
            # 记录训练损失
            mloss = (mloss * i + loss_value) / (i + 1)  # update mean losses

            if not math.isfinite(loss_value):  # 当计算的损失为无穷大时停止训练
                print("Loss is {}, stopping training".format(loss_value))
                print(loss_dict_reduced)
                sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:  # 第一轮使用warmup训练方式
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)

    return mloss, now_lr


@torch.no_grad()
def evaluate(model, data_loader, device):

    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test: "

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets,rains,depths in metric_logger.log_every(data_loader, 100, header):
        # images = list(img.to(device) for img in images)
        rains = list(rain.to(device) for rain in rains)
        # depths = list(dp.to(device) for dp in depths)
        # 当使用CPU时，跳过GPU相关指令
        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        # outputs = model(images, targets=targets)
        imgs,outputs = model(rains=rains)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    coco_info = coco_evaluator.coco_eval[iou_types[0]].stats.tolist()  # numpy to list

    # 计算IOU=0.5
    json_file = open('pascal_voc_classes_cityscapes.json', 'r')
    class_dict = json.load(json_file)
    json_file.close()
    category_index = {v: k for k, v in class_dict.items()}
    voc_map_info_list = []
    coco_eval = coco_evaluator.coco_eval["bbox"]

    # for i in range(len(category_index)):
    #     stats, _ = summarize(coco_eval, catId=i)
    #     voc_map_info_list.append(" {:15}: {}".format(category_index[i + 1], stats[1]))

    for i in range(len(category_index)):
        if i == 5:
            continue
        elif i == 6:
            stats, _ = summarize(coco_eval, catId=5)
            voc_map_info_list.append(" {:15}: {}".format(category_index[6 + 1], stats[1]))
        elif i == 7:
            stats, _ = summarize(coco_eval, catId=6)
            voc_map_info_list.append(" {:15}: {}".format(category_index[7 + 1], stats[1]))
        else:
            stats, _ = summarize(coco_eval, catId=i)
            voc_map_info_list.append(" {:15}: {}".format(category_index[i + 1], stats[1]))
        # stats, _ = summarize(coco_eval, catId=i)
        # voc_map_info_list.append(" {:15}: {}".format(category_index[i + 1], stats[1]))
    print_voc = "\n".join(voc_map_info_list)
    print(print_voc)

    return coco_info


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    return iou_types
