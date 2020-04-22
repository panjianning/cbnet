# cbnet
implement cbnet with mmdetection

## Result

On COCO val2017

* cascade_rcnn + dual res2net101 + dcnv2 + fpn + 1x: 48.7 mAP
* cascade_rcnn + dual resnet_vd200 + dcnv2 + nonlocal + 2.5x softnms: 52.2 mAP (weights are transfered from [CBResNet200-vd-FPN-Nonlocal](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.2/docs/MODEL_ZOO_cn.md) in paddle detection)

## backbone config example
```python
backbone=dict(
    type='CBNet',
    num_repeat=2,
    use_act=False,
    connect_norm_eval=True,
    backbone_type='Res2Net',
    depth=101,
    scale=4,
    baseWidth=26,
    num_stages=4,
    out_indices=(0, 1, 2, 3),
    norm_cfg=dict(type='SyncBN', requires_grad=True),
    dcn=dict(type='DCNv2', deformable_groups=1, fallback_on_stride=False),
    stage_with_dcn=(False, True, True, True),
    frozen_stages=1,
    style='pytorch')
```
