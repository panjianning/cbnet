# cbnet
implement cbnet with [mmdetection](https://github.com/open-mmlab/mmdetection)

## Result

On COCO val2017

1. `cascade_rcnn + dual res2net101 + dcnv2 + fpn + 1x`: 48.8 mAP
* Single scale training and testing

2. `cascade_rcnn + dual resnet_vd200 + dcnv2 + nonlocal + fpn + 2.5x + soft-nms`: 52.2 mAP
* Multiscale-training, single-scale testing
* Weights are transfered from [CBResNet200-vd-FPN-Nonlocal](https://github.com/PaddlePaddle/PaddleDetection/blob/release/0.2/docs/MODEL_ZOO_cn.md) in paddle detection
* mAP with vanilla nms： 50.8

---
`non_local.py` is modified to adapt `NonLocal` module from paddledetection to mmdetection

## backbone config example
``` python
backbone=dict(
    type='CBNet',
    num_repeat=2,
    pretrained='/workspace/nas-data/checkpoint/imagenet/dual_res2net101_v1b_26w_4s-0812c246.pth',
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
To get the pretrained model on imagenet like `dual_xxx.pth`:
``` python
def make_pretrained_model(input_path, output_path, repeat_num=2):
    cp = torch.load(input_path)
    keys = list(cp['state_dict'].keys())
    for key in keys:
        for i in range(repeat_num):
            cp['state_dict']['cb{}.{}'.format(i + 1, key)] = cp['state_dict'][key]
        cp['state_dict'].pop(key)
    torch.save(cp, output_path)
```

## FAQ
### How to get the numpy-array-style weights in paddle detection
```python
from paddle import fluid
import os
from ppdet.core.workspace import load_config, create
import ppdet.utils.checkpoint as checkpoint
import numpy as np
from collections import OrderedDict
import torch


def load_var_dict(cfg_path, weight_path, device_id=-1):
    place = fluid.CUDAPlace(0) if device_id >= 0 else fluid.CPUPlace()
    exe = fluid.Executor(place)

    cfg = load_config(cfg_path)
    main_arch = cfg.architecture
    model = create(main_arch)
    startup_prog = fluid.Program()
    infer_prog = fluid.Program()
    with fluid.program_guard(infer_prog, startup_prog):
        with fluid.unique_name.guard():
            inputs_def = cfg['TestReader']['inputs_def']
            inputs_def['iterable'] = True
            feed_vars, loader = model.build_inputs(**inputs_def)
            _ = model.test(feed_vars)
    checkpoint.load_params(exe, infer_prog, weight_path)
    var_dict = {}
    for var in startup_prog.list_vars():
        fluid_tensor = fluid.global_scope().find_var(var.name).get_tensor()
        var_dict[var.name] = np.array(fluid_tensor)

    return var_dict
```
