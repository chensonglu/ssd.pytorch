MEANS = (104, 117, 123)


def change_cfg_for_ssd512(cfg):
    cfg['min_dim'] = 512
    cfg['steps'] = [8, 16, 32, 64, 128, 256, 512]
    cfg['feature_maps'] = [64, 32, 16, 8, 4, 2, 1]
    cfg['min_sizes'] = [35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8]
    cfg['max_sizes'] = [76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6]
    cfg['aspect_ratios']= [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]]
    return cfg


def change_cfg_for_ssd512_mobilenet(cfg):
    cfg['min_dim'] = 512
    cfg['steps'] = [16, 32, 64, 128, 256, 512]
    cfg['feature_maps'] = [32, 16, 8, 4, 2, 1]
    cfg['min_sizes'] = [35.84, 76.8, 168.96, 261.12, 353.28, 445.44]
    cfg['max_sizes'] = [76.8, 168.96, 261.12, 353.28, 445.44, 537.6]
    cfg['aspect_ratios']= [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]
    return cfg

voc = {
    'num_classes': 21,
    'lr_steps': (80000, 100000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30.0, 60.0, 111.0, 162.0, 213.0, 264.0],
    'max_sizes': [60.0, 111.0, 162.0, 213.0, 264.0, 315.0],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'VOC',
}

coco = {
    'num_classes': 81,
    'lr_steps': (280000, 360000, 400000),
    'max_iter': 400000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [21, 45, 99, 153, 207, 261],
    'max_sizes': [45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'COCO',
}

car_carplate = {
    'num_classes': 3,
    'lr_steps': (20000, 40000, 60000),
    'max_iter': 60000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30.0, 60.0, 111.0, 162.0, 213.0, 264.0],
    'max_sizes': [60.0, 111.0, 162.0, 213.0, 264.0, 315.0],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'CAR_CARPLATE',
}

car = {
    'num_classes': 2,
    'lr_steps': (20000, 40000, 60000),
    'max_iter': 60000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30.0, 60.0, 111.0, 162.0, 213.0, 264.0],
    'max_sizes': [60.0, 111.0, 162.0, 213.0, 264.0, 315.0],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'CAR',
}

carplate = {
    'BB_losstype': 'Ciou',
    'FV_losstype': 'Ciou',
    'num_classes': 2,
    'lr_steps': (20000, 40000, 60000),
    'max_iter': 60000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'min_dim': 300,
    'steps': [8, 16, 32, 64, 100, 300],
    'min_sizes': [30.0, 60.0, 111.0, 162.0, 213.0, 264.0],
    'max_sizes': [60.0, 111.0, 162.0, 213.0, 264.0, 315.0],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'CARPLATE',
}

mobilenet_carplate = {
    'num_classes': 2,
    'lr_steps': (20000, 40000, 60000),
    'max_iter': 60000,
    'feature_maps': [19, 10, 5, 3, 2, 1],
    'min_dim': 300,
    'steps': [16, 32, 64, 100, 150, 300],
    'min_sizes': [30.0, 60.0, 111.0, 162.0, 213.0, 264.0],
    'max_sizes': [60.0, 111.0, 162.0, 213.0, 264.0, 315.0],
    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'MOBILENET_CARPLATE',
}

two_stage_end2end = {
    'num_classes': 2,
    'lr_steps': (20000, 40000, 60000),
    'max_iter': 60000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'feature_maps_2': [56, 28, 14],
    'min_dim': 300,
    'min_dim_2': 56,
    'expand_num': 3,
    'steps': [8, 16, 32, 64, 100, 300],
    'steps_2': [1, 2, 4],
    'min_sizes': [30.0, 60.0, 111.0, 162.0, 213.0, 264.0],
    'max_sizes': [60.0, 111.0, 162.0, 213.0, 264.0, 315.0],
    'min_sizes_2': [5.6, 11.2, 50.4],
    'max_sizes_2': [11.2, 50.4, 89.6],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'aspect_ratios_2': [[2, 3], [2, 3], [2, 3]],
    'variance': [0.1, 0.2],
    'clip': True,
    'name': 'TWO_STAGE_END2END',
}

text = {
    'num_class':2,
    'min_dim':300,  #暂定输入图像大小为固定的300*300，同ssd一样
    'lr_steps': (40000, 80000, 120000),
    'max_iter': 120000,
    'feature_maps': [38, 19, 10, 5, 3, 1],
    'steps':[8, 16, 32, 64, 100, 300],
    'min_sizes':[30.0, 60.0, 114.0, 168.0, 222.0, 276.0],
    'max_sizes':[60.0, 114.0, 168.0, 222.0, 276.0, 330.0],
    'aspect_ratios':[2, 3, 5, 7, 10],
    'variance':[0.1, 0.1, 0.2, 0.2],
    'clip': True,
    'name': 'TEXT'
}
