# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
import json
import os

import cntk
import numpy as np

from Detection.CNN.FasterRCNN.lib.FasterRCNN_eval import compute_test_set_aps, FasterRCNN_Evaluator
from Detection.CNN.FasterRCNN.lib.FasterRCNN_train import prepare, train_faster_rcnn, store_eval_model_with_native_udf
from Detection.CNN.FasterRCNN.lib.utils.config_helpers import merge_configs
from Detection.CNN.FasterRCNN.lib.utils.plot_helpers import plot_test_set_results


def get_configuration():
    # load configs for detector, base network and data set
    from Detection.CNN.FasterRCNN.lib.FasterRCNN_config import cfg as detector_cfg
    # for VGG16 base model use:
    # from utils.configs.VGG16_config import cfg as network_cfg
    # for AlexNet base model use:
    from Detection.CNN.FasterRCNN.configs.AlexNet_config import cfg as network_cfg
    # for Pascal VOC 2007 data set use: from utils.configs.Pascal_config import cfg as dataset_cfg
    # for the Grocery data set use:     from utils.configs.Grocery_config import cfg as dataset_cfg
    # from utils.configs.Grocery_config import cfg as dataset_cfg
    # from utils.configs.LEGO_config import cfg as dataset_cfg
    # from utils.configs.LEGO_Synthetic_config import cfg as dataset_cfg
    # from Detection.CNN.FasterRCNN.configs.LEGO_Detection_Natural_config import cfg as dataset_cfg
    from Detection.CNN.FasterRCNN.configs.LEGO_Localization_Synthetic_config import cfg as dataset_cfg
    # from utils.configs.LEGO_Localization_Natural_config import cfg as dataset_cfg

    return merge_configs([detector_cfg, network_cfg, dataset_cfg])


def write_results(map, eval_results, cfg):
    results = dict(
        mAP=map,
        training_amount=cfg.DATA.NUM_TRAIN_IMAGES,
        testing_amount=cfg.DATA.NUM_TEST_IMAGES,
        source=cfg.DATA.MAP_FILE_PATH,
        epochs=cfg.CNTK.E2E_MAX_EPOCHS,
        backbone=cfg.MODEL.BASE_MODEL,
        AP_per_class=eval_results
    )
    if not os.path.exists(cfg.OUTPUT_PATH):
        os.makedirs(cfg.OUTPUT_PATH)
    with open(os.path.join(cfg.OUTPUT_PATH, 'results.json'), 'a') as fp:
        json.dump(results, fp)
    with open(os.path.join(cfg.OUTPUT_PATH, 'settings.json'), 'a') as fp:
        json.dump(cfg, fp)


# trains and evaluates a Fast R-CNN model.
if __name__ == '__main__':
    cfg = get_configuration()
    prepare(cfg, False)
    cntk.device.try_set_default_device(cntk.device.gpu(cfg.GPU_ID))

    # train and test
    trained_model = train_faster_rcnn(cfg)
    eval_results = compute_test_set_aps(trained_model, cfg)

    # write AP results to output
    for class_name in eval_results:
        print('AP for {:>15} = {:.4f}'.format(class_name, eval_results[class_name]))
    mAP = np.nanmean(list(eval_results.values()))
    print('Mean AP = {:.4f}'.format(mAP))
    write_results(mAP, eval_results, cfg)
    # Plot results on test set images
    if cfg.VISUALIZE_RESULTS:
        num_eval = min(cfg["DATA"].NUM_TEST_IMAGES, 100)
        results_folder = os.path.join(cfg.OUTPUT_PATH, cfg["DATA"].DATASET)
        evaluator = FasterRCNN_Evaluator(trained_model, cfg)
        plot_test_set_results(evaluator, num_eval, results_folder, cfg)

    if cfg.STORE_EVAL_MODEL_WITH_NATIVE_UDF:
        store_eval_model_with_native_udf(trained_model, cfg)
