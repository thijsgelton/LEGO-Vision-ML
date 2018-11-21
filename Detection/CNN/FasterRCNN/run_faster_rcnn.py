# Copyright (c) Microsoft. All rights reserved.

# Licensed under the MIT license. See LICENSE.md file in the project root
# for full license information.
# ==============================================================================
import json
import os
import parser
import sys
from azureml.core.run import Run
import cntk
import numpy as np

abs_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(abs_path, ".."))
from FasterRCNN_eval import compute_test_set_aps, FasterRCNN_Evaluator
from FasterRCNN_train import prepare, train_faster_rcnn, store_eval_model_with_native_udf
from utils.config_helpers import merge_configs
from utils.plot_helpers import plot_test_set_results


def get_configuration():
    from FasterRCNN_config import cfg as detector_cfg
    from configs.AlexNet_config import cfg as network_cfg
    from configs.LEGO_Localization_Natural_config import cfg as dataset_cfg

    return merge_configs([detector_cfg, network_cfg, dataset_cfg])


def write_results(map, eval_results, cfg, run):
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


def log(is_remote, metric, message, run=None):
    if is_remote:
        run.log(metric, message)
    else:
        print(metric, message)


# trains and evaluates a Fast R-CNN model.
if __name__ == '__main__':
    parser.add_argument('--data_set_name', required=True)
    parser.add_argument('--base_directory', required=True)
    parser.add_argument('--fast_mode', default=False)
    parser.add_argument('--is_remote', default=True)
    args = parser.parse_args()
    run = None
    if args.is_remote:
        run = Run.get_submitted_run()
    cfg = get_configuration()
    cfg['CNTK'].FAST_MODE = args.fast_mode
    cfg['DATA'].DATASET = args.data_set_name
    cfg['DATA'].MAP_FILE_PATH = args.base_directory
    cfg['DATA'].NUM_TRAIN_IMAGES = len(open(os.path.join(args.base_directory, "train_img_file.txt")).readlines())
    cfg['DATA'].NUM_TEST_IMAGES = len(open(os.path.join(args.base_directory, "test_img_file.txt")).readlines())

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
    write_results(mAP, eval_results, cfg, run)
    # Plot results on test set images
    if cfg.VISUALIZE_RESULTS:
        num_eval = min(cfg["DATA"].NUM_TEST_IMAGES, 100)
        results_folder = os.path.join(cfg.OUTPUT_PATH, cfg["DATA"].DATASET)
        evaluator = FasterRCNN_Evaluator(trained_model, cfg)
        plot_test_set_results(evaluator, num_eval, results_folder, cfg)

    if cfg.STORE_EVAL_MODEL_WITH_NATIVE_UDF:
        store_eval_model_with_native_udf(trained_model, cfg)
