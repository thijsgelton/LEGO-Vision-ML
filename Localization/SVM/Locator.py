from sklearn.externals import joblib

from Detection.SVM.SelectiveSearchEvaluator import SelectiveSearchEvaluator

if __name__ == "__main__":
    from Localization.SVM.config.localization_natural import *

    selective_search_config = dict(
        scale=600,
        min_size=400,
        sigma=0.9
    )

    evaluator = SelectiveSearchEvaluator(
        image_paths=image_paths,
        gt_bbox_paths=gt_bbox_paths,
        gt_labels_path=gt_labels_paths,
        classifier=joblib.load(classifier),
        label_lookup=label_lookup,
        plot_every_n_images=1,
        image_dimension=1024,
        # output_directory=output_directory,
        selective_search_config=selective_search_config
    )
    evaluator.eval()