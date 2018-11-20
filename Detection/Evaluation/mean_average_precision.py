import glob
import json
import operator
import os
import shutil
import numpy as np
import re
import matplotlib.pyplot as plt
from collections import Counter
from Helpers import utils


class MeanAveragePrecision:

    def __init__(self, base_directory, iou_threshold=0.5, show_animation=False, plot=True, quiet=False,
                 ignore_classes=None):
        self.base_directory = base_directory
        self.iou_threshold = iou_threshold
        self.show_animation = show_animation
        self.plot = plot
        self.quiet = quiet
        self.ignore_classes = ignore_classes if ignore_classes else []
        self.ground_truth_classes_counter = Counter()
        self.temporary_files_path = "tmp_files"
        self.results_files_path = os.path.join(self.base_directory, "results")
        self.coordinates = re.compile(r"\d+")
        self.results_file = os.path.join(self.results_files_path, "results.txt")
        self.predictions_per_class_counter = Counter()
        self.true_positives = Counter()

    def evaluate(self):
        self.create_folder_structure()
        self.convert_and_store_ground_truth_as_json()
        self.convert_and_store_predicted_as_json()
        mean_average_precision, average_precision_per_class = self.calculate_average_precision()
        self.clean_up_temporary_files()
        self.count_total_predictions()
        self.plot_ground_truth_info()
        self.write_ground_truth_info_to_results()
        self.plot_predicted_objects_info()
        self.write_predicted_objects_info_to_results()
        self.plot_mean_average_precision(mean_average_precision, average_precision_per_class)

    def create_folder_structure(self):
        if not os.path.exists(self.temporary_files_path):
            os.makedirs(self.temporary_files_path)

        if os.path.exists(self.results_files_path):
            shutil.rmtree(self.results_files_path)
        os.makedirs(self.results_files_path, exist_ok=True)

        if self.plot:
            os.makedirs(self.results_files_path + "/classes")
        if self.show_animation:
            os.makedirs(self.results_files_path + "/images")
            os.makedirs(self.results_files_path + "/images/single_predictions")

    def convert_and_store_ground_truth_as_json(self):
        ground_truth_files = glob.glob(os.path.join(self.base_directory, "ground-truth", "*.txt"))
        if len(ground_truth_files) == 0:
            raise FileNotFoundError("No ground-truth text files found.")

        for text_file in ground_truth_files:
            bounding_boxes = []
            file_id = os.path.basename(os.path.normpath(text_file.split(".txt", 1)[0]))
            if not self.corresponding_prediction_file_present(file_id):
                raise FileNotFoundError("No corresponding prediction file found.")
            for line in self.lines_to_list(text_file):
                class_name, x1, y1, x2, y2 = line.split()
                if class_name in self.ignore_classes:
                    continue
                bounding_boxes.append(dict(class_name=class_name, bbox=" ".join([x1, y1, x2, y2]), used=False))
                self.ground_truth_classes_counter.update({class_name: 1})

            with open(os.path.join(self.temporary_files_path, file_id + "_ground_truth.json"), 'w') as outfile:
                json.dump(bounding_boxes, outfile)

    def corresponding_prediction_file_present(self, file_id):
        return os.path.exists(os.path.join(self.base_directory, "predicted", file_id + ".txt"))

    @staticmethod
    def lines_to_list(path):
        with open(path) as f:
            content = f.readlines()
        content = [x.strip() for x in content]
        return content

    def convert_and_store_predicted_as_json(self):
        predicted_files = glob.glob(os.path.join(self.base_directory, "predicted", "*.txt"))
        if len(predicted_files) == 0:
            raise FileNotFoundError("No predicted text files found.")

        for index, class_name in enumerate(self.ground_truth_classes_counter.keys()):
            bounding_boxes = []
            for text_file in predicted_files:
                file_id = os.path.basename(os.path.normpath(text_file.split(".txt", 1)[0]))
                if not self.corresponding_ground_truth_file_present(file_id):
                    raise FileNotFoundError("No corresponding prediction file found.")
                for line in self.lines_to_list(text_file):
                    temp_class_name, confidence, x1, y1, x2, y2 = line.split()
                    if temp_class_name == class_name:
                        bounding_boxes.append(dict(confidence=confidence, bbox=" ".join([x1, y1, x2, y2]),
                                                   file_id=file_id))
            bounding_boxes.sort(key=lambda bbox: bbox['confidence'], reverse=True)
            with open(os.path.join(self.temporary_files_path, class_name + "_predictions.json"), 'w') as outfile:
                json.dump(bounding_boxes, outfile)

    def corresponding_ground_truth_file_present(self, file_id):
        return os.path.exists(os.path.join(self.base_directory, "ground-truth", file_id + ".txt"))

    def calculate_average_precision(self):
        sum_average_precision = 0.0
        average_precision_per_class = dict()
        with open(self.results_file, "w") as result_file:
            result_file.write("# AP and precision/recall per class\n")
            for index, class_name in enumerate(sorted(self.ground_truth_classes_counter.keys())):
                average_precision, recall, precision = self.compute_average_precision_for_class(class_name)
                title = "{}: {}".format(class_name, average_precision)
                self.plot_area_under_curve_for_class(title, recall, precision, class_name)
                sum_average_precision += average_precision
                average_precision_per_class[class_name] = average_precision
                self.write_to_average_precision_to_results(title, precision, recall, result_file)
            mean_average_precision = sum_average_precision / len(self.ground_truth_classes_counter.keys())
            self.write_mean_average_precision_to_results(mean_average_precision, result_file)
        return mean_average_precision, average_precision_per_class

    def compute_average_precision_for_class(self, class_name):
        predictions = self.load_predictions(class_name=class_name)
        tp = np.zeros((len(predictions))).tolist()
        fp = np.zeros((len(predictions))).tolist()
        for index, prediction in enumerate(predictions):
            ground_truths, ground_truths_file = self.load_ground_truths(file_id=prediction['file_id'])
            highest_overlap = -1
            matching_ground_truth = -1
            bbox_prediction = self.extract_coordinates(prediction["bbox"])
            for ground_truth in ground_truths:
                highest_overlap, matching_ground_truth = self.find_most_matching_ground_truth(class_name,
                                                                                              matching_ground_truth,
                                                                                              ground_truth,
                                                                                              highest_overlap,
                                                                                              bbox_prediction)

            if highest_overlap >= self.iou_threshold:
                if not bool(matching_ground_truth["used"]):
                    tp[index] = 1
                    matching_ground_truth["used"] = True
                    self.true_positives.update({class_name: 1})
                    self.update_ground_truth_file(ground_truths_file, ground_truths)
                else:
                    fp[index] = 1
            else:
                fp[index] = 1
        return self.compute_recall_precision(fp, tp, class_name)

    def find_most_matching_ground_truth(self, class_name, matching_ground_truth, ground_truth, highest_overlap,
                                        bbox_prediction):
        if ground_truth["class_name"] == class_name:
            overlap = utils.calculate_intersection_over_union(self.extract_coordinates(ground_truth["bbox"]),
                                                              bbox_prediction)
            if overlap > highest_overlap:
                return overlap, ground_truth
        return highest_overlap, matching_ground_truth

    def extract_coordinates(self, bbox):
        return np.array(list(map(lambda coordinate: float(coordinate), re.findall(self.coordinates, bbox))))

    def load_predictions(self, class_name):
        predictions_file = os.path.join(self.temporary_files_path, class_name + "_predictions.json")
        return json.load(open(predictions_file))

    def load_ground_truths(self, file_id):
        ground_truths_file = os.path.join(self.temporary_files_path, file_id + "_ground_truth.json")
        return json.load(open(ground_truths_file)), ground_truths_file

    def update_ground_truth_file(self, ground_truths_file, ground_truths):
        with open(ground_truths_file, 'w') as gt:
            gt.write(json.dumps(ground_truths))

    def plot_area_under_curve_for_class(self, title, recall, precision, class_name):
        plt.plot(recall, precision, '-o')
        area_under_curve_x = recall[:-1] + [recall[-2]] + [recall[-1]]
        area_under_curve_y = precision[:-1] + [0.0] + [precision[-1]]
        plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
        fig = plt.gcf()
        fig.canvas.set_window_title('AP ' + class_name)
        plt.title(title)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        axes = plt.gca()
        axes.set_xlim([0.0, 1.0])
        axes.set_ylim([0.0, 1.05])
        fig.savefig(os.path.join(self.results_files_path, "classes", class_name + ".png"))
        plt.cla()

    def compute_recall_precision(self, fp, tp, class_name):
        cumsum = 0
        for index, value in enumerate(fp):
            fp[index] += cumsum
            cumsum += value
        cumsum = 0
        for index, value in enumerate(tp):
            tp[index] += cumsum
            cumsum += value
        recall = tp[:]
        for index, value in enumerate(tp):
            recall[index] = float(tp[index]) / self.ground_truth_classes_counter[class_name]
        precision = tp[:]
        for index, value in enumerate(tp):
            precision[index] = float(tp[index]) / (fp[index] + tp[index])

        average_precision, descending_recall, descending_precision = self.voc_ap(recall, precision)
        return average_precision, descending_recall, descending_precision

    def write_to_average_precision_to_results(self, title, recall, precision, file):
        rounded_precision = ['%.2f' % elem for elem in recall]
        rounded_recall = ['%.2f' % elem for elem in precision]
        file.write("{}\nPrecision: {}\nRecall: {}\n\n".format(title, str(rounded_precision), str(rounded_recall)))
        if not self.quiet:
            print(title)

    @staticmethod
    def write_mean_average_precision_to_results(mean_average_precision, file):
        message = "mAP = {0:.2f}%\n".format(mean_average_precision)
        file.write("\n# mAP of all classes\n{}".format(message))
        print(message)

    def clean_up_temporary_files(self):
        shutil.rmtree(self.temporary_files_path)

    def draw_plot_func(self, dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color,
                       true_p_bar):
        # TODO: Rewrite this function to something readable
        # sort the dictionary by decreasing value, into a list of tuples
        sorted_dic_by_value = sorted(dictionary.items(), key=lambda x: x[1])
        # unpacking the list of tuples into two lists
        sorted_keys, sorted_values = zip(*sorted_dic_by_value)
        #
        if true_p_bar != "":
            fp_sorted = []
            tp_sorted = []
            for key in sorted_keys:
                fp_sorted.append(dictionary[key] - true_p_bar[key])
                tp_sorted.append(true_p_bar[key])
            plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Predictions')
            plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Predictions',
                     left=fp_sorted)
            plt.legend(loc='lower right')
            """
             Write number on side of bar
            """
            fig = plt.gcf()  # gcf - get current figure
            axes = plt.gca()
            r = fig.canvas.get_renderer()
            for i, val in enumerate(sorted_values):
                fp_val = fp_sorted[i]
                tp_val = tp_sorted[i]
                fp_str_val = " " + str(fp_val)
                tp_str_val = fp_str_val + " " + str(tp_val)
                # trick to paint multicolor with offset:
                #   first paint everything and then repaint the first number
                t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
                plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
                if i == (len(sorted_values) - 1):  # largest bar
                    self.adjust_axes(r, t, fig, axes)
        else:
            plt.barh(range(n_classes), sorted_values, color=plot_color)
            """
             Write number on side of bar
            """
            fig = plt.gcf()  # gcf - get current figure
            axes = plt.gca()
            r = fig.canvas.get_renderer()
            for i, val in enumerate(sorted_values):
                str_val = " " + str(val)  # add a space before
                if val < 1.0:
                    str_val = " {0:.2f}".format(val)
                t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
                # re-set axes to show number inside the figure
                if i == (len(sorted_values) - 1):  # largest bar
                    self.adjust_axes(r, t, fig, axes)
        # set window title
        fig.canvas.set_window_title(window_title)
        # write classes in y axis
        tick_font_size = 12
        plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
        """
         Re-scale height accordingly
        """
        init_height = fig.get_figheight()
        # comput the matrix height in points and inches
        dpi = fig.dpi
        height_pt = n_classes * (tick_font_size * 1.4)  # 1.4 (some spacing)
        height_in = height_pt / dpi
        # compute the required figure height
        top_margin = 0.15  # in percentage of the figure height
        bottom_margin = 0.05  # in percentage of the figure height
        figure_height = height_in / (1 - top_margin - bottom_margin)
        # set new height
        if figure_height > init_height:
            fig.set_figheight(figure_height)

        # set plot title
        plt.title(plot_title, fontsize=14)
        # set axis titles
        # plt.xlabel('classes')
        plt.xlabel(x_label, fontsize='large')
        # adjust size of window
        fig.tight_layout()
        # save the plot
        fig.savefig(output_path)
        # show image
        if to_show:
            plt.show()
        # close the plot
        plt.close()

    def count_total_predictions(self):
        predicted_files = glob.glob(os.path.join(self.base_directory, "predicted", "*.txt"))
        for text_file in predicted_files:
            for line in self.lines_to_list(text_file):
                self.predictions_per_class_counter.update({line.split()[0]: 1})
        return self.predictions_per_class_counter

    def plot_ground_truth_info(self):
        # TODO: rewrite this to something more pythonic
        ground_truth_files = glob.glob(os.path.join(self.base_directory, "ground-truth", "*.txt"))
        window_title = "Ground-Truth Info"
        plot_title = "Ground-Truth\n"
        plot_title += "(" + str(len(ground_truth_files)) + " files and " \
                      + str(len(self.ground_truth_classes_counter.keys())) + " classes)"
        x_label = "Number of objects per class"
        output_path = os.path.join(self.results_files_path, "Ground-Truth Info.png")
        to_show = False
        plot_color = "forestgreen"
        self.draw_plot_func(
            self.ground_truth_classes_counter,
            len(self.ground_truth_classes_counter.keys()),
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            '',
        )

    def write_ground_truth_info_to_results(self):
        with open(self.results_file, "a") as results_file:
            results_file.write("\n# Number of ground-truth objects per class\n")
            for class_name, count in sorted(self.ground_truth_classes_counter.items(), key=lambda x: x[0]):
                results_file.write("{}: {}\n".format(class_name, count))

    def plot_predicted_objects_info(self):
        # TODO: rewrite this to something more pythonic
        predicted_files = glob.glob(os.path.join(self.base_directory, "predicted", "*.txt"))
        window_title = "Predicted Objects Info"
        plot_title = "Predicted Objects\n"
        plot_title += "(" + str(len(predicted_files)) + " files and "
        count_non_zero_values_in_dictionary = sum(int(x) > 0 for x in list(self.predictions_per_class_counter.values()))
        plot_title += str(count_non_zero_values_in_dictionary) + " detected classes)"
        x_label = "Number of objects per class"
        output_path = os.path.join(self.results_files_path, "Predicted Objects Info.png")
        to_show = False
        plot_color = 'forestgreen'
        true_p_bar = self.true_positives
        self.draw_plot_func(
            self.predictions_per_class_counter,
            len(self.predictions_per_class_counter.keys()),
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            true_p_bar
        )

    def write_predicted_objects_info_to_results(self):
        with open(self.results_file, 'a') as results_file:
            results_file.write("\n# Number of predicted objects per class\n")
            for class_name, count in sorted(self.predictions_per_class_counter.items(), key=lambda x: x[0]):
                tp = self.true_positives[class_name]
                results_file.write("{}: {} (tp:{}, fp:{})\n".format(class_name, count, tp, count - tp))

    def plot_mean_average_precision(self, map, ap_per_class):
        # TODO: rewrite this to something more pythonic
        window_title = "mAP"
        plot_title = "mAP = {0:.2f}%".format(map * 100)
        x_label = "Average Precision"
        output_path = os.path.join(self.results_files_path, "mAP.png")
        to_show = True
        plot_color = 'royalblue'
        self.draw_plot_func(
            ap_per_class,
            len(self.ground_truth_classes_counter.keys()),
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            "")

    def voc_ap(self, recall, precision):
        recall.insert(0, 0.0)  # insert 0.0 at begining of list
        recall.append(1.0)  # insert 1.0 at end of list
        descending_recall = recall[:]

        precision.insert(0, 0.0)  # insert 0.0 at begining of list
        precision.append(0.0)  # insert 0.0 at end of list
        descending_precision = precision[:]

        for i in range(len(descending_precision) - 2, -1, -1):
            descending_precision[i] = max(descending_precision[i], descending_precision[i + 1])

        i_list = []
        for i in range(1, len(descending_recall)):
            if descending_recall[i] != descending_recall[i - 1]:
                i_list.append(i)  # if it was matlab would be i + 1

        average_precision = 0.0
        for i in i_list:
            average_precision += ((descending_recall[i] - descending_recall[i - 1]) * descending_precision[i])
        return average_precision, descending_recall, descending_precision

    @staticmethod
    def adjust_axes(r, t, fig, axes):
        # TODO: rewrite this to something more readable
        # get text width for re-scaling
        bb = t.get_window_extent(renderer=r)
        text_width_inches = bb.width / fig.dpi
        # get axis width in inches
        current_fig_width = fig.get_figwidth()
        new_fig_width = current_fig_width + text_width_inches
        propotion = new_fig_width / current_fig_width
        # get axis limit
        x_lim = axes.get_xlim()
        axes.set_xlim([x_lim[0], x_lim[1] * propotion])


if __name__ == "__main__":
    MeanAveragePrecision(base_directory=r"D:\LEGO Vision Datasets\Localization\Natural Data\results\svm").evaluate()
