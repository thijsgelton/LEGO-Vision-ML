import argparse
import glob
import random
import xml.dom.minidom
import xml.etree.cElementTree as et

import cntk.io.transforms as xforms
import numpy as np
from PIL import Image
from cntk import training_session, Trainer, cross_entropy_with_softmax, input_variable, \
    classification_error, UnitType, learning_rate_schedule, momentum_sgd, softmax, CheckpointConfig, \
    gpu, try_set_default_device, element_times, load_model, logging, CloneMethod
from cntk.io import MinibatchSource, StreamDef, ImageDeserializer, StreamDefs
from cntk.layers import Dense, combine, \
    placeholder
from cntk.logging import ProgressPrinter, os, log_number_of_parameters

success = try_set_default_device(gpu(0))
print(f"Using GPU: {success}")


def create_map_files_from_folders(image_directory):
    with open(os.path.join(image_directory, 'images.txt'), mode='w') as f:
        path, classes, file = list(os.walk(image_directory))[0]
        files = []
        for cls in classes:
            for file in [x for x in glob.glob(os.path.join(path, cls, '*')) if not x.endswith('txt')][:3200]:
                if file.endswith(('png', 'jpg', 'jpeg')):
                    files.append(f"{os.path.abspath(file)}\t{classes.index(cls)}\n")
        random.shuffle(files)
        f.writelines(files)


def create_reader(map_file, mean_file, train, pixel_dimensions, classes, total_number_of_samples):
    print(f"Reading map file: {map_file} with number of samples {total_number_of_samples}")
    transforms = [
        xforms.scale(width=pixel_dimensions['width'],
                     height=pixel_dimensions['height'],
                     channels=pixel_dimensions['depth'],
                     interpolations='linear'),
        xforms.mean(mean_file)
    ]

    source = MinibatchSource(
        deserializers=ImageDeserializer(map_file, StreamDefs(features=StreamDef(field='image', transforms=transforms),
                                                             labels=StreamDef(field='label', shape=len(classes)))),
        randomize=train, max_samples=total_number_of_samples)
    return source


def save_mean(filename, data, pixel_dimensions):
    root = et.Element('opencv_storage')
    et.SubElement(root, 'Channel').text = str(pixel_dimensions['depth'])
    et.SubElement(root, 'Row').text = str(pixel_dimensions['height'])
    et.SubElement(root, 'Col').text = str(pixel_dimensions['width'])
    mean_img = et.SubElement(root, 'MeanImg', type_id='opencv-matrix')
    et.SubElement(mean_img, 'rows').text = '1'
    et.SubElement(mean_img, 'cols').text = str(
        pixel_dimensions['width'] * pixel_dimensions['height'] * pixel_dimensions['depth'])
    et.SubElement(mean_img, 'dt').text = 'f'
    flattened = data.flatten()
    flattened_with_breaks = np.insert(np.asarray(flattened, dtype="O"), range(4, len(flattened), 4), "\n")
    et.SubElement(mean_img, 'data').text = ' '.join([str(n) for n in flattened_with_breaks])

    tree = et.ElementTree(root)
    tree.write(filename)
    x = xml.dom.minidom.parse(filename)
    with open(filename, 'w') as f:
        f.write(x.toprettyxml(indent='  '))


def create_mean_file(map_file, pixel_dimensions, filename):
    with open(map_file, mode="r") as f:
        summed = np.zeros((pixel_dimensions['depth'], pixel_dimensions['width'], pixel_dimensions['height']))
        total = list(f.readlines())
        for file in total:
            image = Image.open(os.path.join(file.split("\t")[0]))
            image = image.resize((pixel_dimensions['width'], pixel_dimensions['height']))
            image = np.array(image).transpose((2, 0, 1)) if len(np.array(image).shape) == 3 else np.array(image)
            summed += image
        mean = summed / len(total)
    save_path = os.path.join(map_file.strip("images.txt"), filename)
    save_mean(save_path, mean, pixel_dimensions)
    return save_path


def create_tf_model(model_details, num_classes, input_features, new_prediction_node_name='prediction', freeze=False):
    base_model = load_model(os.path.abspath(model_details['model_file']))
    feature_node = logging.find_by_name(base_model, model_details['feature_node_name'])
    last_node = logging.find_by_name(base_model, model_details['last_hidden_node_name'])

    cloned_layers = combine([last_node.owner]).clone(
        CloneMethod.freeze if freeze else CloneMethod.clone,
        {feature_node: placeholder(name='features')})

    cloned_out = cloned_layers(input_features)
    z = Dense(num_classes, activation=None, name=new_prediction_node_name)(cloned_out)

    return z


def finalize_network(reader, model_details, max_amount_of_epochs, samples_per_epoch,
                     samples_per_minibatch, pixel_dimensions, classes, learning_rate):
    features = input_variable(shape=(pixel_dimensions['depth'], pixel_dimensions['height'], pixel_dimensions['width']))
    label = input_variable(shape=len(classes))

    # speeds up training
    normalized_features = element_times(1.0 / 256.0, features)

    model = create_tf_model(model_details, num_classes=len(classes), input_features=normalized_features, freeze=True)

    loss = cross_entropy_with_softmax(model, label)
    metric = classification_error(model, label)
    learner = momentum_sgd(parameters=model.parameters,
                           lr=learning_rate_schedule(learning_rate, UnitType.minibatch),
                           momentum=0.9, l2_regularization_weight=0.0005)

    reporter = ProgressPrinter(tag='training', num_epochs=max_amount_of_epochs)

    trainer = Trainer(model=model, criterion=(loss, metric), parameter_learners=[learner], progress_writers=[reporter])

    log_number_of_parameters(model)

    map_input_to_streams_train = {
        features: reader.streams.features,
        label: reader.streams.labels
    }

    training_session(
        trainer=trainer, mb_source=reader,
        model_inputs_to_streams=map_input_to_streams_train,
        mb_size=samples_per_minibatch,
        progress_frequency=samples_per_epoch,
        checkpoint_config=CheckpointConfig(frequency=samples_per_epoch,
                                           filename=os.path.join("./checkpoints", "ConvNet_Lego_VisiOn"),
                                           restore=True)
    ).train()
    network = {
        'features': features,
        'label': label,
        'model': softmax(model)
    }
    model_name = f"CNN-3200-224-resnet-18.model"
    export_path = os.path.abspath(os.path.join("..", "..", "Final models", "CNN", model_name))
    model.save(export_path)
    return network


def main(max_epochs, image_directory, lr, number_of_samples, samples_per_minibatch, model_details):
    create_map_files_from_folders(image_directory)
    pixel_dimensions = model_details['image_dims']
    classes = list(os.walk(os.path.join(image_directory)))[0][1]
    all_images_map_file = os.path.join(image_directory, 'images.txt')
    mean_file = create_mean_file(all_images_map_file, pixel_dimensions, 'mean_file.xml')
    samples_per_epoch_train = len(open(all_images_map_file).readlines())

    print(f"Classes: {classes}")
    print(f"Max epochs: {max_epochs}")
    print(f"Learning rate: {lr}")
    print(f"Minibatch Size: {samples_per_minibatch}")
    print(f"Width and height of: {pixel_dimensions}")
    print(f"Number of training samples: {number_of_samples}")
    reader = create_reader(map_file=all_images_map_file, pixel_dimensions=pixel_dimensions, classes=classes, train=True,
                           total_number_of_samples=max_epochs * samples_per_epoch_train,
                           mean_file=mean_file)

    finalize_network(reader=reader,
                     samples_per_epoch=samples_per_epoch_train,
                     max_amount_of_epochs=max_epochs,
                     samples_per_minibatch=samples_per_minibatch,
                     pixel_dimensions=pixel_dimensions,
                     classes=classes,
                     learning_rate=lr,
                     model_details=model_details)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_directory', help='Directory with images')
    args = parser.parse_args()

    model_details = dict(
        model_file=os.path.join("..", "..", "..", "Pretrained Models", "ResNet_18.model"),
        feature_node_name='features',
        last_hidden_node_name='z.x',
        image_dims=dict(depth=3, width=224, height=224)
    )
    main(max_epochs=100, image_directory=args.image_directory,
         lr=0.01, number_of_samples=3200, samples_per_minibatch=64, model_details=model_details)
