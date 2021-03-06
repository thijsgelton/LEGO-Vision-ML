import argparse
import glob
import random
from urllib.request import urlretrieve

import cntk.io.transforms as xforms
import numpy as np
from cntk import relu, glorot_uniform, training_session, Trainer, Function, cross_entropy_with_softmax, input_variable, \
    classification_error, UnitType, learning_rate_schedule, momentum_sgd, softmax, CheckpointConfig, \
    TestConfig, gpu, try_set_default_device, element_times, load_model, logging, CloneMethod
from cntk.io import MinibatchSource, StreamDef, ImageDeserializer, StreamDefs, FULL_DATA_SWEEP
from cntk.layers import default_options, Convolution, BatchNormalization, MaxPooling, Dense, For, Sequential, combine, \
    placeholder, Constant
from cntk.logging import ProgressPrinter, os, log_number_of_parameters
from sklearn.metrics import fbeta_score

import helpers

from azureml.core.run import Run

# get the Azure ML run object
run = Run.get_submitted_run()
success = try_set_default_device(gpu(0))
print(f"Using GPU: {success}")


def create_map_files_from_folders(data_dir, split=0.8, number_of_training_samples=800):
    with open(os.path.join(data_dir, 'images.txt'), mode='w') as f:
        path, classes, file = list(os.walk(data_dir))[0]
        for cls in classes:
            for file in glob.glob(os.path.join(path, cls, '*'))[:number_of_training_samples]:
                if file.endswith(('png', 'jpg')):
                    f.write(f"{os.path.abspath(file)}\t{classes.index(cls)}\n")

    with open(os.path.join(data_dir, 'images.txt')) as f:
        images = f.readlines()
        is_it_shuffled = set([image.split('\t')[-1] for image in images[:20]])
        if len(is_it_shuffled.difference()) < 2:  # if smaller than 2, it is not shuffled yet.
            random.shuffle(images)
        eval_images = [images.pop() for i in range(100)]
        train_images = images[:int(len(images) * split)]
        test_images = images[int(len(images) * split):]
        with open(os.path.join(data_dir, 'finalize_network.txt'), mode='w') as t:
            t.writelines([f"{image}" for image in train_images])
        with open(os.path.join(data_dir, 'test.txt'), mode='w') as test:
            test.writelines([f"{image}" for image in test_images])
        with open(os.path.join(data_dir, 'evaluate.txt'), mode='w') as e:
            e.writelines([f"{image}" for image in eval_images])


def create_reader(map_file, train, dimensions, classes, total_number_of_samples):
    print(f"Reading map file: {map_file} with number of samples {total_number_of_samples}")

    # transformation pipeline for the features has jitter/crop only when training
    transforms = []
    # finalize_network uses data augmentation (translation only)
    if train:
        transforms += [
            xforms.crop(crop_type='randomside', area_ratio=(0.08, 1.0), aspect_ratio=(0.75, 1.3333),
                        jitter_type='uniratio'),
            xforms.color(brightness_radius=0.4, contrast_radius=0.4, saturation_radius=0.4)
        ]
    transforms += [
        xforms.scale(width=dimensions['width'], height=dimensions['height'], channels=dimensions['depth'],
                     interpolations='linear')
    ]
    source = MinibatchSource(
        ImageDeserializer(map_file, StreamDefs(
            features=StreamDef(field='image', transforms=transforms),
            labels=StreamDef(field='label', shape=len(classes))
        )), randomize=train,
        max_samples=total_number_of_samples,
        multithreaded_deserializer=True)
    return source


def create_model(feature_dimensions, classes):
    with default_options(activation=relu, init=glorot_uniform()):
        model = Sequential([
            For(range(3), lambda i: [
                Convolution((5, 5), [32, 32, 64][i], pad=True),
                BatchNormalization(map_rank=1),
                MaxPooling((3, 3), strides=(2, 2))
            ]),
            Dense(64),
            BatchNormalization(map_rank=1),
            Dense(len(classes), activation=None)
        ])

    return model(feature_dimensions)


def download_unless_exists(url, filename):
    if os.path.exists(filename):
        print('Reusing locally stored: ', filename)
    else:
        print('Starting download of {} to {}'.format(url, filename))
        try:
            urlretrieve(url, filename)
            print('Download completed.')
        except:
            print('Failed.')


def download_model(model_root, download_url, output_filename):
    model_uri = download_url
    model_local = os.path.join(model_root, output_filename)
    download_unless_exists(model_uri, model_local)
    return model_local


# Creates the network model for transfer learning
def create_tf_model(model_details, num_classes, input_features, new_prediction_node_name='prediction', freeze=False):
    # Load the pretrained classification net and find nodes
    print(os.path.abspath(model_details['model_file']))
    base_model = load_model(os.path.abspath(model_details['model_file']))
    feature_node = logging.find_by_name(base_model, model_details['feature_node_name'])
    last_node = logging.find_by_name(base_model, model_details['last_hidden_node_name'])

    # Clone the desired layers with fixed weights
    cloned_layers = combine([last_node.owner]).clone(
        CloneMethod.freeze if freeze else CloneMethod.clone,
        {feature_node: placeholder(name='features')})

    # Add new dense layer for class prediction
    cloned_out = cloned_layers(input_features)
    z = Dense(num_classes, activation=None, name=new_prediction_node_name)(cloned_out)

    return z


@Function
def create_criterion(output_layer, label):
    loss = cross_entropy_with_softmax(output_layer, label)
    metric = classification_error(output_layer, label)
    return loss, metric


def train(reader_train, reader_test, samples_per_epoch, max_amount_of_epochs, samples_per_minibatch, dimensions,
          classes, learning_rate, output_directory, tf_model_url):
    features = input_variable(shape=(dimensions['depth'], dimensions['height'], dimensions['width']))
    label = input_variable(shape=len(classes))

    # speeds up training
    normalized_features = element_times(1.0 / 256.0, features)
    if tf_model_url:
        model_file = download_model(output_directory, tf_model_url, 'ResNet_18.model')
        base_model = {
            'model_file': model_file,
            'feature_node_name': 'features',
            'last_hidden_node_name': 'z.x',
            # Channel Depth x Height x Width
            'image_dims': (3, 224, 224)
        }
        model = create_tf_model(base_model, num_classes=len(classes), input_features=normalized_features, freeze=True)
    else:
        model = create_model(feature_dimensions=normalized_features, classes=classes)

    criterion = create_criterion(output_layer=model, label=label)
    learner = momentum_sgd(parameters=model.parameters,
                           lr=learning_rate_schedule(learning_rate, UnitType.minibatch),
                           momentum=0.9,
                           l2_regularization_weight=0.01)

    reporter = ProgressPrinter(tag='training', num_epochs=max_amount_of_epochs)

    trainer = Trainer(model=model, criterion=criterion, parameter_learners=[learner], progress_writers=[reporter])

    log_number_of_parameters(model)

    map_input_to_streams_train = {
        features: reader_train.streams.features,
        label: reader_train.streams.labels
    }

    map_input_to_streams_test = {
        features: reader_test.streams.features,
        label: reader_test.streams.labels
    }

    training_session(
        trainer=trainer, mb_source=reader_train,
        model_inputs_to_streams=map_input_to_streams_train,
        mb_size=samples_per_minibatch,
        progress_frequency=samples_per_epoch,
        checkpoint_config=CheckpointConfig(frequency=samples_per_epoch,
                                           filename=os.path.join(output_directory, "ConvNet_Lego_VisiOn"),
                                           restore=False),
        test_config=TestConfig(reader_test, minibatch_size=samples_per_minibatch,
                               model_inputs_to_streams=map_input_to_streams_test)
    ).train()
    network = {
        'features': features,
        'label': label,
        'model': softmax(model)
    }
    return network


def evaluate_batch(network, reader_eval, samples_per_epoch_eval, classes, output_directory, number_of_samples):
    features = network['features']
    label = network['label']
    model = network['model']

    map_input_to_streams = {
        features: reader_eval.streams.features,
        label: reader_eval.streams.labels
    }

    data = reader_eval.next_minibatch(samples_per_epoch_eval, input_map=map_input_to_streams)

    labels = data[label].as_sequences(label)
    images = data[features].as_sequences(features)
    predictions = [model.eval(image) for image in images]

    # Find the index with the maximum value for both predicted as well as the ground truth
    y_pred = [np.argmax(prediction) for prediction in predictions]
    y_true = [np.argmax(label) for label in labels]

    helpers.plot_confusion_matrix_with_acc_and_fbeta(y_true, y_pred, classes, save_path='outputs/yoo.png',
                                                     normalize=True)

    fscore = fbeta_score(y_true, y_pred, beta=0.5, average='macro')
    run.log('f_score', np.float(fscore))
    # save model to outputs folder
    model.save(f'{output_directory}/cntk-{features.shape}-{number_of_samples}.model')


def main(max_epochs, data_dir, test_dir, output_dir, lr, dimensions, number_of_samples, samples_per_minibatch,
         tf_model_url):
    create_map_files_from_folders(data_dir, split=0.7, number_of_training_samples=number_of_samples)

    classes = list(os.walk(os.path.join(data_dir)))[0][1]
    train_map_file = os.path.join(data_dir, 'finalize_network.txt')
    test_map_file = os.path.join(data_dir, 'test.txt')
    # eval_map_file = os.path.join(data_dir, 'evaluate.txt')

    if test_dir:
        create_map_files_from_folders(test_dir, split=0.7, number_of_training_samples=number_of_samples)
        test_map_file = os.path.join(test_dir, 'test.txt')

    max_amount_of_epochs = max_epochs
    samples_per_epoch_train = len(open(train_map_file).readlines())
    samples_per_epoch_test = len(open(test_map_file).readlines())

    print(f"Classes: {classes}")
    print(f"Max epochs: {max_epochs}")
    print(f"Learning rate: {lr}")
    run.log('learning_rate', lr)
    print(f"Minibatch Size: {samples_per_minibatch}")
    run.log("Minibatch Size", samples_per_minibatch)
    print(f"Width and height of: {dimensions}")
    run.log("shape", dimensions['width'])
    print(f"Number of training samples: {number_of_samples}")
    run.log("samples", number_of_samples)
    run.log("data_dir", data_dir)
    run.log("test_dir", test_dir)

    reader_train = create_reader(map_file=train_map_file, dimensions=dimensions, classes=classes, train=True,
                                 total_number_of_samples=max_amount_of_epochs * samples_per_epoch_train)
    reader_test = create_reader(map_file=test_map_file, dimensions=dimensions, classes=classes, train=False,
                                total_number_of_samples=FULL_DATA_SWEEP)
    reader_eval = create_reader(map_file=test_map_file, dimensions=dimensions, classes=classes, train=False,
                                total_number_of_samples=FULL_DATA_SWEEP)
    network = train(reader_train=reader_train, reader_test=reader_test, samples_per_epoch=samples_per_epoch_train,
                    max_amount_of_epochs=max_amount_of_epochs,
                    samples_per_minibatch=samples_per_minibatch, dimensions=dimensions, classes=classes,
                    learning_rate=lr, output_directory=output_dir, tf_model_url=tf_model_url)

    evaluate_batch(network, reader_eval, samples_per_epoch_test, classes, output_directory=output_dir,
                   number_of_samples=number_of_samples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_epochs', help='Total number of epochs to finalize_network', type=int, default='40')
    parser.add_argument('--lr', help='Learning rate', type=float, default='0.01')
    parser.add_argument('--mb_size', help='Learning rate', type=int, default='64')
    parser.add_argument('--dimension', help='the width and height e.g. (128,128)', type=int, default='64')
    parser.add_argument('--number_of_samples', help='the width and height e.g. (128,128)', type=int, default='400')
    parser.add_argument('--output_dir', help='Output directory', required=False, default='outputs')
    parser.add_argument('--data_dir', help='Directory with training data')
    parser.add_argument('--test_dir', help='Directory with test data', default=None)
    parser.add_argument('--tf_model_url', help="specify transfer learning model download url", default=None)
    args = parser.parse_args()

    dimensions = dict(
        height=args.dimension,
        width=args.dimension,
        depth=3
    )
    main(max_epochs=args.max_epochs, data_dir=args.data_dir, test_dir=args.test_dir,
         output_dir=args.output_dir, lr=args.lr, samples_per_minibatch=args.mb_size,
         dimensions=dimensions, number_of_samples=args.number_of_samples, tf_model_url=args.tf_model_url)
