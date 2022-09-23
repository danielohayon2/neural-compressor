import argparse
import tensorflow as tf
def get_inception_resnet_v2_model(saved_path):
    model = tf.keras.applications.InceptionResNetV2(weights='imagenet')
    model.save(saved_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description='Export pretained keras model',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--output_model',
        type=str,
        help='path to exported model file')

    args = parser.parse_args()
    get_inception_resnet_v2_model(args.output_model)
