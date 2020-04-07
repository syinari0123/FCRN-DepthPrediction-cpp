import argparse
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def preprocess(image, img_size):
    image = image.resize(img_size, Image.ANTIALIAS)
    image = np.array(image).astype(np.float32)
    images = image[None, :, :, :]
    return images


def postprocess(depth):
    depth_final = depth[0, :, :, 0]
    return depth_final


def main(args):
    # Read image
    image = Image.open(args.input_image)
    image = preprocess(image, [args.in_width, args.in_height])

    # Load model
    graph = tf.Graph()
    with tf.gfile.GFile(args.pb_file, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Inference
    with graph.as_default():
        # Define input tensor
        input_node = tf.placeholder(np.float32, shape=[None, args.in_height, args.in_width, 3], name='input_image')
        tf.import_graph_def(graph_def, {'input_image': input_node})
        graph.finalize()

        # inference
        sess = tf.Session(graph=graph)
        with tf.Session() as sess:
            output_tensor = graph.get_tensor_by_name(args.out_node_name)
            output = sess.run(output_tensor, feed_dict={input_node: image})

    # Postprocess
    output = postprocess(output)

    # Plot result
    fig = plt.figure()
    ii = plt.imshow(output, interpolation='nearest')
    fig.colorbar(ii)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-image', type=str, default="sample.png")
    parser.add_argument('--pb-file', type=str, default="pb_file/NYU_FCRN.pb")
    parser.add_argument('--in-height', type=int, default=228)
    parser.add_argument('--in-width', type=int, default=304)
    parser.add_argument('--out-node-name', type=str, default="import/ConvPred/ConvPred:0")

    args = parser.parse_args()
    print(args)
    main(args)
