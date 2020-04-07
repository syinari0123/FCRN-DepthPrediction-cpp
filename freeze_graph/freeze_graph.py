"""
MIT License

Copyright (c) 2018 Shing Yan Loo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
from pathlib import Path
import tensorflow as tf
import models


def main(argv):
    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(arg.batch_size, arg.img_height, arg.img_width, 3),
                                name='input_image')

    # Load model and get output
    model = models.ResNet50UpProj({'data': input_node}, 1, 1, False)
    y = model.get_output()

    # initialise the saver
    saver = tf.train.Saver()

    # Session
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # Initialize variables
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # restore all variables from checkpoint
    saver.restore(sess, arg.ckpt_file)

    # node that are required output nodes
    output_node_names = [arg.output_node_name]

    # We use a built-in TF helper to export variables to constants
    output_graph_def = tf.graph_util.convert_variables_to_constants(
        sess,
        tf.get_default_graph().as_graph_def(),
        # The graph_def is used to retrieve the nodes
        output_node_names  # The output node names are used to select the useful nodes
    )

    # Convert variables to constants
    output_graph_def = tf.graph_util.remove_training_nodes(output_graph_def)

    # Finally we serialize and dump the output graph to the filesystem
    if not os.path.exists(arg.output_dir):
        os.makedirs(arg.output_dir)
    out_pb_file = os.path.join(arg.output_dir, Path(arg.ckpt_file).stem + ".pb")
    with tf.gfile.GFile(out_pb_file, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    print("Frozen graph file {} created successfully".format(out_pb_file))


if __name__ == "__main__":
    args = tf.app.flags
    args.DEFINE_integer('batch_size', 1, 'The size of of a sample batch')
    args.DEFINE_integer('img_height', 228, 'Image height')
    args.DEFINE_integer('img_width', 304, 'Image width')
    args.DEFINE_string('output_node_name', 'ConvPred/ConvPred', 'Output placeholder name')
    args.DEFINE_string('ckpt_file', 'ckpt_files/NYU_FCRN.ckpt', 'checkpoint file path')
    args.DEFINE_string('output_dir', 'pb_files', 'Output directory of a graph file')
    arg = args.FLAGS
    tf.app.run()
