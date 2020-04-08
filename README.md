# TensorFlow C++ inference of FCRN-Depth-Prediction
This is a TensorFlow c++ inference code of [FCRN-DepthPrediction](https://github.com/iro-cp/FCRN-DepthPrediction) (single-view deep depth prediction) for indoor scenes.

<p align="center">
    <img src='https://github.com/syinari0123/FCRN-DepthPrediction-cpp/blob/master/view/demo.gif' width=60%/></a>
</p>

This code uses [cppflow](https://github.com/serizba/cppflow), which doesn't require either TensorFlow compile or bazel (this installation is known as a very difficult task [[1](https://github.com/yan99033/monodepth-cpp/tree/master/Tensorflow_build_instructions), [2](https://github.com/muskie82/CNN-DSO)]).

## Environment
- Ubuntu18.04 (GPU: NVIDIA GeForce GTX 1080)
- CUDA10.0
- cuDNN7.4
- Python3.6.9 (for preparing pb-file)
- TensorFlow for C 1.13.1 (this version requires CUDA10.0 & cuDNN7.4)

You need to download libtensorflow & install it referring to [official page](https://www.tensorflow.org/install/lang_c).
```sh
wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-1.13.1.tar.gz
sudo tar -C /usr/local -xzf libtensorflow-gpu-linux-x86_64-1.13.1.tar.gz
sudo ldconfig
```

## How to run it
See [README](https://github.com/syinari0123/FCRN-DepthPrediction-cpp/tree/master/freeze_graph) in freeze_graph, and prepare pb-file of pretrained FCRN-DepthPrediction model.
```
cd freeze_graph/pb_file
./download.sh
```
Build our code (based on [cppflow](https://github.com/serizba/cppflow)) with following commands.
```sh
mkdir build && cd build
cmake ..
make
```
Specifying the path to the [pb-file](https://github.com/syinari0123/FCRN-DepthPrediction-cpp/tree/master/freeze_graph/pb_file) & target image directory ([samples](https://github.com/syinari0123/FCRN-DepthPrediction-cpp/tree/master/samples) from [NYUDepthv2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)), execute as follows.
```sh
./inference_fcrn \
    pb_file="../freeze_graph/pb_file/NYU_FCRN.pb" \
    img_dir="../samples/home_office_0013"
```

## References
- [FCRN-DepthPrediction](https://github.com/iro-cp/FCRN-DepthPrediction)
- [cppflow](https://github.com/serizba/cppflow)
- [monodepth-cpp](https://github.com/yan99033/monodepth-cpp)
- [NYU Depth Dataset v2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html)