# Freeze tensorflow graph from ckpt file
## How to obtain pb-file
Download checkpoint file provided in [official web-page](https://github.com/iro-cp/FCRN-DepthPrediction).
```sh
cd ckpt_files
./get_fcrn.sh
```
Freeze graph-file and output pb-file (this code is based on [monodepth-cpp](https://github.com/yan99033/monodepth-cpp/tree/master/freeze_graph)).
```sh
python freeze_graph.py \
    --batch_size 1 \
    --img_height 228 \
    --img_width 304 \
    --output_node_name "ConvPred/ConvPred" \
    --ckpt_file "ckpt_file/NYU_FCRN.ckpt" \
    --output_dir "pb_file"
```
You can download converted pb-file from [this link](https://www.dropbox.com/s/aud653q4naeeav1/NYU_FCRN.pb).

## Python inference from pb-file
By running following code, you can easily check whether your pb-file is correct or not.
```sh
python inference_from_pb.py \
    --input-image "sample.png" \
    --pb-file "pb_file/NYU_FCRN.pb" \
    --in-height 228 \
    --in-width 304 \
    --out-node-name "import/ConvPred/ConvPred:0"
```