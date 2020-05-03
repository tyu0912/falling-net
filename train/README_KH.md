# W251 - Fall Detection

### Docker

Requires `/data/` directory and expects contents to be in `w251fall` subdirectory.

**Expects that host has NVIDIA GPU & Cuda (tested using Version 10.1).**

Run `docker_run.sh`


## Data

### Raw Video Files

Video files should be `avi` files.  Each file should have a single fall.  The files need to be be stored in `/data/w251fall/videos/Fall` for preprocessing.

### Preprocessing

The preprocessing is going to output frames from the videos in to `/data/w251fall/jpg` (this directory needs to exist).  The directory structure will mimic the above Raw structure, but each video file will have its own directory with the same name excluding the file extension.  Inside will be a sequence of jpg files `img_00001.jpg`.

The the script:
**Will keep old files, which will speed things up, but may leave garbage behind if videos were removed or renamed** 
**For some reason, after running this script the terminal will stop echoing (can't see what you're typing).  I just restart docker.**

`python tools/vid2img_w251Fall.py /data/w251fall/videos/ /data/w251fall/jpg/`

### Create train and validation sets

`python tools/gen_label_w251fall.py`

## Training


### Transfer learn on Fall data

Run `fall_train.sh`

Checkpoint ends up in `checkpoint` subdirectory.

## Test

**Batch size is only set to 2, 4 causes OOM error**
**The directory in checkpoint may change with args used while training.**

### Mobile Net V2
```
# test TSN using non-local testing protocol
python test_models.py w251fall \
    --weights=checkpoint/TSM_w251fall_RGB_mobilenetv2_shift8_blockres_avg_segment8_e25/ckpt.best.pth.tar  \
    --test_segments=8 --test_crops=3 \
    --batch_size=2 --dense_sample --full_res
```


### resnet50
```
# test TSN using non-local testing protocol
python test_models.py w251fall \
    --weights=checkpoint/TSM_w251fall_RGB_resnet50_shift8_blockres_avg_segment8_e25/ckpt.best.pth.tar  \
    --test_segments=8 --test_crops=3 \
    --batch_size=2 --dense_sample --full_res
```

### Jupyter Lab

Run `jupyterlab_run.sh`
