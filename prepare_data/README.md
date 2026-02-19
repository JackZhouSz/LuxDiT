# Data Preparation

This folder contains some training/testing data list (`./data_list`) from [Polyhaven](https://polyhaven.com/) and [WEB360](https://github.com/Akaneqwq/360DVD) used during our model development.

We also provide scripts that converts 360 pano images/videos to perspective views for testing:

1. To convert HDR image into perspective image:

```bash
# run in the base repo dir
python -m prepare_data.hdri_to_pers_img \
    --hdri_folder examples/hdri \
    --output_folder examples/hdri_pers_img
```

2. To convert HDR image into perspective video:

```bash
python -m prepare_data.hdri_to_pers_vid \
    --hdri_folder examples/hdri \
    --output_folder examples/hdri_pers_vid
```

3. To convert 360 video into perspective video:

```bash
python -m prepare_data.pano_vid_to_pers \
    --pano_vid_folder examples/360vid \
    --output_folder examples/360vid_pers
```