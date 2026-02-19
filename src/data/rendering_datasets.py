import src.data.env_rgbxenv_dataset as env_rgbxenv_dataset
import src.data.env_panovideo_dataset as env_panovideo_dataset


READ_EXAMPLE_FUNC = {
    'env_rgbxenv_hdri': env_rgbxenv_dataset.read_img_example,
    'env_panovideo_hdri': env_panovideo_dataset.read_img_example,
}