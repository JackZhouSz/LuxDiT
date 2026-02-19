
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2
import imageio.v3 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as F_t
from torchvision.transforms.functional import gaussian_blur
try:
    import nvdiffrast.torch as dr
except:
    print('nvdiffrast not found!')
    dr = None

def rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(f <= 0.0031308, f * 12.92, torch.pow(torch.clamp(f, 0.0031308), 1.0/2.4)*1.055 - 0.055)


def srgb_to_rgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(f <= 0.04045, f / 12.92, torch.pow((torch.clamp(f, 0.04045) + 0.055) / 1.055, 2.4))


#----------------------------------------------------------------------------
# Vector operations
#----------------------------------------------------------------------------

def dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sum(x*y, -1, keepdim=True)

def reflect(x: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    return 2*dot(x, n)*n - x

def length(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return torch.sqrt(torch.clamp(dot(x,x), min=eps)) # Clamp to avoid nan gradients because grad(sqrt(0)) = NaN

def safe_normalize(x: torch.Tensor, eps: float =1e-20) -> torch.Tensor:
    return x / length(x, eps)

def cube_to_dir(s, x, y):
    if s == 0:   rx, ry, rz = torch.ones_like(x), -y, -x
    elif s == 1: rx, ry, rz = -torch.ones_like(x), -y, x
    elif s == 2: rx, ry, rz = x, torch.ones_like(x), y
    elif s == 3: rx, ry, rz = x, -torch.ones_like(x), -y
    elif s == 4: rx, ry, rz = x, -y, torch.ones_like(x)
    elif s == 5: rx, ry, rz = -x, -y, -torch.ones_like(x)
    return torch.stack((rx, ry, rz), dim=-1)

def latlong_to_cubemap(latlong_map, res):
    cubemap = torch.zeros(6, res[0], res[1], latlong_map.shape[-1], dtype=torch.float32, device='cuda')
    for s in range(6):
        gy, gx = torch.meshgrid(torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device='cuda'), 
                                torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device='cuda'),
                                indexing='ij')
        v = safe_normalize(cube_to_dir(s, gx, gy))

        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
        texcoord = torch.cat((tu, tv), dim=-1)

        cubemap[s, ...] = dr.texture(latlong_map[None, ...], texcoord[None, ...], filter_mode='linear')[0]
    return cubemap

def latlong_to_cubemap_torch(latlong_map, res): # no dr dependency
    ndim = latlong_map.ndim
    batch_size = 1 if ndim == 3 else latlong_map.shape[0]
    if ndim == 3:
        latlong_map = latlong_map.unsqueeze(0)
    device = latlong_map.device
    cubemap = torch.zeros(batch_size, 6, res[0], res[1], latlong_map.shape[-1], dtype=torch.float32, device=device)
    
    for s in range(6):
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device=device),
            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device=device),
            indexing='ij'
        )
        v = safe_normalize(cube_to_dir(s, gx, gy))  # Shape: (res[0], res[1], 3)
        
        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5  # Shape: (res[0], res[1], 1)
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi  # Shape: (res[0], res[1], 1)
        
        tu = tu * 2 - 1  # Shape: (res[0], res[1], 1)
        tv = tv * 2 - 1  # Shape: (res[0], res[1], 1)

        grid = torch.cat((tu, tv), dim=-1)  # Shape: (res[0], res[1], 2)
        grid = grid.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # Shape: (batch_size, res[0], res[1], 2)
        texture = latlong_map.permute(0, 3, 1, 2)  # Shape: (1, channels, H_latlong, W_latlong)
        
        # Perform sampling
        sampled = F.grid_sample(texture, grid, mode='bilinear', padding_mode='border', align_corners=False)
        # Result shape: (1, channels, res[0], res[1])
        
        # Reshape and store in cubemap
        # cubemap[s] = sampled.squeeze(0).permute(1, 2, 0)  # Shape: (res[0], res[1], channels)
        cubemap[:, s] = sampled.permute(0, 2, 3, 1)  # Shape: (batch_size, res[0], res[1], channels)
    
    if ndim == 3:
        cubemap = cubemap.squeeze(0)

    return cubemap

def prefilter_cubemap(cubemap, roughness, max_sigma=10.0):
    """
    Prefilter a cubemap based on roughness to simulate rough surface reflections.
    
    Args:
        cubemap (torch.Tensor): Cubemap of shape [6, H, W, C].
        roughness (float): Roughness value in [0, 1]. Higher values increase blur.
        max_sigma (float): Maximum sigma for Gaussian blur to cap blur strength.
    
    Returns:
        torch.Tensor: Prefiltered cubemap of same shape as input.
    """
    if roughness <= 0:
        return cubemap.clone()  # No filtering for roughness = 0
    
    # Map roughness to blur parameters
    # Roughness [0, 1] -> kernel_size [3, 21], sigma [0.5, max_sigma]
    kernel_size = int(3 + roughness * 18)  # 3 to 21
    kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size  # Ensure odd
    sigma = 0.5 + roughness * (max_sigma - 0.5)  # 0.5 to max_sigma
    
    # Initialize output cubemap
    prefiltered = torch.zeros_like(cubemap)  # [6, H, W, C]
    
    # Blur each face
    for face in range(6):
        face_data = cubemap[face].permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        blurred_face = gaussian_blur(face_data, kernel_size=kernel_size, sigma=sigma)
        prefiltered[face] = blurred_face.squeeze(0).permute(1, 2, 0)  # [H, W, C]
    
    return prefiltered

def cubemap_sample_torch(cubemap, dirs):
    """Sample from cubemap with smoother transitions across face edges."""
    device = cubemap.device
    N = dirs.shape[0]  # Number of directions (e.g., H * W)
    C = cubemap.shape[-1]  # Number of channels (e.g., 3 for RGB)
    h, w = cubemap.shape[-3:-1]  # Cubemap face resolution
    ndim = cubemap.ndim
    batch_size = cubemap.shape[0] if ndim == 5 else 1  # Handle batch size
    if ndim == 4:
        cubemap = cubemap.unsqueeze(0)

    # Normalize directions
    dirs = safe_normalize(dirs)  # Shape: (N, 3)
    
    # Compute absolute directions and max axis
    abs_dirs = torch.abs(dirs)  # Shape: (N, 3)
    max_axis = torch.argmax(abs_dirs, dim=-1)  # Shape: (N,)
    max_vals = abs_dirs[torch.arange(N, device=device), max_axis]  # Shape: (N,)
    
    # Get signs for face selection
    sign = torch.sign(dirs[torch.arange(N, device=device), max_axis])  # Shape: (N,)
    
    # Preallocate result tensor
    result = torch.zeros(batch_size, N, C, device=device)  # Shape: (N, channels)
    
    # Process each face iteratively
    for s in range(6):
        # Determine face index based on max_axis and sign
        face_idx = torch.where(max_axis == 0, torch.where(sign > 0, 0, 1),  # +X, -X
                               torch.where(max_axis == 1, torch.where(sign > 0, 2, 3),  # +Y, -Y
                                           torch.where(sign > 0, 4, 5)))  # +Z, -Z
        mask = (face_idx == s)  # Shape: (N,)
        
        if not mask.any():
            continue  # Skip if no rays hit this face
        
        # Select directions for this face
        dirs_s = dirs[mask]  # Shape: (N_s, 3)
        max_axis_s = max_axis[mask]  # Shape: (N_s,)
        max_vals_s = max_vals[mask]  # Shape: (N_s,)
        
        # Compute UV coordinates based on face
        # We need to map the two non-max axes to U and V
        u = torch.zeros_like(max_vals_s)
        v = torch.zeros_like(max_vals_s)
        
        # Face-specific UV mapping
        if s == 0:  # +X
            u = -dirs_s[..., 2] / max_vals_s  # -Z
            v = -dirs_s[..., 1] / max_vals_s  # -Y
        elif s == 1:  # -X
            u = dirs_s[..., 2] / max_vals_s   # +Z
            v = -dirs_s[..., 1] / max_vals_s  # -Y
        elif s == 2:  # +Y
            u = dirs_s[..., 0] / max_vals_s   # +X
            v = dirs_s[..., 2] / max_vals_s   # +Z
        elif s == 3:  # -Y
            u = dirs_s[..., 0] / max_vals_s   # +X
            v = -dirs_s[..., 2] / max_vals_s  # -Z
        elif s == 4:  # +Z
            u = dirs_s[..., 0] / max_vals_s   # +X
            v = -dirs_s[..., 1] / max_vals_s  # -Y
        elif s == 5:  # -Z
            u = -dirs_s[..., 0] / max_vals_s  # -X
            v = -dirs_s[..., 1] / max_vals_s  # -Y
        
        # Normalize UV to [0, 1]
        u = (u + 1) * 0.5  # Shape: (N_s,)
        v = (v + 1) * 0.5  # Shape: (N_s,)
        
        # Convert UV to grid_sample format: [0, 1] -> [-1, 1]
        u = u * 2 - 1
        v = v * 2 - 1
        grid = torch.stack((u, v), dim=-1)  # Shape: (N_s, 2)
        
        # Reshape for grid_sample: expects (batch, H, W, 2)
        grid = grid.view(-1, 1, 1, 2)  # Shape: (N_s, 1, 1, 2)
        
        # Prepare texture: grid_sample expects (batch, channels, height, width)
        texture = cubemap[:, s].permute(0, 3, 1, 2).flatten(0,1).unsqueeze(0)  # Shape: (batch_size, channels, h, w)
        texture = texture.expand(grid.shape[0], -1, -1, -1)  # Shape: (N_s, channels, h, w)

        # Perform sampling
        sampled = F.grid_sample(texture, grid, mode='bilinear', padding_mode='border', align_corners=False)
        # Result shape: (batch_size * N_s, channels, 1, 1)

        # Reshape and store in result
        result[:, mask] = sampled.view(-1, batch_size, C).permute(1, 0, 2)  # Shape: (batch_size, N_s, channels)
    
    if ndim == 4:
        result = result.squeeze(0) # Shape: (N, channels)

    return result


# envmap part
def latlong_vec(res, device=None):
    gy, gx = torch.meshgrid(torch.linspace( 0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device=device), 
                            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device=device),
                            indexing='ij')
    
    sintheta, costheta = torch.sin(gy*np.pi), torch.cos(gy*np.pi)
    sinphi, cosphi     = torch.sin(gx*np.pi), torch.cos(gx*np.pi)
    
    dir_vec = torch.stack((
        sintheta*sinphi, 
        costheta, 
        -sintheta*cosphi
        ), dim=-1)
    # return dr.texture(cubemap[None, ...], dir_vec[None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')[0]
    return dir_vec #[H, W, 3]


def rotate_x(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[1,  0, 0, 0], 
                         [0,  c, s, 0], 
                         [0, -s, c, 0], 
                         [0,  0, 0, 1]], dtype=torch.float32, device=device)

def rotate_y(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[ c, 0, s, 0], 
                         [ 0, 1, 0, 0], 
                         [-s, 0, c, 0], 
                         [ 0, 0, 0, 1]], dtype=torch.float32, device=device)

def envmap_vec(res, device=None):
    return -latlong_vec(res, device).flip(0).flip(1) #[H, W, 3]

def envmap_xfm(vec, env_rot, cam_rot):
    # env_rot: envmap rotation
    # cam_rot: camera rotation, camera2world
    vec_inv = vec @ env_rot[:3, :3]
    vec_inv = vec_inv @ cam_rot[:3, :3]
    return vec_inv

def get_ideal_ball(size, flip_x=False):
    """
    Generate normal ball for specific size 
    Normal map is x "left", y up, z into the screen
    @params
        - size (int) - single value of height and width
    @return:
        - normal_map (np.array) - normal map [size, size, 3]
        - mask (np.array) - mask that make a valid normal map [size,size]
    """
    # we flip x to match sobel operator
    x = torch.linspace(1, -1, size)
    y = torch.linspace(1, -1, size)
    x = x.flip(dims=(-1,)) if not flip_x else x
    y, x = torch.meshgrid(y, x)
    z = (1 - x**2 - y**2)
    mask = z >= 0
    # clean up invalid value outsize the mask
    x = x * mask
    y = y * mask
    z = z * mask
    # get real z value
    z = torch.sqrt(z)
    
    # clean up normal map value outside mask 
    normal_map = torch.cat([x[..., None], y[..., None], z[..., None]], dim=-1)
    # normal_map = normal_map.numpy()
    # mask = mask.numpy()
    return normal_map, mask

def get_ref_vector(normal, incoming_vector):
    #R = 2(N ⋅ I)N - I
    R = 2 * (normal * incoming_vector).sum(-1, keepdims=True) * normal - incoming_vector
    return R

def envmap_chrome_ball(size):
    normal_map, mask = get_ideal_ball(size, flip_x=False)
    vec_ref = get_ref_vector(normal_map, torch.tensor([0, 0, 1], dtype=torch.float32))
    return vec_ref

def luminance(rgb):
    lumi = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    return lumi[..., None]

def rgb2srgb(rgb):
    ten_where = torch.where if isinstance(rgb, torch.Tensor) else np.where
    return ten_where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb**(1/2.4) - 0.055)

def reinhard(x, max_point=16):
    # lumi = 0.2126 * x[..., 0] + 0.7152 * x[..., 1] + 0.0722 * x[..., 2]
    # lumi = lumi[..., None]
    # y_rein = x * (1 + lumi / (max_point ** 2)) / (1 + lumi)
    # y_rein = x / (1+x)
    y_rein = x * (1 + x / (max_point ** 2)) / (1 + x)
    return y_rein

def percentile_tone(
    numpy_img, gamma=2.4, percentile=99, max_mapping=0.9, 
    clip=True, alpha=None, with_gamma=True, ret_alpha=False
):
    # Borrowed from StyleLight
    if with_gamma:
        power_numpy_img = np.power(numpy_img, 1 / gamma)
    else:
        power_numpy_img = numpy_img
    non_zero = power_numpy_img > 0
    if non_zero.any():
        r_percentile = np.percentile(power_numpy_img[non_zero], percentile)
    else:
        r_percentile = np.percentile(power_numpy_img, percentile)
    if alpha is None:
        alpha = max_mapping / (r_percentile + 1e-10)
    tonemapped_img = np.multiply(alpha, power_numpy_img)

    if clip:
        tonemapped_img = np.clip(tonemapped_img, 0, 1)

    tonemapped_img = tonemapped_img.astype('float32')

    if ret_alpha:
        return tonemapped_img, alpha
    else:
        return tonemapped_img

# OCIO version
def ocio_tonemapping(
        ocio_config='configs/colormanagement/config.ocio', 
        src_space='Linear', 
        dst_space='AgX Base sRGB'
    ):
    import PyOpenColorIO as OCIO
    # Use Blender's OCIO config (e.g.,  blender/3.3/datafiles/colormanagement/config.ocio)
    # this exactly matches the color management in Blender
    config = OCIO.Config.CreateFromFile(ocio_config) 
    processor = config.getProcessor(src_space, dst_space).getDefaultCPUProcessor()
    # given RGB: processor.applyRGB(RGB) for the inplace color transformation
    
    def apply_tone_mapping(img):
        img_in = img.copy()
        processor.applyRGB(img_in)
        return img_in
    
    return apply_tone_mapping

def cam_intrinsics(fov, width, height, device=None):
    """
    fov is along the height axis
    """
    focal = 0.5 * height / np.tan(0.5 * fov)
    intrinsics = torch.tensor([
        [focal, 0, 0.5 * width],
        [0, focal, 0.5 * height],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)
    return intrinsics

def get_cam_matrix(phi, theta, t=None, radius=1, device=None):
    z         = np.sin(theta) # elevation angle
    r         = np.cos(theta)
    pos       = torch.tensor([r*np.cos(phi), z, r*np.sin(phi)], dtype=torch.float32, device=device) * radius        
    look_at   = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    up        = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32, device=device)
    if t is not None:
        look_at += torch.tensor(t, dtype=torch.float32, device=device)
    # world_to_cam = util.lookAt(pos, look_at, up) # [4, 4]
    w = safe_normalize(pos - look_at)
    u = safe_normalize(torch.cross(up, w, dim=-1))
    v = safe_normalize(torch.cross(w, u, dim=-1))
    translate = torch.tensor([[1, 0, 0, -pos[0]], 
                              [0, 1, 0, -pos[1]], 
                              [0, 0, 1, -pos[2]], 
                              [0, 0, 0, 1]], dtype=pos.dtype, device=pos.device)
    rotate = torch.tensor([[u[0], u[1], u[2], 0], 
                           [v[0], v[1], v[2], 0], 
                           [w[0], w[1], w[2], 0], 
                           [0, 0, 0, 1]], dtype=pos.dtype, device=pos.device)
    world_to_cam = rotate @ translate
    return world_to_cam

def uv_mesh(width, height, device=None):
    uv = torch.stack(
        torch.meshgrid(torch.arange(width) + 0.5, torch.arange(height) + 0.5, indexing='xy'), dim=-1
    ).float().to(device)
    uv = torch.cat([uv, torch.ones((height, width, 1), device=device)], dim=-1) # [H, W, 3]
    return uv

def ray2zdepth(ray_depth, width, height, fov=0.82, uv=None, device=None):
    if uv is None:
        uv = uv_mesh(width, height, device=device)

    intrinsics = cam_intrinsics(fov, width, height, device=device)
    ray_dir = uv @ torch.inverse(intrinsics).T
    z_depth = ray_depth * ray_dir[..., 2:3] / torch.norm(ray_dir, dim=-1, keepdim=True) # [H, W, 1]
    return z_depth


def depth2disparity(depth):
    if isinstance(depth, torch.Tensor):
        disparity = torch.zeros_like(depth)
    elif isinstance(depth, np.ndarray):
        disparity = np.zeros_like(depth)
    non_negtive_mask = depth > 0
    disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]
    return disparity

def disparity2depth(disparity):
    return depth2disparity(disparity)

def normalize_depth(depth, mask=None, min_percentile=None, max_percentile=None, bg_value=1.0):
    # NOTE: only work on single image, not batch
    depth_m = depth[mask] if mask is not None else depth
    depth_min = depth_m.min() if min_percentile is None else np.percentile(depth_m, min_percentile)
    depth_max = depth_m.max() if max_percentile is None else np.percentile(depth_m, max_percentile)

    depth = (depth - depth_min) / (depth_max - depth_min) # normalize to [0, 1]
    depth = np.clip(depth, 0, 1)
    if mask is not None:
        depth[~mask] = bg_value
    return depth

def read_image(file_name):
    if file_name.endswith('.exr'): # imageio may not properly read exr file
        img = imageio.imread(file_name, flags=cv2.IMREAD_UNCHANGED, plugin='opencv')
    else:
        img = imageio.imread(file_name)
    if img.ndim == 2:
        img = img[..., None]
    return img

def read_video(file_name):
    return imageio.imread(file_name)

def center_crop(img):
    # make sure HWC format
    H, W = img.shape[:2]
    if H == W:
        return img
    elif H > W:
        start = (H - W) // 2
        return img[start:start+W, :, :]
    else:
        start = (W - H) // 2
        return img[:, start:start+H, :]
    
def resize_crop(img, size, resize_only=False, is_depth=False):
    is_tensor = isinstance(img, torch.Tensor)
    img = torch.from_numpy(img) if not is_tensor else img
    img = img.permute(2, 0, 1) if not is_tensor else img
    hi, wi = img.shape[-2:]
    ho, wo = size
    resize_kwargs = {
        'depth': {
            'interpolation': transforms.InterpolationMode.NEAREST,
            'antialias': False
        },
        'rgb': {
            'interpolation': transforms.InterpolationMode.BILINEAR,
            'antialias': True
        }
    }
    if resize_only:
        img = F_t.resize(img, size=(ho, wo), **resize_kwargs['depth' if is_depth else 'rgb'])
    else:
        resize_factor = max(ho / hi, wo / wi)
        hr, wr = int(hi * resize_factor), int(wi * resize_factor)
        img = F_t.resize(img, size=(hr, wr), **resize_kwargs['depth' if is_depth else 'rgb'])
        img = F_t.center_crop(img, output_size=(ho, wo))
    img = img.permute(1, 2, 0) if not is_tensor else img
    return img.numpy() if not is_tensor else img
