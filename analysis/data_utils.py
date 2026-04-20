import numpy as np
import torch
from pathlib import Path
import nibabel as nib
import json
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.run.load_pretrained_weights import load_pretrained_weights
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from sklearn.decomposition import PCA

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Helper functions: loading image, extracting patch, and initializing predictor
import nibabel as nib
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.run.load_pretrained_weights import load_pretrained_weights
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
import torch

def find_case_file(image_dir: Path, case_name: str):
    p = Path(image_dir) / case_name
    if p.exists():
        return p
    # try to find by pattern
    files = list(p.parent.rglob(case_name)) if p.parent.exists() else []
    if files:
        return files[0]
    raise FileNotFoundError(f'Case file {case_name} not found under {image_dir}')

def load_case_array(case_file: Path):
    if case_file.suffix == '.npz':
        arr = np.load(case_file)
        # heuristics for key
        for k in ('data','x','arr_0','image','images'):
            if k in arr:
                return arr[k]  # may be C,D,H,W or D,H,W
        return arr[list(arr.files)[0]]
    elif case_file.suffix == '.npy':
        return np.load(case_file)
    else:
        img = nib.load(str(case_file))
        return img.get_fdata()

def ensure_4d_CDHW(x: np.ndarray):
    # convert to (C,D,H,W)
    if x.ndim == 3:
        return np.expand_dims(x, 0)
    if x.ndim == 4:
        # ambiguous: could be (B,C,D,H,W) or (C,D,H,W) - we assume (C,D,H,W)
        return x
    if x.ndim == 5:
        return x[0]
    raise ValueError(f'Unexpected image array shape: {x.shape}')

def extract_patch_from_array(arr: np.ndarray, offset: tuple, patch_size: tuple):
    # arr: (C,D,H,W)
    C, D, H, W = arr.shape
    d0 = int(np.clip(offset[0], 0, D))
    h0 = int(np.clip(offset[1], 0, H))
    w0 = int(np.clip(offset[2], 0, W))
    d1 = min(d0 + patch_size[0], D)
    h1 = min(h0 + patch_size[1], H)
    w1 = min(w0 + patch_size[2], W)
    
    # Extract available region
    patch = arr[:, d0:d1, h0:h1, w0:w1].copy()
    
    # Zero pad if necessary
    pad_d = patch_size[0] - (d1 - d0)
    pad_h = patch_size[1] - (h1 - h0)
    pad_w = patch_size[2] - (w1 - w0)
    
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        patch = np.pad(patch, ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    
    return patch, (d0, h0, w0)

def initialize_untrained_predictor(predictor, plans_file, dataset_json):
    """Initialize predictor with an untrained model based on plans file.
    def initialize_untrained_predictor(predictor, plans_file, dataset_json):
    Args:
        predictor: nnUNetPredictor instance
        plans_file: Path to the plans.json file
    """
    # Load network from plans
    plans_file = Path(plans_file)
    if not plans_file.exists():
        raise FileNotFoundError(f"Plans file not found: {plans_file}")
    
    # Load and parse plans
    with open(plans_file) as f:
        plans = json.load(f)
    
    # Initialize managers
    plans_manager = PlansManager(plans_file)
    configuration = "3d_32x160x128_b10"
    if configuration not in plans['configurations']:
        configuration = list(plans['configurations'].keys())[0]
    
    configuration_manager = plans_manager.get_configuration(configuration)
    label_manager = plans_manager.get_label_manager(dataset_json)
    num_input_channels = determine_num_input_channels(plans_manager, configuration_manager,dataset_json)
    # Get network parameters from configuration manager
    network = get_network_from_plans(
        arch_class_name=configuration_manager.network_arch_class_name,
        arch_kwargs=configuration_manager.network_arch_init_kwargs,
        arch_kwargs_req_import=configuration_manager.network_arch_init_kwargs_req_import,
        input_channels=num_input_channels,
        output_channels=label_manager.num_segmentation_heads,
        allow_init=True,
        deep_supervision=True
    )
    network.to(predictor.device)
    
    # Initialize predictor with untrained network
    predictor.network = network
    predictor.plans_manager = plans_manager
    predictor.configuration_manager = configuration_manager
    predictor.label_manager = label_manager
    predictor.plans = plans
    predictor.configuration_name = configuration
    
    print(f"Initialized untrained network from {plans_file}")
    print(f"Using configuration: {configuration}")
    print(f"Architecture: {configuration_manager.network_arch_class_name}")
    print(f"Input channels: {num_input_channels}, Output channels: {label_manager.num_segmentation_heads}")
    return predictor

def extract_activation_vector(predictor, input_tensor: torch.Tensor, target_module_name: str = None) -> np.ndarray:
    """Run forward and capture a single pooled feature vector for the target module.
    We perform global average pooling over spatial dims of the activation to get a (C,) vector.
    """
    net = predictor.network.to(DEVICE)
    activations = {}
    hooks = []

    def find_module_by_name(root, name):
        if not name:
            return None
        cur = root
        for p in name.split('.'):
            if p.isdigit():
                cur = cur[int(p)]
            else:
                cur = getattr(cur, p)
        return cur

    # select module: default to last encoder stage if available
    if target_module_name is None and hasattr(net, 'encoder'):
        enc = net.encoder
        if hasattr(enc, 'stages') and len(enc.stages) > 0:
            target = enc.stages[-1]
            target_name = 'encoder.stages.-1'
        else:
            target = enc
            target_name = 'encoder'
    elif target_module_name is not None:
        target = find_module_by_name(net, target_module_name)
        target_name = target_module_name
    else:
        target = net
        target_name = 'network'

    def hook_fn(m, inp, out):
        activations[target_name] = out.detach().cpu()

    hooks.append(target.register_forward_hook(hook_fn))
    was_training = net.training
    net.eval()
    with torch.no_grad():
        x = input_tensor.to(DEVICE).float()
        out = net(x)

    for h in hooks:
        h.remove()
    if was_training:
        net.train()

    act = activations[target_name]  # [B,C,(D,),H,W] or [B,C,H,W]
    a = act.cpu().numpy()
    # global average pool spatial dims for first batch
    if a.ndim == 5:
        # B,C,D,H,W -> pool D,H,W -> C vector
        v = a[0].mean(axis=(1,2,3))
    elif a.ndim == 4:
        # B,C,H,W -> pool H,W
        v = a[0].mean(axis=(1,2))
    else:
        raise RuntimeError('Unexpected activation rank: ' + str(a.shape))
    return v

# Texture Enhancement/Reduction Functions with Advanced Options

def enhance_reduce_texture(patch, enhancement_factor=1.5, mode='enhance', method='unsharp', 
                          sigma=None, preserve_range=True, adaptive=False):
    """
    Enhance or reduce texture (high-frequency features) in a 3D patch.
    
    Args:
        patch: numpy array of shape (C, D, H, W) where C is channels
        enhancement_factor: float > 1 to enhance, 0 < factor < 1 to reduce
        mode: 'enhance' or 'reduce'
        method: 'unsharp' (recommended) or 'fft_bandpass'
        sigma: Gaussian blur sigma for unsharp method. Auto-scales with patch size if None
        preserve_range: If True, clips output to original intensity range
        adaptive: If True, apply enhancement only where intensity changes significantly
        
    Returns:
        modified_patch: numpy array with same shape as input
    """
    from scipy.ndimage import gaussian_filter
    
    # Work with first channel (C=0) for simplicity
    if patch.ndim == 4:
        patch_data = patch[0].copy()
    else:
        patch_data = patch.copy()
    
    # Auto-scale sigma based on patch size
    if sigma is None:
        avg_dim = np.mean(patch_data.shape)
        sigma = max(1.0, avg_dim / 80.0)
    
    # Store original range for clipping
    orig_min, orig_max = patch_data.min(), patch_data.max()
    orig_range = orig_max - orig_min + 1e-6
    
    if method == 'unsharp':
        # Unsharp Masking: captures fine textures and details naturally
        patch_float = patch_data.astype(np.float32)
        blurred = gaussian_filter(patch_float, sigma=sigma)
        high_pass = patch_float - blurred
        
        # Adaptive masking: only enhance where there's contrast
        if adaptive:
            local_std = gaussian_filter((high_pass ** 2), sigma=sigma) ** 0.5
            adaptive_mask = local_std / (local_std.max() + 1e-6)
            high_pass = high_pass * adaptive_mask
        
        # Apply unsharp mask
        if mode == 'enhance':
            modified = patch_float + (enhancement_factor - 1.0) * high_pass
        elif mode == 'reduce':
            factor = 1.0 / (enhancement_factor + 1e-6)
            modified = patch_float + (factor - 1.0) * high_pass
        else:
            raise ValueError(f"Unknown mode: {mode}")
    
    elif method == 'fft_bandpass':
        # FFT-based band-pass filtering with Gaussian window
        fft_patch = np.fft.fftn(patch_data)
        fft_shifted = np.fft.fftshift(fft_patch)
        magnitude = np.abs(fft_shifted)
        phase = np.angle(fft_shifted)
        
        D, H, W = patch_data.shape
        d_center, h_center, w_center = D // 2, H // 2, W // 2
        
        # Create distance matrix from center
        d_idx = np.arange(D)
        h_idx = np.arange(H)
        w_idx = np.arange(W)
        dd, hh, ww = np.meshgrid(d_idx - d_center, h_idx - h_center, w_idx - w_center, indexing='ij')
        distance = np.sqrt(dd**2 + hh**2 + ww**2)
        
        max_distance = np.sqrt(d_center**2 + h_center**2 + w_center**2)
        normalized_distance = distance / (max_distance + 1e-6)
        
        # Smoother Gaussian band-pass filter
        band_pass = np.exp(-((normalized_distance - 0.3) ** 2) / (2 * 0.2 ** 2))
        band_pass = np.clip(band_pass, 0, 1)
        
        if mode == 'enhance':
            modified_magnitude = magnitude * (1 + band_pass * (enhancement_factor - 1))
        elif mode == 'reduce':
            factor = 1.0 / (enhancement_factor + 1e-6)
            modified_magnitude = magnitude * (1 + band_pass * (factor - 1))
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Reconstruct
        modified_fft_shifted = modified_magnitude * np.exp(1j * phase)
        modified_fft = np.fft.ifftshift(modified_fft_shifted)
        modified = np.real(np.fft.ifftn(modified_fft))
    
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'unsharp' or 'fft_bandpass'")
    
    # Preserve original range
    if preserve_range:
        modified = np.clip(modified, orig_min, orig_max)
    
    # Reconstruct full patch with channels
    if patch.ndim == 4:
        modified_patch = patch.copy().astype(np.float32)
        modified_patch[0] = modified
        return modified_patch
    else:
        return modified


def enhance_multiscale_texture(patch, enhancement_factor=1.5, mode='enhance', scales=None):
    """
    Multi-scale texture enhancement: apply enhancement at multiple scales and combine.
    Better for capturing details at different sizes.
    
    Args:
        patch: numpy array of shape (C, D, H, W)
        enhancement_factor: amplification strength
        mode: 'enhance' or 'reduce'
        scales: list of sigma values (auto-generated if None)
        
    Returns:
        enhanced_patch: combined multi-scale enhancement
    """
    if patch.ndim == 4:
        patch_data = patch[0].copy()
    else:
        patch_data = patch.copy()
    
    if scales is None:
        # Default scales: fine, medium, coarse
        avg_dim = np.mean(patch_data.shape)
        scales = [avg_dim / 120.0, avg_dim / 60.0, avg_dim / 30.0]
    
    # Enhance at each scale
    enhanced_scales = []
    for sigma in scales:
        enhanced = enhance_reduce_texture(
            patch, 
            enhancement_factor=enhancement_factor, 
            mode=mode, 
            method='unsharp',
            sigma=sigma,
            adaptive=False
        )
        enhanced_scales.append(enhanced[0] if enhanced.ndim == 4 else enhanced)
    
    # Combine scales with equal weighting
    combined = np.mean(enhanced_scales, axis=0)
    
    if patch.ndim == 4:
        result = patch.copy().astype(np.float32)
        result[0] = combined
        return result
    else:
        return combined


def adjust_intensity(patch, intensity_factor=1.2):
    """
    Adjust the intensity (brightness) of a patch.
    
    Args:
        patch: numpy array of shape (C, D, H, W) or (D, H, W)
        intensity_factor: float > 1 to brighten, 0 < factor < 1 to darken
        
    Returns:
        modified_patch: numpy array with same shape as input
    """
    # Normalize, apply factor, then rescale
    original_min = patch.min()
    original_max = patch.max()
    original_range = original_max - original_min + 1e-6
    
    normalized = (patch - original_min) / original_range
    brightened = normalized * intensity_factor
    brightened = np.clip(brightened, 0, 1.0)
    modified_patch = brightened * original_range + original_min
    
    return modified_patch

def create_texture_modified_patches(patch, enhancement_factors=None):
    """
    Create multiple texture-modified versions of a single patch.
    
    Args:
        patch: Original patch array (C, D, H, W)
        enhancement_factors: List of factors to apply. Default includes reduction and enhancement.
                           Factors < 1.0 = reduce texture, > 1.0 = enhance texture
    
    Returns:
        dict with keys:
            'patches': list of modified patch arrays
            'factors': list of factors used
            'labels': list of descriptive labels
    """
    if enhancement_factors is None:
        # Default: range from strong reduction to strong enhancement
        enhancement_factors = [0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0]
    
    modified_patches = []
    labels = []
    
    for factor in enhancement_factors:
        if factor < 1.0:
            # Reduce texture
            mode = 'reduce'
            actual_factor = 1.0 / factor  # Convert to reduction factor
            label = f'Reduce {factor:.2f}x'
        elif factor == 1.0:
            # Original (no modification)
            modified_patches.append(patch.copy())
            labels.append('Original')
            continue
        else:
            # Enhance texture
            mode = 'enhance'
            actual_factor = factor
            label = f'Enhance {factor:.2f}x'
        
        modified = enhance_reduce_texture(
            patch,
            enhancement_factor=actual_factor,
            mode=mode,
            method='unsharp',
            adaptive=False
        )
        modified_patches.append(modified)
        labels.append(label)
    
    return {
        'patches': modified_patches,
        'factors': enhancement_factors,
        'labels': labels
    }


def compute_embeddings_for_texture_variants(patch, patch_info, pretrained_models, plans_file, dataset_json, patch_size=(32, 160, 128), target_module=None, enhancement_factors=None):
    """
    Compute embeddings for texture-modified variants of a single patch across multiple models.
    
    Args:
        patch: Original patch array (C, D, H, W)
        patch_info: Tuple (case_name, offset) for identification
        pretrained_models: List of (model_path, model_name) tuples
        patch_size: Size of patch
        target_module: Target module for activation extraction
        enhancement_factors: List of enhancement factors to use
    
    Returns:
        dict: model_name -> {
            'embeddings': (N_variants, embedding_dim),
            'coords': (N_variants, 2) after PCA,
            'labels': list of variant labels,
            'factors': list of factors used
        }
    """
    # Create texture variants
    variants = create_texture_modified_patches(patch, enhancement_factors=enhancement_factors)
    
    results = {}
    for m_idx, model in enumerate(pretrained_models):
        pretrained_weights_file, model_name = model
        print(f'Processing model {m_idx+1}/{len(pretrained_models)}: {model_name}')
        
        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False, 
                                    perform_everything_on_device=True, device=DEVICE)
        initialize_untrained_predictor(predictor, plans_file, dataset_json)
        
        if pretrained_weights_file.exists():
            print(f'  Loading pretrained weights from {pretrained_weights_file}')
            load_pretrained_weights(predictor.network, pretrained_weights_file)
        
        embeddings = []
        for variant_patch in variants['patches']:
            x_in = np.expand_dims(variant_patch, 0)  # (1,C,D,H,W)
            t = torch.from_numpy(x_in).float()
            try:
                v = extract_activation_vector(predictor, t, target_module_name=target_module)
            except Exception as e:
                print(f'  Failed to extract activation for variant: {e}')
                v = np.zeros(64)
            embeddings.append(v)
        
        embeddings = np.stack(embeddings, axis=0)  # (N_variants, embedding_dim)
        
        # Apply PCA to 2D for visualization
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(embeddings)
        
        results[str(model_name)] = {
            'embeddings': embeddings,
            'coords': coords,
            'labels': variants['labels'],
            'factors': variants['factors'],
            'patch_info': patch_info,
            'patches': variants['patches']
        }
        
        # Free GPU memory
        del predictor
        torch.cuda.empty_cache()
    
    return results


def create_intensity_modified_patches(patch, intensity_factors=None):
    """
    Create multiple intensity-modified versions of a single patch.
    
    Args:
        patch: Original patch array (C, D, H, W)
        intensity_factors: List of factors to apply. Default includes reduction and enhancement.
                          Factors < 1.0 = darken, > 1.0 = brighten
    
    Returns:
        dict with keys:
            'patches': list of modified patch arrays
            'factors': list of factors used
            'labels': list of descriptive labels
    """
    if intensity_factors is None:
        # Default: range from dark to bright
        intensity_factors = [0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0]
    
    modified_patches = []
    labels = []
    
    for factor in intensity_factors:
        if factor < 1.0:
            # Darken
            label = f'Darken {factor:.2f}x'
        elif factor == 1.0:
            # Original (no modification)
            modified_patches.append(patch.copy())
            labels.append('Original')
            continue
        else:
            # Brighten
            label = f'Brighten {factor:.2f}x'
        
        modified = adjust_intensity(patch, factor)
        modified_patches.append(modified)
        labels.append(label)
    
    return {
        'patches': modified_patches,
        'factors': intensity_factors,
        'labels': labels
    }


def compute_embeddings_for_intensity_variants(patch, patch_info, pretrained_models, plans_file, dataset_json, patch_size=(32, 160, 128), target_module=None, intensity_factors=None):
    """
    Compute embeddings for intensity-modified variants of a single patch across multiple models.
    
    Args:
        patch: Original patch array (C, D, H, W)
        patch_info: Tuple (case_name, offset) for identification
        pretrained_models: List of (model_path, model_name) tuples
        patch_size: Size of patch
        target_module: Target module for activation extraction
        intensity_factors: List of intensity factors to use
    
    Returns:
        dict: model_name -> {
            'embeddings': (N_variants, embedding_dim),
            'coords': (N_variants, 2) after PCA,
            'labels': list of variant labels,
            'factors': list of factors used
        }
    """
    # Create intensity variants
    variants = create_intensity_modified_patches(patch, intensity_factors=intensity_factors)
    
    results = {}
    for m_idx, model in enumerate(pretrained_models):
        pretrained_weights_file, model_name = model
        print(f'Processing model {m_idx+1}/{len(pretrained_models)}: {model_name}')
        
        predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=False, 
                                    perform_everything_on_device=True, device=DEVICE)
        initialize_untrained_predictor(predictor, plans_file, dataset_json)
        
        if pretrained_weights_file.exists():
            print(f'  Loading pretrained weights from {pretrained_weights_file}')
            load_pretrained_weights(predictor.network, pretrained_weights_file)
        
        embeddings = []
        for variant_patch in variants['patches']:
            x_in = np.expand_dims(variant_patch, 0)  # (1,C,D,H,W)
            t = torch.from_numpy(x_in).float()
            try:
                v = extract_activation_vector(predictor, t, target_module_name=target_module)
            except Exception as e:
                print(f'  Failed to extract activation for variant: {e}')
                v = np.zeros(64)
            embeddings.append(v)
        
        embeddings = np.stack(embeddings, axis=0)  # (N_variants, embedding_dim)
        
        # Apply PCA to 2D for visualization
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(embeddings)
        
        results[str(model_name)] = {
            'embeddings': embeddings,
            'coords': coords,
            'labels': variants['labels'],
            'factors': variants['factors'],
            'patch_info': patch_info,
            'patches': variants['patches']
        }
        
        # Free GPU memory
        del predictor
        torch.cuda.empty_cache()
    
    return results


