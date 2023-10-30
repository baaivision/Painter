import torch
import torch.nn.functional as F
import numpy as np
import cv2

from PIL import Image


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


class Cache(list):
    def __init__(self, max_size=0):
        super().__init__()
        self.max_size = max_size

    def append(self, x):
        if self.max_size <= 0:
            return
        super().append(x)
        if len(self) > self.max_size:
            self.pop(0)


@torch.no_grad()
def run_one_image(img, tgt, model, device):
    x = torch.tensor(img)
    # make it a batch-like
    x = torch.einsum('nhwc->nchw', x)

    tgt = torch.tensor(tgt)
    # make it a batch-like
    tgt = torch.einsum('nhwc->nchw', tgt)

    bool_masked_pos = torch.zeros(model.patch_embed.num_patches)
    bool_masked_pos[model.patch_embed.num_patches//2:] = 1
    bool_masked_pos = bool_masked_pos.unsqueeze(dim=0)
    valid = torch.ones_like(tgt)

    if model.seg_type == 'instance':
        seg_type = torch.ones([valid.shape[0], 1])
    else:
        seg_type = torch.zeros([valid.shape[0], 1])
    
    feat_ensemble = 0 if len(x) > 1 else -1
    _, y, mask = model(x.float().to(device), tgt.float().to(device), bool_masked_pos.to(device), valid.float().to(device), seg_type.to(device), feat_ensemble)
    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()

    output = y[0, y.shape[1]//2:, :, :]
    output = torch.clip((output * imagenet_std + imagenet_mean) * 255, 0, 255)
    return output


def inference_image(model, device, img_path, img2_paths, tgt2_paths, out_path, ovl_path=None, return_mask=False, upscale=True):
    res, hres = 448, 448

    image = Image.open(img_path).convert("RGB")
    input_image = np.array(image)
    size = image.size
    image = np.array(image.resize((res, hres))) / 255.

    image_batch, target_batch = [], []
    for img2_path, tgt2_path in zip(img2_paths, tgt2_paths):
        img2 = Image.open(img2_path).convert("RGB")
        img2 = img2.resize((res, hres))
        img2 = np.array(img2) / 255.

        tgt2 = Image.open(tgt2_path).convert("RGB")
        tgt2 = tgt2.resize((res, hres), Image.NEAREST)
        tgt2 = np.array(tgt2) / 255.

        tgt = tgt2  # tgt is not available
        tgt = np.concatenate((tgt2, tgt), axis=0)
        img = np.concatenate((img2, image), axis=0)
    
        assert img.shape == (2*res, res, 3), f'{img.shape}'
        # normalize by ImageNet mean and std
        img = img - imagenet_mean
        img = img / imagenet_std

        assert tgt.shape == (2*res, res, 3), f'{img.shape}'
        # normalize by ImageNet mean and std
        tgt = tgt - imagenet_mean
        tgt = tgt / imagenet_std

        image_batch.append(img)
        target_batch.append(tgt)

    img = np.stack(image_batch, axis=0)
    tgt = np.stack(target_batch, axis=0)
    """### Run SegGPT on the image"""
    # make random mask reproducible (comment out to make it change)
    torch.manual_seed(2)
    output = run_one_image(img, tgt, model, device)
    if upscale:
        output = F.interpolate(
            output[None, ...].permute(0, 3, 1, 2), 
            size=[size[1], size[0]], 
            mode='nearest',
        ).permute(0, 2, 3, 1)[0]
        image = input_image / 255  # needed for overlay
    output =  output.numpy()

    if ovl_path is not None:
        overlay = Image.fromarray((image * 255 * (0.6 * output / 255 + 0.4)).astype(np.uint8))
        overlay.save(ovl_path)

    output_img = Image.fromarray(output.astype(np.uint8))
    output_img.save(out_path)

    if return_mask:
        return output


def inference_video(model, device, vid_path, num_frames, img2_paths, tgt2_paths, out_path, ovl_path):
    res, hres = 448, 448

    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height), True)
    
    if ovl_path is not None:
        overlay_video_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height), True)

    if img2_paths is None:
        _, frame = cap.read()
        img2 = Image.fromarray(frame[:, :, ::-1]).convert('RGB')
    else:
        img2 = Image.open(img2_paths[0]).convert("RGB")
    img2 = img2.resize((res, hres))
    img2 = np.array(img2) / 255.

    tgt2 = Image.open(tgt2_paths[0]).convert("RGB")
    tgt2 = tgt2.resize((res, hres), Image.NEAREST)
    tgt2 = np.array(tgt2) / 255.

    frames_cache, target_cache = Cache(num_frames), Cache(num_frames)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image_batch, target_batch = [], []
        image = Image.fromarray(frame[:, :, ::-1]).convert('RGB')
        input_image = np.array(image)
        size = image.size
        image = np.array(image.resize((res, hres))) / 255.

        for prompt, target in zip([img2] + frames_cache, [tgt2] + target_cache):
            tgt = target  # tgt is not available
            tgt = np.concatenate((target, tgt), axis=0)
            img = np.concatenate((prompt, image), axis=0)

            assert img.shape == (2*res, res, 3), f'{img.shape}'
            # normalize by ImageNet mean and std
            img = img - imagenet_mean
            img = img / imagenet_std

            assert tgt.shape == (2*res, res, 3), f'{img.shape}'
            # normalize by ImageNet mean and std
            tgt = tgt - imagenet_mean
            tgt = tgt / imagenet_std

            image_batch.append(img)
            target_batch.append(tgt)

        img = np.stack(image_batch, axis=0)
        tgt = np.stack(target_batch, axis=0)
        
        torch.manual_seed(2)
        output = run_one_image(img, tgt, model, device)

        frames_cache.append(image)
        target_cache.append(
            output.mean(-1) \
                .gt(128).float() \
                .unsqueeze(-1).expand(-1, -1, 3) \
                .numpy()
        )

        output = F.interpolate(
            output[None, ...].permute(0, 3, 1, 2), 
            size=[size[1], size[0]], 
            mode='nearest',
        ).permute(0, 2, 3, 1)[0].numpy()
        overlay = input_image * (0.6 * output / 255 + 0.4)
        output_video_writer.write(np.ascontiguousarray(output.astype(np.uint8)[:, :, ::-1]))

        if ovl_path is not None:
            overlay_video_writer.write(np.ascontiguousarray(overlay.astype(np.uint8)[:, :, ::-1]))
    
    output_video_writer.release()
    overlay_video_writer.release()
