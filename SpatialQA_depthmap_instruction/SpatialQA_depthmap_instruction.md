SpatialBot uses depth information from sensors if possible. Convert depth information to millimeter unit so DepthAPI and SpatialBot can accurately query depth information.

Otherwise, we convert RGB to RGB-D with [ZoeDepth](https://github.com/isl-org/ZoeDepth) and save depth information in uint16 by default, which covers 0-65535mm.
(You can also do it in e.g. uint64 to cover more distance ranges).

After downloading ZoeDepth, please replace ```save_raw_16bit``` function in ```zoedepth/utils/misc.py``` with:
```
def save_raw_16bit(depth, fpath="raw.png"):
    if isinstance(depth, torch.Tensor):
        depth = depth.squeeze().cpu().numpy()
    assert isinstance(depth, np.ndarray), "Depth must be a torch tensor or numpy array"
    assert depth.ndim == 2, "Depth must be 2D"
    depth = depth * 1000
    depth = depth.astype(np.uint16)
    depth = Image.fromarray(depth)
    depth.save(fpath)
```

Use or [codes](https://github.com/BAAI-DCAI/SpatialBot/blob/main/SpatialQA_depthmap_instruction/zoe.py) to run ZoeDepth for a directory:
```
python zoe.py --img-dir </path/of/rgb> --save-dir </path/to/save/depth>
```