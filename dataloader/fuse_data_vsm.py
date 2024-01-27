import pathlib

import cv2
import numpy as np
import kornia.utils
import torch.utils.data

class TrainingData(torch.utils.data.Dataset):
    """
    Load dataset with ir folder path and vis folder path
    """

    # TODO: remove ground truth reference
    def __init__(self, ir_folder: pathlib.Path, vi_folder: pathlib.Path, ir_reverse_folder: pathlib.Path, vi_reverse_folder: pathlib.Path, ir_map: pathlib.Path, vi_map: pathlib.Path):
    # def __init__(self, ir_folder: pathlib.Path, vi_folder: pathlib.Path, ir_map: pathlib.Path, vi_map: pathlib.Path):
        super(TrainingData, self).__init__()
        # self.crop = crop
        # gain ir and vis images list
        self.ps = 120

        self.ir_list = [x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.vi_list = [x for x in sorted(vi_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]

        self.ir_reverse_list = [x for x in sorted(ir_reverse_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.vi_reverse_list = [x for x in sorted(vi_reverse_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]

        self.ir_map_list = [x for x in sorted(ir_map.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.vi_map_list = [x for x in sorted(vi_map.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]

        # self.ir_reverse_map_list = [x for x in sorted(ir_reverse_map.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        # self.vi_reverse_map_list = [x for x in sorted(vi_reverse_map.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]

    def get_patch(self, ir, vis, ir_reverse, vis_reverse, ir_map, vis_map):
        H, W = ir.shape[1], ir.shape[2]

        #1.
        x, y = np.random.randint(10, H-10-self.ps+1), np.random.randint(10, W-10-self.ps+1)
        ir_crop = ir[:, x:x+self.ps, y:y+self.ps]
        vis_crop = vis[:, x:x+self.ps, y:y+self.ps]
        ir_reverse_crop = ir_reverse[:, x:x+self.ps, y:y+self.ps]
        vis_reverse_crop = vis_reverse[:, x:x+self.ps, y:y+self.ps]
        ir_map_crop = ir_map[:, x:x+self.ps, y:y+self.ps]
        vis_map_crop = vis_map[:, x:x+self.ps, y:y+self.ps]


        return ir_crop, vis_crop, ir_reverse_crop, vis_reverse_crop, ir_map_crop, vis_map_crop

    def __getitem__(self, index):
        # gain image path
        ir_path = self.ir_list[index]
        vi_path = self.vi_list[index]

        ir_reverse_path = self.ir_reverse_list[index]
        vi_reverse_path = self.vi_reverse_list[index]

        ir_map_path = self.ir_map_list[index]
        vi_map_path = self.vi_map_list[index]

        # ir_reverse_map_path = self.ir_reverse_map_list[index]
        # vi_reverse_map_path = self.vi_reverse_map_list[index]

        assert ir_path.name == vi_path.name, f"Mismatch ir:{ir_path.name} vi:{vi_path.name}."

        # read image as type Tensor
        ir = self.imread(path=ir_path, flags=cv2.IMREAD_GRAYSCALE)
        vi = self.imread(path=vi_path, flags=cv2.IMREAD_GRAYSCALE)

        ir_reverse = self.imread(path=ir_reverse_path, flags=cv2.IMREAD_GRAYSCALE)
        vi_reverse = self.imread(path=vi_reverse_path, flags=cv2.IMREAD_GRAYSCALE)

        ir_map = self.imread(path=ir_map_path, flags=cv2.IMREAD_GRAYSCALE)
        vi_map = self.imread(path=vi_map_path, flags=cv2.IMREAD_GRAYSCALE)

        # ir_reverse_map = self.imread(path=ir_reverse_map_path, flags=cv2.IMREAD_GRAYSCALE)
        # vi_reverse_map = self.imread(path=vi_reverse_map_path, flags=cv2.IMREAD_GRAYSCALE)

        ir_crop, vis_crop, ir_reverse_crop, vis_reverse_crop, ir_map_crop, vis_map_crop = self.get_patch(ir, vi, ir_reverse, vi_reverse, ir_map, vi_map)

        # return ir_crop, vis_crop, ir_reverse_crop, vis_reverse_crop, ir_map_crop, vis_map_crop


        return (ir_crop, vis_crop), (str(ir_path), str(vi_path)), (ir_reverse_crop, vis_reverse_crop), (ir_map_crop, vis_map_crop)
        # return (ir, vi), (str(ir_path), str(vi_path)), (ir_reverse, vi_reverse), (ir_map, vi_map), (ir_reverse_map, vi_reverse_map)   11111111111111111

    def __len__(self):
        return len(self.ir_list)

    @staticmethod
    def imread(path: pathlib.Path, flags=cv2.IMREAD_GRAYSCALE):
        im_cv = cv2.imread(str(path), flags)
        assert im_cv is not None, f"Image {str(path)} is invalid."
        im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
        return im_ts

class TestData(torch.utils.data.Dataset):
    """
    Load dataset with ir folder path and vis folder path
    """

    # TODO: remove ground truth reference
    def __init__(self, ir_folder: pathlib.Path, vis_folder: pathlib.Path, ir_reverse_folder: pathlib.Path, vis_reverse_folder: pathlib.Path):
        super(TestData, self).__init__()
        # gain ir and vis images list

        self.ir_list = [x for x in sorted(ir_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.vis_list = [x for x in sorted(vis_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]

        self.ir_reverse_list = [x for x in sorted(ir_reverse_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]
        self.vis_reverse_list = [x for x in sorted(vis_reverse_folder.glob('*')) if x.suffix in ['.png', '.jpg', '.bmp']]

    def __getitem__(self, index):
        # gain image path
        ir_path = self.ir_list[index]
        vis_path = self.vis_list[index]

        ir_reverse_path = self.ir_reverse_list[index]
        vis_reverse_path = self.vis_reverse_list[index]

        assert ir_path.name == vis_path.name, f"Mismatch ir:{ir_path.name} vis:{vis_path.name}."

        # read image as type Tensor
        ir = self.imread(path=ir_path, flags=cv2.IMREAD_GRAYSCALE)
        vis = self.imread(path=vis_path, flags=cv2.IMREAD_GRAYSCALE)

        ir_reverse = self.imread(path=ir_reverse_path, flags=cv2.IMREAD_GRAYSCALE)
        vis_reverse = self.imread(path=vis_reverse_path, flags=cv2.IMREAD_GRAYSCALE)


        return (ir, vis), (str(ir_path), str(vis_path)), (ir_reverse, vis_reverse), (str(ir_reverse_path), str(vis_reverse_path))

    def __len__(self):
        return len(self.ir_list)

    @staticmethod
    def imread(path: pathlib.Path, flags=cv2.IMREAD_GRAYSCALE):
        im_cv = cv2.imread(str(path), flags)
        assert im_cv is not None, f"Image {str(path)} is invalid."
        im_ts = kornia.utils.image_to_tensor(im_cv / 255.).type(torch.FloatTensor)
        return im_ts