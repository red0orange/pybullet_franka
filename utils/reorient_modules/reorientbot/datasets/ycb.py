import os
import sys
import gdown
import numpy as np
import path

from .. import geometry

home = path.Path("~").expanduser()

# class_names = [
#     "__background__",
#     "002_master_chef_can",
#     "003_cracker_box",
#     "004_sugar_box",
#     "005_tomato_soup_can",
#     "006_mustard_bottle",
#     "007_tuna_fish_can",
#     "008_pudding_box",
#     "009_gelatin_box",
#     "010_potted_meat_can",
#     "011_banana",
#     "019_pitcher_base",
#     "021_bleach_cleanser",
#     "024_bowl",
#     "025_mug",
#     "035_power_drill",
#     "036_wood_block",
#     "037_scissors",
#     "040_large_marker",
#     "051_large_clamp",
#     "052_extra_large_clamp",
#     "061_foam_brick",
# ]

class_names = [
    "__background__",
    "025_mug",
    "004_sugar_box",
    "005_tomato_soup_can",
    "006_mustard_bottle",
    "007_tuna_fish_can",
    "008_pudding_box",
    "021_bleach_cleanser",
    "YcbCrackerBox",
    "YcbGelatinBox",
    "YcbMasterChefCan",
    "YcbPottedMeatCan"
]
class_names = np.array(class_names)
class_names.setflags(write=0)

# https://www.ri.cmu.edu/pub_files/2015/8/ycb_journal_v26.pdf
masses = [
    0,
    0.414,
    0.411,
    0.514,
    0.349,
    0.603,
    0.171,
    0.187,
    0.097,
    0.370,
    0.066,
    0.178 + 0.066,
    1.131,
    0.147,
    0.118,
    0.895,
    0.729,
    0.082,
    0.0158,
    0.125,
    0.202,
    0.028,
]
masses = np.array(masses)
masses.setflags(write=0)


def init():
    root_dir = path.Path("/home/huangdehao/github_projects/multi_view_rearr/pybullet_pipeline/data/my_models")

    if not root_dir.exists():
        gdown.cached_download(
            url="https://drive.google.com/uc?id=1BoXR3rNqWIoILDQK8yiB6FWgvHGpjtJe",  # NOQA
            path=root_dir + ".zip",
            md5="054b845708318a9d38a3f080572dcb3c",
            postprocess=gdown.extractall,
        )

    class_names = []
    for model_dir in sorted(root_dir.listdir()):
        class_name = str(model_dir.basename())
        class_names.append(class_name)

    return root_dir


def get_visual_file(class_id):
    assert class_id > 0
    root_dir = init()
    class_name = class_names[class_id]
    if os.path.exists(root_dir / class_name / "textured_simple.obj"):
        return root_dir / class_name / "textured_simple.obj"
    elif os.path.exists(root_dir / class_name / "textured_simple_reoriented.obj"):
        return root_dir / class_name / "textured_simple_reoriented.obj"
    else:
        raise BaseException("")


def get_canonical_quaternion(class_id):
    c = geometry.Coordinate()
    if class_names[class_id] == "006_mustard_bottle":
        c.rotate([0, 0, 0])
    elif class_names[class_id] == "005_tomato_soup_can":
        c.rotate([0, 0, 0])
    elif class_names[class_id] == "025_mug":
        c.rotate([0, 0, 0])
    elif class_names[class_id] == "YcbCrackerBox":
        c.rotate([0, 0, 0])
    elif class_names[class_id] == "YcbGelatinBox":
        c.rotate([0, 0, 0])
    else:
        pass
    return c.quaternion


def get_pcd_file(class_id):
    assert class_id > 0
    root_dir = init()
    class_name = class_names[class_id]
    return root_dir / class_name / "points.xyz"

