from mmdet.apis import init_detector, inference_detector

config_file = 'rtmdet_tiny_8xb32-300e_coco.py'
checkpoint_file = 'rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:1')  # or device='cuda:0'
a = inference_detector(model, '/home/huangdehao/github_projects/pybullet_franka/inbox_codes/image.png')
print(a)