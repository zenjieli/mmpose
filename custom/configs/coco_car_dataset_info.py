dataset_info = dict(
    dataset_name='coco_car',
    paper_info=dict(),
    keypoint_info={
        0: dict(name='left',   id=0, color=[0, 0, 255],   type='', swap='right'),
        1: dict(name='bottom', id=1, color=[0, 200, 0],   type='', swap=''),
        2: dict(name='right',  id=2, color=[255, 80, 0],  type='', swap='left'),
        3: dict(name='top',    id=3, color=[0, 220, 220], type='', swap=''),
    },
    skeleton_info={
        0: dict(link=('left', 'bottom'), id=0, color=[96, 96, 255]),
        1: dict(link=('bottom', 'right'), id=1, color=[96, 96, 255]),
        2: dict(link=('right', 'top'), id=2, color=[96, 96, 255]),
        3: dict(link=('top', 'left'), id=3, color=[96, 96, 255]),
    },
    joint_weights=[1., 1., 1., 1.],
    sigmas=[0.1, 0.1, 0.1, 0.1],
)
