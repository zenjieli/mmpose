dataset_info = dict(
    dataset_name='pascal3d_car',
    paper_info=dict(),
    keypoint_info={
        0: dict(name='RF', id=0, color=[100, 255, 100], type='', swap='LF'),
        1: dict(name='LF', id=1, color=[255, 100, 100], type='', swap='RF'),
        2: dict(name='LB', id=2, color=[100, 100, 255], type='', swap='RB'),
        3: dict(name='RB', id=3, color=[255, 255, 100], type='', swap='LB'),
    },
    skeleton_info={
        0: dict(link=('RF', 'LF'), id=0, color=[96, 255, 96]),
        1: dict(link=('LF', 'LB'), id=1, color=[255, 96, 96]),
        2: dict(link=('LB', 'RB'), id=2, color=[96, 96, 255]),
        3: dict(link=('RB', 'RF'), id=3, color=[255, 255, 96]),
    },
    joint_weights=[1., 1., 1., 1.],
    sigmas=[0.1, 0.1, 0.1, 0.1],
)
