import cv2
import glob
import os
import numpy as np
from PIL import Image

def cal_for_frames_0(video_path):
    frames = glob.glob(os.path.join(video_path, '*.jpg'))
    frames.sort()

    flow = []
    prev = cv2.imread(frames[0],1)
    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    for i, frame_curr in enumerate(frames):
        curr = cv2.imread(frame_curr)
        curr = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        tmp_flow = compute_TVL1(prev, curr)
        flow.append(tmp_flow)
        prev = curr

    return flow

def compute_TVL1(prev, curr, bound=15):
    """Compute the TV-L1 optical flow."""
#     print('prev',prev.shape)
#     print('curr',curr.shape)
    TVL1 = cv2.optflow.DualTVL1OpticalFlow_create()
    #HS = cv2.optflow.DenseRLOFOpticalFlow_create
#     TVL1 = cv2.DualTVL1OpticalFlow_create()
#     TVL1=cv2.createOptFlow_DualTVL1()
    flow = TVL1.calc(prev, curr, None)
    assert flow.dtype == np.float32

    flow = (flow + bound) * (255.0 / (2 * bound))
    flow = np.round(flow).astype(int)
    flow[flow >= 255] = 255
    flow[flow <= 0] = 0

    return flow
