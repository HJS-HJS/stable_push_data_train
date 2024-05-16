import copy
import numpy as np
from typing import List, Union
from scipy.spatial.transform import Rotation as R

def tmat(pose):
    ''' Pose datatype conversion
    
    gymapi.Transform -> Homogeneous transformation matrix (4 x 4)
    
    '''
    t = np.eye(4)
    t[0, 3], t[1, 3], t[2, 3] = pose.p.x, pose.p.y, pose.p.z
    quat = np.array([pose.r.x, pose.r.y, pose.r.z, pose.r.w])
    t[:3,:3] = R.from_quat(quat).as_matrix()
    return t

def fibonacci_lattice(samples: int=2000) -> List[Union[float, float]]:
    """generate ICR in fibonacci lattice

    Args:
        samples (int, optional): Number to make lattice velocity samples. Defaults to 2000.

    Returns:
        List[Union[float, float]]: Point list with [x, y]. x=[-1,1], y=[0,1]
    """
    points = []
    phi = (1. + np.sqrt(5.)) / 2.  # golden angle in radians

    for i in range(samples):
        x = 1. - 2. * ((i / phi) % 1.)
        y = i / (samples - 1)
        points.append((x, y))
    return np.array(points)

def model_input(samples: int=2000, mode: List=[None, None, None]) -> Union[Union[float, float, float], Union[float, float, float]]:
    """Created model input and real value list

    Args:
        samples (int, optional): Number of model inputs(points). Defaults to 2000.
        mode (List, optional): Fix input value. Defaults to [None, None, None]. Each list means icr[m], gripper angle[deg], and gripper width[m] in that order. Set to none if you do not want to change it.

    Returns:
        Union[Union[float, float, float], Union[float, float, float]]: Returns model input and real value. The model value is a value directly inserted into the learned model, and the real value indicates what each value actually means.
    """
    
    # Parameters
    MAX_R=0.5
    MIN_R=-1.5
    MAX_A=np.pi/2
    MIN_A=0
    MAX_L=0.08
    MIN_L=0.04

    _model_input = np.vstack((fibonacci_lattice(samples=samples).T,MIN_L + (MAX_L - MIN_L) * np.random.rand(samples)))
    _real_value = copy.deepcopy(_model_input)
    # ICR biased
    # _real_value[0,:] = (np.sign(_model_input[0,:]) * np.power(10, MIN_R + np.abs(_model_input[0,:]) * (MAX_R - MIN_R)))
    # ICR uniform distribution
    _real_value[0,:] = np.sign(_model_input[0,:]) * (np.power(10, MIN_R) + np.abs(_model_input[0,:]) * (np.power(10, MAX_R) - np.power(10, MIN_R)))
    _model_input[0,:] = np.sign(_real_value[0,:]) * (np.log10(np.abs(_real_value[0,:])) - MIN_R) / (MAX_R - MIN_R)
    _real_value[1,:] = 90 - np.rad2deg( MIN_A + _model_input[1,:] * (MAX_A - MIN_A))

    for i, _num in enumerate(mode):
        if _num is None:
            continue
        else:
            if i == 0:
                _real_value[0,:] = _num
                _model_input[0,:] = np.sign(_real_value[0,:]) * (np.log10(np.abs(_real_value[0,:])) - MIN_R) / (MAX_R - MIN_R)
                pass

            elif i == 1:
                _model_input[1,:] = (np.pi/2 - _num - MIN_A) / (MAX_A - MIN_A)
                _real_value[1,:] = np.rad2deg(_num)
                pass

            elif i == 2:
                _model_input[2,:] = _num
                _real_value[2,:] = _num
                pass
            
    return _model_input.T, _real_value.T

