import copy
from itertools import product
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
        
    return np.vstack((np.array(points).T,np.random.rand(samples)))

def square_board(samples: int=2000) -> List[Union[float, float]]:
    """generate ICR in square board

    Args:
        samples (int, optional): Number to make lattice velocity samples. Defaults to 2000.

    Returns:
        List[Union[float, float]]: Point list with [x, y]. x=[-1,1], y=[0,1]
    """
    _z_num = 4
    # _root = int(np.sqrt(samples / _z_num))
    # while True:
    #     if (samples / _z_num)%_root==0:
    #         break
    #     else:
    #         _root += 1
    _root = 100

    _x = np.linspace(-1, 1, _root)
    _x = np.sign(_x) * np.power(_x, 2)
    _y = np.linspace(0, 1, int(samples / _z_num /_root))
    _z = np.linspace(0, 1, _z_num)

    _model_input = np.array(list(product(_x, _y, _z)))

    return _model_input.T, np.array([_root, samples / _z_num /_root, _z_num]).astype(np.uint)

def model_input(samples: int=2000, model_config: dict = {'MAX_R': 0.5, 'MIN_R': -1.5, 'MAX_A': 90, 'MIN_A': 0, 'MAX_L': 0.08, 'MIN_L': 0.04}, mode: List=[None, None, None]) -> Union[Union[float, float, float], Union[float, float, float]]:
    """Created model input and real value list

    Args:
        samples (int, optional): Number of model inputs(points). Defaults to 2000.
        model_config (dict, optional): Radius, angle, lengh data.
        mode (List, optional): Fix input value. Defaults to [None, None, None]. Each list means icr[m], gripper angle[deg], and gripper width[m] in that order. Set to none if you do not want to change it.

    Returns:
        Union[Union[float, float, float], Union[float, float, float]]: Returns model input and real value. The model value is a value directly inserted into the learned model, and the real value indicates what each value actually means.
    """
    
    # Parameters
    MAX_R = model_config["MAX_R"]
    MIN_R = model_config["MIN_R"]
    MAX_A = np.deg2rad(model_config["MAX_A"])
    MIN_A = np.deg2rad(model_config["MIN_A"])
    MAX_L = model_config["MAX_L"]
    MIN_L = model_config["MIN_L"]
    print("MAX_R: ", MAX_R)
    print("MIN_R: ", MIN_R)
    print("MAX_A: ", MAX_A)
    print("MIN_A: ", MIN_A)
    print("MAX_L: ", MAX_L)
    print("MIN_L: ", MIN_L)

    _model_input = (fibonacci_lattice(samples=samples))
    _model_input[2,:] = MAX_L - (MAX_L - MIN_L) * _model_input[2,:]
                    
    _real_value = copy.deepcopy(_model_input)
    # ICR biased
    # _real_value[0,:] = (np.sign(_model_input[0,:]) * np.power(10, MIN_R + np.abs(_model_input[0,:]) * (MAX_R - MIN_R)))
    # ICR uniform distribution
    _real_value[0,:] = np.sign(_model_input[0,:]) * (np.power(10, MIN_R) + np.abs(_model_input[0,:]) * (np.power(10, MAX_R) - np.power(10, MIN_R)))
    _model_input[0,:] = np.sign(_real_value[0,:]) * (np.log10(np.abs(_real_value[0,:])) - MIN_R) / (MAX_R - MIN_R)
    _real_value[1,:] = np.rad2deg( MIN_A + _model_input[1,:] * (MAX_A - MIN_A))

    for i, _num in enumerate(mode):
        if _num is None:
            continue
        else:
            if i == 0:
                _real_value[0,:] = _num
                _model_input[0,:] = np.sign(_real_value[0,:]) * (np.log10(np.abs(_real_value[0,:])) - MIN_R) / (MAX_R - MIN_R)
                pass

            elif i == 1:
                _model_input[1,:] = (_num - MIN_A) / (MAX_A - MIN_A)
                _real_value[1,:] = np.rad2deg(_num)
                pass

            elif i == 2:
                _model_input[2,:] = _num
                _real_value[2,:] = _num
                pass
            
    return _model_input.T, _real_value.T

def checker_input(samples: int=2000, model_config: dict = {'MAX_R': 0.5, 'MIN_R': -1.5, 'MAX_A': 90, 'MIN_A': 0, 'MAX_L': 0.08, 'MIN_L': 0.04}, mode: List=[None, None, None]) -> Union[Union[float, float, float], Union[float, float, float]]:
    """Created model input and real value list

    Args:
        samples (int, optional): Number of model inputs(points). Defaults to 2000.
        mode (List, optional): Fix input value. Defaults to [None, None, None]. Each list means icr[m], gripper angle[deg], and gripper width[m] in that order. Set to none if you do not want to change it.

    Returns:
        Union[Union[float, float, float], Union[float, float, float]]: Returns model input and real value. The model value is a value directly inserted into the learned model, and the real value indicates what each value actually means.
    """
    
    # Parameters
    MAX_R = model_config["MAX_R"]
    MIN_R = model_config["MIN_R"]
    MAX_A = np.deg2rad(model_config["MAX_A"])
    MIN_A = np.deg2rad(model_config["MIN_A"])
    MAX_L = model_config["MAX_L"]
    MIN_L = model_config["MIN_L"]
    print("MAX_R: ", MAX_R)
    print("MIN_R: ", MIN_R)
    print("MAX_A: ", MAX_A)
    print("MIN_A: ", MIN_A)
    print("MAX_L: ", MAX_L)
    print("MIN_L: ", MIN_L)

    _model_input, _shape = square_board(samples=samples)
    _model_input[2,:] = MAX_L - (MAX_L - MIN_L) * _model_input[2,:]

    _real_value = copy.deepcopy(_model_input)
    # ICR uniform distribution
    _real_value[0,:] = np.sign(_model_input[0,:]) * (np.power(10, MIN_R) + np.abs(_model_input[0,:]) * (np.power(10, MAX_R) - np.power(10, MIN_R)))
    _model_input[0,:] = np.sign(_real_value[0,:]) * (np.log10(np.abs(_real_value[0,:])) - MIN_R) / (MAX_R - MIN_R)
    _real_value[1,:] = np.rad2deg( MIN_A + _model_input[1,:] * (MAX_A - MIN_A))

    for i, _num in enumerate(mode):
        if _num is None:
            continue
        else:
            if i == 0:
                _real_value[0,:] = _num
                _model_input[0,:] = np.sign(_real_value[0,:]) * (np.log10(np.abs(_real_value[0,:])) - MIN_R) / (MAX_R - MIN_R)
                pass

            elif i == 1:
                _model_input[1,:] = (_num - MIN_A) / (MAX_A - MIN_A)
                _real_value[1,:] = np.rad2deg(_num)
                pass

            elif i == 2:
                _model_input[2,:] = _num
                _real_value[2,:] = _num
                pass
    print(_model_input)
    print(_real_value)
    return _model_input.T, _real_value.T, _shape