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
 
def fibonacci_sphere(samples: int=2000) -> List[Union[float, float, float]]:
    """generate ICR in fibonacci sphere

    Args:
        samples (int, optional): Number to make circle velocity samples. Defaults to 2000.

    Returns:
        List[Union[float, float, float]]: Velocities list with [x, y, z]
    """
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(samples//2):
        x = 1 - (i / float(samples - 1)) * 2  # x goes from 1 to -1
        radius = np.sqrt(1 - x * x)  # radius at x

        theta = phi * i  # golden angle increment

        y = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append((x, y, z))
    return np.array(points)

def linear_velocities(samples: int=2000) -> List[Union[float, float, float]]:
    """_summary_

    Args:
        samples (int, optional): _description_. Defaults to 2000.

    Returns:
        List[Union[float, float, float]]: _description_
    """
    # log_radius = np.linspace(np.log10(1e-4), np.log10(100), samples)
    # log_radius = np.linspace(np.log10(1e-2), np.log10(10), samples)
    # log_radius = np.linspace(np.log10(1e-1), np.log10(10), samples)
    # log_radius = np.linspace(-1.5, np.log10(10), samples)
    # log_radius = np.linspace(-1.2, 0.5, samples)
    log_radius = np.linspace(-1.5, 0.5, samples)
    # log_radius = np.linspace(np.log10(10), np.log10(100), samples)
    radius_positive = np.power(10, log_radius)
    radius = np.concatenate((np.flip(-radius_positive), radius_positive))
    icrs = np.vstack((radius, np.zeros_like(radius), np.ones_like(radius))).T
    return icrs

def velocities2icrs(velocity: List[Union[float, float, float]]) -> List[Union[float, float]]:
    """Calculate ICR (Instantaneous Center of Rotation) for each velocity.

    Args:
        velocity (List[Union[float, float, float]]): Velocities list with [x, y, z]

    Returns:
        List[Union[float, float]]: List of ICR [x/z, y/z]
    """
    vx, vy, w = velocity[:,0], velocity[:,1], velocity[:,2]
    ICRs = []
    for i in range(len(vx)):
        if w[i] == 0:
            w[i] = 1e-6
        icr= np.array([-vy[i] / w[i], vx[i] / w[i]])
        ICRs.append(icr)
        
    return np.array(ICRs)
