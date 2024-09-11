import os
import numpy as np
import torch
import yaml
import multiprocessing
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from utils.model import PushNet
from utils.model_test import PushNetTest
from utils.dataloader import PushNetDataset
from utils.utils import *
torch.multiprocessing.set_sharing_strategy('file_system')
# import matplotlib
# matplotlib.use('Qt5Agg')

import matplotlib.pyplot as plt

'''
Plot the network output (velocity sphere in which each velocity is colored by the estimated push success rate)
and the corresponding image.
'''

def model_loop(dataloader, model, input):
    predictions, images = [], []
    with torch.no_grad():
        input = torch.from_numpy(input.astype(np.float32)).to(DEVICE)
        
        for image, _, _  in tqdm(dataloader):
            
            image_device = image.to(DEVICE)
            images_tensor = torch.tile(image_device,(len(input),1,1,1))
            
            outputs = model(images_tensor,input).cpu()
            result = torch.nn.Softmax(dim=1)(outputs)[:,1] # Estimated push success rate.
            
            predictions.append(result.detach().numpy())
            images.append(image[0].permute(1,2,0).numpy())
            
    return np.array(predictions), np.array(images)

def plot_model(dataloader, model, model_input, num_samples):
    _threshold = 0.8
    # figure configuration
    plot_fig = plt.figure()
    image_fig = plt.figure()
    num_samples_per_edge = int(np.ceil(np.sqrt(num_samples)))
    stability, images = model_loop(dataloader,model,model_input)
    _stability = stability.reshape(_shape)
    
    _stability = np.where(_stability>_threshold, 1, 0).T
    print(np.sum(_stability, axis=2))
    _arg = np.argmax(np.sum(_stability, axis=2))
    # _arg = np.argmin(np.sum(_stability, axis=2))
    _arg = int((_arg % _shape[1]) * _shape[2] + (_arg // _shape[1]))
    print("arg:", _arg)
    print("max arg:", np.max(np.sum(_stability, axis=2)))
    max_input = real_inputs[np.arange(_arg, _shape[0]*_shape[1]*_shape[2], _shape[1]*_shape[2]).astype(int),:]
    # print(max_input.shape)
    # print(max_input)
    max_results = np.squeeze(stability)[np.arange(_arg, _shape[0]*_shape[1]*_shape[2], _shape[1]*_shape[2]).astype(int)]
    # print(max_results.shape)
    # print(max_results)
    # print(np.where(max_results>0.8,1,0))
    icr_list = max_input[np.where(max_results>_threshold)]
    # print(icr_list)
    # print(np.where(icr_list[:,0] > 0, 1, 0))
    right_min_radius = np.argmin(np.where(icr_list[:,0] > 0, 1, -10000) * icr_list[:,0])
    left_min_radius = np.argmax(np.where(icr_list[:,0] < 0, 1, -10000) * icr_list[:,0])
    print(right_min_radius, left_min_radius)
    print(icr_list[right_min_radius], icr_list[left_min_radius])

    stability = stability.reshape(num_samples,-1)
    # exit()
    for idx in range(num_samples):
        # Plot model output
        ax = plot_fig.add_subplot(num_samples_per_edge,num_samples_per_edge,idx+1,projection='3d')
        ax.set_title(f"Result {idx}")
        ax.view_init(elev = 0,azim = 0)
        ax.set_xlabel(r"$IRC$ [m]")
        ax.set_ylabel(r"$Angl$ [degree]", rotation=0)
        ax.set_zlabel(r"$Gripper Width$ [m]")
        ax.set_box_aspect((1,2,2))
        ax.grid(False)
        p = ax.scatter(real_inputs[:,0], real_inputs[:,1], real_inputs[:,2], stability[idx], c=stability[idx], cmap="jet", s=100, vmin=0, vmax=1)
        plot_fig.colorbar(p, ax=ax)
        
        # Plot image
        ax = image_fig.add_subplot(num_samples_per_edge,num_samples_per_edge,idx+1)
        ax.set_title(f"Image {idx}")
        ax.imshow(images[idx])
        ax.axis('off')

    plt.show()
    
if __name__ == '__main__':
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get current file path
    current_file_path = os.path.dirname(os.path.realpath(__file__))
    config_file = os.path.abspath(os.path.join(current_file_path, '..', 'config', 'config.yaml'))

    # Load configuation file
    with open(config_file,'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Get model config file path
    model_config_dir = os.path.abspath(os.path.join(os.path.expanduser('~'), config["model_dir"], config['network']["model_name"], 'pusher.yaml'))
    config_file = os.path.abspath(os.path.join(model_config_dir))
    
    # Load configuation file
    with open(config_file,'r') as f:
        model_config = yaml.load(f, Loader=yaml.FullLoader)

    model_dir = os.path.abspath(os.path.join(os.path.expanduser('~'), config["model_dir"], config['network']["model_name"], 'model.pt'))
    DATA_DIR = config["data_dir"]
    # num_push_cases = config['network_output']['num_pushes']
    num_push_cases = 1
    # data_stats_dir = os.path.expanduser('~') + '/' + DATA_DIR + '/data_stats'
    data_stats_dir = os.path.expanduser('~') + '/' + config["model_dir"] + "/" + config['network']["model_name"] + '/data_stats'
    
    # VEL_NUM=config['network_output']['num_data_points']
    VEL_NUM=2000

    network_inputs, real_inputs, _shape = checker_input(samples=VEL_NUM, model_config=model_config, mode=[None,None,None])

    velocity_mean = np.load(data_stats_dir + "/velocity_mean.npy")
    velocity_std = np.load(data_stats_dir + "/velocity_std.npy")
    print("velocity_mean", velocity_mean)
    print("velocity_std", velocity_std)
    input_normalized = (network_inputs - velocity_mean) / velocity_std
    
    # model = PushNet()
    model = PushNetTest()
    model.to(DEVICE)
    model.load_state_dict(torch.load(model_dir))
    
    # Getting features and confusion index
    print("Getting features and confusion index")
    test_dataset = PushNetDataset(dataset_dir=DATA_DIR, type='test', image_type=config['planner']['image_type'], num_debug_samples = num_push_cases)
    num_workers = multiprocessing.cpu_count()
    dataloader = DataLoader(test_dataset, shuffle=True, num_workers=num_workers)
    plot_model(dataloader, model, input_normalized, num_push_cases)

