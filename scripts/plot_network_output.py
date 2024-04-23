import os
import numpy as np
import torch
import yaml
import multiprocessing
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm
from utils.model import PushNet
from utils.dataloader import PushNetDataset
from utils.utils import fibonacci_sphere, linear_velocities
torch.multiprocessing.set_sharing_strategy('file_system')
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# from matplotlib import pyplot as plt


'''
Plot the network output (velocity sphere in which each velocity is colored by the estimated push success rate)
and the corresponding image.


'''
def model_loop(dataloader,model,velocities):
    predictions, images = [], []
    velocities = torch.from_numpy(velocities.astype(np.float32)).to(DEVICE)
    with torch.no_grad():
        
        for image, velocity, label_onehot  in tqdm(dataloader):
            
            image_device = image.to(DEVICE)
            images_tensor = torch.tile(image_device,(len(velocities),1,1,1))
            
            outputs = model(images_tensor,velocities).cpu()
            result = torch.nn.Softmax(dim=1)(outputs)[:,1] # Estimated push success rate.
            
            predictions.append(result.detach().numpy())
            images.append(image[0].permute(1,2,0).numpy())
            
    return np.array(predictions), np.array(images)


def plot_model(dataloader,model,velocities,num_samples):
    # figure configuration
    plot_fig = plt.figure()
    image_fig = plt.figure()
    num_samples_per_edge = int(np.ceil(np.sqrt(num_samples)))
    stability, images = model_loop(dataloader,model,velocities)
    stability = stability.reshape(num_samples,-1)
    for idx in range(num_samples):
        
        # Plot model output
        ax = plot_fig.add_subplot(num_samples_per_edge,num_samples_per_edge,idx+1,projection='3d')
        ax.set_title(f"Result {idx}")
        ax.view_init(elev = 0,azim = 0)
        ax.set_xlabel(r"$IRC$ [m]")
        ax.set_ylabel(r"$Height$ [m]")
        ax.set_zlabel(r"$Angl$ [rad]", rotation=0)
        ax.set_box_aspect((1,2,2))
        ax.grid(False)
        p = ax.scatter(_velocity[:,0], _velocity[:,1], _velocity[:,2], stability[idx], c=stability[idx], cmap="jet", s=100, vmin=0, vmax=1)
        plot_fig.colorbar(p, ax=ax)
        
        # Plot image
        ax = image_fig.add_subplot(num_samples_per_edge,num_samples_per_edge,idx+1)
        ax.set_title(f"Image {idx}")
        ax.imshow(images[idx])
        ax.axis('off')

    plt.show()
    
def load_sampler(dataset):
    
    label_list = dataset.label_list
    indices = dataset.indices
    num_true=0
    for index in indices:
        if label_list[index] == 1:
            num_true += 1
    num_false = len(indices) - num_true
    
    portion_true = num_true/len(indices)
    portion_false = num_false/len(indices)
    weights = [1/portion_true if label_list[index] == 1 else 1/portion_false for index in indices]
    
    sampler = WeightedRandomSampler(weights, len(weights))
    
    return sampler
    
if __name__ == '__main__':
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Get current file path
    current_file_path = os.path.dirname(os.path.realpath(__file__))
    config_file = os.path.abspath(os.path.join(current_file_path, '..', 'config', 'config.yaml'))

    # Load configuation file
    with open(config_file,'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_dir = os.path.abspath(os.path.join(os.path.expanduser('~'), config["model_dir"], config['network']["model_name"], 'model.pt'))
    DATA_DIR = config["data_dir"]
    num_push_cases = config['network_output']['num_pushes']
    data_stats_dir = os.path.expanduser('~') + '/' + DATA_DIR + '/data_stats'
    
    MAX_H=0.02
    MIN_H=0.02
    MAX_A=np.pi/2
    MIN_A=0
    VEL_NUM=2000
    _velocity=np.vstack((linear_velocities(samples=int(VEL_NUM/2))[:,0], MIN_H + (MAX_H - MIN_H) * np.random.rand(VEL_NUM), MIN_A + (MAX_A - MIN_A) * np.random.rand(VEL_NUM))).T

    velocity_mean = np.load(data_stats_dir + "/velocity_mean.npy")
    velocity_std = np.load(data_stats_dir + "/velocity_std.npy")
    velocity_normalized = (_velocity - velocity_mean) / velocity_std
    
    model = PushNet()
    model.to(DEVICE)
    model.load_state_dict(torch.load(model_dir))
    
    # Getting features and confusion index
    print("Getting features and confusion index")
    test_dataset = PushNetDataset(dataset_dir=DATA_DIR, type='test', image_type=config['planner']['image_type'], num_debug_samples = num_push_cases)
    test_sampler = load_sampler(test_dataset)
    num_workers = multiprocessing.cpu_count()
    dataloader = DataLoader(test_dataset, shuffle=True, num_workers=num_workers)
    plot_model(dataloader,model,velocity_normalized, num_push_cases)
