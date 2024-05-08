import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from utils.model import PushNet
# from utils.dataloader_confusion import PushNetDataset
from utils.dataloader import PushNetDataset
from utils.utils import fibonacci_sphere, linear_velocities
import yaml
torch.multiprocessing.set_sharing_strategy('file_system')
import matplotlib.pyplot as plt
import os 

# Feature extraction loop
def sort_confusion(dataloader, model):
    TP, FP, FN, TN = [], [], [], []

    with torch.no_grad():
        index = 0
        for image, velocity, label_onehot in dataloader:
            image = image.to(DEVICE)
            velocity = velocity.to(DEVICE)
            label_onehot = label_onehot.to(DEVICE)

            
            # for index in range(velocity.size(1)):
            # output = model(image,velocity).cpu()
            output = model(image,velocity).to(DEVICE)
            # confusion
            pred = torch.argmax(output, dim=1) # selects the "class" that the model chose
            label = torch.argmax(label_onehot, dim=1)
            
            # index - class
            if   pred == 1 and label == 1: # TP
                TP.append(index)
            elif pred == 1 and label == 0: # FP
                FP.append(index)
            elif pred == 0 and label == 1: # FN
                FN.append(index)
            elif pred == 0 and label == 0: # TN
                TN.append(index)
            index += 1
              
    return TP, FP, FN, TN, image

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
    
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        
    image_type = config['planner']['image_type']
    MODEL_NAME = config['network']["model_name"]
    VEL_NUM = config['confusion']['num_data_points']
    num_contact_samples= config['confusion']['num_pushes']
    DATA_DIR = config["data_dir"]
    num_push_cases = config['network_output']['num_pushes']
    data_stats_dir = os.path.expanduser('~') + '/' + DATA_DIR + '/data_stats'
    MAX_L=0.08
    MIN_L=0.04
    MAX_A=np.pi/2
    MIN_A=0

    model_dir = os.path.abspath(os.path.join(os.path.expanduser('~'), config["model_dir"], config['network']["model_name"], 'model.pt'))

    model = PushNet()
    model.to(DEVICE)
    model.load_state_dict(torch.load(model_dir)) #, map_location=torch.device('cpu')
    
    # Getting features and confusion index
    print("Getting features and confusion index")
    test_dataset = PushNetDataset(dataset_dir=DATA_DIR, type='test', image_type=config['planner']['image_type'], num_debug_samples = VEL_NUM)
    test_sampler = load_sampler(test_dataset)
    dataloader = DataLoader(test_dataset, 1, test_sampler, num_workers=16)
    
    velocity=np.vstack((linear_velocities(samples=int(VEL_NUM/2))[:,0], MIN_L + (MAX_L - MIN_L) * np.random.rand(VEL_NUM), MIN_A + (MAX_A - MIN_A) * np.random.rand(VEL_NUM))).T
    # velocity = fibonacci_sphere()
    
    fig_velocity = plt.figure(figsize=(20,20))
    # fig_icr      = plt.figure(figsize=(20,20))
    fig_img      = plt.figure(figsize=(20,20))
    
    for i in range(num_contact_samples):
        TP, FP, FN, TN, image  = sort_confusion(dataloader, model)
        classes = [TP  , FP  ,  FN , TN  ]
        legends = ['TP', 'FP', 'FN', 'TN']
        colors  = ['b' , 'y' , 'r' , 'k' ]
        
        num_row = int(np.ceil(np.sqrt(num_contact_samples)))
        ax_velocity = fig_velocity.add_subplot(num_row,num_row,i+1,projection='3d')
        # ax_icr      = fig_icr.add_subplot(num_row,num_row,i+1)
        ax_img = fig_img.add_subplot(num_row,num_row,i+1)
        ax_img.imshow( image[0].cpu().permute(1,2,0).numpy())
        
        for color, each_class in enumerate(classes):
            indices = each_class
            color = colors[color]
            # Plot 3d velocity
            velocity_x = np.take(velocity[:,0],indices)
            velocity_y = np.take(velocity[:,1],indices)
            velocity_z = np.rad2deg(np.take(velocity[:,2],indices))
            # ax_velocity.scatter(velocity_x, velocity_y, velocity_z, c=color, s = 1.5) 
            ax_velocity.scatter(velocity_x, velocity_y, velocity_z, c=color, s = 200) 
            
            # Plot 2d icr
            # icr_x = np.take(icr[:,0],indices)
            # icr_y = np.take(icr[:,1],indices)
            # ax_icr.scatter(icr_x, icr_y, c=color, s = 1.5)
            
        ax_velocity.view_init(elev = 0,azim = 0)
        ax_velocity.set_xlabel(r"$IRC$ [m]")
        ax_velocity.set_ylabel(r"$Gripper Width$ [m]")
        ax_velocity.set_zlabel(r"$Angl$ [degree]", rotation=0)
        # ax_velocity.set_xlabel(r"$v_x$ [m/s]")
        # ax_velocity.set_ylabel(r"$v_y$ [m/s]")
        # ax_velocity.set_zlabel(r"$\omega$ [rad]", rotation=0)
        ax_velocity.set_box_aspect((1,2,2))
        ax_velocity.legend(legends)
        ax_velocity.grid(False)
        ax_velocity.set_title(f'Acc: {100*(len(TP)+len(TN))/(len(TP)+len(TN)+len(FP)+len(FN)): 0.2f}% | Prec: {100*len(TP)/(len(TP)+len(FP)): 0.2f}% | Rec: {100*len(TP)/(len(TP)+len(FN)): 0.2f}%')
        
        
    plt.show()  