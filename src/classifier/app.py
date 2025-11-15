import torch

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)

#from torchsummary import summary
from model_utils import cnnAutoencoder

from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

# auto_cnn = cnnAutoencoder(input_shape=(3, 128, 128), latent_dim=1000)
# auto_cnn.to(device)

# auto_cnn.load_state_dict(torch.load('./cnn_autoencoder_dl.pth'))
# auto_cnn.eval()


import mlflow
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
model_name="tracking-quickstart"
model_version = "latest"
model_uri = f"models:/{model_name}/{model_version}"

# Load model
auto_cnn = mlflow.sklearn.load_model(model_uri)
auto_cnn.eval()


import umap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def perform_umap(f_vec, n_neighbors=None, min_dist=None, random_state=None):
    # Perform U-MAP
    if n_neighbors is None and min_dist is None:
        umap_model = umap.UMAP(n_components=2, 
                               random_state=random_state)
    else:
        umap_model = umap.UMAP(n_components=2, 
                               n_neighbors=n_neighbors,
                               min_dist=min_dist,
                               random_state=random_state)
    umap_result = umap_model.fit_transform(f_vec)
    return umap_result

def plot_reduction(reduced_data, original_data, groups=None, category=None, zoom=0.35, savefig=False, outname=None):
    """ This function plots the dimensionally reduced data."""
    
    if 'pca' in category.lower():
        title = 'PCA Results'; xtitle = 'Principal Component 1'; ytitle = 'Principal Component 2'
    elif ('umap' in category.lower()) or ('u-map' in category.lower()):
        title = 'U-Map Results'; xtitle = 'U-Map Dimension 1'; ytitle = 'U-Map Dimension 2'
    else:
        title = 'Dimensionality Reduction Results'; xtitle = 'Reduced Dimension 1'; ytitle = 'Reduced Dimension 2'
    
    plt.figure(figsize=(10, 12))
    plt.rcParams.update({'font.size': 14})
    
    # Plot 2d reduction
    plt.subplot(2,1,1)
    if not groups:
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.5)
    else:
        # group colors
        #cs = ['y','red', 'green', 'lightcoral','limegreen','rosybrown','aquamarine','royalblue','deepskyblue','navy']
        cs =['red','darkgreen','magenta','greenyellow','pink','cyan']
        #cs = ['y','r', 'g', 'b']
        colors = [cs[i] for i in groups]
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], alpha=0.5, c=colors)
        #plt.legend(*scatter.legend_elements(),loc="lower left", title="Classes")
        
        
        #plt.legend((cs[0],cs[1],cs[2],cs[3]),['exp', 'generated_fake', 'generated_real', 'train_set'],loc='upper left',ncol=4)
        #plt.legend(loc='upper left',ncol=4)
        
    plt.title(title); plt.xlabel(xtitle); plt.ylabel(ytitle)
    plt.grid(True)

    # Plot 2d reduction with original images
    plt.subplot(2,1,2)
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], marker='o', s=30, c='b')
    
    for i in range(len(reduced_data)):

        if original_data[i].shape[0] > 1:
            orig_im = original_data[i][0,:]
        else:
            orig_im = original_data[i]
            
        imagebox = OffsetImage(orig_im, zoom=zoom)  # Adjust the zoom factor as needed
        ab = AnnotationBbox(imagebox, (reduced_data[i, 0], reduced_data[i, 1]), frameon=False)
        plt.gca().add_artist(ab)

    plt.title('Original Images Embedded')
    plt.xlabel(xtitle); plt.ylabel(ytitle); plt.title(title + ' with Original Images')
    plt.grid(True)

    plt.tight_layout()
    if savefig:
        plt.savefig(outname)
    plt.show()
    
    
    pass


import torchvision.datasets as dset
import numpy as np

input_size = 128
data_transform=transforms.Compose([transforms.Resize(input_size),
                              transforms.CenterCrop(input_size),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                             ])


# sorted paths, alphabetic order
mix_datasets = dset.ImageFolder(root="/lovelace/xiaoya/GenAI_dataset/dynamic_latent/testset/", transform=data_transform)
val_loader_mix = DataLoader(mix_datasets, batch_size=16, shuffle=False)

f_vec = []
groups = []
val_set_mix = []
with torch.no_grad(): 
    for batch, gs in val_loader_mix:
        batch_f_vec = auto_cnn.encoder(batch.to(device))
        f_vec.append(batch_f_vec.detach().cpu().numpy())
        groups.extend(gs)
        val_set_mix.append(batch)
        
f_vec = np.vstack(f_vec)
val_set_mix = np.vstack(val_set_mix)

print(val_set_mix.shape)
print(f_vec.shape)
print(len(groups))


import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

umap_features = perform_umap(f_vec, n_neighbors=5, min_dist=0.5)
plot_reduction(umap_features, np.squeeze(val_set_mix), groups=groups, category='umap', zoom=0.4, savefig=True, outname='./autoencoder_img_128/all_latent_space_mix.png')


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

optimal_clusters = 2
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
kmeans.fit(f_vec)

# Get the cluster labels
cluster_labels = kmeans.labels_

# Print the cluster labels
print(cluster_labels)


from sklearn.neighbors import NearestNeighbors
import numpy as np
from model_utils import myModels
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def select_model(index):
    model_names = ["classification", "detection"]
    model_name = model_names[cluster_labels[index]]

    if model_name=="classification":
        folder_name="classifier_checkpoints_rings"
        model_name ="vit_p16"
    
        round_info = 'round2'
        num_classes = 2
        feature_extract = True
        input_size = 224
        round2_dir = '/lovelace/xiaoya/scientific_txt2image-main/'+str(folder_name)+'/'+round_info+'/'
        model, input_size = myModels.initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)
        model = torch.load(round2_dir + model_name + '/' + model_name + '_' + str(1000) + '_epochs_classifier.pth')
    
    
    else:
        # Load the pre-trained model
        model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()  # Set the model to evaluation mode
    return model