import os
os.environ["KERAS_BACKEND"] = "torch" 
from functools import partial
from molexpress import layers
from molexpress.datasets import featurizers
from molexpress.datasets import encoders
from molexpress.ops.chem_ops import get_molecule
import torch
import pandas as pd 
from tqdm import tqdm


class GraphNeuralNetwork(torch.nn.Module):
    
    def __init__(self, dim):
        super().__init__()
        self.gcn1 = layers.GINConv(dim)
        self.gcn2 = layers.GINConv(dim)
        self.gcn3 = layers.GINConv(dim)
        self.gcn4 = layers.GINConv(dim)
        
    def forward(self, x):
        x = self.gcn1(x)
        x = self.gcn2(x)
        x = self.gcn3(x)
        x = self.gcn4(x)
        return x


class NodePrediction(torch.nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, input_dim) 
        self.linear2 = torch.nn.Linear(input_dim, output_dim) 
        
    def forward(self, x):
        x = self.linear1(x['node_state'])
        x = torch.nn.functional.relu(x,inplace=False)
        x = self.linear2(x)
        return x


class EdgePrediction(torch.nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, input_dim) 
        self.linear2 = torch.nn.Linear(input_dim, output_dim)
        self.gather_incident = layers.GatherIncident()
        
    def forward(self, x):
        x = self.gather_incident(x) # We do not use edge states but incident node states.
        x = self.linear1(x)
        x = torch.nn.functional.relu(x,inplace=False)
        x = self.linear2(x)
        return x
    

atom_featurizers = [
    featurizers.AtomType(vocab={'C', 'N', 'O'}),
    featurizers.Hybridization(),
]

bond_featurizers = [
    featurizers.BondType(),
    featurizers.Conjugated()
]

peptide_graph_encoder = encoders.PeptideGraphEncoder(
    atom_featurizers=atom_featurizers, 
    bond_featurizers=bond_featurizers,
    self_loops=False, # self_loops True adds one feature dim to edge state
    supports_masking=True, # supports_masking True adds one feature dim to node and edge state
)

data = pd.read_csv("/home/harikrishnan/molexpress-main/molexpress/pretraining/canon_filtered_pubchem.txt",names=["smiles"])

dataset = data["smiles"].apply(lambda x: [x]).to_list()
# print(len(dataset))

class Dataset(torch.utils.data.Dataset):
    
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, index):
        graph = peptide_graph_encoder(self.x[index])
        return graph

torch_dataset = Dataset(dataset)

batch_size = 1024
partial_collate_fn = partial(
    peptide_graph_encoder.masked_collate_fn, node_masking_rate=0.3, edge_masking_rate=0.3)

dataset = torch.utils.data.DataLoader(
    torch_dataset, batch_size=batch_size, collate_fn=partial_collate_fn,num_workers= 8)


graph_model = GraphNeuralNetwork(32).to('cuda')
node_pred_model = NodePrediction(32, 11).to('cuda')
edge_pred_model = EdgePrediction(32 * 2, 6).to('cuda')



optimizer = torch.optim.SGD(
    (
        list(graph_model.parameters()) + 
        list(node_pred_model.parameters()) + 
        list(edge_pred_model.parameters())
    ),
    lr=0.001, momentum=0.5
)
loss_fn = torch.nn.BCELoss(reduction='none') # use BCELoss if node/edge label (initial node/edge state) is multi-hot.
# loss_fn = torch.nn.CrossEntropyLoss(reduction='none') # use CrossEntropyLoss if node/edge label is one-hot.

def weighted_loss(pred, true, weight):
    log = torch.sigmoid(pred)    # Sigmoid() only with BCELoss
    loss = loss_fn(log, true)
    w_loss = loss * weight[:, None]      # weight[:, None] only with BCELoss
    return torch.mean(w_loss,)
    
log_file = "training_log.txt"
epochs = 250
error_log_file = "error_log.txt"

with open(log_file, "w") as f:
    f.write("Epoch,Loss\n")  

# Training loop

def train():

    for epoch in tqdm(range(epochs), total=epochs, desc="Training Progress"):
        loss_sum = 0
        optimizer.zero_grad()

        for ind, x in tqdm(enumerate(dataset),total=dataset.__len__()):


            graph = graph_model(x)

            try:
                node_pred = node_pred_model(graph)

            except Exception as e:
                print(f"Issue is in batch number {ind+1}")
                
                raise e


            try:
                edge_pred = edge_pred_model(graph)

            except Exception as e:
                print(f"Issue is in batch number {ind+1}")
                
                raise e


            node_loss = weighted_loss(node_pred, graph['node_label'], graph['node_loss_weight'])
            edge_loss = weighted_loss(edge_pred, graph['edge_label'], graph['edge_loss_weight'])

            loss = node_loss + edge_loss
            loss.backward()

            loss_sum += loss.item()

            optimizer.step()

                #
        # tqdm.write(f"Epoch {epoch:<3} - Loss {bat:.3f}")


        with open(log_file, "a") as f:
            f.write(f"{epoch},{loss_sum}\n")


    #     if epoch % 5 == 0:
    #         print(f"Epoch {epoch:<3} - Loss {loss_sum:.3f}")




    torch.save({
        'epoch': epochs,
        'model_state_dict': graph_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss_sum,
    }, 'model_checkpoint.pth')

    print("Training complete. Model saved to 'model_checkpoint.pth'.")



if __name__ == "__main__":
    train()




















