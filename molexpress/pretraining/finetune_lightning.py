import os
os.environ["KERAS_BACKEND"] = "torch" 

from functools import partial
from molexpress import layers
from molexpress.datasets import featurizers
from molexpress.datasets import encoders
from molexpress.ops.chem_ops import get_molecule
import torch
import pandas as pd 
import pytorch_lightning as pl
import wandb
from torch.utils.data import random_split, DataLoader
from functools import partial
from tqdm import tqdm



# Initialize Weights & Biases
wandb.init(project="ms2dip_gnn_finetune")

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

# Graph Neural Network using PyTorch Lightning
class GraphNeuralNetwork(torch.nn.Module):
    
    def __init__(self, dim):
        super().__init__()
        self.gcn1 = layers.GINConv(dim)
        self.gcn2 = layers.GINConv(dim)
        self.gcn3 = layers.GINConv(dim)
        self.gcn4 = layers.GINConv(dim)
        self.gcn5 = layers.GINConv(dim)
        self.gcn6 = layers.GINConv(dim)
        self.readout = layers.ResidueReadout()
        self.mode = "train"

    def forward(self, x):
        x = self.gcn1(x)
        x = self.gcn2(x)
        x = self.gcn3(x)
        x = self.gcn4(x)
        x = self.gcn5(x)
        x = self.gcn6(x)
        if self.mode == "train":
            return x
        elif self.mode == "inference":
            return self.readout(x)

    def set_mode(self, mode):
        self.mode = mode


class NodePrediction(torch.nn.Module):
    
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, input_dim) 
        self.linear2 = torch.nn.Linear(input_dim, output_dim) 
        
    def forward(self, x):
        x = self.linear1(x['node_state'])
        x = torch.nn.functional.relu(x, inplace=False)
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
        x = torch.nn.functional.relu(x, inplace=False)
        x = self.linear2(x)
        return x


class GraphDataset(torch.utils.data.Dataset):
    
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, index):
        graph = peptide_graph_encoder(self.x[index])
        return graph


class GraphModelModule(pl.LightningModule):
    
    def __init__(self, graph_model, node_pred_model, edge_pred_model, lr=1e-7):
        super().__init__()
        self.graph_model = graph_model
        self.node_pred_model = node_pred_model
        self.edge_pred_model = edge_pred_model
        self.loss_fn = torch.nn.BCELoss(reduction='none')
        self.lr = lr

    def forward(self, x):
        graph = self.graph_model(x)
        node_pred = self.node_pred_model(graph)
        edge_pred = self.edge_pred_model(graph)
        return node_pred, edge_pred

    def training_step(self, batch, batch_idx):
        graph = self.graph_model(batch)
        node_pred = self.node_pred_model(graph)
        edge_pred = self.edge_pred_model(graph)
        
        node_loss = self.weighted_loss(node_pred, batch['node_label'], batch['node_loss_weight'])
        edge_loss = self.weighted_loss(edge_pred, batch['edge_label'], batch['edge_loss_weight'])
        loss = node_loss + edge_loss

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,batch_size = 64)
        return loss

    def validation_step(self, batch, batch_idx):
        graph = self.graph_model(batch)
        node_pred = self.node_pred_model(graph)
        edge_pred = self.edge_pred_model(graph)
        
        node_loss = self.weighted_loss(node_pred, batch['node_label'], batch['node_loss_weight'])
        edge_loss = self.weighted_loss(edge_pred, batch['edge_label'], batch['edge_loss_weight'])
        loss = node_loss + edge_loss

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True,batch_size = 64)
        return loss

 
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            list(self.graph_model.parameters()) + 
            list(self.node_pred_model.parameters()) + 
            list(self.edge_pred_model.parameters()), 
            lr=self.lr,
        )
        return optimizer

    def weighted_loss(self, pred, true, weight):
        log = torch.sigmoid(pred)    # Sigmoid() only with BCELoss
        true = torch.from_numpy(true)
        true  = true.to("cuda") 
        weight = torch.from_numpy(weight)
        weight = weight.to("cuda")
        assert true.shape ==log.shape, f"Expected the two inputs to have the same shape"
   
        loss = self.loss_fn(log, true)
        # print(true.get_device(),log.get_device(),loss.get_device(),)
        w_loss = loss * weight[:, None]  # weight[:, None] only with BCELoss
        return torch.mean(w_loss)




graph_model = GraphNeuralNetwork(1280)
node_pred_model = NodePrediction(1280, 11)
edge_pred_model = EdgePrediction(1280 * 2, 6)

model = GraphModelModule.load_from_checkpoint(
    "/home/harikrishnan/molexpress/molexpress/pretraining/model_checkpoint_6_layers_dim1280.ckpt",
    graph_model=graph_model,
    node_pred_model=node_pred_model,
    edge_pred_model=edge_pred_model,
    strict= False
      )

# model = model.load_from_checkpoint("model_checkpoint_shuffled_adamW.ckpt")
# print(model)


data = pd.read_csv("smiles_finetune.csv",names=["smiles"])
data = data.sample(frac=1)
dataset = data["smiles"].apply(lambda x: [x]).to_list()


dataset = GraphDataset(dataset)

# Validation split and DataLoaders
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size],generator= torch.Generator().manual_seed(42))


batch_size = 64
partial_collate_fn = partial(
    peptide_graph_encoder.masked_collate_fn, node_masking_rate=0.3, edge_masking_rate=0.3
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=partial_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=partial_collate_fn,)

# Logging with Weights & Biases
wandb_logger = pl.loggers.WandbLogger()

# Training with PyTorch Lightning Trainer
trainer = pl.Trainer(
    max_epochs=250,
    logger=wandb_logger,
    accelerator="gpu",
    devices="auto",
    log_every_n_steps=1
)

trainer.fit(model, train_loader, val_loader)

# Save model checkpoint
trainer.save_checkpoint("finetuned_model_6_l_d_1280_adam.ckpt")










