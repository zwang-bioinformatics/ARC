import warnings

import torch
from torch_geometric.nn import GATv2Conv, summary
from torch_geometric.nn import TransformerConv
from torch_geometric.nn import GraphNorm
from torch_geometric.nn import ResGatedGraphConv
from torch_geometric.nn import GINEConv
from torch_geometric.nn import MLP
from torch_geometric.nn import GENConv
from torch_geometric.nn import GeneralConv
from torch_geometric.nn import PDNConv

warnings.filterwarnings(
    "ignore",
    message=r".*deprecated.*",
    category=UserWarning,
)

def create_model(model_name, config):
    if model_name == "GLFP": return GLFP_F(node_dim = config["node_dim"], edge_dim = config["edge_dim"] ).to(config["device"])
    if model_name == "mTransformerConv": return mTransformerConv_F(node_dim = config["node_dim"], edge_dim = config["edge_dim"] ).to(config["device"])
    if model_name == "mix": return mix(node_dim = config["node_dim"], edge_dim = config["edge_dim"] ).to(config["device"])
    if model_name == "ResGatedGraphConv_": return ResGatedGraphConv_(node_dim = config["node_dim"], edge_dim = config["edge_dim"] ).to(config["device"])
    if model_name == "GINEConv_": return GINEConv_(node_dim = config["node_dim"], edge_dim = config["edge_dim"] ).to(config["device"])
    if model_name == "GENConv_": return GENConv_(node_dim = config["node_dim"], edge_dim = config["edge_dim"] ).to(config["device"])
    if model_name == "GeneralConv_": return GeneralConv_(node_dim = config["node_dim"], edge_dim = config["edge_dim"] ).to(config["device"])
    if model_name == "PDNConv_": return PDNConv_(node_dim = config["node_dim"], edge_dim = config["edge_dim"] ).to(config["device"])
    else: raise ValueError(f"Model {model_name} not found")

class GLFP(torch.nn.Module):
    def __init__(self, node_dim, edge_dim):
        super(GLFP, self).__init__()
        # print("GLFP")
        self.conv1 = GATv2Conv(node_dim, out_channels=5, edge_dim=edge_dim, heads=10, concat=True, negative_slope=0.2, dropout=0.3, bias=True)
        self.graph_norm1 = GraphNorm(in_channels=5 * 10)
        self.conv2 = GATv2Conv(5 * 10, out_channels=2, edge_dim=edge_dim, heads=10, concat=True, negative_slope=0.2, dropout=0.3, bias=True)
        self.graph_norm2 = GraphNorm(in_channels=2 * 10)
        self.conv3 = GATv2Conv(2 * 10, out_channels=1, edge_dim=edge_dim, heads=1, concat=False, negative_slope=0.2, dropout=0.3, bias=True)
        

    def forward(self, x, edge_index, edge_attr, batch_idx):
        
        x = self.conv1(x, edge_index, edge_attr)
        x = self.graph_norm1(x, batch_idx)
        x = x.relu()
        
        x = self.conv2(x, edge_index, edge_attr)
        x = self.graph_norm2(x, batch_idx)
        x = x.relu()
        
        x = self.conv3(x, edge_index, edge_attr)
        
        return torch.sigmoid(x)
    
class mTransformerConv(torch.nn.Module):
    def __init__(self, node_dim, edge_dim):
        super(mTransformerConv, self).__init__()
        # print("mTransformerConv")
        self.conv1 = TransformerConv(node_dim, out_channels=5, edge_dim=edge_dim, heads=10, concat=True, beta=True, dropout=0.3, bias=True)
        self.graph_norm1 = GraphNorm(in_channels=5 * 10)
        self.conv2 = TransformerConv(5 * 10, out_channels=2, edge_dim=edge_dim, heads=10, concat=True, beta=True, dropout=0.3, bias=True)
        self.graph_norm2 = GraphNorm(in_channels=2 * 10)
        self.conv3 = TransformerConv(2 * 10, out_channels=1, edge_dim=edge_dim, heads=1, concat=False, beta=True, dropout=0.3, bias=True)
        

    def forward(self, x, edge_index, edge_attr, batch_idx):
        
        x = self.conv1(x, edge_index, edge_attr)
        x = self.graph_norm1(x, batch_idx)
        x = x.relu()
        
        x = self.conv2(x, edge_index, edge_attr)
        x = self.graph_norm2(x, batch_idx)
        x = x.relu()
        
        x = self.conv3(x, edge_index, edge_attr)
        
        return torch.sigmoid(x)
    

class GLFP_F(torch.nn.Module):
    def __init__(self, node_dim, edge_dim):
        super(GLFP_F, self).__init__()
        # print("GLFP")
        self.conv1 = GATv2Conv(node_dim, out_channels=5, edge_dim=edge_dim, heads=10, concat=False, negative_slope=0.2, dropout=0.3, bias=True)
        self.graph_norm1 = GraphNorm(in_channels=5)
        self.conv2 = GATv2Conv(5, out_channels=1, edge_dim=edge_dim, heads=10, concat=False, negative_slope=0.2, dropout=0.3, bias=True)
        

    def forward(self, x, edge_index, edge_attr, batch_idx):
        
        x = self.conv1(x, edge_index, edge_attr)
        x = self.graph_norm1(x, batch_idx)
        x = x.relu()
        
        x = self.conv2(x, edge_index, edge_attr)
        
        return torch.sigmoid(x)
    
class mTransformerConv_F(torch.nn.Module):
    def __init__(self, node_dim, edge_dim):
        super(mTransformerConv_F, self).__init__()
        # print("mTransformerConv_F")
        self.conv1 = TransformerConv(node_dim, out_channels=5, edge_dim=edge_dim, heads=10, concat=False, beta=True, dropout=0.3, bias=True)
        self.graph_norm1 = GraphNorm(in_channels=5)
        self.conv2 = TransformerConv(5, out_channels=1, edge_dim=edge_dim, heads=10, concat=False, beta=True, dropout=0.3, bias=True)
        

    def forward(self, x, edge_index, edge_attr, batch_idx):
        
        x = self.conv1(x, edge_index, edge_attr)
        x = self.graph_norm1(x, batch_idx)
        x = x.relu()
        
        x = self.conv2(x, edge_index, edge_attr)
        
        return torch.sigmoid(x)
    


class mix(torch.nn.Module):
    def __init__(self, node_dim, edge_dim):
        super(mix, self).__init__()
        # print("mix")
        self.conv1 = TransformerConv(node_dim, out_channels=5, edge_dim=edge_dim, heads=10, concat=False, beta=True, dropout=0.3, bias=True)
        self.graph_norm1 = GraphNorm(in_channels=5)
        self.conv2 = GATv2Conv(5, out_channels=1, edge_dim=edge_dim, heads=10, concat=False, negative_slope=0.2, dropout=0.3, bias=True)

    def forward(self, x, edge_index, edge_attr, batch_idx):
        
        x = self.conv1(x, edge_index, edge_attr)
        x = self.graph_norm1(x, batch_idx)
        x = x.relu()
        
        x = self.conv2(x, edge_index, edge_attr)
        
        return torch.sigmoid(x)
    

class GeneralConv_(torch.nn.Module):
    def __init__(self, node_dim, edge_dim):
        super().__init__()
        # print("GeneralConv_")
        self.conv1 = GeneralConv(in_channels=node_dim, out_channels=5, aggr='add', heads=5, attention=True, attention_type="additive", in_edge_channels=edge_dim, bias=True)
        self.graph_norm1 = GraphNorm(in_channels=5)
        self.conv2 = GeneralConv(in_channels=5, out_channels=1, aggr='add', heads=5, attention=True, attention_type="additive", in_edge_channels=edge_dim, bias=True)
        
    def forward(self, x, edge_index, edge_attr, batch_idx):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.graph_norm1(x, batch_idx)
        x = x.relu()
        
        x = self.conv2(x, edge_index, edge_attr)
        
        return torch.sigmoid(x)

class ResGatedGraphConv_(torch.nn.Module):
    def __init__(self, node_dim, edge_dim):
        super(ResGatedGraphConv_, self).__init__()
        # print("ResGatedGraphConv_")
        self.conv1 = ResGatedGraphConv(node_dim, out_channels=5, edge_dim=edge_dim, bias=True)
        self.graph_norm1 = GraphNorm(in_channels=5)
        self.conv2 = ResGatedGraphConv(5, out_channels=1, edge_dim=edge_dim, bias=True)

    def forward(self, x, edge_index, edge_attr, batch_idx):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.graph_norm1(x, batch_idx)
        x = x.relu()
        
        x = self.conv2(x, edge_index, edge_attr)
        
        return torch.sigmoid(x)
    
class GENConv_(torch.nn.Module):
    def __init__(self, node_dim, edge_dim):
        super().__init__()
        # print("GENConv_")
        self.conv1 = GENConv(in_channels=node_dim, out_channels=5, aggr='add', bias=True, msg_norm=True, edge_dim=edge_dim)
        self.graph_norm1 = GraphNorm(in_channels=5)
        self.conv2 = GENConv(in_channels=5, out_channels=1, aggr='add', bias=True, msg_norm=True, edge_dim=edge_dim)
        
    def forward(self, x, edge_index, edge_attr, batch_idx):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.graph_norm1(x, batch_idx)
        x = x.relu()
        
        x = self.conv2(x, edge_index, edge_attr)
        
        return torch.sigmoid(x)
        

class GINEConv_(torch.nn.Module):
    def __init__(self, node_dim, edge_dim):
        super().__init__()
        # print("GINEConv_")
        self.conv1 = GINEConv(nn=MLP([node_dim, 5], bias=True, dropout=0.3, batch_norm=True), train_eps=True, edge_dim=edge_dim)
        self.graph_norm1 = GraphNorm(in_channels=5)
        self.conv2 = GINEConv(nn=MLP([5, 1], bias=True, dropout=0.3, batch_norm=True), train_eps=True, edge_dim=edge_dim)
        
    def forward(self, x, edge_index, edge_attr, batch_idx):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.graph_norm1(x, batch_idx)
        x = x.relu()
        
        x = self.conv2(x, edge_index, edge_attr)
        
        return torch.sigmoid(x)
    
class PDNConv_(torch.nn.Module):
    def __init__(self, node_dim, edge_dim):
        super().__init__()
        # print("PDNConv_")
        self.conv1 = PDNConv(node_dim, out_channels=5, edge_dim=edge_dim, add_self_loops=True, normalize=True, bias=True, hidden_channels=5)
        self.graph_norm1 = GraphNorm(in_channels=5)
        self.conv2 = PDNConv(5, out_channels=1, edge_dim=edge_dim, add_self_loops=True, normalize=True, bias=True, hidden_channels=5)
        
    def forward(self, x, edge_index, edge_attr, batch_idx):
        x = self.conv1(x, edge_index, edge_attr)
        x = self.graph_norm1(x, batch_idx)
        x = x.relu()
        
        x = self.conv2(x, edge_index, edge_attr)
        
        return torch.sigmoid(x)