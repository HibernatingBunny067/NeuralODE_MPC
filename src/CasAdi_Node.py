import numpy as np
from casadi import tanh,DM,mtimes,vertcat
import torch 

## Cluster -> Load Checkpoint -> Get state -> make network -> expose it 

class CasAdiMLP:
    def __init__(self,cluster:int,
                 device= torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 PREFIX: str = "problem.nodes.0.nodes.0.callable.rk4.block.net.") -> None:
        # self.cluster = cluster
        self.device = device
        self.PREFIX = PREFIX

        self.W1 = torch.zeros((64,5),device=self.device)
        self.b1 = None

        self.W2 = None
        self.b2 = None

        self.W3 = None
        self.b3 = None

        self.W4 = torch.zeros((4,64),device=self.device)
        self.b4 = None

        self.X_scale = DM(np.zeros((4,1),dtype=np.float64))
        self.U_scale = DM(np.zeros((1,1),dtype=np.float64))
        # self.eps = DM(np.full((4,1),1e-6,dtype=np.float64))

        self.__post_init__()

        assert self.W1.shape == (64,5)
        assert self.W4.shape == (4,64)

        print("Model Initialized !")

    def __test__(self):
        """ Function to check the consistency between CasAdi Neural Network and PyTorch Neural Network """
        print("Test passed !")

    def __post_init__(self):

        weights_path = f"weights"

        ckpt = torch.load(
            f"{weights_path}/best_model_64_n.ckpt",
            map_location=self.device
        )

        norm = torch.load(
            f"{weights_path}/normalization_64_n.pt",
            map_location=self.device,
            weights_only=False
        )

        self.X_scale = DM(norm['x_scale'].cpu().numpy().reshape(-1,1))
        self.U_scale = DM(norm['u_scale'].cpu().numpy().reshape(-1,1))

        state = ckpt['state_dict']

        self.W1 = DM(state[self.PREFIX+"linear.0.weight"].cpu().numpy())
        self.b1 = DM(state[self.PREFIX+"linear.0.bias"].cpu().numpy())

        self.W2 = DM(state[self.PREFIX+"linear.1.weight"].cpu().numpy())
        self.b2 = DM(state[self.PREFIX+"linear.1.bias"].cpu().numpy())

        self.W3 = DM(state[self.PREFIX+"linear.2.weight"].cpu().numpy())
        self.b3 = DM(state[self.PREFIX+"linear.2.bias"].cpu().numpy())

        self.W4 = DM(state[self.PREFIX+"linear.3.weight"].cpu().numpy())
        self.b4 = DM(state[self.PREFIX+"linear.3.bias"].cpu().numpy())

    def __call__(self,x,u):

        x_norm = x /( self.X_scale+1e-6)
        u_norm = u / (self.U_scale+1e-6)

        inp = vertcat(x_norm,u_norm)

        z = tanh(mtimes(self.W1,inp)+self.b1)
        z = tanh(mtimes(self.W2,z)+self.b2)
        z = tanh(mtimes(self.W3,z)+self.b3)
        xdot_norm = mtimes(self.W4,z) + self.b4

        xdot = xdot_norm*self.X_scale

        return xdot
