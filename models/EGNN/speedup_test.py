import torch
import torch.nn as nn
import torch.nn.functional as F

# Importamos las utilidades de PyTorch Geometric para el dataset real
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader

# Satorras E_GCL — import the bare layer, no embedding wrappers
from egnn_clean import E_GCL
from layer import EGNN_Triton_Layer

# ── Parámetros del modelo ─────────────────────────────────────────────────────
BATCH_SIZE = 32

F_NODE     = 32
F_EDGE     = 16   # must be >= 16 for tl.dot in the Triton kernel
HIDDEN_DIM = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Carga de datos reales (QM9) ───────────────────────────────────────────────
print("Cargando dataset QM9 (se descargará automáticamente la primera vez)...")
dataset = QM9(root='./data/QM9')

# DataLoader agrupa automáticamente las moléculas calculando el "offset" internamente
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Extraemos un único batch para el benchmark
batch = next(iter(loader)).to(device)

# QM9 usa 11 características para nodos y 4 para aristas.
# Hacemos "padding" con ceros para igualarlas a tus requerimientos (32 y 16)
x_real = F.pad(batch.x, (0, F_NODE - batch.x.shape[1]))
edge_attr_real = F.pad(batch.edge_attr, (0, F_EDGE - batch.edge_attr.shape[1]))
pos_real = batch.pos
edge_index_real = batch.edge_index

print(f"\n--- Estadísticas del Batch QM9 (Real) ---")
print(f"Batch size      : {BATCH_SIZE} moléculas")
print(f"Nodos totales   : {x_real.shape[0]} (Promedio de ~{x_real.shape[0]/BATCH_SIZE:.1f} átomos por molécula)")
print(f"Aristas totales : {edge_index_real.shape[1]}")
print(f"Dimensión x     : {x_real.shape}")
print(f"Dimensión edges : {edge_attr_real.shape}")

# ── Satorras E_GCL wrapper ────────────────────────────────────────────────────
egcl = E_GCL(
    input_nf=F_NODE,
    output_nf=F_NODE,
    hidden_nf=HIDDEN_DIM,
    edges_in_d=F_EDGE,
).to(device)

class SatorrasWrapper(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x, pos, edge_index, edge_attr):
        h, new_pos, _ = self.layer(x, edge_index, pos, edge_attr=edge_attr)
        return h, new_pos

# ── Triton wrapper ────────────────────────────────────────────────────────────
class TritonWrapper(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x, pos, edge_index, edge_attr):
        pos4 = F.pad(pos, (0, 1))
        new_feat, new_coord4 = self.layer(x, pos4, edge_index, edge_attr)
        return new_feat, new_coord4[:, :3]

my_egnn = EGNN_Triton_Layer(
    f_node=F_NODE,
    f_edge=F_EDGE,
    msg_hidden_dim=HIDDEN_DIM,
    msg_out_feat=F_NODE,
    mov_hidden_dim=HIDDEN_DIM,
    node_hidden_dim=HIDDEN_DIM,
    rbf_dim=1,
    rbf_gamma=10.0,
).to(device)

# ── Benchmark ─────────────────────────────────────────────────────────────────
def run_benchmark(model, name, x, pos, edge_index, edge_attr, iters=200, warmup=20):
    model.eval()
    print(f"\n[{name}] Warming up ({warmup} iters)...")
    with torch.no_grad():
        for _ in range(warmup):
            model(x, pos, edge_index, edge_attr)
        torch.cuda.synchronize()

        print(f"[{name}] Measuring {iters} iters...")
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(iters):
            model(x, pos, edge_index, edge_attr)
        end.record()
        torch.cuda.synchronize()

    ms = start.elapsed_time(end) / iters
    print(f"[{name}] {ms:.3f} ms / iter")
    return ms

# Nota: Los grafos de PyG (QM9) ya son bidireccionales por defecto, 
# por lo que pasamos el mismo edge_index a ambos modelos de forma justa.
time_satorras = run_benchmark(
    SatorrasWrapper(egcl), "Satorras E_GCL",
    x_real, pos_real, edge_index_real, edge_attr_real
)

time_triton = run_benchmark(
    TritonWrapper(my_egnn), "EGNN Triton",
    x_real, pos_real, edge_index_real, edge_attr_real
)

print("\n" + "=" * 50)
print(f"Batch size     : {BATCH_SIZE} moléculas (Dataset QM9)")
print(f"Satorras E_GCL : {time_satorras:.3f} ms")
print(f"Triton EGNN    : {time_triton:.3f} ms")
print(f"Speedup        : {time_satorras / time_triton:.2f}x")
print("=" * 50)