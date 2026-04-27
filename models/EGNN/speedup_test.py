import torch
import torch.nn as nn
import torch.nn.functional as F

from egnn import E_GCL

from layer import EGNN_Triton_Layer

NUM_NODES = 5000
NUM_EDGES = 40000

F_NODE = 32
F_EDGE = 8
HIDDEN_DIM = 64

print(f"Creating graph with {NUM_NODES} nodes and {NUM_EDGES} edges...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Base (unidirectional) edges ───────────────────────────────────────────────
edge_index = torch.randint(0, NUM_NODES, (2, NUM_EDGES), device=device)
edge_attr  = torch.randn((NUM_EDGES, F_EDGE), device=device)

# Satorras E_GCL is unidirectional (src→dst only), so double the edges so it
# sees the same total message volume as the bidirectional Triton kernel.
edge_index_bi = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # (2, 2*NUM_EDGES)
edge_attr_bi  = torch.cat([edge_attr,  edge_attr],           dim=0)  # (2*NUM_EDGES, F_EDGE)

x   = torch.randn((NUM_NODES, F_NODE), device=device)
pos = torch.randn((NUM_NODES, 3),      device=device)

# ── Satorras E_GCL wrapper ────────────────────────────────────────────────────
# Use E_GCL directly — no embedding_in / embedding_out — so the parameter count
# and compute match the Triton layer exactly.
egcl = E_GCL(
    input_nf=F_NODE,
    output_nf=F_NODE,
    hidden_nf=HIDDEN_DIM,
    edges_in_d=F_EDGE,
).to(device)

class SatorrasWrapper(nn.Module):
    """Feeds the bidirectional edge list so message volume == Triton kernel."""
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x, pos, edge_index, edge_attr):
        # edge_index / edge_attr already doubled by the caller
        h, new_pos, _ = self.layer(x, edge_index, pos, edge_attr=edge_attr)
        return h, new_pos

# ── Triton wrapper ────────────────────────────────────────────────────────────
# The kernel loads coord with tl.arange(0,4) but masks col>=3, so we pad (N,3)
# to (N,4) to satisfy stride arithmetic without leaking garbage data.
class TritonWrapper(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x, pos, edge_index, edge_attr):
        pos4 = F.pad(pos, (0, 1))                           # (N,3) -> (N,4)
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

# Satorras gets the doubled edge list; Triton gets the original (it doubles internally)
time_satorras = run_benchmark(
    SatorrasWrapper(egcl), "Satorras E_GCL",
    x, pos, edge_index_bi, edge_attr_bi
)
time_triton = run_benchmark(
    TritonWrapper(my_egnn), "EGNN Triton",
    x, pos, edge_index, edge_attr
)

print("\n" + "=" * 50)
print(f"Satorras E_GCL : {time_satorras:.3f} ms  (bidirectional edges, no embeddings)")
print(f"Triton EGNN    : {time_triton:.3f} ms  (bidirectional kernel, original edges)")
print(f"Speedup        : {time_satorras / time_triton:.2f}x")
print("=" * 50)