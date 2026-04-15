import numpy as np
from PIL.features import features
from rdkit import RDLogger
import torch
import torch.nn as nn

RDLogger.DisableLog('rdApp.*')


class AtomEmbedding(nn.Module):
    """
    Neural network embedding for atom features.
    Converts categorical features to learned embeddings and concatenates with continuous features.
    """

    def __init__(self, n_atoms: int = 44,n_aa: int = 22,n_hyb: int = 8,emb_dim_atom: int = 16,emb_dim_aa: int = 8,
                 emb_dim_hyb: int = 8):
        super().__init__()

        self.atom_emb = nn.Embedding(n_atoms, emb_dim_atom)
        self.aa_emb = nn.Embedding(n_aa, emb_dim_aa)
        self.hyb_emb = nn.Embedding(n_hyb, emb_dim_hyb)

        self.emb_dim_atom = emb_dim_atom
        self.emb_dim_aa = emb_dim_aa
        self.emb_dim_hyb = emb_dim_hyb

    def forward(self, atom_idx: torch.Tensor, aa_idx: torch.Tensor,
                hyb_idx: torch.Tensor, cont_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            atom_idx: Atom type indices [N]
            aa_idx: Amino acid indices [N]
            hyb_idx: Hybridization indices [N]
            cont_feat: Continuous features [N, num_cont_features]

        Returns:
            Embedded features [N, total_emb_dim]
        """
        e_atom = self.atom_emb(atom_idx)
        e_aa = self.aa_emb(aa_idx)
        e_hyb = self.hyb_emb(hyb_idx)

        features = to_tensor(cont_feat)
        # Concatenate embeddings + continuous features
        x = torch.cat([e_atom, e_aa, e_hyb, features], dim=-1)
        return x

    @property
    def output_dim(self) -> int:
        """Total output dimension (excluding continuous features)."""
        return self.emb_dim_atom + self.emb_dim_aa + self.emb_dim_hyb


class BondEmbedding(nn.Module):
    """Neural network embedding for bond features."""

    def __init__(self, n_bonds: int = 14, emb_dim: int = 8):
        super().__init__()
        self.bond_emb = nn.Embedding(n_bonds, emb_dim)
        self.emb_dim_bond = emb_dim

    def forward(self, bond_idx: torch.Tensor,non_cov_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bond_idx: Bond type indices [E]
            non_cov_feat: Continuous features [N, num_non_cov_feat]

        Returns:
            Embedded bond features [E, emb_dim_bond]
        """
        e_bond = self.bond_emb(bond_idx)
        nc_features = to_tensor(non_cov_feat)
        x = torch.cat([e_bond, nc_features], dim=-1)
        return x

    @property
    def output_dim(self) -> int:
        return self.emb_dim_bond


def to_tensor(x, dtype=None, device=None):
    """
    Convert x to a torch.Tensor if it isn't one already.
    Handles numpy arrays and Python lists efficiently.
    """
    if torch.is_tensor(x):
        t = x
    elif isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
    else:
        t = torch.tensor(x)

    if dtype is not None:
        t = t.to(dtype)
    if device is not None:
        t = t.to(device)

    return t

