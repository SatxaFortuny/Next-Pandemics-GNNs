import math
import os
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdPartialCharges
from rdkit import RDLogger
from typing import List, Tuple, Optional, Dict
import re
from torch_geometric.data import Data
import torch.nn as nn
from rdkit.Chem.rdmolfiles import SDMolSupplier
from rdkit.Chem.rdchem import Mol

RDLogger.DisableLog('rdApp.*')

from atom_and_bond_embedding import *

class EmbeddingEncoder(torch.nn.Module):
    def __init__(self, atom_emb_dim, hybrid_emb_dim, bond_emb_dim):
        super().__init__()
        n_bond_types = 6
        hybridization_types = 8
        periodic_elements = 39
        # El embedding del atomo con el numero de atomos diferentes y la dimnension del vector que queremos
        self.atom_embedding = torch.nn.Embedding(periodic_elements, atom_emb_dim)
        # Lo mismo con lo demás
        self.hybrid_embedding = torch.nn.Embedding(hybridization_types, hybrid_emb_dim)
        self.bond_embedding = torch.nn.Embedding(n_bond_types, bond_emb_dim)

    def encode_nodes(self, x):
        symbol_idx = x[:, 0].long()
        hybrid_idx = x[:, 1].long()
        cont_features = x[:, 2:]  # [degree, numH, aromatic]

        symbol_emb = self.atom_embedding(symbol_idx)
        hybrid_emb = self.hybrid_embedding(hybrid_idx)
        return torch.cat([symbol_emb, hybrid_emb, cont_features], dim=1)

    def encode_edges(self, edge_attr):
        bond_idx = edge_attr[:, 0].long()
        bond_dist = edge_attr[:, 1].unsqueeze(1)
        bond_emb = self.bond_embedding(bond_idx)
        return torch.cat([bond_emb, bond_dist], dim=1)


class Ligand:
    """
    Represents a ligand molecule with its graph representation.

    Attributes:
        sdf_path: Path to the SDF file
        mol: RDKit molecule object
        name: Ligand name from PDB
        features: Node features for each atom
        edges: Edge list (connectivity)
        edge_features: Edge features
        coordinates: 3D coordinates (normalized)
        atom_names_pdb: Atom names from PDB file (if available)
        num_atoms: Number of non-hydrogen atoms
    """

    # --- Definiciones de Patrones SMARTS para H-Bonds ---
    # Donador: Generalmente N u O con al menos un H unido
    HBD_PATTERN = Chem.MolFromSmarts('[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]')

    # Aceptor: N u O con pares libres disponibles (definición estándar de Lipinski)
    HBA_PATTERN = Chem.MolFromSmarts(
        '[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]')

    PERIODIC_ELEMENTS = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                         'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Sb', 'Sn', 'Ag',
                         'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu',  'Ni',
                         'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pb', 'Unknown']

    HYBRIDIZATION = ['S', 'SP', 'SP2', 'SP2D', 'SP3', 'SP3D', 'OTHER', 'UNSPECIFIED']

    BOND_TYPES_COVALENT = ["NONE","SINGLE","DOUBLE","TRIPLE","QUADRUPLE","QUINTUPLE","HEXTUPLE","ONEANDAHALF",
                           "TWOANDAHALF","THREEANDAHALF", "FOURANDAHALF","FIVEANDAHALF","AROMATIC","IONIC", "HYDROGEN",
                           "THREECENTER","DATIVEONE","DATIVE","DATIVEL","DATIVER","OTHER", "ZERO","PEPTIDE"]



    BOND_TYPES_NON_COVALENT = ["hbond","weak_hbond", "xbond", "ionic", "metal", "aromatic", "hydrophobic",
                               "carbonyl", "polar","weak_polar","CARBONPI", "CATIONPI", "METSULPHURPI", "EF", "FT"]

    BOND_TYPES = {"SINGLE":0,"DOUBLE":1,"TRIPLE":2,"AROMATIC":3,"IONIC":4, "HYDROGEN":4, "ZERO":4,"OTHER":5,"UNSPECIFIED":5,"NON_COVALENT":4}

    AA_3TO1 = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E",
               "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
               "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
               "TYR": "Y", "VAL": "V", "LIG": "X"}

    ATOM_TO_IDX = {e: i for i, e in enumerate(PERIODIC_ELEMENTS)}
    AA_TO_IDX = {a: i for i, a in enumerate(AA_3TO1.keys())}
    HYB_TO_IDX = {h: i for i, h in enumerate(HYBRIDIZATION)}
    BOND_TO_IDX = {b: i for i, b in enumerate(BOND_TYPES_COVALENT)}

    def __init__(self, sdf_path: str, node_feature_selection: int = 0,
                 use_embedding_nodes: bool = False, use_embedding_edges: bool = False):
        """
        Initialize a Ligand object from an SDF file.

        Args:
            sdf_path: Path to the SDF file
            node_feature_selection: Feature set to use (0 or 1)
            use_embeddings: If True, use learned embeddings instead of one-hot encoding
        """
        self.sdf_path = sdf_path
        self.node_feature_selection = node_feature_selection
        self.use_embedding_nodes = use_embedding_nodes
        self.use_embedding_edges = use_embedding_edges
        self.mol = None
        self.name = None
        self.features = []
        self.edges = []
        self.edge_features = []
        self.coordinates = []
        self.atom_names_pdb = None
        self.num_atoms = 0
        self.index_H_atoms = []

        # Embedding-specific attributes
        self.atom_indices = None
        self.aa_indices = None
        self.hyb_indices = None
        #self.continuous_features = None
        #self.covalent_bonds = None
        self.bond_indices = None
        self.edge_distances = None
        self.hbd_indices = {}
        self.hba_indices = {}

        # Load molecule
        self._load_molecule()

    def _extract_atom_names_from_cif(self):
        """
        Read a CIF file and extract atom names from the _atom_site.label_atom_id column.
        Returns a list of atom names (e.g., ['N', 'C', 'C1', 'N1', ...]).
        """
        cif_path = self.sdf_path.replace("Ligand_SDF", "Ligand_CIF").replace(".sdf", ".cif")
        atom_names = []
        if self.sdf_path.split('\\')[-1].split('_')[0] == '9G0I':
            d = 0
        with open(cif_path, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith(("HETATM", "ATOM")):
                    fields = line.split()
                    if len(fields) >= 4:  # ensure column exists
                       if not fields[3].startswith('H') and (fields[4] == '.' or fields[4] == 'A'):
                            atom_names.append(fields[3])
        return atom_names


    def _load_molecule(self):
        """Load molecule from SDF file."""

        mol_supplier = Chem.SDMolSupplier(self.sdf_path, sanitize=True, removeHs=False)

        self.mol = mol_supplier[0]

        if self.mol is None:
            raise ValueError(f"Invalid SDF file or molecule at: {self.sdf_path}")

        # Extract name
        self.name = self.mol.GetProp("_Name") if self.mol.HasProp("_Name") else "UNK"
        self.atom_names_pdb = self._extract_atom_names_from_cif()
        # Compute charges
        rdPartialCharges.ComputeGasteigerCharges(self.mol)

        hbd_matches = self.mol.GetSubstructMatches(self.HBD_PATTERN)
        self.hbd_indices = {idx[0] for idx in hbd_matches}  # Usamos set para búsqueda rápida O(1)

        hba_matches = self.mol.GetSubstructMatches(self.HBA_PATTERN)
        self.hba_indices = {idx[0] for idx in hba_matches}


    def build_graph(self, include_coords: bool = False,
                    use_pdb_coords: bool = False,
                    pdb_path: Optional[str] = None,
                    ligand_name_conversion: Optional[object] = None,
                    min_coord: Optional[List[float]] = None,
                    max_coord: Optional[List[float]] = None,
                    include_edge_distances: bool = False, simplified_edge_distances: bool = False) -> bool:
        """
        Build the molecular graph representation.

        Args:
            include_coords: Whether to include 3D coordinates in node features
            use_pdb_coords: Whether to use coordinates from PDB file
            pdb_path: Path to PDB file (required if use_pdb_coords=True)
            ligand_name_conversion: DataFrame for ligand name conversion
            min_coord: Minimum coordinates for normalization
            max_coord: Maximum coordinates for normalization
            include_edge_distances: Whether to include distances in edge features

        Returns:
            True if successful, False otherwise
        """
        # Extract atom features
        try:
            self._extract_atom_features()
            # Add coordinates if requested
            if include_coords or include_edge_distances:
                if use_pdb_coords and pdb_path:
                    success = self._add_pdb_coordinates(pdb_path, ligand_name_conversion,min_coord, max_coord)
                    if not success:
                        return False
                else:
                    self._add_sdf_coordinates(min_coord, max_coord)

            # Build edges
            self._build_edges(simplified_edge_distances=simplified_edge_distances)

            # Add distances to edges if requested
            if include_edge_distances:
                self._add_edge_distances()

                # Remove coords from node features if not explicitly requested
                if not include_coords:
                    self.features = [f[:-3] for f in self.features]


            return True
        except Exception as e:

            print(f"Error building graph for {self.name}: {e}")

            return False


    def _extract_atom_features(self):
        """Extract features for each non-hydrogen atom."""
        if self.use_embedding_nodes:
            self._extract_atom_features_for_embedding()
            #self.num_atoms = len(self.continuous_features)
        else:
            self._extract_atom_features_one_hot()
        if not len(self.features) == len(self.atom_names_pdb):
            print(f'{self.sdf_path.split('\\')[-1].split('_')[0]}: NOT SAME NUMBER OF NODES AND ATOMS NAMES IN THE PDB, CHECK IT')
        self.num_atoms = len(self.features)


    def _extract_atom_features_one_hot(self):
        """Extract one-hot encoded features for each non-hydrogen atom."""
        self.features = []
        self.index_H_atoms = []

        for i, atom in enumerate(self.mol.GetAtoms()):
            if atom.GetSymbol() != 'H':
                feat = self._atom_features_one_hot(atom)
                self.features.append(feat)
            else:
                self.index_H_atoms.append(i)


    def _extract_atom_features_for_embedding(self):
        """Extract features as indices for neural network embedding."""
        atom_idx_list = []
        aa_idx_list = []
        hyb_idx_list = []
        cont_list = []
        self.index_H_atoms = []

        for i, atom in enumerate(self.mol.GetAtoms()):
            if atom.GetSymbol() != 'H':
                cont = self._atom_features_for_embedding(atom)
                #atom_idx_list.append(atom_idx)
                #aa_idx_list.append(aa_idx)
                #hyb_idx_list.append(hyb_idx)
                cont_list.append(cont)
            else:
                self.index_H_atoms.append(i)

        # Store as tensors for embedding
       # self.atom_indices = torch.tensor(atom_idx_list, dtype=torch.long)
       # self.aa_indices = torch.tensor(aa_idx_list, dtype=torch.long)
       # self.hyb_indices = torch.tensor(hyb_idx_list, dtype=torch.long)
        cont_array = np.array(cont_list)
       # self.continuous_features = torch.tensor(cont_array, dtype=torch.float)

        # Keep features empty - will be filled by embedding layer
        self.features = torch.tensor(cont_array, dtype=torch.float)


    def _atom_features_for_embedding(self, atom) -> Tuple[int, int, int, np.ndarray]:
        """
        Extract atom features as indices for embedding.

        Returns:
            atom_idx: Index in PERIODIC_ELEMENTS
            aa_idx: Index in AA_3TO1 (always LIG for ligands)
            hyb_idx: Index in HYBRIDIZATION
            cont: Continuous features [degree, num_h, aromatic]
        """



        atom_symbol = atom.GetSymbol()
        atom_idx = self.ATOM_TO_IDX.get(atom_symbol, self.ATOM_TO_IDX['Unknown'])

        aa_idx = self.AA_TO_IDX.get("LIG", self.AA_TO_IDX['LIG'])

        hyb = atom.GetHybridization().name
        hyb_idx = self.HYB_TO_IDX.get(hyb, self.HYB_TO_IDX['UNSPECIFIED'])

        # Continuous features
        degree = atom.GetDegree() / 10.0
        num_h = atom.GetTotalNumHs() / 10.0
        aromatic = float(atom.GetIsAromatic())
      #  formal_charge = float(atom.GetFormalCharge())
        pos_idx = atom.GetIdx()
        is_donor = float(pos_idx in self.hbd_indices)
        is_acceptor = float(pos_idx in self.hba_indices)

      #  try:
      #      gasteiger_charge = atom.GetDoubleProp('_GasteigerCharge')
      #      if math.isnan(gasteiger_charge) or math.isinf(gasteiger_charge):
      #          gasteiger_charge = 0.0
      #  except KeyError:
      #      gasteiger_charge = 0.0

     #   if self.node_feature_selection == 0:
     #       cont = np.array([degree, num_h, aromatic], dtype=float)
     #   elif self.node_feature_selection == 1:
     #       implicit_val = atom.GetImplicitValence() / 10.0
     #       cont = np.array([degree, num_h, aromatic, implicit_val], dtype=float)

        return [atom_idx, hyb_idx, degree, num_h, aromatic, is_donor, is_acceptor] # [atom_idx, hyb_idx, degree, num_h, aromatic, formal_charge, gasteiger_charge, is_donor, is_acceptor]


    def _atom_features_one_hot(self, atom) -> np.ndarray:
        """
        Extract features for a single atom.

        Args:
            atom: RDKit atom object

        Returns:
            Feature vector as numpy array
        """
        element_vec = self._one_of_k_encoding_unk(atom.GetSymbol(), self.PERIODIC_ELEMENTS)
        degree = [atom.GetDegree() / 10.0]
        num_h = [atom.GetTotalNumHs() / 10.0]
        is_aromatic = [float(atom.GetIsAromatic())]
        resname_vec = self._one_of_k_encoding_unk("LIG", list(self.AA_3TO1.keys()))

        if self.node_feature_selection == 0:
            hybrid = atom.GetHybridization().name
            hybridization = self._one_of_k_encoding_unk(hybrid, self.HYBRIDIZATION)
            features = element_vec + resname_vec + hybridization + degree + num_h + is_aromatic

        elif self.node_feature_selection == 1:
            implicit_val = [atom.GetImplicitValence() / 10.0]
            features = element_vec + resname_vec  + degree + num_h + is_aromatic + implicit_val

        return np.array(features)


    def _add_sdf_coordinates(self, min_coord: List[float], max_coord: List[float]):
        """Add normalized coordinates from SDF conformer."""
        conf = self.mol.GetConformer()
        self.coordinates = []
        features_with_coords = []

        #if self.use_embedding_nodes:
        #    features = self.continuous_features
        #else:
        #    features = self.features

        for i, feat in enumerate(self.features):
            pos = conf.GetAtomPosition(i)
            coords = self._normalize_3d_coord([pos.x, pos.y, pos.z], min_coord, max_coord)
            self.coordinates.append(coords)
            features_with_coords.append(np.concatenate([feat, coords]))

        self.features = features_with_coords


    def _add_pdb_coordinates(self, pdb_path: str, ligand_name_conversion: object,
                             min_coord: List[float], max_coord: List[float]) -> bool:
        """Add normalized coordinates from PDB/CIF file."""
        # Construct paths
        pdb_dir = os.path.dirname(os.path.dirname(pdb_path))
        pdb_id = os.path.basename(pdb_path).split('_')[0]

        cif_path = os.path.join(pdb_dir, 'Ligand', 'Ligand_CIF', f'{pdb_id}_ligand.cif')

        # Get ligand name from conversion table
        ligand_pdb = ligand_name_conversion.loc[
            ligand_name_conversion['PDB_ID'] == self.name.split("-")[0],
            'Ligand'
        ].tolist()[0]

        # Extract atom names and coordinates
        self.atom_names_pdb, self.coordinates = self._get_atom_names_and_coords_from_cif(
            cif_path, ligand_pdb, min_coord, max_coord
        )

        if len(self.atom_names_pdb) == 0 or len(self.coordinates) == 0:
            return False

        # Append coordinates to features
        features_with_coords = []
        for feat, coord in zip(self.features, self.coordinates):
            features_with_coords.append(np.concatenate([feat, coord]))

        self.features = features_with_coords
        return True


    def _get_atom_names_and_coords_from_cif(self, cif_path: str, ligand_name: str,
                                            min_coord: List[float],
                                            max_coord: List[float]) -> Tuple[List[str], List[np.ndarray]]:
        """Extract atom names and coordinates from CIF file."""
        atom_names = []
        coords = []

        with open(cif_path, 'r') as f:
            for line in f:
                if line.startswith("HETATM") or line.startswith("ATOM"):
                    res_name = line.split(' ')[6]
                    if res_name == ligand_name:
                        split_line = line.split()
                        atom_name = line.split(' ')[4]

                        if not atom_name.startswith('H'):
                            atom_name = self._clean_atom_name(atom_name)
                            if atom_name not in atom_names:
                                atom_names.append(atom_name)
                                xyz = self._normalize_3d_coord(
                                    [split_line[-6], split_line[-5], split_line[-4]],
                                    min_coord, max_coord
                                )
                                coords.append(xyz)

        return atom_names, coords


    def _build_edges(self, simplified_edge_distances:bool =False):
        """Build edge list and edge features."""
        if simplified_edge_distances:
            self._simplified_edge_features()
        elif self.use_embedding_edges:
            self._build_edges_for_embedding()
        else:
            self._build_edges_one_hot()
        assert len(self.edges) == len(self.edge_features), f"Number of Edges: {len(self.edges)}, Number of rows with Edge Features: {len(self.edge_features)}"
        self.num_edges = len(self.edges)

    def _simplified_edge_features(self):
        """Build edges with one-hot encoded bond features."""
        self.edges = []
        self.edge_features = []
        for bond in self.mol.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            # Skip hydrogen atoms
            if a not in self.index_H_atoms and b not in self.index_H_atoms:
                if a < self.num_atoms and b < self.num_atoms:
                    self.edges.append([a, b])
                    self.edges.append([b,a])
                   # bond_feat =  self._one_of_k_encoding_unk("covalent", ["covalent","non_covalent"])
                    bond_type = bond.GetBondType().name
                    if bond_type not in list(self.BOND_TYPES.keys()):
                        print(bond_type)
                    number = self.BOND_TYPES.get(bond_type, self.BOND_TYPES['OTHER'])
                    self.edge_features.append(number)
                    self.edge_features.append(number)
                    #try:
                    #    self.edge_features.append(bond_feat / sum(bond_feat))
                    #except TypeError:
                    #    self.edge_features.append(np.array(bond_feat) / sum(bond_feat))

    def _build_edges_one_hot(self):
        """Build edges with one-hot encoded bond features."""
        self.edges = []
        self.edge_features = []
        #bonds_kekulized = Chem.SDMolSupplier(self.sdf_path.replace("Ligand_SDF", "Ligand_SDF_Kekulized"), sanitize=False, removeHs=True)[0]
        for bond in self.mol.GetBonds():

       # if bonds_kekulized is None:
      #     raise ValueError(f"Invalid SDF file or molecule at: {self.sdf_path.replace("Ligand_SDF", "Ligand_SDF_Kekulized")}")

        #for bond in bonds_kekulized.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

            # Skip hydrogen atoms
            if a not in self.index_H_atoms and b not in self.index_H_atoms:
                if a < self.num_atoms and b < self.num_atoms:
                    self.edges.append([a, b])
                    bond_feat = self._bond_features_one_hot(bond.GetBondType().name)
                    self.edge_features.append(bond_feat / sum(bond_feat))


    def _build_edges_for_embedding(self):
        """Build edges with bond type indices for embedding."""
        edges = []
        bond_idx_list = []

        for bond in self.mol.GetBonds():
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

            # Skip hydrogen atoms
            if a not in self.index_H_atoms and b not in self.index_H_atoms:
                # Get number of atoms (for embeddings, use indices length)
                if a < self.num_atoms and b < self.num_atoms:
                    edges.append([a, b])
                   # edges.append([b, a])
                    bond_type = bond.GetBondType().name
                    if bond_type not in list(self.BOND_TYPES.keys()):
                        print(bond_type)
                    bond_idx = self.BOND_TO_IDX.get(bond_type, self.BOND_TO_IDX['OTHER'])
                    bond_idx_list.append(bond_idx)

        self.edges = edges
        self.bond_indices = torch.tensor(bond_idx_list, dtype=torch.long)
        # Keep edge_features empty - will be filled by embedding layer
        self.edge_features = torch.zeros(len(self.bond_indices), len(self.BOND_TYPES_NON_COVALENT))


    def _bond_features_one_hot(self, bond_type: str) -> np.ndarray:
        """Extract features for a bond."""
        return np.array(self._one_of_k_encoding_unk(bond_type,self.BOND_TYPES_COVALENT + self.BOND_TYPES_NON_COVALENT))


    def _add_edge_distances(self):
        """Add Euclidean distances to edge features."""
        edge_features_updated = []
        for (src_idx, tgt_idx), edge_feat in zip(self.edges, self.edge_features):
            src_coords = self.coordinates[src_idx]
            tgt_coords = self.coordinates[tgt_idx]

            distance = np.linalg.norm(src_coords - tgt_coords)
            updated_feat = np.append(edge_feat, distance)
            edge_features_updated.append(updated_feat)

        self.edge_features = edge_features_updated

    @staticmethod
    def _normalize_3d_coord(coord: List[float], min_coord: List[float],
                            max_coord: List[float]) -> np.ndarray:
        """Normalize 3D coordinates using min-max normalization."""
        coord = np.array(coord, dtype=float)
        min_coord = np.array(min_coord, dtype=float)
        max_coord = np.array(max_coord, dtype=float)

        range_coord = max_coord - min_coord
        range_coord[range_coord == 0] = 1.0

        return (coord - min_coord) / range_coord

    @staticmethod
    def _clean_atom_name(atom_name: str) -> str:
        """Clean atom name by extracting element and number."""
        match = re.match(r'^([A-Z]+)(\d+)', atom_name)
        return match.group(1) + match.group(2) if match else atom_name

    @staticmethod
    def _one_of_k_encoding_unk(x, allowable_set: List) -> List[bool]:
        """One-hot encoding with unknown handling."""
        if x not in allowable_set:
            x = allowable_set[-1]
        return [x == s for s in allowable_set]


    def to_graph_dict(self) -> Dict:
        """
        Convert ligand to graph dictionary format.

        Returns:
            Dictionary with graph data
        """
        graph = {
            'name': self.name,
            'num_atoms': self.num_atoms,
            'num_edges': self.num_edges,
            'edges': torch.tensor(self.edges, dtype=torch.long).t().contiguous(),
            'coordinates':  self.coordinates,  #torch.from_numpy(np.asarray(self.coordinates, dtype=np.float32))
            'features': self.features,
            'atom_names_pdb':self.atom_names_pdb,
            'edge_features': torch.stack([torch.from_numpy(x) for x in self.edge_features]).float()
        }

       # if self.use_embedding_nodes:
       #     graph.update({
       #         'atom_indices': self.atom_indices,
       #         'aa_indices': self.aa_indices,
       #         'hyb_indices': self.hyb_indices,
               # 'continuous_features': self.continuous_features,
       #     })
       # if self.use_embedding_edges:
       #     graph.update({'bond_indices': self.bond_indices})
        return graph


    def to_graph_to_data(self, encoder = None):
        """
        Convert ligand graph to Data format.
        """
        self.features = torch.as_tensor(self.features, dtype=torch.float)
        self.y= torch.as_tensor([self.y], dtype=torch.float)
        self.edges = torch.as_tensor(self.edges, dtype=torch.long).t().contiguous()

        if self.edge_features:
            #self.edge_features = torch.stack([torch.as_tensor(x, dtype=torch.float) for x in self.edge_features])
            self.edge_features = torch.tensor(self.edge_features,dtype=torch.float)
        else:
            self.edge_features = torch.empty((0, self.features.size(1)))

        if encoder is not None:
            self.features = encoder.encode_nodes(self.features)
            if self.edge_features.numel() > 0:
                self.edge_features = encoder.encode_edges(self.edge_features)

        if not self.name:
            print("No name added to a ligand, check code to fix it")
        #return Data(x=self.features, edge_index=self.edges, edge_attr=self.edge_features, name=self.name, y = self.y)

        return Data(x=self.features, edge_index=self.edges, edge_attr=self.edge_features, num_nodes = self.num_atoms, name=self.name)


    def apply_embeddings(self, atom_embedder=None, bond_embedder=None):

        """
        Apply embedding layers to convert indices to feature vectors.

        Args:
            atom_embedder: AtomEmbedding layer
            bond_embedder: BondEmbedding layer (optional)

        Returns:
            node_features: Embedded node features [N, emb_dim]
            edge_features: Embedded edge features [E, emb_dim] or None
        """

        if self.use_embedding_nodes:
            node_features = atom_embedder(self.atom_indices,self.aa_indices,self.hyb_indices,self.features)

            #if self.coordinates:
            #    node_features = torch.cat(
            #        [node_features, torch.tensor(self.coordinates, dtype=torch.float)],
            #        dim=-1
            #    )
            self.features = node_features

        # Apply bond embedding if provided
        if self.use_embedding_edges and bond_embedder is not None:
            edge_features = bond_embedder(self.bond_indices, self.edge_features)
            if hasattr(self, 'edge_distances'):
                if self.edge_distances is not None:
                    edge_features = torch.cat([edge_features, self.edge_distances], dim=-1)
            self.edge_features = edge_features


    def get_smiles(self, canonical: bool = True) -> str:
        """
        Get SMILES representation of the molecule.

        Args:
            canonical: Whether to return canonical SMILES

        Returns:
            SMILES string
        """
        return Chem.MolToSmiles(self.mol, isomericSmiles=not canonical)

    def __repr__(self) -> str:
        return (f"Ligand(name='{self.name}', num_atoms={self.num_atoms}, "
                f"num_edges={len(self.edges)})")


class LigandCollection:
    """
    Manages a collection of Ligand objects.

    Attributes:
        ligands: Dictionary mapping ligand names/IDs to Ligand objects
        graph_data: Dictionary mapping ligand IDs to graph representations
    """

    def __init__(self):
        self.ligands = {}
        self.graph_data = {}
        self.failed_ligands = []
        self.use_embedding_nodes = False
        self.use_embedding_edges = False

        # Embedding layers (initialized when needed)
        self.atom_embedder = None
        self.bond_embedder = None

    def add_ligand(self, ligand_id: str, sdf_path: str,
                   node_feature_selection: int = 0,
                   use_embedding_nodes: bool = False,
                   use_embedding_edges:bool = False, simplified_edge_distances:bool = False) -> bool:
        """
        Add a single ligand to the collection.

        Args:
            ligand_id: Unique identifier for the ligand
            sdf_path: Path to the SDF file
            node_feature_selection: Feature set to use
            use_embeddings: Whether to use neural network embeddings

        Returns:
            True if successful, False otherwise
        """
        try:
            ligand = Ligand(sdf_path, node_feature_selection, use_embedding_nodes, use_embedding_edges)
            self.ligands[ligand_id] = ligand
            self.use_embedding_nodes = use_embedding_nodes
            self.use_embedding_edges = use_embedding_edges
            return True
        except Exception as e:
            print(f"Failed to load ligand {ligand_id}: {e}")
            self.failed_ligands.append(ligand_id)
            return False

    def load_from_directory(self, sdf_directory: str,
                            node_feature_selection: int = 0,
                            use_embedding_nodes: bool = False,
                            use_embedding_edges: bool = False,
                            file_pattern: str = "*.sdf", simplified_edge_distances:bool = False) -> int:
        """
        Load all ligands from a directory.

        Args:
            sdf_directory: Path to directory containing SDF files
            node_feature_selection: Feature set to use
            use_embeddings: Whether to use neural network embeddings
            file_pattern: File pattern to match (default: "*.sdf")

        Returns:
            Number of successfully loaded ligands
        """
        import glob
        self.use_embedding_nodes = use_embedding_nodes
        if simplified_edge_distances:
            use_embedding_edges = False
        self.use_embedding_edges = use_embedding_edges
        sdf_files = glob.glob(os.path.join(sdf_directory, file_pattern))
        success_count = 0

        for sdf_path in sdf_files:
            # Extract ligand ID from filename (e.g., "1A2B_ligand.sdf" -> "1A2B")
            ligand_id = os.path.basename(sdf_path).split('_')[0]
            ligand_id = ligand_id.split('-')[0]
            ligand_id = ligand_id.upper()
            if self.add_ligand(ligand_id, sdf_path, node_feature_selection, use_embedding_nodes, use_embedding_edges, simplified_edge_distances=simplified_edge_distances):
                success_count += 1

        print(f"Loaded {success_count}/{len(sdf_files)} ligands from {sdf_directory}")
        if self.failed_ligands:
            print(f"Failed ligands: {self.failed_ligands}")

        return success_count

    def load_from_dict(self, ligand_dict: Dict[str, str],
                       sdf_directory: str,
                       node_feature_selection: int = 0,
                       use_embedding_nodes: bool = False,
                       use_embedding_edges: bool = False) -> int:
        """
        Load ligands from a dictionary mapping IDs to names.

        Args:
            ligand_dict: Dictionary mapping ligand IDs to names
            sdf_directory: Directory containing SDF files
            node_feature_selection: Feature set to use
            use_embeddings: Whether to use neural network embeddings

        Returns:
            Number of successfully loaded ligands
        """
        success_count = 0

        for ligand_id, ligand_name in ligand_dict.items():
            sdf_path = os.path.join(sdf_directory, f"{ligand_id}_ligand.sdf")

            if os.path.exists(sdf_path):
                if self.add_ligand(ligand_id, sdf_path, node_feature_selection, use_embedding_nodes, use_embedding_edges):
                    success_count += 1
            else:
                print(f"SDF file not found for {ligand_id}: {sdf_path}")
                self.failed_ligands.append(ligand_id)

        print(f"Loaded {success_count}/{len(ligand_dict)} ligands")
        return success_count


    def initialize_node_embedders(self, emb_dim_atom: int = 16, emb_dim_aa: int = 8,
                                  emb_dim_hyb: int = 3):
        """
        Initialize embedding layers for node features (atoms and amino acids).

        Args:
            emb_dim_atom: Embedding dimension for atom types
            emb_dim_aa: Embedding dimension for amino acid types
            emb_dim_hyb: Embedding dimension for hybridization
        """
        if not self.use_embedding_nodes:
            raise ValueError("Cannot initialize node embedders when use_embeddings_nodes=False")

        self.atom_embedder = AtomEmbedding(
            n_atoms=len(Ligand.PERIODIC_ELEMENTS),
            n_aa=len(Ligand.AA_3TO1),
            n_hyb=len(Ligand.HYBRIDIZATION),
            emb_dim_atom=emb_dim_atom,
            emb_dim_aa=emb_dim_aa,
            emb_dim_hyb=emb_dim_hyb
        )

        print(f"Initialized node embedder: Atom output dim = {self.atom_embedder.output_dim}")


    def initialize_edge_embedders(self, emb_dim_bond: int = 8):
        """
        Initialize embedding layers for edge features (bonds).

        Args:
            emb_dim_bond: Embedding dimension for bond types
        """
        if not self.use_embedding_edges:
            raise ValueError("Cannot initialize edge embedders when use_embeddings_edges=False")

        self.bond_embedder = BondEmbedding(
            n_bonds=len(Ligand.BOND_TYPES_COVALENT),
            emb_dim=emb_dim_bond
        )

        print(f"Initialized edge embedder: Bond output dim = {self.bond_embedder.output_dim}")


    def build_all_graphs(self, include_coords: bool = False,
                         use_pdb_coords: bool = False,
                         pdb_directory: Optional[str] = None,
                         ligand_name_conversion: Optional[object] = None,
                         min_coord: Optional[List[float]] = None,
                         max_coord: Optional[List[float]] = None,
                         affinity_data=[],
                         include_edge_distances: bool = False, simplified_edge_distances: bool = False) -> int:
        """
        Build graph representations for all ligands in the collection.

        Args:
            include_coords: Whether to include 3D coordinates
            use_pdb_coords: Whether to use PDB coordinates
            pdb_directory: Directory containing PDB files
            ligand_name_conversion: DataFrame for name conversion
            min_coord: Minimum coordinates for normalization
            max_coord: Maximum coordinates for normalization
            include_edge_distances: Whether to include edge distances

        Returns:
            Number of successfully built graphs
        """
        success_count = 0
        failed = []
        data_list = []

        encoder = None
       # encoder = EmbeddingEncoder(atom_emb_dim=12,hybrid_emb_dim=3,bond_emb_dim=2)

        for ligand_id, ligand in self.ligands.items():
            pdb_path = None
            if use_pdb_coords and pdb_directory:
                pdb_path = os.path.join(pdb_directory, f"{ligand_id}_protein.pdb")
            if simplified_edge_distances:
                self.use_embedding_edges = False
            try:
                success = ligand.build_graph(include_coords=include_coords,use_pdb_coords=use_pdb_coords,
                                             pdb_path=pdb_path,ligand_name_conversion=ligand_name_conversion,
                                             min_coord=min_coord,max_coord=max_coord,
                                             include_edge_distances=include_edge_distances, simplified_edge_distances=simplified_edge_distances)
                if success:
                   # if self.use_embedding_nodes or self.use_embedding_edges:
                   #     ligand.apply_embeddings(atom_embedder=self.atom_embedder,bond_embedder=self.bond_embedder)
                    ligand.name = ligand_id
                    self.graph_data[ligand_id] = ligand.to_graph_dict()
                    ligand.y =  torch.tensor([affinity_data.loc[affinity_data[0] == ligand_id, 1].iloc[0]], dtype=torch.float)
                    data_list.append(ligand.to_graph_to_data(encoder=encoder))
                    success_count += 1
                else:
                    failed.append(ligand_id)

            except Exception as e:
                print(f"Failed to build graph for {ligand_id}: {e}")
                failed.append(ligand_id)

        print(f"Built {success_count}/{len(self.ligands)} graphs")
        if failed:
            print(f"Failed graphs: {failed}")

        return success_count, data_list

    def get_ligand(self, ligand_id: str) -> Optional[Ligand]:
        """Get a specific ligand by ID."""
        return self.ligands.get(ligand_id)

    def get_graph(self, ligand_id: str) -> Optional[Dict]:
        """Get graph data for a specific ligand."""
        return self.graph_data.get(ligand_id)

    def get_smiles_dict(self) -> Dict[str, str]:
        """Get dictionary mapping ligand IDs to SMILES strings."""
        return {lid: lig.get_smiles() for lid, lig in self.ligands.items()}

    def filter_by_ids(self, ligand_ids: List[str]) -> 'LigandCollection':
        """
        Create a new collection with only specified ligand IDs.

        Args:
            ligand_ids: List of ligand IDs to keep

        Returns:
            New LigandCollection with filtered ligands
        """
        filtered = LigandCollection()
        for lid in ligand_ids:
            if lid in self.ligands:
                filtered.ligands[lid] = self.ligands[lid]
                if lid in self.graph_data:
                    filtered.graph_data[lid] = self.graph_data[lid]

        return filtered

    def export_graph_data(self, output_path: str):
        """
        Export all graph data to a pickle file.

        Args:
            output_path: Path to save the pickle file
        """
        import pickle

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(self.graph_data, f)

        print(f"Exported {len(self.graph_data)} graphs to {output_path}")

    def load_graph_data(self, input_path: str):
        """
        Load graph data from a pickle file.

        Args:
            input_path: Path to the pickle file
        """
        import pickle

        with open(input_path, 'rb') as f:
            self.graph_data = pickle.load(f)

        print(f"Loaded {len(self.graph_data)} graphs from {input_path}")
        return self.graph_data


    def __len__(self) -> int:
        return len(self.ligands)

    def __iter__(self):
        return iter(self.ligands.items())

    def __getitem__(self, ligand_id: str) -> Ligand:
        return self.ligands[ligand_id]

    def __repr__(self) -> str:
        return f"LigandCollection(n_ligands={len(self.ligands)}, n_graphs={len(self.graph_data)})"


# Example usage
if __name__ == "__main__":

    min_coord = [-28, -36, -34]
    max_coord = [39, 37, 42]
    use_embedding_nodes = False
    use_embedding_edges = False

    collection_ligands = LigandCollection()
    collection_ligands.load_from_directory(
        sdf_directory=r"C:\Users\natal\code_binding_data_and_results\data\Mpro-URV\Ligand\Ligand_SDF",
        node_feature_selection=0, use_embedding_nodes=use_embedding_nodes, use_embedding_edges=use_embedding_edges)

    # Initialize embedding layers
    if use_embedding_nodes == True:
        collection_ligands.initialize_node_embedders(emb_dim_atom=16, emb_dim_aa=8, emb_dim_hyb=8)

    if use_embedding_edges == True:
        collection_ligands.initialize_edge_embedders(emb_dim_bond = 8)


    collection_ligands.build_all_graphs(include_coords=True, min_coord=min_coord, max_coord=max_coord,
                                        include_edge_distances=True)

    # Get ligand and apply embeddings
    ligand = collection_ligands.get_ligand("5RGV")


    print("\n" + "=" * 60)
    print("Example 3: Loading from dictionary (like your workflow)")
    print("=" * 60)

    ligands_dict = {"1A2B": "CCCC...", "3XYZ": "CCOC..."}

    collection = LigandCollection()
    collection.load_from_dict(
        ligand_dict=ligands_dict,
        sdf_directory="path/to/Ligand/Ligand_SDF/",
        node_feature_selection=0,
        use_embedding_nodes=True,
        use_embedding_edges=False
    )

    collection.build_all_graphs(
        include_coords=False,
        include_edge_distances=False
    )

    # Export in your existing format
    smile_graph = collection.graph_data
    print(f"Total graphs: {len(smile_graph)}")

    # Export to pickle (compatible with existing code)
    collection.export_graph_data("output/ligand_graphs.pkl")