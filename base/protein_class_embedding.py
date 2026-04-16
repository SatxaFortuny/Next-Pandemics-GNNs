from Bio.PDB import PDBParser, NeighborSearch
from scipy.spatial import cKDTree
import json

from ligand_class_embedding import *
from aminoacids_infor import *
import pickle

from atom_and_bond_embedding import AtomEmbedding, BondEmbedding


def export_dict_graph_data(graph, output_path: str):
    """Export all graph data to a pickle file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(graph, f)
    print(f"Exported {len(graph)} pocket graphs to {output_path}")


def load_graph(input_path: str):
    """    Load graph data from a pickle file.
    Args: input_path: Path to the pickle file     """

    with open(input_path, 'rb') as f:
        graph = pickle.load(f)

    print(f"Loaded {len(graph)} graphs from {input_path}")
    return graph


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


class Pocket:
    """
    Represents a protein pocket (binding site) as a graph.

    The pocket can be defined by:
    1. Residues within a distance threshold from the ligand
    2. Residues specified in an interaction JSON file

    Attributes:
        pocket_id: Unique identifier
        pdb_path: Path to PDB file
        residues: Dictionary of residues in the pocket {residue_id: residue_object}
        atom_nodes: Dictionary of atoms in the pocket
        edges: Edge list (covalent and peptide bonds)
        edge_features: Features for each edge
        num_atoms: Number of atoms in the pocket
        coordinates: 3D coordinates of atoms
        atom_index_map: Mapping from atom ID to node index
        use_embedding_nodes: Whether to use learned embeddings for nodes
        use_embedding_edges: Whether to use learned embeddings for edges
    """

    AA_3TO1 =  {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E",
               "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
               "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
               "TYR": "Y", "VAL": "V", "LIG": "X"}

    PERIODIC_ELEMENTS = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
                         'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Sb', 'Sn', 'Ag',
                         'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Ni',
                         'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pb', 'Unknown']

    HYBRIDIZATION = ['S', 'SP', 'SP2', 'SP2D', 'SP3', 'SP3D', 'OTHER', 'UNSPECIFIED']

    BOND_TYPES_COVALENT = ["NONE","UNSPECIFIED","SINGLE","DOUBLE","TRIPLE","QUADRUPLE","QUINTUPLE","HEXTUPLE","ONEANDAHALF",
                           "TWOANDAHALF","THREEANDAHALF", "FOURANDAHALF","FIVEANDAHALF","AROMATIC","IONIC", "HYDROGEN",
                           "THREECENTER","DATIVEONE","DATIVE","DATIVEL","DATIVER","OTHER", "ZERO","PEPTIDE"]
    BOND_TYPES_NON_COVALENT = ["hbond", "weak_hbond", "xbond", "ionic", "metal", "aromatic", "hydrophobic",
                               "carbonyl", "polar", "weak_polar", "CARBONPI", "CATIONPI", "METSULPHURPI", "EF", "FT"]

    BOND_TYPES = {"SINGLE":0,"DOUBLE":1,"TRIPLE":2,"AROMATIC":3,"IONIC":4, "HYDROGEN":4, "ZERO":4,"OTHER":5,"UNSPECIFIED":5,"NON_COVALENT":4}

    ATOM_TO_IDX = {e: i for i, e in enumerate(PERIODIC_ELEMENTS)}
    AA_TO_IDX = {a: i for i, a in enumerate(AA_3TO1.keys())}
    HYB_TO_IDX = {h: i for i, h in enumerate(HYBRIDIZATION)}
    BOND_TO_IDX = {b: i for i, b in enumerate(BOND_TYPES_COVALENT)}


    def __init__(self, pocket_id: str, pdb_path: str,
                 use_embedding_nodes: bool = False,
                 use_embedding_edges: bool = False,
                 chain_id: str = 'A'):
        """
        Initialize a Pocket object.

        Args:
            pocket_id: Unique identifier for the pocket
            pdb_path: Path to the PDB file
            use_embedding_nodes: Whether to use embeddings for node features
            use_embedding_edges: Whether to use embeddings for edge features
            chain_id: Chain ID to extract from PDB (default: 'A')
        """
        self.pocket_id = pocket_id
        self.pdb_path = pdb_path
        self.chain_id = chain_id
        self.use_embedding_nodes = use_embedding_nodes
        self.use_embedding_edges = use_embedding_edges

        # Parse PDB structure
        self.structure = None
        self.parser = PDBParser(QUIET=True)
        self._load_structure()

        # Pocket components
        self.residues = {}
        self.filtered_rows = []
        self.number_atoms_by_residue = {}
        self.selected_residue_ids = []
        self.atom_nodes = {}
        self.edges = []
        self.edge_features = []
        self.atom_index_map = {}
        self.features = []
        self.coordinates = []
        self.not_normalized_coordinates = []
        self.num_atoms = 0
        self.num_edges = 0
        self.bond_indices = None
        self.atom_indices = None
        self.aa_indices = None
        self.hyb_indices = None
        self.edge_distances = None
        self.hbd_indices = {}
        self.hba_indices = {}

    def _load_structure(self):
        """Load structure from PDB file."""
        if not os.path.exists(self.pdb_path):
            raise ValueError(f"PDB file not found: {self.pdb_path}")
        try:
            self.structure = self.parser.get_structure(self.pocket_id, self.pdb_path)
        except Exception as e:
            raise ValueError(f"Failed to load PDB file {self.pdb_path}: {e}")

    def _find_residues_near_coords(self, coords_denorm, ns, distance_threshold, chain_id='A'):
        """        Find residues near a set of 3D coordinates, optionally filtering by chain.        """

        residues_set = set()

        for coord in coords_denorm:
            # Search all residues within distance_threshold
            for res in ns.search(coord, distance_threshold, level="R"):
                # Filter by amino acid type
                if res.get_resname() in self.AA_3TO1:
                    # Filter by chain if specified
                    if chain_id is None or res.get_parent().id == chain_id:
                        residues_set.add(res)

        return residues_set

    def select_residues_by_distance(self, ligand_coords: bool = True, collection_ligands: Dict = None,
                                    distance_threshold: float = 5.0,min_coord: [List[float]] = None,
                                    max_coord: [List[float]] = None,) -> int:
        """
        Select pocket residues based on distance from ligand atoms.

        Args:
            ligand_coords: List of ligand atom coordinates
            distance_threshold: Distance threshold in Angstroms

        Returns:
            Number of selected residues
        """
        if ligand_coords:
            self.selected_residue_ids = []
            residues_set = set()

            # Get all atoms from the protein
            atoms = [a for a in self.structure[0].get_atoms() if a.element != "H"]

            # Build neighbor search
            ns = NeighborSearch(atoms)

            norm_ligand_coords = collection_ligands['features'][:,-3:]
            coords_denorm = self._denormalize_3d_coord(norm_ligand_coords.detach().numpy(), min_coord, max_coord)
            # Find residues within threshold distance

            chain_priority = ['A', 'B', 'AA']
            residues_set = set()

            for chain_name in chain_priority:
                residues_set = self._find_residues_near_coords(coords_denorm, ns, distance_threshold, chain_id=chain_name)
                if residues_set:  # stop at first non-empty set
                    break


            # Store selected residues
            for res in sorted(residues_set, key=lambda r: r.id[1]):
                res_id = f"{res.get_resname()}_{res.get_id()[1]}"
                self.selected_residue_ids.append(res_id)
                self.residues[res_id] = res

            print(f"Selected {len(self.residues)} residues within {distance_threshold}Å of ligand")
            #return len(self.residues)


    def _filter_interactions(self, entry: dict, distance_threshold: float = 7.0, chain: str = "A") :
        """
        Filters molecular interaction entries based on multiple criteria.

        Args:
            entry: Single interaction entry dictionary
            dist: Maximum distance threshold
            chain: Chain ID to filter

        Returns:
            Filtered row dictionary or None
        """


        accepted_types = {"atom-plane", "plane-plane", "group-plane"}
        amino_acid_types = list(self.AA_3TO1.keys())

        entities = entry.get("interacting_entities", "")
        contact_types = set(entry.get("contact", []))
        interaction_type = entry.get("type", "")
        bgn = entry.get("bgn", {})
        row = None

        # Apply initial filters
        if (entities == "INTER"
                and bgn.get("auth_asym_id", "") == chain
                and entry["end"].get("auth_asym_id") == chain):

            # Check contact and interaction type
           # if contact_types in self.BOND_TYPES_NON_COVALENT or interaction_type in accepted_types:
            if any(c in self.BOND_TYPES_NON_COVALENT for c in contact_types) or interaction_type in accepted_types:
                # Extract relevant information
                bgn_resname = bgn.get("label_comp_id", "")
                end_resname = entry["end"].get("label_comp_id", "")

                # Check if residues are amino acids
                bgn_is_residue = bgn_resname in amino_acid_types
                end_is_residue = end_resname in amino_acid_types

                # Skip if neither is an amino acid residue
                if not (bgn_is_residue or end_is_residue):
                    return None

                # Distance check
                distance = entry.get("distance")
                if distance is None or distance > distance_threshold:
                    return None

                # Build row depending on which side is the residue
                if bgn_is_residue:
                    row = {
                        'ligand_atom': entry["end"].get("auth_atom_id"),
                        'protein_residue': bgn.get("label_comp_id"),
                        'protein_id': bgn.get("auth_seq_id"),
                        'protein_atom': bgn.get("auth_atom_id"),
                        "contact": ";".join(entry.get("contact", [])),
                        "distance": distance,
                        "type": interaction_type
                    }
                else:  # end_is_residue
                    row = {
                        'ligand_atom': bgn.get("auth_atom_id"),
                        'protein_residue': entry["end"].get("label_comp_id"),
                        'protein_id': entry["end"].get("auth_seq_id"),
                        'protein_atom': entry["end"].get("auth_atom_id"),
                        "contact": ";".join(entry.get("contact", [])),
                        "distance": distance,
                        "type": interaction_type
                    }
        self.chain_id = chain
        return row

    def select_residues_from_interactions(self, interaction_json_path: str,
                                          distance_threshold: float = 7.0, chain: str = "A") -> int:
        """
        Select pocket residues from an interaction JSON file.

        Args:
            interaction_json_path: Path to interaction JSON file
            distance_threshold: Maximum distance for interactions (Angstroms)

        Returns:
            Number of selected residues
        """

        if not os.path.exists(interaction_json_path):
            print(f"⚠ Interaction file not found: {interaction_json_path}")

            return 0

        self.selected_residue_ids = []

        try:
            with open(interaction_json_path, 'r') as f:
                interaction_data = json.load(f)
        except Exception as e:
            print(f"Failed to load interaction file: {e}")
            return 0



        if interaction_data is not None:
            for entry in interaction_data:
                row = self._filter_interactions(entry, distance_threshold)
                if row:
                    self.filtered_rows.append(row)

            if len(self.filtered_rows) == 0:
                # Define the chain priorities
                chain_options = ["AAA_2", "AAA", "B"]

                for chain in chain_options:
                    for entry in interaction_data:
                        row = self._filter_interactions(entry, distance_threshold, chain=chain) if chain else self._filter_interactions(entry, distance_threshold)
                        if row:
                            self.filtered_rows.append(row)
                    # Stop checking other chains if we already have results
                    if self.filtered_rows:
                        break


    def build_pocket_graph(self, ligand_coords: bool = True,min_coord: [List[float]] = None,max_coord: [List[float]] = None,
                           include_edge_distances: bool = False,interaction_json_path: [str] = None,
                           distance_threshold: float = 5.0,pocket_by_distance: bool = False,
                           collection_ligands: Dict = None,simplified_edge_distances: bool = False) -> bool:
        """
        Build the pocket graph representation.

        Args:
            ligand_coords: Ligand coordinates for distance-based selection
            min_coord: Minimum coordinates for normalization
            max_coord: Maximum coordinates for normalization
            include_edge_distances: Whether to include distances in edge features
            interaction_json_path: Path to interaction JSON (use instead of distance-based)
            distance_threshold: Distance threshold for selection

        Returns:
            True if successful, False otherwise
        """
        try:
            # Select residues and Extract atoms and build graph
            if pocket_by_distance:
                self.select_residues_by_distance(ligand_coords, collection_ligands, distance_threshold, min_coord, max_coord)
                self._extract_pocket_atoms_by_distance(min_coord, max_coord)
            elif interaction_json_path and os.path.exists(interaction_json_path):
                self.select_residues_from_interactions(interaction_json_path, distance_threshold)
                self._extract_pocket_atoms(min_coord, max_coord)
            else:
                print("Must provide either pocket_by_distance or interaction_json_path")
                return False

            if ligand_coords:
                self.features = [np.concatenate((f, c), axis=-1)
                                 for f, c in zip(self.features, self.coordinates)]
            if self.num_atoms > 0:
                self._build_pocket_edges(include_edge_distances, simplified_edge_distances=simplified_edge_distances)

            return True

        except Exception as e:
            print(f"Error building pocket graph: {e}")
            return False

    def _sort_residues_by_number(self, residue_list: List[str]) -> List[str]:
        """
        Sort residues by residue number.
        Args:
            residue_list: List of residue strings like ['ARG_7', 'PHE_8']
        Returns:
            Sorted residue list
        """
        self.selected_residue_ids = sorted(residue_list, key=lambda x: int(x.split('_')[-1]))


    def _detect_chain(self, model):
        """
        Detects the correct chain in the model that contains the first selected residue.
        Caches the chain ID in self._detected_chain_id for future use.
        Args:
            model: Bio.PDB model object.
        Returns:
            str: detected chain ID
        """

        # Extract residue number from the first selected residue
        if not self.selected_residue_ids:
            raise ValueError("No selected residues available for chain detection.")

        res_num = int(self.selected_residue_ids[0].split('_')[1])

        # Try the default chain first
        try:
            model.child_dict[self.chain_id][(' ', res_num, ' ')]
            return
        except KeyError:
            # Scan all chains for the residue
            for chain_id, chain_obj in model.child_dict.items():
                if (' ', res_num, ' ') in chain_obj:
                    self.chain_id = chain_id
                    return

        # Residue not found in any chain
        raise KeyError(f"Residue {res_num} not found in any chain of the model.")


    def _extract_pocket_atoms(self, min_coord,max_coord):
        """Extract atoms from selected residues."""
        self.features = []
        self.coordinates = []
        self.not_normalized_coordinates = []
        self.atom_nodes = {}
        self.atom_index_map = {}
        self.number_atoms_by_residue = {}
        self.index_H_atoms = []

        # For embedding mode: collect indices
        atom_idx_list = []
        aa_idx_list = []
        hyb_idx_list = []
        cont_list = []

        model = self.structure[0]

        # Derive residue IDs defensively when coming from JSON interactions
        if not self.selected_residue_ids:
            if self.filtered_rows:
                aa_in_interaction_contacts = {
                    f"{r['protein_residue']}_{r['protein_id']}" for r in self.filtered_rows
                }
                self._sort_residues_by_number(list(aa_in_interaction_contacts))
            else:
                raise ValueError(
                    "No residues to extract: both selected_residue_ids and filtered_rows are empty."
                )

        #self._sort_residues_by_number(list(res_ids))
        #aa_in_interaction_contacts = list(self.residues.keys())
        #self._sort_residues_by_number(aa_in_interaction_contacts)

        # Detect the chain that contains the first selected residue
        self._detect_chain(model)
        atom_total_count = 0
        for res_id in self.selected_residue_ids:
            res_name, res_num = res_id.split('_')
            res_num = int(res_num)

            try:
                residue = model[self.chain_id][(' ', res_num, ' ')]
            except KeyError:
                print(f'⚠ Chain {self.chain_id} or residue {res_id} not found.')
                continue

            # Extract non-hydrogen atoms
            atoms_in_residue = 0
            for atom in residue:
                if atom.element != "H":
                    atom_id = self._get_atom_id(atom)
                    self.atom_nodes[atom_id] = atom
                    self.atom_index_map[atom_id] = atom_total_count

                    # Get coordinates
                    coord = atom.get_coord()
                    self.not_normalized_coordinates.append(coord)

                    coord_normalized = self._normalize_3d_coord(coord, min_coord, max_coord)
                    self.coordinates.append(coord_normalized)

                    # Extract features
                    if self.use_embedding_nodes:
                        cont = self._atom_features_for_embedding(atom, res_name)
                       # atom_idx_list.append(atom_idx)
                       # aa_idx_list.append(aa_idx)
                       # hyb_idx_list.append(hyb_idx)
                        cont_list.append(cont)
                    else:
                        feat = self._atom_features_one_hot(atom, res_name)
                        self.features.append(feat)

                    atom_total_count += 1
                    atoms_in_residue += 1

            self.number_atoms_by_residue[res_id] = atoms_in_residue

        self.num_atoms = atom_total_count

        # Convert to tensors if using embeddings
       # if self.use_embedding_nodes and self.num_atoms > 0:
       #     self._convert_embedding_features_to_tensors(atom_idx_list, aa_idx_list, hyb_idx_list, cont_list)
        cont_array = np.array(cont_list)
        self.features = torch.tensor(cont_array, dtype=torch.float)

    def _extract_pocket_atoms_by_distance(self, min_coord,max_coord):
        """Extract atoms from selected residues."""
        self.features = []
        self.coordinates = []
        self.not_normalized_coordinates = []
        self.atom_nodes = {}
        self.atom_index_map = {}
        self.number_atoms_by_residue = {}
        self.index_H_atoms = []

        # For embedding mode: collect indices
        atom_idx_list = []
        aa_idx_list = []
        hyb_idx_list = []
        cont_list = []

        model = self.structure[0]

        aa_in_interaction_contacts = list(self.residues.keys())
        self._sort_residues_by_number(aa_in_interaction_contacts)
        #self._detect_chain(model)

        if self.residues[self.selected_residue_ids[0]].full_id[2] == self.residues[self.selected_residue_ids[-1]].full_id[2]:
            self.chain_id = self.residues[self.selected_residue_ids[0]].full_id[2]

        atom_total_count = 0
        for res_id in self.selected_residue_ids:
            res_name, res_num = res_id.split('_')
            res_num = int(res_num)

            try:
                residue = model[self.chain_id][(' ', res_num, ' ')]
            except KeyError:
                print(f'⚠ Chain {self.chain_id} or residue {res_id} not found.')
                continue

            # Extract non-hydrogen atoms
            atoms_in_residue = 0
            for atom in residue:
                if atom.element != "H":
                    atom_id = self._get_atom_id(atom)
                    self.atom_nodes[atom_id] = atom
                    self.atom_index_map[atom_id] = atom_total_count

                    # Get coordinates
                    coord = atom.get_coord()
                    self.not_normalized_coordinates.append(coord)

                    coord_normalized = self._normalize_3d_coord(coord, min_coord, max_coord)
                    self.coordinates.append(coord_normalized)

                    # Extract features
                    if self.use_embedding_nodes:
                        atom_idx, aa_idx, hyb_idx, cont = self._atom_features_for_embedding(atom, res_name)
                        atom_idx_list.append(atom_idx)
                        aa_idx_list.append(aa_idx)
                        hyb_idx_list.append(hyb_idx)
                        cont_list.append(cont)
                    else:
                        feat = self._atom_features_one_hot(atom, res_name)
                        self.features.append(feat)

                    atom_total_count += 1
                    atoms_in_residue += 1

            self.number_atoms_by_residue[res_id] = atoms_in_residue

        self.num_atoms = atom_total_count

        # Convert to tensors if using embeddings
        if self.use_embedding_nodes and self.num_atoms > 0:
            self._convert_embedding_features_to_tensors(atom_idx_list, aa_idx_list, hyb_idx_list, cont_list)



    def _get_atom_features_from_dict(self, atom_id: str, resname: str) -> Tuple[float, float, str]:
        try:
            atom_data = aminoacids[resname][atom_id]
            degree = atom_data.get("degree", 0.0) / 10.0
            num_h = atom_data.get("num_h", 0.0) / 10.0
            hybrid = atom_data.get("hybrid", "SP3")
        except KeyError:
            # Defaults if not found in dictionary
            degree = 0.0
            num_h = 0.0
            hybrid = "SP3"

        return degree, num_h, hybrid

    def _is_atom_aromatic(self,resname: str) -> int:
        aromatic_residues = {'PHE', 'TYR', 'TRP', 'HIS'}
        return 1 if resname in aromatic_residues else 0

    def _atom_features_one_hot(self, atom, resname: str) -> np.ndarray:
        """Extract one-hot encoded features for an atom."""
        element = atom.element if hasattr(atom, 'element') else 'Unknown'
        element_vec = self._one_of_k_encoding_unk(element, self.PERIODIC_ELEMENTS)

        degree, num_h, hybrid = self._get_atom_features_from_dict(atom.id, resname) if aminoacids else (0.0, 0.0, "SP3")
        degree_list = [degree]
        num_h_list = [num_h]
        is_aromatic = [float(self._is_atom_aromatic(resname))]

        resname_vec = self._one_of_k_encoding_unk(resname, list(self.AA_3TO1.keys()))


        hybridization = self._one_of_k_encoding_unk('SP3', self.HYBRIDIZATION)
        features = element_vec  + resname_vec + hybridization + degree_list + num_h_list + is_aromatic

        return np.array(features)

    def _is_hb(self, resname: str, dictionary: dict, atom_name: str) -> bool:
        """Check if the atom is in the donor/acceptor dictionary for the residue."""
        return atom_name in dictionary.get(resname, [])

    def _atom_features_for_embedding(self, atom, resname: str) -> Tuple:
        """Extract features as indices for embedding."""
        element = atom.element if hasattr(atom, 'element') else 'Unknown'
        atom_idx = self.ATOM_TO_IDX.get(element, self.ATOM_TO_IDX['Unknown'])
        aa_idx = self.AA_TO_IDX.get(resname, self.AA_TO_IDX.get('ALA', 0))

        # Continuous features (placeholders for protein atoms)
        degree, num_h, hybrid = self._get_atom_features_from_dict(atom.id, resname) if aminoacids else (0.0, 0.0, "SP3")
        hyb_idx = self.HYB_TO_IDX.get(hybrid)
        is_donor = float(self._is_hb(resname, HBD_RESIDUE_ATOMS, atom.get_name()))
        is_acceptor = float(self._is_hb(resname, HBA_RESIDUE_ATOMS, atom.get_name()))
       # degree_list = [degree]
       # num_h_list = [num_h]
       # is_aromatic = [float(self._is_atom_aromatic(resname))]
       # cont = np.array([degree_list, num_h_list, is_aromatic], dtype=float)
        aromatic = float(self._is_atom_aromatic(resname))
        cont = np.array([degree, num_h, aromatic], dtype=float)
      #  return atom_idx, aa_idx, hyb_idx, cont
        return [atom_idx, hyb_idx, degree, num_h, aromatic, is_donor, is_acceptor] #[atom_idx, hyb_idx, degree, num_h, aromatic, 0.0, 0.0, is_donor, is_acceptor]



    def _convert_embedding_features_to_tensors(self, atom_idx_list, aa_idx_list, hyb_idx_list, cont_list):
        """Convert extracted embedding features to tensors."""

        self.atom_indices = torch.tensor(atom_idx_list, dtype=torch.long)
        self.aa_indices = torch.tensor(aa_idx_list, dtype=torch.long)
        self.hyb_indices = torch.tensor(hyb_idx_list, dtype=torch.long)
        self.features = torch.tensor(np.array(cont_list), dtype=torch.float)

    def _build_pocket_edges(self, include_edge_distances: bool, cutoff: float = 1.9, simplified_edge_distances: bool = False):
        """Build edges within the pocket (covalent and peptide bonds)."""
        self.edges = []
        self.edge_features = []
        bond_idx_list = []

        # Build KDTree for neighbor search within residues
        # Add edges within residues (covalent bonds)

        start = 0
        end = 0
        atom_list = list(self.atom_index_map.keys())
        for i, res in enumerate(self.selected_residue_ids):
            end += self.number_atoms_by_residue[res]
            res_coords = np.array(self.not_normalized_coordinates[start:end])
            if res_coords.shape[0] > 0:
                # Build KDTree for this residue
                try:
                    tree = cKDTree(res_coords)
                    neighbors = tree.query_ball_tree(tree, cutoff)

                    for j, nbrs in enumerate(neighbors):
                        if len(nbrs) > 1:
                            atom_j_idx = start + j
                            atom_j = self.atom_nodes[atom_list[start]] #self.atom_nodes[atom_list[atom_j_idx
                            for k in nbrs:
                                if k > j:
                                    atom_k_idx = start + k
                                    atom_k = self.atom_nodes[atom_list[atom_k_idx]]
                                    sorted_bond = sorted((atom_j_idx, atom_k_idx))
                                    if sorted_bond not in self.edges:
                                        self.edges.append(sorted_bond)
                                        self.edges.append(sorted((atom_j_idx, atom_k_idx), reverse=True))
                                        if simplified_edge_distances:
                                            #bond_feat = self._one_of_k_encoding_unk("covalent", ["covalent", "non_covalent"])
                                            #self.edge_features.append(bond_feat / np.sum(bond_feat))
                                            try:
                                                new_bond_option_1 = aminoacid_bonds.get(res.split('_')[0], {}).get(
                                                    atom_j.id, {}).get(atom_k.id)
                                                new_bond_option_2 = aminoacid_bonds.get(res.split('_')[0], {}).get(
                                                    atom_k.id, {}).get(atom_j.id)
                                                if new_bond_option_1 not in (None, "SINGLE"):
                                                    new_bond = new_bond_option_1
                                                elif new_bond_option_2 not in (None, "SINGLE"):
                                                    new_bond = new_bond_option_2
                                                else:
                                                    new_bond = "SINGLE"
                                            except (KeyError, AttributeError):
                                                new_bond = "SINGLE"
                                            if new_bond not in list(self.BOND_TYPES.keys()):
                                                print(new_bond)
                                            number = self.BOND_TYPES.get(new_bond, self.BOND_TYPES['OTHER'])
                                            self.edge_features.append(number)
                                            self.edge_features.append(number)
                                        else:
                                            try:
                                                new_bond_option_1 = aminoacid_bonds.get(res.split('_')[0], {}).get(atom_j.id, {}).get(atom_k.id)
                                                new_bond_option_2 = aminoacid_bonds.get(res.split('_')[0], {}).get(atom_k.id,{}).get(atom_j.id)
                                                if new_bond_option_1 is not None:
                                                    if new_bond_option_1!= "SINGLE":
                                                        new_bond = new_bond_option_1
                                                elif new_bond_option_2 is not None:
                                                    if new_bond_option_2!= "SINGLE":
                                                        new_bond = new_bond_option_2
                                                else:
                                                    new_bond = "SINGLE"
                                            except (KeyError, AttributeError):
                                                new_bond = "SINGLE"

                                            if self.use_embedding_edges:
                                                bond_idx_list.append(self.BOND_TO_IDX[new_bond])
                                            else:
                                                bond_feat = self._bond_features_one_hot(new_bond, self.BOND_TYPES_COVALENT+self.BOND_TYPES_NON_COVALENT)
                                                self.edge_features.append(bond_feat / np.sum(bond_feat))
                except Exception as e:
                    print(f"Error building edges for residue {res}: {e}")
            start = end

        # Add peptide bonds between consecutive residues
        beg_res1 = 0
        end_res1 = 0
        for i in range(len(self.selected_residue_ids) - 1):
            res1 = self.selected_residue_ids[i]
            res2 = self.selected_residue_ids[i + 1]
            end_res1 += self.number_atoms_by_residue[res1]
            # Check if residues are consecutive in sequence
            if int(res2.split('_')[1]) == int(res1.split('_')[1]) + 1:
                try:
                    atom_c = False
                    for option in range(self.number_atoms_by_residue[res1]):
                        if self.atom_nodes[atom_list[beg_res1:end_res1][option]].id == 'C':
                            atom_c = True
                            c_idx = beg_res1 + option
                            break
                    atom_n = False
                    for option in range(self.number_atoms_by_residue[res2]):
                        if self.atom_nodes[atom_list[end_res1:end_res1+self.number_atoms_by_residue[res2]][option]].id == 'N':
                            atom_n = True
                            n_idx = end_res1 + option
                            break

                    if atom_c and atom_n:
                        dist = np.linalg.norm(self.not_normalized_coordinates[c_idx] - self.not_normalized_coordinates[n_idx])
                        if 1.2 < dist < 1.45:
                            self.edges.append([c_idx,n_idx])
                            self.edges.append([n_idx, c_idx])
                            if simplified_edge_distances:
                                number = self.BOND_TYPES['OTHER'] #to indicate peptide bond
                                self.edge_features.append(number)
                                self.edge_features.append(number)
                              #  bond_feat = self._one_of_k_encoding_unk("covalent", ["covalent", "non_covalent"])
                              #  self.edge_features.append(bond_feat / np.sum(bond_feat))
                            elif self.use_embedding_edges:
                                bond_idx_list.append(self.BOND_TO_IDX.get("PEPTIDE", 14))
                            else:
                                bond_feat = self._bond_features_one_hot("PEPTIDE", self.BOND_TYPES_COVALENT + self.BOND_TYPES_NON_COVALENT)
                                self.edge_features.append(bond_feat / np.sum(bond_feat))

                except KeyError:
                    pass
            beg_res1 = end_res1

        if self.use_embedding_edges and bond_idx_list:
            self.bond_indices = torch.tensor(bond_idx_list, dtype=torch.long)
            self.edge_features = torch.zeros(len(self.bond_indices), len(self.BOND_TYPES_NON_COVALENT))


        # Add distances to edges if requested
        if include_edge_distances:
            distances = []
            edge_features_updated = []
            for (src_idx, tgt_idx), edge_feat in zip(self.edges, self.edge_features):
                src_coords = self.coordinates[src_idx]
                tgt_coords = self.coordinates[tgt_idx]
                distance = np.linalg.norm(src_coords - tgt_coords)
                updated_feat = np.append(edge_feat, distance)
                edge_features_updated.append(updated_feat)
                distances.append(distance)
            self.edge_distances = torch.tensor(distances, dtype=torch.float).unsqueeze(-1)
            self.edge_features = edge_features_updated

        self.num_edges = len(self.edges)

    def _bond_features_one_hot(self, bond_type: str, bond_type_list = BOND_TYPES_COVALENT) -> np.ndarray:
        """Extract one-hot encoded features for a bond."""
        return np.array(self._one_of_k_encoding_unk(bond_type, bond_type_list))

    @staticmethod
    def _get_atom_id(atom) -> str:
        """Generate unique ID for an atom."""
        res = atom.get_parent()
        return f"{res.get_resname()}_{res.get_id()[1]}_{res.get_parent().id}_{atom.get_id()}"

    @staticmethod
    def _normalize_3d_coord(coord: np.ndarray, min_coord: List[float],
                            max_coord: List[float]) -> np.ndarray:
        """Normalize 3D coordinates."""
        coord = np.array(coord, dtype=float)
        min_coord = np.array(min_coord, dtype=float)
        max_coord = np.array(max_coord, dtype=float)

        range_coord = max_coord - min_coord
        range_coord[range_coord == 0] = 1.0

        return (coord - min_coord) / range_coord

    def _denormalize_3d_coord(self,norm_coord: np.ndarray, min_coord: List[float], max_coord: List[float]) -> np.ndarray:
        """Convert normalized 3D coordinates back to original scale."""
        norm_coord = np.array(norm_coord, dtype=float)
        min_coord = np.array(min_coord, dtype=float)
        max_coord = np.array(max_coord, dtype=float)

        range_coord = max_coord - min_coord
        range_coord[range_coord == 0] = 1.0

        return norm_coord * range_coord + min_coord

    @staticmethod
    def _one_of_k_encoding_unk(x, allowable_set: List) -> List[bool]:
        """One-hot encoding with unknown handling."""
        if x not in allowable_set:
            x = allowable_set[-1]
        return [x == s for s in allowable_set]

    def to_graph_dict(self) -> Dict:
        """Convert pocket to graph dictionary format."""
        graph = {
            'pocket_id': self.pocket_id,
            'num_atoms': self.num_atoms,
            'num_edges': len(self.edges),
            'edges': self.edges,
            'num_residues': len(self.residues),
            'residue_ids': self.selected_residue_ids,
        }

        if self.coordinates:
            graph['coordinates'] = torch.tensor(np.array(self.coordinates), dtype=torch.float)

        if self.use_embedding_nodes:
            graph.update({
                'atom_indices': self.atom_indices,
                'aa_indices': self.aa_indices,
                'hyb_indices': self.hyb_indices,
            })
        else:
            graph['features'] = self.features

        if self.use_embedding_edges:
            graph['bond_indices'] = self.bond_indices
            if self.edge_distances is not None:
                graph['edge_distances'] = self.edge_distances
        else:
            graph['edge_features'] = self.edge_features

        return graph

    def to_graph_to_data(self, pocket_graph, encoder = None):
        """
        Convert ligand graph to Data format.
        """
        if isinstance(pocket_graph['features'], list):
            pocket_graph['features'] = torch.from_numpy(np.array(pocket_graph['features'])).float()
        else:
            pocket_graph['features'] = torch.as_tensor(pocket_graph['features'], dtype=torch.float)

        pocket_graph['y'] = torch.as_tensor([pocket_graph['y']], dtype=torch.float)
        pocket_graph['edges'] = torch.as_tensor(pocket_graph['edges'], dtype=torch.long)


      #  pocket_graph['edge_features'] = torch.stack([torch.as_tensor(x, dtype=torch.float) for x in pocket_graph['edge_features']])

        pocket_graph['edge_features'] = torch.tensor(pocket_graph['edge_features'], dtype=torch.float)

        if not pocket_graph['name']:
            print("No name added to a ligand, check code to fix it")

        features = pocket_graph['features']  # shape: [num_nodes, feat_dim]
        edges = pocket_graph['edges']  # shape: [2, num_edges] or [num_edges, 2]

        num_nodes = features.shape[0]

        min_idx = edges.min().item()
        max_idx = edges.max().item()

        if min_idx < 0 or max_idx >= num_nodes:
            print(f"{pocket_graph['name']}")
            print(f"Invalid edge index detected.")
            print(f"Node count: {num_nodes}")
            print(f"Edge index range: [{min_idx}, {max_idx}]")

        if encoder is not None:
            pocket_graph['features'] = encoder.encode_nodes(pocket_graph['features'])
            if pocket_graph['edge_features'].numel() > 0:
                pocket_graph['edge_features'] = encoder.encode_edges(pocket_graph['edge_features'])

        #return Data(x=pocket_graph['features'], edge_index=pocket_graph['edges'], edge_attr=pocket_graph['edge_features'], name=pocket_graph['name'], y = pocket_graph['y'])
        return Data(x=pocket_graph['features'], edge_index=pocket_graph['edges'],
                    edge_attr=pocket_graph['edge_features'], name=pocket_graph['name'], num_nodes = pocket_graph['num_atoms'])


    def apply_embeddings(self, atom_embedder: Optional[AtomEmbedding] = None,
                         bond_embedder: Optional[BondEmbedding] = None):
        """Apply embedding layers."""
        node_features = None
        edge_features = None

        if self.use_embedding_nodes and atom_embedder is not None:
            node_features = atom_embedder(self.atom_indices, self.aa_indices,self.hyb_indices, self.features)
            self.features = node_features

    #    if self.use_embedding_edges and bond_embedder is not None:
    #        edge_features = bond_embedder(self.bond_indices, self.edge_features)
    #        self.edge_features = edge_features



    def __repr__(self) -> str:
        return (f"Pocket(id='{self.pocket_id}', num_atoms={self.num_atoms}, "
                f"num_residues={len(self.residues)}, num_edges={len(self.edges)})")


class PocketCollection:
    """Manages a collection of Pocket objects."""

    def __init__(self, use_embedding_nodes: bool = False,
                 use_embedding_edges: bool = False, simplified_edge_distances: bool = False):
        """Initialize PocketCollection."""
        self.pockets = {}
        self.graph_data = {}
        self.failed_pockets = []
        self.use_embedding_nodes = use_embedding_nodes
        if simplified_edge_distances:
            use_embedding_edges = False
        self.use_embedding_edges = use_embedding_edges

        # Embedding layers (initialized when needed)
        self.atom_embedder = None
        self.bond_embedder = None


    def add_pocket(self, pocket_id: str, pdb_path: str, chain_id: str = 'A') -> bool:
        """Add a pocket to the collection."""
        try:
            pocket = Pocket(
                pocket_id=pocket_id,
                pdb_path=pdb_path,
                use_embedding_nodes=self.use_embedding_nodes,
                use_embedding_edges=self.use_embedding_edges,
                chain_id=chain_id)
            self.pockets[pocket_id] = pocket
            return pocket
        except Exception as e:
            print(f"Failed to add pocket {pocket_id}: {e}")
            self.failed_pockets.append(pocket_id)
            return False

    def initialize_node_embedders(self, emb_dim_atom: int = 16, emb_dim_aa: int = 8,
                                  emb_dim_hyb: int = 8):
        """Initialize atom embedding layers."""
        if not self.use_embedding_nodes:
            raise ValueError("Cannot initialize embedders when use_embedding_nodes=False")

        self.atom_embedder = AtomEmbedding(
            n_atoms=len(Pocket.PERIODIC_ELEMENTS),
            n_aa=len(Pocket.AA_3TO1),
            n_hyb=len(Pocket.HYBRIDIZATION),
            emb_dim_atom=emb_dim_atom,
            emb_dim_aa=emb_dim_aa,
            emb_dim_hyb=emb_dim_hyb
        )
        print(f"Initialized pocket node embedder: output dim = {self.atom_embedder.output_dim}")

    def initialize_edge_embedders(self, emb_dim_bond: int = 8):
        """Initialize bond embedding layers."""
        if not self.use_embedding_edges:
            raise ValueError("Cannot initialize embedders when use_embedding_edges=False")

        self.bond_embedder = BondEmbedding(
            n_bonds=len(Pocket.BOND_TYPES_COVALENT),
            emb_dim=emb_dim_bond
        )
        print(f"Initialized pocket edge embedder: output dim = {self.bond_embedder.output_dim}")


    def _find_protein_atoms_near_ligand(self,ligand_coords: torch.Tensor,protein_coords: torch.Tensor,threshold: float = 5.0):
        """mask: (N_p,) boolean tensor
            indices: indices of protein atoms in pocket        """

        ligand_coords = ligand_coords.float()
        protein_coords = protein_coords.float()

        dists = torch.cdist(ligand_coords, protein_coords)
        contact_pairs = (dists <= threshold).nonzero(as_tuple=False)

        ligand_idx = contact_pairs[:, 0]
        protein_idx = contact_pairs[:, 1]
        distances = dists[ligand_idx, protein_idx]

        return ligand_idx, protein_idx, distances


    def non_covalent_interactions_by_distance(self, pocket_graph, ligand_graph, threshold = 5.0, ligand_coords: bool = True,include_edge_distances: bool = False):
        """ Add non-covalent interactions between ligand and protein pocket.
            Updated graph representation """

        # Append protein atoms

        if len(ligand_graph['features'][0]) == len(pocket_graph.features[0]):
            ligand_count = len(ligand_graph['atom_names_pdb'])

            if self.use_embedding_nodes:
                ligand_graph['features'] = torch.cat([ligand_graph['features'], pocket_graph.features], dim=0)
            else:
                ligand_graph['features'].extend(pocket_graph.features)

            ligand_graph['num_atoms'] = len(ligand_graph['features'])

            for e in  pocket_graph.edges:
                ligand_graph['edges'].append([e[0] + ligand_count, e[1] + ligand_count])
            ligand_graph['edge_features'].extend(pocket_graph.edge_features)

            new_edges = []
            new_edge_features = []
            mask, indices = self._find_protein_atoms_near_ligand(ligand_graph['features'][0:ligand_count][:, -3:],
                                                 pocket_graph.features[:, -3:], threshold)
            if len(pocket_graph.filtered_rows) > 0:
                for interaction in pocket_graph.filtered_rows:
                    for atom_x in interaction['ligand_atom'].split(','):
                        if atom_x not in ligand_graph['atom_names_pdb']:
                            print(f'{atom_x} not in listed ligand atom names')
                            continue
                        pos_ligand = ligand_graph['atom_names_pdb'].index(atom_x)

                        for atom_y in interaction['protein_atom'].split(','):
                            try:
                                atom_key = f"{interaction['protein_residue']}_{interaction['protein_id']}_{pocket_graph.chain_id}_{atom_y}"
                                raw_pos_protein = pocket_graph.atom_index_map[atom_key]
                            except KeyError:
                                continue

                            pos_protein = raw_pos_protein + ligand_count

                            for contact in interaction['contact'].split(';'):
                                edge_contact = [pos_ligand, pos_protein]
                                edge_contact_features = pocket_graph._bond_features_one_hot(contact, self.BOND_TYPES_NON_COVALENT)

                                #if include_edge_distances and not ligand_coords:
                                #    coord_ligand = length_ligand_coords[pos_ligand]
                                #    coord_protein = graph[1][pos_protein][-3:]
                                #    dist = np.linalg.norm(coord_ligand - coord_protein)
                                #    e_features = np.concatenate([e_features, [dist]])

                                #edge_contact_features = np.array(e_features)

                                if edge_contact not in new_edges:# or (include_edge_distances and not ligand_coords):
                                    new_edges.append(edge_contact)
                                    new_edge_features.append(edge_contact_features)
                                else:
                                    index = new_edges.index(edge_contact)
                                    if not np.all(new_edge_features[index] == edge_contact_features):
                                        updated_edge_features = np.array([e or c for e, c in
                                                                          zip(new_edge_features[index],
                                                                              edge_contact_features)])
                                        new_edge_features[index] = updated_edge_features

                if self.use_embedding_edges and self.bond_embedder is not None:
                    edge_features = self.bond_embedder(pocket_graph.bond_indices, self.edge_features)
                    self.edge_features = edge_features

                ligand_graph['edges'].extend(new_edges)
                ligand_graph['edge_features'].extend(new_edge_features)
                ligand_graph['num_edges'] = len(ligand_graph['edges'])

        else:
            print('stmg')

    def _merge_edges(self, ligand_graph, pocket_graph, ligand_count):
        """
        Merge edges from pocket_graph into ligand_graph, offsetting by ligand_count.

        Returns:
            edges: updated edges (list or tensor)
            was_tensor: True if the original ligand_graph['edges'] was a tensor
        """
        edges = ligand_graph.get('edges', [])
        was_tensor = torch.is_tensor(edges)

        # Convert tensor to list for easier appending
        if was_tensor:
            edges = edges.tolist()

        if isinstance(edges, list) and isinstance(pocket_graph.edges, list):
            if len(edges) == 2 and all(isinstance(e, list) for e in edges):
                for e in pocket_graph.edges:
                    edges[0].append(e[0] + ligand_count)
                    edges[1].append(e[1] + ligand_count)
            else:
                for e in pocket_graph.edges:
                    edges.append([e[0] + ligand_count, e[1] + ligand_count])
        else:
            raise TypeError(f"Unsupported type for ligand_graph['edges'] ({type(edges)}) "
                            f"or pocket_graph.edges ({type(pocket_graph.edges)})")

        # Convert back to tensor if the original was a tensor
       # if was_tensor:
       #     edges = torch.tensor(edges, dtype=torch.long)

        # Update ligand_graph in place
       # ligand_graph['edges'] = edges

        return edges, was_tensor


    def non_covalent_interactions(self, pocket_graph, ligand_graph, ligand_coords: bool = True,include_edge_distances: bool = False, simplified_edge_distances:bool = False):
        """ Add non-covalent interactions between ligand and protein pocket.
            Updated graph representation """

        # Append protein atoms

        if len(ligand_graph['features'][0]) == len(pocket_graph.features[0]):
            ligand_count = len(ligand_graph['atom_names_pdb'])

          #  if self.use_embedding_nodes:
          #      ligand_graph['features'] = torch.cat([ligand_graph['features'], pocket_graph.features], dim=0)
          #  else:
            ligand_graph['features'].extend(pocket_graph.features)

            ligand_graph['num_atoms'] = len(ligand_graph['features'])

            edges, was_tensor = self._merge_edges(ligand_graph, pocket_graph, ligand_count)

            new_edges = []
            new_edge_features = []
            if len(pocket_graph.filtered_rows) > 0:
                for interaction in pocket_graph.filtered_rows:
                    for atom_x in interaction['ligand_atom'].split(','):
                        if atom_x not in ligand_graph['atom_names_pdb']:
                            print(f'{atom_x} not in listed ligand atom names')
                            continue
                        pos_ligand = ligand_graph['atom_names_pdb'].index(atom_x)

                        for atom_y in interaction['protein_atom'].split(','):
                            try:
                                atom_key = f"{interaction['protein_residue']}_{interaction['protein_id']}_{pocket_graph.chain_id}_{atom_y}"
                                raw_pos_protein = pocket_graph.atom_index_map[atom_key]
                            except KeyError:
                                continue

                            pos_protein = raw_pos_protein + ligand_count

                            for contact in interaction['contact'].split(';'):
                                print(contact)
                                edge_contact = [pos_ligand, pos_protein]
                                if simplified_edge_distances:
                                    #edge_contact_features = pocket_graph._one_of_k_encoding_unk("non_covalent", ["covalent", "non_covalent"])
                                    edge_contact_features = pocket_graph.BOND_TYPES['NON_COVALENT']
                                else:
                                    if not self.use_embedding_edges:
                                        edge_contact_features = pocket_graph._bond_features_one_hot(contact,pocket_graph.BOND_TYPES_COVALENT + pocket_graph.BOND_TYPES_NON_COVALENT)
                                    else:
                                        edge_contact_features = pocket_graph._bond_features_one_hot(contact, pocket_graph.BOND_TYPES_NON_COVALENT)
                                if include_edge_distances:
                                    coord_ligand = ligand_graph['coordinates'][pos_ligand]
                                    coord_protein = pocket_graph.coordinates[raw_pos_protein]
                                    dist = np.linalg.norm(coord_ligand - coord_protein)
                                    if simplified_edge_distances:
                                        edge_contact_features = np.array([edge_contact_features, dist])
                                    else:
                                        e_features = np.concatenate([edge_contact_features, [dist]])
                                        edge_contact_features = np.array(e_features)

                                if edge_contact not in new_edges:# or (include_edge_distances and not ligand_coords):
                                    new_edges.append(edge_contact)
                                    new_edges.append([pos_protein, pos_ligand])
                                    new_edge_features.append(edge_contact_features)
                                    new_edge_features.append(edge_contact_features)
                                    #if pocket_graph.bond_indices:
                                    try:
                                        pocket_graph.bond_indices = torch.cat([pocket_graph.bond_indices,torch.tensor([0],dtype=pocket_graph.bond_indices.dtype)],dim=0 )
                                    except AttributeError:
                                        pass
                                else:
                                    if not simplified_edge_distances:
                                        index = new_edges.index(edge_contact)
                                        if not np.all(new_edge_features[index] == edge_contact_features):
                                            updated_edge_features = np.array([e or c for e, c in
                                                                          zip(new_edge_features[index],
                                                                              edge_contact_features)])
                                            new_edge_features[index] = updated_edge_features
                if not include_edge_distances:
                    try:
                        try:
                            new_edge_features = torch.tensor(np.stack(new_edge_features), dtype=pocket_graph.edge_features.dtype,
                                         device=pocket_graph.edge_features.device)
                        except AttributeError:
                            new_edge_features = torch.tensor(np.stack(new_edge_features),dtype=torch.float32)
                        if not isinstance(pocket_graph.edge_features, torch.Tensor) and isinstance(new_edge_features, torch.Tensor):
                            pocket_graph.edge_features = torch.cat([torch.tensor(pocket_graph.edge_features, dtype=torch.float32), new_edge_features], dim=0)
                        else:
                            pocket_graph.edge_features = torch.cat([pocket_graph.edge_features, new_edge_features], dim=0)
                    except ValueError as e:
                        print("Stack failed: shape mismatch")
                        print(e)
                elif include_edge_distances:
                    pocket_graph.edge_features.extend(new_edge_features)

                if self.use_embedding_edges and self.bond_embedder is not None:
                    edge_features = self.bond_embedder(pocket_graph.bond_indices, pocket_graph.edge_features)
                    pocket_graph.edge_features = edge_features

                if not isinstance(ligand_graph['edge_features'], torch.Tensor):
                    if isinstance(pocket_graph.edge_features, torch.Tensor):
                        ligand_graph['edge_features'] = torch.cat([torch.tensor(ligand_graph['edge_features'], dtype=torch.float32), pocket_graph.edge_features],dim=-0)
                    elif  not isinstance(pocket_graph.edge_features, torch.Tensor):
                        ligand_graph['edge_features'] = torch.cat([torch.tensor(ligand_graph['edge_features'], dtype=torch.float32), torch.tensor(pocket_graph.edge_features, dtype=torch.float32)], dim=-0)
                else:
                     if isinstance(pocket_graph.edge_features, list):
                         pocket_graph.edge_features = torch.from_numpy(np.array(pocket_graph.edge_features)).float()
                     ligand_graph['edge_features'] = torch.cat([ligand_graph['edge_features'], pocket_graph.edge_features],dim=-0)
                if len(edges) == 2:
                    for e in new_edges:
                        edges[0].append(e[0])
                        edges[1].append(e[1])
                else:
                    for e in new_edges:
                        edges.append([e[0], e[1]])
                if was_tensor:
                    edges = torch.tensor(edges, dtype=torch.long)

                ligand_graph['edges'] = edges
               # ligand_graph['edge_features'].extend(new_edge_features)
                ligand_graph['num_edges'] = len(ligand_graph['edges'])

        else:
            print('The number of ligand features is not the same as the ones in the pocket graph')



    def build_all_pockets(self, collection_ligands, ligand_coords: bool = True,min_coord: [List[float]] = None,
                          max_coord: [List[float]] = None,affinity_data=[],include_edge_distances: bool = False,
                          interaction_json_dir: [str] = None,distance_threshold: float = 5.0,
                          pocket_by_distance: bool = False, simplified_edge_distances: bool = False) -> int:

        """Build graph representations for all pockets."""
        success_count = 0
        failed = []
        data_list = []

        encoder = None
        #encoder = EmbeddingEncoder(atom_emb_dim=12, hybrid_emb_dim=3, bond_emb_dim=2)

        for ligand_id in list(collection_ligands.keys()):
            interaction_json = os.path.join(interaction_json_dir, f"{ligand_id}_ligand.json")
            try:
                pdb_path = os.path.join(os.path.dirname(interaction_json_dir),  "Protein", "Protein_PDB",f'{ligand_id}_protein.pdb')
                pocket = self.add_pocket(ligand_id, pdb_path)
                if pocket is not None:
                    success = pocket.build_pocket_graph(ligand_coords=ligand_coords,min_coord=min_coord, max_coord=max_coord,
                        include_edge_distances=include_edge_distances,interaction_json_path=interaction_json, distance_threshold=distance_threshold,
                           pocket_by_distance=pocket_by_distance, collection_ligands=collection_ligands[ligand_id],simplified_edge_distances=simplified_edge_distances)

                    if success:
                      #  if self.use_embedding_nodes:
                      #      pocket.apply_embeddings(atom_embedder=self.atom_embedder,bond_embedder=self.bond_embedder)
                        if pocket_by_distance:
                            self.non_covalent_interactions_by_distance(pocket, collection_ligands[ligand_id], threshold= distance_threshold, ligand_coords = ligand_coords,include_edge_distances= include_edge_distances)
                        else:
                            self.non_covalent_interactions(pocket, collection_ligands[ligand_id], ligand_coords = ligand_coords,include_edge_distances= include_edge_distances,simplified_edge_distances=simplified_edge_distances)

                       # self.graph_data[ligand_id] = pocket.to_graph_dict()
                        success_count += 1
                        collection_ligands[ligand_id]['y'] = torch.tensor([affinity_data.loc[affinity_data[0] == ligand_id, 1].iloc[0]],
                                            dtype=torch.float)
                        data_list.append(pocket.to_graph_to_data(collection_ligands[ligand_id], encoder=encoder))
                    else:
                        failed.append(ligand_id)

            except Exception as e:
                print(f"Failed to build pocket {ligand_id}: {e}")
                failed.append(ligand_id)

        print(f"Built {success_count}/{len(self.pockets)} pocket graphs")
        if failed:
            print(f"Failed pockets: {failed}")

        return collection_ligands, data_list



    def get_pocket(self, pocket_id: str) -> Optional[Pocket]:
        """Get a specific pocket by ID."""
        return self.pockets.get(pocket_id)

    def get_graph(self, pocket_id: str) -> Optional[Dict]:
        """Get graph data for a specific pocket."""
        return self.graph_data.get(pocket_id)

    def export_graph_data(self, output_path: str):
        """Export all graph data to a pickle file."""
        import pickle
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(self.graph_data, f)
        print(f"Exported {len(self.graph_data)} pocket graphs to {output_path}")

    def __len__(self) -> int:
        return len(self.pockets)

    def __repr__(self) -> str:
        return f"PocketCollection(n_pockets={len(self.pockets)}, n_graphs={len(self.graph_data)})"

if __name__ == "__main__":
    min_coord = [-28, -36, -34]
    max_coord = [39, 37, 42]

    DATA_PATH = r"C:\Users\natal\code_binding_data_and_results\data\Mpro-URV"  # Path for MPro data
    results_folder = r"C:\Users\natal\code_binding_data_and_results\results\Mpro-URV"
    ligand_sdf_directory = os.path.join(DATA_PATH, "Ligand", "Ligand_SDF")
    exported_ligand_path = os.path.join(results_folder, "ligand_graphs.pkl")
    split_data_folder = "split_data"

    use_embedding_nodes = True
    use_embedding_edges = False
    include_edge_distances = True
    interaction_json_dir = os.path.join(DATA_PATH, "Interaction")
    pocket_by_distance = False #PL interactions by distance not json
    include_coords = True
    protein_interaction = True

    exported_pocket_path = os.path.join(results_folder, "protein_ligand_graphs.pkl")


    collection_ligands = LigandCollection()
   ## collection_ligands.load_from_directory(
   #     sdf_directory=r"C:\Users\natal\code_binding_data_and_results\data\Mpro-URV\Ligand\Ligand_SDF",
   #      node_feature_selection=0, use_embedding_nodes=use_embedding_nodes, use_embedding_edges=use_embedding_edges)
   # collection_ligands.build_all_graphs(include_coords=True, min_coord=min_coord, max_coord=max_coord,
   #                                     include_edge_distances=True)
    exported_ligand_path = os.path.join(results_folder, "ligand_graphs_embed.pkl")
   # collection_ligands.export_graph_data(exported_ligand_path)
    collection_ligands = collection_ligands.load_graph_data(exported_ligand_path)


    pockets_interaction = PocketCollection(use_embedding_nodes=use_embedding_nodes,use_embedding_edges=use_embedding_edges)

    # Initialize embedders
    if use_embedding_nodes:
        pockets_interaction.initialize_node_embedders(emb_dim_atom=16)

    if use_embedding_edges:
        pockets_interaction.initialize_edge_embedders(emb_dim_bond= 8)

    # Build graphs (and apply embeddings internally)

    ligand_protein_collection = pockets_interaction.build_all_pockets(collection_ligands=collection_ligands,ligand_coords=include_coords,min_coord=min_coord, max_coord=max_coord, include_edge_distances=include_edge_distances,
                                          interaction_json_dir=interaction_json_dir,distance_threshold=7.0, pocket_by_distance=pocket_by_distance)


    export_dict_graph_data(ligand_protein_collection,exported_pocket_path)

    ligand_protein_collection = load_graph(exported_pocket_path)
