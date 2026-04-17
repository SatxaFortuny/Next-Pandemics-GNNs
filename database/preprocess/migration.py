"""
Migration script: file-based MPro-URV dataset → PostgreSQL

Tables created:
    1. mpro_dataset          – one row per PDB entry (metadata + paths)
    2. fold_assignments      – k-fold split membership (pdb_id, fold, role)
    3. interactions_pl       – protein-ligand non-covalent interactions from JSONs
    4. ligand_atoms          – per-atom data extracted from ligand CIF files
    5. pocket_residues       – residues selected for each pocket (from interaction JSON)
    6. pocket_atoms          – per-atom data for each pocket residue (from PDB files)
    7. protein_secondary_structure – per-residue DSSP secondary structure data
"""

import os
import ast
import glob
import json
import gzip
import re

import polars as pl
from Bio.PDB import PDBParser, DSSP

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "MPro-URV")
URI_DB = "postgresql://admin:admin@127.0.0.1:5432/db_preproc"

# Amino acids recognised by the GNN code
AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E",
    "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}

# Non-covalent contact types recognised by the GNN code
BOND_TYPES_NON_COVALENT = {
    "hbond", "weak_hbond", "xbond", "ionic", "metal", "aromatic",
    "hydrophobic", "carbonyl", "polar", "weak_polar",
    "CARBONPI", "CATIONPI", "METSULPHURPI", "EF", "FT",
}

ACCEPTED_INTERACTION_TYPES = {"atom-plane", "plane-plane", "group-plane"}

# Chain priority order used by the GNN code when searching for pocket residues
CHAIN_PRIORITY = ["A", "B", "AA", "AAA_2", "AAA"]


def load_splits(data_path: str) -> pl.DataFrame:
    """
    Read train/valid/test split files and return a long-format DataFrame with
    columns (pdb_id, fold_index, split_role).
    """
    split_paths = {
        "train": os.path.join(data_path, "Splits", "train_index_folder.txt"),
        "valid": os.path.join(data_path, "Splits", "valid_index_folder.txt"),
        "test":  os.path.join(data_path, "Splits", "test_index_folder.txt"),
    }

    rows = []
    for role, path in split_paths.items():
        if not os.path.exists(path):
            print(f"  [!] Split file not found, skipping: {path}")
            continue

        with open(path) as fh:
            content = fh.read().strip()

        try:
            parsed = ast.literal_eval(content)
        except Exception:
            parsed = [line.strip() for line in content.splitlines() if line.strip()]

        if parsed and not isinstance(parsed[0], list):
            parsed = [parsed]

        for fold_idx, fold_ids in enumerate(parsed):
            for pdb_id in fold_ids:
                rows.append({
                    "pdb_id":     str(pdb_id).strip(),
                    "fold_index": fold_idx,
                    "split_role": role,
                })

    if not rows:
        return pl.DataFrame(
            {"pdb_id": [], "fold_index": [], "split_role": []},
            schema={"pdb_id": pl.Utf8, "fold_index": pl.Int32, "split_role": pl.Utf8},
        )

    return pl.DataFrame(rows).with_columns(pl.col("fold_index").cast(pl.Int32))


# ---------------------------------------------------------------------------
# Table 1 – mpro_dataset
# ---------------------------------------------------------------------------

def migrate_dataset(data_path: str, uri: str) -> None:
    print("[*] Reading Info.csv ...")
    df = pl.read_csv(os.path.join(data_path, "Info.csv"), separator=";")
    df = df.rename({c: c.lower().replace(" ", "_") for c in df.columns})

    if "pdb_id" not in df.columns:
        for candidate in ("pdb_id", "PDB_ID", "pdbid"):
            if candidate in df.columns:
                df = df.rename({candidate: "pdb_id"})
                break

    print("[*] Generating file paths ...")
    df = df.with_columns([
        pl.concat_str([pl.lit(f"{data_path}/Complex/ALIGNED/"),
                       pl.col("pdb_id"), pl.lit(".cif")]).alias("path_complex_aligned"),
        pl.concat_str([pl.lit(f"{data_path}/Complex/CIF_FROM_PDB_NOT_ALIGNED/"),
                       pl.col("pdb_id"), pl.lit(".cif.gz")]).alias("path_complex_cif_gz"),
        pl.concat_str([pl.lit(f"{data_path}/Complex/PROTONATED_NOT_ALIGNED/"),
                       pl.col("pdb_id"), pl.lit(".cif")]).alias("path_complex_protonated"),
        pl.concat_str([pl.lit(f"{data_path}/Interaction/"),
                       pl.col("pdb_id"), pl.lit("_ligand.json")]).alias("path_interaction_json"),
        pl.concat_str([pl.lit(f"{data_path}/Ligand/Ligand_CIF/"),
                       pl.col("pdb_id"), pl.lit("_ligand.cif")]).alias("path_ligand_cif"),
        pl.concat_str([pl.lit(f"{data_path}/Ligand/Ligand_SDF/"),
                       pl.col("pdb_id"), pl.lit("_ligand.sdf")]).alias("path_ligand_sdf"),
        pl.concat_str([pl.lit(f"{data_path}/Ligand/Ligand_SMI/"),
                       pl.col("pdb_id"), pl.lit("_"), pl.col("ligand"),
                       pl.lit(".smi")]).alias("path_ligand_smi"),
        pl.concat_str([pl.lit(f"{data_path}/Protein/Protein_CIF/"),
                       pl.col("pdb_id"), pl.lit("_protein.cif")]).alias("path_protein_cif"),
        pl.concat_str([pl.lit(f"{data_path}/Protein/Protein_DSSP/"),
                       pl.col("pdb_id"), pl.lit("_protein.dssp")]).alias("path_protein_dssp"),
        pl.concat_str([pl.lit(f"{data_path}/Protein/Protein_PDB/"),
                       pl.col("pdb_id"), pl.lit("_protein.pdb")]).alias("path_protein_pdb"),
    ])

    print(f"[*] Writing mpro_dataset ({df.height} rows, {df.width} cols) ...")
    df.write_database(
        table_name="mpro_dataset",
        connection=uri,
        if_table_exists="replace",
        engine="adbc",
    )
    print("[V] mpro_dataset done.\n")


# ---------------------------------------------------------------------------
# Table 2 – fold_assignments
# ---------------------------------------------------------------------------

def migrate_fold_assignments(data_path: str, uri: str) -> None:
    print("[*] Loading split files ...")
    df = load_splits(data_path)

    if df.height == 0:
        print("  [!] No split data found – skipping fold_assignments.\n")
        return

    num_folds = df["fold_index"].n_unique()
    print(f"[*] Writing fold_assignments ({df.height} rows, {num_folds} fold(s)) ...")
    df.write_database(
        table_name="fold_assignments",
        connection=uri,
        if_table_exists="replace",
        engine="adbc",
    )
    print("[V] fold_assignments done.\n")


# ---------------------------------------------------------------------------
# Table 3 – interactions_pl
# ---------------------------------------------------------------------------

def _parse_universal_interactions(pdb_id: str, path: str) -> list[dict]:
    with open(path) as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError:
            return []

    found = []
    for entry in data:
        bgn = entry.get("bgn", {})
        end = entry.get("end", {})
        contact_types = entry.get("contact", [])

        found.append({
            "pdb_id": pdb_id,
            
            # Metadades de la interacció
            "interaction_class": entry.get("interacting_entities"), # Ex: "INTRA_NON_SELECTION" o "INTER"
            "interaction_type":  entry.get("type"),                 # Ex: "atom-atom"
            "contact_types":     ";".join(sorted(contact_types)),   # Ex: "proximal"
            "distance":          float(entry.get("distance", 0)),

            # Dades de l'element 1 (bgn / origen)
            "bgn_chain":         bgn.get("auth_asym_id"),
            "bgn_residue":       bgn.get("label_comp_id"),
            "bgn_res_id":        bgn.get("auth_seq_id"),
            "bgn_atom":          bgn.get("auth_atom_id"),
            "bgn_type":          bgn.get("label_comp_type"),        # Ens dirà si és "P" (Proteïna), etc.

            # Dades de l'element 2 (end / destí)
            "end_chain":         end.get("auth_asym_id"),
            "end_residue":       end.get("label_comp_id"),
            "end_res_id":        end.get("auth_seq_id"),
            "end_atom":          end.get("auth_atom_id"),
            "end_type":          end.get("label_comp_type"),
        })

    return found

def migrate_interactions(data_path: str, uri: str) -> None:
    json_files = sorted(glob.glob(os.path.join(data_path, "Interaction", "*.json")))
    print(f"[*] Parsing {len(json_files)} interaction JSON files ...")

    all_rows: list[dict] = []
    for path in json_files:
        pdb_id = os.path.basename(path).split("_")[0]
        all_rows.extend(_parse_universal_interactions(pdb_id, path))

    if not all_rows:
        print("  [!] No interaction rows found – skipping.\n")
        return

    df = pl.DataFrame(all_rows).with_columns([
        pl.col("bgn_res_id").cast(pl.Utf8).cast(pl.Int32, strict=False),
        pl.col("end_res_id").cast(pl.Utf8).cast(pl.Int32, strict=False),
        pl.col("distance").cast(pl.Float32),
    ])

    print(f"[*] Writing interactions_all ({df.height} rows) ...")
    df.write_database(
        table_name="interactions_all",
        connection=uri,
        if_table_exists="replace",
        engine="adbc",
    )
    print("[V] interactions_all done.\n")


# ---------------------------------------------------------------------------
# Table 4 – ligand_atoms  (from Ligand_CIF files)
# ---------------------------------------------------------------------------

def _parse_ligand_cif(pdb_id: str, cif_path: str) -> list[dict]:
    """
    Extract heavy-atom names and 3-D coordinates from a ligand CIF file.
    Mirrors Ligand._extract_atom_names_from_cif() and
    Ligand._get_atom_names_and_coords_from_cif().
    """
    rows = []
    seen_names: set[str] = set()

    with open(cif_path) as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith(("HETATM", "ATOM")):
                continue

            fields = line.split()
            if len(fields) < 10:
                continue

            atom_name = fields[3]

            # Skip hydrogens and alternate conformations (field 4 != '.' or 'A')
            if len(fields) > 4 and fields[4] not in (".", "A"):
                continue
            if atom_name in seen_names:
                continue

            seen_names.add(atom_name)

            # Coordinates are in the last columns; positions vary by CIF writer.
            # The GNN code uses split_line[-6], [-5], [-4] for x, y, z.
            try:
                x = float(fields[-6])
                y = float(fields[-5])
                z = float(fields[-4])
            except (ValueError, IndexError):
                x = y = z = None

            element = re.sub(r"[^A-Za-z]", "", atom_name)[:2]

            rows.append({
                "pdb_id":    pdb_id,
                "atom_name": atom_name,
                "element":   element,
                "x":         x,
                "y":         y,
                "z":         z,
            })

    return rows

def migrate_ligand_atoms(data_path: str, uri: str) -> None:
    cif_files = sorted(glob.glob(
        os.path.join(data_path, "Ligand", "Ligand_CIF", "*.cif")
    ))
    print(f"[*] Parsing {len(cif_files)} ligand CIF files ...")

    all_rows: list[dict] = []
    for path in cif_files:
        pdb_id = os.path.basename(path).split("_")[0]
        all_rows.extend(_parse_ligand_cif(pdb_id, path))

    if not all_rows:
        print("  [!] No ligand atom rows found – skipping.\n")
        return

    df = pl.DataFrame(all_rows).with_columns([
        pl.col("x").cast(pl.Float32),
        pl.col("y").cast(pl.Float32),
        pl.col("z").cast(pl.Float32),
    ])

    print(f"[*] Writing ligand_atoms ({df.height} rows) ...")
    df.write_database(
        table_name="ligand_atoms",
        connection=uri,
        if_table_exists="replace",
        engine="adbc",
    )
    print("[V] ligand_atoms done.\n")



# ---------------------------------------------------------------------------
#  ligand_bonds   (from Ligand_CIF files)
# ---------------------------------------------------------------------------

def _parse_ligand_sdf(pdb_id: str, sdf_path: str) -> list[dict]:
    """
    Extracts topology from a SDF filr with format V2000.
    """
    rows = []
    
    with open(sdf_path) as fh:
        lines = fh.readlines()
        
    if len(lines) < 4:
        return rows

    counts_line = lines[3]
    try:
        num_atoms = int(counts_line[0:3].strip())
        num_bonds = int(counts_line[3:6].strip())
    except ValueError:
        return rows

    bond_start_idx = 4 + num_atoms
    bond_end_idx = bond_start_idx + num_bonds

    for i in range(bond_start_idx, min(bond_end_idx, len(lines))):
        line = lines[i]
        if len(line) < 9:
            continue
            
        try:
            atom_a = int(line[0:3].strip())
            atom_b = int(line[3:6].strip())
            bond_order = int(line[6:9].strip())
            
            rows.append({
                "pdb_id": pdb_id,
                "atom_a_idx": atom_a,  
                "atom_b_idx": atom_b,  
                "bond_order": bond_order
            })
        except ValueError:
            continue

    return rows

def migrate_ligand_bonds(data_path: str, uri: str) -> None:
    sdf_files = sorted(glob.glob(
        os.path.join(data_path, "Ligand", "Ligand_SDF", "*.sdf")
    ))
    print(f"[*] Parsing {len(sdf_files)} ligand SDF files for bonds ...")

    all_rows: list[dict] = []
    for path in sdf_files:
        pdb_id = os.path.basename(path).split("_")[0]
        all_rows.extend(_parse_ligand_sdf(pdb_id, path))

    if not all_rows:
        print("  [!] No ligand bond rows found – skipping.\n")
        return

    df = pl.DataFrame(all_rows).with_columns([
        pl.col("atom_a_idx").cast(pl.Int32),
        pl.col("atom_b_idx").cast(pl.Int32),
        pl.col("bond_order").cast(pl.Int32),
    ])

    print(f"[*] Writing ligand_bonds ({df.height} rows) ...")
    df.write_database(
        table_name="ligand_bonds",
        connection=uri,
        if_table_exists="replace",
        engine="adbc",
    )
    print("[V] ligand_bonds done.\n")

# ---------------------------------------------------------------------------
# Taules 5 & 6 – protein_residues + protein_atoms (TOTA la proteïna)
# ---------------------------------------------------------------------------

def _parse_full_protein_pdb(pdb_id: str, pdb_path: str) -> tuple[list[dict], list[dict]]:
    """
    Reads the all the PDB file without filters
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, pdb_path)
    except Exception as exc:
        print(f"  [!] Cannot parse PDB {pdb_id}: {exc}")
        return [], []

    model = structure[0]

    residue_rows: list[dict] = []
    atom_rows: list[dict] = []

    for chain in model:
        chain_id = chain.id
        
        for residue in chain:
            
            res_num = residue.get_id()[1]
            res_name = residue.resname

            if residue.get_id()[0].strip() != "":
                continue

            residue_rows.append({
                "pdb_id":       pdb_id,
                "chain_id":     chain_id,
                "residue_name": res_name,
                "residue_num":  int(res_num),
            })

            for atom in residue.get_atoms():
                coord = atom.get_vector().get_array()
                
                atom_rows.append({
                    "pdb_id":       pdb_id,
                    "chain_id":     chain_id,
                    "residue_name": res_name,
                    "residue_num":  int(res_num),
                    "atom_name":    atom.id,
                    "element":      atom.element,
                    "x":            float(coord[0]),
                    "y":            float(coord[1]),
                    "z":            float(coord[2]),
                })

    return residue_rows, atom_rows


def migrate_full_protein_data(data_path: str, uri: str) -> None:
    """Creates protein_residues and protein_atoms."""
    
    pdb_files = sorted(glob.glob(os.path.join(data_path, "Protein", "Protein_PDB", "*.pdb")))
    print(f"[*] Extracting full protein data for {len(pdb_files)} structures ...")

    all_residue_rows: list[dict] = []
    all_atom_rows: list[dict] = []

    for pdb_path in pdb_files:
        pdb_id = os.path.basename(pdb_path).split("_")[0]
        
        res_rows, atm_rows = _parse_full_protein_pdb(pdb_id, pdb_path)
        all_residue_rows.extend(res_rows)
        all_atom_rows.extend(atm_rows)

    if all_residue_rows:
        df_res = pl.DataFrame(all_residue_rows).with_columns(
            pl.col("residue_num").cast(pl.Int32)
        )
        print(f"[*] Writing protein_residues ({df_res.height} rows) ...")
        df_res.write_database(
            table_name="protein_residues",
            connection=uri,
            if_table_exists="replace",
            engine="adbc",
        )
        print("[V] protein_residues done.")

    if all_atom_rows:
        df_atm = pl.DataFrame(all_atom_rows).with_columns([
            pl.col("residue_num").cast(pl.Int32),
            pl.col("x").cast(pl.Float32),
            pl.col("y").cast(pl.Float32),
            pl.col("z").cast(pl.Float32),
        ])
        print(f"[*] Writing protein_atoms ({df_atm.height} rows) ...")
        df_atm.write_database(
            table_name="protein_atoms",
            connection=uri,
            if_table_exists="replace",
            engine="adbc",
        )
        print("[V] protein_atoms done.\n")

# ---------------------------------------------------------------------------
# Table 7 – protein_secondary_structure  (from DSSP files)
# ---------------------------------------------------------------------------

# DSSP column layout (fixed-width):
# cols  0- 4 : residue sequential number
# cols  5- 5 : chain break indicator
# cols  6- 9 : residue number (PDB)
# col  10    : insertion code
# col  11    : chain ID
# col  13    : one-letter amino-acid code
# col  16    : secondary structure assignment
# cols 34-37 : accessible surface area (ASA)
# cols 85-90 : phi angle
# cols 91-97 : psi angle

def _parse_dssp_file(pdb_id: str, dssp_path: str) -> list[dict]:
    rows = []
    in_data = False

    with open(dssp_path, errors="replace") as fh:
        for line in fh:
            if line.startswith("  #"):
                in_data = True
                continue
            if not in_data:
                continue
            if len(line) < 38:
                continue

            chain_id  = line[11].strip()
            aa_code   = line[13].strip()
            ss        = line[16].strip() or "C"   # blank = coil

            # Skip chain-break markers
            if aa_code == "!":
                continue

            try:
                res_num = int(line[6:10].strip())
            except ValueError:
                continue

            try:
                asa = float(line[34:38].strip())
            except ValueError:
                asa = None

            try:
                phi = float(line[103:109].strip())
            except (ValueError, IndexError):
                phi = None

            try:
                psi = float(line[109:115].strip())
            except (ValueError, IndexError):
                psi = None

            rows.append({
                "pdb_id":      pdb_id,
                "chain_id":    chain_id,
                "residue_num": res_num,
                "aa_code":     aa_code,
                "ss_type":     ss,
                "asa":         asa,
                "phi":         phi,
                "psi":         psi,
            })

    return rows


def migrate_secondary_structure(data_path: str, uri: str) -> None:
    dssp_files = sorted(glob.glob(
        os.path.join(data_path, "Protein", "Protein_DSSP", "*.dssp")
    ))
    print(f"[*] Parsing {len(dssp_files)} DSSP files ...")

    all_rows: list[dict] = []
    for path in dssp_files:
        pdb_id = os.path.basename(path).split("_")[0]
        all_rows.extend(_parse_dssp_file(pdb_id, path))

    if not all_rows:
        print("  [!] No DSSP rows found – skipping.\n")
        return

    df = pl.DataFrame(all_rows).with_columns([
        pl.col("residue_num").cast(pl.Int32),
        pl.col("asa").cast(pl.Float32),
        pl.col("phi").cast(pl.Float32),
        pl.col("psi").cast(pl.Float32),
    ])

    print(f"[*] Writing protein_secondary_structure ({df.height} rows) ...")
    df.write_database(
        table_name="protein_secondary_structure",
        connection=uri,
        if_table_exists="replace",
        engine="adbc",
    )
    print("[V] protein_secondary_structure done.\n")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("MPro-URV → PostgreSQL Data Lake migration")
    print("=" * 60)


    migrate_dataset(DATA_PATH, URI_DB)
    migrate_fold_assignments(DATA_PATH, URI_DB)
    
    migrate_interactions(DATA_PATH, URI_DB)
    
    migrate_ligand_atoms(DATA_PATH, URI_DB)
    migrate_ligand_bonds(DATA_PATH, URI_DB)
    
    migrate_full_protein_data(DATA_PATH, URI_DB)
    migrate_secondary_structure(DATA_PATH, URI_DB)

    print("=" * 60)
    print("[V] All tables migrated successfully to the Data Lake.")
    print("=" * 60)