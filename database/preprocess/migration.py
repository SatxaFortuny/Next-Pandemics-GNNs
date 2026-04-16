import polars as pl
import os
import ast
import json
import glob

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(SCRIPT_DIR, "MPro-URV", "MPro-URV_Version2", "MPro-URV_Version2")

URI_DB = "postgresql://admin:admin@127.0.0.1:5432/db_preproc"

def load_splits(data_path):
    splits_paths = {
        "train": os.path.join(data_path, "Splits", "train_index_folder.txt"),
        "valid": os.path.join(data_path, "Splits", "valid_index_folder.txt"),
        "test": os.path.join(data_path, "Splits", "test_index_folder.txt")
    }
    
    dfs_splits = []
    for split_name, path in splits_paths.items():
        if os.path.exists(path):
            with open(path, 'r') as f:
                content = f.read()
                try:
                    trash_list = ast.literal_eval(content)
                except:
                    trash_list = [line.strip() for line in content.split('\n') if line.strip()]
            
            flat_id_list = []
            for element in trash_list:
                if isinstance(element, list):
                    flat_id_list.extend(element)
                else:
                    flat_id_list.append(element)
            
            df_temp = pl.DataFrame({"PDB_ID": flat_id_list, "split_type": split_name})
            df_temp = df_temp.with_columns(pl.col("PDB_ID").cast(pl.Utf8))
            dfs_splits.append(df_temp)
            
    if dfs_splits:
        return pl.concat(dfs_splits)
        
    return pl.DataFrame({"PDB_ID": [], "split_type": []}, schema={"PDB_ID": pl.Utf8, "split_type": pl.Utf8})


def migrate():
    
    print("[*] Reading Info.csv...")
    df_info = pl.read_csv(f"{DATA_PATH}/Info.csv", separator=";")
    
    print("[*] Pairing with Train/Valid/Test...")
    df_splits = load_splits(DATA_PATH)
    
    if df_splits.height > 0:
        df_final = df_info.join(df_splits, on="PDB_ID", how="left")
    else:
        df_final = df_info.with_columns(pl.lit(None).alias("split_type"))

    print("[*] Generating paths...")
    
    df_final = df_final.with_columns([
        pl.concat_str([pl.lit(f"{DATA_PATH}/Complex/ALIGNED/"), pl.col("PDB_ID"), pl.lit(".cif")]).alias("path_complex_aligned"),
        pl.concat_str([pl.lit(f"{DATA_PATH}/Complex/CIF_FROM_PDB_NOT_ALIGNED/"), pl.col("PDB_ID"), pl.lit(".cif.gz")]).alias("path_complex_cif_gz"),
        pl.concat_str([pl.lit(f"{DATA_PATH}/Complex/PROTONATED_NOT_ALIGNED/"), pl.col("PDB_ID"), pl.lit(".cif")]).alias("path_complex_protonated"),
        pl.concat_str([pl.lit(f"{DATA_PATH}/Interaction/"), pl.col("PDB_ID"), pl.lit("_ligand.json")]).alias("path_interaction_json"),
        pl.concat_str([pl.lit(f"{DATA_PATH}/Ligand/Ligand_CIF/"), pl.col("PDB_ID"), pl.lit("_ligand.cif")]).alias("path_ligand_cif"),
        pl.concat_str([pl.lit(f"{DATA_PATH}/Ligand/Ligand_SDF/"), pl.col("PDB_ID"), pl.lit("_ligand.sdf")]).alias("path_ligand_sdf"),
        pl.concat_str([pl.lit(f"{DATA_PATH}/Ligand/Ligand_SMI/"), pl.col("PDB_ID"), pl.lit("_"), pl.col("Ligand"), pl.lit(".smi")]).alias("path_ligand_smi"),
        pl.concat_str([pl.lit(f"{DATA_PATH}/Protein/Protein_CIF/"), pl.col("PDB_ID"), pl.lit("_protein.cif")]).alias("path_protein_cif"),
        pl.concat_str([pl.lit(f"{DATA_PATH}/Protein/Protein_DSSP/"), pl.col("PDB_ID"), pl.lit("_protein.dssp")]).alias("path_protein_dssp"),
        pl.concat_str([pl.lit(f"{DATA_PATH}/Protein/Protein_PDB/"), pl.col("PDB_ID"), pl.lit("_protein.pdb")]).alias("path_protein_pdb")
    ])

    df_final = df_final.rename({col: col.lower().replace(" ", "_") for col in df_final.columns})

    print(f"[*] Writing the table ({df_final.height} files, {df_final.width} columns) into PostgreSQL...")
    df_final.write_database(
        table_name="mpro_dataset",
        connection=URI_DB,
        if_table_exists="replace",
        engine="adbc"
    )
    print("[V] Table 'mpro_dataset' successfully created.\n")


def migrate_interactions():
    
    json_files = glob.glob(os.path.join(DATA_PATH, "Interaction", "*.json"))
    print(f"[*] Processing {len(json_files)} JSON interaction files...")
    
    interaction_files = []
    
    for file in json_files:
        # ARREGLAT: fitxer -> file
        pdb_id = os.path.basename(file).split("_")[0]
        
        with open(file, 'r') as f:
            try:
                data = json.load(f)
            except:
                continue 
                
        for interaction in data:
            if interaction.get("interacting_entities") == "INTER":
                bgn = interaction.get("bgn", {})
                end = interaction.get("end", {})
                
                if bgn.get("label_comp_type") in ["P", "p", "Protein", "protein"] or end.get("label_comp_type") in ["B", "b", "Ligand", "ligand"]:
                    prot = bgn
                    lig = end
                else:
                    prot = end
                    lig = bgn
                
                row = {
                    "pdb_id": pdb_id,
                    "protein_chain": prot.get("auth_asym_id"),
                    "protein_residue": prot.get("label_comp_id"),
                    "protein_res_id": prot.get("auth_seq_id"),
                    "protein_atom": prot.get("auth_atom_id"),
                    "ligand_atom": lig.get("auth_atom_id"),
                    "contact_types": ";".join(interaction.get("contact", [])),
                    "interaction_type": interaction.get("type"),
                    "distance": interaction.get("distance")
                }
                interaction_files.append(row)

    df_interactions = pl.DataFrame(interaction_files)
    print(f"[*] Writing {df_interactions.height} interactions into PostgreSQL...")
    
    df_interactions.write_database(
        table_name="interactions_pl",
        connection=URI_DB,
        if_table_exists="replace",
        engine="adbc"
    )
    print("[V] Table 'interactions_pl' successfully created.\n")


if __name__ == "__main__":
    migrate()
    migrate_interactions()
    print("[V] Migration completed successfully")