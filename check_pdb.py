import os
import glob

def count_residues_in_pdb(filepath):
    """Counts unique residues in a PDB file based on ATOM records."""
    residues = set()
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                # Extract Chain ID, Residue Sequence Number, and Insertion Code
                chain_id = line[21]
                res_seq = line[22:26].strip()
                ins_code = line[26]
                # Create a unique identifier for the residue
                res_id = f"{chain_id}_{res_seq}_{ins_code}"
                residues.add(res_id)
    return len(residues)

pdb_dir = "/home/subhadip/rin_encoder_1/PDB-Uniprot"  # Update this path to your PDB directory
pdb_files = sorted(glob.glob(os.path.join(pdb_dir, "*.pdb")))

residue_counts = {}
for f in pdb_files:
    count = count_residues_in_pdb(f)
    if count not in residue_counts:
        residue_counts[count] = []
    residue_counts[count].append(os.path.basename(f))

print("📊 Residue count summary:")
for count, files in sorted(residue_counts.items()):
    print(f"\n  ✔️ {count} residues: {len(files)} files")
    if len(files) <= 5:
        for fname in files:
            print(f"      - {fname}")
    else:
        print(f"      - (showing first 5): {files[:5]} ...")