import glob
from rin_encodermap.pdb_parsing import parse_pdb

for f in sorted(glob.glob("/home/subhadip/rin_encoder_1/filtered_pdb/*.pdb")):
    residues = parse_pdb(f)
    print(len(residues), f)