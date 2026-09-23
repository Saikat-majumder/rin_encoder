"""
pdb_parsing.py
==============

Turn a protein structure (PDB file) into a Residue Interaction Network
(RIN) and extract its closeness-centrality fingerprint, following:

    Franke, L.; Peter, C. "Visualizing the Residue Interaction Landscape of
    Proteins by Temporal Network Embedding." J. Chem. Theory Comput. 2023,
    19, 2985-2995. https://doi.org/10.1021/acs.jctc.2c01228

Workflow (Section 4.1-4.2 of the paper):
  1. Parse a PDB frame and collect backbone atom coordinates per residue.
  2. Build an undirected, unweighted RIN: two residues are connected if the
     minimum distance between any of their atoms is below a cutoff
     (default 6.0 A), excluding directly-neighboring residues in sequence.
  3. Compute the closeness centrality of every residue -- the reciprocal
     mean shortest-path length to all other residues (Eq. 1) -- which
     forms the N-dimensional "closeness fingerprint" for that structure.
  4. (Optional) visualize the RIN and project fingerprints across many
     frames into 2D with classical MDS, as a quick, non-neural-network
     sanity check before running the full EncoderMap pipeline
     (see train_encodermap.py / test_encodermap.py).

This module has no dependency on PyTorch -- it only needs Biopython,
NetworkX, NumPy, scikit-learn and Matplotlib, so it can be run as a
lightweight, independent featurization step whose output (.npy fingerprint
files) is later consumed by the training/testing scripts.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser
from sklearn.manifold import MDS
from scipy.spatial.distance import pdist, squareform

STANDARD_AA = {
    "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE",
    "LYS", "LEU", "MET", "ASN", "PRO", "GLN", "ARG", "SER",
    "THR", "VAL", "TRP", "TYR",
}
BACKBONE_ATOMS = ("N", "CA", "C", "O")


# ----------------------------------------------------------------------
# 1. Parse PDB
# ----------------------------------------------------------------------
def parse_pdb(pdb_file, model_index=0):
    """
    Parse a PDB file and return a list of (residue_id, coords) tuples.

    coords is an (n_backbone_atoms, 3) array of the residue's N, CA, C, O
    coordinates. Non-amino-acid residues (water, ligands, ions, ...) are
    skipped.

    model_index selects which MODEL to read for multi-model PDB files
    (e.g. NMR ensembles); default is the first model.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    model = list(structure)[model_index]

    residues = []
    for chain in model:
        for residue in chain:
            if residue.get_resname() not in STANDARD_AA:
                continue
            atoms = [atom for atom in residue if atom.name in BACKBONE_ATOMS]
            if not atoms:
                continue
            coords = np.array([atom.coord for atom in atoms])
            residues.append((residue.get_id()[1], coords))
    return residues


# ----------------------------------------------------------------------
# 2. Build Residue Interaction Network (RIN)
# ----------------------------------------------------------------------
def build_rin(residues, cutoff=6.0):
    """
    Build an undirected, unweighted RIN.

    Nodes are residues (graph node index i corresponds to residues[i]).
    An edge connects residues i and j if the minimum distance between any
    of their backbone atoms is <= cutoff (Angstrom), excluding residues
    that are directly adjacent in the protein sequence (|res_id_i -
    res_id_j| == 1), as in the paper's Section 4.2.
    """
    G = nx.Graph()
    n_residues = len(residues)

    for i, (res_id, _) in enumerate(residues):
        G.add_node(i, residue_id=res_id)

    for i in range(n_residues):
        for j in range(i + 1, n_residues):
            if abs(residues[i][0] - residues[j][0]) == 1:
                continue
            coords_i = residues[i][1]
            coords_j = residues[j][1]
            min_dist = np.min(np.linalg.norm(coords_i[:, np.newaxis] - coords_j, axis=2))
            if min_dist <= cutoff:
                G.add_edge(i, j, weight=min_dist)

    return G


# ----------------------------------------------------------------------
# 3. Closeness centrality -> fingerprint (Eq. 1)
# ----------------------------------------------------------------------
def calculate_closeness_centrality(G, normalize=True):
    """
    Closeness centrality c_i = N / sum_j d(v_i, v_j), where d is the
    shortest (unweighted geodesic) path length in the RIN, and N is the
    number of residues.

    Returns a dict {node_index: closeness_centrality}. NetworkX's
    `distance="weight"` argument is intentionally NOT used here -- the
    paper defines d(v_i, v_j) as an unweighted edge count (Eq. 1), so the
    default (unweighted) closeness_centrality is used, and this function
    just applies the N-based normalization NetworkX already returns by
    default (kept explicit for clarity/consistency with the paper).
    """
    closeness = nx.closeness_centrality(G)
    if not normalize:
        return closeness
    return closeness  # nx.closeness_centrality already returns the (N-1)/sum(d) normalized form


def fingerprint_from_pdb(pdb_file, cutoff=6.0):
    """
    Convenience wrapper: PDB file -> RIN -> ordered closeness fingerprint.

    Returns:
        fingerprint: (n_residues,) array, ordered by residue index as
            encountered in the file (matching the protein sequence).
        G: the underlying RIN (networkx.Graph)
        residues: the parsed residue list (for coordinate/labeling use)
    """
    residues = parse_pdb(pdb_file)
    if not residues:
        raise ValueError(f"No valid amino-acid residues found in {pdb_file}")

    G = build_rin(residues, cutoff=cutoff)
    closeness = calculate_closeness_centrality(G)
    fingerprint = np.array([closeness[i] for i in range(len(residues))])
    return fingerprint, G, residues


def build_fingerprint_dataset(pdb_glob_pattern, cutoff=6.0, scale_to_unit_range=True, verbose=True):
    """
    Build an (n_frames, n_residues) fingerprint matrix from many PDB files
    (e.g. one file per MD trajectory frame), matching the paper's approach
    of stacking one fingerprint per simulation frame (Section 4.2) and
    scaling by the maximum value observed over all frames so that the
    fingerprint lies in [0, 1].

    pdb_glob_pattern: glob pattern matching one PDB file per frame, e.g.
        "frames/frame_*.pdb"

    Returns:
        X: (n_frames, n_residues) float array
        file_list: the sorted list of files used, in row order
    """
    file_list = sorted(glob.glob(pdb_glob_pattern))
    if not file_list:
        raise FileNotFoundError(f"No files matched pattern: {pdb_glob_pattern}")

    fingerprints = []
    for i, f in enumerate(file_list):
        fp, _, _ = fingerprint_from_pdb(f, cutoff=cutoff)
        fingerprints.append(fp)
        if verbose and i % 100 == 0:
            print(f"[{i + 1}/{len(file_list)}] {os.path.basename(f)}")

    X = np.stack(fingerprints, axis=0)

    if scale_to_unit_range:
        X = X / X.max()

    return X, file_list


# ----------------------------------------------------------------------
# 4. Visualization
# ----------------------------------------------------------------------
def visualize_rin(G, closeness, residues, title="Residue Interaction Network (RIN)"):
    """Draw the RIN with nodes colored by closeness centrality (node
    positions taken as the mean backbone-atom coordinate of each residue,
    matching the earlier draft's layout convention)."""
    plt.figure(figsize=(10, 8))

    pos = {}
    for node in G.nodes():
        res_id = G.nodes[node]["residue_id"]
        coords = residues[node][1]
        pos[node] = np.mean(coords, axis=0)[:2]  # project onto xy for a 2D plot

    node_colors = [closeness[n] for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_size=200, node_color=node_colors, cmap=plt.cm.viridis, alpha=0.9)
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    nx.draw_networkx_labels(G, pos, labels={n: G.nodes[n]["residue_id"] for n in G.nodes()}, font_size=8)

    sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                norm=plt.Normalize(vmin=min(node_colors), vmax=max(node_colors)))
    sm.set_array([])
    plt.colorbar(sm, label="Closeness Centrality", ax=plt.gca())

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def project_to_2d_mds(fingerprints, n_components=2, random_state=42):
    """
    Quick classical-MDS projection of a set of closeness fingerprints
    (one row per frame/structure) into 2D, as a fast sanity check before
    running the full nonlinear EncoderMap embedding.

    fingerprints: (n_frames, n_residues) array
    Returns: (n_frames, n_components) array of MDS coordinates
    """
    distances = squareform(pdist(fingerprints, metric="euclidean"))
    mds = MDS(n_components=n_components, dissimilarity="precomputed", random_state=random_state, normalized_stress="auto")
    return mds.fit_transform(distances)


def plot_2d_projection(coords_2d, color_values=None, title="Residue Interaction Landscape (MDS)",
                        cbar_label="value"):
    """Scatter-plot a 2D fingerprint projection, optionally colored by an
    external per-frame observable (e.g. RMSD, radius of gyration)."""
    plt.figure(figsize=(8, 7))
    if color_values is not None:
        sc = plt.scatter(coords_2d[:, 0], coords_2d[:, 1], c=color_values, cmap="viridis", s=10, alpha=0.8)
        plt.colorbar(sc, label=cbar_label)
    else:
        plt.scatter(coords_2d[:, 0], coords_2d[:, 1], s=10, alpha=0.8)
    plt.xlabel("MDS dimension 1")
    plt.ylabel("MDS dimension 2")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------------------
# 5. Single-structure end-to-end workflow
# ----------------------------------------------------------------------
def analyze_pdb(pdb_file, cutoff=6.0, plot=True):
    """Single-PDB demo: parse -> RIN -> closeness centrality -> (optional) plot."""
    residues = parse_pdb(pdb_file)
    if not residues:
        raise ValueError("No valid residues found in the PDB file.")

    G = build_rin(residues, cutoff=cutoff)
    closeness = calculate_closeness_centrality(G)
    print("Closeness centrality (by residue index):", closeness)

    if plot:
        visualize_rin(G, closeness, residues)

    return G, closeness


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse a PDB file into a RIN and print/plot its closeness fingerprint.")
    parser.add_argument("pdb_file", type=str, help="Path to a .pdb file")
    parser.add_argument("--cutoff", type=float, default=6.0, help="Contact-distance cutoff in Angstrom (default 6.0)")
    parser.add_argument("--no-plot", action="store_true", help="Skip the RIN visualization")
    args = parser.parse_args()

    analyze_pdb(args.pdb_file, cutoff=args.cutoff, plot=not args.no_plot)
