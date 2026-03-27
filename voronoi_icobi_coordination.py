#!/usr/bin/env python3
"""
Coordination number calculation via Voronoi polyhedron from a VASP POSCAR file.

Dependencies:
    pip install numpy scipy pymatgen

Usage:
    python voronoi_coordination.py POSCAR [--atom 0] [--cutoff 10.0] [--econ] [--all]
                                          [--threshold 0.02] [--icobilist ICOBILIST.lobster]
                                          [--icobi-threshold 0.01] [--show-excluded]
"""

import argparse
import sys
import numpy as np


# ─────────────────────────────────────────────
# 1. POSCAR reader
# ─────────────────────────────────────────────

def read_poscar(filepath):
    """
    Reads a POSCAR file (VASP 4 or 5) and returns:
        lattice   : np.ndarray (3×3), lattice vectors as rows (Å)
        positions : np.ndarray (N×3), fractional coordinates in [0,1[
        species   : list[str], chemical symbol for each atom
    """
    with open(filepath) as f:
        lines = f.readlines()

    # Line 1: comment
    # Line 2: scaling factor
    scale = float(lines[1].strip())

    # Lines 3-5: lattice vectors
    lattice = np.array([[float(x) for x in lines[i].split()] for i in range(2, 5)]) * scale

    # Line 6 (VASP5: species names) or line 7 (VASP4: direct counts)
    tokens6 = lines[5].split()
    try:
        int(tokens6[0])  # VASP4: no species names
        species_names = None
        counts = [int(x) for x in tokens6]
        coord_start = 7
    except ValueError:
        species_names = tokens6          # VASP5
        counts = [int(x) for x in lines[6].split()]
        coord_start = 8

    # Build species list per atom
    if species_names is None:
        species_names = [f"X{i}" for i in range(len(counts))]
    species = []
    for sym, cnt in zip(species_names, counts):
        species.extend([sym] * cnt)

    # Coordinate type (Direct / Cartesian)
    coord_type = lines[coord_start - 1].strip()[0].upper()  # 'D' or 'C'/'K'

    # Read positions
    n_atoms = sum(counts)
    raw = np.array([[float(x) for x in lines[coord_start + i].split()[:3]]
                    for i in range(n_atoms)])

    if coord_type == 'D':
        positions = raw % 1.0  # wrap into [0,1[
    else:
        # Cartesian -> fractional
        positions = (np.linalg.inv(lattice.T) @ raw.T).T % 1.0

    return lattice, positions, species


# ─────────────────────────────────────────────
# 2. ICOBILIST.lobster reader
# ─────────────────────────────────────────────

import re

def _lobster_label_to_index(label, species):
    """
    Converts a LOBSTER label (e.g. 'Na1', 'H9', 'Be5') to a 0-based index
    by extracting the number and comparing with the species list.

    LOBSTER numbers atoms 1-based in the POSCAR order.
    → Python index = LOBSTER_number - 1
    """
    m = re.match(r'^([A-Za-z]+)(\d+)$', label)
    if not m:
        return None
    elem, num = m.group(1), int(m.group(2))
    idx = num - 1  # 1-based → 0-based
    if idx < len(species) and species[idx].lower() == elem.lower():
        return idx
    # Fallback: linear search (in case LOBSTER order differs)
    count = 0
    for i, sp in enumerate(species):
        if sp.lower() == elem.lower():
            count += 1
            if count == num:
                return i
    return None


def read_icobilist(filepath, species):
    """
    Reads an ICOBILIST.lobster file and returns a dict:
        icobi_data[(idx_mu, idx_nu)] = ICOBI_total

    Only total lines (atoms without orbital suffix) are read.
    Distance is not used for identification; matching is done via
    pair indices + translation vector.

    Returned structure:
        {
          (idx_mu, idx_nu): [
              {'icobi': float, 'distance': float, 'translation': np.array([na,nb,nc])},
              ...
          ]
        }
    """
    icobi_data = {}

    with open(filepath) as f:
        lines = f.readlines()

    for line in lines[2:]:   # skip first two header lines
        parts = line.split()
        if len(parts) < 8:
            continue

        label_mu = parts[1]
        label_nu = parts[2]

        # Total line = labels without underscore (no orbital suffix)
        if '_' in label_mu or '_' in label_nu:
            continue

        try:
            dist  = float(parts[3])
            na    = int(parts[4])
            nb    = int(parts[5])
            nc    = int(parts[6])
            icobi = float(parts[7])
        except ValueError:
            continue

        idx_mu = _lobster_label_to_index(label_mu, species)
        idx_nu = _lobster_label_to_index(label_nu, species)
        if idx_mu is None or idx_nu is None:
            continue

        key = (idx_mu, idx_nu)
        if key not in icobi_data:
            icobi_data[key] = []
        icobi_data[key].append({
            'icobi'      : icobi,
            'distance'   : dist,
            'translation': np.array([na, nb, nc]),
        })

    return icobi_data


def match_icobi(neighbor, center_index, icobi_data, tol=1e-3):
    """
    Finds the ICOBI corresponding to a Voronoi neighbor.

    The neighbor dict contains:
        neighbor['atom_index'] : index of the neighboring atom
        neighbor['distance']   : distance (Å)

    Looks in icobi_data[(center, neighbor)] or (neighbor, center)
    for an entry whose distance matches within ±tol Å.
    Returns the ICOBI value or None if not found.
    """
    idx_v = neighbor['atom_index']
    d_v   = neighbor['distance']

    for key in [(center_index, idx_v), (idx_v, center_index)]:
        if key in icobi_data:
            for entry in icobi_data[key]:
                if abs(entry['distance'] - d_v) < tol:
                    return entry['icobi']
    return None


# ─────────────────────────────────────────────
# 3. Periodic image generation
# ─────────────────────────────────────────────

def get_periodic_images(positions, lattice, cutoff):
    """
    Returns all Cartesian positions (including images) within a sphere of
    radius `cutoff` around the origin, along with the base atom index and
    translation vector.
    """
    # Number of replicas needed along each axis
    norms = np.linalg.norm(lattice, axis=1)
    reps = np.ceil(cutoff / norms).astype(int) + 1

    images = []   # (cart_pos, atom_index, translation_vector)
    for i, pos in enumerate(positions):
        cart_base = lattice.T @ pos
        for na in range(-reps[0], reps[0] + 1):
            for nb in range(-reps[1], reps[1] + 1):
                for nc in range(-reps[2], reps[2] + 1):
                    t = na * lattice[0] + nb * lattice[1] + nc * lattice[2]
                    images.append((cart_base + t, i, np.array([na, nb, nc])))
    return images


# ─────────────────────────────────────────────
# 4. Voronoi polyhedron calculation
# ─────────────────────────────────────────────

def voronoi_analysis(center_index, positions, lattice, species, cutoff=12.0):
    """
    Computes the Voronoi polyhedron for the atom `center_index`.

    Returns a list of dicts:
        {
          'atom_index': int,
          'species'   : str,
          'distance'  : float (Å),
          'face_area' : float (Å²),
        }
    sorted by increasing distance.
    """
    from scipy.spatial import Voronoi

    # Cartesian position of the central atom
    center_cart = lattice.T @ positions[center_index]

    # Retrieve all periodic images
    all_images = get_periodic_images(positions, lattice, cutoff)

    # Keep only images within cutoff sphere around the center
    neighbor_pts = []
    neighbor_meta = []
    for (cart, idx, trans) in all_images:
        d = np.linalg.norm(cart - center_cart)
        if d < cutoff and not (idx == center_index and np.all(trans == 0)):
            neighbor_pts.append(cart)
            neighbor_meta.append({'atom_index': idx,
                                   'species': species[idx],
                                   'distance': d})

    if len(neighbor_pts) < 4:
        raise ValueError("Not enough neighbors to build the Voronoi polyhedron.")

    # Build Voronoi tessellation
    # First point is the central atom
    all_pts = np.vstack([[center_cart], neighbor_pts])
    vor = Voronoi(all_pts)

    # Identify the region of the central point (index 0)
    region_index = vor.point_region[0]
    region = vor.regions[region_index]

    if -1 in region:
        print("⚠  Open Voronoi region — increase --cutoff.")

    # For each ridge adjacent to the central point
    coordination = []
    for (p1, p2), vertices in zip(vor.ridge_points, vor.ridge_vertices):
        if p1 != 0 and p2 != 0:
            continue
        neighbor_pt_idx = p2 if p1 == 0 else p1
        if neighbor_pt_idx == 0:
            continue
        if -1 in vertices:
            continue

        # Vertices of the face
        face_verts = vor.vertices[vertices]
        area = polygon_area(face_verts)

        # Metadata of the neighbor
        meta = neighbor_meta[neighbor_pt_idx - 1].copy()
        meta['face_area'] = area
        meta['face_vertices'] = face_verts
        coordination.append(meta)

    coordination.sort(key=lambda x: x['distance'])
    return coordination


# ─────────────────────────────────────────────
# 5. Geometric utilities
# ─────────────────────────────────────────────

def polygon_area(vertices):
    """
    Area of a planar 3D polygon (cross product method).
    `vertices` : np.ndarray (N×3)
    """
    if len(vertices) < 3:
        return 0.0
    c = vertices.mean(axis=0)
    area = 0.0
    n = len(vertices)
    for i in range(n):
        v1 = vertices[i] - c
        v2 = vertices[(i + 1) % n] - c
        area += np.linalg.norm(np.cross(v1, v2))
    return area / 2.0


def effective_coordination_number(coordination):
    """
    ECoN (O'Keeffe 1979):
        ECoN = Σ exp(1 - (d_i / d_min)^6)

    where d_min is the distance to the first neighbor.
    """
    if not coordination:
        return 0.0
    d_min = min(c['distance'] for c in coordination)
    econ = sum(np.exp(1 - (c['distance'] / d_min) ** 6) for c in coordination)
    return econ


def weighted_econ_area(coordination):
    """
    Area-weighted ECoN:
        ECoN_area = Σ (A_i / A_max)

    A_max = largest face area.
    """
    if not coordination:
        return 0.0
    a_max = max(c['face_area'] for c in coordination)
    if a_max == 0:
        return 0.0
    return sum(c['face_area'] / a_max for c in coordination)


# ─────────────────────────────────────────────
# 6. Output formatting
# ─────────────────────────────────────────────

def print_results(atom_idx, species, coordination, econ=False, threshold=0.0,
                  icobi_data=None, icobi_threshold=0.0, center_index=None,
                  show_excluded=False):
    """
    Prints the Voronoi polyhedron results, optionally with ICOBI information.

    threshold         : minimum face area fraction to retain a neighbor (0 = no filter).
    icobi_data        : dict from read_icobilist(), or None.
    icobi_threshold   : minimum ICOBI to retain a neighbor (0 = no filter).
    center_index      : index of the central atom (required for ICOBI matching).
    show_excluded     : whether to print the list of excluded neighbors.
    """
    sp = species[atom_idx]
    total_area = sum(c['face_area'] for c in coordination)

    # Compute area fractions
    for c in coordination:
        c['frac_area'] = c['face_area'] / total_area if total_area > 0 else 0.0

    # Inject ICOBI values if available
    has_icobi = icobi_data is not None
    if has_icobi:
        ctr = center_index if center_index is not None else atom_idx
        for c in coordination:
            c['icobi'] = match_icobi(c, ctr, icobi_data)

    # Filtering
    def keep(c):
        if c['frac_area'] < threshold:
            return False
        if has_icobi and icobi_threshold > 0.0:
            iv = c.get('icobi')
            if iv is None or iv < icobi_threshold:
                return False
        return True

    active   = [c for c in coordination if keep(c)]
    excluded = [c for c in coordination if not keep(c)]

    cn_raw   = len(coordination)
    cn_filt  = len(active)

    # Header
    width = 80 if has_icobi else 68
    print(f"\n{'═'*width}")
    print(f"  Atom {atom_idx}  ({sp})")
    print(f"  Raw CN (strict Voronoi)      = {cn_raw}")
    if threshold > 0.0 or (has_icobi and icobi_threshold > 0.0):
        filters = []
        if threshold > 0.0:
            filters.append(f"area ≥ {threshold:.1%}")
        if has_icobi and icobi_threshold > 0.0:
            filters.append(f"ICOBI ≥ {icobi_threshold:.4f}")
        print(f"  Filtered CN ({', '.join(filters)}) = {cn_filt}")

    if econ:
        ec_filt  = effective_coordination_number(active)
        ew_filt  = weighted_econ_area(active)
        ec_raw   = effective_coordination_number(coordination)
        ew_raw   = weighted_econ_area(coordination)
        if cn_filt != cn_raw:
            print(f"  ECoN (O'Keeffe)  (filtered) = {ec_filt:.4f}   [raw : {ec_raw:.4f}]")
            print(f"  ECoN (area-weighted) (filtered) = {ew_filt:.4f}   [raw : {ew_raw:.4f}]")
        else:
            print(f"  ECoN (O'Keeffe, distance)   = {ec_raw:.4f}")
            print(f"  ECoN (area-weighted)        = {ew_raw:.4f}")

    # Average ICOBI of kept bonds
    if has_icobi and cn_filt > 0:
        icobi_vals = [c['icobi'] for c in active if c.get('icobi') is not None]
        if icobi_vals:
            avg_icobi = np.mean(icobi_vals)
            print(f"  Average ICOBI (filtered bonds) = {avg_icobi:.5f}")

    # Table of kept neighbors
    print(f"{'─'*width}")
    if has_icobi:
        print(f"  {'#':>4}  {'Species':<8} {'Dist (Å)':>10}  {'Area (Å²)':>10}  "
              f"{'Area frac':>10}  {'ICOBI':>10}  {'Bonding':>7}")
    else:
        print(f"  {'#':>4}  {'Species':<8} {'Dist (Å)':>10}  {'Area (Å²)':>10}  {'Area frac':>10}")
    print(f"{'─'*width}")

    for i, c in enumerate(active):
        if has_icobi:
            iv = c.get('icobi')
            iv_str  = f"{iv:10.5f}" if iv is not None else f"{'N/A':>10}"
            bonding = _bonding_label(iv) if iv is not None else "  ?"
            print(f"  {i+1:>4}  {c['species']:<8} {c['distance']:>10.4f}  "
                  f"{c['face_area']:>10.4f}  {c['frac_area']:>10.4%}  "
                  f"{iv_str}  {bonding:>7}")
        else:
            print(f"  {i+1:>4}  {c['species']:<8} {c['distance']:>10.4f}  "
                  f"{c['face_area']:>10.4f}  {c['frac_area']:>10.4%}")

    # Optional excluded neighbors list
    if show_excluded and excluded:
        print(f"{'┄'*width}")
        label_parts = []
        if threshold > 0.0:
            label_parts.append(f"area < {threshold:.1%}")
        if has_icobi and icobi_threshold > 0.0:
            label_parts.append(f"ICOBI < {icobi_threshold:.4f}")
        print(f"  Excluded ({' or '.join(label_parts) if label_parts else 'filtered'}) :")
        for c in excluded:
            if has_icobi:
                iv = c.get('icobi')
                iv_str = f"{iv:10.5f}" if iv is not None else f"{'N/A':>10}"
                bonding = _bonding_label(iv) if iv is not None else "  ?"
                print(f"  {'—':>4}  {c['species']:<8} {c['distance']:>10.4f}  "
                      f"{c['face_area']:>10.4f}  {c['frac_area']:>10.4%}  "
                      f"{iv_str}  {bonding:>7}")
            else:
                print(f"  {'—':>4}  {c['species']:<8} {c['distance']:>10.4f}  "
                      f"{c['face_area']:>10.4f}  {c['frac_area']:>10.4%}")

    print(f"{'═'*width}\n")


def _bonding_label(icobi):
    """Qualitative bond strength label from ICOBI."""
    if icobi >= 0.5:
        return "strong"
    elif icobi >= 0.1:
        return "medium"
    elif icobi >= 0.02:
        return "weak"
    else:
        return "none"


# ─────────────────────────────────────────────
# 7. Command line interface
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Voronoi polyhedron and coordination number calculation from a VASP POSCAR.")
    parser.add_argument("poscar", help="Path to the POSCAR file")
    parser.add_argument("--atom", type=int, default=0,
                        help="Index of the central atom (0-based, default=0)")
    parser.add_argument("--all", action="store_true",
                        help="Compute CN for all atoms")
    parser.add_argument("--cutoff", type=float, default=12.0,
                        help="Cutoff radius for periodic images (Å, default=12)")
    parser.add_argument("--econ", action="store_true",
                        help="Also compute ECoN (effective coordination number)")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help=("Area fraction threshold (0–1) to filter non‑bonding neighbors. "
                              "E.g., 0.02 excludes faces <2%% of total area. Default: 0 (no filter)."))
    parser.add_argument("--icobilist", type=str, default=None,
                        help="Path to ICOBILIST.lobster file (optional).")
    parser.add_argument("--icobi-threshold", type=float, default=0.0,
                        dest="icobi_threshold",
                        help=("Minimum ICOBI to retain a neighbor as bonded. "
                              "E.g., 0.01. Default: 0 (no filter)."))
    parser.add_argument("--show-excluded", action="store_true",
                        help="Show the list of excluded neighbors (hidden by default).")
    args = parser.parse_args()

    # Read POSCAR
    try:
        lattice, positions, species = read_poscar(args.poscar)
    except Exception as e:
        print(f"Error reading POSCAR: {e}", file=sys.stderr)
        sys.exit(1)

    n_atoms = len(positions)
    print(f"\nStructure read: {n_atoms} atoms")
    print(f"Species: {', '.join(sorted(set(species)))}")
    print(f"Lattice:\n{np.round(lattice, 5)}")

    # Optionally load ICOBILIST
    icobi_data = None
    if args.icobilist:
        try:
            icobi_data = read_icobilist(args.icobilist, species)
            n_pairs = sum(len(v) for v in icobi_data.values())
            print(f"ICOBILIST loaded: {n_pairs} bond pairs read.")
        except Exception as e:
            print(f"⚠  Could not read ICOBILIST: {e}", file=sys.stderr)

    atoms_to_process = list(range(n_atoms)) if args.all else [args.atom]

    for idx in atoms_to_process:
        if idx < 0 or idx >= n_atoms:
            print(f"⚠  Atom {idx} out of range [0, {n_atoms-1}]", file=sys.stderr)
            continue
        try:
            coord = voronoi_analysis(idx, positions, lattice, species,
                                     cutoff=args.cutoff)
            print_results(idx, species, coord,
                          econ=args.econ,
                          threshold=args.threshold,
                          icobi_data=icobi_data,
                          icobi_threshold=args.icobi_threshold,
                          center_index=idx,
                          show_excluded=args.show_excluded)
        except Exception as e:
            print(f"⚠  Error for atom {idx}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
