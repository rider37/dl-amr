#!/usr/bin/env python3
"""
Audit an OpenFOAM polyMesh for 2-D integrity: is it still exactly one cell
thick along the empty direction?

Reads the mesh files directly - no OpenFOAM run required.

Reports
  - the empty-patch face count against the 2*nCells invariant
  - how many cells have 2 / 1 / 0 empty faces
  - the distinct coordinate planes along the span direction
  - internal faces whose normal lies along the span direction (these carry
    spurious out-of-plane flux and must not exist in a 2-D mesh)
  - the equivalent hexRef4 cell count and the cell overhead hexRef8 imposes

Usage
    check2D.py <case>/constant/polyMesh
    check2D.py <case>/<time>/polyMesh
    check2D.py <case>              # audits every polyMesh found under it
"""

import sys
import os
import re
import math
import collections


def strip_header(txt):
    """Drop the FoamFile { ... } header block."""
    i = txt.find('FoamFile')
    if i < 0:
        return txt
    j = txt.index('}', i)
    return txt[j + 1:]


def is_binary(path):
    with open(path, 'rb') as f:
        head = f.read(2048)
    return b'format' in head and b'binary' in head


def read_labels(path):
    body = strip_header(open(path).read())
    m = re.search(r'(\d+)\s*\(', body)
    if not m:
        return []
    n = int(m.group(1))
    start = m.end()
    end = body.index(')', start)
    vals = body[start:end].split()
    return [int(v) for v in vals[:n]]


def read_points(path):
    body = strip_header(open(path).read())
    m = re.search(r'(\d+)\s*\(', body)
    n = int(m.group(1))
    out = []
    for p in re.findall(r'\(\s*([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s*\)',
                        body[m.end() - 1:]):
        out.append((float(p[0]), float(p[1]), float(p[2])))
        if len(out) == n:
            break
    return out


def read_faces(path):
    body = strip_header(open(path).read())
    m = re.search(r'(\d+)\s*\(', body)
    n = int(m.group(1))
    out = []
    for f in re.finditer(r'(\d+)\s*\(([\d\s]+)\)', body[m.end() - 1:]):
        out.append([int(x) for x in f.group(2).split()])
        if len(out) == n:
            break
    return out


def read_boundary(path):
    body = strip_header(open(path).read())
    i = body.index('(')
    out = []
    for name, blk in re.findall(r'(\w+)\s*\{([^}]*)\}', body[i:]):
        t = re.search(r'type\s+(\w+)', blk)
        nf = re.search(r'nFaces\s+(\d+)', blk)
        sf = re.search(r'startFace\s+(\d+)', blk)
        if t and nf and sf:
            out.append((name, t.group(1), int(nf.group(1)), int(sf.group(1))))
    return out


def newell(face, pts):
    ax = ay = az = 0.0
    k = len(face)
    for i in range(k):
        p = pts[face[i]]
        q = pts[face[(i + 1) % k]]
        ax += (p[1] - q[1]) * (p[2] + q[2])
        ay += (p[2] - q[2]) * (p[0] + q[0])
        az += (p[0] - q[0]) * (p[1] + q[1])
    return (ax / 2, ay / 2, az / 2)


def audit(meshDir):
    print("=" * 74)
    print(meshDir)
    print("=" * 74)

    need = ['points', 'faces', 'owner', 'boundary']
    for f in need:
        p = os.path.join(meshDir, f)
        if not os.path.exists(p):
            print("  skip: no %s" % f)
            return None
        if is_binary(p):
            print("  skip: %s is in binary format; rewrite the case with "
                  "writeFormat ascii to audit it" % f)
            return None

    owner = read_labels(os.path.join(meshDir, 'owner'))
    pts = read_points(os.path.join(meshDir, 'points'))
    faces = read_faces(os.path.join(meshDir, 'faces'))
    patches = read_boundary(os.path.join(meshDir, 'boundary'))

    nCells = max(owner) + 1
    nFaces = len(faces)
    nInternal = min(sf for (_, _, _, sf) in patches) if patches else nFaces

    empties = [p for p in patches if p[1] == 'empty']

    print("  nPoints %d   nCells %d   nFaces %d   nInternalFaces %d"
          % (len(pts), nCells, nFaces, nInternal))

    # ---- span direction --------------------------------------------------
    # Prefer the empty patches. Without them (a symmetry-patch "2-D" case,
    # which is the usual hexRef8 workaround) fall back on the thinnest
    # bounding box dimension.
    if empties:
        acc = [0.0, 0.0, 0.0]
        for (_, _, nf, sf) in empties:
            for fi in range(sf, sf + nf):
                a = newell(faces[fi], pts)
                m = math.sqrt(sum(c * c for c in a))
                if m > 0:
                    for c in range(3):
                        acc[c] += abs(a[c]) / m
        spanDir = max(range(3), key=lambda c: acc[c])
        how = 'from empty patches'
    else:
        ext = [max(p[c] for p in pts) - min(p[c] for p in pts)
               for c in range(3)]
        spanDir = min(range(3), key=lambda c: ext[c])
        how = 'thinnest bbox dimension, extents %.3g/%.3g/%.3g' % tuple(ext)

    print("  span direction: %s   (%s)" % ('xyz'[spanDir], how))

    # ---- boundary faces normal to the span -------------------------------
    # These are the caps of each in-plane column, whatever patch they sit on
    # (empty, symmetry, symmetryPlane, ...). A one-cell-thick mesh has
    # exactly 2 per cell.
    spanPatches = collections.Counter()
    cnt = collections.Counter()
    nSpanBoundary = 0

    for (name, t, nf, sf) in patches:
        for fi in range(sf, sf + nf):
            a = newell(faces[fi], pts)
            m = math.sqrt(sum(c * c for c in a))
            if m > 0 and abs(a[spanDir]) > 0.999 * m:
                cnt[owner[fi]] += 1
                nSpanBoundary += 1
                spanPatches[(name, t)] += 1

    if not nSpanBoundary:
        print("  no boundary faces normal to the span -> cannot audit")
        return None

    print("  span-normal boundary faces are on: %s"
          % ', '.join('%s (%s) x%d' % (n, t, c)
                      for (n, t), c in spanPatches.items()))

    nEmpty = nSpanBoundary
    hist = collections.Counter(cnt.values())
    n0 = nCells - len(cnt)

    ok = (nEmpty == 2 * nCells)
    print("  span-normal boundary faces %d   2*nCells %d   ->  %s"
          % (nEmpty, 2 * nCells,
             "ONE CELL THICK" if ok else "*** MORE THAN ONE CELL THICK ***"))
    print("     cells with 2 span faces: %d" % hist.get(2, 0))
    print("     cells with 1 span face : %d" % hist.get(1, 0))
    print("     cells with 0 span faces: %d" % n0)
    if empties and not ok:
        print("     -> these are EMPTY patches, so the empty BC no longer"
              " covers the whole front/back: the case is not 2-D")
    elif not empties and not ok:
        print("     -> symmetry patches, so this is consistent, but the extra"
              " layers add no in-plane resolution (pure cost)")

    # ---- span planes -----------------------------------------------------
    planes = sorted(set(round(p[spanDir], 9) for p in pts))
    print("  distinct %s planes: %s%s"
          % ('xyz'[spanDir],
             planes[:8],
             " ..." if len(planes) > 8 else ""))
    if len(planes) > 2:
        print("     -> the mesh is up to %d cells thick along the span"
              % (len(planes) - 1))

    # ---- spurious internal span-normal faces -----------------------------
    zn = 0
    zarea = 0.0
    for fi in range(nInternal):
        a = newell(faces[fi], pts)
        m = math.sqrt(sum(c * c for c in a))
        if m > 0 and abs(a[spanDir]) > 0.999 * m:
            zn += 1
            zarea += m
    print("  internal faces normal to the span: %d  (total area %.4g)"
          % (zn, zarea))
    if zn:
        print("     -> these carry out-of-plane flux; a true 2-D mesh has none")

    # ---- cost -------------------------------------------------------------
    # Each in-plane column contributes exactly 2 empty faces, so the number of
    # columns is the cell count an equivalent hexRef4 mesh would have at the
    # same in-plane resolution.
    nColumns = nEmpty // 2
    print("  equivalent hexRef4 cell count (same in-plane resolution): %d"
          % nColumns)
    if nColumns:
        print("  hexRef8 overhead: %+.1f%% cells (%d extra)"
              % (100.0 * (nCells - nColumns) / nColumns, nCells - nColumns))
    stacked = nCells - hist.get(2, 0)
    stackedCols = nColumns - hist.get(2, 0)
    if stackedCols > 0:
        print("  within the refined region only: %d cells vs %d needed"
              " (%.2fx)" % (stacked, stackedCols, stacked / stackedCols))

    return dict(nCells=nCells, nEmpty=nEmpty, ok=ok, nColumns=nColumns,
                spanFaces=zn, planes=len(planes))


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1

    targets = []
    for a in argv[1:]:
        if os.path.basename(a.rstrip('/')) == 'polyMesh':
            targets.append(a)
        else:
            for root, dirs, _ in os.walk(a):
                if os.path.basename(root) == 'polyMesh':
                    targets.append(root)
                    dirs[:] = []

    if not targets:
        print("no polyMesh directories found")
        return 1

    bad = 0
    for t in sorted(targets):
        r = audit(t)
        print()
        if r and not r['ok']:
            bad += 1

    print("%d of %d meshes are NOT one cell thick" % (bad, len(targets)))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))
