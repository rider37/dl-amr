# hexRef4 — 2-D AMR for OpenFOAM v2506 that keeps `empty` patches valid

`hexRef8` splits every hex into 8, including along the span direction. On a
one-cell-thick 2-D mesh that gives two cells across the span, which breaks the
`empty` boundary condition — so 2-D AMR normally forces you onto `symmetry`
(or `wedge`) patches and a two-layer mesh.

`hexRef4` splits a cell into 4 **in the plane only**. Span-parallel edges are
never split, so the mesh stays exactly one cell thick and `empty` patches
remain valid for the whole run.

Built against **OpenFOAM v2506** (`/usr/lib/openfoam/openfoam2506`).

---

## What is in here

| Path | Contents |
|---|---|
| `hexRef4/hexRef4.{H,C}` | The 2-D cutter. Port of `hexRef8` with the span direction frozen. |
| `hexRef4/hexRef4Data.{H,C}` | Level/history bundle for mesh redistribution (unchanged port). |
| `dynamicRefine2DFvMesh/` | `dynamicRefineFvMesh` rebuilt on `hexRef4`, incl. flux handling. |
| `test/Test-hexRef4/` | Level 1: topology checker, drives `hexRef4` directly. |
| `test/case2D/` | Minimal 2-D box case for the Level 1 checker. |
| `test/Test-refine2DFlux/` | Level 2: full `dynamicRefine2DFvMesh` stack + flux conservation. |
| `test/caseFlux/` | Case for the Level 2 checker. |
| `test/validate.sh` | Runs Levels 1 and 2 end to end. |
| `test/dynamicMeshDict.template` | Drop-in dict for your own cases. |
| `tools/check2D.py` | Audits any written `polyMesh` for 2-D integrity. No OpenFOAM run. |
| `tools/migrate2D.sh` | Converts a case from `dynamicRefineFvMesh` to the 2-D one. `-n` for dry run. |

Library: `$FOAM_USER_LIBBIN/libhexRef4.so`
Apps: `$FOAM_USER_APPBIN/{Test-hexRef4,Test-refine2DFlux}`

---

## Build

```bash
source /usr/lib/openfoam/openfoam2506/etc/bashrc

cd /mnt/hdd/Jongmin/AMR1/amr/hexRef4
wmake libso

cd test/Test-hexRef4
wmake
```

Both already build clean as committed.

---

## How the 2-D split works

For a span direction along `z`:

```
        span faces (front/back, on the empty patches)   -> split into 4
        side faces (the other four)                     -> split into 2
        span-parallel edges (the four verticals)        -> never split
```

Points introduced per refined cell:

* one mid point on each span face — `M0` (back), `M1` (front);
* one mid point on each split in-plane edge.

There is **no cell-interior point**: `M0`/`M1` take the role of `hexRef8`'s
single cell mid point. The 4 new internal faces are the quads
`(S, M0, M1, S')`, where `S`/`S'` are corresponding split points on the
back/front span faces. Each spans the full thickness, so the sub-cells are
still one cell thick.

The 8 hex corner points map onto only **4** added cells — two corners joined by
a span edge share a sub-cell. That is what lets `getAnchorCell` /
`getFaceNeighbours` be reused from `hexRef8` unchanged.

Refinement history reuses `refinementHistory`'s 8-slot `splitCell8`, filling
slots 0–3 and leaving 4–7 at `-1`; `refinementHistory` already tolerates `-1`
entries (that is how it marks freed sub-cells), so no fork of that class was
needed.

Unrefinement is driven by split points on **one** span face only (the lower
numbered of `M0`/`M1`), otherwise the same 4 internal faces would be removed
twice. Unlike `hexRef8`, points on `empty` boundary faces are *not* excluded
from the candidate list — in 2-D the group's centre point necessarily lies on
the empty patch. Points on any other boundary face are still excluded.

---

## Flux conservation

Identical in mechanism to `dynamicRefineFvMesh` (that was the requirement), and
it applies to all three categories of face:

1. **Split faces and their masters** — recomputed from the `correctFluxes`
   table as `interpolate(U) & Sf` (`mapFields`). Both split kinds are covered:
   span faces into 4, side faces into 2. The second half of a bisected side
   face is added *from* the original face (`addFace(..., facei, ...)`), so it
   inherits the parent's mapping before being overwritten, exactly as
   `hexRef8`'s three added quarter-faces do.

2. **Injected internal faces** — the 4 new internal faces per split cell have
   no correspondence in the old mesh (`faceMap == -1`), so
   `mapNewInternalFaces<T>()` reconstructs them from the hull of the
   already-mapped faces of their owner and neighbour. Oriented (flux) fields
   are converted to intensive form, interpolated, and converted back.

3. **Unrefinement** — faces merged back together are recomputed from
   `correctFluxes` via the `faceToSplitPoint` map, which walks
   `pointEdges(M0) -> otherPoint -> pointFaces`. In 2-D that reaches `M1` and
   the four back-face spokes `S`, whose `pointFaces` cover the split span
   faces and the bisected side faces (a bisected side face contains both `S`
   and `S'`). So the same code covers the 2-D case without modification.

`maxCells` accounting was corrected from `/7` to `/3` extra cells per split,
and the old-time volume correction in `mapFields` from `nSubCells == 8` to
`== 4`.

**This is an approximate reconstruction, exactly as in `hexRef8`** — the
sub-face fluxes are re-interpolated rather than redistributed from the parent,
so they are not discretely conservative at the instant of refinement. The
pressure equation re-projects onto a divergence-free field on the next
PISO/PIMPLE iteration. See *Possible improvement* below.

---

## Using it in a case

1. Load the library in `system/controlDict`:

   ```
   libs            ("libhexRef4.so");
   ```

2. In `constant/dynamicMeshDict`, switch the mesh type **and rename the coeffs
   sub-dict** (it is looked up as `<typeName>Coeffs`):

   ```
   dynamicFvMesh   dynamicRefine2DFvMesh;

   dynamicRefine2DFvMeshCoeffs
   {
       ...            // unchanged contents
   }
   ```

   A ready template is in `test/dynamicMeshDict.template`.

3. Everything else — `refineInterval`, `field`, `lowerRefineLevel`,
   `unrefineLevel`, `nBufferLayers`, `maxRefinement`, `maxCells`,
   `correctFluxes`, `dumpLevel`, `protectedCells` — is unchanged. Solver code
   that does `refCast<dynamicRefineFvMesh>(mesh).protectedCell()` must become
   `refCast<dynamicRefine2DFvMesh>(mesh)` and include
   `dynamicRefine2DFvMesh.H`.

`maxCells` now buys 3× more refinement events than before for the same
number, since each split adds 3 cells rather than 7 — worth revisiting if you
carry a phase-4 value over.

### Level files are shared with hexRef8

`hexRef4` reads and writes the same `constant/polyMesh/{cellLevel,pointLevel,
level0Edge,refinementHistory}` files as `hexRef8`. Start from a fresh mesh; do
not resume a `hexRef8`-refined case with `hexRef4` (the history would describe
8-way splits).

---

## Why this is not optional: the existing phase-4 meshes are not 2-D

`tools/check2D.py` audits a written `polyMesh` directly — no OpenFOAM run
needed. Run on a phase-4 AMR snapshot it reports:

```
BO_Re150_2D_phase4/exp_G_mr2_lr004/254.5/polyMesh
  nCells 38986
  empty faces 64524   2*nCells 77972   ->  *** NOT 2-D ***
     cells with 2 empty faces: 29794
     cells with 1 empty face : 4936
     cells with 0 empty faces: 4256
  distinct z planes: [-0.5, -0.25, 0.0, 0.25, 0.5]
     -> the mesh is up to 4 cells thick along the span
  internal faces normal to the span: 6727  (total area 11.33)
  equivalent hexRef4 cell count (same in-plane resolution): 32262
  hexRef8 overhead: +20.8% cells (6724 extra)
  within the refined region only: 9192 cells vs 2468 needed (3.72x)
```

The baseline `Re150_GCI_v2/Coarse` mesh audits clean (`ONE CELL THICK`,
0 span-normal internal faces), so this is introduced purely by `hexRef8`
refinement. With `maxRefinement 2` the refined wake ends up 4 cells thick.

Two consequences:

1. **Cost** — the refined region carries 3.7× the cells its in-plane
   resolution needs; 21% overhead on the whole mesh at this snapshot, growing
   toward 4× as the refined fraction grows. For a BO study whose objective
   includes cost, and whose driver has an 80k hard kill, a large part of the
   cell budget is being spent stacking z-layers that add no resolution.

2. **Physics** — the case is no longer strictly 2-D. Measured in that same
   snapshot: `max|Uz| = 3.98e-3` against `max|U_inplane| = 1.37`, i.e. 0.29%
   spurious out-of-plane velocity. Small, but non-zero and unphysical; the
   6727 span-normal internal faces carry real flux that a 2-D discretisation
   should not have.

Use `tools/check2D.py <case>` (it walks a whole case tree) to audit any other
runs before deciding what to keep.

---

## Validation

**Not yet run here** — solvers and mesh utilities are not executed
automatically in this project, so the ladder below is for you to run.

### Levels 1 and 2 — scripted, no solver

```bash
cd /mnt/hdd/Jongmin/AMR1/amr/hexRef4/test
./validate.sh          # or: ./validate.sh 1   /   ./validate.sh 2
```

**Level 1 — mesh engine** (`Test-hexRef4` on `case2D`). Refines a shrinking
cylinder three times, then unrefines everything possible, asserting after
*every* topology change:

| | check |
|---|---|
| Test 1 | `nCells_new == nCells_old + 3*nRefined` (and `- 3*nMerged` on unrefine). A 2-D split adds **3**, not 7. |
| Test 2 | `nEmptyFaces == 2*nCells`, and every cell has exactly 2 span-normal faces |
| Test 3 | total volume unchanged to 1e-12 relative |
| Test 4 | `max abs(level_i - level_j) <= 1` across every internal and coupled face |
| Test 5 | a uniform volume field stays uniform to 1e-12 |
| — | `primitiveMesh::checkMesh`, `hexRef4::checkRefinementLevels`, `hexRef4::checkMesh` |

**Level 2 — flux and mapping** (`Test-refine2DFlux` on `caseFlux`). Drives
`dynamicRefine2DFvMesh` with a disc sweeping across the domain so both
refinement and unrefinement fire, and adds the decisive test:

> For a uniform `U` with no span component, `phi = interpolate(U) & Sf` is
> **exactly** discretely divergence-free on any mesh, since
> `sum_f U.Sf = U . sum_f Sf = 0` for a closed cell and the empty faces
> contribute nothing. So `max_cells |sum_f phi_f|` must stay at round-off
> after every topology change.

This is what catches a wrong sign: if any new internal face has its
owner/neighbour swapped relative to its normal, or an injected face keeps a
stale value, the error jumps from ~1e-16 to O(|U|·A) and the test aborts with
that diagnosis. It also re-checks Tests 1–5 (cell-count delta divisible by 3,
empty invariant, volume, 2:1, uniform scalar *and* uniform vector).

### Levels 3–5 — your solver

Not scripted; they need `pthermalpimpleFoam_*`.

3. Coarse cylinder, `hexRef4` AMR + `empty`, against the non-AMR `Coarse`
   reference — compare Cd, Cl, St, U, ω_z.
4. Same indicator, `hexRef8` + `empty` (i.e. the current phase-4 setup) vs
   `hexRef4` + `empty` — cost at equal in-plane resolution. Use
   `tools/check2D.py` on both to state the comparison exactly. Note the
   apples-to-apples baseline here is *not* `hexRef8 + symmetry`: your cases
   already use `empty`, and `hexRef8` has been silently 3-D-ifying them.
5. One good indicator against a uniform ~100k mesh, then restart BO.

Do not restart the BO campaign until 1–3 pass.

### Known gaps

* **Parallel is untested.** The span-direction detection is reduced across
  processors so it is consistent, but the ordering of the two halves of a
  bisected side face on a processor patch has not been checked. Run the
  single-processor test first.
* A cell whose `empty`-patch face is *already* more refined than the cell
  itself is rejected with an explicit `FatalError` rather than handled.
  `hexRef8` supports that case; it does not arise from refine/unrefine alone,
  only from subsetting.
* `level0EdgeLength()` is inherited unchanged and sees the constant-length span
  edges. It only feeds distance-based refinement (`consistentSlowRefinement2`),
  which `dynamicRefine2DFvMesh` does not use, so it is harmless here.

---

## Possible improvement: genuinely conservative refinement

The current behaviour follows `hexRef8`, as asked. If you later want the flux
to be *discretely* conservative across a refinement event:

* On a split face the sub-faces already hold the parent's mapped flux value —
  scaling by `magSf_sub/magSf_parent` instead of re-interpolating makes the
  face exactly conservative (in 2-D a bisected side face is an exact half).
* The 4 new internal fluxes are then fixed up to one degree of freedom by
  requiring each of the 4 sub-cells to be divergence-free (4 equations, rank 3
  around the cycle); taking the minimum-norm solution closes it.

That would make refinement flux-neutral and remove the pressure-solver
transient after each AMR step. It is a separate change and is *not*
implemented here.
