#pragma once
#include "fvCFD.H"
#include "interpolationCellPoint.H"
#include "cellPointWeight.H"
#include "meshSearch.H"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cctype>

namespace Foam {

struct GridGeom {
    label nx, ny;
    scalar x0, y0;
    scalar dx, dy;
    boundBox bb;
};

struct GridSample {
    GridGeom geom;
    scalarField T;
    scalarField Ux;
    scalarField Uy;
    scalarField Vort;
};

struct NormParams {
    scalar lo, hi, eps;
    bool clip;
    scalar Tmin, Tmax;
    scalar Umin, Umax;
    scalar Vmin, Vmax;
    scalar Wmin, Wmax;
	// normalization method: "minmax" (default) or "zscore"
	word method;
	// z-score parameters (used when method == "zscore")
	scalar Tmean, Tstd;
	scalar Umean, Ustd;
	scalar Vmean, Vstd;
	scalar Wmean, Wstd;
	// optional z-score clipping bounds
	bool zClip;
	scalar zLo, zHi;
};

// Safe clamping
inline scalar clamp01(const scalar v, const scalar lo, const scalar hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

// Dictionary utilities
inline scalar lookupScalarOrDefault(const dictionary& dict, const word& key, const scalar defVal)
{
    return dict.found(key) ? readScalar(dict.lookup(key)) : defVal;
}

inline bool lookupBoolOrDefault(const dictionary& dict, const word& key, const bool defVal)
{
    if (dict.found(key)) {
        const Switch sw(dict.lookup(key));
        return bool(sw);
    }
    return defVal;
}

inline word lookupWordOrDefault(const dictionary& dict, const word& key, const word& defVal)
{
	if (dict.found(key)) {
		word w(dict.lookup(key));
		return w;
	}
	return defVal;
}

// lowercase utility compatible with OpenFOAM word
inline word toLowerWord(const word& w)
{
	Foam::string s(w);
	for (size_t i = 0; i < s.size(); ++i)
	{
		unsigned char c = static_cast<unsigned char>(s[i]);
		s[i] = static_cast<char>(std::tolower(c));
	}
	return word(s);
}

// Load normalization parameters
inline NormParams loadNormParams(const fvMesh& mesh, const word& dictName = "mlNormDict", const bool useCp = false)
{
    IOdictionary d(
        IOobject(
            dictName,
            mesh.time().constant(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );

    const dictionary* rootPtr = &d;
    if (d.found("mlNormDict")) rootPtr = &d.subDict("mlNormDict");
    const dictionary& root = *rootPtr;

    NormParams np;
	np.lo   = lookupScalarOrDefault(root, "outRangeLow",  0.0);
	np.hi   = lookupScalarOrDefault(root, "outRangeHigh", 1.0);
	np.eps  = lookupScalarOrDefault(root, "eps",          1e-12);
	np.clip = lookupBoolOrDefault(root,   "clip",         true);
	np.method = lookupWordOrDefault(root, "method", "minmax");
	np.method = toLowerWord(np.method);

    const word scalarKey = (useCp && root.found("cp")) ? word("cp") : word("p");
    const dictionary& Td = root.subDict(scalarKey);
    const dictionary& Ud = root.subDict("u");
    const dictionary& Vd = root.subDict("v");
    const dictionary* WdPtr = root.found("vorticity") ? &root.subDict("vorticity") : nullptr;

	// Defaults for min-max
	np.Tmin = lookupScalarOrDefault(Td, "min", 0.0);
	np.Tmax = lookupScalarOrDefault(Td, "max", 1.0);
	np.Umin = lookupScalarOrDefault(Ud, "min", 0.0);
	np.Umax = lookupScalarOrDefault(Ud, "max", 1.0);
	np.Vmin = lookupScalarOrDefault(Vd, "min", 0.0);
	np.Vmax = lookupScalarOrDefault(Vd, "max", 1.0);
	np.Wmin = WdPtr ? lookupScalarOrDefault(*WdPtr, "min", 0.0) : 0.0;
	np.Wmax = WdPtr ? lookupScalarOrDefault(*WdPtr, "max", 1.0) : 1.0;

	// z-score parameters
	np.Tmean = lookupScalarOrDefault(Td, "mean", 0.0);
	np.Tstd  = lookupScalarOrDefault(Td, "std",  1.0);
	np.Umean = lookupScalarOrDefault(Ud, "mean", 0.0);
	np.Ustd  = lookupScalarOrDefault(Ud, "std",  1.0);
	np.Vmean = lookupScalarOrDefault(Vd, "mean", 0.0);
	np.Vstd  = lookupScalarOrDefault(Vd, "std",  1.0);
	np.Wmean = WdPtr ? lookupScalarOrDefault(*WdPtr, "mean", 0.0) : 0.0;
	np.Wstd  = WdPtr ? lookupScalarOrDefault(*WdPtr, "std",  1.0) : 1.0;

	np.zClip = lookupBoolOrDefault(root, "zClip", false);
	np.zLo   = lookupScalarOrDefault(root, "zClipLow", -GREAT);
	np.zHi   = lookupScalarOrDefault(root, "zClipHigh",  GREAT);
    
    return np;
}

// Generate uniform grid points
inline pointField makePlaneGrid(
    const fvMesh& mesh,
    const label nx, const label ny,
    GridGeom& gg
)
{
    gg.nx = nx; 
    gg.ny = ny;
    gg.bb = mesh.bounds();

    const point pMin = gg.bb.min();
    const point pMax = gg.bb.max();
    const scalar zMid = 0.5*(pMin.z() + pMax.z());

    gg.dx = (nx > 1) ? (pMax.x() - pMin.x())/scalar(nx - 1) : 0.0;
    gg.dy = (ny > 1) ? (pMax.y() - pMin.y())/scalar(ny - 1) : 0.0;
    gg.x0 = pMin.x();
    gg.y0 = pMin.y();

    pointField pts(nx*ny);
    
    label k = 0;
    for (label j = 0; j < ny; ++j) {
        const scalar y = gg.y0 + gg.dy*j;
        for (label i = 0; i < nx; ++i) {
            const scalar x = gg.x0 + gg.dx*i;
            pts[k++] = point(x, y, zMid);
        }
    }
    
    return pts;
}

// (moved makeGridAndCellIdx definitions below the region makePlaneGrid)

// Generate uniform grid points limited to user region
inline pointField makePlaneGrid(
    const fvMesh& mesh,
    const label nx, const label ny,
    GridGeom& gg,
    const boundBox& regionBb
)
{
    gg.nx = nx;
    gg.ny = ny;
    gg.bb = regionBb;

    const point pMin = gg.bb.min();
    const point pMax = gg.bb.max();
    const scalar zMid = 0.5*(pMin.z() + pMax.z());

    gg.dx = (nx > 1) ? (pMax.x() - pMin.x())/scalar(nx - 1) : 0.0;
    gg.dy = (ny > 1) ? (pMax.y() - pMin.y())/scalar(ny - 1) : 0.0;
    gg.x0 = pMin.x();
    gg.y0 = pMin.y();

    pointField pts(nx*ny);

    label k = 0;
    for (label j = 0; j < ny; ++j) {
        const scalar y = gg.y0 + gg.dy*j;
        for (label i = 0; i < nx; ++i) {
            const scalar x = gg.x0 + gg.dx*i;
            pts[k++] = point(x, y, zMid);
        }
    }

    return pts;
}

// Compute grid points and corresponding containing-cell indices (full domain)
inline void makeGridAndCellIdx(
    const fvMesh& mesh,
    const label nx, const label ny,
    GridGeom& gg,
    List<label>& cellIdx
)
{
    pointField pts = makePlaneGrid(mesh, nx, ny, gg);
    const boundBox& bb = gg.bb;
    const scalar eps = SMALL;
    const label nPoints = nx*ny;
    cellIdx.setSize(nPoints, -1);
    meshSearch mSearch(mesh);
    for (label k = 0; k < nPoints; ++k) {
        point q = pts[k];
        q.x() = Foam::clamp(q.x(), bb.min().x()+eps, bb.max().x()-eps);
        q.y() = Foam::clamp(q.y(), bb.min().y()+eps, bb.max().y()-eps);
        q.z() = Foam::clamp(q.z(), bb.min().z()+eps, bb.max().z()-eps);
        label celli = mSearch.findCell(q);
        if (celli < 0) {
            celli = mSearch.findNearestCell(q);
        }
        cellIdx[k] = celli;
    }
}

// Compute grid points and corresponding containing-cell indices (region)
inline void makeGridAndCellIdx(
    const fvMesh& mesh,
    const label nx, const label ny,
    GridGeom& gg,
    List<label>& cellIdx,
    const boundBox& regionBb
)
{
    pointField pts = makePlaneGrid(mesh, nx, ny, gg, regionBb);
    const boundBox& bb = gg.bb;
    const scalar eps = SMALL;
    const label nPoints = nx*ny;
    cellIdx.setSize(nPoints, -1);
    meshSearch mSearch(mesh);
    for (label k = 0; k < nPoints; ++k) {
        point q = pts[k];
        q.x() = Foam::clamp(q.x(), bb.min().x()+eps, bb.max().x()-eps);
        q.y() = Foam::clamp(q.y(), bb.min().y()+eps, bb.max().y()-eps);
        q.z() = Foam::clamp(q.z(), bb.min().z()+eps, bb.max().z()-eps);
        label celli = mSearch.findCell(q);
        if (celli < 0) {
            celli = mSearch.findNearestCell(q);
        }
        cellIdx[k] = celli;
    }
}

// Sample fields to grid - FIXED VERSION without problematic parallelization
inline GridSample samplePUtoGrid(
    const fvMesh& mesh,
    const volScalarField& P,
    const volVectorField& U,
    const volScalarField& Vort,
    const label nx = 128,
	const label ny = 128,
	const bool useNearest = false
)
{
    GridSample out;
	out.geom.nx = nx;
	out.geom.ny = ny;
	
	// Static cache across calls (rebuild only if grid or mesh bounds/topology changed)
	static label lastNX = -1, lastNY = -1;
	static label lastNCells = -1;
	static boundBox lastBB;
	static GridGeom cachedGeom;
	static pointField ptsCache;
	static List<label> cellIdxCache;
	const boundBox curBB = mesh.bounds();
	const bool bbChanged =
		(mag(curBB.min().x() - lastBB.min().x()) > SMALL) ||
		(mag(curBB.min().y() - lastBB.min().y()) > SMALL) ||
		(mag(curBB.min().z() - lastBB.min().z()) > SMALL) ||
		(mag(curBB.max().x() - lastBB.max().x()) > SMALL) ||
		(mag(curBB.max().y() - lastBB.max().y()) > SMALL) ||
		(mag(curBB.max().z() - lastBB.max().z()) > SMALL);
	const bool needRebuild = (nx != lastNX) || (ny != lastNY)
		|| (mesh.nCells() != lastNCells) || bbChanged
		|| (ptsCache.size() != nx*ny);
	
	pointField& pts = ptsCache;
	List<label>& cellIdx = cellIdxCache;
	if (needRebuild)
	{
		// Build grid points and indices
		pts = makePlaneGrid(mesh, nx, ny, out.geom);
		pts.setSize(nx*ny);
		cellIdx.setSize(nx*ny, -1);
		cachedGeom = out.geom;
		const boundBox& bb = out.geom.bb;
		const scalar eps = SMALL;
		meshSearch mSearch(mesh);
		for (label k = 0; k < nx*ny; ++k) {
			point q = pts[k];
			q.x() = Foam::clamp(q.x(), bb.min().x()+eps, bb.max().x()-eps);
			q.y() = Foam::clamp(q.y(), bb.min().y()+eps, bb.max().y()-eps);
			q.z() = Foam::clamp(q.z(), bb.min().z()+eps, bb.max().z()-eps);
			pts[k] = q;
			label celli = mSearch.findCell(q);
			if (celli < 0) {
				celli = mSearch.findNearestCell(q);
			}
			cellIdx[k] = celli;
		}
		lastNX = nx;
		lastNY = ny;
		lastNCells = mesh.nCells();
		lastBB = curBB;
	}
	// Ensure geometry matches cached grid
	out.geom = cachedGeom;
	const pointField& ptsRO = pts;
	const List<label>& cellIdxRO = cellIdx;
    
    const label nPoints = nx * ny;
    out.T.setSize(nPoints);
    out.Ux.setSize(nPoints);
    out.Uy.setSize(nPoints);
    out.Vort.setSize(nPoints);
    
    // Create interpolators (only if not using nearest sampling)
	// Info<< "Creating interpolators..." << endl;
    interpolationCellPoint<scalar> Tint(P);
    interpolationCellPoint<vector> Uint(U);
    interpolationCellPoint<scalar> Vint(Vort);
    
    // Info<< "Sampling " << nPoints << " points..." << endl;
    
    #ifdef _OPENMP
    {
        // const int nThreads = omp_get_max_threads();
        // Info<< "OpenMP enabled: using " << nThreads << " threads" << endl;
        #pragma omp parallel
        {
            interpolationCellPoint<scalar> TintLocal(P);
            interpolationCellPoint<vector> UintLocal(U);
            interpolationCellPoint<scalar> VintLocal(Vort);
            #pragma omp for schedule(static)
            for (label k = 0; k < nPoints; ++k) {
				const point& q = ptsRO[k];
				const label celli = cellIdxRO[k];
                if (celli >= 0) {
					if (useNearest) {
						out.T[k] = P[celli];
						const vector Uv = U[celli];
						out.Ux[k] = Uv.x();
						out.Uy[k] = Uv.y();
                        out.Vort[k] = Vort[celli];
					} else {
						cellPointWeight cpw(mesh, q, celli, -1);
						out.T[k] = TintLocal.interpolate(cpw);
						const vector Uv = UintLocal.interpolate(cpw);
						out.Ux[k] = Uv.x();
						out.Uy[k] = Uv.y();
                        out.Vort[k] = VintLocal.interpolate(cpw);
					}
                } else {
                    out.T[k] = 0.0;
                    out.Ux[k] = 0.0;
                    out.Uy[k] = 0.0;
                    out.Vort[k] = 0.0;
                }
            }
        }
    }
    #else
    // Sequential processing (fallback if OpenMP disabled)
    for (label k = 0; k < nPoints; ++k) {
		const point& q = ptsRO[k];
		const label celli = cellIdxRO[k];
        if (celli >= 0) {
			if (useNearest) {
				out.T[k] = P[celli];
				const vector Uv = U[celli];
				out.Ux[k] = Uv.x();
				out.Uy[k] = Uv.y();
                out.Vort[k] = Vort[celli];
			} else {
				cellPointWeight cpw(mesh, q, celli, -1);
				out.T[k] = Tint.interpolate(cpw);
				const vector Uv = Uint.interpolate(cpw);
				out.Ux[k] = Uv.x();
				out.Uy[k] = Uv.y();
                out.Vort[k] = Vint.interpolate(cpw);
			}
        } else {
            out.T[k] = 0.0;
            out.Ux[k] = 0.0;
            out.Uy[k] = 0.0;
            out.Vort[k] = 0.0;
        }
        
        if (nPoints > 10000 && k % 5000 == 0) {
            Info<< "  Sampled " << k << "/" << nPoints << " points" << endl;
        }
    }
    #endif
    
    Info<< "Sampling complete" << endl;
    
    return out;
}

// Sample fields to grid limited to user region
inline GridSample samplePUtoGrid(
    const fvMesh& mesh,
    const volScalarField& P,
    const volVectorField& U,
    const volScalarField& Vort,
    const label nx,
    const label ny,
	const boundBox& regionBb,
	const bool useNearest = false
)
{
    GridSample out;
	out.geom.nx = nx;
	out.geom.ny = ny;

	// Static cache across calls for region version
	static label lastNXr = -1, lastNYr = -1;
	static label lastNCellsr = -1;
	static boundBox lastRegionBB;
	static GridGeom cachedGeom_r;
	static pointField ptsCache;
	static List<label> cellIdxCache;
	const bool regionChanged =
		(mag(regionBb.min().x() - lastRegionBB.min().x()) > SMALL) ||
		(mag(regionBb.min().y() - lastRegionBB.min().y()) > SMALL) ||
		(mag(regionBb.min().z() - lastRegionBB.min().z()) > SMALL) ||
		(mag(regionBb.max().x() - lastRegionBB.max().x()) > SMALL) ||
		(mag(regionBb.max().y() - lastRegionBB.max().y()) > SMALL) ||
		(mag(regionBb.max().z() - lastRegionBB.max().z()) > SMALL);
	const bool needRebuild = (nx != lastNXr) || (ny != lastNYr)
		|| (mesh.nCells() != lastNCellsr) || regionChanged
		|| (ptsCache.size() != nx*ny);
	
	pointField& pts = ptsCache;
	List<label>& cellIdx = cellIdxCache;
	if (needRebuild)
	{
		pts = makePlaneGrid(mesh, nx, ny, out.geom, regionBb);
		pts.setSize(nx*ny);
		cellIdx.setSize(nx*ny, -1);
		cachedGeom_r = out.geom;
		const boundBox& bb = out.geom.bb;
		const scalar eps = SMALL;
		meshSearch mSearch(mesh);
		for (label k = 0; k < nx*ny; ++k) {
			point q = pts[k];
			q.x() = Foam::clamp(q.x(), bb.min().x()+eps, bb.max().x()-eps);
			q.y() = Foam::clamp(q.y(), bb.min().y()+eps, bb.max().y()-eps);
			q.z() = Foam::clamp(q.z(), bb.min().z()+eps, bb.max().z()-eps);
			pts[k] = q;
			label celli = mSearch.findCell(q);
			if (celli < 0) {
				celli = mSearch.findNearestCell(q);
			}
			cellIdx[k] = celli;
		}
		lastNXr = nx;
		lastNYr = ny;
		lastNCellsr = mesh.nCells();
		lastRegionBB = regionBb;
	}
	// Ensure geometry matches cached grid
	out.geom = cachedGeom_r;
	const pointField& ptsRO = pts;
	const List<label>& cellIdxRO = cellIdx;

    const label nPoints = nx * ny;
    out.T.setSize(nPoints);
    out.Ux.setSize(nPoints);
    out.Uy.setSize(nPoints);
    out.Vort.setSize(nPoints);

    // Create interpolators (only if not using nearest sampling)
	// Info<< "Creating interpolators..." << endl;
    interpolationCellPoint<scalar> Tint(P);
    interpolationCellPoint<vector> Uint(U);
    interpolationCellPoint<scalar> Vint(Vort);

    // Info<< "Sampling " << nPoints << " points..." << endl;

    #ifdef _OPENMP
    {
        // const int nThreads = omp_get_max_threads();
        // Info<< "OpenMP enabled: using " << nThreads << " threads" << endl;
        #pragma omp parallel
        {
            interpolationCellPoint<scalar> TintLocal(P);
            interpolationCellPoint<vector> UintLocal(U);
            interpolationCellPoint<scalar> VintLocal(Vort);
            #pragma omp for schedule(static)
            for (label k = 0; k < nPoints; ++k) {
				const point& q = ptsRO[k];
				const label celli = cellIdxRO[k];
                if (celli >= 0) {
					if (useNearest) {
						out.T[k] = P[celli];
						const vector Uv = U[celli];
						out.Ux[k] = Uv.x();
						out.Uy[k] = Uv.y();
                        out.Vort[k] = Vort[celli];
					} else {
						cellPointWeight cpw(mesh, q, celli, -1);
						out.T[k] = TintLocal.interpolate(cpw);
						const vector Uv = UintLocal.interpolate(cpw);
						out.Ux[k] = Uv.x();
						out.Uy[k] = Uv.y();
                        out.Vort[k] = VintLocal.interpolate(cpw);
					}
                } else {
                    out.T[k] = 0.0;
                    out.Ux[k] = 0.0;
                    out.Uy[k] = 0.0;
                    out.Vort[k] = 0.0;
                }
            }
        }
    }
    #else
    for (label k = 0; k < nPoints; ++k) {
		const point& q = ptsRO[k];
		const label celli = cellIdxRO[k];
        if (celli >= 0) {
			if (useNearest) {
				out.T[k] = P[celli];
				const vector Uv = U[celli];
				out.Ux[k] = Uv.x();
				out.Uy[k] = Uv.y();
                out.Vort[k] = Vort[celli];
			} else {
				cellPointWeight cpw(mesh, q, celli, -1);
				out.T[k] = Tint.interpolate(cpw);
				const vector Uv = Uint.interpolate(cpw);
				out.Ux[k] = Uv.x();
				out.Uy[k] = Uv.y();
                out.Vort[k] = Vint.interpolate(cpw);
			}
        } else {
            out.T[k] = 0.0;
            out.Ux[k] = 0.0;
            out.Uy[k] = 0.0;
            out.Vort[k] = 0.0;
        }
    }
    #endif

    Info<< "Sampling complete (region)" << endl;
    return out;
}

// Normalization with bounds checking
inline void normalizeMinMaxInPlace(
    scalarField& ch, 
    const scalar minv, const scalar maxv,
    const scalar lo, const scalar hi, 
    const scalar eps, const bool doClip
)
{
    const scalar denom = max(mag(maxv - minv), eps);
    const scalar a = (hi - lo) / denom;
    const scalar b = lo - a*minv;
    
    const scalar loB = min(lo, hi);
    const scalar hiB = max(lo, hi);
    
    forAll(ch, i) {
        scalar v = a*ch[i] + b;
        if (doClip) {
            v = clamp01(v, loB, hiB);
        }
        ch[i] = v;
    }
}

// Z-score normalization: (x - mean) / std, with optional clipping
inline void normalizeZScoreInPlace(
	scalarField& ch,
	const scalar meanv, const scalar stdv,
	const scalar eps, const bool doClip,
	const scalar loZ, const scalar hiZ
)
{
	const scalar denom = max(mag(stdv), eps);
	forAll(ch, i) {
		scalar v = (ch[i] - meanv) / denom;
		if (doClip) {
			const scalar loB = min(loZ, hiZ);
			const scalar hiB = max(loZ, hiZ);
			v = clamp01(v, loB, hiB);
		}
		ch[i] = v;
	}
}

}
