/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2019 OpenCFD Ltd.
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.
\*---------------------------------------------------------------------------*/

#include "torch/torch.h"
#include "torch/script.h"
#include <ATen/Context.h>
#include "fvCFD.H"
#include "dynamicFvMesh.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "pimpleControl.H"
#include "CorrectPhi.H"
#include "fvOptions.H"
#include "localEulerDdtScheme.H"
#include "fvcSmooth.H"
#include "Resize.H"
#include "aiInference.H"
#ifdef _OPENMP
#include <omp.h>
#endif

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList::addNote
    (
        "Transient solver for incompressible, turbulent flow"
        " of Newtonian fluids on a moving mesh with AI-guided AMR."
    );

    #include "postProcess.H"
    #include "addCheckCaseOptions.H"
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createDynamicFvMesh.H"
    #include "initContinuityErrs.H"
    #include "createDyMControls.H"
    #include "createFields.H"
    #include "createUfIfPresent.H"
    #include "CourantNo.H"
    #include "setInitialDeltaT.H"

    turbulence->validate();

    if (!LTS)
    {
        #include "CourantNo.H"
        #include "setInitialDeltaT.H"
    }

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nStarting time loop\n" << endl;

        // Disable cuDNN for stability (debug) - keep CUDA, but avoid cuDNN engine issues
        at::globalContext().setUserEnabledCuDNN(false);
        Info<< "[AI DEBUG] cuDNN disabled via at::globalContext()" << nl;

    // --- Read refineInterval from dynamicMeshDict
    label refineInterval = 1;
    {
        IOdictionary dyMDict
        (
            IOobject
            (
                "dynamicMeshDict",
                runTime.constant(),
                mesh,
                IOobject::MUST_READ_IF_MODIFIED,
                IOobject::NO_WRITE
            )
        );
        
        // Try to find refineInterval in various locations
        if (dyMDict.found("refineInterval"))
        {
            refineInterval = readLabel(dyMDict.lookup("refineInterval"));
        }
        else if (dyMDict.found("dynamicRefineFvMeshCoeffs"))
        {
            const dictionary& refineDict = dyMDict.subDict("dynamicRefineFvMeshCoeffs");
            if (refineDict.found("refineInterval"))
            {
                refineInterval = readLabel(refineDict.lookup("refineInterval"));
            }
        }
        else if (dyMDict.found("dynamicRefineBalancedFvMeshCoeffs"))
        {
            const dictionary& refineDict = dyMDict.subDict("dynamicRefineBalancedFvMeshCoeffs");
            if (refineDict.found("refineInterval"))
            {
                refineInterval = readLabel(refineDict.lookup("refineInterval"));
            }
        }
        
        refineInterval = max(refineInterval, label(1));
        
        Info<< "\n==================================" << endl;
        Info<< "AMR Configuration:" << endl;
        Info<< "  refineInterval = " << refineInterval << endl;
        Info<< "  Mesh refinement will occur every " << refineInterval << " timesteps" << endl;
        Info<< "  AI inference will run immediately before refinement" << endl;
        Info<< "==================================\n" << endl;
    }

    // --- ML Configuration - Lazy loading
    torch::jit::script::Module aiModule;
    NormParams normParams;
    label NX = 128, NY = 128;
    scalar attnThr = 0.5;
    scalar noRefineRadius = 2.0;   // default: radius 2.0 (r^2=4.0 as before)
    word noRefineShape("circle"); // circle | rect | none
    vector noRefineCenter(0, 0, 0);
    vector noRefineMin(0, 0, 0);
    vector noRefineMax(0, 0, 0);
    bool useRegion = false;
    bool useRe = false;
    scalar ReValue = 1.0;
    scalar ReRef = 1.0;
    word reMode("concat"); // "concat" (Re channel) | "film" (Re scalar input)
    bool useCp = false;
    bool uncStats = false;
    bool uncUseSigma = false;
    word refineMode("sigma"); // "sigma" (default, log-variance head) | "delta_norm" (mean head L2 norm, ablation)
    scalar uInf = 1.0;
    scalar pRef = 0.0;
    bool useVort = true;
    bool cudnnEnabled = true;
    bool cudnnBenchmark = false;
    bool cudnnDeterministic = true;
    bool cudnnTF32 = false;
    boundBox regionBb;
    bool mlConfigLoaded = false;
    bool thrGreaterEqual = true; // default: ">="
    bool sampleNearest = false;  // default: linear interpolation
    // CUDA device management
    at::Device aiDevice(torch::kCPU);
    bool useCuda = false;
    
    // Initialize refineFlag to zero (but don't write it yet)
    Info<< "Initializing refineFlag field" << endl;
    refineFlag = dimensionedScalar("zero", dimless, 1.0);

    // Main time loop
    while (runTime.run())
    {
        #include "readDyMControls.H"

        if (LTS)
        {
            #include "setRDeltaT.H"
        }
        else
        {
            #include "CourantNo.H"
            #include "setDeltaT.H"
        }

        ++runTime;

        Info<< "Time = " << runTime.timeName() << nl << endl;

        // --- Pressure-velocity PIMPLE corrector loop
        while (pimple.loop())
        {
            if (pimple.firstIter() || moveMeshOuterCorrectors)
            {
                // Check if mesh refinement will happen this iteration
                // This happens BEFORE mesh.controlledUpdate()
                bool meshWillRefine = false;
                if (refineInterval > 0)
                {
                    // OpenFOAM typically refines at timesteps: refineInterval, 2*refineInterval, 3*refineInterval, etc.
                    // We need to generate refineFlag right before the refinement
                    if ((runTime.timeIndex() % refineInterval) == 0 && runTime.timeIndex() > 0)
                    {
                        meshWillRefine = true;
                    }
                }

                // Generate refineFlag only if mesh is about to refine
                if (meshWillRefine)
                {
                    Info<< "\n>>> MESH REFINEMENT IMMINENT - GENERATING REFINEFLAG <<<" << endl;
                    Info<< "    Timestep: " << runTime.timeIndex() << endl;
                    
                    // Load ML configuration on first use
                    if (!mlConfigLoaded)
                    {
                        Info<< "\nLoading ML configuration..." << endl;
                        
                        try {
                            // Read ML inference dictionary
                            IOdictionary infDict
                            (
                                IOobject
                                (
                                    "mlInferDict",
                                    runTime.constant(),
                                    mesh,
                                    IOobject::MUST_READ,
                                    IOobject::NO_WRITE
                                )
                            );
                            
                            const dictionary& mlDict = infDict.found("mlInferDict") 
                                ? infDict.subDict("mlInferDict") 
                                : infDict;
                            
                            // Extract configuration
                            const fileName modelPath(mlDict.lookup("modelPath"));
                            NX = readLabel(mlDict.lookup("NX"));
                            NY = readLabel(mlDict.lookup("NY"));
                            attnThr = readScalar(mlDict.lookup("attnThr"));
                            
                            // Optional threshold direction
                            if (mlDict.found("thrDirection"))
                            {
                                const word dir(mlDict.lookup("thrDirection"));
                                word dLower = word(dir);
                                dLower = toLowerWord(dLower);
                                if (dLower == "le" || dLower == "lessequal" || dLower == "leq")
                                {
                                    thrGreaterEqual = false;
                                }
                                else
                                {
                                    thrGreaterEqual = true;
                                }
                            }
                            else if (mlDict.found("thrIsGE"))
                            {
                                const Switch sw(mlDict.lookup("thrIsGE"));
                                thrGreaterEqual = bool(sw);
                            }
                            
                            // Optional region restriction
                            if (mlDict.found("useRegion"))
                            {
                                const Switch sw(mlDict.lookup("useRegion"));
                                useRegion = bool(sw);
                            }
                            // Optional no-refine inner radius (around origin)
                            if (mlDict.found("noRefineRadius"))
                            {
                                noRefineRadius = readScalar(mlDict.lookup("noRefineRadius"));
                            }
                            if (mlDict.found("noRefineShape"))
                            {
                                word s(mlDict.lookup("noRefineShape"));
                                noRefineShape = toLowerWord(s);
                            }
                            if (mlDict.found("noRefineCenter"))
                            {
                                const vector c = vector(mlDict.lookup("noRefineCenter"));
                                noRefineCenter = c;
                            }
                            if (mlDict.found("noRefineMin") && mlDict.found("noRefineMax"))
                            {
                                const vector rmin = vector(mlDict.lookup("noRefineMin"));
                                const vector rmax = vector(mlDict.lookup("noRefineMax"));
                                noRefineMin = rmin;
                                noRefineMax = rmax;
                                if (noRefineShape == word("circle"))
                                {
                                    noRefineShape = word("rect");
                                }
                            }
                            // Optional sampling method: nearest or linear
                            if (mlDict.found("sampleMethod"))
                            {
                                word sm(mlDict.lookup("sampleMethod"));
                                sm = toLowerWord(sm);
                                sampleNearest = (sm == "nearest");
                            }
                            else if (mlDict.found("sampleNearest"))
                            {
                                const Switch sw(mlDict.lookup("sampleNearest"));
                                sampleNearest = bool(sw);
                            }

                            // Optional: append Re channel as constant (broadcast) input
                            if (mlDict.found("useRe"))
                            {
                                const Switch sw(mlDict.lookup("useRe"));
                                useRe = bool(sw);
                            }
                            // Optional: use Cp instead of p
                            if (mlDict.found("useCp"))
                            {
                                const Switch sw(mlDict.lookup("useCp"));
                                useCp = bool(sw);
                            }
                            if (mlDict.found("uncStats"))
                            {
                                const Switch sw(mlDict.lookup("uncStats"));
                                uncStats = bool(sw);
                            }
                            if (mlDict.found("uncUseSigma"))
                            {
                                const Switch sw(mlDict.lookup("uncUseSigma"));
                                uncUseSigma = bool(sw);
                            }
                            if (mlDict.found("refineMode"))
                            {
                                refineMode = word(mlDict.lookup("refineMode"));
                            }
                            if (mlDict.found("useVorticity"))
                            {
                                const Switch sw(mlDict.lookup("useVorticity"));
                                useVort = bool(sw);
                            }
                            if (mlDict.found("cudnnEnabled"))
                            {
                                const Switch sw(mlDict.lookup("cudnnEnabled"));
                                cudnnEnabled = bool(sw);
                            }
                            if (mlDict.found("disableCuDNN"))
                            {
                                const Switch sw(mlDict.lookup("disableCuDNN"));
                                cudnnEnabled = !bool(sw);
                            }
                            if (mlDict.found("cudnnBenchmark"))
                            {
                                const Switch sw(mlDict.lookup("cudnnBenchmark"));
                                cudnnBenchmark = bool(sw);
                            }
                            if (mlDict.found("cudnnDeterministic"))
                            {
                                const Switch sw(mlDict.lookup("cudnnDeterministic"));
                                cudnnDeterministic = bool(sw);
                            }
                            if (mlDict.found("cudnnTF32"))
                            {
                                const Switch sw(mlDict.lookup("cudnnTF32"));
                                cudnnTF32 = bool(sw);
                            }
                            if (useCp)
                            {
                                if (mlDict.found("uInf")) uInf = readScalar(mlDict.lookup("uInf"));
                                else if (mlDict.found("UInf")) uInf = readScalar(mlDict.lookup("UInf"));
                                if (mlDict.found("pRef")) pRef = readScalar(mlDict.lookup("pRef"));
                                if (uInf <= SMALL)
                                {
                                    FatalErrorInFunction
                                        << "uInf must be positive for Cp computation"
                                        << exit(FatalError);
                                }
                            }
                            if (useRe)
                            {
                                if (mlDict.found("ReValue")) ReValue = readScalar(mlDict.lookup("ReValue"));
                                else if (mlDict.found("Re")) ReValue = readScalar(mlDict.lookup("Re"));
                                if (mlDict.found("ReRef")) ReRef = readScalar(mlDict.lookup("ReRef"));
                                if (mlDict.found("reMode"))
                                {
                                    reMode = word(mlDict.lookup("reMode"));
                                    reMode = toLowerWord(reMode);
                                }
                            }
                            // Load normalization parameters after useCp is known
                            normParams = loadNormParams(mesh, "mlNormDict", useCp);
                            Info<< "Normalization parameters loaded" << endl;
                            if (useRegion)
                            {
                                if (!(mlDict.found("regionMin") && mlDict.found("regionMax")))
                                {
                                    FatalErrorInFunction
                                        << "useRegion is true but regionMin/regionMax not provided in mlInferDict"
                                        << exit(FatalError);
                                }
                                const vector rmin = vector(mlDict.lookup("regionMin"));
                                const vector rmax = vector(mlDict.lookup("regionMax"));
                                regionBb = boundBox(point(rmin.x(), rmin.y(), rmin.z()),
                                                    point(rmax.x(), rmax.y(), rmax.z()));
                            }
                            
                            Info<< "ML Configuration:" << endl;
                            Info<< "  Model: " << modelPath << endl;
                            Info<< "  Grid:  " << NX << " x " << NY << endl;
                            Info<< "  Threshold: " << attnThr << endl;
                            Info<< "  Threshold compare: " << (thrGreaterEqual ? ">=" : "<=") << endl;
                            Info<< "  useCp: " << (useCp ? "true" : "false") << " (uInf=" << uInf << ", pRef=" << pRef << ")" << endl;
                            Info<< "  uncStats: " << (uncStats ? "true" : "false")
                                 << " (uncUseSigma=" << (uncUseSigma ? "true" : "false") << ")" << endl;
                            Info<< "  noRefineShape: " << noRefineShape << " | radius=" << noRefineRadius
                                 << " | center=" << noRefineCenter << " | min=" << noRefineMin << " | max=" << noRefineMax << endl;
                            Info<< "  useVorticity: " << (useVort ? "true" : "false") << endl;
                            Info<< "  cudnnEnabled: " << (cudnnEnabled ? "true" : "false")
                                 << " | cudnnBenchmark: " << (cudnnBenchmark ? "true" : "false")
                                 << " | cudnnDeterministic: " << (cudnnDeterministic ? "true" : "false")
                                 << " | cudnnTF32: " << (cudnnTF32 ? "true" : "false") << endl;
                            if (useRe)
                            {
                                Info<< "  useRe: true (ReValue=" << ReValue << ", ReRef=" << ReRef << ", reMode=" << reMode << ")" << endl;
                            }
                            if (useRegion)
                            {
                                Info<< "  Region: min=" << regionBb.min() << " max=" << regionBb.max() << endl;
                            }
                            
                            // Optional: control OpenMP thread count for sampling to avoid OMP runtime conflicts
                            #ifdef _OPENMP
                            int ompThreads = 1; // default to 1 to minimize conflicts with libtorch
                            if (mlDict.found("ompThreads"))
                            {
                                ompThreads = readLabel(mlDict.lookup("ompThreads"));
                                ompThreads = max(1, ompThreads);
                            }
                            omp_set_num_threads(ompThreads);
                            Info<< "  OMP threads for sampling: " << ompThreads << endl;
                            #endif
                            
                            // Load PyTorch model
                            if (!isFile(modelPath))
                            {
                                FatalErrorInFunction
                                    << "Model file not found: " << modelPath
                                    << exit(FatalError);
                            }
                            
                            aiModule = torch::jit::load(modelPath.c_str(), torch::kCPU);
                            aiModule.eval();
                            // Move model to CUDA once if available
                            if (torch::cuda::is_available())
                            {
                                aiDevice = at::Device(torch::kCUDA);
                                aiModule.to(aiDevice);
                                useCuda = true;
                                // cuDNN 설정 (mlInferDict로 제어)
                                at::globalContext().setUserEnabledCuDNN(cudnnEnabled);
                                at::globalContext().setBenchmarkCuDNN(cudnnBenchmark);
                                at::globalContext().setDeterministicCuDNN(cudnnDeterministic);
                                at::globalContext().setAllowTF32CuDNN(cudnnTF32);
                                Info<< "AI model loaded to CUDA device" << endl;
                            }
                            else
                            {
                                aiDevice = at::Device(torch::kCPU);
                                useCuda = false;
                                Info<< "AI model loaded on CPU (CUDA not available)" << endl;
                            }
                            Info<< "AI model loaded successfully" << endl;
                            
                            mlConfigLoaded = true;
                            
                        } catch (const std::exception& e) {
                            FatalErrorInFunction
                                << "Failed to load ML configuration: " << e.what()
                                << exit(FatalError);
                        }
                    }
                    
                    // Perform AI inference for refineFlag generation
                    try {
                        Info<< "Running AI inference..." << endl;

                        // Compute vorticity (z-component of curl(U)) for 2D input
                        tmp<volVectorField> tCurlU = fvc::curl(U);
                        volScalarField vort
                        (
                            IOobject
                            (
                                "vorticity",
                                runTime.timeName(),
                                mesh,
                                IOobject::NO_READ,
                                IOobject::NO_WRITE
                            ),
                            tCurlU().component(vector::Z)
                        );
                        
                        // Sample and inference
                        if (sampleNearest && useCuda)
                        {
                            // 1) Build or reuse cached grid and cell indices
                            static GridGeom ggIdxCache;
                            static List<label> cellIdxCache;
                            static bool cacheValid = false;
                            static label lastNXgpu = -1, lastNYgpu = -1, lastNCellsgpu = -1;
                            static boundBox lastRegionBb;
                            static bool lastUseRegion = false;

                            // Aggressive cache: build indices only on first use,
                            // and reuse even after mesh refinement for speed.
                            const bool needRebuildIdx = !cacheValid;

                            if (needRebuildIdx)
                            {
                                if (useRegion) {
                                    makeGridAndCellIdx(mesh, NX, NY, ggIdxCache, cellIdxCache, regionBb);
                                } else {
                                    makeGridAndCellIdx(mesh, NX, NY, ggIdxCache, cellIdxCache);
                                }
                                cacheValid = true;
                                lastNXgpu = NX;
                                lastNYgpu = NY;
                                lastNCellsgpu = mesh.nCells();
                                lastRegionBb = regionBb;
                                lastUseRegion = useRegion;
                            }

                            const GridGeom& ggIdx = ggIdxCache;
                            const List<label>& cellIdx = cellIdxCache;
                            const label H = ggIdx.ny;
                            const label W = ggIdx.nx;
                            const label HW = H*W;

                            // 2) Prepare/reuse host/device channel buffers (size HW only)
                            static at::Tensor hostUxCh, hostUyCh, hostPCh, hostVortCh;
                            static at::Tensor devUxCh, devUyCh, devPCh, devVortCh;
                            static at::Tensor devIdx;
                            static long lastHWDev = -1;
                            static bool devBuffersInit = false;

                            const scalarField& pInt = p.primitiveField();
                            const vectorField& uInt = U.primitiveField();
                            const scalarField& vortInt = vort.primitiveField();

                            if (!devBuffersInit || lastHWDev != static_cast<long>(HW))
                            {
                                hostUxCh = torch::empty({static_cast<long>(HW)}, at::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true));
                                hostUyCh = torch::empty({static_cast<long>(HW)}, at::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true));
                                hostPCh  = torch::empty({static_cast<long>(HW)}, at::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true));
                                hostVortCh = torch::empty({static_cast<long>(HW)}, at::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU).pinned_memory(true));
                                devUxCh  = torch::empty({static_cast<long>(HW)}, at::TensorOptions().dtype(torch::kFloat32).device(aiDevice));
                                devUyCh  = torch::empty({static_cast<long>(HW)}, at::TensorOptions().dtype(torch::kFloat32).device(aiDevice));
                                devPCh   = torch::empty({static_cast<long>(HW)}, at::TensorOptions().dtype(torch::kFloat32).device(aiDevice));
                                devVortCh= torch::empty({static_cast<long>(HW)}, at::TensorOptions().dtype(torch::kFloat32).device(aiDevice));

                                // Indices tensor (copy from List<label> to int64 on device)
                                at::Tensor idxCpu = torch::empty({static_cast<long>(HW)}, at::TensorOptions().dtype(torch::kLong).device(torch::kCPU).pinned_memory(true));
                                int64_t* idxPtr = idxCpu.data_ptr<int64_t>();
                                for (label i = 0; i < HW; ++i) {
                                    idxPtr[i] = static_cast<int64_t>(cellIdx[i]);
                                }
                                devIdx = idxCpu.to(aiDevice, /*non_blocking=*/true);

                                lastHWDev = static_cast<long>(HW);
                                devBuffersInit = true;
                            }

                            // 3) Gather nearest directly on host (O(HW), no O(nCells) loop)
                            auto aUxCh = hostUxCh.accessor<float,1>();
                            auto aUyCh = hostUyCh.accessor<float,1>();
                            auto aPCh  = hostPCh.accessor<float,1>();
                            auto aVortCh = hostVortCh.accessor<float,1>();
                            const label nCells = pInt.size();
                            for (label i = 0; i < HW; ++i)
                            {
                                const label c = cellIdx[i];
                                if (c >= 0 && c < nCells)
                                {
                                    const vector& uVal = uInt[c];
                                    aUxCh[i] = static_cast<float>(uVal.x());
                                    aUyCh[i] = static_cast<float>(uVal.y());
                                    float pVal = static_cast<float>(pInt[c]);
                                    if (useCp)
                                    {
                                        pVal = static_cast<float>((pVal - pRef) / (0.5 * uInf * uInf));
                                    }
                                    aPCh[i]  = pVal;
                                    aVortCh[i] = useVort ? static_cast<float>(vortInt[c]) : 0.0f;
                                }
                                else
                                {
                                    aUxCh[i] = 0.0f;
                                    aUyCh[i] = 0.0f;
                                    aPCh[i]  = 0.0f;
                                    aVortCh[i] = 0.0f;
                                }
                            }

                            // 4) Move channels to CUDA (non-blocking)
                            devUxCh.copy_(hostUxCh, /*non_blocking=*/true);
                            devUyCh.copy_(hostUyCh, /*non_blocking=*/true);
                            devPCh.copy_(hostPCh,   /*non_blocking=*/true);
                            devVortCh.copy_(hostVortCh, /*non_blocking=*/true);

                            // 5) Use device channel tensors directly
                            at::Tensor chUx = devUxCh;
                            at::Tensor chUy = devUyCh;
                            at::Tensor chP  = devPCh;
                            at::Tensor chV  = devVortCh;
                            // 6) Normalize on device
                            auto toScalar = [](scalar v)->float { return static_cast<float>(v); };
                            if (normParams.method == "zscore")
                            {
                                const float e = toScalar(normParams.eps);
                                // Ux
                                const float uDen = std::max(std::abs(toScalar(normParams.Ustd)), e);
                                chUx = (chUx - toScalar(normParams.Umean)) / uDen;
                                if (normParams.zClip) chUx = chUx.clamp(toScalar(normParams.zLo), toScalar(normParams.zHi));
                                // Uy
                                const float vDen = std::max(std::abs(toScalar(normParams.Vstd)), e);
                                chUy = (chUy - toScalar(normParams.Vmean)) / vDen;
                                if (normParams.zClip) chUy = chUy.clamp(toScalar(normParams.zLo), toScalar(normParams.zHi));
                                // P
                                const float tDen = std::max(std::abs(toScalar(normParams.Tstd)), e);
                                chP = (chP - toScalar(normParams.Tmean)) / tDen;
                                if (normParams.zClip) chP = chP.clamp(toScalar(normParams.zLo), toScalar(normParams.zHi));
                                // Vorticity
                                if (useVort)
                                {
                                    const float wDen = std::max(std::abs(toScalar(normParams.Wstd)), e);
                                    chV = (chV - toScalar(normParams.Wmean)) / wDen;
                                    if (normParams.zClip) chV = chV.clamp(toScalar(normParams.zLo), toScalar(normParams.zHi));
                                }
                                else
                                {
                                    chV = torch::zeros_like(chV);
                                }
                            }
                            else
                            {
                                const float lo = toScalar(normParams.lo), hi = toScalar(normParams.hi);
                                const float loB = std::min(lo, hi), hiB = std::max(lo, hi);
                                const float e = toScalar(normParams.eps);
                                // Ux
                                {
                                    const float den = std::max(std::abs(toScalar(normParams.Umax - normParams.Umin)), e);
                                    const float a = (hi - lo)/den;
                                    const float b = lo - a*toScalar(normParams.Umin);
                                    chUx = chUx * a + b;
                                    if (normParams.clip) chUx = chUx.clamp(loB, hiB);
                                }
                                // Uy
                                {
                                    const float den = std::max(std::abs(toScalar(normParams.Vmax - normParams.Vmin)), e);
                                    const float a = (hi - lo)/den;
                                    const float b = lo - a*toScalar(normParams.Vmin);
                                    chUy = chUy * a + b;
                                    if (normParams.clip) chUy = chUy.clamp(loB, hiB);
                                }
                                // P
                                {
                                    const float den = std::max(std::abs(toScalar(normParams.Tmax - normParams.Tmin)), e);
                                    const float a = (hi - lo)/den;
                                    const float b = lo - a*toScalar(normParams.Tmin);
                                    chP = chP * a + b;
                                    if (normParams.clip) chP = chP.clamp(loB, hiB);
                                }
                                // Vorticity
                                if (useVort)
                                {
                                    const float den = std::max(std::abs(toScalar(normParams.Wmax - normParams.Wmin)), e);
                                    const float a = (hi - lo)/den;
                                    const float b = lo - a*toScalar(normParams.Wmin);
                                    chV = chV * a + b;
                                    if (normParams.clip) chV = chV.clamp(loB, hiB);
                                }
                                else
                                {
                                    chV = torch::zeros_like(chV);
                                }
                            }
                            // 7) Build input tensor [1,C,H,W], where C depends on useVort/useRe
                            at::Tensor x;
                            const float reNorm = static_cast<float>(ReValue / (ReRef != 0 ? ReRef : 1.0));
                            at::Tensor reTensor;
                            if (useRe && reMode == "film")
                            {
                                x = useVort
                                    ? torch::stack({chUx, chUy, chP, chV}, 0)
                                          .reshape({4, H, W})
                                          .unsqueeze(0)
                                          .to(aiDevice)
                                    : torch::stack({chUx, chUy, chP}, 0)
                                          .reshape({3, H, W})
                                          .unsqueeze(0)
                                          .to(aiDevice);
                                reTensor = torch::full({1}, reNorm, at::TensorOptions().dtype(torch::kFloat32).device(aiDevice));
                            }
                            else if (useRe)
                            {
                                // legacy concat mode
                                at::Tensor chRe = torch::full({static_cast<long>(HW)}, reNorm,
                                    at::TensorOptions().dtype(torch::kFloat32).device(aiDevice));
                                x = useVort
                                    ? torch::stack({chUx, chUy, chP, chV, chRe}, 0)
                                          .reshape({5, H, W})
                                          .unsqueeze(0)
                                          .to(aiDevice)
                                    : torch::stack({chUx, chUy, chP, chRe}, 0)
                                          .reshape({4, H, W})
                                          .unsqueeze(0)
                                          .to(aiDevice);
                            }
                            else
                            {
                                x = useVort
                                    ? torch::stack({chUx, chUy, chP, chV}, 0)
                                          .reshape({4, H, W})
                                          .unsqueeze(0)
                                          .to(aiDevice)
                                    : torch::stack({chUx, chUy, chP}, 0)
                                          .reshape({3, H, W})
                                          .unsqueeze(0)
                                          .to(aiDevice);
                            }
                            x = x.contiguous();

                            // --- DEBUG: log input/weights device, shape ---
                            Info<< "\n[AI DEBUG] --- Input tensor info ---" << nl;
                            Info<< "  x.device = " << x.device().str() << nl;
                            Info<< "  x.sizes  = [";
                            for (int i = 0; i < x.dim(); ++i)
                            {
                                Info<< x.size(i);
                                if (i < x.dim()-1) Info<< ", ";
                            }
                            Info<< "]" << nl;

                            bool firstLogged = false;
                            for (const at::Tensor& w0 : aiModule.parameters())
                            {
                                Info<< "[AI DEBUG] --- First weight tensor info ---" << nl;
                                Info<< "  w0.device = " << w0.device().str() << nl;
                                Info<< "  w0.sizes  = [";
                                for (int i = 0; i < w0.dim(); ++i)
                                {
                                    Info<< w0.size(i);
                                    if (i < w0.dim()-1) Info<< ", ";
                                }
                                Info<< "]" << nl;
                                firstLogged = true;
                                break;
                            }
                            Info<< "[AI DEBUG] torch::cuda::is_available() = "
                                 << (torch::cuda::is_available() ? "true" : "false") << nl;
                            Info<< "[AI DEBUG] aiDevice = " << aiDevice.str() << nl;
                            Info<< "--------------------------------------" << nl << endl;
                            // 8) Forward
                            std::vector<torch::jit::IValue> inputs;
                            inputs.push_back(x);
                            if (useRe && reMode == "film")
                            {
                                inputs.push_back(reTensor);
                            }
                            auto output = aiModule.forward(inputs);
                            at::Tensor attn;
                            if (output.isTuple()) {
                                auto tuple_outputs = output.toTuple();
                                if (refineMode == "delta_norm") {
                                    // Ablation: use mean head L2 norm across channels
                                    at::Tensor meanOut = tuple_outputs->elements()[0].toTensor();
                                    // meanOut shape: [B, C, H, W]; compute sqrt(sum_c x^2)
                                    attn = torch::sqrt(meanOut.pow(2).sum(/*dim=*/1, /*keepdim=*/true));
                                } else {
                                    // Default: use log-variance head
                                    attn = tuple_outputs->elements()[1].toTensor();
                                }
                            } else if (output.isTensor()) {
                                attn = output.toTensor();
                            } else {
                                FatalErrorInFunction << "Unexpected model output type" << exit(FatalError);
                            }
                            // 9) Post-process attention
                            attn = attn.to(torch::kCPU).contiguous();
                            if (attn.dim() == 4) attn = attn.squeeze(0).squeeze(0);
                            else if (attn.dim() == 3) attn = attn.squeeze(0);
                            if (attn.numel() != HW) {
                                FatalErrorInFunction << "Attention tensor size mismatch. Expected " << HW 
                                                     << ", got " << attn.numel() << exit(FatalError);
                            }
                            attn = attn.reshape({HW});
                            const float* aPtr = attn.data_ptr<float>();
                            scalarField attnImg(HW);
                            for (label i = 0; i < HW; ++i) attnImg[i] = static_cast<scalar>(aPtr[i]);

                            if (uncStats)
                            {
                                scalar uMin = GREAT, uMax = -GREAT, uSum = 0.0;
                                forAll(attnImg, i)
                                {
                                    scalar v = attnImg[i];
                                    if (uncUseSigma)
                                    {
                                        v = Foam::sqrt(Foam::exp(v));
                                    }
                                    uMin = min(uMin, v);
                                    uMax = max(uMax, v);
                                    uSum += v;
                                }
                                Info<< "Unc stats: min=" << uMin
                                     << ", max=" << uMax
                                     << ", mean=" << (uSum/attnImg.size()) << endl;
                            }

                            // Generate refineFlag
                            Info<< "Generating refineFlag field..." << endl;
                            makeAndWriteRefineFlag(mesh, attnImg, ggIdx, NX, NY,
                                                   attnThr, noRefineRadius, noRefineShape,
                                                   noRefineCenter, noRefineMin, noRefineMax,
                                                   refineFlag, thrGreaterEqual);
                            // Ensure binary values (0/1) on current mesh
                            {
                                scalarField& rf = refineFlag.primitiveFieldRef();
                                forAll(rf, i) rf[i] = (rf[i] > 0.5 ? 1.0 : 0.0);
                                refineFlag.correctBoundaryConditions();
                                scalar rMin = GREAT, rMax = -GREAT, rSum = 0.0;
                                forAll(rf, i) { rMin = min(rMin, rf[i]); rMax = max(rMax, rf[i]); rSum += rf[i]; }
                                Info<< "refineFlag stats (pre-update): min=" << rMin
                                     << ", max=" << rMax
                                     << ", mean=" << (rSum/rf.size()) << endl;
                            }
                        }
                        else
                        {
                            // Fallback: CPU sampling + CPU normalization + CPU inference (existing path)
                            Info<< "Sampling fields (" << NX << "x" << NY << ")..." << endl;
                            GridSample gs = useRegion
                                ? samplePUtoGrid(mesh, p, U, vort, NX, NY, regionBb, sampleNearest)
                                : samplePUtoGrid(mesh, p, U, vort, NX, NY, sampleNearest);

                            if (useCp)
                            {
                                forAll(gs.T, i)
                                {
                                    gs.T[i] = (gs.T[i] - pRef) / (0.5 * uInf * uInf);
                                }
                            }
                            if (!useVort)
                            {
                                gs.Vort = 0.0;
                            }
                            
                            Info<< "Normalizing (" << normParams.method << ")..." << endl;
                            if (normParams.method == "zscore")
                            {
                                normalizeZScoreInPlace(gs.T,  normParams.Tmean, normParams.Tstd,
                                                       normParams.eps, normParams.zClip,
                                                       normParams.zLo, normParams.zHi);
                                normalizeZScoreInPlace(gs.Ux, normParams.Umean, normParams.Ustd,
                                                       normParams.eps, normParams.zClip,
                                                       normParams.zLo, normParams.zHi);
                                normalizeZScoreInPlace(gs.Uy, normParams.Vmean, normParams.Vstd,
                                                       normParams.eps, normParams.zClip,
                                                       normParams.zLo, normParams.zHi);
                                if (useVort)
                                {
                                    normalizeZScoreInPlace(gs.Vort, normParams.Wmean, normParams.Wstd,
                                                           normParams.eps, normParams.zClip,
                                                           normParams.zLo, normParams.zHi);
                                }
                            }
                            else
                            {
                                normalizeMinMaxInPlace(gs.T,  normParams.Tmin, normParams.Tmax, 
                                                      normParams.lo, normParams.hi, 
                                                      normParams.eps, normParams.clip);
                                normalizeMinMaxInPlace(gs.Ux, normParams.Umin, normParams.Umax, 
                                                      normParams.lo, normParams.hi, 
                                                      normParams.eps, normParams.clip);
                                normalizeMinMaxInPlace(gs.Uy, normParams.Vmin, normParams.Vmax, 
                                                      normParams.lo, normParams.hi, 
                                                      normParams.eps, normParams.clip);
                                if (useVort)
                                {
                                    normalizeMinMaxInPlace(gs.Vort, normParams.Wmin, normParams.Wmax, 
                                                          normParams.lo, normParams.hi, 
                                                          normParams.eps, normParams.clip);
                                }
                            }
                            Info<< "Running neural network (CPU path)..." << endl;
                            // Ensure model on CPU for this path
                            aiModule.to(at::Device(torch::kCPU));
                            const float reNorm = static_cast<float>(ReValue / (ReRef != 0 ? ReRef : 1.0));
                            scalarField attnImg = inferAttention(aiModule, gs, useVort, useRe, reNorm, useRe && reMode == "film", refineMode);
                            if (uncStats)
                            {
                                scalar uMin = GREAT, uMax = -GREAT, uSum = 0.0;
                                forAll(attnImg, i)
                                {
                                    scalar v = attnImg[i];
                                    if (uncUseSigma)
                                    {
                                        v = Foam::sqrt(Foam::exp(v));
                                    }
                                    uMin = min(uMin, v);
                                    uMax = max(uMax, v);
                                    uSum += v;
                                }
                                Info<< "Unc stats: min=" << uMin
                                     << ", max=" << uMax
                                     << ", mean=" << (uSum/attnImg.size()) << endl;
                            }
                            Info<< "Generating refineFlag field..." << endl;
                            makeAndWriteRefineFlag(mesh, attnImg, gs.geom, NX, NY,
                                                   attnThr, noRefineRadius, noRefineShape,
                                                   noRefineCenter, noRefineMin, noRefineMax,
                                                   refineFlag, thrGreaterEqual);
                            // Ensure binary values (0/1) on current mesh
                            {
                                scalarField& rf = refineFlag.primitiveFieldRef();
                                forAll(rf, i) rf[i] = (rf[i] > 0.5 ? 1.0 : 0.0);
                                refineFlag.correctBoundaryConditions();
                                scalar rMin = GREAT, rMax = -GREAT, rSum = 0.0;
                                forAll(rf, i) { rMin = min(rMin, rf[i]); rMax = max(rMax, rf[i]); rSum += rf[i]; }
                                Info<< "refineFlag stats (pre-update): min=" << rMin
                                     << ", max=" << rMax
                                     << ", mean=" << (rSum/rf.size()) << endl;
                            }
                        }

                        // (removed duplicate refineFlag generation)
                        
                        Info<< ">>> REFINEFLAG READY FOR MESH REFINEMENT <<<\n" << endl;
                        
                    } catch (const std::exception& e) {
                        WarningInFunction
                            << "AI refinement failed: " << e.what()
                            << "\nUsing zero refineFlag" << endl;
                        refineFlag = dimensionedScalar("zero", dimless, 0.0);
                        refineFlag.write();
                    }
                }
                
                // Now perform mesh changes (this will use the refineFlag we just generated)
                mesh.controlledUpdate();

                if (mesh.changing())
                {
                    Info<< "Mesh topology change detected" << endl;
                    // Keep refineFlag binary after mesh change for clean visualization
                    {
                        scalarField& rf = refineFlag.primitiveFieldRef();
                        forAll(rf, i) rf[i] = (rf[i] > 0.5 ? 1.0 : 0.0);
                        refineFlag.correctBoundaryConditions();
                    }
                    
                    MRF.update();

                    if (correctPhi)
                    {
                        phi = mesh.Sf() & Uf();
                        #include "correctPhi.H"
                        fvc::makeRelative(phi, U);
                    }

                    if (checkMeshCourantNo)
                    {
                        #include "meshCourantNo.H"
                    }
                    
                    // Do not reset refineFlag here; keep values for write/debug visibility
                }
            }

            #include "UEqn.H"

            while (pimple.correct())
            {
                #include "pEqn.H"
            }

            if (pimple.turbCorr())
            {
                laminarTransport.correct();
                turbulence->correct();
            }
        }

        runTime.write();
        runTime.printExecutionTime(Info);
    }

    Info<< "End\n" << endl;

    return 0;
}

// ************************************************************************* //
