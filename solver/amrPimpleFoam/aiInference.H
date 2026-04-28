#pragma once
#include <torch/script.h>
#include "fvCFD.H"
#include "Resize.H"
#include <cmath>

namespace Foam {

// Safe bilinear sampling
inline scalar bilinearSample(
    const scalarField& img,
    const GridGeom& g,
    const point& p
)
{
    // Compute continuous grid coordinates
    scalar u = (g.dx > VSMALL ? (p.x() - g.x0)/g.dx : 0.0);
    scalar v = (g.dy > VSMALL ? (p.y() - g.y0)/g.dy : 0.0);
    
    // Clamp to grid bounds
    u = Foam::clamp(u, 0.0, scalar(g.nx - 1));
    v = Foam::clamp(v, 0.0, scalar(g.ny - 1));
    
    const label i0 = static_cast<label>(std::floor(u));
    const label j0 = static_cast<label>(std::floor(v));
    const label i1 = min(i0 + 1, g.nx - 1);
    const label j1 = min(j0 + 1, g.ny - 1);
    
    const scalar fu = u - scalar(i0);
    const scalar fv = v - scalar(j0);
    
    const auto idx = [nx=g.nx](label i, label j) { return j*nx + i; };
    
    const scalar v00 = img[idx(i0, j0)];
    const scalar v10 = img[idx(i1, j0)];
    const scalar v01 = img[idx(i0, j1)];
    const scalar v11 = img[idx(i1, j1)];
    
    return (1 - fu)*(1 - fv)*v00
         +      fu *(1 - fv)*v10
         + (1 - fu)*     fv *v01
         +      fu *     fv *v11;
}

// Convert grid sample to PyTorch tensor
inline at::Tensor gridToTensorCHW(
    const GridSample& gs, const label H, const label W,
    const bool useVort = true,
    const bool useRe = false,
    const float reNorm = 1.0f
)
{
    const long baseC = useVort ? 4 : 3;
    const long C = useRe ? (baseC + 1) : baseC;
    Info<< "Creating tensor of size [1, " << C << ", " << H << ", " << W << "]" << endl;
    
    // Create new tensor each time (safer than reusing)
    at::Tensor x = torch::zeros({1, C, H, W}, 
        at::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
    
    // Access tensor data
    auto x_a = x.accessor<float, 4>(); // [b,c,h,w]
    
    for (label j = 0; j < H; ++j) {
        for (label i = 0; i < W; ++i) {
            const label k = j*W + i;
            x_a[0][0][j][i] = static_cast<float>(gs.Ux[k]);   // channel 0: u
            x_a[0][1][j][i] = static_cast<float>(gs.Uy[k]);   // channel 1: v
            x_a[0][2][j][i] = static_cast<float>(gs.T[k]);    // channel 2: p or cp
            if (useVort)
            {
                x_a[0][3][j][i] = static_cast<float>(gs.Vort[k]); // channel 3: vorticity
            }
            if (useRe)
            {
                x_a[0][baseC][j][i] = reNorm;               // extra Re channel
            }
        }
    }
    
    Info<< "Tensor created successfully" << endl;
    return x;
}

// Perform attention inference
inline scalarField inferAttention(
    torch::jit::script::Module& module,
    const GridSample& gs,
    const bool useVort = true,
    const bool useRe = false,
    const float reNorm = 1.0f,
    const bool reAsScalar = false,
    const Foam::word& refineMode = Foam::word("sigma")
)
{
    const label H = gs.geom.ny;
    const label W = gs.geom.nx;
    const label HW = H * W;
    
    Info<< "Starting AI inference for grid " << W << "x" << H << endl;
    
    try {
        // Create input tensor
        at::Tensor x = gridToTensorCHW(gs, H, W, useVort, useRe, reNorm);
        
        // Verify tensor shape
        Info<< "Input tensor shape: [" 
             << x.size(0) << ", " << x.size(1) << ", " 
             << x.size(2) << ", " << x.size(3) << "]" << endl;
        
        // Perform inference
        Info<< "Running model forward pass..." << endl;
        at::Tensor attn;
        {
            torch::NoGradGuard noGrad;
            
            // Call model
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(x);
            if (useRe && reAsScalar)
            {
                inputs.push_back(torch::full({1}, reNorm, at::TensorOptions().dtype(torch::kFloat32).device(x.device())));
            }
            
            auto output = module.forward(inputs);
            
            // Handle output based on model architecture
            if (output.isTuple()) {
                auto tuple_outputs = output.toTuple();
                Info<< "Model returned tuple with "
                     << tuple_outputs->elements().size() << " elements" << endl;

                if (refineMode == "delta_norm") {
                    at::Tensor meanOut = tuple_outputs->elements()[0].toTensor();
                    attn = torch::sqrt(meanOut.pow(2).sum(/*dim=*/1, /*keepdim=*/true));
                } else {
                    attn = tuple_outputs->elements()[1].toTensor();
                }
            } else if (output.isTensor()) {
                Info<< "Model returned single tensor" << endl;
                attn = output.toTensor();
            } else {
                FatalErrorInFunction
                    << "Unexpected model output type"
                    << exit(FatalError);
            }
        }
        
        Info<< "Model inference complete" << endl;
        
        // Process attention tensor
        Info<< "Attention tensor shape before processing: [";
        for (int64_t i = 0; i < attn.dim(); ++i) {
            Info<< attn.size(i);
            if (i < attn.dim() - 1) Info<< ", ";
        }
        Info<< "]" << endl;
        
        // Move to CPU and reshape
        attn = attn.to(torch::kCPU).contiguous();
        
        // Reshape to (H*W) if needed
        if (attn.dim() == 4) {
            // (1, 1, H, W) -> (H, W)
            attn = attn.squeeze(0).squeeze(0);
        } else if (attn.dim() == 3) {
            // (1, H, W) -> (H, W)
            attn = attn.squeeze(0);
        }
        
        // Ensure correct shape
        if (attn.numel() != HW) {
            FatalErrorInFunction
                << "Attention tensor size mismatch. Expected " << HW 
                << " elements, got " << attn.numel()
                << exit(FatalError);
        }
        
        // Convert to scalarField
        attn = attn.reshape({HW});
        const float* aPtr = attn.data_ptr<float>();
        
        scalarField attnImg(HW);
        for (label i = 0; i < HW; ++i) {
            attnImg[i] = static_cast<scalar>(aPtr[i]);
        }
        
        // Debug: print statistics
        scalar aMin = GREAT, aMax = -GREAT, aSum = 0.0;
        forAll(attnImg, i) {
            aMin = min(aMin, attnImg[i]);
            aMax = max(aMax, attnImg[i]);
            aSum += attnImg[i];
        }
        Info<< "Attention map stats: min=" << aMin 
             << ", max=" << aMax 
             << ", mean=" << (aSum/HW) << endl;
        
        return attnImg;
        
    } catch (const c10::Error& e) {
        FatalErrorInFunction
            << "PyTorch error during inference: " << e.what()
            << exit(FatalError);
    } catch (const std::exception& e) {
        FatalErrorInFunction
            << "Standard exception during inference: " << e.what()
            << exit(FatalError);
    }
    
    // Should never reach here
    return scalarField(HW, 0.0);
}

// Generate refineFlag field
inline void makeAndWriteRefineFlag(
    const fvMesh& mesh,
    const scalarField& attnImg,
    const GridGeom& geom,
    const label nx, const label ny,
    const scalar thr,
    const scalar noRefineRadius,
    const word& noRefineShape,
    const vector& noRefineCenter,
    const vector& noRefineMin,
    const vector& noRefineMax,
    Foam::volScalarField& refineFlag,
    const bool thrGE
)
{
    Info<< "Generating refineFlag field with threshold " << thr << endl;
    
    scalarField& rf = refineFlag.primitiveFieldRef();
    const vectorField& Cc = mesh.C();
    const boundBox& regionBb = geom.bb;
    
    // Set refine flag based on attention
    forAll(rf, cellI) {
        const point p = Cc[cellI];
        if (regionBb.contains(p)) {
            const scalar a = bilinearSample(attnImg, geom, p);
            rf[cellI] = (thrGE ? (a >= thr) : (a <= thr)) ? 1.0 : 0.0;
        } else {
            // Outside the user-specified extraction region: force 1
            rf[cellI] = 0.0;
        }
        // Force no-refine inside user-defined region (circle or rectangle)
        if (noRefineRadius > 0 || noRefineShape.size())
        {
            const word shape = toLowerWord(noRefineShape.size() ? noRefineShape : word("circle"));
            if (shape == "circle")
            {
                const scalar dx = p.x() - noRefineCenter.x();
                const scalar dy = p.y() - noRefineCenter.y();
                const scalar r2 = dx*dx + dy*dy;
                if (r2 <= noRefineRadius*noRefineRadius)
                {
                    rf[cellI] = 0.0;
                }
            }
            else if (shape == "rect" || shape == "rectangle" || shape == "box")
            {
                if (p.x() >= noRefineMin.x() && p.x() <= noRefineMax.x() &&
                    p.y() >= noRefineMin.y() && p.y() <= noRefineMax.y())
                {
                    rf[cellI] = 0.0;
                }
            }
        }
    }
    
    // Count cells to be refined
    label refineCount = 0;
    forAll(rf, cellI) {
        if (rf[cellI] > 0.5) refineCount++;
    }
    Info<< "Initial cells marked for refinement: " << refineCount << endl;
    
    // Expand refinement region by rings
	// (removed unused owner/neighbour/nInt to avoid warnings)
    
    // Final count
    refineCount = 0;
    forAll(rf, cellI) {
        if (rf[cellI] > 0.5) refineCount++;
    }
    Info<< "Final cells marked for refinement (after ring expansion): " << refineCount << endl;
}

// Load model from file (for compatibility)
inline scalarField inferAttention(
    const fileName& modelPath,
    const GridSample& gs
)
{
    torch::jit::script::Module module = torch::jit::load(modelPath.c_str(), torch::kCPU);
    module.eval();
    return inferAttention(module, gs);
}

}
