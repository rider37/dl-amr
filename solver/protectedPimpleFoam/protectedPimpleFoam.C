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

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

Application
    pimpleFoam.C

Group
    grpIncompressibleSolvers

Description
    Transient solver for incompressible, turbulent flow of Newtonian fluids
    on a moving mesh.

    \heading Solver details
    The solver uses the PIMPLE (merged PISO-SIMPLE) algorithm to solve the
    continuity equation:

        \f[
            \div \vec{U} = 0
        \f]

    and momentum equation:

        \f[
            \ddt{\vec{U}} + \div \left( \vec{U} \vec{U} \right) - \div \gvec{R}
          = - \grad p + \vec{S}_U
        \f]

    Where:
    \vartable
        \vec{U} | Velocity
        p       | Pressure
        \vec{R} | Stress tensor
        \vec{S}_U | Momentum source
    \endvartable

    Sub-models include:
    - turbulence modelling, i.e. laminar, RAS or LES
    - run-time selectable MRF and finite volume options, e.g. explicit porosity

    \heading Required fields
    \plaintable
        U       | Velocity [m/s]
        p       | Kinematic pressure, p/rho [m2/s2]
        \<turbulence fields\> | As required by user selection
    \endplaintable

Note
   The motion frequency of this solver can be influenced by the presence
   of "updateControl" and "updateInterval" in the dynamicMeshDict.

\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "dynamicFvMesh.H"
#include "singlePhaseTransportModel.H"
#include "turbulentTransportModel.H"
#include "pimpleControl.H"
#include "CorrectPhi.H"
#include "fvOptions.H"
#include "localEulerDdtScheme.H"
#include "fvcSmooth.H"
#include "dynamicRefineFvMesh.H"
#include "cellSet.H"


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    argList::addNote
    (
        "Transient solver for incompressible, turbulent flow"
        " of Newtonian fluids on a moving mesh."
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

    Info<< "\nLoading protected cells for dynamic mesh\n" << endl;
    // Get a reference to the protectedCell bitSet from the mesh object
    bitSet& protectedCell = refCast<dynamicRefineFvMesh>(mesh).protectedCell();
    // Initialise the set if it's empty
    if (protectedCell.empty())
    {
        protectedCell.setSize(mesh.nCells());
        protectedCell = false; // Set all cells to 'not protected' by default
    }

    // Read dynamicMeshDict for user-defined protection region
    IOdictionary dyMDict
    (
        IOobject
        (
            "dynamicMeshDict",
            runTime.constant(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );

    const dictionary& refineDict = dyMDict.found("dynamicRefineFvMeshCoeffs")
        ? dyMDict.subDict("dynamicRefineFvMeshCoeffs")
        : dyMDict;

    auto toLowerWord = [](const word& w)
    {
        Foam::string s(w);
        for (size_t i = 0; i < s.size(); ++i)
        {
            unsigned char c = static_cast<unsigned char>(s[i]);
            s[i] = static_cast<char>(std::tolower(c));
        }
        return word(s);
    };

    word protectMode("cellset"); // cellset | circle | rect | none
    if (refineDict.found("protectMode"))
    {
        protectMode = word(refineDict.lookup("protectMode"));
    }
    else if (refineDict.found("protectShape"))
    {
        protectMode = word(refineDict.lookup("protectShape"));
    }
    protectMode = toLowerWord(protectMode);

    bool combineCellSet = false;
    if (refineDict.found("protectCombineCellSet"))
    {
        const Switch sw(refineDict.lookup("protectCombineCellSet"));
        combineCellSet = bool(sw);
    }

    const vectorField& Cc = mesh.C();
    label protectCount = 0;

    // Apply user-defined region
    if (protectMode == "circle")
    {
        if (!refineDict.found("protectCenter") || !refineDict.found("protectRadius"))
        {
            WarningInFunction << "protectMode=circle but protectCenter/protectRadius missing. Falling back to cellSet."
                              << endl;
            protectMode = "cellset";
        }
        else
        {
            const vector center = vector(refineDict.lookup("protectCenter"));
            const scalar radius = readScalar(refineDict.lookup("protectRadius"));
            const scalar r2 = radius*radius;
            forAll(Cc, celli)
            {
                const vector d = Cc[celli] - center;
                if ((d.x()*d.x() + d.y()*d.y()) <= r2)
                {
                    protectedCell[celli] = 1;
                    protectCount++;
                }
            }
            Info<< "Protected cells (circle): " << protectCount << nl;
        }
    }
    else if (protectMode == "rect" || protectMode == "rectangle" || protectMode == "box")
    {
        if (!refineDict.found("protectMin") || !refineDict.found("protectMax"))
        {
            WarningInFunction << "protectMode=rect but protectMin/protectMax missing. Falling back to cellSet."
                              << endl;
            protectMode = "cellset";
        }
        else
        {
            const vector rmin = vector(refineDict.lookup("protectMin"));
            const vector rmax = vector(refineDict.lookup("protectMax"));
            forAll(Cc, celli)
            {
                const vector p = Cc[celli];
                if (p.x() >= rmin.x() && p.x() <= rmax.x() &&
                    p.y() >= rmin.y() && p.y() <= rmax.y())
                {
                    protectedCell[celli] = 1;
                    protectCount++;
                }
            }
            Info<< "Protected cells (rect): " << protectCount << nl;
        }
    }
    else if (protectMode == "none")
    {
        Info<< "Protection disabled (protectMode=none)." << nl;
    }

    // Optionally load cellSet (default or combined)
    if (protectMode == "cellset" || combineCellSet)
    {
        const fileName setsDir = runTime.constant()/"polyMesh"/"sets";
        const fileName setPath = setsDir/"protectedCells";
        if (isFile(setPath))
        {
            cellSet protectedCells
            (
                IOobject
                (
                    "protectedCells",
                    "constant/polyMesh/sets",
                    mesh,
                    IOobject::MUST_READ,
                    IOobject::NO_WRITE
                )
            );
            forAll(protectedCells.toc(), i)
            {
                label celli = protectedCells.toc()[i];
                if (celli < protectedCell.size())
                {
                    if (!protectedCell[celli]) protectCount++;
                    protectedCell[celli] = 1;
                }
            }
            Info<< "Protected cells (cellSet): " << protectedCells.size() << nl;
        }
        else if (protectMode == "cellset")
        {
            WarningInFunction << "protectedCells set not found at " << setPath
                              << " (no protection applied)." << endl;
        }
    }

    Info<< "Total protected cells: " << protectCount << nl;

    Info<< "\nStarting time loop\n" << endl;

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
                // Do any mesh changes
                mesh.controlledUpdate();

                if (mesh.changing())
                {
                    MRF.update();

                    if (correctPhi)
                    {
                        // Calculate absolute flux
                        // from the mapped surface velocity
                        phi = mesh.Sf() & Uf();

                        #include "correctPhi.H"

                        // Make the flux relative to the mesh motion
                        fvc::makeRelative(phi, U);
                    }

                    if (checkMeshCourantNo)
                    {
                        #include "meshCourantNo.H"
                    }
                }
            }

            #include "UEqn.H"

            // --- Pressure corrector loop
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
