/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2011-2017 OpenFOAM Foundation
    Copyright (C) 2016-2023 OpenCFD Ltd.
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

\*---------------------------------------------------------------------------*/

#include "hexRef4.H"

#include "polyMesh.H"
#include "polyTopoChange.H"
#include "meshTools.H"
#include "polyAddFace.H"
#include "polyAddPoint.H"
#include "polyAddCell.H"
#include "polyModifyFace.H"
#include "syncTools.H"
#include "faceSet.H"
#include "cellSet.H"
#include "pointSet.H"
#include "labelPairHashes.H"
#include "OFstream.H"
#include "Time.H"
#include "FaceCellWave.H"
#include "mapDistributePolyMesh.H"
#include "refinementData.H"
#include "refinementDistanceData.H"
#include "degenerateMatcher.H"
#include "emptyPolyPatch.H"

//#include "fvMesh.H"
//#include "volFields.H"
//#include "OBJstream.H"

// OpenFOAM v2312 backport: syncTools::swapBoundaryCellList<refinementData>
// instantiates transformList, which needs a transform() overload for the
// element type. refinementData carries two labels only (refinementCount_,
// count_), so rotation across coupled transforms is the identity.
namespace Foam
{
    inline refinementData transform
    (
        const tensor&,
        const refinementData& rd
    )
    {
        return rd;
    }
}

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(hexRef4, 0);

    //- Reduction class. If x and y are not equal assign value.
    template<label value>
    struct ifEqEqOp
    {
        void operator()(label& x, const label y) const
        {
            x = (x == y) ? x : value;
        }
    };
}


// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

void Foam::hexRef4::reorder
(
    const labelList& map,
    const label len,
    const label null,
    labelList& elems
)
{
    labelList newElems(len, null);

    forAll(elems, i)
    {
        label newI = map[i];

        if (newI >= len)
        {
            FatalErrorInFunction << abort(FatalError);
        }

        if (newI >= 0)
        {
            newElems[newI] = elems[i];
        }
    }

    elems.transfer(newElems);
}


void Foam::hexRef4::getFaceInfo
(
    const label facei,
    label& patchID,
    label& zoneID,
    label& zoneFlip
) const
{
    patchID = -1;

    if (!mesh_.isInternalFace(facei))
    {
        patchID = mesh_.boundaryMesh().whichPatch(facei);
    }

    zoneID = mesh_.faceZones().whichZone(facei);

    zoneFlip = false;

    if (zoneID >= 0)
    {
        const faceZone& fZone = mesh_.faceZones()[zoneID];

        zoneFlip = fZone.flipMap()[fZone.whichFace(facei)];
    }
}


// Adds a face on top of existing facei.
Foam::label Foam::hexRef4::addFace
(
    polyTopoChange& meshMod,
    const label facei,
    const face& newFace,
    const label own,
    const label nei
) const
{
    label patchID, zoneID, zoneFlip;

    getFaceInfo(facei, patchID, zoneID, zoneFlip);

    label newFacei = -1;

    if ((nei == -1) || (own < nei))
    {
        // Ordering ok.
        newFacei = meshMod.setAction
        (
            polyAddFace
            (
                newFace,                    // face
                own,                        // owner
                nei,                        // neighbour
                -1,                         // master point
                -1,                         // master edge
                facei,                      // master face for addition
                false,                      // flux flip
                patchID,                    // patch for face
                zoneID,                     // zone for face
                zoneFlip                    // face zone flip
            )
        );
    }
    else
    {
        // Reverse owner/neighbour
        newFacei = meshMod.setAction
        (
            polyAddFace
            (
                newFace.reverseFace(),      // face
                nei,                        // owner
                own,                        // neighbour
                -1,                         // master point
                -1,                         // master edge
                facei,                      // master face for addition
                false,                      // flux flip
                patchID,                    // patch for face
                zoneID,                     // zone for face
                zoneFlip                    // face zone flip
            )
        );
    }
    return newFacei;
}


// Adds an internal face from an edge. Assumes orientation correct.
// Problem is that the face is between four new vertices. So what do we provide
// as master? The only existing mesh item we have is the edge we have split.
// Have to be careful in only using it if it has internal faces since otherwise
// polyMeshMorph will complain (because it cannot generate a sensible mapping
// for the face)
Foam::label Foam::hexRef4::addInternalFace
(
    polyTopoChange& meshMod,
    const label meshFacei,
    const label meshPointi,
    const face& newFace,
    const label own,
    const label nei
) const
{
    if (mesh_.isInternalFace(meshFacei))
    {
        return meshMod.setAction
        (
            polyAddFace
            (
                newFace,                    // face
                own,                        // owner
                nei,                        // neighbour
                -1,                         // master point
                -1,                         // master edge
                -1,                         // master face for addition
                false,                      // flux flip
                -1,                         // patch for face
                -1,                         // zone for face
                false                       // face zone flip
            )
        );
    }
    else
    {
        // Two choices:
        // - append (i.e. create out of nothing - will not be mapped)
        //   problem: field does not get mapped.
        // - inflate from point.
        //   problem: does interpolative mapping which constructs full
        //   volPointInterpolation!

        // For now create out of nothing

        return meshMod.setAction
        (
            polyAddFace
            (
                newFace,                    // face
                own,                        // owner
                nei,                        // neighbour
                -1,                         // master point
                -1,                         // master edge
                -1,                         // master face for addition
                false,                      // flux flip
                -1,                         // patch for face
                -1,                         // zone for face
                false                       // face zone flip
            )
        );


        ////- Inflate-from-point:
        //// Check if point has any internal faces we can use.
        //label masterPointi = -1;
        //
        //const labelList& pFaces = mesh_.pointFaces()[meshPointi];
        //
        //forAll(pFaces, i)
        //{
        //    if (mesh_.isInternalFace(pFaces[i]))
        //    {
        //        // meshPoint uses internal faces so ok to inflate from it
        //        masterPointi = meshPointi;
        //
        //        break;
        //    }
        //}
        //
        //return meshMod.setAction
        //(
        //    polyAddFace
        //    (
        //        newFace,                    // face
        //        own,                        // owner
        //        nei,                        // neighbour
        //        masterPointi,               // master point
        //        -1,                         // master edge
        //        -1,                         // master face for addition
        //        false,                      // flux flip
        //        -1,                         // patch for face
        //        -1,                         // zone for face
        //        false                       // face zone flip
        //    )
        //);
    }
}


// Modifies existing facei for either new owner/neighbour or new face points.
void Foam::hexRef4::modFace
(
    polyTopoChange& meshMod,
    const label facei,
    const face& newFace,
    const label own,
    const label nei
) const
{
    label patchID, zoneID, zoneFlip;

    getFaceInfo(facei, patchID, zoneID, zoneFlip);

    if
    (
        (own != mesh_.faceOwner()[facei])
     || (
            mesh_.isInternalFace(facei)
         && (nei != mesh_.faceNeighbour()[facei])
        )
     || (newFace != mesh_.faces()[facei])
    )
    {
        if ((nei == -1) || (own < nei))
        {
            meshMod.setAction
            (
                polyModifyFace
                (
                    newFace,            // modified face
                    facei,              // label of face being modified
                    own,                // owner
                    nei,                // neighbour
                    false,              // face flip
                    patchID,            // patch for face
                    false,              // remove from zone
                    zoneID,             // zone for face
                    zoneFlip            // face flip in zone
                )
            );
        }
        else
        {
            meshMod.setAction
            (
                polyModifyFace
                (
                    newFace.reverseFace(),  // modified face
                    facei,                  // label of face being modified
                    nei,                    // owner
                    own,                    // neighbour
                    false,                  // face flip
                    patchID,                // patch for face
                    false,                  // remove from zone
                    zoneID,                 // zone for face
                    zoneFlip                // face flip in zone
                )
            );
        }
    }
}


// Bit complex way to determine the unrefined edge length.
Foam::scalar Foam::hexRef4::getLevel0EdgeLength() const
{
    if (cellLevel_.size() != mesh_.nCells())
    {
        FatalErrorInFunction
            << "Number of cells in mesh:" << mesh_.nCells()
            << " does not equal size of cellLevel:" << cellLevel_.size()
            << endl
            << "This might be because of a restart with inconsistent cellLevel."
            << abort(FatalError);
    }

    // Determine minimum edge length per refinement level
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    const scalar GREAT2 = sqr(GREAT);

    label nLevels = gMax(cellLevel_)+1;

    scalarField typEdgeLenSqr(nLevels, GREAT2);


    // 1. Look only at edges surrounded by cellLevel cells only.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    {
        // Per edge the cellLevel of connected cells. -1 if not set,
        // labelMax if different levels, otherwise levels of connected cells.
        labelList edgeLevel(mesh_.nEdges(), -1);

        forAll(cellLevel_, celli)
        {
            const label cLevel = cellLevel_[celli];

            const labelList& cEdges = mesh_.cellEdges(celli);

            forAll(cEdges, i)
            {
                label edgeI = cEdges[i];

                if (edgeLevel[edgeI] == -1)
                {
                    edgeLevel[edgeI] = cLevel;
                }
                else if (edgeLevel[edgeI] == labelMax)
                {
                    // Already marked as on different cellLevels
                }
                else if (edgeLevel[edgeI] != cLevel)
                {
                    edgeLevel[edgeI] = labelMax;
                }
            }
        }

        // Make sure that edges with different levels on different processors
        // are also marked. Do the same test (edgeLevel != cLevel) on coupled
        // edges.
        syncTools::syncEdgeList
        (
            mesh_,
            edgeLevel,
            ifEqEqOp<labelMax>(),
            labelMin
        );

        // Now use the edgeLevel with a valid value to determine the
        // length per level.
        forAll(edgeLevel, edgeI)
        {
            const label eLevel = edgeLevel[edgeI];

            if (eLevel >= 0 && eLevel < labelMax)
            {
                const edge& e = mesh_.edges()[edgeI];

                scalar edgeLenSqr = e.magSqr(mesh_.points());

                typEdgeLenSqr[eLevel] = min(typEdgeLenSqr[eLevel], edgeLenSqr);
            }
        }
    }

    // Get the minimum per level over all processors. Note minimum so if
    // cells are not cubic we use the smallest edge side.
    Pstream::listCombineReduce(typEdgeLenSqr, minEqOp<scalar>());

    if (debug)
    {
        Pout<< "hexRef4::getLevel0EdgeLength() :"
            << " After phase1: Edgelengths (squared) per refinementlevel:"
            << typEdgeLenSqr << endl;
    }


    // 2. For any levels where we haven't determined a valid length yet
    //    use any surrounding cell level. Here we use the max so we don't
    //    pick up levels between celllevel and higher celllevel (will have
    //    edges sized according to highest celllevel)
    //    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    scalarField maxEdgeLenSqr(nLevels, -GREAT2);

    forAll(cellLevel_, celli)
    {
        const label cLevel = cellLevel_[celli];

        const labelList& cEdges = mesh_.cellEdges(celli);

        forAll(cEdges, i)
        {
            const edge& e = mesh_.edges()[cEdges[i]];

            scalar edgeLenSqr = e.magSqr(mesh_.points());

            maxEdgeLenSqr[cLevel] = max(maxEdgeLenSqr[cLevel], edgeLenSqr);
        }
    }

    Pstream::listCombineReduce(maxEdgeLenSqr, maxEqOp<scalar>());

    if (debug)
    {
        Pout<< "hexRef4::getLevel0EdgeLength() :"
            << " Poor Edgelengths (squared) per refinementlevel:"
            << maxEdgeLenSqr << endl;
    }


    // 3. Combine the two sets of lengths
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    forAll(typEdgeLenSqr, levelI)
    {
        if (typEdgeLenSqr[levelI] == GREAT2 && maxEdgeLenSqr[levelI] >= 0)
        {
            typEdgeLenSqr[levelI] = maxEdgeLenSqr[levelI];
        }
    }

    if (debug)
    {
        Pout<< "hexRef4::getLevel0EdgeLength() :"
            << " Final Edgelengths (squared) per refinementlevel:"
            << typEdgeLenSqr << endl;
    }

    // Find lowest level present
    scalar level0Size = -1;

    forAll(typEdgeLenSqr, levelI)
    {
        scalar lenSqr = typEdgeLenSqr[levelI];

        if (lenSqr < GREAT2)
        {
            level0Size = Foam::sqrt(lenSqr)*(1<<levelI);

            if (debug)
            {
                Pout<< "hexRef4::getLevel0EdgeLength() :"
                    << " For level:" << levelI
                    << " have edgeLen:" << Foam::sqrt(lenSqr)
                    << " with equivalent level0 len:" << level0Size
                    << endl;
            }
            break;
        }
    }

    if (level0Size == -1)
    {
        FatalErrorInFunction
            << "Problem : typEdgeLenSqr:" << typEdgeLenSqr << abort(FatalError);
    }

    return level0Size;
}


// Check whether pointi is an anchor on celli.
// If it is not check whether any other point on the face is an anchor cell.
Foam::label Foam::hexRef4::getAnchorCell
(
    const labelListList& cellAnchorPoints,
    const labelListList& cellAddedCells,
    const label celli,
    const label facei,
    const label pointi
) const
{
    if (cellAnchorPoints[celli].size())
    {
        label index = cellAnchorPoints[celli].find(pointi);

        if (index != -1)
        {
            return cellAddedCells[celli][index];
        }


        // pointi is not an anchor cell.
        // Maybe we are already a refined face so check all the face
        // vertices.
        const face& f = mesh_.faces()[facei];

        forAll(f, fp)
        {
            label index = cellAnchorPoints[celli].find(f[fp]);

            if (index != -1)
            {
                return cellAddedCells[celli][index];
            }
        }

        // Problem.
        dumpCell(celli);
        Perr<< "cell:" << celli << " anchorPoints:" << cellAnchorPoints[celli]
            << endl;

        FatalErrorInFunction
            << "Could not find point " << pointi
            << " in the anchorPoints for cell " << celli << endl
            << "Does your original mesh obey the 2:1 constraint and"
            << " did you use consistentRefinement to make your cells to refine"
            << " obey this constraint as well?"
            << abort(FatalError);

        return -1;
    }
    else
    {
        return celli;
    }
}


// Get new owner and neighbour
void Foam::hexRef4::getFaceNeighbours
(
    const labelListList& cellAnchorPoints,
    const labelListList& cellAddedCells,
    const label facei,
    const label pointi,

    label& own,
    label& nei
) const
{
    // Is owner split?
    own = getAnchorCell
    (
        cellAnchorPoints,
        cellAddedCells,
        mesh_.faceOwner()[facei],
        facei,
        pointi
    );

    if (mesh_.isInternalFace(facei))
    {
        nei = getAnchorCell
        (
            cellAnchorPoints,
            cellAddedCells,
            mesh_.faceNeighbour()[facei],
            facei,
            pointi
        );
    }
    else
    {
        nei = -1;
    }
}


// Get point with the lowest pointLevel
Foam::label Foam::hexRef4::findMinLevel(const labelList& f) const
{
    label minLevel = labelMax;
    label minFp = -1;

    forAll(f, fp)
    {
        label level = pointLevel_[f[fp]];

        if (level < minLevel)
        {
            minLevel = level;
            minFp = fp;
        }
    }

    return minFp;
}


// Get point with the highest pointLevel
Foam::label Foam::hexRef4::findMaxLevel(const labelList& f) const
{
    label maxLevel = labelMin;
    label maxFp = -1;

    forAll(f, fp)
    {
        label level = pointLevel_[f[fp]];

        if (level > maxLevel)
        {
            maxLevel = level;
            maxFp = fp;
        }
    }

    return maxFp;
}


Foam::label Foam::hexRef4::countAnchors
(
    const labelList& f,
    const label anchorLevel
) const
{
    label nAnchors = 0;

    forAll(f, fp)
    {
        if (pointLevel_[f[fp]] <= anchorLevel)
        {
            nAnchors++;
        }
    }
    return nAnchors;
}


void Foam::hexRef4::dumpCell(const label celli) const
{
    OFstream str(mesh_.time().path()/"cell_" + Foam::name(celli) + ".obj");
    Pout<< "hexRef4 : Dumping cell as obj to " << str.name() << endl;

    const cell& cFaces = mesh_.cells()[celli];

    Map<label> pointToObjVert;
    label objVertI = 0;

    forAll(cFaces, i)
    {
        const face& f = mesh_.faces()[cFaces[i]];

        forAll(f, fp)
        {
            if (pointToObjVert.insert(f[fp], objVertI))
            {
                meshTools::writeOBJ(str, mesh_.points()[f[fp]]);
                objVertI++;
            }
        }
    }

    forAll(cFaces, i)
    {
        const face& f = mesh_.faces()[cFaces[i]];

        forAll(f, fp)
        {
            label pointi = f[fp];
            label nexPointi = f[f.fcIndex(fp)];

            str << "l " << pointToObjVert[pointi]+1
                << ' ' << pointToObjVert[nexPointi]+1 << nl;
        }
    }
}


// Find point with certain pointLevel. Skip any higher levels.
Foam::label Foam::hexRef4::findLevel
(
    const label facei,
    const face& f,
    const label startFp,
    const bool searchForward,
    const label wantedLevel
) const
{
    label fp = startFp;

    forAll(f, i)
    {
        label pointi = f[fp];

        if (pointLevel_[pointi] < wantedLevel)
        {
            dumpCell(mesh_.faceOwner()[facei]);
            if (mesh_.isInternalFace(facei))
            {
                dumpCell(mesh_.faceNeighbour()[facei]);
            }

            FatalErrorInFunction
                << "face:" << f
                << " level:" << labelUIndList(pointLevel_, f)
                << " startFp:" << startFp
                << " wantedLevel:" << wantedLevel
                << abort(FatalError);
        }
        else if (pointLevel_[pointi] == wantedLevel)
        {
            return fp;
        }

        if (searchForward)
        {
            fp = f.fcIndex(fp);
        }
        else
        {
            fp = f.rcIndex(fp);
        }
    }

    dumpCell(mesh_.faceOwner()[facei]);
    if (mesh_.isInternalFace(facei))
    {
        dumpCell(mesh_.faceNeighbour()[facei]);
    }

    FatalErrorInFunction
        << "face:" << f
        << " level:" << labelUIndList(pointLevel_, f)
        << " startFp:" << startFp
        << " wantedLevel:" << wantedLevel
        << abort(FatalError);

    return -1;
}


// Gets cell level such that the face has four points <= level.
Foam::label Foam::hexRef4::faceLevel(const label facei) const
{
    const face& f = mesh_.faces()[facei];

    if (f.size() <= 4)
    {
        return pointLevel_[f[findMaxLevel(f)]];
    }
    else
    {
        label ownLevel = cellLevel_[mesh_.faceOwner()[facei]];

        if (countAnchors(f, ownLevel) == 4)
        {
            return ownLevel;
        }
        else if (countAnchors(f, ownLevel+1) == 4)
        {
            return ownLevel+1;
        }
        else
        {
            return -1;
        }
    }
}


void Foam::hexRef4::checkInternalOrientation
(
    polyTopoChange& meshMod,
    const label celli,
    const label facei,
    const point& ownPt,
    const point& neiPt,
    const face& newFace
)
{
    face compactFace(identity(newFace.size()));
    pointField compactPoints(meshMod.points(), newFace);

    const vector areaNorm(compactFace.areaNormal(compactPoints));

    const vector dir(neiPt - ownPt);

    if ((dir & areaNorm) < 0)
    {
        FatalErrorInFunction
            << "cell:" << celli << " old face:" << facei
            << " newFace:" << newFace << endl
            << " coords:" << compactPoints
            << " ownPt:" << ownPt
            << " neiPt:" << neiPt
            << abort(FatalError);
    }

    const vector fcToOwn(compactFace.centre(compactPoints) - ownPt);

    const scalar s = (fcToOwn & areaNorm) / (dir & areaNorm);

    if (s < 0.1 || s > 0.9)
    {
        FatalErrorInFunction
            << "cell:" << celli << " old face:" << facei
            << " newFace:" << newFace << endl
            << " coords:" << compactPoints
            << " ownPt:" << ownPt
            << " neiPt:" << neiPt
            << " s:" << s
            << abort(FatalError);
    }
}


void Foam::hexRef4::checkBoundaryOrientation
(
    polyTopoChange& meshMod,
    const label celli,
    const label facei,
    const point& ownPt,
    const point& boundaryPt,
    const face& newFace
)
{
    face compactFace(identity(newFace.size()));
    pointField compactPoints(meshMod.points(), newFace);

    const vector areaNorm(compactFace.areaNormal(compactPoints));

    const vector dir(boundaryPt - ownPt);

    if ((dir & areaNorm) < 0)
    {
        FatalErrorInFunction
            << "cell:" << celli << " old face:" << facei
            << " newFace:" << newFace
            << " coords:" << compactPoints
            << " ownPt:" << ownPt
            << " boundaryPt:" << boundaryPt
            << abort(FatalError);
    }

    const vector fcToOwn(compactFace.centre(compactPoints) - ownPt);

    const scalar s = (fcToOwn & dir) / magSqr(dir);

    if (s < 0.7 || s > 1.3)
    {
        WarningInFunction
            << "cell:" << celli << " old face:" << facei
            << " newFace:" << newFace
            << " coords:" << compactPoints
            << " ownPt:" << ownPt
            << " boundaryPt:" << boundaryPt
            << " s:" << s
            << endl;
            //<< abort(FatalError);
    }
}


// If p0 and p1 are existing vertices check if edge is split and insert
// splitPoint.
void Foam::hexRef4::insertEdgeSplit
(
    const labelList& edgeMidPoint,
    const label p0,
    const label p1,
    DynamicList<label>& verts
) const
{
    if (p0 < mesh_.nPoints() && p1 < mesh_.nPoints())
    {
        label edgeI = meshTools::findEdge(mesh_, p0, p1);

        if (edgeI != -1 && edgeMidPoint[edgeI] != -1)
        {
            verts.append(edgeMidPoint[edgeI]);
        }
    }
}


// Determine which of x,y,z is the 2-D (empty) direction. Preferred source is
// the empty patches themselves; polyMesh::geometricD is the fallback for
// meshes that (temporarily) have none, e.g. after subsetting.
void Foam::hexRef4::calcSpanDirection()
{
    // Sum of |normal| components so the result is independent of face
    // orientation and therefore identical on every processor.
    vector sumNormal(Zero);
    label nEmptyFaces = 0;

    const polyBoundaryMesh& patches = mesh_.boundaryMesh();

    forAll(patches, patchi)
    {
        const polyPatch& pp = patches[patchi];

        if (isA<emptyPolyPatch>(pp))
        {
            forAll(pp, i)
            {
                const label facei = pp.start() + i;

                const vector& a = mesh_.faceAreas()[facei];

                sumNormal += cmptMag(a/(mag(a) + VSMALL));
                nEmptyFaces++;
            }
        }
    }

    reduce(sumNormal, sumOp<vector>());
    reduce(nEmptyFaces, sumOp<label>());

    if (!nEmptyFaces)
    {
        // No empty patches at all. Ask the mesh which direction carries no
        // solution.
        const Vector<label> geomD(mesh_.geometricD());

        sumNormal = Zero;
        forAll(geomD, dir)
        {
            if (geomD[dir] == -1)
            {
                sumNormal[dir] = 1;
            }
        }

        if (mag(sumNormal) < SMALL)
        {
            FatalErrorInFunction
                << "Cannot determine the 2-D span direction: the mesh has no"
                << " empty patches and polyMesh::geometricD() returns "
                << geomD << " i.e. the mesh is 3-D." << nl
                << "hexRef4 only refines 2-D (one cell thick) meshes."
                << " Use hexRef8 for 3-D meshes."
                << abort(FatalError);
        }
    }

    // Snap onto the dominant coordinate axis
    spanDir_ = -1;
    scalar maxCmpt = 0;

    for (direction dir = 0; dir < vector::nComponents; dir++)
    {
        if (mag(sumNormal[dir]) > maxCmpt)
        {
            maxCmpt = mag(sumNormal[dir]);
            spanDir_ = dir;
        }
    }

    // Every empty face must agree, otherwise the "2-D" direction is not
    // unique and the refinement below would be ill-defined.
    if (spanDir_ == -1 || maxCmpt < (1 - 1e-3)*mag(sumNormal))
    {
        FatalErrorInFunction
            << "The empty-patch normals of this mesh do not all point along a"
            << " single coordinate axis (summed |normal| = " << sumNormal
            << ")." << nl
            << "hexRef4 requires the 2-D direction to be along x, y or z."
            << abort(FatalError);
    }

    spanVec_ = Zero;
    spanVec_[spanDir_] = 1;

    if (debug)
    {
        Pout<< "hexRef4 : 2-D span (empty) direction is component "
            << label(spanDir_) << " " << spanVec_
            << " from " << nEmptyFaces << " empty faces" << endl;
    }
}


bool Foam::hexRef4::isSpanFace(const label facei) const
{
    const vector& a = mesh_.faceAreas()[facei];

    return mag(a[spanDir_]) > (1 - 1e-3)*(mag(a) + VSMALL);
}


bool Foam::hexRef4::isSpanEdge(const label edgeI) const
{
    const vector d(mesh_.edges()[edgeI].vec(mesh_.points()));

    return mag(d[spanDir_]) > (1 - 1e-3)*(mag(d) + VSMALL);
}


// Point on the opposite span face connected to pointi by a span-parallel edge
Foam::label Foam::hexRef4::spanPartnerPoint
(
    const label pointi,
    const label otherFacei
) const
{
    const face& otherF = mesh_.faces()[otherFacei];
    const labelList& pEdges = mesh_.pointEdges()[pointi];

    forAll(pEdges, i)
    {
        const label edgeI = pEdges[i];

        if (isSpanEdge(edgeI))
        {
            const label otherPointi = mesh_.edges()[edgeI].otherVertex(pointi);

            if (otherF.find(otherPointi) != -1)
            {
                return otherPointi;
            }
        }
    }

    dumpCell(mesh_.faceOwner()[otherFacei]);

    FatalErrorInFunction
        << "Could not find the span-direction partner of point " << pointi
        << " at " << mesh_.points()[pointi] << nl
        << "on the opposite span face " << otherFacei
        << " with points " << otherF << nl
        << "Is the mesh really one cell thick along component "
        << label(spanDir_) << "?"
        << abort(FatalError);

    return -1;
}


// For a span face that is about to be split into four, collect per anchor
// point the point where the perimeter splits (walking forward) and the anchor
// that follows it.
void Foam::hexRef4::collectSpokes
(
    const labelList& edgeMidPoint,
    const label cLevel,
    const label facei,
    labelList& anchors,
    labelList& splitPoints,
    labelList& nextAnchors
) const
{
    const face& f = mesh_.faces()[facei];
    const labelList& fEdges = mesh_.faceEdges(facei);

    DynamicList<label> anch(4);
    DynamicList<label> split(4);
    DynamicList<label> nextAnch(4);

    forAll(f, fp)
    {
        if (pointLevel_[f[fp]] > cLevel)
        {
            continue;
        }

        // f[fp] is an anchor. Find where the perimeter splits ahead of it.
        label splitPointi = -1;

        const label fp1 = f.fcIndex(fp);

        if (pointLevel_[f[fp1]] <= cLevel)
        {
            // Next vertex is the next anchor, so the edge in between is the
            // one being split right now.
            splitPointi = edgeMidPoint[fEdges[fp]];

            if (splitPointi == -1)
            {
                dumpCell(mesh_.faceOwner()[facei]);

                FatalErrorInFunction
                    << "Span face " << facei << " f:" << f
                    << " pointLevel:" << labelUIndList(pointLevel_, f)
                    << " has edge " << fEdges[fp]
                    << " between two anchors at level " << cLevel
                    << " that was not marked for splitting"
                    << abort(FatalError);
            }
        }
        else
        {
            // An earlier refinement already split this edge: pick up the
            // existing cLevel+1 point.
            splitPointi = f[findLevel(facei, f, fp1, true, cLevel+1)];
        }

        // Walk on to the next anchor
        label fpNext = fp1;
        while (pointLevel_[f[fpNext]] > cLevel)
        {
            fpNext = f.fcIndex(fpNext);
        }

        anch.append(f[fp]);
        split.append(splitPointi);
        nextAnch.append(f[fpNext]);
    }

    anchors.transfer(anch);
    splitPoints.transfer(split);
    nextAnchors.transfer(nextAnch);
}


// Split a side face into two along the span direction, keeping its full span
// extent. Both halves inherit the orientation of the original face.
void Foam::hexRef4::splitSideFace
(
    const labelListList& cellAnchorPoints,
    const labelListList& cellAddedCells,
    const labelList& edgeMidPoint,
    const label facei,
    polyTopoChange& meshMod
) const
{
    const face& f = mesh_.faces()[facei];
    const labelList& fEdges = mesh_.faceEdges(facei);

    // Vertex loop with the new edge mid points inserted; remember where the
    // two split points ended up.
    DynamicList<label> loop(f.size() + 2);
    DynamicList<label> splitAt(2);

    forAll(f, fp)
    {
        loop.append(f[fp]);

        const label edgeI = fEdges[fp];

        if (edgeMidPoint[edgeI] >= 0)
        {
            splitAt.append(loop.size());
            loop.append(edgeMidPoint[edgeI]);
        }
    }

    if (splitAt.size() != 2)
    {
        dumpCell(mesh_.faceOwner()[facei]);

        FatalErrorInFunction
            << "Side face " << facei << " f:" << f
            << " pointLevel:" << labelUIndList(pointLevel_, f)
            << " of a cell being refined has " << splitAt.size()
            << " split edges instead of 2."
            << abort(FatalError);
    }

    // Cut the loop at the two split points into two faces
    const label i0 = splitAt[0];
    const label i1 = splitAt[1];

    face f0(i1 - i0 + 1);
    forAll(f0, i)
    {
        f0[i] = loop[i0 + i];
    }

    face f1(loop.size() - i1 + i0 + 1);
    forAll(f1, i)
    {
        f1[i] = loop[(i1 + i) % loop.size()];
    }

    // Owner/neighbour follow from an anchor of each half. Only pre-existing
    // points have a pointLevel_, and the corner points are exactly those.
    const face* halves[2] = {&f0, &f1};
    label anchorPoint[2] = {-1, -1};

    for (label half = 0; half < 2; half++)
    {
        const face& sub = *halves[half];

        label minLevel = labelMax;

        forAll(sub, i)
        {
            const label pointi = sub[i];

            if (pointi < mesh_.nPoints() && pointLevel_[pointi] < minLevel)
            {
                minLevel = pointLevel_[pointi];
                anchorPoint[half] = pointi;
            }
        }

        if (anchorPoint[half] == -1)
        {
            dumpCell(mesh_.faceOwner()[facei]);

            FatalErrorInFunction
                << "Half " << half << " of split side face " << facei
                << " (" << sub << ") has no pre-existing point"
                << abort(FatalError);
        }
    }

    label own0, nei0;
    getFaceNeighbours
    (
        cellAnchorPoints,
        cellAddedCells,
        facei,
        anchorPoint[0],
        own0,
        nei0
    );

    label own1, nei1;
    getFaceNeighbours
    (
        cellAnchorPoints,
        cellAddedCells,
        facei,
        anchorPoint[1],
        own1,
        nei1
    );

    // Original face gets modified, the second half is added from it so that
    // it inherits the mapping (and hence the flux) of the original.
    modFace(meshMod, facei, f0, own0, nei0);
    addFace(meshMod, facei, f1, own1, nei1);
}


// Creates the 4 internal faces that split celli into 4.
void Foam::hexRef4::createInternalFaces
(
    const labelListList& cellAnchorPoints,
    const labelListList& cellAddedCells,
    const labelList& faceMidPoint,
    const labelList& edgeMidPoint,
    const label celli,

    polyTopoChange& meshMod
) const
{
    const cell& cFaces = mesh_.cells()[celli];
    const label cLevel = cellLevel_[celli];

    // Locate the two span faces. "back" is the one whose outward normal
    // points against spanVec_, which fixes the orientation convention used
    // when the internal faces are built below.
    label backFacei = -1;
    label frontFacei = -1;

    forAll(cFaces, i)
    {
        const label facei = cFaces[i];

        if (!isSpanFace(facei))
        {
            continue;
        }

        vector n = mesh_.faceAreas()[facei];

        if (mesh_.faceOwner()[facei] != celli)
        {
            n = -n;
        }

        label& slot = ((n & spanVec_) < 0 ? backFacei : frontFacei);

        if (slot != -1)
        {
            dumpCell(celli);

            FatalErrorInFunction
                << "Cell " << celli << " has more than one span face on the"
                << " same side. Mesh is not one cell thick along component "
                << label(spanDir_) << "."
                << abort(FatalError);
        }

        slot = facei;
    }

    if (backFacei == -1 || frontFacei == -1)
    {
        dumpCell(celli);

        FatalErrorInFunction
            << "Cell " << celli << " does not have two faces normal to the"
            << " span direction (back:" << backFacei
            << " front:" << frontFacei << ")." << nl
            << "hexRef4 needs a 2-D, one cell thick mesh."
            << abort(FatalError);
    }

    const label midBack = faceMidPoint[backFacei];
    const label midFront = faceMidPoint[frontFacei];

    if (midBack < 0 || midFront < 0)
    {
        dumpCell(celli);

        FatalErrorInFunction
            << "Span faces " << backFacei << " and " << frontFacei
            << " of refined cell " << celli << " were not both marked for"
            << " splitting (mid points " << midBack << ", " << midFront
            << ")." << nl
            << "hexRef4 needs the empty-patch faces of a refined cell to be"
            << " at the same refinement level as the cell."
            << abort(FatalError);
    }

    labelList backAnchors, backSplits, backNext;
    collectSpokes
    (
        edgeMidPoint,
        cLevel,
        backFacei,
        backAnchors,
        backSplits,
        backNext
    );

    labelList frontAnchors, frontSplits, frontNext;
    collectSpokes
    (
        edgeMidPoint,
        cLevel,
        frontFacei,
        frontAnchors,
        frontSplits,
        frontNext
    );

    if (backAnchors.size() != 4 || frontAnchors.size() != 4)
    {
        dumpCell(celli);

        FatalErrorInFunction
            << "Span faces of cell " << celli << " at level " << cLevel
            << " have " << backAnchors.size() << " and "
            << frontAnchors.size() << " anchor points instead of 4." << nl
            << "hexRef4 cannot refine a cell whose empty-patch face is"
            << " already more refined than the cell itself."
            << abort(FatalError);
    }

    // Index the front spokes on their (unordered) anchor pair so a back spoke
    // can look up its partner across the span.
    HashTable<label, edge, Hash<edge>> frontSplitOfPair(8);

    forAll(frontAnchors, i)
    {
        frontSplitOfPair.insert
        (
            edge(frontAnchors[i], frontNext[i]),
            frontSplits[i]
        );
    }

    forAll(backAnchors, i)
    {
        const label anchorA = backAnchors[i];
        const label anchorB = backNext[i];
        const label splitBack = backSplits[i];

        const edge frontPair
        (
            spanPartnerPoint(anchorA, frontFacei),
            spanPartnerPoint(anchorB, frontFacei)
        );

        const auto fnd = frontSplitOfPair.cfind(frontPair);

        if (!fnd.good())
        {
            dumpCell(celli);

            FatalErrorInFunction
                << "Cell " << celli << ": the back-face split between anchors "
                << anchorA << " and " << anchorB
                << " has no matching front-face split (looked for anchor pair "
                << frontPair << ")."
                << abort(FatalError);
        }

        const label splitFront = fnd.val();

        // With back = -spanVec_ side, the loop (splitBack, midBack, midFront,
        // splitFront) has a normal pointing from subcell(anchorB) towards
        // subcell(anchorA).
        face newFace(4);
        newFace[0] = splitBack;
        newFace[1] = midBack;
        newFace[2] = midFront;
        newFace[3] = splitFront;

        label own = getAnchorCell
        (
            cellAnchorPoints,
            cellAddedCells,
            celli,
            backFacei,
            anchorB
        );
        label nei = getAnchorCell
        (
            cellAnchorPoints,
            cellAddedCells,
            celli,
            backFacei,
            anchorA
        );

        point ownPt = mesh_.points()[anchorB];
        point neiPt = mesh_.points()[anchorA];

        if (own > nei)
        {
            std::swap(own, nei);
            std::swap(ownPt, neiPt);
            newFace.flip();
        }

        if (debug)
        {
            checkInternalOrientation
            (
                meshMod,
                celli,
                backFacei,
                ownPt,
                neiPt,
                newFace
            );
        }

        addInternalFace
        (
            meshMod,
            backFacei,
            anchorA,
            newFace,
            own,
            nei
        );
    }
}


void Foam::hexRef4::walkFaceToMid
(
    const labelList& edgeMidPoint,
    const label cLevel,
    const label facei,
    const label startFp,
    DynamicList<label>& faceVerts
) const
{
    const face& f = mesh_.faces()[facei];
    const labelList& fEdges = mesh_.faceEdges(facei);

    label fp = startFp;

    // Starting from fp store all (1 or 2) vertices until where the face
    // gets split

    while (true)
    {
        if (edgeMidPoint[fEdges[fp]] >= 0)
        {
            faceVerts.append(edgeMidPoint[fEdges[fp]]);
        }

        fp = f.fcIndex(fp);

        if (pointLevel_[f[fp]] <= cLevel)
        {
            // Next anchor. Have already append split point on edge in code
            // above.
            return;
        }
        else if (pointLevel_[f[fp]] == cLevel+1)
        {
            // Mid level
            faceVerts.append(f[fp]);

            return;
        }
        else if (pointLevel_[f[fp]] == cLevel+2)
        {
            // Store and continue to cLevel+1.
            faceVerts.append(f[fp]);
        }
    }
}


// Same as walkFaceToMid but now walk back.
void Foam::hexRef4::walkFaceFromMid
(
    const labelList& edgeMidPoint,
    const label cLevel,
    const label facei,
    const label startFp,
    DynamicList<label>& faceVerts
) const
{
    const face& f = mesh_.faces()[facei];
    const labelList& fEdges = mesh_.faceEdges(facei);

    label fp = f.rcIndex(startFp);

    while (true)
    {
        if (pointLevel_[f[fp]] <= cLevel)
        {
            // anchor.
            break;
        }
        else if (pointLevel_[f[fp]] == cLevel+1)
        {
            // Mid level
            faceVerts.append(f[fp]);
            break;
        }
        else if (pointLevel_[f[fp]] == cLevel+2)
        {
            // Continue to cLevel+1.
        }
        fp = f.rcIndex(fp);
    }

    // Store
    while (true)
    {
        if (edgeMidPoint[fEdges[fp]] >= 0)
        {
            faceVerts.append(edgeMidPoint[fEdges[fp]]);
        }

        fp = f.fcIndex(fp);

        if (fp == startFp)
        {
            break;
        }
        faceVerts.append(f[fp]);
    }
}


// Updates refineCell (cells marked for refinement) so across all faces
// there will be 2:1 consistency after refinement.
Foam::label Foam::hexRef4::faceConsistentRefinement
(
    const bool maxSet,
    const labelUList& cellLevel,
    bitSet& refineCell
) const
{
    label nChanged = 0;

    // Internal faces.
    for (label facei = 0; facei < mesh_.nInternalFaces(); facei++)
    {
        label own = mesh_.faceOwner()[facei];
        label ownLevel = cellLevel[own] + refineCell.get(own);

        label nei = mesh_.faceNeighbour()[facei];
        label neiLevel = cellLevel[nei] + refineCell.get(nei);

        if (ownLevel > (neiLevel+1))
        {
            if (maxSet)
            {
                refineCell.set(nei);
            }
            else
            {
                refineCell.unset(own);
            }
            nChanged++;
        }
        else if (neiLevel > (ownLevel+1))
        {
            if (maxSet)
            {
                refineCell.set(own);
            }
            else
            {
                refineCell.unset(nei);
            }
            nChanged++;
        }
    }


    // Coupled faces. Swap owner level to get neighbouring cell level.
    // (only boundary faces of neiLevel used)
    labelList neiLevel(mesh_.nBoundaryFaces());

    forAll(neiLevel, i)
    {
        label own = mesh_.faceOwner()[i+mesh_.nInternalFaces()];

        neiLevel[i] = cellLevel[own] + refineCell.get(own);
    }

    // Swap to neighbour
    syncTools::swapBoundaryFaceList(mesh_, neiLevel);

    // Now we have neighbour value see which cells need refinement
    forAll(neiLevel, i)
    {
        label own = mesh_.faceOwner()[i+mesh_.nInternalFaces()];
        label ownLevel = cellLevel[own] + refineCell.get(own);

        if (ownLevel > (neiLevel[i]+1))
        {
            if (!maxSet)
            {
                refineCell.unset(own);
                nChanged++;
            }
        }
        else if (neiLevel[i] > (ownLevel+1))
        {
            if (maxSet)
            {
                refineCell.set(own);
                nChanged++;
            }
        }
    }

    return nChanged;
}


// Debug: check if wanted refinement is compatible with 2:1
void Foam::hexRef4::checkWantedRefinementLevels
(
    const labelUList& cellLevel,
    const labelList& cellsToRefine
) const
{
    bitSet refineCell(mesh_.nCells(), cellsToRefine);

    for (label facei = 0; facei < mesh_.nInternalFaces(); facei++)
    {
        label own = mesh_.faceOwner()[facei];
        label ownLevel = cellLevel[own] + refineCell.get(own);

        label nei = mesh_.faceNeighbour()[facei];
        label neiLevel = cellLevel[nei] + refineCell.get(nei);

        if (mag(ownLevel-neiLevel) > 1)
        {
            dumpCell(own);
            dumpCell(nei);
            FatalErrorInFunction
                << "cell:" << own
                << " current level:" << cellLevel[own]
                << " level after refinement:" << ownLevel
                << nl
                << "neighbour cell:" << nei
                << " current level:" << cellLevel[nei]
                << " level after refinement:" << neiLevel
                << nl
                << "which does not satisfy 2:1 constraints anymore."
                << abort(FatalError);
        }
    }

    // Coupled faces. Swap owner level to get neighbouring cell level.
    // (only boundary faces of neiLevel used)
    labelList neiLevel(mesh_.nBoundaryFaces());

    forAll(neiLevel, i)
    {
        label own = mesh_.faceOwner()[i+mesh_.nInternalFaces()];

        neiLevel[i] = cellLevel[own] + refineCell.get(own);
    }

    // Swap to neighbour
    syncTools::swapBoundaryFaceList(mesh_, neiLevel);

    // Now we have neighbour value see which cells need refinement
    forAll(neiLevel, i)
    {
        label facei = i + mesh_.nInternalFaces();

        label own = mesh_.faceOwner()[facei];
        label ownLevel = cellLevel[own] + refineCell.get(own);

        if (mag(ownLevel - neiLevel[i]) > 1)
        {
            label patchi = mesh_.boundaryMesh().whichPatch(facei);

            dumpCell(own);
            FatalErrorInFunction
                << "Celllevel does not satisfy 2:1 constraint."
                << " On coupled face "
                << facei
                << " on patch " << patchi << " "
                << mesh_.boundaryMesh()[patchi].name()
                << " owner cell " << own
                << " current level:" << cellLevel[own]
                << " level after refinement:" << ownLevel
                << nl
                << " (coupled) neighbour cell will get refinement "
                << neiLevel[i]
                << abort(FatalError);
        }
    }
}


// Set instance for mesh files
void Foam::hexRef4::setInstance(const fileName& inst)
{
    if (debug)
    {
        Pout<< "hexRef4::setInstance(const fileName& inst) : "
            << "Resetting file instance to " << inst << endl;
    }

    cellLevel_.instance() = inst;
    pointLevel_.instance() = inst;
    level0Edge_.instance() = inst;
    history_.instance() = inst;
}


void Foam::hexRef4::collectLevelPoints
(
    const labelList& f,
    const label level,
    DynamicList<label>& points
) const
{
    forAll(f, fp)
    {
        if (pointLevel_[f[fp]] <= level)
        {
            points.append(f[fp]);
        }
    }
}


void Foam::hexRef4::collectLevelPoints
(
    const labelList& meshPoints,
    const labelList& f,
    const label level,
    DynamicList<label>& points
) const
{
    forAll(f, fp)
    {
        label pointi = meshPoints[f[fp]];
        if (pointLevel_[pointi] <= level)
        {
            points.append(pointi);
        }
    }
}


// Return true if we've found 6 quads. faces guaranteed to be outwards pointing.
bool Foam::hexRef4::matchHexShape
(
    const label celli,
    const label cellLevel,
    DynamicList<face>& quads
) const
{
    const cell& cFaces = mesh_.cells()[celli];

    // Work arrays
    DynamicList<label> verts(4);
    quads.clear();


    // 1. pick up any faces with four cellLevel points

    forAll(cFaces, i)
    {
        label facei = cFaces[i];
        const face& f = mesh_.faces()[facei];

        verts.clear();
        collectLevelPoints(f, cellLevel, verts);
        if (verts.size() == 4)
        {
            if (mesh_.faceOwner()[facei] != celli)
            {
                reverse(verts);
            }
            quads.emplace_back(verts);
        }
    }


    if (quads.size() < 6)
    {
        Map<labelList> pointFaces(2*cFaces.size());

        forAll(cFaces, i)
        {
            label facei = cFaces[i];
            const face& f = mesh_.faces()[facei];

            // Pick up any faces with only one level point.
            // See if there are four of these where the common point
            // is a level+1 point. This common point is then the mid of
            // a split face.

            verts.clear();
            collectLevelPoints(f, cellLevel, verts);
            if (verts.size() == 1)
            {
                // Add to pointFaces for any level+1 point (this might be
                // a midpoint of a split face)
                for (const label pointi : f)
                {
                    if (pointLevel_[pointi] == cellLevel+1)
                    {
                        pointFaces(pointi).push_uniq(facei);
                    }
                }
            }
        }

        // 2. Check if we've collected any midPoints.
        forAllConstIters(pointFaces, iter)
        {
            const labelList& pFaces = iter.val();

            if (pFaces.size() == 4)
            {
                // Collect and orient.
                faceList fourFaces(pFaces.size());
                forAll(pFaces, pFacei)
                {
                    label facei = pFaces[pFacei];
                    const face& f = mesh_.faces()[facei];
                    if (mesh_.faceOwner()[facei] == celli)
                    {
                        fourFaces[pFacei] = f;
                    }
                    else
                    {
                        fourFaces[pFacei] = f.reverseFace();
                    }
                }

                primitivePatch bigFace
                (
                    SubList<face>(fourFaces),
                    mesh_.points()
                );
                const labelListList& edgeLoops = bigFace.edgeLoops();

                if (edgeLoops.size() == 1)
                {
                    // Collect the 4 cellLevel points
                    verts.clear();
                    collectLevelPoints
                    (
                        bigFace.meshPoints(),
                        bigFace.edgeLoops()[0],
                        cellLevel,
                        verts
                    );

                    if (verts.size() == 4)
                    {
                        quads.emplace_back(verts);
                    }
                }
            }
        }
    }

    return (quads.size() == 6);
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

// Construct from mesh, read refinement data
Foam::hexRef4::hexRef4(const polyMesh& mesh, const bool readHistory)
:
    mesh_(mesh),
    cellLevel_
    (
        IOobject
        (
            "cellLevel",
            mesh_.facesInstance(),
            polyMesh::meshSubDir,
            mesh_,
            IOobject::READ_IF_PRESENT,
            IOobject::NO_WRITE
        ),
        labelList(mesh_.nCells(), Zero)
    ),
    pointLevel_
    (
        IOobject
        (
            "pointLevel",
            mesh_.facesInstance(),
            polyMesh::meshSubDir,
            mesh_,
            IOobject::READ_IF_PRESENT,
            IOobject::NO_WRITE
        ),
        labelList(mesh_.nPoints(), Zero)
    ),
    level0Edge_
    (
        IOobject
        (
            "level0Edge",
            mesh_.facesInstance(),
            polyMesh::meshSubDir,
            mesh_,
            IOobject::READ_IF_PRESENT,
            IOobject::NO_WRITE
        ),
        // Needs name:
        dimensionedScalar("level0Edge", dimLength, getLevel0EdgeLength())
    ),
    history_
    (
        IOobject
        (
            "refinementHistory",
            mesh_.facesInstance(),
            polyMesh::meshSubDir,
            mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        // All cells visible if not read or readHistory = false
        (readHistory ? mesh_.nCells() : 0)
    ),
    faceRemover_(mesh_, GREAT),     // merge boundary faces wherever possible
    savedPointLevel_(0),
    savedCellLevel_(0),
    spanDir_(-1),
    spanVec_(Zero)
{
    calcSpanDirection();

    if (readHistory)
    {
        history_.readOpt(IOobject::READ_IF_PRESENT);
        if (history_.typeHeaderOk<refinementHistory>(true))
        {
            history_.read();
        }
    }

    if (history_.active() && history_.visibleCells().size() != mesh_.nCells())
    {
        FatalErrorInFunction
            << "History enabled but number of visible cells "
            << history_.visibleCells().size() << " in "
            << history_.objectPath()
            << " is not equal to the number of cells in the mesh "
            << mesh_.nCells()
            << abort(FatalError);
    }

    if
    (
        cellLevel_.size() != mesh_.nCells()
     || pointLevel_.size() != mesh_.nPoints()
    )
    {
        FatalErrorInFunction
            << "Restarted from inconsistent cellLevel or pointLevel files."
            << endl
            << "cellLevel file " << cellLevel_.objectPath() << endl
            << "pointLevel file " << pointLevel_.objectPath() << endl
            << "Number of cells in mesh:" << mesh_.nCells()
            << " does not equal size of cellLevel:" << cellLevel_.size() << endl
            << "Number of points in mesh:" << mesh_.nPoints()
            << " does not equal size of pointLevel:" << pointLevel_.size()
            << abort(FatalError);
    }


    // Check refinement levels for consistency
    checkRefinementLevels(-1, labelList(0));


    // Check initial mesh for consistency

    //if (debug)
    {
        checkMesh();
    }
}


Foam::hexRef4::hexRef4
(
    const polyMesh& mesh,
    const labelList& cellLevel,
    const labelList& pointLevel,
    const refinementHistory& history,
    const scalar level0Edge
)
:
    mesh_(mesh),
    cellLevel_
    (
        IOobject
        (
            "cellLevel",
            mesh_.facesInstance(),
            polyMesh::meshSubDir,
            mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        cellLevel
    ),
    pointLevel_
    (
        IOobject
        (
            "pointLevel",
            mesh_.facesInstance(),
            polyMesh::meshSubDir,
            mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        pointLevel
    ),
    level0Edge_
    (
        IOobject
        (
            "level0Edge",
            mesh_.facesInstance(),
            polyMesh::meshSubDir,
            mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        // Needs name:
        dimensionedScalar
        (
            "level0Edge",
            dimLength,
            (level0Edge >= 0 ? level0Edge : getLevel0EdgeLength())
        )
    ),
    history_
    (
        IOobject
        (
            "refinementHistory",
            mesh_.facesInstance(),
            polyMesh::meshSubDir,
            mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        history
    ),
    faceRemover_(mesh_, GREAT),     // merge boundary faces wherever possible
    savedPointLevel_(0),
    savedCellLevel_(0),
    spanDir_(-1),
    spanVec_(Zero)
{
    calcSpanDirection();

    if (history_.active() && history_.visibleCells().size() != mesh_.nCells())
    {
        FatalErrorInFunction
            << "History enabled but number of visible cells in it "
            << history_.visibleCells().size()
            << " is not equal to the number of cells in the mesh "
            << mesh_.nCells() << abort(FatalError);
    }

    if
    (
        cellLevel_.size() != mesh_.nCells()
     || pointLevel_.size() != mesh_.nPoints()
    )
    {
        FatalErrorInFunction
            << "Incorrect cellLevel or pointLevel size." << endl
            << "Number of cells in mesh:" << mesh_.nCells()
            << " does not equal size of cellLevel:" << cellLevel_.size() << endl
            << "Number of points in mesh:" << mesh_.nPoints()
            << " does not equal size of pointLevel:" << pointLevel_.size()
            << abort(FatalError);
    }

    // Check refinement levels for consistency
    checkRefinementLevels(-1, labelList(0));


    // Check initial mesh for consistency

    //if (debug)
    {
        checkMesh();
    }
}


Foam::hexRef4::hexRef4
(
    const polyMesh& mesh,
    const labelList& cellLevel,
    const labelList& pointLevel,
    const scalar level0Edge
)
:
    mesh_(mesh),
    cellLevel_
    (
        IOobject
        (
            "cellLevel",
            mesh_.facesInstance(),
            polyMesh::meshSubDir,
            mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        cellLevel
    ),
    pointLevel_
    (
        IOobject
        (
            "pointLevel",
            mesh_.facesInstance(),
            polyMesh::meshSubDir,
            mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        pointLevel
    ),
    level0Edge_
    (
        IOobject
        (
            "level0Edge",
            mesh_.facesInstance(),
            polyMesh::meshSubDir,
            mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        // Needs name:
        dimensionedScalar
        (
            "level0Edge",
            dimLength,
            (level0Edge >= 0 ? level0Edge : getLevel0EdgeLength())
        )
    ),
    history_
    (
        IOobject
        (
            "refinementHistory",
            mesh_.facesInstance(),
            polyMesh::meshSubDir,
            mesh_,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        List<refinementHistory::splitCell8>(0),
        labelList(0),
        false
    ),
    faceRemover_(mesh_, GREAT),     // merge boundary faces wherever possible
    savedPointLevel_(0),
    savedCellLevel_(0),
    spanDir_(-1),
    spanVec_(Zero)
{
    calcSpanDirection();

    if
    (
        cellLevel_.size() != mesh_.nCells()
     || pointLevel_.size() != mesh_.nPoints()
    )
    {
        FatalErrorInFunction
            << "Incorrect cellLevel or pointLevel size." << endl
            << "Number of cells in mesh:" << mesh_.nCells()
            << " does not equal size of cellLevel:" << cellLevel_.size() << endl
            << "Number of points in mesh:" << mesh_.nPoints()
            << " does not equal size of pointLevel:" << pointLevel_.size()
            << abort(FatalError);
    }

    // Check refinement levels for consistency
    checkRefinementLevels(-1, labelList(0));

    // Check initial mesh for consistency

    //if (debug)
    {
        checkMesh();
    }
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::labelList Foam::hexRef4::consistentRefinement
(
    const labelUList& cellLevel,
    const labelList& cellsToRefine,
    const bool maxSet
) const
{
    // Loop, modifying cellsToRefine, until no more changes to due to 2:1
    // conflicts.
    // maxSet = false : unselect cells to refine
    // maxSet = true  : select cells to refine

    bitSet refineCell(mesh_.nCells(), cellsToRefine);

    while (true)
    {
        label nChanged = faceConsistentRefinement
        (
            maxSet,
            cellLevel,
            refineCell
        );

        reduce(nChanged, sumOp<label>());

        if (debug)
        {
            Pout<< "hexRef4::consistentRefinement : Changed " << nChanged
                << " refinement levels due to 2:1 conflicts."
                << endl;
        }

        if (nChanged == 0)
        {
            break;
        }
    }

    // Convert back to labelList.
    labelList newCellsToRefine(refineCell.toc());

    if (debug)
    {
        checkWantedRefinementLevels(cellLevel, newCellsToRefine);
    }

    return newCellsToRefine;
}


// Given a list of cells to refine determine additional cells to refine
// such that the overall refinement:
// - satisfies maxFaceDiff (e.g. 2:1) across neighbouring faces
// - satisfies maxPointDiff (e.g. 4:1) across selected point connected
//   cells. This is used to ensure that e.g. cells on the surface are not
//   point connected to cells which are 8 times smaller.
Foam::labelList Foam::hexRef4::consistentSlowRefinement
(
    const label maxFaceDiff,
    const labelList& cellsToRefine,
    const labelList& facesToCheck,
    const label maxPointDiff,
    const labelList& pointsToCheck
) const
{
    const labelList& faceOwner = mesh_.faceOwner();
    const labelList& faceNeighbour = mesh_.faceNeighbour();


    if (maxFaceDiff <= 0)
    {
        FatalErrorInFunction
            << "Illegal maxFaceDiff " << maxFaceDiff << nl
            << "Value should be >= 1" << exit(FatalError);
    }


    // Bit tricky. Say we want a distance of three cells between two
    // consecutive refinement levels. This is done by using FaceCellWave to
    // transport out the new refinement level. It gets decremented by one
    // every cell it crosses so if we initialize it to maxFaceDiff
    // we will get a field everywhere that tells us whether an unselected cell
    // needs refining as well.


    // Initial information about (distance to) cellLevel on all cells
    List<refinementData> allCellInfo(mesh_.nCells());

    // Initial information about (distance to) cellLevel on all faces
    List<refinementData> allFaceInfo(mesh_.nFaces());

    forAll(allCellInfo, celli)
    {
        // maxFaceDiff since refinementData counts both
        // faces and cells.
        allCellInfo[celli] = refinementData
        (
            maxFaceDiff*(cellLevel_[celli]+1),// when cell is to be refined
            maxFaceDiff*cellLevel_[celli]     // current level
        );
    }

    // Cells to be refined will have cellLevel+1
    forAll(cellsToRefine, i)
    {
        label celli = cellsToRefine[i];

        allCellInfo[celli].count() = allCellInfo[celli].refinementCount();
    }


    // Labels of seed faces
    DynamicList<label> seedFaces(mesh_.nFaces()/100);
    // refinementLevel data on seed faces
    DynamicList<refinementData> seedFacesInfo(mesh_.nFaces()/100);

    // Dummy additional info for FaceCellWave
    int dummyTrackData = 0;


    // Additional buffer layer thickness by changing initial count. Usually
    // this happens on boundary faces. Bit tricky. Use allFaceInfo to mark
    // off thus marked faces so they're skipped in the next loop.
    forAll(facesToCheck, i)
    {
        label facei = facesToCheck[i];

        if (allFaceInfo[facei].valid(dummyTrackData))
        {
            // Can only occur if face has already gone through loop below.
            FatalErrorInFunction
                << "Argument facesToCheck seems to have duplicate entries!"
                << endl
                << "face:" << facei << " occurs at positions "
                << findIndices(facesToCheck, facei)
                << abort(FatalError);
        }


        const refinementData& ownData = allCellInfo[faceOwner[facei]];

        label maxDataCount = ownData.count();

        if (mesh_.isInternalFace(facei))
        {
            // Seed face if neighbouring cell (after possible refinement)
            // will be refined one more than the current owner or neighbour.

            const refinementData& neiData = allCellInfo[faceNeighbour[facei]];

            if (maxDataCount < neiData.count())
            {
                maxDataCount = neiData.count();
            }
        }

        label faceCount = maxDataCount + maxFaceDiff;
        label faceRefineCount = faceCount + maxFaceDiff;

        seedFaces.push_back(facei);
        allFaceInfo[facei] =
            seedFacesInfo.emplace_back(faceRefineCount, faceCount);
    }


    // Just seed with all faces inbetween different refinement levels for now
    // (alternatively only seed faces on cellsToRefine but that gives problems
    //  if no cells to refine)
    forAll(faceNeighbour, facei)
    {
        // Check if face already handled in loop above
        if (!allFaceInfo[facei].valid(dummyTrackData))
        {
            label own = faceOwner[facei];
            label nei = faceNeighbour[facei];

            // Seed face with transported data from highest cell.

            if (allCellInfo[own].count() > allCellInfo[nei].count())
            {
                allFaceInfo[facei].updateFace
                (
                    mesh_,
                    facei,
                    own,
                    allCellInfo[own],
                    FaceCellWave<refinementData, int>::propagationTol(),
                    dummyTrackData
                );
                seedFaces.append(facei);
                seedFacesInfo.append(allFaceInfo[facei]);
            }
            else if (allCellInfo[own].count() < allCellInfo[nei].count())
            {
                allFaceInfo[facei].updateFace
                (
                    mesh_,
                    facei,
                    nei,
                    allCellInfo[nei],
                    FaceCellWave<refinementData, int>::propagationTol(),
                    dummyTrackData
                );
                seedFaces.append(facei);
                seedFacesInfo.append(allFaceInfo[facei]);
            }
        }
    }

    // Seed all boundary faces with owner value. This is to make sure that
    // they are visited (probably only important for coupled faces since
    // these need to be visited from both sides)
    List<refinementData> nbrCellInfo;
    syncTools::swapBoundaryCellList(mesh_, allCellInfo, nbrCellInfo);

    for (label facei = mesh_.nInternalFaces(); facei < mesh_.nFaces(); facei++)
    {
        // Check if face already handled in loop above
        if (!allFaceInfo[facei].valid(dummyTrackData))
        {
            const label own = faceOwner[facei];
            const auto& nbrInfo = nbrCellInfo[facei-mesh_.nInternalFaces()];

            if (allCellInfo[own].count() > nbrInfo.count())
            {
                allFaceInfo[facei].updateFace
                (
                    mesh_,
                    facei,
                    own,
                    allCellInfo[own],
                    FaceCellWave<refinementData, int>::propagationTol(),
                    dummyTrackData
                );
                seedFaces.append(facei);
                seedFacesInfo.append(allFaceInfo[facei]);
            }
            else if (allCellInfo[own].count() < nbrInfo.count())
            {
                allFaceInfo[facei].updateFace
                (
                    mesh_,
                    facei,
                    -1,         // Lucky! neighbCelli not used!
                    nbrInfo,
                    FaceCellWave<refinementData, int>::propagationTol(),
                    dummyTrackData
                );
                seedFaces.append(facei);
                seedFacesInfo.append(allFaceInfo[facei]);
            }
        }
    }


    // face-cell-face transport engine
    FaceCellWave<refinementData, int> levelCalc
    (
        mesh_,
        allFaceInfo,
        allCellInfo,
        dummyTrackData
    );

    while (true)
    {
        if (debug)
        {
            Pout<< "hexRef4::consistentSlowRefinement : Seeded "
                << seedFaces.size() << " faces between cells with different"
                << " refinement level." << endl;
        }

        // Set seed faces
        levelCalc.setFaceInfo(seedFaces.shrink(), seedFacesInfo.shrink());
        seedFaces.clear();
        seedFacesInfo.clear();

        // Iterate until no change. Now 2:1 face difference should be satisfied
        levelCalc.iterate(mesh_.globalData().nTotalFaces()+1);


        // Now check point-connected cells (face-connected cells already ok):
        // - get per point max of connected cells
        // - sync across coupled points
        // - check cells against above point max

        if (maxPointDiff == -1)
        {
            // No need to do any point checking.
            break;
        }

        // Determine per point the max cell level. (done as count, not
        // as cell level purely for ease)
        labelList maxPointCount(mesh_.nPoints(), Zero);

        forAll(maxPointCount, pointi)
        {
            label& pLevel = maxPointCount[pointi];

            const labelList& pCells = mesh_.pointCells(pointi);

            forAll(pCells, i)
            {
                pLevel = max(pLevel, allCellInfo[pCells[i]].count());
            }
        }

        // Sync maxPointCount to neighbour
        syncTools::syncPointList
        (
            mesh_,
            maxPointCount,
            maxEqOp<label>(),
            labelMin            // null value
        );

        // Update allFaceInfo from maxPointCount for all points to check
        // (usually on boundary faces)

        // Per face the new refinement data
        Map<refinementData> changedFacesInfo(pointsToCheck.size());

        forAll(pointsToCheck, i)
        {
            label pointi = pointsToCheck[i];

            // Loop over all cells using the point and check whether their
            // refinement level is much less than the maximum.

            const labelList& pCells = mesh_.pointCells(pointi);

            forAll(pCells, pCelli)
            {
                label celli = pCells[pCelli];

                refinementData& cellInfo = allCellInfo[celli];

                if
                (
                   !cellInfo.isRefined()
                 && (
                        maxPointCount[pointi]
                      > cellInfo.count() + maxFaceDiff*maxPointDiff
                    )
                )
                {
                    // Mark cell for refinement
                    cellInfo.count() = cellInfo.refinementCount();

                    // Insert faces of cell as seed faces.
                    const cell& cFaces = mesh_.cells()[celli];

                    forAll(cFaces, cFacei)
                    {
                        label facei = cFaces[cFacei];

                        refinementData faceData;
                        faceData.updateFace
                        (
                            mesh_,
                            facei,
                            celli,
                            cellInfo,
                            FaceCellWave<refinementData, int>::propagationTol(),
                            dummyTrackData
                        );

                        if (faceData.count() > allFaceInfo[facei].count())
                        {
                            changedFacesInfo.insert(facei, faceData);
                        }
                    }
                }
            }
        }

        label nChanged = changedFacesInfo.size();
        reduce(nChanged, sumOp<label>());

        if (nChanged == 0)
        {
            break;
        }


        // Transfer into seedFaces, seedFacesInfo
        seedFaces.setCapacity(changedFacesInfo.size());
        seedFacesInfo.setCapacity(changedFacesInfo.size());

        forAllConstIters(changedFacesInfo, iter)
        {
            seedFaces.append(iter.key());
            seedFacesInfo.append(iter.val());
        }
    }


    if (debug)
    {
        for (label facei = 0; facei < mesh_.nInternalFaces(); facei++)
        {
            label own = mesh_.faceOwner()[facei];
            label ownLevel =
                cellLevel_[own]
              + (allCellInfo[own].isRefined() ? 1 : 0);

            label nei = mesh_.faceNeighbour()[facei];
            label neiLevel =
                cellLevel_[nei]
              + (allCellInfo[nei].isRefined() ? 1 : 0);

            if (mag(ownLevel-neiLevel) > 1)
            {
                dumpCell(own);
                dumpCell(nei);
                FatalErrorInFunction
                    << "cell:" << own
                    << " current level:" << cellLevel_[own]
                    << " current refData:" << allCellInfo[own]
                    << " level after refinement:" << ownLevel
                    << nl
                    << "neighbour cell:" << nei
                    << " current level:" << cellLevel_[nei]
                    << " current refData:" << allCellInfo[nei]
                    << " level after refinement:" << neiLevel
                    << nl
                    << "which does not satisfy 2:1 constraints anymore." << nl
                    << "face:" << facei << " faceRefData:" << allFaceInfo[facei]
                    << abort(FatalError);
            }
        }


        // Coupled faces. Swap owner level to get neighbouring cell level.
        // (only boundary faces of neiLevel used)

        labelList neiLevel(mesh_.nBoundaryFaces());
        labelList neiCount(mesh_.nBoundaryFaces());
        labelList neiRefCount(mesh_.nBoundaryFaces());

        forAll(neiLevel, i)
        {
            label own = mesh_.faceOwner()[i+mesh_.nInternalFaces()];
            neiLevel[i] = cellLevel_[own];
            neiCount[i] = allCellInfo[own].count();
            neiRefCount[i] = allCellInfo[own].refinementCount();
        }

        // Swap to neighbour
        syncTools::swapBoundaryFaceList(mesh_, neiLevel);
        syncTools::swapBoundaryFaceList(mesh_, neiCount);
        syncTools::swapBoundaryFaceList(mesh_, neiRefCount);

        // Now we have neighbour value see which cells need refinement
        forAll(neiLevel, i)
        {
            label facei = i+mesh_.nInternalFaces();

            label own = mesh_.faceOwner()[facei];
            label ownLevel =
                cellLevel_[own]
              + (allCellInfo[own].isRefined() ? 1 : 0);

            label nbrLevel =
                neiLevel[i]
              + ((neiCount[i] >= neiRefCount[i]) ? 1 : 0);

            if (mag(ownLevel - nbrLevel) > 1)
            {
                dumpCell(own);
                label patchi = mesh_.boundaryMesh().whichPatch(facei);

                FatalErrorInFunction
                    << "Celllevel does not satisfy 2:1 constraint."
                    << " On coupled face "
                    << facei
                    << " refData:" << allFaceInfo[facei]
                    << " on patch " << patchi << " "
                    << mesh_.boundaryMesh()[patchi].name() << nl
                    << "owner cell " << own
                    << " current level:" << cellLevel_[own]
                    << " current count:" << allCellInfo[own].count()
                    << " current refCount:"
                    << allCellInfo[own].refinementCount()
                    << " level after refinement:" << ownLevel
                    << nl
                    << "(coupled) neighbour cell"
                    << " has current level:" << neiLevel[i]
                    << " current count:" << neiCount[i]
                    << " current refCount:" << neiRefCount[i]
                    << " level after refinement:" << nbrLevel
                    << abort(FatalError);
            }
        }
    }

    // Convert back to labelList of cells to refine.

    label nRefined = 0;

    forAll(allCellInfo, celli)
    {
        if (allCellInfo[celli].isRefined())
        {
            nRefined++;
        }
    }

    // Updated list of cells to refine
    labelList newCellsToRefine(nRefined);
    nRefined = 0;

    forAll(allCellInfo, celli)
    {
        if (allCellInfo[celli].isRefined())
        {
            newCellsToRefine[nRefined++] = celli;
        }
    }

    if (debug)
    {
        Pout<< "hexRef4::consistentSlowRefinement : From "
            << cellsToRefine.size() << " to " << newCellsToRefine.size()
            << " cells to refine." << endl;
    }

    return newCellsToRefine;
}


Foam::labelList Foam::hexRef4::consistentSlowRefinement2
(
    const label maxFaceDiff,
    const labelList& cellsToRefine,
    const labelList& facesToCheck
) const
{
    const labelList& faceOwner = mesh_.faceOwner();
    const labelList& faceNeighbour = mesh_.faceNeighbour();

    if (maxFaceDiff <= 0)
    {
        FatalErrorInFunction
            << "Illegal maxFaceDiff " << maxFaceDiff << nl
            << "Value should be >= 1" << exit(FatalError);
    }

    const scalar level0Size = 2*maxFaceDiff*level0EdgeLength();


    // Bit tricky. Say we want a distance of three cells between two
    // consecutive refinement levels. This is done by using FaceCellWave to
    // transport out the 'refinement shell'. Anything inside the refinement
    // shell (given by a distance) gets marked for refinement.

    // Initial information about (distance to) cellLevel on all cells
    List<refinementDistanceData> allCellInfo(mesh_.nCells());

    // Initial information about (distance to) cellLevel on all faces
    List<refinementDistanceData> allFaceInfo(mesh_.nFaces());

    // Dummy additional info for FaceCellWave
    int dummyTrackData = 0;


    // Mark cells with wanted refinement level
    forAll(cellsToRefine, i)
    {
        label celli = cellsToRefine[i];

        allCellInfo[celli] = refinementDistanceData
        (
            level0Size,
            mesh_.cellCentres()[celli],
            cellLevel_[celli]+1             // wanted refinement
        );
    }
    // Mark all others with existing refinement level
    forAll(allCellInfo, celli)
    {
        if (!allCellInfo[celli].valid(dummyTrackData))
        {
            allCellInfo[celli] = refinementDistanceData
            (
                level0Size,
                mesh_.cellCentres()[celli],
                cellLevel_[celli]           // wanted refinement
            );
        }
    }


    //{
    //    const fvMesh& fMesh = reinterpret_cast<const fvMesh&>(mesh_);
    //
    //    // Dump origin level
    //    volScalarField originLevel
    //    (
    //        IOobject
    //        (
    //            "originLevel_before_walk",
    //            fMesh.time().timeName(),
    //            fMesh,
    //            IOobject::NO_READ,
    //            IOobject::NO_WRITE,
    //            IOobject::NO_REGISTER
    //        ),
    //        fMesh,
    //        dimensionedScalar(dimless, Zero)
    //    );
    //
    //    forAll(originLevel, celli)
    //    {
    //        originLevel[celli] = allCellInfo[celli].originLevel();
    //    }
    //    Pout<< "Writing " << originLevel.objectPath() << endl;
    //    originLevel.write();
    //}
    //{
    //    const auto& cc = mesh_.cellCentres();
    //
    //    mkDir(mesh_.time().timePath());
    //    OBJstream os(mesh_.time().timePath()/"origin_before_walk.obj");
    //    forAll(allCellInfo, celli)
    //    {
    //        os.write(linePointRef(cc[celli], allCellInfo[celli].origin()));
    //    }
    //}


    // Labels of seed faces
    DynamicList<label> seedFaces(mesh_.nFaces()/100);
    // refinementLevel data on seed faces
    DynamicList<refinementDistanceData> seedFacesInfo(mesh_.nFaces()/100);

    const pointField& cc = mesh_.cellCentres();
    const polyBoundaryMesh& patches = mesh_.boundaryMesh();

    // Get neighbour boundary data:
    // - coupled faces      : owner data
    // - non-coupled faces  : owner level + 1 so we can treat
    pointField nbrCc(mesh_.nBoundaryFaces(), point::max);
    labelList nbrLevel(mesh_.nBoundaryFaces(), labelMax);
    bitSet isBoundary(mesh_.nFaces());
    {
        for (const polyPatch& pp : patches)
        {
            if (pp.coupled())
            {
                const auto& faceCells = pp.faceCells();
                forAll(faceCells, i)
                {
                    const label own = faceCells[i];
                    nbrCc[pp.offset()+i] = cc[own];

                    const label ownLevel =
                    (
                        allCellInfo[own].valid(dummyTrackData)
                      ? allCellInfo[own].originLevel()
                      : cellLevel_[own]
                    );
                    nbrLevel[pp.offset()+i] = ownLevel;
                }
            }
            else
            {
                isBoundary.set(pp.range());
            }
        }
        syncTools::swapBoundaryFaceList(mesh_, nbrCc);
        syncTools::swapBoundaryFaceList(mesh_, nbrLevel);
    }


    for (const label facei : facesToCheck)
    {
        if (allFaceInfo[facei].valid(dummyTrackData))
        {
            // Can only occur if face has already gone through loop below.
            FatalErrorInFunction
                << "Argument facesToCheck seems to have duplicate entries!"
                << endl
                << "face:" << facei << " occurs at positions "
                << findIndices(facesToCheck, facei)
                << abort(FatalError);
        }

        const label own = faceOwner[facei];
        const label ownLevel =
        (
            allCellInfo[own].valid(dummyTrackData)
          ? allCellInfo[own].originLevel()
          : cellLevel_[own]
        );
        const point& ownCc = cc[own];

        if (isBoundary(facei))
        {
            // Do as if boundary face would have neighbour with one higher
            // refinement level.
            const point& fc = mesh_.faceCentres()[facei];

            refinementDistanceData neiData
            (
                level0Size,
                2*fc - cc[own],    // est'd cell centre
                ownLevel+1
            );

            allFaceInfo[facei].updateFace
            (
                mesh_,
                facei,
                own,        // not used (should be nei)
                neiData,
                FaceCellWave<refinementDistanceData, int>::propagationTol(),
                dummyTrackData
            );
        }
        else
        {
            label neiLevel;
            point neiCc;
            if (mesh_.isInternalFace(facei))
            { 
                const label nei = faceNeighbour[facei];
                neiLevel =
                (
                    allCellInfo[nei].valid(dummyTrackData)
                  ? allCellInfo[nei].originLevel()
                  : cellLevel_[nei]
                );
                neiCc = cc[nei];
            }
            else
            {
                neiLevel = nbrLevel[facei-mesh_.nInternalFaces()];
                neiCc = nbrCc[facei-mesh_.nInternalFaces()];
            }

            if (ownLevel == neiLevel)
            {
                // Fake as if nei>own or own>nei (whichever one 'wins')
                allFaceInfo[facei].updateFace
                (
                    mesh_,
                    facei,
                    own,    // not used, should be nei
                    refinementDistanceData(level0Size, neiCc, neiLevel+1),
                    FaceCellWave<refinementDistanceData, int>::propagationTol(),
                    dummyTrackData
                );
                allFaceInfo[facei].updateFace
                (
                    mesh_,
                    facei,
                    own,    // not used
                    refinementDistanceData(level0Size, ownCc, ownLevel+1),
                    FaceCellWave<refinementDistanceData, int>::propagationTol(),
                    dummyTrackData
                );
            }
            else
            {
                // Difference in level anyway.
                allFaceInfo[facei].updateFace
                (
                    mesh_,
                    facei,
                    own,    // not used, should be nei
                    refinementDistanceData(level0Size, neiCc, neiLevel),
                    FaceCellWave<refinementDistanceData, int>::propagationTol(),
                    dummyTrackData
                );
                allFaceInfo[facei].updateFace
                (
                    mesh_,
                    facei,
                    own,    // not used
                    refinementDistanceData(level0Size, ownCc, ownLevel),
                    FaceCellWave<refinementDistanceData, int>::propagationTol(),
                    dummyTrackData
                );
            }
        }
        seedFaces.append(facei);
        seedFacesInfo.append(allFaceInfo[facei]);
    }



    // Create some initial seeds to start walking from. This is only if there
    // are no facesToCheck.
    // Just seed with all faces inbetween different refinement levels for now
    // Note: no need to handle coupled faces since FaceCellWave below
    //       already swaps seedInfo upon start
    //forAll(faceNeighbour, facei)
    forAll(faceOwner, facei)
    {
        // Check if face already handled in loop above
        if (!allFaceInfo[facei].valid(dummyTrackData) && !isBoundary(facei))
        {
            const label own = faceOwner[facei];
            const label ownLevel =
            (
                allCellInfo[own].valid(dummyTrackData)
              ? allCellInfo[own].originLevel()
              : cellLevel_[own]
            );
            const point& ownCc = cc[own];

            label neiLevel;
            point neiCc;
            if (mesh_.isInternalFace(facei))
            { 
                const label nei = faceNeighbour[facei];
                neiLevel =
                (
                    allCellInfo[nei].valid(dummyTrackData)
                  ? allCellInfo[nei].originLevel()
                  : cellLevel_[nei]
                );
                neiCc = cc[nei];
            }
            else
            {
                neiLevel = nbrLevel[facei-mesh_.nInternalFaces()];
                neiCc = nbrCc[facei-mesh_.nInternalFaces()];
            }

            if (ownLevel > neiLevel)
            {
                // Set face to owner data. (since face not yet would be copy)
                seedFaces.append(facei);
                allFaceInfo[facei].updateFace
                (
                    mesh_,
                    facei,
                    own,
                    refinementDistanceData(level0Size, ownCc, ownLevel),
                    FaceCellWave<refinementDistanceData, int>::propagationTol(),
                    dummyTrackData
                );
                seedFacesInfo.append(allFaceInfo[facei]);
            }
            else if (neiLevel > ownLevel)
            {
                seedFaces.append(facei);
                allFaceInfo[facei].updateFace
                (
                    mesh_,
                    facei,
                    own,    // not used, should be nei,
                    refinementDistanceData(level0Size, neiCc, neiLevel),
                    FaceCellWave<refinementDistanceData, int>::propagationTol(),
                    dummyTrackData
                );
                seedFacesInfo.append(allFaceInfo[facei]);
            }
        }
    }

    seedFaces.shrink();
    seedFacesInfo.shrink();

    // face-cell-face transport engine
    FaceCellWave<refinementDistanceData, int> levelCalc
    (
        mesh_,
        seedFaces,
        seedFacesInfo,
        allFaceInfo,
        allCellInfo,
        mesh_.globalData().nTotalCells()+1,
        dummyTrackData
    );


    //- noted: origin is different face (? or cell) between non-parallel
    //         and parallel
    //{
    //    const auto& cc = mesh_.cellCentres();
    //
    //    mkDir(mesh_.time().timePath());
    //    OBJstream os(mesh_.time().timePath()/"origin_after_walk.obj");
    //    forAll(allCellInfo, celli)
    //    {
    //        os.write(linePointRef(cc[celli], allCellInfo[celli].origin()));
    //    }
    //}



    //if (debug)
    //{
    //    const fvMesh& fMesh = reinterpret_cast<const fvMesh&>(mesh_);
    //
    //    // Dump origin level
    //    volScalarField originLevel
    //    (
    //        IOobject
    //        (
    //            "originLevel_after_walk",
    //            fMesh.time().timeName(),
    //            fMesh,
    //            IOobject::NO_READ,
    //            IOobject::NO_WRITE,
    //            IOobject::NO_REGISTER
    //        ),
    //        fMesh,
    //        dimensionedScalar(dimless, Zero)
    //    );
    //
    //    forAll(originLevel, celli)
    //    {
    //        originLevel[celli] = allCellInfo[celli].originLevel();
    //    }
    //    // Dump wanted level
    //    volScalarField wantedLevel
    //    (
    //        IOobject
    //        (
    //            "wantedLevel",
    //            fMesh.time().timeName(),
    //            fMesh,
    //            IOobject::NO_READ,
    //            IOobject::NO_WRITE,
    //            IOobject::NO_REGISTER
    //        ),
    //        fMesh,
    //        dimensionedScalar(dimless, Zero)
    //    );
    //
    //    forAll(wantedLevel, celli)
    //    {
    //        wantedLevel[celli] = allCellInfo[celli].wantedLevel(cc[celli]);
    //    }
    //
    //    Pout<< "Writing " << originLevel.objectPath() << endl;
    //    //fMesh.write();
    //    originLevel.write();
    //    Pout<< "Writing " << wantedLevel.objectPath() << endl;
    //    wantedLevel.write();
    //}


    // Convert back to labelList of cells to refine.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // 1. Force original refinement cells to be picked up by setting the
    // originLevel of input cells to be a very large level (but within range
    // of 1<< shift inside refinementDistanceData::wantedLevel)
    forAll(cellsToRefine, i)
    {
        label celli = cellsToRefine[i];

        allCellInfo[celli].originLevel() = sizeof(label)*8-2;
        allCellInfo[celli].origin() = cc[celli];
    }

    // 2. Extend to 2:1. I don't understand yet why this is not done
    // 2. Extend to 2:1. For non-cube cells the scalar distance does not work
    // so make sure it at least provides 2:1.
    bitSet refineCell(mesh_.nCells());
    forAll(allCellInfo, celli)
    {
        label wanted = allCellInfo[celli].wantedLevel(cc[celli]);

        if (wanted > cellLevel_[celli]+1)
        {
            refineCell.set(celli);
        }
    }
    faceConsistentRefinement(true, cellLevel_, refineCell);

    while (true)
    {
        label nChanged = faceConsistentRefinement(true, cellLevel_, refineCell);

        reduce(nChanged, sumOp<label>());

        if (debug)
        {
            Pout<< "hexRef4::consistentSlowRefinement2 : Changed " << nChanged
                << " refinement levels due to 2:1 conflicts."
                << endl;
        }

        if (nChanged == 0)
        {
            break;
        }
    }

    // 3. Convert back to labelList.
    labelList newCellsToRefine(refineCell.toc());

    if (debug)
    {
        Pout<< "hexRef4::consistentSlowRefinement2 : From "
            << cellsToRefine.size() << " to " << newCellsToRefine.size()
            << " cells to refine." << endl;

        // Check that newCellsToRefine obeys at least 2:1.

        {
            cellSet cellsIn(mesh_, "cellsToRefineIn", cellsToRefine);
            Pout<< "hexRef4::consistentSlowRefinement2 : writing "
                << cellsIn.size() << " to cellSet "
                << cellsIn.objectPath() << endl;
            cellsIn.write();
        }
        {
            cellSet cellsOut(mesh_, "cellsToRefineOut", newCellsToRefine);
            Pout<< "hexRef4::consistentSlowRefinement2 : writing "
                << cellsOut.size() << " to cellSet "
                << cellsOut.objectPath() << endl;
            cellsOut.write();
        }

        // Extend to 2:1
        bitSet refineCell(mesh_.nCells(), newCellsToRefine);

        const bitSet savedRefineCell(refineCell);
        label nChanged = faceConsistentRefinement(true, cellLevel_, refineCell);

        {
            cellSet cellsOut2
            (
                mesh_, "cellsToRefineOut2", newCellsToRefine.size()
            );
            forAll(refineCell, celli)
            {
                if (refineCell.test(celli))
                {
                    cellsOut2.insert(celli);
                }
            }
            Pout<< "hexRef4::consistentSlowRefinement2 : writing "
                << cellsOut2.size() << " to cellSet "
                << cellsOut2.objectPath() << endl;
            cellsOut2.write();
        }

        if (nChanged > 0)
        {
            forAll(refineCell, celli)
            {
                if (refineCell.test(celli) && !savedRefineCell.test(celli))
                {
                    dumpCell(celli);
                    FatalErrorInFunction
                        << "Cell:" << celli << " cc:"
                        << mesh_.cellCentres()[celli]
                        << " was not marked for refinement but does not obey"
                        << " 2:1 constraints."
                        << abort(FatalError);
                }
            }
        }
    }

    return newCellsToRefine;
}


// Top level driver to insert topo changes to do all refinement.
Foam::labelListList Foam::hexRef4::setRefinement
(
    const labelList& cellLabels,
    polyTopoChange& meshMod
)
{
    if (debug)
    {
        Pout<< "hexRef4::setRefinement :"
            << " Checking initial mesh just to make sure" << endl;

        checkMesh();
        // Cannot call checkRefinementlevels since hanging points might
        // get triggered by the mesher after subsetting.
        //checkRefinementLevels(-1, labelList(0));
    }

    // Clear any saved point/cell data.
    savedPointLevel_.clear();
    savedCellLevel_.clear();


    // New point/cell level. Copy of pointLevel for existing points.
    DynamicList<label> newCellLevel(cellLevel_.size());
    forAll(cellLevel_, celli)
    {
        newCellLevel.append(cellLevel_[celli]);
    }
    DynamicList<label> newPointLevel(pointLevel_.size());
    forAll(pointLevel_, pointi)
    {
        newPointLevel.append(pointLevel_[pointi]);
    }


    // Cells to split
    // ~~~~~~~~~~~~~~
    // Unlike hexRef8 no point is introduced in the cell interior: the two
    // span face mid points play that role, so the set of cells to refine is
    // just a marker.

    bitSet refineCell(mesh_.nCells(), cellLabels);

    if (debug)
    {
        cellSet splitCells(mesh_, "splitCells", cellLabels.size());

        for (const label celli : refineCell)
        {
            splitCells.insert(celli);
        }

        Pout<< "hexRef4::setRefinement : Dumping " << splitCells.size()
            << " cells to split to cellSet " << splitCells.objectPath()
            << endl;

        splitCells.write();
    }


    // Split edges
    // ~~~~~~~~~~~
    // Only edges lying in the 2-D plane. Edges along the span direction are
    // never split - that is what keeps the mesh exactly one cell thick and
    // the empty patches valid.

    if (debug)
    {
        Pout<< "hexRef4::setRefinement :"
            << " Allocating in-plane edge midpoints."
            << endl;
    }

    // -1  : no need to split edge
    // >=0 : label of introduced mid point
    labelList edgeMidPoint(mesh_.nEdges(), -1);

    for (const label celli : refineCell)
    {
        const labelList& cEdges = mesh_.cellEdges(celli);

        forAll(cEdges, i)
        {
            const label edgeI = cEdges[i];

            const edge& e = mesh_.edges()[edgeI];

            if
            (
                pointLevel_[e[0]] <= cellLevel_[celli]
             && pointLevel_[e[1]] <= cellLevel_[celli]
             && !isSpanEdge(edgeI)
            )
            {
                edgeMidPoint[edgeI] = 12345;    // mark need for splitting
            }
        }
    }

    // Synchronize edgeMidPoint across coupled patches. Take max so that
    // any split takes precedence.
    syncTools::syncEdgeList
    (
        mesh_,
        edgeMidPoint,
        maxEqOp<label>(),
        labelMin
    );


    // Introduce edge points
    // ~~~~~~~~~~~~~~~~~~~~~

    {
        // Phase 1: calculate midpoints and sync.
        // This needs doing for if people do not write binary and we slowly
        // get differences.

        pointField edgeMids(mesh_.nEdges(), point(-GREAT, -GREAT, -GREAT));

        forAll(edgeMidPoint, edgeI)
        {
            if (edgeMidPoint[edgeI] >= 0)
            {
                // Edge marked to be split.
                edgeMids[edgeI] = mesh_.edges()[edgeI].centre(mesh_.points());
            }
        }
        syncTools::syncEdgePositions
        (
            mesh_,
            edgeMids,
            maxEqOp<vector>(),
            point(-GREAT, -GREAT, -GREAT)
        );


        // Phase 2: introduce points at the synced locations.
        forAll(edgeMidPoint, edgeI)
        {
            if (edgeMidPoint[edgeI] >= 0)
            {
                // Edge marked to be split. Replace edgeMidPoint with actual
                // point label.

                const edge& e = mesh_.edges()[edgeI];

                edgeMidPoint[edgeI] = meshMod.setAction
                (
                    polyAddPoint
                    (
                        edgeMids[edgeI],            // point
                        e[0],                       // master point
                        -1,                         // zone for point
                        true                        // supports a cell
                    )
                );

                newPointLevel(edgeMidPoint[edgeI]) =
                    max
                    (
                        pointLevel_[e[0]],
                        pointLevel_[e[1]]
                    )
                  + 1;
            }
        }
    }

    if (debug)
    {
        OFstream str(mesh_.time().path()/"edgeMidPoint.obj");

        forAll(edgeMidPoint, edgeI)
        {
            if (edgeMidPoint[edgeI] >= 0)
            {
                const edge& e = mesh_.edges()[edgeI];

                meshTools::writeOBJ(str, e.centre(mesh_.points()));
            }
        }

        Pout<< "hexRef4::setRefinement :"
            << " Dumping edge centres to split to file " << str.name() << endl;
    }


    // Calculate face level
    // ~~~~~~~~~~~~~~~~~~~~
    // (after splitting)

    if (debug)
    {
        Pout<< "hexRef4::setRefinement :"
            << " Marking faces to split."
            << endl;
    }

    // Face anchor level. There are guaranteed 4 points with level
    // <= anchorLevel. These are the corner points.
    labelList faceAnchorLevel(mesh_.nFaces());

    for (label facei = 0; facei < mesh_.nFaces(); facei++)
    {
        faceAnchorLevel[facei] = faceLevel(facei);
    }

    // Faces that have to be split. The criterion is exactly hexRef8's, but
    // the resulting split differs by face orientation:
    //  - span face (normal along the span direction): split into 4 about a
    //    new mid point;
    //  - side face: bisected into 2, keeping its full span extent.
    labelList faceNeedsSplit(mesh_.nFaces(), Zero);

    // Internal faces: look at cells on both sides. Uniquely determined since
    // face itself guaranteed to be same level as most refined neighbour.
    for (label facei = 0; facei < mesh_.nInternalFaces(); facei++)
    {
        if (faceAnchorLevel[facei] >= 0)
        {
            label own = mesh_.faceOwner()[facei];
            label ownLevel = cellLevel_[own];
            label newOwnLevel = ownLevel + (refineCell.test(own) ? 1 : 0);

            label nei = mesh_.faceNeighbour()[facei];
            label neiLevel = cellLevel_[nei];
            label newNeiLevel = neiLevel + (refineCell.test(nei) ? 1 : 0);

            if
            (
                newOwnLevel > faceAnchorLevel[facei]
             || newNeiLevel > faceAnchorLevel[facei]
            )
            {
                faceNeedsSplit[facei] = 1;
            }
        }
    }

    // Coupled patches handled like internal faces except now all information
    // from neighbour comes from across processor.
    // Boundary faces are more complicated since the boundary face can
    // be more refined than its owner (or neighbour for coupled patches)
    // (does not happen if refining/unrefining only, but does e.g. when
    //  refinining and subsetting)

    {
        labelList newNeiLevel(mesh_.nBoundaryFaces());

        forAll(newNeiLevel, i)
        {
            label own = mesh_.faceOwner()[i+mesh_.nInternalFaces()];
            label ownLevel = cellLevel_[own];
            label newOwnLevel = ownLevel + (refineCell.test(own) ? 1 : 0);

            newNeiLevel[i] = newOwnLevel;
        }

        // Swap.
        syncTools::swapBoundaryFaceList(mesh_, newNeiLevel);

        // So now we have information on the neighbour.

        forAll(newNeiLevel, i)
        {
            label facei = i+mesh_.nInternalFaces();

            if (faceAnchorLevel[facei] >= 0)
            {
                label own = mesh_.faceOwner()[facei];
                label ownLevel = cellLevel_[own];
                label newOwnLevel = ownLevel + (refineCell.test(own) ? 1 : 0);

                if
                (
                    newOwnLevel > faceAnchorLevel[facei]
                 || newNeiLevel[i] > faceAnchorLevel[facei]
                )
                {
                    faceNeedsSplit[facei] = 1;
                }
            }
        }
    }


    // Synchronize across coupled patches. (logical or)
    syncTools::syncFaceList
    (
        mesh_,
        faceNeedsSplit,
        maxEqOp<label>()
    );


    // Introduce span face mid points
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Side faces do not get one: they are bisected using the mid points of
    // their two split in-plane edges.

    // -1  : face not split into four
    // >=0 : label of introduced mid point
    labelList faceMidPoint(mesh_.nFaces(), -1);

    {
        // Phase 1: determine mid points and sync. See comment for edgeMids
        // above
        pointField bFaceMids
        (
            mesh_.nBoundaryFaces(),
            point(-GREAT, -GREAT, -GREAT)
        );

        forAll(bFaceMids, i)
        {
            label facei = i+mesh_.nInternalFaces();

            if (faceNeedsSplit[facei] && isSpanFace(facei))
            {
                bFaceMids[i] = mesh_.faceCentres()[facei];
            }
        }
        syncTools::syncBoundaryFacePositions
        (
            mesh_,
            bFaceMids,
            maxEqOp<vector>()
        );

        forAll(faceNeedsSplit, facei)
        {
            if (faceNeedsSplit[facei] && isSpanFace(facei))
            {
                // Span face marked to be split into four.

                const face& f = mesh_.faces()[facei];

                faceMidPoint[facei] = meshMod.setAction
                (
                    polyAddPoint
                    (
                        (
                            facei < mesh_.nInternalFaces()
                          ? mesh_.faceCentres()[facei]
                          : bFaceMids[facei-mesh_.nInternalFaces()]
                        ),                          // point
                        f[0],                       // master point
                        -1,                         // zone for point
                        true                        // supports a cell
                    )
                );

                // Determine the level of the corner points and midpoint will
                // be one higher.
                newPointLevel(faceMidPoint[facei]) = faceAnchorLevel[facei]+1;
            }
        }
    }

    if (debug)
    {
        faceSet splitFaces(mesh_, "splitFaces", cellLabels.size());

        forAll(faceNeedsSplit, facei)
        {
            if (faceNeedsSplit[facei])
            {
                splitFaces.insert(facei);
            }
        }

        Pout<< "hexRef4::setRefinement : Dumping " << splitFaces.size()
            << " faces to split to faceSet " << splitFaces.objectPath() << endl;

        splitFaces.write();
    }


    // Information complete
    // ~~~~~~~~~~~~~~~~~~~~
    // At this point we have all the information we need. We should no
    // longer reference the cellLabels to refine. All the information is:
    // - refineCell        : cell needs to be split into four
    // - faceNeedsSplit    : face needs to be split
    // - faceMidPoint >= 0 : span face, split into four about this point
    // - edgeMidPoint >= 0 : in-plane edge needs to be split



    // Get the corner/anchor points
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if (debug)
    {
        Pout<< "hexRef4::setRefinement :"
            << " Finding cell anchorPoints (8 per cell)"
            << endl;
    }

    // There will always be 8 points on the hex that were introduced
    // with the hex and will have the same or lower refinement level. They map
    // onto only 4 sub-cells: two anchors connected by a span edge end up in
    // the same one.

    // Per cell the 8 corner points.
    labelListList cellAnchorPoints(mesh_.nCells());

    {
        labelList nAnchorPoints(mesh_.nCells(), Zero);

        for (const label celli : refineCell)
        {
            cellAnchorPoints[celli].setSize(8);
        }

        forAll(pointLevel_, pointi)
        {
            const labelList& pCells = mesh_.pointCells(pointi);

            forAll(pCells, pCelli)
            {
                label celli = pCells[pCelli];

                if
                (
                    refineCell.test(celli)
                 && pointLevel_[pointi] <= cellLevel_[celli]
                )
                {
                    if (nAnchorPoints[celli] == 8)
                    {
                        dumpCell(celli);
                        FatalErrorInFunction
                            << "cell " << celli
                            << " of level " << cellLevel_[celli]
                            << " uses more than 8 points of equal or"
                            << " lower level" << nl
                            << "Points so far:" << cellAnchorPoints[celli]
                            << abort(FatalError);
                    }

                    cellAnchorPoints[celli][nAnchorPoints[celli]++] = pointi;
                }
            }
        }

        for (const label celli : refineCell)
        {
            if (nAnchorPoints[celli] != 8)
            {
                dumpCell(celli);

                const labelList& cPoints = mesh_.cellPoints(celli);

                FatalErrorInFunction
                    << "cell " << celli
                    << " of level " << cellLevel_[celli]
                    << " does not seem to have 8 points of equal or"
                    << " lower level" << endl
                    << "cellPoints:" << cPoints << endl
                    << "pointLevels:"
                    << labelUIndList(pointLevel_, cPoints) << endl
                    << abort(FatalError);
            }
        }
    }


    // Add the cells
    // ~~~~~~~~~~~~~
    // Four per cell, but indexed by all 8 anchor points so that the
    // owner/neighbour bookkeeping inherited from hexRef8 keeps working
    // unchanged: cellAddedCells[celli][i] is the sub-cell that anchor
    // cellAnchorPoints[celli][i] ends up in, and span partners share one.

    if (debug)
    {
        Pout<< "hexRef4::setRefinement :"
            << " Adding cells (1 per pair of span-partner anchorPoints)"
            << endl;
    }

    // Per cell the sub-cell per anchor point (8 entries, 4 distinct)
    labelListList cellAddedCells(mesh_.nCells());

    // Per cell the 3 added cells (+ original cell), in creation order
    labelListList cellSubCells(mesh_.nCells());

    for (const label celli : refineCell)
    {
        const labelList& cAnchors = cellAnchorPoints[celli];

        labelList& cAdded = cellAddedCells[celli];
        cAdded.setSize(8, -1);

        DynamicList<label> subCells(4);

        // Update cell level
        newCellLevel[celli] = cellLevel_[celli]+1;

        forAll(cAnchors, i)
        {
            if (cAdded[i] != -1)
            {
                continue;
            }

            // Find the anchor directly across the span
            label partneri = -1;

            const labelList& pEdges = mesh_.pointEdges()[cAnchors[i]];

            forAll(pEdges, k)
            {
                const label edgeI = pEdges[k];

                if (!isSpanEdge(edgeI))
                {
                    continue;
                }

                const label otherPointi =
                    mesh_.edges()[edgeI].otherVertex(cAnchors[i]);

                const label index = cAnchors.find(otherPointi);

                if (index != -1 && cAdded[index] == -1)
                {
                    partneri = index;
                    break;
                }
            }

            if (partneri == -1)
            {
                dumpCell(celli);

                FatalErrorInFunction
                    << "Anchor point " << cAnchors[i]
                    << " at " << mesh_.points()[cAnchors[i]]
                    << " of cell " << celli
                    << " has no span-direction partner among the cell's"
                    << " anchor points " << cAnchors << nl
                    << "Is the mesh one cell thick along component "
                    << label(spanDir_) << "?"
                    << abort(FatalError);
            }

            label newCelli = -1;

            if (subCells.empty())
            {
                // Original cell is re-used as the first sub-cell
                newCelli = celli;
            }
            else
            {
                newCelli = meshMod.setAction
                (
                    polyAddCell
                    (
                        -1,                                 // master point
                        -1,                                 // master edge
                        -1,                                 // master face
                        celli,                              // master cell
                        mesh_.cellZones().whichZone(celli)  // zone for cell
                    )
                );

                newCellLevel(newCelli) = cellLevel_[celli]+1;
            }

            subCells.append(newCelli);

            cAdded[i] = newCelli;
            cAdded[partneri] = newCelli;
        }

        if (subCells.size() != 4)
        {
            dumpCell(celli);

            FatalErrorInFunction
                << "Cell " << celli << " was split into " << subCells.size()
                << " cells instead of 4. Anchor points " << cAnchors
                << " do not form 4 span-direction pairs."
                << abort(FatalError);
        }

        cellSubCells[celli].transfer(subCells);
    }


    // Faces
    // ~~~~~
    // 1. existing span faces that get split (into four)
    // 2. existing side faces that get bisected (into two)
    // 3. existing faces that do not get split but only edges get split
    // 4. existing faces that do not get split but get new owner/neighbour
    // 5. new internal faces inside split cells.

    if (debug)
    {
        Pout<< "hexRef4::setRefinement :"
            << " Marking faces to be handled"
            << endl;
    }

    // Get all affected faces.
    bitSet affectedFace(mesh_.nFaces());

    {
        for (const label celli : refineCell)
        {
            const cell& cFaces = mesh_.cells()[celli];

            affectedFace.set(cFaces);
        }

        forAll(faceNeedsSplit, facei)
        {
            if (faceNeedsSplit[facei])
            {
                affectedFace.set(facei);
            }
        }

        forAll(edgeMidPoint, edgeI)
        {
            if (edgeMidPoint[edgeI] >= 0)
            {
                const labelList& eFaces = mesh_.edgeFaces(edgeI);

                affectedFace.set(eFaces);
            }
        }
    }


    // 1. Span faces that get split into four
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if (debug)
    {
        Pout<< "hexRef4::setRefinement : Splitting span faces" << endl;
    }

    forAll(faceMidPoint, facei)
    {
        if (faceMidPoint[facei] >= 0 && affectedFace.test(facei))
        {
            // Face needs to be split and hasn't yet been done in some way
            // (affectedFace - is impossible since this is first change but
            //  just for completeness)

            const face& f = mesh_.faces()[facei];

            // Has original facei been used (three faces added, original gets
            // modified)
            bool modifiedFace = false;

            label anchorLevel = faceAnchorLevel[facei];

            face newFace(4);

            forAll(f, fp)
            {
                label pointi = f[fp];

                if (pointLevel_[pointi] <= anchorLevel)
                {
                    // point is anchor. Start collecting face.

                    DynamicList<label> faceVerts(4);

                    faceVerts.append(pointi);

                    // Walk forward to mid point.
                    // - if next is +2 midpoint is +1
                    // - if next is +1 it is midpoint
                    // - if next is +0 there has to be edgeMidPoint

                    walkFaceToMid
                    (
                        edgeMidPoint,
                        anchorLevel,
                        facei,
                        fp,
                        faceVerts
                    );

                    faceVerts.append(faceMidPoint[facei]);

                    walkFaceFromMid
                    (
                        edgeMidPoint,
                        anchorLevel,
                        facei,
                        fp,
                        faceVerts
                    );

                    // Convert dynamiclist to face.
                    newFace.transfer(faceVerts);

                    // Get new owner/neighbour
                    label own, nei;
                    getFaceNeighbours
                    (
                        cellAnchorPoints,
                        cellAddedCells,
                        facei,
                        pointi,          // Anchor point

                        own,
                        nei
                    );


                    if (debug)
                    {
                        if (mesh_.isInternalFace(facei))
                        {
                            label oldOwn = mesh_.faceOwner()[facei];
                            label oldNei = mesh_.faceNeighbour()[facei];

                            checkInternalOrientation
                            (
                                meshMod,
                                oldOwn,
                                facei,
                                mesh_.cellCentres()[oldOwn],
                                mesh_.cellCentres()[oldNei],
                                newFace
                            );
                        }
                        else
                        {
                            label oldOwn = mesh_.faceOwner()[facei];

                            checkBoundaryOrientation
                            (
                                meshMod,
                                oldOwn,
                                facei,
                                mesh_.cellCentres()[oldOwn],
                                mesh_.faceCentres()[facei],
                                newFace
                            );
                        }
                    }


                    if (!modifiedFace)
                    {
                        modifiedFace = true;

                        modFace(meshMod, facei, newFace, own, nei);
                    }
                    else
                    {
                        addFace(meshMod, facei, newFace, own, nei);
                    }
                }
            }

            // Mark face as having been handled
            affectedFace.unset(facei);
        }
    }


    // 2. Side faces that get bisected
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if (debug)
    {
        Pout<< "hexRef4::setRefinement : Bisecting side faces" << endl;
    }

    forAll(faceNeedsSplit, facei)
    {
        if
        (
            faceNeedsSplit[facei]
         && faceMidPoint[facei] < 0
         && affectedFace.test(facei)
        )
        {
            splitSideFace
            (
                cellAnchorPoints,
                cellAddedCells,
                edgeMidPoint,
                facei,
                meshMod
            );

            // Mark face as having been handled
            affectedFace.unset(facei);
        }
    }


    // 3. faces that do not get split but use edges that get split
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if (debug)
    {
        Pout<< "hexRef4::setRefinement :"
            << " Adding edge splits to unsplit faces"
            << endl;
    }

    DynamicList<label> eFacesStorage;
    DynamicList<label> fEdgesStorage;

    forAll(edgeMidPoint, edgeI)
    {
        if (edgeMidPoint[edgeI] >= 0)
        {
            // Split edge. Check that face not already handled above.

            const labelList& eFaces = mesh_.edgeFaces(edgeI, eFacesStorage);

            forAll(eFaces, i)
            {
                label facei = eFaces[i];

                if (!faceNeedsSplit[facei] && affectedFace.test(facei))
                {
                    // Unsplit face. Add edge splits to face.

                    const face& f = mesh_.faces()[facei];
                    const labelList& fEdges = mesh_.faceEdges
                    (
                        facei,
                        fEdgesStorage
                    );

                    DynamicList<label> newFaceVerts(f.size());

                    forAll(f, fp)
                    {
                        newFaceVerts.append(f[fp]);

                        label edgeI = fEdges[fp];

                        if (edgeMidPoint[edgeI] >= 0)
                        {
                            newFaceVerts.append(edgeMidPoint[edgeI]);
                        }
                    }

                    face newFace;
                    newFace.transfer(newFaceVerts);

                    // The point with the lowest level should be an anchor
                    // point of the neighbouring cells.
                    label anchorFp = findMinLevel(f);

                    label own, nei;
                    getFaceNeighbours
                    (
                        cellAnchorPoints,
                        cellAddedCells,
                        facei,
                        f[anchorFp],          // Anchor point

                        own,
                        nei
                    );


                    if (debug)
                    {
                        if (mesh_.isInternalFace(facei))
                        {
                            label oldOwn = mesh_.faceOwner()[facei];
                            label oldNei = mesh_.faceNeighbour()[facei];

                            checkInternalOrientation
                            (
                                meshMod,
                                oldOwn,
                                facei,
                                mesh_.cellCentres()[oldOwn],
                                mesh_.cellCentres()[oldNei],
                                newFace
                            );
                        }
                        else
                        {
                            label oldOwn = mesh_.faceOwner()[facei];

                            checkBoundaryOrientation
                            (
                                meshMod,
                                oldOwn,
                                facei,
                                mesh_.cellCentres()[oldOwn],
                                mesh_.faceCentres()[facei],
                                newFace
                            );
                        }
                    }

                    modFace(meshMod, facei, newFace, own, nei);

                    // Mark face as having been handled
                    affectedFace.unset(facei);
                }
            }
        }
    }


    // 4. faces that do not get split but whose owner/neighbour change
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if (debug)
    {
        Pout<< "hexRef4::setRefinement :"
            << " Changing owner/neighbour for otherwise unaffected faces"
            << endl;
    }

    forAll(affectedFace, facei)
    {
        if (affectedFace.test(facei))
        {
            const face& f = mesh_.faces()[facei];

            // The point with the lowest level should be an anchor
            // point of the neighbouring cells.
            label anchorFp = findMinLevel(f);

            label own, nei;
            getFaceNeighbours
            (
                cellAnchorPoints,
                cellAddedCells,
                facei,
                f[anchorFp],          // Anchor point

                own,
                nei
            );

            modFace(meshMod, facei, f, own, nei);

            // Mark face as having been handled
            affectedFace.unset(facei);
        }
    }


    // 5. new internal faces inside split cells.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if (debug)
    {
        Pout<< "hexRef4::setRefinement :"
            << " Create new internal faces for split cells"
            << endl;
    }

    for (const label celli : refineCell)
    {
        createInternalFaces
        (
            cellAnchorPoints,
            cellAddedCells,
            faceMidPoint,
            edgeMidPoint,
            celli,
            meshMod
        );
    }

    // Extend pointLevels and cellLevels for the new cells. Could also be done
    // in updateMesh but saves passing cellAddedCells out of this routine.

    // Check
    if (debug)
    {
        label minPointi = labelMax;
        label maxPointi = labelMin;

        forAll(faceMidPoint, facei)
        {
            if (faceMidPoint[facei] >= 0)
            {
                minPointi = min(minPointi, faceMidPoint[facei]);
                maxPointi = max(maxPointi, faceMidPoint[facei]);
            }
        }
        forAll(edgeMidPoint, edgeI)
        {
            if (edgeMidPoint[edgeI] >= 0)
            {
                minPointi = min(minPointi, edgeMidPoint[edgeI]);
                maxPointi = max(maxPointi, edgeMidPoint[edgeI]);
            }
        }

        if (minPointi != labelMax && minPointi != mesh_.nPoints())
        {
            FatalErrorInFunction
                << "Added point labels not consecutive to existing mesh points."
                << nl
                << "mesh_.nPoints():" << mesh_.nPoints()
                << " minPointi:" << minPointi
                << " maxPointi:" << maxPointi
                << abort(FatalError);
        }
    }

    pointLevel_.transfer(newPointLevel);
    cellLevel_.transfer(newCellLevel);

    // Mark files as changed
    setInstance(mesh_.facesInstance());


    // Update the live split cells tree.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // New unrefinement structure
    if (history_.active())
    {
        if (debug)
        {
            Pout<< "hexRef4::setRefinement :"
                << " Updating refinement history to " << cellLevel_.size()
                << " cells" << endl;
        }

        // Extend refinement history for new cells
        history_.resize(cellLevel_.size());

        forAll(cellSubCells, celli)
        {
            const labelList& subCells = cellSubCells[celli];

            if (subCells.size())
            {
                // Cell was split. Only 4 of the 8 slots of the underlying
                // splitCell8 get used; the rest stay at -1, which
                // refinementHistory tolerates.
                history_.storeSplit(celli, subCells);
            }
        }
    }

    // Compact cellSubCells.

    labelListList refinedCells(cellLabels.size());

    forAll(cellLabels, i)
    {
        label celli = cellLabels[i];

        refinedCells[i].transfer(cellSubCells[celli]);
    }

    return refinedCells;
}


void Foam::hexRef4::storeData
(
    const labelList& pointsToStore,
    const labelList& facesToStore,
    const labelList& cellsToStore
)
{
    savedPointLevel_.clear();
    savedPointLevel_.reserve(pointsToStore.size());
    forAll(pointsToStore, i)
    {
        label pointi = pointsToStore[i];
        savedPointLevel_.insert(pointi, pointLevel_[pointi]);
    }

    savedCellLevel_.clear();
    savedCellLevel_.reserve(cellsToStore.size());
    forAll(cellsToStore, i)
    {
        label celli = cellsToStore[i];
        savedCellLevel_.insert(celli, cellLevel_[celli]);
    }
}


// Gets called after the mesh change. setRefinement will already have made
// sure the pointLevel_ and cellLevel_ are the size of the new mesh so we
// only need to account for reordering.
void Foam::hexRef4::updateMesh(const mapPolyMesh& map)
{
    Map<label> dummyMap(0);

    updateMesh(map, dummyMap, dummyMap, dummyMap);
}


// Gets called after the mesh change. setRefinement will already have made
// sure the pointLevel_ and cellLevel_ are the size of the new mesh so we
// only need to account for reordering.
void Foam::hexRef4::updateMesh
(
    const mapPolyMesh& map,
    const Map<label>& pointsToRestore,
    const Map<label>& facesToRestore,
    const Map<label>& cellsToRestore
)
{
    // Update celllevel
    if (debug)
    {
        Pout<< "hexRef4::updateMesh :"
            << " Updating various lists"
            << endl;
    }

    {
        const labelList& reverseCellMap = map.reverseCellMap();

        if (debug)
        {
            Pout<< "hexRef4::updateMesh :"
                << " reverseCellMap:" << map.reverseCellMap().size()
                << " cellMap:" << map.cellMap().size()
                << " nCells:" << mesh_.nCells()
                << " nOldCells:" << map.nOldCells()
                << " cellLevel_:" << cellLevel_.size()
                << " reversePointMap:" << map.reversePointMap().size()
                << " pointMap:" << map.pointMap().size()
                << " nPoints:" << mesh_.nPoints()
                << " nOldPoints:" << map.nOldPoints()
                << " pointLevel_:" << pointLevel_.size()
                << endl;
        }

        if (reverseCellMap.size() == cellLevel_.size())
        {
            // Assume it is after hexRef4 that this routine is called.
            // Just account for reordering. We cannot use cellMap since
            // then cells created from cells would get cellLevel_ of
            // cell they were created from.
            reorder(reverseCellMap, mesh_.nCells(), -1, cellLevel_);
        }
        else
        {
            // Map data
            const labelList& cellMap = map.cellMap();

            labelList newCellLevel(cellMap.size());
            forAll(cellMap, newCelli)
            {
                label oldCelli = cellMap[newCelli];

                if (oldCelli == -1)
                {
                    newCellLevel[newCelli] = -1;
                }
                else
                {
                    newCellLevel[newCelli] = cellLevel_[oldCelli];
                }
            }
            cellLevel_.transfer(newCellLevel);
        }

        // See if any cells to restore. This will be for some new cells
        // the corresponding old cell.
        forAllConstIters(cellsToRestore, iter)
        {
            const label newCelli = iter.key();
            const label storedCelli = iter.val();

            const auto fnd = savedCellLevel_.cfind(storedCelli);

            if (!fnd.good())
            {
                FatalErrorInFunction
                    << "Problem : trying to restore old value for new cell "
                    << newCelli << " but cannot find old cell " << storedCelli
                    << " in map of stored values " << savedCellLevel_
                    << abort(FatalError);
            }
            cellLevel_[newCelli] = fnd.val();
        }

        //if (cellLevel_.found(-1))
        //{
        //    WarningInFunction
        //        << "Problem : "
        //        << "cellLevel_ contains illegal value -1 after mapping
        //        << " at cell " << cellLevel_.find(-1) << endl
        //        << "This means that another program has inflated cells"
        //        << " (created cells out-of-nothing) and hence we don't know"
        //        << " their cell level. Continuing with illegal value."
        //        << abort(FatalError);
        //}
    }


    // Update pointlevel
    {
        const labelList& reversePointMap = map.reversePointMap();

        if (reversePointMap.size() == pointLevel_.size())
        {
            // Assume it is after hexRef4 that this routine is called.
            reorder(reversePointMap, mesh_.nPoints(), -1,  pointLevel_);
        }
        else
        {
            // Map data
            const labelList& pointMap = map.pointMap();

            labelList newPointLevel(pointMap.size());

            forAll(pointMap, newPointi)
            {
                const label oldPointi = pointMap[newPointi];

                if (oldPointi == -1)
                {
                    //FatalErrorInFunction
                    //    << "Problem : point " << newPointi
                    //    << " at " << mesh_.points()[newPointi]
                    //    << " does not originate from another point"
                    //    << " (i.e. is inflated)." << nl
                    //    << "Hence we cannot determine the new pointLevel"
                    //    << " for it." << abort(FatalError);
                    newPointLevel[newPointi] = -1;
                }
                else
                {
                    newPointLevel[newPointi] = pointLevel_[oldPointi];
                }
            }
            pointLevel_.transfer(newPointLevel);
        }

        // See if any points to restore. This will be for some new points
        // the corresponding old point (the one from the call to storeData)
        forAllConstIters(pointsToRestore, iter)
        {
            const label newPointi = iter.key();
            const label storedPointi = iter.val();

            auto fnd = savedPointLevel_.find(storedPointi);

            if (!fnd.good())
            {
                FatalErrorInFunction
                    << "Problem : trying to restore old value for new point "
                    << newPointi << " but cannot find old point "
                    << storedPointi
                    << " in map of stored values " << savedPointLevel_
                    << abort(FatalError);
            }
            pointLevel_[newPointi] = fnd.val();
        }

        //if (pointLevel_.found(-1))
        //{
        //    WarningInFunction
        //        << "Problem : "
        //        << "pointLevel_ contains illegal value -1 after mapping"
        //        << " at point" << pointLevel_.find(-1) << endl
        //        << "This means that another program has inflated points"
        //        << " (created points out-of-nothing) and hence we don't know"
        //        << " their point level. Continuing with illegal value."
        //        //<< abort(FatalError);
        //}
    }

    // Update refinement tree
    if (history_.active())
    {
        history_.updateMesh(map);
    }

    // Mark files as changed
    setInstance(mesh_.facesInstance());

    // Update face removal engine
    faceRemover_.updateMesh(map);

    // Clear cell shapes
    cellShapesPtr_.clear();
}


// Gets called after mesh subsetting. Maps are from new back to old.
void Foam::hexRef4::subset
(
    const labelList& pointMap,
    const labelList& faceMap,
    const labelList& cellMap
)
{
    // Update celllevel
    if (debug)
    {
        Pout<< "hexRef4::subset :"
            << " Updating various lists"
            << endl;
    }

    if (history_.active())
    {
        WarningInFunction
            << "Subsetting will not work in combination with unrefinement."
            << nl
            << "Proceed at your own risk." << endl;
    }


    // Update celllevel
    {
        labelList newCellLevel(cellMap.size());

        forAll(cellMap, newCelli)
        {
            newCellLevel[newCelli] = cellLevel_[cellMap[newCelli]];
        }

        cellLevel_.transfer(newCellLevel);

        if (cellLevel_.found(-1))
        {
            FatalErrorInFunction
                << "Problem : "
                << "cellLevel_ contains illegal value -1 after mapping:"
                << cellLevel_
                << abort(FatalError);
        }
    }

    // Update pointlevel
    {
        labelList newPointLevel(pointMap.size());

        forAll(pointMap, newPointi)
        {
            newPointLevel[newPointi] = pointLevel_[pointMap[newPointi]];
        }

        pointLevel_.transfer(newPointLevel);

        if (pointLevel_.found(-1))
        {
            FatalErrorInFunction
                << "Problem : "
                << "pointLevel_ contains illegal value -1 after mapping:"
                << pointLevel_
                << abort(FatalError);
        }
    }

    // Update refinement tree
    if (history_.active())
    {
        history_.subset(pointMap, faceMap, cellMap);
    }

    // Mark files as changed
    setInstance(mesh_.facesInstance());

    // Nothing needs doing to faceRemover.
    //faceRemover_.subset(pointMap, faceMap, cellMap);

    // Clear cell shapes
    cellShapesPtr_.clear();
}


// Gets called after the mesh distribution
void Foam::hexRef4::distribute(const mapDistributePolyMesh& map)
{
    if (debug)
    {
        Pout<< "hexRef4::distribute :"
            << " Updating various lists"
            << endl;
    }

    // Update celllevel
    map.distributeCellData(cellLevel_);
    // Update pointlevel
    map.distributePointData(pointLevel_);

    // Update refinement tree
    if (history_.active())
    {
        history_.distribute(map);
    }

    // Update face removal engine
    faceRemover_.distribute(map);

    // Clear cell shapes
    cellShapesPtr_.clear();
}


void Foam::hexRef4::checkMesh() const
{
    const scalar smallDim = 1e-6 * mesh_.bounds().mag();

    if (debug)
    {
        Pout<< "hexRef4::checkMesh : Using matching tolerance smallDim:"
            << smallDim << endl;
    }

    // Check owner on coupled faces.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    // There should be only one coupled face between two cells. Why? Since
    // otherwise mesh redistribution might cause multiple faces between two
    // cells
    {
        labelList nei(mesh_.nBoundaryFaces());
        forAll(nei, i)
        {
            nei[i] = mesh_.faceOwner()[i+mesh_.nInternalFaces()];
        }

        // Replace data on coupled patches with their neighbour ones.
        syncTools::swapBoundaryFaceList(mesh_, nei);

        const polyBoundaryMesh& patches = mesh_.boundaryMesh();

        forAll(patches, patchi)
        {
            const polyPatch& pp = patches[patchi];

            if (pp.coupled())
            {
                // Check how many faces between owner and neighbour.
                // Should be only one.
                labelPairLookup cellToFace(2*pp.size());

                label facei = pp.start();

                forAll(pp, i)
                {
                    label own = mesh_.faceOwner()[facei];
                    label bFacei = facei-mesh_.nInternalFaces();

                    if (!cellToFace.insert(labelPair(own, nei[bFacei]), facei))
                    {
                        dumpCell(own);
                        FatalErrorInFunction
                            << "Faces do not seem to be correct across coupled"
                            << " boundaries" << endl
                            << "Coupled face " << facei
                            << " between owner " << own
                            << " on patch " << pp.name()
                            << " and coupled neighbour " << nei[bFacei]
                            << " has two faces connected to it:"
                            << facei << " and "
                            << cellToFace[labelPair(own, nei[bFacei])]
                            << abort(FatalError);
                    }

                    facei++;
                }
            }
        }
    }

    // Check face areas.
    // ~~~~~~~~~~~~~~~~~

    {
        scalarField neiFaceAreas(mesh_.nBoundaryFaces());
        forAll(neiFaceAreas, i)
        {
            neiFaceAreas[i] = mag(mesh_.faceAreas()[i+mesh_.nInternalFaces()]);
        }

        // Replace data on coupled patches with their neighbour ones.
        syncTools::swapBoundaryFaceList(mesh_, neiFaceAreas);

        forAll(neiFaceAreas, i)
        {
            label facei = i+mesh_.nInternalFaces();

            const scalar magArea = mag(mesh_.faceAreas()[facei]);

            if (mag(magArea - neiFaceAreas[i]) > smallDim)
            {
                const face& f = mesh_.faces()[facei];
                label patchi = mesh_.boundaryMesh().whichPatch(facei);

                dumpCell(mesh_.faceOwner()[facei]);

                FatalErrorInFunction
                    << "Faces do not seem to be correct across coupled"
                    << " boundaries" << endl
                    << "Coupled face " << facei
                    << " on patch " << patchi
                    << " " << mesh_.boundaryMesh()[patchi].name()
                    << " coords:" << UIndirectList<point>(mesh_.points(), f)
                    << " has face area:" << magArea
                    << " (coupled) neighbour face area differs:"
                    << neiFaceAreas[i]
                    << " to within tolerance " << smallDim
                    << abort(FatalError);
            }
        }
    }


    // Check number of points on faces.
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
        labelList nVerts(mesh_.nBoundaryFaces());

        forAll(nVerts, i)
        {
            nVerts[i] = mesh_.faces()[i+mesh_.nInternalFaces()].size();
        }

        // Replace data on coupled patches with their neighbour ones.
        syncTools::swapBoundaryFaceList(mesh_, nVerts);

        forAll(nVerts, i)
        {
            label facei = i+mesh_.nInternalFaces();

            const face& f = mesh_.faces()[facei];

            if (f.size() != nVerts[i])
            {
                dumpCell(mesh_.faceOwner()[facei]);

                label patchi = mesh_.boundaryMesh().whichPatch(facei);

                FatalErrorInFunction
                    << "Faces do not seem to be correct across coupled"
                    << " boundaries" << endl
                    << "Coupled face " << facei
                    << " on patch " << patchi
                    << " " << mesh_.boundaryMesh()[patchi].name()
                    << " coords:" << UIndirectList<point>(mesh_.points(), f)
                    << " has size:" << f.size()
                    << " (coupled) neighbour face has size:"
                    << nVerts[i]
                    << abort(FatalError);
            }
        }
    }


    // Check points of face
    // ~~~~~~~~~~~~~~~~~~~~
    {
        // Anchor points.
        pointField anchorPoints(mesh_.nBoundaryFaces());

        forAll(anchorPoints, i)
        {
            label facei = i+mesh_.nInternalFaces();
            const point& fc = mesh_.faceCentres()[facei];
            const face& f = mesh_.faces()[facei];
            const vector anchorVec(mesh_.points()[f[0]] - fc);

            anchorPoints[i] = anchorVec;
        }

        // Replace data on coupled patches with their neighbour ones. Apply
        // rotation transformation (but not separation since is relative vector
        // to point on same face.
        syncTools::swapBoundaryFaceList(mesh_, anchorPoints);

        forAll(anchorPoints, i)
        {
            label facei = i+mesh_.nInternalFaces();
            const point& fc = mesh_.faceCentres()[facei];
            const face& f = mesh_.faces()[facei];
            const vector anchorVec(mesh_.points()[f[0]] - fc);

            if (mag(anchorVec - anchorPoints[i]) > smallDim)
            {
                dumpCell(mesh_.faceOwner()[facei]);

                label patchi = mesh_.boundaryMesh().whichPatch(facei);

                FatalErrorInFunction
                    << "Faces do not seem to be correct across coupled"
                    << " boundaries" << endl
                    << "Coupled face " << facei
                    << " on patch " << patchi
                    << " " << mesh_.boundaryMesh()[patchi].name()
                    << " coords:" << UIndirectList<point>(mesh_.points(), f)
                    << " has anchor vector:" << anchorVec
                    << " (coupled) neighbour face anchor vector differs:"
                    << anchorPoints[i]
                    << " to within tolerance " << smallDim
                    << abort(FatalError);
            }
        }
    }

    if (debug)
    {
        Pout<< "hexRef4::checkMesh : Returning" << endl;
    }
}


void Foam::hexRef4::checkRefinementLevels
(
    const label maxPointDiff,
    const labelList& pointsToCheck
) const
{
    if (debug)
    {
        Pout<< "hexRef4::checkRefinementLevels :"
            << " Checking 2:1 refinement level" << endl;
    }

    if
    (
        cellLevel_.size() != mesh_.nCells()
     || pointLevel_.size() != mesh_.nPoints()
    )
    {
        FatalErrorInFunction
            << "cellLevel size should be number of cells"
            << " and pointLevel size should be number of points."<< nl
            << "cellLevel:" << cellLevel_.size()
            << " mesh.nCells():" << mesh_.nCells() << nl
            << "pointLevel:" << pointLevel_.size()
            << " mesh.nPoints():" << mesh_.nPoints()
            << abort(FatalError);
    }


    // Check 2:1 consistency.
    // ~~~~~~~~~~~~~~~~~~~~~~

    {
        // Internal faces.
        for (label facei = 0; facei < mesh_.nInternalFaces(); facei++)
        {
            label own = mesh_.faceOwner()[facei];
            label nei = mesh_.faceNeighbour()[facei];

            if (mag(cellLevel_[own] - cellLevel_[nei]) > 1)
            {
                dumpCell(own);
                dumpCell(nei);

                FatalErrorInFunction
                    << "Celllevel does not satisfy 2:1 constraint." << nl
                    << "On face " << facei << " owner cell " << own
                    << " has refinement " << cellLevel_[own]
                    << " neighbour cell " << nei << " has refinement "
                    << cellLevel_[nei]
                    << abort(FatalError);
            }
        }

        // Coupled faces. Get neighbouring value
        labelList neiLevel(mesh_.nBoundaryFaces());

        forAll(neiLevel, i)
        {
            label own = mesh_.faceOwner()[i+mesh_.nInternalFaces()];

            neiLevel[i] = cellLevel_[own];
        }

        // No separation
        syncTools::swapBoundaryFaceList(mesh_, neiLevel);

        forAll(neiLevel, i)
        {
            label facei = i+mesh_.nInternalFaces();

            label own = mesh_.faceOwner()[facei];

            if (mag(cellLevel_[own] - neiLevel[i]) > 1)
            {
                dumpCell(own);

                label patchi = mesh_.boundaryMesh().whichPatch(facei);

                FatalErrorInFunction
                    << "Celllevel does not satisfy 2:1 constraint."
                    << " On coupled face " << facei
                    << " on patch " << patchi << " "
                    << mesh_.boundaryMesh()[patchi].name()
                    << " owner cell " << own << " has refinement "
                    << cellLevel_[own]
                    << " (coupled) neighbour cell has refinement "
                    << neiLevel[i]
                    << abort(FatalError);
            }
        }
    }


    // Check pointLevel is synchronized
    // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    {
        labelList syncPointLevel(pointLevel_);

        // Get min level
        syncTools::syncPointList
        (
            mesh_,
            syncPointLevel,
            minEqOp<label>(),
            labelMax
        );


        forAll(syncPointLevel, pointi)
        {
            if (pointLevel_[pointi] != syncPointLevel[pointi])
            {
                FatalErrorInFunction
                    << "PointLevel is not consistent across coupled patches."
                    << endl
                    << "point:" << pointi << " coord:" << mesh_.points()[pointi]
                    << " has level " << pointLevel_[pointi]
                    << " whereas the coupled point has level "
                    << syncPointLevel[pointi]
                    << abort(FatalError);
            }
        }
    }


    // Check 2:1 across points (instead of faces)
    if (maxPointDiff != -1)
    {
        // Determine per point the max cell level.
        labelList maxPointLevel(mesh_.nPoints(), Zero);

        forAll(maxPointLevel, pointi)
        {
            const labelList& pCells = mesh_.pointCells(pointi);

            label& pLevel = maxPointLevel[pointi];

            forAll(pCells, i)
            {
                pLevel = max(pLevel, cellLevel_[pCells[i]]);
            }
        }

        // Sync maxPointLevel to neighbour
        syncTools::syncPointList
        (
            mesh_,
            maxPointLevel,
            maxEqOp<label>(),
            labelMin            // null value
        );

        // Check 2:1 across boundary points
        forAll(pointsToCheck, i)
        {
            label pointi = pointsToCheck[i];

            const labelList& pCells = mesh_.pointCells(pointi);

            forAll(pCells, i)
            {
                label celli = pCells[i];

                if
                (
                    mag(cellLevel_[celli]-maxPointLevel[pointi])
                  > maxPointDiff
                )
                {
                    dumpCell(celli);

                    FatalErrorInFunction
                        << "Too big a difference between"
                        << " point-connected cells." << nl
                        << "cell:" << celli
                        << " cellLevel:" << cellLevel_[celli]
                        << " uses point:" << pointi
                        << " coord:" << mesh_.points()[pointi]
                        << " which is also used by a cell with level:"
                        << maxPointLevel[pointi]
                        << abort(FatalError);
                }
            }
        }
    }


    //- Gives problems after first splitting off inside mesher.
    //// Hanging points
    //{
    //    // Any patches with points having only two edges.
    //
    //    boolList isHangingPoint(mesh_.nPoints(), false);
    //
    //    const polyBoundaryMesh& patches = mesh_.boundaryMesh();
    //
    //    forAll(patches, patchi)
    //    {
    //        const polyPatch& pp = patches[patchi];
    //
    //        const labelList& meshPoints = pp.meshPoints();
    //
    //        forAll(meshPoints, i)
    //        {
    //            label pointi = meshPoints[i];
    //
    //            const labelList& pEdges = mesh_.pointEdges()[pointi];
    //
    //            if (pEdges.size() == 2)
    //            {
    //                isHangingPoint[pointi] = true;
    //            }
    //        }
    //    }
    //
    //    syncTools::syncPointList
    //    (
    //        mesh_,
    //        isHangingPoint,
    //        andEqOp<bool>(),        // only if all decide it is hanging point
    //        true,                   // null
    //        false                   // no separation
    //    );
    //
    //    //OFstream str(mesh_.time().path()/"hangingPoints.obj");
    //
    //    label nHanging = 0;
    //
    //    forAll(isHangingPoint, pointi)
    //    {
    //        if (isHangingPoint[pointi])
    //        {
    //            nHanging++;
    //
    //            Pout<< "Hanging boundary point " << pointi
    //                << " at " << mesh_.points()[pointi]
    //                << endl;
    //            //meshTools::writeOBJ(str, mesh_.points()[pointi]);
    //        }
    //    }
    //
    //    if (returnReduceOr(nHanging))
    //    {
    //        FatalErrorInFunction
    //            << "Detected a point used by two edges only (hanging point)"
    //            << nl << "This is not allowed"
    //            << abort(FatalError);
    //    }
    //}
}


const Foam::cellShapeList& Foam::hexRef4::cellShapes() const
{
    if (!cellShapesPtr_)
    {
        if (debug)
        {
            Pout<< "hexRef4::cellShapes() : calculating splitHex cellShapes."
                << " cellLevel:" << cellLevel_.size()
                << " pointLevel:" << pointLevel_.size()
                << endl;
        }

        const cellShapeList& meshShapes = mesh_.cellShapes();
        cellShapesPtr_.reset(new cellShapeList(meshShapes));

        label nSplitHex = 0;
        label nUnrecognised = 0;

        forAll(cellLevel_, celli)
        {
            if (meshShapes[celli].model().index() == 0)
            {
                label level = cellLevel_[celli];

                // Return true if we've found 6 quads
                DynamicList<face> quads;
                bool haveQuads = matchHexShape
                (
                    celli,
                    level,
                    quads
                );

                if (haveQuads)
                {
                    faceList faces(std::move(quads));
                    cellShapesPtr_()[celli] = degenerateMatcher::match(faces);
                    nSplitHex++;
                }
                else
                {
                    nUnrecognised++;
                }
            }
        }
        if (debug)
        {
            Pout<< "hexRef4::cellShapes() :"
                << " nCells:" << mesh_.nCells() << " of which" << nl
                << "    primitive:" << (mesh_.nCells()-nSplitHex-nUnrecognised)
                << nl
                << "    split-hex:" << nSplitHex << nl
                << "    poly     :" << nUnrecognised << nl
                << endl;
        }
    }
    return *cellShapesPtr_;
}


Foam::labelList Foam::hexRef4::getSplitPoints() const
{
    if (debug)
    {
        checkRefinementLevels(-1, labelList(0));
    }

    if (debug)
    {
        Pout<< "hexRef4::getSplitPoints :"
            << " Calculating unrefineable points" << endl;
    }


    if (!history_.active())
    {
        FatalErrorInFunction
            << "Only call if constructed with history capability"
            << abort(FatalError);
    }

    // Master cell
    // -1 undetermined
    // -2 certainly not split point
    // >= label of master cell
    labelList splitMaster(mesh_.nPoints(), -1);
    labelList splitMasterLevel(mesh_.nPoints(), Zero);

    // Unmark all with not 4 cells. In 2-D the point at the centre of a
    // refinement group is the mid point of a span face, shared by exactly the
    // 4 sub-cells.
    for (label pointi = 0; pointi < mesh_.nPoints(); pointi++)
    {
        const labelList& pCells = mesh_.pointCells(pointi);

        if (pCells.size() != 4)
        {
            splitMaster[pointi] = -2;
        }
    }

    // Unmark all with different master cells
    const labelList& visibleCells = history_.visibleCells();

    forAll(visibleCells, celli)
    {
        const labelList& cPoints = mesh_.cellPoints(celli);

        if (visibleCells[celli] != -1 && history_.parentIndex(celli) >= 0)
        {
            label parentIndex = history_.parentIndex(celli);

            // Check same master.
            forAll(cPoints, i)
            {
                label pointi = cPoints[i];

                label masterCelli = splitMaster[pointi];

                if (masterCelli == -1)
                {
                    // First time visit of point. Store parent cell and
                    // level of the parent cell (with respect to celli). This
                    // is additional guarantee that we're referring to the
                    // same master at the same refinement level.

                    splitMaster[pointi] = parentIndex;
                    splitMasterLevel[pointi] = cellLevel_[celli] - 1;
                }
                else if (masterCelli == -2)
                {
                    // Already decided that point is not splitPoint
                }
                else if
                (
                    (masterCelli != parentIndex)
                 || (splitMasterLevel[pointi] != cellLevel_[celli] - 1)
                )
                {
                    // Different masters so point is on two refinement
                    // patterns
                    splitMaster[pointi] = -2;
                }
            }
        }
        else
        {
            // Either not visible or is unrefined cell
            forAll(cPoints, i)
            {
                label pointi = cPoints[i];

                splitMaster[pointi] = -2;
            }
        }
    }

    // Unmark boundary faces.
    // Unlike hexRef8 the split point of a 2-D refinement group sits *on* the
    // boundary - it is the mid point of the span (empty patch) face - so only
    // the faces that are not span faces disqualify a point.
    for
    (
        label facei = mesh_.nInternalFaces();
        facei < mesh_.nFaces();
        facei++
    )
    {
        if (isSpanFace(facei))
        {
            continue;
        }

        const face& f = mesh_.faces()[facei];

        forAll(f, fp)
        {
            splitMaster[f[fp]] = -2;
        }
    }

    // Both span face mid points (front and back) of a group pass the tests
    // above. Keep only one of the two, otherwise the same set of internal
    // faces would be removed twice.
    for (label pointi = 0; pointi < mesh_.nPoints(); pointi++)
    {
        if (splitMaster[pointi] < 0)
        {
            continue;
        }

        const labelList& pEdges = mesh_.pointEdges()[pointi];

        forAll(pEdges, i)
        {
            const label edgeI = pEdges[i];

            if (!isSpanEdge(edgeI))
            {
                continue;
            }

            const label otherPointi = mesh_.edges()[edgeI].otherVertex(pointi);

            if (splitMaster[otherPointi] >= 0 && otherPointi < pointi)
            {
                // Partner across the span is also a candidate and has the
                // lower label: let that one represent the group.
                splitMaster[pointi] = -2;
            }
        }
    }


    // Collect into labelList

    label nSplitPoints = 0;

    forAll(splitMaster, pointi)
    {
        if (splitMaster[pointi] >= 0)
        {
            nSplitPoints++;
        }
    }

    labelList splitPoints(nSplitPoints);
    nSplitPoints = 0;

    forAll(splitMaster, pointi)
    {
        if (splitMaster[pointi] >= 0)
        {
            splitPoints[nSplitPoints++] = pointi;
        }
    }

    return splitPoints;
}


//void Foam::hexRef4::markIndex
//(
//    const label maxLevel,
//    const label level,
//    const label index,
//    const label markValue,
//    labelList& indexValues
//) const
//{
//    if (level < maxLevel && indexValues[index] == -1)
//    {
//        // Mark
//        indexValues[index] = markValue;
//
//        // Mark parent
//        const splitCell8& split = history_.splitCells()[index];
//
//        if (split.parent_ >= 0)
//        {
//            markIndex
//            (
//              maxLevel, level+1, split.parent_, markValue, indexValues);
//            )
//        }
//    }
//}
//
//
//// Get all cells which (down to level) originate from the same cell.
//// level=0 returns cell only, level=1 returns the 8 cells this cell
//// originates from, level=2 returns 64 cells etc.
//// If the cell does not originate from refinement returns just itself.
//void Foam::hexRef4::markCellClusters
//(
//    const label maxLevel,
//    labelList& cluster
//) const
//{
//    cluster.setSize(mesh_.nCells());
//    cluster = -1;
//
//    const DynamicList<splitCell8>& splitCells = history_.splitCells();
//
//    // Mark all splitCells down to level maxLevel with a cell originating from
//    // it.
//
//    labelList indexLevel(splitCells.size(), -1);
//
//    forAll(visibleCells, celli)
//    {
//        label index = visibleCells[celli];
//
//        if (index >= 0)
//        {
//            markIndex(maxLevel, 0, index, celli, indexLevel);
//        }
//    }
//
//    // Mark cells with splitCell
//}


Foam::labelList Foam::hexRef4::consistentUnrefinement
(
    const labelList& pointsToUnrefine,
    const bool maxSet
) const
{
    if (debug)
    {
        Pout<< "hexRef4::consistentUnrefinement :"
            << " Determining 2:1 consistent unrefinement" << endl;
    }

    if (maxSet)
    {
        FatalErrorInFunction
            << "maxSet not implemented yet."
            << abort(FatalError);
    }

    // Loop, modifying pointsToUnrefine, until no more changes to due to 2:1
    // conflicts.
    // maxSet = false : unselect points to refine
    // maxSet = true: select points to refine

    // Maintain bitset for pointsToUnrefine and cellsToUnrefine
    bitSet unrefinePoint(mesh_.nPoints(), pointsToUnrefine);

    while (true)
    {
        // Construct cells to unrefine
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~

        bitSet unrefineCell(mesh_.nCells());

        forAll(unrefinePoint, pointi)
        {
            if (unrefinePoint.test(pointi))
            {
                const labelList& pCells = mesh_.pointCells(pointi);

                unrefineCell.set(pCells);
            }
        }


        label nChanged = 0;


        // Check 2:1 consistency taking refinement into account
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        // Internal faces.
        for (label facei = 0; facei < mesh_.nInternalFaces(); facei++)
        {
            label own = mesh_.faceOwner()[facei];
            label nei = mesh_.faceNeighbour()[facei];

            label ownLevel = cellLevel_[own] - unrefineCell.get(own);
            label neiLevel = cellLevel_[nei] - unrefineCell.get(nei);

            if (ownLevel < (neiLevel-1))
            {
                // Since was 2:1 this can only occur if own is marked for
                // unrefinement.

                if (maxSet)
                {
                    unrefineCell.set(nei);
                }
                else
                {
                    // could also combine with unset:
                    // if (!unrefineCell.unset(own))
                    // {
                    //     FatalErrorInFunction
                    //         << "problem cell already unset"
                    //         << abort(FatalError);
                    // }
                    if (!unrefineCell.test(own))
                    {
                        FatalErrorInFunction
                            << "problem" << abort(FatalError);
                    }

                    unrefineCell.unset(own);
                }
                nChanged++;
            }
            else if (neiLevel < (ownLevel-1))
            {
                if (maxSet)
                {
                    unrefineCell.set(own);
                }
                else
                {
                    if (!unrefineCell.test(nei))
                    {
                        FatalErrorInFunction
                            << "problem" << abort(FatalError);
                    }

                    unrefineCell.unset(nei);
                }
                nChanged++;
            }
        }


        // Coupled faces. Swap owner level to get neighbouring cell level.
        labelList neiLevel(mesh_.nBoundaryFaces());

        forAll(neiLevel, i)
        {
            label own = mesh_.faceOwner()[i+mesh_.nInternalFaces()];

            neiLevel[i] = cellLevel_[own] - unrefineCell.get(own);
        }

        // Swap to neighbour
        syncTools::swapBoundaryFaceList(mesh_, neiLevel);

        forAll(neiLevel, i)
        {
            label facei = i+mesh_.nInternalFaces();
            label own = mesh_.faceOwner()[facei];
            label ownLevel = cellLevel_[own] - unrefineCell.get(own);

            if (ownLevel < (neiLevel[i]-1))
            {
                if (!maxSet)
                {
                    if (!unrefineCell.test(own))
                    {
                        FatalErrorInFunction
                            << "problem" << abort(FatalError);
                    }

                    unrefineCell.unset(own);
                    nChanged++;
                }
            }
            else if (neiLevel[i] < (ownLevel-1))
            {
                if (maxSet)
                {
                    if (unrefineCell.test(own))
                    {
                        FatalErrorInFunction
                            << "problem" << abort(FatalError);
                    }

                    unrefineCell.set(own);
                    nChanged++;
                }
            }
        }

        reduce(nChanged, sumOp<label>());

        if (debug)
        {
            Pout<< "hexRef4::consistentUnrefinement :"
                << " Changed " << nChanged
                << " refinement levels due to 2:1 conflicts."
                << endl;
        }

        if (nChanged == 0)
        {
            break;
        }


        // Convert cellsToUnrefine back into points to unrefine
        // ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        // Knock out any point whose cell neighbour cannot be unrefined.
        forAll(unrefinePoint, pointi)
        {
            if (unrefinePoint.test(pointi))
            {
                const labelList& pCells = mesh_.pointCells(pointi);

                forAll(pCells, j)
                {
                    if (!unrefineCell.test(pCells[j]))
                    {
                        unrefinePoint.unset(pointi);
                        break;
                    }
                }
            }
        }
    }


    // Convert back to labelList.
    label nSet = 0;

    forAll(unrefinePoint, pointi)
    {
        if (unrefinePoint.test(pointi))
        {
            nSet++;
        }
    }

    labelList newPointsToUnrefine(nSet);
    nSet = 0;

    forAll(unrefinePoint, pointi)
    {
        if (unrefinePoint.test(pointi))
        {
            newPointsToUnrefine[nSet++] = pointi;
        }
    }

    return newPointsToUnrefine;
}


void Foam::hexRef4::setUnrefinement
(
    const labelList& splitPointLabels,
    polyTopoChange& meshMod
)
{
    if (!history_.active())
    {
        FatalErrorInFunction
            << "Only call if constructed with history capability"
            << abort(FatalError);
    }

    if (debug)
    {
        Pout<< "hexRef4::setUnrefinement :"
            << " Checking initial mesh just to make sure" << endl;

        checkMesh();

        forAll(cellLevel_, celli)
        {
            if (cellLevel_[celli] < 0)
            {
                FatalErrorInFunction
                    << "Illegal cell level " << cellLevel_[celli]
                    << " for cell " << celli
                    << abort(FatalError);
            }
        }


        // Write to sets.
        pointSet pSet(mesh_, "splitPoints", splitPointLabels);
        pSet.write();

        cellSet cSet(mesh_, "splitPointCells", splitPointLabels.size());

        forAll(splitPointLabels, i)
        {
            const labelList& pCells = mesh_.pointCells(splitPointLabels[i]);

            cSet.insert(pCells);
        }
        cSet.write();

        Pout<< "hexRef4::setRefinement : Dumping " << pSet.size()
            << " points and "
            << cSet.size() << " cells for unrefinement to" << nl
            << "    pointSet " << pSet.objectPath() << nl
            << "    cellSet " << cSet.objectPath()
            << endl;
    }


    labelList cellRegion;
    labelList cellRegionMaster;
    labelList facesToRemove;

    {
        labelHashSet splitFaces(8*splitPointLabels.size());

        forAll(splitPointLabels, i)
        {
            const labelList& pFaces = mesh_.pointFaces()[splitPointLabels[i]];

            // Only the 4 internal faces of the group are to be removed. The
            // span face quads that also share the split point are boundary
            // faces on the empty patch; removeFaces merges those back into
            // one itself once the cells have been combined.
            forAll(pFaces, j)
            {
                if (mesh_.isInternalFace(pFaces[j]))
                {
                    splitFaces.insert(pFaces[j]);
                }
            }
        }

        // Check with faceRemover what faces will get removed. Note that this
        // can be more (but never less) than splitFaces provided.
        faceRemover_.compatibleRemoves
        (
            splitFaces.toc(),   // pierced faces
            cellRegion,         // per cell -1 or region it is merged into
            cellRegionMaster,   // per region the master cell
            facesToRemove       // new faces to be removed.
        );

        if (facesToRemove.size() != splitFaces.size())
        {
            // Dump current mesh
            {
                const_cast<polyMesh&>(mesh_).setInstance
                (
                    mesh_.time().timeName()
                );
                mesh_.write();
                pointSet pSet(mesh_, "splitPoints", splitPointLabels);
                pSet.write();
                faceSet fSet(mesh_, "splitFaces", splitFaces);
                fSet.write();
                faceSet removeSet(mesh_, "facesToRemove", facesToRemove);
                removeSet.write();
            }

            FatalErrorInFunction
                << "Ininitial set of split points to unrefine does not"
                << " seem to be consistent or not mid points of refined cells"
                << " splitPoints:" << splitPointLabels.size()
                << " splitFaces:" << splitFaces.size()
                << " facesToRemove:" << facesToRemove.size()
                << abort(FatalError);
        }
    }

    // Redo the region master so it is consistent with our master.
    // This will guarantee that the new cell (for which faceRemover uses
    // the region master) is already compatible with our refinement structure.

    forAll(splitPointLabels, i)
    {
        label pointi = splitPointLabels[i];

        // Get original cell label

        const labelList& pCells = mesh_.pointCells(pointi);

        // Check
        if (pCells.size() != 4)
        {
            FatalErrorInFunction
                << "splitPoint " << pointi
                << " should have 4 cells using it. It has " << pCells
                << abort(FatalError);
        }


        // Check that the lowest numbered pCells is the master of the region
        // (should be guaranteed by directRemoveFaces)
        //if (debug)
        {
            label masterCelli = min(pCells);

            forAll(pCells, j)
            {
                label celli = pCells[j];

                label region = cellRegion[celli];

                if (region == -1)
                {
                    FatalErrorInFunction
                        << "Ininitial set of split points to unrefine does not"
                        << " seem to be consistent or not mid points"
                        << " of refined cells" << nl
                        << "cell:" << celli << " on splitPoint " << pointi
                        << " has no region to be merged into"
                        << abort(FatalError);
                }

                if (masterCelli != cellRegionMaster[region])
                {
                    FatalErrorInFunction
                        << "cell:" << celli << " on splitPoint:" << pointi
                        << " in region " << region
                        << " has master:" << cellRegionMaster[region]
                        << " which is not the lowest numbered cell"
                        << " among the pointCells:" << pCells
                        << abort(FatalError);
                }
            }
        }
    }

    // Insert all commands to combine cells. Never fails so don't have to
    // test for success.
    faceRemover_.setRefinement
    (
        facesToRemove,
        cellRegion,
        cellRegionMaster,
        meshMod
    );

    // Remove the 4 cells that originated from merging around the split point
    // and adapt cell levels (not that pointLevels stay the same since points
    // either get removed or stay at the same position.
    forAll(splitPointLabels, i)
    {
        label pointi = splitPointLabels[i];

        const labelList& pCells = mesh_.pointCells(pointi);

        label masterCelli = min(pCells);

        forAll(pCells, j)
        {
            cellLevel_[pCells[j]]--;
        }

        history_.combineCells(masterCelli, pCells);
    }

    // Mark files as changed
    setInstance(mesh_.facesInstance());

    // history_.updateMesh will take care of truncating.
}


// Write refinement to polyMesh directory.
bool Foam::hexRef4::write(const bool writeOnProc) const
{
    bool writeOk =
        cellLevel_.write(writeOnProc)
     && pointLevel_.write(writeOnProc)
     && level0Edge_.write(writeOnProc);

    if (history_.active())
    {
        writeOk = writeOk && history_.write(writeOnProc);
    }
    else
    {
        refinementHistory::removeFiles(mesh_);
    }

    return writeOk;
}


void Foam::hexRef4::removeFiles(const polyMesh& mesh)
{
    IOobject io
    (
        "dummy",
        mesh.facesInstance(),
        polyMesh::meshSubDir,
        mesh
    );
    fileName setsDir(io.path());

    if (topoSet::debug) DebugVar(setsDir);

    if (exists(setsDir/"cellLevel"))
    {
        rm(setsDir/"cellLevel");
    }
    if (exists(setsDir/"pointLevel"))
    {
        rm(setsDir/"pointLevel");
    }
    if (exists(setsDir/"level0Edge"))
    {
        rm(setsDir/"level0Edge");
    }
    refinementHistory::removeFiles(mesh);
}


// ************************************************************************* //
