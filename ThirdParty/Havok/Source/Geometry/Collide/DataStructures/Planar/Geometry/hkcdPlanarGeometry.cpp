/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Geometry/Collide/hkcdCollide.h>
#include <Geometry/Collide/DataStructures/Planar/Geometry/hkcdPlanarGeometry.h>
#include <Geometry/Collide/DataStructures/Planar/Utils/hkcdPlanarGeometryWeldUtil.h>
#include <Geometry/Collide/DataStructures/Planar/Predicates/hkcdPlanarGeometryPredicates.h>
#include <Geometry/Collide/DataStructures/Planar/ConvexCellsTree/hkcdConvexCellsTree3D.h>
#include <Geometry/Collide/DataStructures/IntAabb/hkcdIntAabb.h>

#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/GeometryUtilities/Mesh/Utils/FindUniquePositionsUtil/hkFindUniquePositionsUtil.h>

//
//	Constructor

hkcdPlanarGeometry::hkcdPlanarGeometry(PlanesCollection* planesCollection, int initialPolyCapacity, hkcdPlanarEntityDebugger* debugger)
:	hkcdPlanarEntity(debugger)
,	m_planes(planesCollection)
{
	m_polys.setAndDontIncrementRefCount(new hkcdPlanarGeometryPolygonCollection());
	if ( initialPolyCapacity )
	{
		m_polys->create(initialPolyCapacity);
	}
}

//
//	Constructor with polygon collection

hkcdPlanarGeometry::hkcdPlanarGeometry(PlanesCollection* planesCollection, hkcdPlanarGeometryPolygonCollection* polygonCollection, hkcdPlanarEntityDebugger* debugger)
:	hkcdPlanarEntity(debugger)
,	m_planes(planesCollection)
,	m_polys(polygonCollection)
{

}

//
//	Copy constructor

hkcdPlanarGeometry::hkcdPlanarGeometry(const hkcdPlanarGeometry& other)
:	hkcdPlanarEntity(other)
,	m_planes(other.m_planes)
{
	m_polys.setAndDontIncrementRefCount(new hkcdPlanarGeometryPolygonCollection());
	m_polys->copy(*other.m_polys);
}

//
//	Welds the planes so that all planes are unique

void hkcdPlanarGeometry::weldPlanes(hkArray<int>* planeRemapTable)
{
	// Save old planes
	PlanesCollection oldPlanes(*m_planes);

	// Weld our planes
	hkArray<int> remapTable;
	if ( !planeRemapTable )
	{
		planeRemapTable = &remapTable;
	}
	m_planes->weldPlanes(planeRemapTable);

	// Remap
	{
		PlanesCollection* newPlanes = m_planes;
		newPlanes->addReference();
		m_planes = &oldPlanes;
	
		setPlanesCollection(newPlanes, planeRemapTable->begin());
		newPlanes->removeReference();
	}
}

//
//	Sets a new planes collection. If the plane remapping table is non-null, the plane Ids on all nodes will be re-set as well (i.e. to match the plane Ids in the new collection)

void hkcdPlanarGeometry::setPlanesCollection(PlanesCollection* newPlanes, const int* HK_RESTRICT planeRemapTable)
{
	if ( m_planes && newPlanes && planeRemapTable && (m_planes != newPlanes) )
	{
		// Remap all polygons
		for (PolygonId polyId = m_polys->getFirstPolygonId(); polyId.isValid(); polyId = m_polys->getNextPolygonId(polyId))
		{
			Polygon& poly		= accessPolygon(polyId);
			const int numBounds = getNumBoundaryPlanes(polyId);

			// Remap boundary
			for (int k = numBounds - 1; k >= 0; k--)
			{
				const PlaneId oldPlaneId = poly.getBoundaryPlaneId(k);

				if ( oldPlaneId.isValid() )
				{
					const int oldPlaneIdx		= oldPlaneId.value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);
					const int newPlaneIdx		= planeRemapTable[oldPlaneIdx] & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);
					Plane oldPlane;				m_planes->getPlane(oldPlaneId, oldPlane);
					Plane newPlane;				newPlanes->getPlane(PlaneId(newPlaneIdx), newPlane);
					const PlaneId newPlaneId	( newPlaneIdx | (hkcdPlanarGeometryPredicates::sameOrientation(oldPlane, newPlane) ? 0 : hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG));

					poly.setBoundaryPlaneId(k, newPlaneId);
				}
			}

			// Remap support
			{
				const PlaneId oldPlaneId	= poly.getSupportPlaneId();

				if ( oldPlaneId.isValid() )
				{
					const int oldPlaneIdx		= oldPlaneId.value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);
					const int newPlaneIdx		= planeRemapTable[oldPlaneIdx] & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);
					Plane oldPlane;				m_planes->getPlane(oldPlaneId, oldPlane);
					Plane newPlane;				newPlanes->getPlane(PlaneId(newPlaneIdx), newPlane);
					const PlaneId newPlaneId	( newPlaneIdx | (hkcdPlanarGeometryPredicates::sameOrientation(oldPlane, newPlane) ? 0 : hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG));

					poly.setSupportPlaneId(newPlaneId);
				}
			}
		}
	}

	m_planes = newPlanes;
}

//
//	Shift all plane ids in the geom polygons

void hkcdPlanarGeometry::shiftPlaneIds(int offsetValue)
{
	// Remap all polygons
	for (PolygonId polyId = m_polys->getFirstPolygonId(); polyId.isValid(); polyId = m_polys->getNextPolygonId(polyId))
	{
		Polygon& poly		= accessPolygon(polyId);
		const int numBounds = getNumBoundaryPlanes(polyId);

		// Remap boundary
		for (int k = numBounds - 1; k >= 0; k--)
		{
			const PlaneId oldPlaneId = poly.getBoundaryPlaneId(k);
			const int oldPlaneIdx		= oldPlaneId.value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);
			if ( oldPlaneIdx >= hkcdPlanarGeometryPlanesCollection::NUM_BOUNDS )
			{
				const PlaneId newPlaneId	( oldPlaneId.value() + offsetValue );
				poly.setBoundaryPlaneId(k, newPlaneId);
			}
		}

		// Remap support
		{
			const PlaneId oldPlaneId	= poly.getSupportPlaneId();
			const int oldPlaneIdx		= oldPlaneId.value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);
			if ( oldPlaneIdx >= hkcdPlanarGeometryPlanesCollection::NUM_BOUNDS )
			{
				const PlaneId newPlaneId	( oldPlaneId.value() + offsetValue );
				poly.setSupportPlaneId(newPlaneId);
			}
		}
	}
}

//
// Checks planes consistency within the geometry

bool hkcdPlanarGeometry::checkPlanesConsistency() const
{
	// Remap all polygons
	for (PolygonId polyId = m_polys->getFirstPolygonId(); polyId.isValid(); polyId = m_polys->getNextPolygonId(polyId))
	{
		const Polygon& poly	= getPolygon(polyId);
		const int numBounds = getNumBoundaryPlanes(polyId);

		// Check boundaries
		for (int k = numBounds - 1; k >= 0; k--)
		{
			const PlaneId oldPlaneId = poly.getBoundaryPlaneId(k);
			const int oldPlaneIdx		= oldPlaneId.value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);
			if ( oldPlaneIdx >= m_planes->getNumPlanes() )
			{
				HK_BREAKPOINT(0);
				return false;
			}
		}

		// Check support
		{
			const PlaneId oldPlaneId	= poly.getSupportPlaneId();
			const int oldPlaneIdx		= oldPlaneId.value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);
			if ( oldPlaneIdx >= m_planes->getNumPlanes() )
			{
				HK_BREAKPOINT(0);
				return false;
			}
		}
	}

	return true;
}

//
//	Adds all polygons from the given geometry to this geometry. The planes are assumed to be already present in this geometry,
//	and a mapping from other to this geometry planes is expected as input

void hkcdPlanarGeometry::appendGeometryPolygons(const hkcdPlanarGeometry& srcGeom, int* HK_RESTRICT planeRemapTable, hkUint32 maxPlaneIdValue, const hkArray<PolygonId>& polygonIdsToAdd, hkArray<PolygonId>& addedPolygonIdsOut, bool flipWinding, int materialOffset)
{
	const PlanesCollection* srcPlanes					= srcGeom.getPlanesCollection();
	const PlanesCollection* thisPlanes					= getPlanesCollection();
	const int inc = flipWinding ? 1 : -1;
	addedPolygonIdsOut.setSize(0);

	for (int pidx = polygonIdsToAdd.getSize() - 1 ; pidx >= 0 ; pidx--)
	{
		const PolygonId srcPolyId	= polygonIdsToAdd[pidx];
		const Polygon& srcPoly		= srcGeom.getPolygon(srcPolyId);
		const int numEdges			= srcGeom.getNumBoundaryPlanes(srcPolyId);
		const PlaneId srcSupportId	= srcPoly.getSupportPlaneId();
		const hkUint32 srcMtlId		= srcPoly.getMaterialId();

		// Add polygon if it has a valid remap value for its support plane
		const int suppPlaneIdx		= srcSupportId.value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);
		if ( (maxPlaneIdValue == 0) || (srcSupportId.isValid() && PlaneId(planeRemapTable[suppPlaneIdx]).isValid() && (PlaneId(planeRemapTable[suppPlaneIdx]).value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG)) < maxPlaneIdValue) )
		{
			// Allocate polygon in the merged geometry
			const PolygonId dstPolyId	= addPolygon(PlaneId::invalid(), srcMtlId + materialOffset, numEdges);
			Polygon& dstPoly			= accessPolygon(dstPolyId);
			addedPolygonIdsOut.pushBack(dstPolyId);

			// Map support plane
			if ( srcSupportId.isValid() )
			{
				const int oldPlaneIdx		= srcSupportId.value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);
				int newPlaneIdx				= planeRemapTable[oldPlaneIdx] & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);;
				Plane oldPlane;				srcPlanes->getPlane(srcSupportId, oldPlane);
				// We may want to add the plane now (only processed if maxPlaneIdValue == 0, to allow including dangling polygons whose support plane might not be part of the solid)
				if ( !PlaneId(newPlaneIdx).isValid() )
				{
					newPlaneIdx						= accessPlanesCollection()->addPlane(oldPlane).value();
					planeRemapTable[oldPlaneIdx]	= newPlaneIdx;
				}
				Plane newPlane;				thisPlanes->getPlane(PlaneId(newPlaneIdx), newPlane);
				PlaneId newPlaneId			( newPlaneIdx | (hkcdPlanarGeometryPredicates::sameOrientation(oldPlane, newPlane) ? 0 : hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG));
				newPlaneId					= flipWinding ? hkcdPlanarGeometryPrimitives::getOppositePlaneId(newPlaneId) : newPlaneId;

				dstPoly.setSupportPlaneId(newPlaneId);
				HK_ASSERT(0x1c0af246, newPlaneId.isValid());
			}

			// Map boundary planes
			for (int sbi = numEdges - 1, dbi = flipWinding ? 0 : sbi; sbi >= 0; sbi--, dbi += inc)
			{
				const PlaneId oldPlaneId = srcPoly.getBoundaryPlaneId(sbi);

				if ( oldPlaneId.isValid() )
				{
					const int oldPlaneIdx		= oldPlaneId.value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);
					int newPlaneIdx				= planeRemapTable[oldPlaneIdx] & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);
					Plane oldPlane;				srcPlanes->getPlane(oldPlaneId, oldPlane);
					// We may want to add the plane now
					if ( !PlaneId(newPlaneIdx).isValid() )
					{
						newPlaneIdx						= accessPlanesCollection()->addPlane(oldPlane).value();
						planeRemapTable[oldPlaneIdx]	= newPlaneIdx;
					}
					Plane newPlane;				thisPlanes->getPlane(PlaneId(newPlaneIdx), newPlane);
					const PlaneId newPlaneId	( newPlaneIdx | (hkcdPlanarGeometryPredicates::sameOrientation(oldPlane, newPlane) ? 0 : hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG));

					dstPoly.setBoundaryPlaneId(dbi, newPlaneId);
				}
			}
		}
	}
}

//
//	Adds the given polygons to this geometry and returns the newly added polygon Ids.

void hkcdPlanarGeometry::appendGeometryPolygons(const hkcdPlanarGeometry& srcGeom, const hkArray<PolygonId>& srcPolygonIds, bool flipWinding, hkArray<PolygonId>& dstPolygonIds, int materialOffset)
{
	const int numPolys = srcPolygonIds.getSize();
	dstPolygonIds.setSize(numPolys);

	hkInplaceArray<PlaneId, 128> bounds;
	const int inc = flipWinding ? 1 : -1;
	for (int k = 0; k < numPolys; k++)
	{
		const PolygonId srcPolyId	= srcPolygonIds[k];
		const Polygon& srcPoly		= srcGeom.getPolygon(srcPolyId);
		const int numEdges			= srcGeom.getNumBoundaryPlanes(srcPolyId);
		const PlaneId srcSupportId	= srcPoly.getSupportPlaneId();
		const hkUint32 srcMtlId		= srcPoly.getMaterialId();

		// Save and eventually flip bounds
		bounds.setSize(numEdges);
		for (int sbi = numEdges - 1, dbi = flipWinding ? 0 : sbi; sbi >= 0; sbi--, dbi += inc)
		{
			bounds[dbi] = srcPoly.getBoundaryPlaneId(sbi);
		}

		// Allocate polygon in the merged geometry
		const PlaneId dstSupportId	= flipWinding ? hkcdPlanarGeometryPrimitives::getOppositePlaneId(srcSupportId) : srcSupportId;
		const PolygonId dstPolyId	= addPolygon(dstSupportId, srcMtlId + materialOffset, numEdges);
		Polygon& dstPoly			= accessPolygon(dstPolyId);
		dstPolygonIds[k]			= dstPolyId;

		// Set boundary planes
		for (int bi = numEdges - 1; bi >= 0; bi--)
		{
			dstPoly.setBoundaryPlaneId(bi, bounds[bi]);
		}
	}
}

//
//	Classifies a triangle w.r.t. a plane. The result is approximative, as it uses floating-point operations

hkcdPlanarGeometryPredicates::Orientation hkcdPlanarGeometry::approxClassify(PolygonId polygonId, PlaneId planeId) const
{
	const PlanesCollection* planeCollection = getPlanesCollection();

	// Get polygon
	const Polygon& polygon = getPolygon(polygonId);
	if ( hkcdPlanarGeometryPrimitives::coplanarPlaneIds(polygon.getSupportPlaneId(), planeId) )
	{
		return hkcdPlanarGeometryPredicates::ON_PLANE;	// Early-out
	}

	Plane splitPlane;	planeCollection->getPlane(planeId, splitPlane);
	Plane s;			planeCollection->getPlane(polygon.getSupportPlaneId(), s);

	// Classify each vertex of the polygon w.r.t the plane
	hkUint32 numBehind = 0, numInFront = 0, numCoplanar = 0;
	const hkUint32 numPolyVerts = getNumBoundaryPlanes(polygonId);
	for (hkUint32 prevVtx = numPolyVerts - 1, crtVtx = 0; crtVtx < numPolyVerts; prevVtx = crtVtx, crtVtx++ )
	{
		Plane prevBound;	planeCollection->getPlane(polygon.getBoundaryPlaneId(prevVtx), prevBound);
		Plane crtBound;		planeCollection->getPlane(polygon.getBoundaryPlaneId(crtVtx), crtBound);

		const hkcdPlanarGeometryPredicates::Orientation ori = hkcdPlanarGeometryPredicates::approximateOrientation(s, prevBound, crtBound, splitPlane);
		switch ( ori )
		{
		case hkcdPlanarGeometryPredicates::BEHIND:		numBehind++;	if ( numInFront )	{	return hkcdPlanarGeometryPredicates::INTERSECT;	}	break;
		case hkcdPlanarGeometryPredicates::IN_FRONT_OF:	numInFront++;	if ( numBehind )	{	return hkcdPlanarGeometryPredicates::INTERSECT;	}	break;
		case hkcdPlanarGeometryPredicates::ON_PLANE:	numCoplanar++;	break;
		default:	break;
		}
	}

	// Return decision
	if ( numBehind && numInFront )	{	return hkcdPlanarGeometryPredicates::INTERSECT;		}
	if ( numInFront )				{	return hkcdPlanarGeometryPredicates::IN_FRONT_OF;	}
	if ( numBehind )				{	return hkcdPlanarGeometryPredicates::BEHIND;		}

	return hkcdPlanarGeometryPredicates::ON_PLANE;
}

//
//	Classifies a triangle w.r.t. a plane

hkcdPlanarGeometryPredicates::Orientation hkcdPlanarGeometry::classify(PolygonId polygonId, PlaneId planeId) const
{
	const PlanesCollection* planeCollection = getPlanesCollection();
	OrientationCache* orientationCache		= planeCollection->getOrientationCache();

	// Get polygon and test for coplanarity
	const Polygon& polygon		= getPolygon(polygonId);
	const PlaneId polySupportId = polygon.getSupportPlaneId();
	if ( hkcdPlanarGeometryPrimitives::coplanarPlaneIds(polySupportId, planeId) )
	{
		return hkcdPlanarGeometryPredicates::ON_PLANE;	// Early-out
	}

	// Classify each vertex of the polygon w.r.t the plane
	hkUint32 numBehind = 0, numInFront = 0, numCoplanar = 0;
	const hkUint32 numPolyVerts = getNumBoundaryPlanes(polygonId);
	PlaneId prevBoundId			= polygon.getBoundaryPlaneId(numPolyVerts - 1);
	Plane splitPlane;			planeCollection->getPlane(planeId, splitPlane);
	Plane s;					planeCollection->getPlane(polySupportId, s);
	Plane prevBound;			planeCollection->getPlane(prevBoundId, prevBound);
	hkIntVector vPlaneIds;		vPlaneIds.set(polySupportId.value(), prevBoundId.value(), 0, planeId.value());

	for (hkUint32 crtVtx = 0; crtVtx < numPolyVerts; crtVtx++ )
	{
		const PlaneId crtBoundId	= polygon.getBoundaryPlaneId(crtVtx);
		Plane crtBound;				planeCollection->getPlane(crtBoundId, crtBound);
		vPlaneIds.setComponent<2>	(crtBoundId.value());

		// Try to get the orientation from cache
		const Orientation ori = hkcdPlanarGeometryPredicates::orientation(s, prevBound, crtBound, splitPlane, vPlaneIds, orientationCache);
		switch ( ori )
		{
		case hkcdPlanarGeometryPredicates::BEHIND:		numBehind++;	if ( numInFront )	{	return hkcdPlanarGeometryPredicates::INTERSECT;	}	break;
		case hkcdPlanarGeometryPredicates::IN_FRONT_OF:	numInFront++;	if ( numBehind )	{	return hkcdPlanarGeometryPredicates::INTERSECT;	}	break;
		case hkcdPlanarGeometryPredicates::ON_PLANE:	numCoplanar++;	break;
		default:	break;
		}

		prevBoundId = crtBoundId;
		prevBound	= crtBound;
		vPlaneIds.	setPermutation<hkVectorPermutation::XZZW>(vPlaneIds);
	}

	// Return decision
	if ( numBehind && numInFront )	{	return hkcdPlanarGeometryPredicates::INTERSECT;		}
	if ( numInFront )				{	return hkcdPlanarGeometryPredicates::IN_FRONT_OF;	}
	if ( numBehind )				{	return hkcdPlanarGeometryPredicates::BEHIND;		}

	HK_ASSERT(0x4add6ee2, numCoplanar == numPolyVerts);
	return hkcdPlanarGeometryPredicates::ON_PLANE;
}

//
//	Clips a polygon with the given planes
//	Encode (oPrev, oCrt, oNext) as a symbol, i.e. -0+:
//	(H added when entering back into the polygon, i.e. on +-!
//		*-*	| B		*+-	| HB		-0-	| B
//					*+0	| NULL		00-	| HB
//					*++	| NULL		+0-	| HB
//									-00	| NULL
//									-0+	| NULL

hkUint32 hkcdPlanarGeometry::clipPolygon(PlaneId supportPlaneId, PlaneId*& boundsIn, PlaneId*& boundsOut, int numBounds, const PlaneId* clippingPlanesIn, int numClippingPlanesIn)
{
	const PlanesCollection* planeCollection = getPlanesCollection();
	OrientationCache* orientationCache		= planeCollection->getOrientationCache();

	// Put input data in out, as it will be swapped in the loop
	hkAlgorithm::swap(boundsIn, boundsOut);
	int numNewBounds	= numBounds;
	numBounds			= 0;
	Plane supportPlane;	planeCollection->getPlane(supportPlaneId, supportPlane);

	hkInplaceArray<Orientation, 128> orientations;

	// Split polygon with each clipping plane
	for (int ci = 0; ci < numClippingPlanesIn; ci++)
	{
		// Input data is in polyOut, swap
		hkAlgorithm::swap(boundsIn, boundsOut);
		hkAlgorithm::swap(numNewBounds, numBounds);

		// Reset new polygon
		numNewBounds = 0;
		if ( !numBounds )
		{
			return 0;	// Empty polygon, nothing to do!
		}

		// Clip current polygon with the current plane
		const PlaneId splitPlaneId(clippingPlanesIn[ci].value() & hkcdPlanarGeometryPrimitives::PLANE_ID_MASK);
		if ( hkcdPlanarGeometryPrimitives::coplanarPlaneIds(supportPlaneId, splitPlaneId) )
		{
			// Check if the polygons have similar orientation
			if ( hkcdPlanarGeometryPrimitives::sameOrientationPlaneIds(supportPlaneId, splitPlaneId) )
			{
				hkAlgorithm::swap(boundsIn, boundsOut);
				hkAlgorithm::swap(numNewBounds, numBounds);
				continue;	// Plane does not affect current polygon
			}
			
			// Polygon is fully clipped, nothing is inside the plane!
			return 0;
		}

		HK_ASSERT(0x4fe84f3a, numBounds >= 3);
		{	
			// Compute all orientations
			orientations.setSize(numBounds);
			{
				Plane boundPrev;			planeCollection->getPlane(boundsIn[numBounds - 1], boundPrev);
				Plane splitPlane;			planeCollection->getPlane(splitPlaneId, splitPlane);
				hkIntVector vPlaneIds;		vPlaneIds.set(supportPlaneId.value(), boundsIn[numBounds - 1].value(), 0, splitPlaneId.value());

				for (int crt = 0; crt < numBounds; crt++)
				{
					// Compute orientation predicate
					const PlaneId boundCrtId	= boundsIn[crt];
					Plane boundCrt;				planeCollection->getPlane(boundCrtId, boundCrt);
					vPlaneIds.setComponent<2>	(boundCrtId.value());
					orientations[crt]			= hkcdPlanarGeometryPredicates::orientation(supportPlane, boundPrev, boundCrt, splitPlane, vPlaneIds, orientationCache);
					vPlaneIds.setPermutation<hkVectorPermutation::XZZW>(vPlaneIds);
					boundPrev					= boundCrt;
				}
			}
			
			PlaneId boundCrtId	= boundsIn[numBounds - 1];
			Orientation oPrev	= orientations[numBounds - 2];
			Orientation oCrt	= orientations[numBounds - 1];
			for (int next = 0; next < numBounds; next++)
			{
				// Compute orientation predicate
				PlaneId boundNextId		= boundsIn[next];
				const Orientation oNext = orientations[next];

				// Add a plane to the boundary based on the orientations
				if ( oCrt == hkcdPlanarGeometryPredicates::BEHIND )
				{
					// Codes *-*
					boundsOut[numNewBounds++] = boundCrtId;
				}
				else if ( oCrt == hkcdPlanarGeometryPredicates::IN_FRONT_OF )
				{
					// Codes *+*
					if ( oNext == hkcdPlanarGeometryPredicates::BEHIND )
					{
						// Code *+-
						boundsOut[numNewBounds++] = splitPlaneId;
						boundsOut[numNewBounds++] = boundCrtId;
					}
				}
				else if ( oNext == hkcdPlanarGeometryPredicates::BEHIND )	// Codes *0*
				{
					// Codes *0-
					if ( oPrev != hkcdPlanarGeometryPredicates::BEHIND )
					{
						// Codes 00-, +0-
						boundsOut[numNewBounds++] = splitPlaneId;
					}
					boundsOut[numNewBounds++] = boundCrtId;
				}

				oPrev	= oCrt;
				oCrt	= oNext;
				boundCrtId	= boundNextId;
			}
		}
	}

	return numNewBounds;
}

//
//	Splits a polygon with the given splitting plane. Returns the part of the polygon inside the given plane
//	Encode (oPrev, oCrt, oNext) as a symbol, i.e. -0+:
//	(H added when entering back into the polygon, i.e. on +-!
//		*-*	| B		*+-	| HB		-0-	| B
//					*+0	| NULL		00-	| HB
//					*++	| NULL		+0-	| HB
//									-00	| NULL
//									-0+	| NULL

void hkcdPlanarGeometry::split(PolygonId polygonId, PlaneId h, PolygonId& insidePolyId, PolygonId& outsidePolyId)
{
	const PlanesCollection* planeCollection = getPlanesCollection();
	OrientationCache* orientationCache		= planeCollection->getOrientationCache();

	// Get planes
	const Polygon& origPoly			= getPolygon(polygonId);
	const PlaneId supportPlaneId	= origPoly.getSupportPlaneId();

	// Check for same plane
	if ( hkcdPlanarGeometryPrimitives::coplanarPlaneIds(supportPlaneId, h) )
	{
		// Check if the polygons have similar orientation
		if ( hkcdPlanarGeometryPrimitives::sameOrientationPlaneIds(supportPlaneId, h) )
		{
			// h == m_support, entire polygon is inside
			insidePolyId	= polygonId;
			outsidePolyId	= PolygonId::invalid();
		}
		else
		{
			// h == -m_support, entire polygon is outside
			insidePolyId	= PolygonId::invalid();
			outsidePolyId	= polygonId;
		}
		return;
	}

	// Support and h are not coincident, we have an intersection!
	const int numOrigBounds	= getNumBoundaryPlanes(polygonId);
	int numBoundsIn			= 0;
	int numBoundsOut		= 0;
	HK_ASSERT(0x28587a13, numOrigBounds >= 3);

	// Alloc some working buffers. Each polygon can have at most numBounds + 1 planes
	hkLocalBuffer<PlaneId> boundsTemp((numOrigBounds + 1) * 2);
	hkLocalBuffer<Orientation> orientations(numOrigBounds);
	PlaneId* boundsIn	= &boundsTemp[0];
	PlaneId* boundsOut	= &boundsTemp[numOrigBounds + 1];

	Plane supportPlane;	planeCollection->getPlane(supportPlaneId, supportPlane);
	Plane splitPlane;	planeCollection->getPlane(h, splitPlane);
	const PlaneId invH	= hkcdPlanarGeometryPrimitives::getOppositePlaneId(h);
	{
		// Compute all orientations
		{
			PlaneId boundPrevId		= origPoly.getBoundaryPlaneId(numOrigBounds - 1);
			Plane boundPrev;		planeCollection->getPlane(boundPrevId, boundPrev);
			hkIntVector vPlaneIds;	vPlaneIds.set(supportPlaneId.value(), boundPrevId.value(), 0, h.value());

			for (int crt = 0; crt < numOrigBounds; crt++)
			{
				// Compute orientation predicate
				const PlaneId boundCrtId	= origPoly.getBoundaryPlaneId(crt);
				Plane boundCrt;				planeCollection->getPlane(boundCrtId, boundCrt);
				vPlaneIds.setComponent<2>	(boundCrtId.value());
				orientations[crt]			= hkcdPlanarGeometryPredicates::orientation(supportPlane, boundPrev, boundCrt, splitPlane, vPlaneIds, orientationCache);
				boundPrevId					= boundCrtId;
				boundPrev					= boundCrt;
				vPlaneIds.setPermutation<hkVectorPermutation::XZZW>(vPlaneIds);
			}
		}

		PlaneId boundCrtId	= origPoly.getBoundaryPlaneId(numOrigBounds - 1);
		Orientation oPrev	= orientations[numOrigBounds - 2];
		Orientation oCrt	= orientations[numOrigBounds - 1];
		for (int next = 0; next < numOrigBounds; next++)
		{
			// Compute orientation predicate
			PlaneId boundNextId	= origPoly.getBoundaryPlaneId(next);
			Plane boundNext;	planeCollection->getPlane(boundNextId, boundNext);
			const Orientation oNext = orientations[next];

			// Add a plane to the boundary based on the orientations
			if ( oCrt == hkcdPlanarGeometryPredicates::BEHIND )
			{
				// Inside Codes:	*-*
				boundsIn[numBoundsIn++] = boundCrtId;

				// Outside Codes:	*+*
				if ( oNext == hkcdPlanarGeometryPredicates::IN_FRONT_OF )
				{
					// Outside Codes:	*+-
					boundsOut[numBoundsOut++] = invH;
					boundsOut[numBoundsOut++] = boundCrtId;
				}
			}
			else if ( oCrt == hkcdPlanarGeometryPredicates::IN_FRONT_OF )
			{
				// Inside Codes:	*+*
				if ( oNext == hkcdPlanarGeometryPredicates::BEHIND )
				{
					// Code *+-
					boundsIn[numBoundsIn++] = h;
					boundsIn[numBoundsIn++] = boundCrtId;
				}

				// Outside Codes:	*-*
				boundsOut[numBoundsOut++] = boundCrtId;
			}
			else
			{
				// Inside / Outside Codes: *0*
				if ( oNext == hkcdPlanarGeometryPredicates::BEHIND )
				{
					// Inside Codes *0-
					if ( oPrev != hkcdPlanarGeometryPredicates::BEHIND )
					{
						// Codes 00-, +0-
						boundsIn[numBoundsIn++] = h;
					}
					boundsIn[numBoundsIn++] = boundCrtId;
				}
				else if ( oNext == hkcdPlanarGeometryPredicates::IN_FRONT_OF )
				{
					// Outside Codes *0-
					if ( oPrev != hkcdPlanarGeometryPredicates::IN_FRONT_OF )
					{
						// Outside Codes 00-, +0-
						boundsOut[numBoundsOut++] = invH;
					}
					boundsOut[numBoundsOut++] = boundCrtId;
				}
			}

			oPrev	= oCrt;
			oCrt	= oNext;
			boundCrtId	= boundNextId;
		}
	}

	// Add the polygons
	{
		// Create the new polygons
		const hkUint32 mtlId	= origPoly.getMaterialId();
		insidePolyId			= addPolygon(supportPlaneId, mtlId, numBoundsIn);
		outsidePolyId			= addPolygon(supportPlaneId, mtlId, numBoundsOut);
		Polygon& insidePoly		= accessPolygon(insidePolyId);
		Polygon& outsidePoly	= accessPolygon(outsidePolyId);
		
		for (int k = numBoundsIn - 1; k >= 0; k--)	{	insidePoly.setBoundaryPlaneId(k, boundsIn[k]);	}
		for (int k = numBoundsOut - 1; k >= 0; k--)	{	outsidePoly.setBoundaryPlaneId(k, boundsOut[k]);}
	}
}

//
//	Returns true if two given polygon on the same support plane potentially overlap.

bool hkcdPlanarGeometry::check2dIntersection(const PolygonId& polygonId1, const PolygonId& polygonId2) const
{
	const PlanesCollection* planeCollection = getPlanesCollection();
	OrientationCache* orientationCache		= planeCollection->getOrientationCache();

	// Check if a separating plane can be found for the two polygons
	int numBounds[2];
	const Polygon* poly[2];
	numBounds[0]					= getNumBoundaryPlanes(polygonId1);
	poly[0]							= &(getPolygon(polygonId1));
	numBounds[1]					= getNumBoundaryPlanes(polygonId2);
	poly[1]							= &(getPolygon(polygonId2));
	bool foundBehind[2]				= { false, false };
	bool foundInFrontOf[2]			= { false, false };
	bool foundOnPlane[2]			= { false, false };
	const PlaneId supportPlaneId	= poly[0]->getSupportPlaneId();
	Plane supportPlane;				planeCollection->getPlane(supportPlaneId, supportPlane);

	// Gather all the possible separating planes (boundaries of A and B)
	hkArray<PlaneId> separatingPlanesIds(numBounds[0] + numBounds[1]);
	int nbSepPlanes = 0;
	for (int l = 0 ; l < 2 ; l++)
	{
		for (int b = 0 ; b < numBounds[l] ; b++)
		{
			separatingPlanesIds[nbSepPlanes] = poly[l]->getBoundaryPlaneId(b);
			nbSepPlanes++;
		}
	}

	// Classify the vertices of A and B against the separating planes
	for (int s = 0 ; s < nbSepPlanes ; s++)
	{
		PlaneId separatingPlaneId	= separatingPlanesIds[s];
		Plane separatingPlane;		planeCollection->getPlane(separatingPlaneId, separatingPlane);

		// Do two checks: planes of poly A against vertices of poly B and planes of poly B against vertices of poly A
		for (int l = 0 ; l < 2 ; l++)
		{
			foundBehind[l] = false;
			foundInFrontOf[l] = false;
			foundOnPlane[l] = false;

			// Check if all the vertices of poly B are on the same side of the plane
			PlaneId boundPrevId		= poly[l]->getBoundaryPlaneId(numBounds[l] - 1);
			Plane boundPrev;		planeCollection->getPlane(boundPrevId, boundPrev);
			hkIntVector vPlaneIds;	vPlaneIds.set(supportPlaneId.value(), boundPrevId.value(), 0, separatingPlaneId.value());

			for (int crt = 0; crt < numBounds[l]; crt++)
			{
				// Compute orientation predicate
				const PlaneId boundCrtId		= poly[l]->getBoundaryPlaneId(crt);
				Plane boundCrt;					planeCollection->getPlane(boundCrtId, boundCrt);
				vPlaneIds.setComponent<2>		(boundCrtId.value());
				const Orientation orientations	= hkcdPlanarGeometryPredicates::orientation(supportPlane, boundPrev, boundCrt, separatingPlane, vPlaneIds, orientationCache);
				boundPrevId						= boundCrtId;
				boundPrev						= boundCrt;
				vPlaneIds.setPermutation<hkVectorPermutation::XZZW>(vPlaneIds);

				if ( orientations == hkcdPlanarGeometryPredicates::BEHIND )			{	foundBehind[l] = true;		}
				if ( orientations == hkcdPlanarGeometryPredicates::IN_FRONT_OF )	{	foundInFrontOf[l] = true;	}
				if ( orientations == hkcdPlanarGeometryPredicates::ON_PLANE )		{	foundOnPlane[l] = true;		}
			}

		}

		// A separating plane has been found, the two polygons do not overlap
		if ( (!foundInFrontOf[0] && !foundBehind[1]) || (!foundBehind[0] && !foundInFrontOf[1]) )
		{
			return false;
		}

	}

	return true;
}

bool HK_CALL hkcdPlanarGeometry::check2dIntersection(	const hkcdPlanarGeometry& geom1, const PolygonId& polygonId1,
														const hkcdPlanarGeometry& geom2, const PolygonId& polygonId2)
{
	HK_ASSERT(0x5e4a6d5b, geom1.getPlanesCollection() == geom2.getPlanesCollection());
	const PlanesCollection* planeColl	= geom1.getPlanesCollection();
	OrientationCache* orientationCache	= planeColl->getOrientationCache();

	// Check if a separating plane can be found for the two polygons
	int numBounds[2];
	const Polygon* poly[2];
	numBounds[0]					= geom1.getNumBoundaryPlanes(polygonId1);
	poly[0]							= &(geom1.getPolygon(polygonId1));
	numBounds[1]					= geom2.getNumBoundaryPlanes(polygonId2);
	poly[1]							= &(geom2.getPolygon(polygonId2));
	bool foundBehind[2]				= { false, false };
	bool foundInFrontOf[2]			= { false, false };
	bool foundOnPlane[2]			= { false, false };
	const PlaneId supportPlaneId	= poly[0]->getSupportPlaneId();
	Plane supportPlane;				planeColl->getPlane(supportPlaneId, supportPlane);

	// Gather all the possible separating planes (boundaries of A and B)
	hkArray<PlaneId> separatingPlanesIds(numBounds[0] + numBounds[1]);
	int nbSepPlanes = 0;
	for (int l = 0 ; l < 2 ; l++)
	{
		for (int b = 0 ; b < numBounds[l] ; b++)
		{
			separatingPlanesIds[nbSepPlanes] = poly[l]->getBoundaryPlaneId(b);
			nbSepPlanes++;
		}
	}

	// Classify the vertices of A and B against the separating planes
	for (int s = 0 ; s < nbSepPlanes ; s++)
	{
		PlaneId separatingPlaneId	= separatingPlanesIds[s];
		Plane separatingPlane;		planeColl->getPlane(separatingPlaneId, separatingPlane);

		// Do two checks: planes of poly A against vertices of poly B and planes of poly B against vertices of poly A
		for (int l = 0 ; l < 2 ; l++)
		{
			foundBehind[l] = false;
			foundInFrontOf[l] = false;
			foundOnPlane[l] = false;

			// Check if all the vertices of poly B are on the same side of the plane
			PlaneId boundPrevId		= poly[l]->getBoundaryPlaneId(numBounds[l] - 1);
			Plane boundPrev;		planeColl->getPlane(boundPrevId, boundPrev);
			hkIntVector vPlaneIds;	vPlaneIds.set(supportPlaneId.value(), boundPrevId.value(), 0, separatingPlaneId.value());

			for (int crt = 0; crt < numBounds[l]; crt++)
			{
				// Compute orientation predicate
				const PlaneId boundCrtId		= poly[l]->getBoundaryPlaneId(crt);
				Plane boundCrt;					planeColl->getPlane(boundCrtId, boundCrt);
				vPlaneIds.setComponent<2>		(boundCrtId.value());
				const Orientation orientations	= hkcdPlanarGeometryPredicates::orientation(supportPlane, boundPrev, boundCrt, separatingPlane, vPlaneIds, orientationCache);
				boundPrevId						= boundCrtId;
				boundPrev						= boundCrt;
				vPlaneIds.setPermutation<hkVectorPermutation::XZZW>(vPlaneIds);

				if ( orientations == hkcdPlanarGeometryPredicates::BEHIND )			{	foundBehind[l] = true;		}
				if ( orientations == hkcdPlanarGeometryPredicates::IN_FRONT_OF )	{	foundInFrontOf[l] = true;	}
				if ( orientations == hkcdPlanarGeometryPredicates::ON_PLANE )		{	foundOnPlane[l] = true;		}
			}
		}

		// A separating plane has been found, the two polygons do not overlap
		if ( (!foundInFrontOf[0] && !foundBehind[1]) || (!foundBehind[0] && !foundInFrontOf[1]) )
		{
			return false;
		}
	}

	return true;
}

//
//	Computes the intersection and difference of two input polygons. Will add new polygons to the geometry. Returns true if the intersection is not empty

// bool hkcdPlanarGeometry::computeIntersectionAndDifferenceOfPolys(const PolygonId polygonIdA, const PolygonId polygonIdB, PolygonId& intersectionPolyIds, hkArray<PolygonId>& differencePolyIds)
// {
// 	// Check for potential intersection between poly A and B		
// 	const hkBool32 intersect = check2dIntersection(polygonIdA, polygonIdB);
// 	if ( !intersect )
// 	{
// 		return false;
// 	}
// 
// 	// If the two polygon intersect, split polygon of A with each boundary plane of B
// 	hkArray<PolygonId> intermediatePolygonIds;
// 	if ( intersect )
// 	{
// 		const int numBoundsB	= getNumBoundaryPlanes(polygonIdB);
// 		PolygonId lastInPid		= polygonIdA;
// 
// 		for (int b = 0; b < numBoundsB; b++)
// 		{
// 			const PlaneId splitPlaneId		= getPolygon(polygonIdB).getBoundaryPlaneId(b);
// 			const Orientation orientation	= classify(lastInPid, splitPlaneId);
// 
// 			if ( orientation == hkcdPlanarGeometryPredicates::INTERSECT )
// 			{
// 				// Update intermediate polygon id
// 				if ( lastInPid.value() != polygonIdA.value() )
// 				{
// 					intermediatePolygonIds.pushBack(lastInPid);
// 				}
// 
// 				// Split
// 				PolygonId polyIn, polyOut;
// 				split(lastInPid, splitPlaneId, polyIn, polyOut);
// 
// 				// Add the outside polygon to the list
// 				differencePolyIds.pushBack(polyOut);
// 				lastInPid = polyIn;
// 			}
// 		}
// 
// 		// Inside poly will be the last inside succesful split poly
// 		intersectionPolyIds = lastInPid;
// 	}
// 
// 	removePolygons(intermediatePolygonIds);
// 
// 	return true;
// }

//
//	Removes the given polygons from the mesh

void hkcdPlanarGeometry::removePolygons(const hkArray<PolygonId>& polygonIds)
{
	for (int k = polygonIds.getSize() - 1; k >= 0; k--)
	{
		const PolygonId polyId = polygonIds[k];
		m_polys->freePolygon(polyId);
	}
}

//
//	Retrieves all valid polygon Ids

void hkcdPlanarGeometry::getAllPolygons(hkArray<PolygonId>& polygonsOut) const
{
	for (PolygonId polyId = m_polys->getFirstPolygonId(); polyId.isValid(); polyId = m_polys->getNextPolygonId(polyId))
	{
		polygonsOut.pushBack(polyId);
	}
}

//
//	Collects all planes used by the given polygons

void hkcdPlanarGeometry::getAllPolygonsPlanes(const hkArray<PolygonId>& polygonsIn, hkArray<PlaneId>& planesOut, bool collectBoundaryPlanes) const
{
	// Reset planes
	planesOut.setSize(0);

	const int numPolys = polygonsIn.getSize();
	for (int k = numPolys - 1; k >= 0; k--)
	{
		const PolygonId polyId	= polygonsIn[k];
		const Polygon& poly		= getPolygon(polyId);
		const int numBounds		= collectBoundaryPlanes ? getNumBoundaryPlanes(polyId) : 0;

		// Add boundary
		PlaneId* planes = planesOut.expandBy(numBounds + 1);
		for (int i = numBounds - 1; i >= 0; i--)
		{
			// Remove the flip bit so we can collapse all identical planes!
			planes[i] = PlaneId(poly.getBoundaryPlaneId(i).value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG));
		}

		// Add support
		planes[numBounds] = PlaneId(poly.getSupportPlaneId().value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG));
	}

	// Sort all plane Ids and remove duplicates
	const int numPlanes = planesOut.getSize();
	hkSort(reinterpret_cast<PlaneId::Type*>(planesOut.begin()), numPlanes);
	const int numUniquePlanes = hkAlgorithm::removeDuplicatesFromSortedList(planesOut.begin(), numPlanes);
	planesOut.setSize(numUniquePlanes);
}

//
//	Builds a vertex-based geometry representation from this entity.

void hkcdPlanarGeometry::extractGeometry(hkGeometry& geomOut) const
{
	hkFindUniquePositionsUtil vtxWelder;
	hkArray<int> polyIb;

	// Triangulate all polygons, they are convex!
	for (PolygonId polyId = m_polys->getFirstPolygonId(); polyId.isValid(); polyId = m_polys->getNextPolygonId(polyId))
	{
		const Polygon& poly = getPolygon(polyId);

		// Compute all vertices
		const int numPolyVerts = getNumBoundaryPlanes(polyId);
		polyIb.setSize(numPolyVerts);

		Plane planes[3];	m_planes->getPlane(poly.getSupportPlaneId(), planes[0]);
							m_planes->getPlane(poly.getBoundaryPlaneId(numPolyVerts - 1), planes[1]);
		for (int crt = 0; crt < numPolyVerts; crt++)
		{
			m_planes->getPlane(poly.getBoundaryPlaneId(crt), planes[2]);
			hkIntVector iv;		hkcdPlanarGeometryPredicates::approximateIntersection(planes, iv);
			hkVector4 fv;		m_planes->convertFixedPosition(iv, fv);
			polyIb[crt] 		= vtxWelder.addPosition(fv);
			planes[1]			= planes[2];
		}

		// Triangulate. The polygon is convex
		for (int k = 2; k < numPolyVerts; k++)
		{
			hkGeometry::Triangle& tri = geomOut.m_triangles.expandOne();
			tri.set(polyIb[0], polyIb[k - 1], polyIb[k], polyId.value());
		}
	}

	geomOut.m_vertices.swap(vtxWelder.m_positions);
}

//
//	Builds a vertex-based geometry representation for the given polygon

void hkcdPlanarGeometry::extractPolygonGeometry(PolygonId polyId, hkGeometry& geomOut) const
{
	const PlanesCollection* planeCollection = getPlanesCollection();

	const Polygon& poly = getPolygon(polyId);

	// Compute all vertices
	const int numPolyVerts = getNumBoundaryPlanes(polyId);
	geomOut.m_vertices.setSize(numPolyVerts);

	Plane planes[3];	planeCollection->getPlane(poly.getSupportPlaneId(), planes[0]);
	planeCollection->getPlane(poly.getBoundaryPlaneId(numPolyVerts - 1), planes[1]);
	for (int crt = 0; crt < numPolyVerts; crt++)
	{
		planeCollection->getPlane(poly.getBoundaryPlaneId(crt), planes[2]);
		hkIntVector iv;		hkcdPlanarGeometryPredicates::approximateIntersection(planes, iv);
		planeCollection->convertFixedPosition(iv, geomOut.m_vertices[crt]);
		planes[1]			= planes[2];
	}

	// Triangulate. The polygon is convex
	for (int k = 2; k < numPolyVerts; k++)
	{
		hkGeometry::Triangle& tri = geomOut.m_triangles.expandOne();
		tri.set(0, k - 1, k, polyId.value());
	}
}

//
//	Builds a vertex-based geometry representation for the given polygon

void hkcdPlanarGeometry::extractPolygonsGeometry(const hkArray<PolygonId>& polyIds, hkGeometry& geomOut) const
{
	const PlanesCollection* planeCollection = getPlanesCollection();
	geomOut.m_vertices.setSize(0);
	geomOut.m_triangles.setSize(0);

	for (int k = 0; k < polyIds.getSize(); k++)
	{
		// Get polygon
		const PolygonId polyId	= polyIds[k];
		const Polygon& poly		= getPolygon(polyId);

		// Compute all vertices
		const int numPolyVerts	= getNumBoundaryPlanes(polyId);
		const int vbIdxBase		= geomOut.m_vertices.getSize();
		hkVector4* vbPtr		= geomOut.m_vertices.expandBy(numPolyVerts);

		Plane planes[3];	planeCollection->getPlane(poly.getSupportPlaneId(), planes[0]);
		planeCollection->getPlane(poly.getBoundaryPlaneId(numPolyVerts - 1), planes[1]);
		for (int crt = 0; crt < numPolyVerts; crt++)
		{
			planeCollection->getPlane(poly.getBoundaryPlaneId(crt), planes[2]);
			hkIntVector iv;		hkcdPlanarGeometryPredicates::approximateIntersection(planes, iv);
			planeCollection->convertFixedPosition(iv, vbPtr[crt]);
			planes[1]			= planes[2];
		}

		// Triangulate. The polygon is convex
		for (int i = 2; i < numPolyVerts; i++)
		{
			hkGeometry::Triangle& tri = geomOut.m_triangles.expandOne();
			tri.set(vbIdxBase, vbIdxBase + i - 1, vbIdxBase + i, polyId.value());
		}
	}
}

//
//	Removes all polygons that are not present in the given list. The given list of polygon Ids must be sorted ascending!

void hkcdPlanarGeometry::keepPolygons(const hkArray<PolygonId>& polygonIds)
{
	PolygonId polyIdB = m_polys->getLastPolygonId();
	for (int ia = polygonIds.getSize() - 1; (ia >= 0) && polyIdB.isValid(); )
	{
		const PolygonId polyIdA = polygonIds[ia];

		if ( polyIdA > polyIdB )
		{
			// Poly A is marked for keep, but not in B? Should not happen!
			HK_ASSERT2(0x7086ac06, false, "Input polygons are not sorted!");
			ia--;
		}
		else if ( polyIdA == polyIdB )
		{
			// Poly A is marked for keep!
			ia--;
			polyIdB = m_polys->getPrevPolygonId(polyIdB);
		}
		else // (polyIdA < polyIdB)
		{
			// Poly B dies
			const PolygonId deletedPolyId = polyIdB;
			polyIdB = m_polys->getPrevPolygonId(polyIdB);
			m_polys->freePolygon(deletedPolyId);
		}
	}

	// Can release the remaining polys!
	while ( polyIdB.isValid() )
	{
		const PolygonId deletedPolyId = polyIdB;
		polyIdB = m_polys->getPrevPolygonId(polyIdB);
		m_polys->freePolygon(deletedPolyId);
	}
}

//
//	Remaps the material ids of the polygons given a remap table

void hkcdPlanarGeometry::remapPolygonMaterialIds(hkArray<hkUint32>& remapTable)
{
	// Loop over all the polygons of the geometry
	for (PolygonId polyId	= m_polys->getFirstPolygonId(); polyId.isValid(); polyId = m_polys->getNextPolygonId(polyId))
	{
		Polygon& poly		= accessPolygon(polyId);
		poly.setMaterialId(remapTable[poly.getMaterialId()]);
	}
}

//
//	Collects all the plane Ids used by the convex cells

void hkcdPlanarGeometry::collectUsedPlaneIds(hkBitField& usedPlaneIdsOut) const
{	
	// Loop over all the polygons of the geometry
	for (PolygonId polyId = m_polys->getFirstPolygonId(); polyId.isValid(); polyId = m_polys->getNextPolygonId(polyId))
	{
		const Polygon& poly		= getPolygon(polyId);
		const int numPolyVerts	= getNumBoundaryPlanes(polyId);

		// Add support plane
		{
			const PlaneId planeId	= poly.getSupportPlaneId();
			const int planeIdx		= planeId.value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);

			usedPlaneIdsOut.set(planeIdx);
		}

		// Add boundary planes
		for (int k = 0; k < numPolyVerts; k++)
		{
			const PlaneId boundId	= poly.getBoundaryPlaneId(k);
			const int planeIdx		= boundId.value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);

			usedPlaneIdsOut.set(planeIdx);
		}
	}
}

//
//	Returns whether the geometry contains polygon with invalid material

bool hkcdPlanarGeometry::containsPolygonsWithInvalidMaterial() const
{
	// Loop over all the polygons of the geometry
	for (PolygonId polyId = m_polys->getFirstPolygonId(); polyId.isValid(); polyId = m_polys->getNextPolygonId(polyId))
	{
		const Polygon& poly		= getPolygon(polyId);
		if ( poly.getMaterialId() >= INVALID_MATERIAL_ID )
		{
			return true;
		}
	}

	return false;
}

/*
 * Havok SDK - Base file, BUILD(#20130912)
 * 
 * Confidential Information of Havok.  (C) Copyright 1999-2013
 * Telekinesys Research Limited t/a Havok. All Rights Reserved. The Havok
 * Logo, and the Havok buzzsaw logo are trademarks of Havok.  Title, ownership
 * rights, and intellectual property rights in the Havok software remain in
 * Havok and/or its suppliers.
 * 
 * Use of this software for evaluation purposes is subject to and indicates
 * acceptance of the End User licence Agreement for this product. A copy of
 * the license is included with this software and is also available from salesteam@havok.com.
 * 
 */
