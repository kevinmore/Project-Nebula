/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Geometry/Collide/hkcdCollide.h>
#include <Geometry/Collide/DataStructures/Planar/CSG/hkcdPlanarCsgOperand.h>
#include <Geometry/Collide/DataStructures/Planar/Geometry/hkcdVertexGeometry.h>
#include <Geometry/Collide/DataStructures/Planar/CSG/hkcdPlanarGeometryBooleanUtil.h>
#include <Geometry/Collide/DataStructures/Planar/Utils/hkcdPlanarGeometrySimplifier.h>

//
//	Types

typedef hkcdPlanarGeometry::Plane					Plane;
typedef hkcdPlanarGeometry::PlaneId					PlaneId;
typedef hkcdPlanarGeometry::Polygon					Polygon;
typedef hkcdPlanarGeometry::PolygonId				PolygonId;

namespace hkcdOperandImpl 
{

	/// Source polygon
	struct SrcPoly
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY, hkcdOperandImpl::SrcPoly);

		/// Compares two polygons
		static HK_FORCE_INLINE bool HK_CALL less(const SrcPoly& pa, const SrcPoly& pb);

		/// Returns the support plane id
		HK_FORCE_INLINE PlaneId getSupportPlaneId() const;

		/// Returns the source entity
		HK_FORCE_INLINE int getSource() const;

		/// Sets the values
		HK_FORCE_INLINE void set(PolygonId polyId, PlaneId planeId, int polySource);

		PolygonId m_polyId;		///< Id of the source polygon
		hkUint32 m_uid;			///< A unique identifier
	};

	//
	//	Computes a unique identifier

	HK_FORCE_INLINE void SrcPoly::set(PolygonId polyId, PlaneId planeId, int polySource)
	{
		// We've got a free bit from the flipped plane flag we just turned-off, so reuse it and add the poly source as the last bit!
		// This way, coplanar polygons will be sorted by source as well.
		const hkUint32 uId = ((planeId.valueUnchecked() & ~(hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG)) << 1);

		m_uid		= uId | (polySource & 1);
		m_polyId	= polyId;
	}

	//
	//	Compares two polygons

	HK_FORCE_INLINE bool HK_CALL SrcPoly::less(const SrcPoly& pa, const SrcPoly& pb)
	{
		return (pa.m_uid < pb.m_uid);
	}

	//
	//	Returns the support plane id

	HK_FORCE_INLINE PlaneId SrcPoly::getSupportPlaneId() const
	{
		return PlaneId(m_uid >> 1);
	}

	//
	//	Returns the source Entity

	HK_FORCE_INLINE int SrcPoly::getSource() const
	{
		return (m_uid & 1);
	}

	//
	//	Build a 2D BSP tree for each face of a given geometry

	static void HK_CALL buildBSPsFromGeometryPlanes(const hkcdPlanarGeometry& geomIn, const hkArray<PolygonId>& boundaryPolyIds, const hkcdPlanarGeometryPlanesCollection* planesCollection, const hkcdConvexCellsTree3D* cells, hkArray< hkRefPtr<hkcdPlanarSolid> >& BSPsOut)
	{
		// Regroup boundary polys by plane, and store surface plane Ids
		hkArray< hkArray<PolygonId> > planeIdToPolyIds;
		const int numBoundaryPlaneIds = planesCollection->getNumPlanes()*2;
		planeIdToPolyIds.setSize(numBoundaryPlaneIds);
		for (int i = boundaryPolyIds.getSize() - 1 ; i >=0 ; i--)
		{
			const PlaneId planeId	= geomIn.getPolygon(boundaryPolyIds[i]).getSupportPlaneId();
			const hkUint32 planeIdx	= planeId.value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);
			const hkUint32 flipped	= ( planeId.value() != planeIdx ) ? 1 : 0;
			const hkUint32 indx		= planeIdx*2 + flipped;
			planeIdToPolyIds[indx].pushBack(boundaryPolyIds[i]);
		}

		// For each plane, build the bsp
		BSPsOut.setSize(numBoundaryPlaneIds, HK_NULL);
		for (int i = 0 ; i < numBoundaryPlaneIds ; i++)
		{
			const hkArray<PolygonId>& polyIdsOnPlane = planeIdToPolyIds[i];
			if ( polyIdsOnPlane.getSize() )
			{
				BSPsOut[i].setAndDontIncrementRefCount(new hkcdPlanarSolid(planesCollection));

				// Get a V-Rep out of the polys on this plane 
				if ( polyIdsOnPlane.getSize() > 1 )
				{
					hkcdVertexGeometry* vRep = hkcdVertexGeometry::createFromPlanarGeometry(&geomIn, polyIdsOnPlane);
					hkArray<hkcdVertexGeometry::VPolygonId> vRepPolyIds;
					vRep->getAllPolygonIds(vRepPolyIds);
					hkcdPlanarGeometrySimplifier::buildSimplifyingSolidForPolygonGroup((hkcdPlanarGeometry*)(&geomIn), vRep, vRepPolyIds, BSPsOut[i]);
					vRep->removeReference();
				}
				else
				{
					// Only one poly: Build bsp direclty from boundary planes
					hkArray<PlaneId> edgePlaneIds;
					const PolygonId polyId			= polyIdsOnPlane[0];
					const Polygon& poly				= geomIn.getPolygon(polyId);
					const int numBPlanes			= geomIn.getNumBoundaryPlanes(polyId);
					edgePlaneIds.setSize(numBPlanes);
					for (int b = 0 ; b < numBPlanes ; b++)
					{
						edgePlaneIds[b]				= poly.getBoundaryPlaneId(b);
					}			
					BSPsOut[i]->buildConvex(edgePlaneIds.begin(), edgePlaneIds.getSize());
				}

			}
		}
	}

	//
	//	Classifies polygon w.r.t. faces of a boundary geometry

	static void HK_CALL classifyPolygons(hkcdPlanarGeometry& geomIn, const hkArray< hkRefPtr<hkcdPlanarSolid> >& BSPsIn, const hkArray<PolygonId>& polyIdsIn, hkArray<PolygonId>& polyIdsOnBoundaryOut)
	{
		// Classify the inputs polys by plane ids
		const int numBoundaryPlaneIds = BSPsIn.getSize();
		hkArray< hkArray<PolygonId> > inputPlaneIdToPolyIds;
		inputPlaneIdToPolyIds.setSize(numBoundaryPlaneIds);
		for (int i = polyIdsIn.getSize() - 1 ; i >=0 ; i--)
		{
			const PlaneId planeId	= geomIn.getPolygon(polyIdsIn[i]).getSupportPlaneId();
			const hkUint32 planeIdx	= planeId.value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);
			const hkUint32 flipped	= ( planeId.value() != planeIdx ) ? 1 : 0;
			const hkUint32 indx		= planeIdx*2 + flipped;
			inputPlaneIdToPolyIds[indx].pushBack(polyIdsIn[i]);
		}

		// For each plane, classify
		polyIdsOnBoundaryOut.setSize(0);
		for (int i = 0 ; i < numBoundaryPlaneIds ; i++)
		{
			const hkArray<PolygonId>& polyIdsOnPlane	= inputPlaneIdToPolyIds[i];
			const hkcdPlanarSolid* planeBSP				= BSPsIn[i];
			if ( polyIdsOnPlane.getSize() && planeBSP )
			{
				// classify the polygons for this plane
				hkArray<PolygonId> inPolyIds, outPolyIds, boundPolyIds;
				planeBSP->classifyPolygons(geomIn, polyIdsOnPlane, inPolyIds, boundPolyIds, outPolyIds);

				// add inside polys to ouput
				polyIdsOnBoundaryOut.append(inPolyIds);
			}
		}
	}

}

//
//	Constructor

hkcdPlanarCsgOperand::hkcdPlanarCsgOperand()
:	hkReferencedObject()
,	m_geometry(HK_NULL)
,	m_geomInfos(HK_NULL)
,	m_solid(HK_NULL)
,	m_regions(HK_NULL)
{}

//
//	GeomSource constructor

hkcdPlanarCsgOperand::GeomSource::GeomSource()
:	m_geometry(HK_NULL)
,	m_materialOffset(0)
,	m_numMaterialIds(0)
,	m_geomInfos(HK_NULL)
,	m_cutoutSolid(HK_NULL)
,	m_flipPolygons(false)
{}

//
//	GeomSource constructor from operand

hkcdPlanarCsgOperand::GeomSource::GeomSource(const hkcdPlanarCsgOperand& operand, const int nbMatIds)
:	m_geometry(operand.m_geometry)
,	m_materialOffset(0)
,	m_numMaterialIds(nbMatIds)
,	m_geomInfos(operand.m_geomInfos)
,	m_cutoutSolid(HK_NULL)
,	m_flipPolygons(false)
{}

//
//	Destructor

hkcdPlanarCsgOperand::~hkcdPlanarCsgOperand()
{
	m_regions = HK_NULL;
	m_solid = HK_NULL;
	m_geometry = HK_NULL;
}

//
//	Returns the convex cell tree corresponding to this solid planar geom. Build it if necessary.

hkcdConvexCellsTree3D* hkcdPlanarCsgOperand::getOrCreateConvexCellTree(bool withConnectivity, bool rebuildIfConnectivityDoesntMatch)
{
	if ( !m_regions || ( rebuildIfConnectivityDoesntMatch && (m_regions->hasManifoldCells() != withConnectivity)) )
	{
		// Build the tree now
		hkcdPlanarGeometry* geom = new hkcdPlanarGeometry(m_solid->accessPlanesCollection(), 0, m_solid->accessDebugger());
		m_regions.setAndDontIncrementRefCount(new hkcdConvexCellsTree3D(geom, withConnectivity));
		geom->removeReference();
		m_regions->buildFromSolid(m_solid);
	}

	return m_regions;
}

//
//	Sets a new planes collection. If the plane remapping table is non-null, the plane Ids on all nodes will be re-set as well (i.e. to match the plane Ids in the new collection)

void hkcdPlanarCsgOperand::setPlanesCollection(const PlanesCollection* newPlanes, const int* HK_RESTRICT planeRemapTable)
{
	// Make sure our geometry and solid share the same planes collection
	HK_ASSERT(0x18271f2c, !m_geometry || !m_solid || (m_geometry->getPlanesCollection() == m_solid->getPlanesCollection()));
	HK_ASSERT(0x1554c34c, !m_geometry || !m_regions || (m_geometry != m_regions->getGeometry()));
	HK_ASSERT(0x54070aa4, !m_solid || !m_regions || (m_solid->getPlanesCollection() == m_regions->getGeometry()->getPlanesCollection()));

	PlanesCollection* planesCol = const_cast<PlanesCollection*>(newPlanes);
	if ( m_geometry )
	{
		m_geometry->setPlanesCollection(planesCol, planeRemapTable);
	}

	if ( m_solid )
	{
		m_solid->setPlanesCollection(newPlanes, (int*)planeRemapTable);
	}

	if ( m_regions )
	{
		hkcdPlanarGeometry* regionsGeom = m_regions->accessGeometry();
		regionsGeom->setPlanesCollection(const_cast<PlanesCollection*>(newPlanes), planeRemapTable);
	}
}

//
//	Shift all plane ids of the operand elements

void hkcdPlanarCsgOperand::shiftPlaneIds(int offsetValue)
{
	// Make sure our geometry and solid share the same planes collection
	HK_ASSERT(0x18271f2c, !m_geometry || !m_solid || (m_geometry->getPlanesCollection() == m_solid->getPlanesCollection()));
	HK_ASSERT(0x1554c34c, !m_geometry || !m_regions || (m_geometry != m_regions->getGeometry()));
	HK_ASSERT(0x54070aa4, !m_solid || !m_regions || (m_solid->getPlanesCollection() == m_regions->getGeometry()->getPlanesCollection()));

	if ( m_geometry )
	{
		m_geometry->shiftPlaneIds(offsetValue);
	}

	if ( m_solid )
	{
		m_solid->shiftPlaneIds(offsetValue);
	}

	if ( m_regions )
	{
		hkcdPlanarGeometry* regionsGeom = m_regions->accessGeometry();
		regionsGeom->shiftPlaneIds(offsetValue);
	}
}

//
//	Retrieves the planes collection

const hkcdPlanarGeometryPlanesCollection* hkcdPlanarCsgOperand::getPlanesCollection() const
{
	HK_ASSERT(0x7d303860, !m_geometry || !m_solid || (m_geometry->getPlanesCollection() == m_solid->getPlanesCollection()));
	HK_ASSERT(0x59ea2ba2, !m_geometry || !m_regions || (m_geometry != m_regions->getGeometry()));
	HK_ASSERT(0x45126e8b, !m_solid || !m_regions || (m_solid->getPlanesCollection() == m_regions->getGeometry()->getPlanesCollection()));

	if ( m_solid )		{	return m_solid->getPlanesCollection();					}
	if ( m_geometry )	{	return m_geometry->getPlanesCollection();				}
	if ( m_regions )	{	return m_regions->getGeometry()->getPlanesCollection();	}

	return HK_NULL;
}

//
//	Simplifies this by rebuilding its tree from its boundaries

void hkcdPlanarCsgOperand::simplifyFromBoundaries()
{
	// Compute the boundaries
	hkcdPlanarGeometryPlanesCollection* pc = new hkcdPlanarGeometryPlanesCollection();
	hkcdPlanarGeometry boundaries(pc);
	pc->removeReference();

	m_solid->computeBoundary(getOrCreateConvexCellTree(true, true), boundaries, HK_NULL);

	// Collect all polygons
	hkArray<PolygonId> polygonIds;
	boundaries.getAllPolygons(polygonIds);

	// If no polygon, no boundary is present, don't change the tree
	if ( polygonIds.getSize() )
	{
		// Collect all unique planes used by the polygons
		hkArray<PlaneId> planeIds;
		boundaries.getAllPolygonsPlanes(polygonIds, planeIds, false);

		// Build the tree
		hkPseudoRandomGenerator rng(13);
		m_solid->clear();
		m_solid->buildTree(boundaries, rng, planeIds, polygonIds, false, HK_NULL);
	}

	// Invalidate the convex cell tree (no longer valid)
	m_regions = HK_NULL;
}

//
//	Build the planar geometry from its geometry sources

void hkcdPlanarCsgOperand::buildGeometryFromGeomSources(bool useStandardClassify, bool intersectCoplanarPolygons)
{
	HK_ASSERT(0xa95136ff, !getGeometry() && getSolid());
	typedef hkcdOperandImpl::SrcPoly SrcPoly;

	HK_TIMER_BEGIN("Build geometry", HK_NULL);

	// Create the output geometry
	hkcdPlanarGeometry tmpGeom(accessPlanesCollection());
	hkArray<PolygonId> currentPolyIds;

	// Add all the geometry sources
	hkArray<hkcdPlanarCsgOperand::GeomSource>& geomSources = accessGeometrySources();
	const int nbGeomSrc = geomSources.getSize();

	hkArray< hkRefPtr<hkcdPlanarSolid> > boundaryPlaneBSPs;	
	if ( !useStandardClassify )
	{
		// First, get the boundary polygons
		hkRefPtr<hkcdConvexCellsTree3D> cells = accessRegions();
		if ( !cells || !cells->hasManifoldCells() )
		{
			hkcdPlanarGeometry* geom = new hkcdPlanarGeometry(accessPlanesCollection(), 0);
			cells.setAndDontIncrementRefCount(new hkcdConvexCellsTree3D(geom, true));
			geom->removeReference();
			cells->buildFromSolid(m_solid);
		}
		hkArray<hkcdConvexCellsTree3D::CellId> solidCellIds;			
		cells->collectSolidCells(solidCellIds);
		hkArray<PolygonId> boundaryPolyIds;		
		hkcdPlanarGeometry boundaryGeometry(accessPlanesCollection());
		cells->extractBoundaryPolygonsFromCellIds(solidCellIds, boundaryGeometry, boundaryPolyIds);

		// for each non empty planes, build a BSP tree
		hkcdOperandImpl::buildBSPsFromGeometryPlanes(boundaryGeometry, boundaryPolyIds, getPlanesCollection(), cells, boundaryPlaneBSPs);
	}

	const hkUint32 nbOriginalPlanes = getPlanesCollection()->getNumPlanes();

	hkArray<PolygonId> danglingPolysIds;	
	hkArray<PolygonId> srcPolyIds, srcDanglingPolyIds, insidePolyIds, boundaryPolyIds, outsidePolyIds, newPolyIds;
	for (int s = 0 ; s < nbGeomSrc ; s++)
	{
		// Create a unique temporary instance of the ith geom source for the operand planes collection
		hkRefPtr<hkcdPlanarGeometry> geomSrc;
		geomSrc.setAndDontIncrementRefCount(new hkcdPlanarGeometry(accessPlanesCollection()));
		hkArray<int> remapTable;
		createGeometryFromGeometrySource(s, geomSrc, nbOriginalPlanes, srcPolyIds, srcDanglingPolyIds, remapTable);

		// Classify the (non dangling) polys of the source against the solid
		if ( useStandardClassify )
		{
			m_solid->classifyPolygons(*geomSrc, srcPolyIds, insidePolyIds, boundaryPolyIds, outsidePolyIds);
		}
		else
		{
			// This (more complicated) classify handles the inverted geom sources cases
			hkcdOperandImpl::classifyPolygons(*geomSrc, boundaryPlaneBSPs, srcPolyIds, boundaryPolyIds);
		}		

		// Add polys on the boundary to the mesh
		tmpGeom.appendGeometryPolygons(*geomSrc, boundaryPolyIds, false, newPolyIds);

		// Add dangling polys to current geometry
		if ( srcDanglingPolyIds.getSize() ) 
		{
			tmpGeom.appendGeometryPolygons(*geomSrc, srcDanglingPolyIds, false, insidePolyIds);
			danglingPolysIds.append(insidePolyIds);
		}

		// For all coplanar polys, cut them to avoid overlapping and add them to the retained poly list
		if ( intersectCoplanarPolygons )
		{
			// Sort new and current polys by support plane Ids
			hkArray<SrcPoly> newPolys;
			const int numNewPolys = newPolyIds.getSize();
			newPolys.setSize(numNewPolys);
			for (int k = 0; k < numNewPolys; k++)
			{
				const PolygonId polyId	= newPolyIds[k]; 
				const PlaneId supportId	= tmpGeom.getPolygon(polyId).getSupportPlaneId();
				newPolys[k].set(polyId, supportId, 0);
			}
			hkArray<SrcPoly> currPolys;
			const int numCurrPolys = currentPolyIds.getSize();
			currPolys.setSize(numCurrPolys);
			for (int k = 0; k < numCurrPolys; k++)
			{
				const PolygonId polyId	= currentPolyIds[k]; 
				const PlaneId supportId	= tmpGeom.getPolygon(polyId).getSupportPlaneId();
				currPolys[k].set(polyId, supportId, 0);
			}
			hkSort(newPolys.begin(), newPolys.getSize(), SrcPoly::less);
			hkSort(currPolys.begin(), currPolys.getSize(), SrcPoly::less);

			// Add only the polygon on the boundary
			int currInd = 0, newInd = 0;
			hkArray<PolygonId> polyIdsOnPlane;
			currentPolyIds.setSize(0);
			while ( newInd < numNewPolys || currInd < numCurrPolys )
			{
				// Choose plane id value as minimum
				hkUint32 currPlaneId;
				if ( (newInd == numNewPolys) || (currInd == numCurrPolys) ) 
				{
					currPlaneId = (newInd == numNewPolys) ? currPolys[currInd].m_uid : newPolys[newInd].m_uid;
				}
				else
				{
					currPlaneId = (currPolys[currInd].m_uid < newPolys[newInd].m_uid) ? currPolys[currInd].m_uid : newPolys[newInd].m_uid;
				}

				// Get all the polys on this plane
				int numSrcPoly = 0;
				polyIdsOnPlane.setSize(0);
				while ( currInd < numCurrPolys && currPolys[currInd].m_uid == currPlaneId )		{ polyIdsOnPlane.pushBack(currPolys[currInd].m_polyId); currInd++; numSrcPoly++; }
				while ( newInd < numNewPolys && newPolys[newInd].m_uid == currPlaneId )			{ polyIdsOnPlane.pushBack(newPolys[newInd].m_polyId); newInd++; }

				// Add them to the geometry
				hkcdPlanarGeometryBooleanUtil::addCoplanarPolygonsToMesh(tmpGeom, polyIdsOnPlane, numSrcPoly, currentPolyIds);
			}
		}
		else
		{
			// Add simply all the boundary polys, without any self intersection test
			currentPolyIds.append(newPolyIds);
		}

		// Meanwhile, in the dangling polygons world... We might need to cut them with the cutouts used during the booleans
		if ( danglingPolysIds.getSize() && geomSources[s].m_cutoutSolid )
		{
			// Create a temporary planar solid remapped to the current plane collection
			hkcdPlanarSolid cutoutClone(*geomSources[s].m_cutoutSolid);
			if ( cutoutClone.getPlanesCollection() != getPlanesCollection() ) cutoutClone.setPlanesCollection(getPlanesCollection(), remapTable.begin(), true);
			cutoutClone.classifyPolygons(tmpGeom, danglingPolysIds, insidePolyIds, boundaryPolyIds, outsidePolyIds);
			// Keep only the inside or outside polygons
			danglingPolysIds.setSize(0);
			if ( geomSources[s].m_flipPolygons ) danglingPolysIds.append(outsidePolyIds); else danglingPolysIds.append(insidePolyIds);
		}
	}

	// Create the new geometry with the final selected polygons
	hkRefPtr<hkcdPlanarGeometry> newGeom;
	newGeom.setAndDontIncrementRefCount(new hkcdPlanarGeometry(accessPlanesCollection()));	
	hkArray<PolygonId> tmpPolyIds;
	newGeom->appendGeometryPolygons(tmpGeom, currentPolyIds, false, tmpPolyIds);
	newGeom->appendGeometryPolygons(tmpGeom, danglingPolysIds, false, tmpPolyIds);

	// Set the geometry on the operand
	setGeometry(newGeom);

	// Save the dangling poly in the operand
	hkRefPtr<hkcdPlanarCsgOperand::GeomExtraInfos> geomInfo;
	geomInfo.setAndDontIncrementRefCount(new hkcdPlanarCsgOperand::GeomExtraInfos());
	geomInfo->m_danglingPolyIds.append(tmpPolyIds);
	hkSort(geomInfo->m_danglingPolyIds.begin(), geomInfo->m_danglingPolyIds.getSize());
	setGeomInfos(geomInfo);	

	// Remove the geometry sources
	removeGeometrySources();

	HK_TIMER_END();
}

//
//	Fast version of buildGeometryFromGeomSources when only 2 geometry sources are present

void hkcdPlanarCsgOperand::buildGeometryFrom2GeomSources(hkcdPlanarGeometryPrimitives::CollectionManager<hkcdPlanarGeometryPolygonCollection>* polysCollManager, hkcdPlanarGeometryPrimitives::CollectionManager<hkcdPlanarSolid::ArrayMgr>* arraysCollManager)
{
#if HK_CONFIG_MONITORS == HK_CONFIG_MONITORS_ENABLED
	hkMonitorStream& timerStream = hkMonitorStream::getInstance();
#endif	

	HK_TIMER_BEGIN2(timerStream, "BuildGeometry", HK_NULL);

	// Add all the geometry sources
	const hkUint32 nbOriginalPlanes = getPlanesCollection()->getNumPlanes();

	// Create the new geometry with the final selected polygons
	hkRefPtr<hkcdPlanarGeometry> newGeom;
	newGeom.setAndDontIncrementRefCount(new hkcdPlanarGeometry(accessPlanesCollection()));	
	hkArray<PolygonId> tmpPolyIds;

	//hkArray<PolygonId> danglingPolysIds;	
	hkArray<PolygonId> tmpSrcPolyIds, srcPolyIds, srcDanglingPolyIds, tmpCutoutPolyIds, cutoutPolyIds, cutoutDanglingPolyIds;
	hkArray<PolygonId> insidePolyIds, boundaryPolyIds, outsidePolyIds, newPolyIds;

	if ( m_solid->isValid() && !m_solid->isEmpty() )
	{		

		HK_TIMER_BEGIN2(timerStream, "SelectPlanesAndPolys", HK_NULL);

#if 0
		// Get the used planes of the solid		
		const int numPlanes			= accessPlanesCollection()->getNumPlanes();
		hkBitField usedPlaneIds;	usedPlaneIds.setSizeAndFill(0, numPlanes, 0);
		m_solid->collectUsedPlaneIds(usedPlaneIds);

		// Get the polygons of the original geometry, and remove the one that are using support planes not in the final solid
		hkRefPtr<hkcdPlanarGeometry> geomSrc;
		hkArray<int> remapTable;
		geomSrc.setAndDontIncrementRefCount(createGeometryFromGeometrySource(0, nbOriginalPlanes, tmpSrcPolyIds, srcDanglingPolyIds, remapTable));
		srcPolyIds.reserve(tmpSrcPolyIds.getSize());
		for (int pid = tmpSrcPolyIds.getSize() - 1 ; pid >= 0 ; pid--)
		{
			const Polygon& poly = geomSrc->getPolygon(tmpSrcPolyIds[pid]);
			if ( usedPlaneIds.get(poly.getSupportPlaneId().value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG)) )
			{
				srcPolyIds.pushBack(tmpSrcPolyIds[pid]);
			}
		}

		hkRefPtr<hkcdPlanarGeometry> geomCutout;
		geomCutout.setAndDontIncrementRefCount(createGeometryFromGeometrySource(1, nbOriginalPlanes, tmpCutoutPolyIds, cutoutDanglingPolyIds, remapTable));
		cutoutPolyIds.reserve(tmpCutoutPolyIds.getSize());
		for (int pid = tmpCutoutPolyIds.getSize() - 1 ; pid >= 0 ; pid--)
		{
			const Polygon& poly = geomCutout->getPolygon(tmpCutoutPolyIds[pid]);
			if ( usedPlaneIds.get(poly.getSupportPlaneId().value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG)) )
			{
				cutoutPolyIds.pushBack(tmpCutoutPolyIds[pid]);
			}
		}
#else
		hkArray<int> remapTable;
		hkRefPtr<hkcdPlanarGeometry> geomSrc;
		hkRefPtr<hkcdPlanarGeometryPolygonCollection> polysColl;
		polysCollManager->getUnusedCollection(polysColl);
		geomSrc.setAndDontIncrementRefCount(new hkcdPlanarGeometry(accessPlanesCollection(), polysColl));
		createGeometryFromGeometrySource(0, geomSrc, nbOriginalPlanes, srcPolyIds, srcDanglingPolyIds, remapTable);
		hkRefPtr<hkcdPlanarGeometry> geomCutout;
		polysCollManager->getUnusedCollection(polysColl);
		geomCutout.setAndDontIncrementRefCount(new hkcdPlanarGeometry(accessPlanesCollection(), polysColl));
		createGeometryFromGeometrySource(1, geomCutout, nbOriginalPlanes, cutoutPolyIds, cutoutDanglingPolyIds, remapTable);
		polysColl = HK_NULL;
#endif
		HK_TIMER_END();

		HK_TIMER_BEGIN2(timerStream, "Classify1", HK_NULL);

		// Get an array manager for the classify
		hkRefPtr<hkcdPlanarSolid::ArrayMgr> arrayMgr(HK_NULL);
		// Classify the poly of the 1st source against the cutout solid
		hkcdPlanarSolid* cutoutSolid = m_geomSources[1].m_cutoutSolid;
		HK_ASSERT(0x69a332ed, cutoutSolid->getPlanesCollection() == getPlanesCollection());
		if ( cutoutSolid->computeNumNodesWithLabel(hkcdPlanarSolid::NODE_TYPE_IN) == 1 )
		{
			// Convex cutout, use fast classify
			cutoutSolid->classifyInsideOrBoundaryPolygons(*geomSrc, srcPolyIds, insidePolyIds);
		}
		else
		{
			arraysCollManager->getUnusedCollection(arrayMgr);
			arrayMgr->reset();
			cutoutSolid->classifyPolygons(*geomSrc, srcPolyIds, insidePolyIds, boundaryPolyIds, outsidePolyIds, arrayMgr);
		}
		HK_TIMER_END();

		HK_TIMER_BEGIN2(timerStream, "AppendPolys", HK_NULL);
		// Add them to the result
		newGeom->appendGeometryPolygons(*geomSrc, insidePolyIds, false, tmpPolyIds);
		HK_TIMER_END();

		HK_TIMER_BEGIN2(timerStream, "Classify2", HK_NULL);		
		// Classify the poly of the 2nd source against the original solid
		//hkcdPlanarSolid* shapeSolid = m_geomSources[0].m_cutoutSolid;
		//HK_ASSERT(0x69a332ed, shapeSolid->getPlanesCollection() == getPlanesCollection());
		//shapeSolid->classifyPolygons(*geomCutout, cutoutPolyIds, insidePolyIds, boundaryPolyIds, outsidePolyIds);
		if ( m_solid->computeNumNodesWithLabel(hkcdPlanarSolid::NODE_TYPE_IN) == 1 )
		{
			// Convex cutout, use fast classify
			m_solid->classifyInsideOrBoundaryPolygons(*geomCutout, cutoutPolyIds, boundaryPolyIds);
		}
		else
		{
			if ( !arrayMgr ) arraysCollManager->getUnusedCollection(arrayMgr);
			arrayMgr->reset();
			m_solid->classifyPolygons(*geomCutout, cutoutPolyIds, insidePolyIds, boundaryPolyIds, outsidePolyIds, arrayMgr);
		}
		HK_TIMER_END();

		// Add them to the result
		HK_TIMER_BEGIN2(timerStream, "AppendPolys", HK_NULL);
		//newGeom->appendGeometryPolygons(*geomCutout, insidePolyIds, false, tmpPolyIds);
		newGeom->appendGeometryPolygons(*geomCutout, boundaryPolyIds, false, tmpPolyIds);
		HK_TIMER_END();
	}

	// Set the geometry on the operand
	setGeometry(newGeom);

	// Remove the geometry sources
	removeGeometrySources();

	HK_TIMER_END();
}

//
//	Copy the desired data from another operand

void hkcdPlanarCsgOperand::copyData(const hkcdPlanarCsgOperand& operandSrc, bool copySolid, bool copyRegions, bool copyGeometry)
{
	HK_TIMER_BEGIN("Clone operand", HK_NULL);

	HK_TIMER_BEGIN("Clone solid", HK_NULL);
	if ( copySolid && operandSrc.getSolid() )
	{
		hkcdPlanarSolid* clonedTree = new hkcdPlanarSolid(*operandSrc.getSolid());
		setSolid(clonedTree);			
		clonedTree->removeReference();
	}
	HK_TIMER_END();

	HK_TIMER_BEGIN("Clone regions", HK_NULL);
	if ( copyRegions && operandSrc.getRegions() )
	{
		hkcdConvexCellsTree3D* clonedRegions = new hkcdConvexCellsTree3D(*operandSrc.getRegions());
		setRegions(clonedRegions);			
		clonedRegions->removeReference();
	}
	HK_TIMER_END();

	HK_TIMER_BEGIN("Clone planar geom", HK_NULL);
	if ( copyGeometry && operandSrc.getGeometry() )
	{
		hkcdPlanarGeometry* clonedGeometry = new hkcdPlanarGeometry(*operandSrc.getGeometry());
		setGeometry(clonedGeometry);
		clonedGeometry->removeReference();
	}
	HK_TIMER_END();

	HK_TIMER_END();
}

/// Copy the desired data from another operand, using provided collection managers
void hkcdPlanarCsgOperand::copyData(const hkcdPlanarCsgOperand& operandSrc, hkcdPlanarGeometryPlanesCollection* dstPlaneCollection, PolyCollManager* polysCollManager, CellCollManager* cellsCollManager, bool copyRegions, bool copyGeometry)
{
	HK_TIMER_BEGIN("Copy operand", HK_NULL);

	hkRefPtr<hkcdPlanarGeometryPolygonCollection> polysColl;
	hkRefPtr<hkcdConvexCellsCollection> cellsColl;

	HK_TIMER_BEGIN("Copy solid", HK_NULL);
	if ( operandSrc.getSolid() )
	{
		hkcdPlanarSolid* solid = new hkcdPlanarSolid(*operandSrc.getSolid());
		solid->setPlanesCollection(dstPlaneCollection, HK_NULL);
		setSolid(solid);
		solid->removeReference();
	}
	HK_TIMER_END();

	HK_TIMER_BEGIN("Copy regions", HK_NULL);
	if ( copyRegions && operandSrc.getRegions() )
	{
		polysCollManager->getUnusedCollection(polysColl);
		polysColl->copy(operandSrc.getRegions()->getGeometry()->getPolygons());
		hkcdPlanarGeometry* cellsGeom			= new hkcdPlanarGeometry(dstPlaneCollection, polysColl);
		cellsCollManager->getUnusedCollection(cellsColl);
		cellsColl->copy(*operandSrc.getRegions()->getCells());
		hkcdConvexCellsTree3D* regions			= new hkcdConvexCellsTree3D(cellsColl, cellsGeom, *operandSrc.getRegions());
		cellsGeom->removeReference();
		setRegions(regions);	
		regions->removeReference();
	}
	HK_TIMER_END();

	HK_TIMER_BEGIN("Copy planar geom", HK_NULL);
	if ( copyGeometry && operandSrc.getGeometry() )
	{
		polysCollManager->getUnusedCollection(polysColl);
		polysColl->copy(operandSrc.getGeometry()->getPolygons());
		hkcdPlanarGeometry* geom				= new hkcdPlanarGeometry(dstPlaneCollection, polysColl);
		setGeometry(geom);
		geom->removeReference();
	}
	HK_TIMER_END();

	HK_TIMER_END();

}

//	Create an operand with the same data, but a duplicated plane collection
void HK_CALL hkcdPlanarCsgOperand::createOperandWithSharedDataAndClonedPlanes(const hkcdPlanarCsgOperand* operandSrc, hkRefPtr<hkcdPlanarCsgOperand>& operandDst)
{
	// Create a new plane collection
	const PlanesCollection* srcPlanes	= operandSrc->getPlanesCollection();
	PlanesCollection* dstPlanes			= new PlanesCollection(*srcPlanes);
	hkcdPlanarCsgOperand* dstOperand	= new hkcdPlanarCsgOperand();
	dstOperand->setPlanesCollection(dstPlanes, HK_NULL);

	// Copy the mesh (don't clone the polygon)
	hkcdPlanarGeometry* srcMesh			= (hkcdPlanarGeometry*)operandSrc->getGeometry();
	hkRefPtr<hkcdPlanarGeometry> dstMesh	= HK_NULL;
	if ( srcMesh )
	{
		dstMesh.setAndDontIncrementRefCount(new hkcdPlanarGeometry(dstPlanes, &(srcMesh->accessPolygons())));
		dstOperand->setGeometry(dstMesh);
	}
	// Copy the regions (don't clone the polygon of the cells)
	hkcdConvexCellsTree3D* srcRegions	= (hkcdConvexCellsTree3D*)operandSrc->getRegions();
	if ( srcRegions )
	{
		hkRefPtr<hkcdPlanarGeometry> newRegionMesh	= HK_NULL;
		newRegionMesh.setAndDontIncrementRefCount(new hkcdPlanarGeometry(dstPlanes, &(srcRegions->accessGeometry()->accessPolygons())));
		hkRefPtr<hkcdConvexCellsTree3D> dstRegions;
		dstRegions.setAndDontIncrementRefCount(new hkcdConvexCellsTree3D(srcRegions->accessCells(), newRegionMesh, *srcRegions));
		dstOperand->setRegions(dstRegions);
	}
	// Copy the solid (should not clone the nodes)
	hkcdPlanarSolid* srcSolid			= (hkcdPlanarSolid*)operandSrc->getSolid();
	if ( srcSolid )
	{
		hkRefPtr<hkcdPlanarSolid> dstSolid	= HK_NULL;
		dstSolid.setAndDontIncrementRefCount(new hkcdPlanarSolid(srcSolid->accessNodes(), srcSolid->getRootNodeId(), dstPlanes));
		dstSolid->setPlanesCollection(dstPlanes, HK_NULL);
		dstOperand->setSolid(dstSolid);
	}

	// Write output
	operandDst.setAndDontIncrementRefCount(dstOperand);
	dstPlanes->removeReference();
}

//
//	Create a unique temporary instance of the ith geom source matching the operand planes collection

void hkcdPlanarCsgOperand::createGeometryFromGeometrySource(int geomSourceId, hkcdPlanarGeometry* geomRes, hkUint32 maxPlaneIdValue, hkArray<PolygonId>& polyIds, hkArray<PolygonId>& danglingPolyIds, hkArray<int>& remapTable)
{
	// Allocate output geometry	
	GeomSource& geomSrc = m_geomSources[geomSourceId];
	danglingPolyIds.setSize(0);

	// Check if we have a proper geometry
	if ( !geomSrc.m_geometry )
	{
		// No valid geometry, it must be a geom source added only to cut dangling polys
		HK_ASSERT(0xd1a23e21, geomSrc.m_cutoutSolid);
		polyIds.setSize(0);
		hkcdPlanarGeometryPlanesCollection::computeMappingBetweenPlaneSets(geomSrc.m_cutoutSolid->accessPlanesCollection(), getPlanesCollection(), remapTable);
		return;
	}

	// Get source and dangling polys
	hkArray<PolygonId> srcPolyIds;
	geomSrc.m_geometry->getAllPolygons(srcPolyIds);
	if ( geomSrc.m_geomInfos && geomSrc.m_geomInfos->m_danglingPolyIds.getSize() )
	{
		// remove the dangling polygons from the source poly, they will be added later
		hkArray<PolygonId> clonedSrcPolyIds;
		clonedSrcPolyIds.append(srcPolyIds);
		hkSort(clonedSrcPolyIds.begin(), clonedSrcPolyIds.getSize());
		hkArray<PolygonId>& srcDanglingPolyIds = geomSrc.m_geomInfos->m_danglingPolyIds;
		srcPolyIds.setSize(hkAlgorithm::differenceOfSortedLists(clonedSrcPolyIds.begin(), clonedSrcPolyIds.getSize(), srcDanglingPolyIds.begin(), srcDanglingPolyIds.getSize(), srcPolyIds.begin()));
	}

	// Add them to the geom dest
	if ( geomSrc.m_geometry->accessPlanesCollection() != getPlanesCollection() )
	{
		// Plane collections do not match: get the mapping from geom source local geometry to common plane collection of the operand
		hkcdPlanarGeometryPlanesCollection::computeMappingBetweenPlaneSets(geomSrc.m_geometry->accessPlanesCollection(), getPlanesCollection(), remapTable);

		// Add the polygon to output geometry
		// First, the non dangling polys
		geomRes->appendGeometryPolygons(*geomSrc.m_geometry, remapTable.begin(), maxPlaneIdValue, srcPolyIds, polyIds, geomSrc.m_flipPolygons, geomSrc.m_materialOffset);

		// Then the dangling polys
		if ( geomSrc.m_geomInfos && geomSrc.m_geomInfos->m_danglingPolyIds.getSize() )
		{
			geomRes->appendGeometryPolygons(
				*geomSrc.m_geometry, remapTable.begin(), 0, 
				geomSrc.m_geomInfos->m_danglingPolyIds, danglingPolyIds, 
				geomSrc.m_flipPolygons, geomSrc.m_materialOffset);
		}
	}
	else
	{
		// Plane collections match: Simply add the polygons without remapping
		geomRes->appendGeometryPolygons(*geomSrc.m_geometry, srcPolyIds, geomSrc.m_flipPolygons, polyIds, geomSrc.m_materialOffset);
		if ( geomSrc.m_geomInfos && geomSrc.m_geomInfos->m_danglingPolyIds.getSize() )
		{
			geomRes->appendGeometryPolygons(*geomSrc.m_geometry, geomSrc.m_geomInfos->m_danglingPolyIds, geomSrc.m_flipPolygons, danglingPolyIds, geomSrc.m_materialOffset);
		}
	}
}

//
//	Collects a bit-field of plane Ids used by the operand

void hkcdPlanarCsgOperand::collectUsedPlaneIds(hkBitField& usedPlaneIdsOut) const
{
	// Collect the planes used by each planar entity
	if ( m_geometry )
	{
		m_geometry->collectUsedPlaneIds(usedPlaneIdsOut);
	}
	
	if ( m_solid )
	{
		m_solid->collectUsedPlaneIds(usedPlaneIdsOut);
	}

	if ( m_regions )
	{
		const hkcdPlanarGeometry* regionsGeom = m_regions->getGeometry();
		regionsGeom->collectUsedPlaneIds(usedPlaneIdsOut);
	}
}

//
//	Removes all planes not used by the entities

void hkcdPlanarCsgOperand::removeUnusedPlanes()
{
	// Get the plane collection and allocate the bit-field
	PlanesCollection* newPlanes		= accessPlanesCollection();
	const int numPlanes				= newPlanes->getNumPlanes();
	hkBitField usedPlaneIds;	usedPlaneIds.setSizeAndFill(0, numPlanes, 0);

	// Gather all the planes in use
	collectUsedPlaneIds(usedPlaneIds);

	// Clone the planes collection so we can perform the remap
	PlanesCollection oldPlanes(*newPlanes);
	newPlanes->addReference();
	setPlanesCollection(&oldPlanes, HK_NULL);

	// Remove the unused planes from the collection
	hkArray<int> remappedPlanes;
	usedPlaneIds.setNot(usedPlaneIds);
	newPlanes->removePlanes(usedPlaneIds, &remappedPlanes);

	// Set the new planes collection
	setPlanesCollection(newPlanes, remappedPlanes.begin());
	newPlanes->removeReference();
}

//
//	Adds a geometry source to the operand

hkcdPlanarCsgOperand::GeomSource* hkcdPlanarCsgOperand::addGeometrySource(const GeomSource& geomSource)
{
	if ( geomSource.m_geometry && geomSource.m_numMaterialIds > 0 )
	{
		// Check that the source is not already present
		for (int s = m_geomSources.getSize() - 1 ; s >= 0 ; s--)
		{
			const GeomSource& gs = m_geomSources[s];
			if ( gs.m_geometry.val() == geomSource.m_geometry.val() && gs.m_flipPolygons == geomSource.m_flipPolygons )
			{
				return HK_NULL;
			}
		}

		m_geomSources.pushBack(geomSource);

		// update mat offset
		GeomSource& newGeom = m_geomSources[m_geomSources.getSize() - 1];
		if ( m_geomSources.getSize() > 1 )
		{
			newGeom.m_materialOffset = m_geomSources[m_geomSources.getSize() - 2].m_materialOffset + m_geomSources[m_geomSources.getSize() - 2].m_numMaterialIds;
		}
		else
		{
			newGeom.m_materialOffset = 0;
		}
		return &newGeom;
	}
	else if( geomSource.m_cutoutSolid )
	{
		// Add an empty source, containing only the cutout solid
		m_geomSources.pushBack(geomSource);

		// update mat offset
		GeomSource& newGeom			= m_geomSources[m_geomSources.getSize() - 1];
		newGeom.m_numMaterialIds	= 0;		// this source has no geometry, hence no material id...
		if ( m_geomSources.getSize() > 1 )
		{
			newGeom.m_materialOffset = m_geomSources[m_geomSources.getSize() - 2].m_materialOffset + m_geomSources[m_geomSources.getSize() - 2].m_numMaterialIds;
		}
		else
		{
			newGeom.m_materialOffset = 0;
		}
	}

	return HK_NULL;
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
