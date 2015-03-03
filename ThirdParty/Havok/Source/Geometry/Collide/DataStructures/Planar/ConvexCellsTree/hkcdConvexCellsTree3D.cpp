/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Geometry/Collide/hkcdCollide.h>
#include <Geometry/Collide/DataStructures/Planar/ConvexCellsTree/hkcdConvexCellsTree3D.h>
#include <Geometry/Collide/DataStructures/Planar/Solid/hkcdPlanarSolid.h>

#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/Base/Algorithm/Collide/1AxisSweep/hk1AxisSweep.h>
#include <Common/GeometryUtilities/Mesh/Utils/FindUniquePositionsUtil/hkFindUniquePositionsUtil.h>

//
//	Constructor 

hkcdConvexCellsTree3D::hkcdConvexCellsTree3D(hkcdPlanarGeometry* geom, bool withConnectivity)
:	hkcdConvexCellsTree<hkcdConvexCellsCollection::Cell, hkcdConvexCellsCollection::CellId, hkcdConvexCellsCollection>(geom)
,	m_buildCellConnectivity(withConnectivity)
{
	for (int k = PlanesCollection::NUM_BOUNDS - 1; k >= 0; k--)
	{
		m_boundingPolys[k]	= PolygonId::invalid();
	}
}

//
//	Constructor from data pointers

hkcdConvexCellsTree3D::hkcdConvexCellsTree3D(hkcdConvexCellsCollection* cells, hkcdPlanarGeometry* geom, const hkcdConvexCellsTree3D& other)
:	hkcdConvexCellsTree<hkcdConvexCellsCollection::Cell, hkcdConvexCellsCollection::CellId, hkcdConvexCellsCollection>(cells, geom)
,	m_buildCellConnectivity(other.m_buildCellConnectivity)
{
	// Bounding planes and polys
	for (int n = 0 ; n < PlanesCollection::NUM_BOUNDS ; n++)
	{
		m_boundingPolys[n] = other.m_boundingPolys[n];
	}
}

//
//	Copy constructor

hkcdConvexCellsTree3D::hkcdConvexCellsTree3D(const hkcdConvexCellsTree3D& other)
:	hkcdConvexCellsTree<hkcdConvexCellsCollection::Cell, hkcdConvexCellsCollection::CellId, hkcdConvexCellsCollection>(other)
,	m_buildCellConnectivity(other.m_buildCellConnectivity)
{
	// Bounding planes and polys
	for (int n = 0 ; n < PlanesCollection::NUM_BOUNDS ; n++)
	{
		m_boundingPolys[n] = other.m_boundingPolys[n];
	}
}

//
//	Build a convex cell tree out of a solid bsp tree

void hkcdConvexCellsTree3D::buildFromSolid(hkcdPlanarSolid* solid)
{
	// Set-up boundary cell
	{
		CellId boundaryCellId = createBoundaryCell();
		solid->accessNode(solid->getRootNodeId()).m_data = boundaryCellId.value();
		accessCell(boundaryCellId).setUserData(solid->getRootNodeId().value());
	}

	// Initialize nodes stack
	hkArray<NodeId> nodeStack;
	nodeStack.pushBack(solid->getRootNodeId());

	// Split boundary cell with each plane in the BSP tree
	while ( nodeStack.getSize() )
	{
		// Pop node from the stack
		const NodeId nodeId = nodeStack[0];
		const Node& node = solid->getNode(nodeId);
		nodeStack.removeAt(0);

		// Get the cell associated with this node
		const CellId nodeCellId(node.m_data);

		// Check if the node is internal
		if ( node.m_typeAndFlags == hkcdPlanarSolid::NODE_TYPE_INTERNAL )
		{
			HK_ASSERT(0x73df24d1, node.m_left.isValid());
			HK_ASSERT(0x43141527, node.m_right.isValid());

			CellId inCell	= CellId::invalid();
			CellId outCell	= CellId::invalid();

			if ( nodeCellId.isValid() )
			{
				// Split it with the plane in the BSP node					
				splitCell(solid, nodeCellId, node.m_planeId, inCell, outCell);
			}

			// Recurse on left
			{
				Node& leftChild		= solid->accessNode(node.m_left);
				leftChild.m_data	= inCell.valueUnchecked();
				if ( inCell.isValid() )
				{
					accessCell(inCell).setUserData(node.m_left.value());
				}
				nodeStack.pushBack(node.m_left);
			}

			// Recurse on right
			{					
				Node& rightChild	= solid->accessNode(node.m_right);	
				rightChild.m_data	= outCell.valueUnchecked();
				if ( outCell.isValid() )
				{
					accessCell(outCell).setUserData(node.m_right.value());
				}
				nodeStack.pushBack(node.m_right);
			}
		}
		else if ( nodeCellId.isValid() )
		{
			Cell& cell =  accessCell(nodeCellId);
			cell.setLeaf(true);
			cell.setLabel( ( node.m_typeAndFlags == hkcdPlanarSolid::NODE_TYPE_IN ) ? hkcdConvexCellsCollection::CELL_SOLID : hkcdConvexCellsCollection::CELL_EMPTY);
		}
	}
}

//
//	Destructor

hkcdConvexCellsTree3D::~hkcdConvexCellsTree3D()
{
	m_mesh = HK_NULL;
}

//
//	Creates a box cell that encloses the entire "known" space

hkcdConvexCellsTree3D::CellId hkcdConvexCellsTree3D::createBoundaryCell()
{
	// Allocate a cell
	const CellId cellId = m_cells->allocCell(PlanesCollection::NUM_BOUNDS);
	Cell& boundsCell = accessCell(cellId);

	// Create 6 polygons, one for each axis
	{
		const Bounds polyBounds[PlanesCollection::NUM_BOUNDS][4] = 
		{
			{ PlanesCollection::BOUND_NEG_Y, PlanesCollection::BOUND_NEG_Z, PlanesCollection::BOUND_POS_Y, PlanesCollection::BOUND_POS_Z }, { PlanesCollection::BOUND_NEG_Y, PlanesCollection::BOUND_POS_Z, PlanesCollection::BOUND_POS_Y, PlanesCollection::BOUND_NEG_Z },
			{ PlanesCollection::BOUND_NEG_Z, PlanesCollection::BOUND_NEG_X, PlanesCollection::BOUND_POS_Z, PlanesCollection::BOUND_POS_X }, { PlanesCollection::BOUND_NEG_Z, PlanesCollection::BOUND_POS_X, PlanesCollection::BOUND_POS_Z, PlanesCollection::BOUND_NEG_X },
			{ PlanesCollection::BOUND_NEG_Y, PlanesCollection::BOUND_POS_X, PlanesCollection::BOUND_POS_Y, PlanesCollection::BOUND_NEG_X }, { PlanesCollection::BOUND_NEG_Y, PlanesCollection::BOUND_NEG_X, PlanesCollection::BOUND_POS_Y, PlanesCollection::BOUND_POS_X },
		};

		for (int k = 0; k < PlanesCollection::NUM_BOUNDS; k++)
		{
			m_boundingPolys[k] = m_mesh->addPolygon(PlaneId(k), 0, 4);
			boundsCell.setBoundaryPolygonId(k, m_boundingPolys[k]);

			Polygon& poly = m_mesh->accessPolygon(m_boundingPolys[k]);

			if ( m_buildCellConnectivity )	
			{
				poly.setNegCellId(CellId::invalid().valueUnchecked());		// there is nothing outside the box
				poly.setPosCellId(cellId.value());
				poly.setMaterialId(POLY_SURFACE_INVALID);
			}

			for (int bi = 0; bi < 4; bi++)
			{
				poly.setBoundaryPlaneId(bi, PlaneId(polyBounds[k][bi]));
			}
		}
	}

	// Return the Id of the boundary cell
	return cellId;
}

//
//	Splits the given cell by the given plane. Updates all adjacency information

void hkcdConvexCellsTree3D::splitCell(hkcdPlanarSolid* solid, CellId cellId, PlaneId splitPlaneId, CellId& insideCellIdOut, CellId& outsideCellIdOut)
{
	// Split each polygon of the original cell
	const int numPolygons = m_cells->getNumBoundaryPolygons(cellId);
	accessCell(cellId).setLeaf(false);

	// Allocate temporary buffers
	hkInplaceArray<PolygonId, 512> polyIdsIn;	polyIdsIn.setSize(numPolygons + 1);
	hkInplaceArray<PolygonId, 512> polyIdsOut;	polyIdsOut.setSize(numPolygons + 1);
	hkInplaceArray<PlaneId, 512> planeIdsIn;	planeIdsIn.setSize(numPolygons + 1);
	hkInplaceArray<PlaneId, 512> planeIdsOut;	planeIdsOut.setSize(numPolygons + 1);
	int numPolysIn	= 0;
	int numPolysOut	= 0;

	for (int pi = 0; pi < numPolygons; pi++)
	{
		// Get a polygon on the cell boundary and test against splitting plane
		const Cell& originalCell = getCell(cellId);
		const PolygonId originalPolyId	= originalCell.getBoundaryPolygonId(pi);
		PlaneId supportId				= m_mesh->getPolygon(originalPolyId).getSupportPlaneId();

		// While building connectivity, the flip information is deduced from the polygon neighboring data
		if ( m_buildCellConnectivity )
		{
			const Polygon& originalPolygon = m_mesh->getPolygon(originalPolyId);
			supportId = ( originalPolygon.getPosCellId() == cellId.value() ) ? supportId : hkcdPlanarGeometryPrimitives::getOppositePlaneId(supportId);
		}

		//const hkcdPlanarGeometryPredicates::Orientation ori = m_mesh->classify(originalPolyId, splitPlaneId, orientationCache, cellId.value() == 99);
		const hkcdPlanarGeometryPredicates::Orientation ori = m_mesh->classify(originalPolyId, splitPlaneId);

		planeIdsIn[numPolysIn]		= supportId;
		planeIdsOut[numPolysOut]	= supportId;

		switch ( ori )
		{
		case hkcdPlanarGeometryPredicates::IN_FRONT_OF:	{	HK_ASSERT(0x14b5441b, numPolysOut < numPolygons);	polyIdsOut[numPolysOut++]	= originalPolyId;	}	break;
		case hkcdPlanarGeometryPredicates::BEHIND:		{	HK_ASSERT(0x5d51195, numPolysIn < numPolygons);		polyIdsIn[numPolysIn++]		= originalPolyId;	}	break;
		case hkcdPlanarGeometryPredicates::INTERSECT:
			{
				PolygonId insidePolyId, outsidePolyId;
				m_mesh->split(originalPolyId, splitPlaneId, insidePolyId, outsidePolyId);
				HK_ASSERT(0x7c933140, numPolysIn < numPolygons);	polyIdsIn[numPolysIn++]		= insidePolyId;
				HK_ASSERT(0x3d638fc9, numPolysOut < numPolygons);	polyIdsOut[numPolysOut++]	= outsidePolyId;
				if ( m_buildCellConnectivity )
				{
					updateConnectivity(solid, cellId, originalPolyId, insidePolyId, outsidePolyId, splitPlaneId);
				}			
			}
			break;

		default:	// Polygon is on plane. Do nothing here, the polygon on the splitting plane will be added last!
			break;
		}
	}

	// Add the cells
	insideCellIdOut			= CellId::invalid();
	outsideCellIdOut		= CellId::invalid();
	
	if ( numPolysIn && numPolysOut )
	{
		// Create spliting polygon
		const int numPlaneIds	= hkMath::min2(numPolysIn, numPolysOut);
		PolygonId cappingPolyId = addClosingCap((numPolysOut > numPlaneIds) ? planeIdsIn.begin() : planeIdsOut.begin(), numPlaneIds, splitPlaneId);
		if ( m_buildCellConnectivity )
		{
			Polygon& capingPoly		= m_mesh->accessPolygon(cappingPolyId);			
			capingPoly.setMaterialId(POLY_SURFACE_OPEN);
			capingPoly.setNegCellId(cellId.value());
			capingPoly.setPosCellId(cellId.value());
		}

		// Create inside cell
		{
			polyIdsIn[numPolysIn++] = cappingPolyId;

			insideCellIdOut		= m_cells->allocCell(numPolysIn);
			Cell& insideCell	= accessCell(insideCellIdOut);
			for (int k = numPolysIn - 1; k >= 0; k--)
			{
				insideCell.setBoundaryPolygonId(k, polyIdsIn[k]);
				// Optionnally, replace all the neighboring info in the polygons
				if ( m_buildCellConnectivity )
				{
					Polygon& poly		= m_mesh->accessPolygon(polyIdsIn[k]);
					HK_ASSERT(0x3d6e5a6f, poly.getNegCellId() == cellId.value() || poly.getPosCellId() == cellId.value());
					poly.setPosCellId( ( poly.getPosCellId() == cellId.value() )			? insideCellIdOut.value() : poly.getPosCellId() );
					poly.setNegCellId( ( poly.getPosCellId() != insideCellIdOut.value() )	? insideCellIdOut.value() : poly.getNegCellId() );
				}
			}
		}

		// Create outside cell
		{			
			if ( m_buildCellConnectivity )
			{
				// While building manifold cells, all the polygon are shared between neighboring cells
				polyIdsOut[numPolysOut++] = cappingPolyId;
			}
			else
			{
				// We've already computed the polygon before, flip it and reuse
				const Polygon& cappingPoly	= m_mesh->getPolygon(cappingPolyId);
				const hkUint32 mtlId		= cappingPoly.getMaterialId();
				const PlaneId invSupportId	= hkcdPlanarGeometryPrimitives::getOppositePlaneId(cappingPoly.getSupportPlaneId());
				const int numBounds			= m_mesh->getNumBoundaryPlanes(cappingPolyId);
				const PolygonId oppCapId	= m_mesh->addPolygon(invSupportId, mtlId, numBounds);

				// Set the bounding planes
				{
					const Polygon& capPoly		= m_mesh->getPolygon(cappingPolyId);
					Polygon& invCappingPoly		= m_mesh->accessPolygon(oppCapId);

					for (int bi = numBounds - 1; bi >= 0; bi--)
					{
						const PlaneId boundId = capPoly.getBoundaryPlaneId(numBounds - 1 - bi);
						invCappingPoly.setBoundaryPlaneId(bi, boundId);
					}
				}

				polyIdsOut[numPolysOut++]	= oppCapId;
			}
			outsideCellIdOut			= m_cells->allocCell(numPolysOut);
			Cell& outsideCell			= accessCell(outsideCellIdOut);

			for (int k = numPolysOut - 1; k >= 0; k--)
			{
				outsideCell.setBoundaryPolygonId(k, polyIdsOut[k]);
				// Optionnally, replace all the neighboring info in the polygons
				if ( m_buildCellConnectivity )
				{
					Polygon& poly		= m_mesh->accessPolygon(polyIdsOut[k]);
					HK_ASSERT(0x3d6e5a6f, poly.getNegCellId() == cellId.value() || poly.getPosCellId() == cellId.value());
					poly.setNegCellId( ( poly.getNegCellId() == cellId.value() )			? outsideCellIdOut.value() : poly.getNegCellId() );
					poly.setPosCellId( ( poly.getNegCellId() != outsideCellIdOut.value() )	? outsideCellIdOut.value() : poly.getPosCellId() );
				}
			}
		}
	}
	else 
	{
		//We need to maintain a bijection between nodes and cells. Copy the cell and update neighbors
		CellId newCellId = CellId::invalid();
		if ( m_buildCellConnectivity )
		{
			const int nbSrcPolys	= m_cells->getNumBoundaryPolygons(cellId);
			newCellId				= m_cells->allocCell(nbSrcPolys);
			Cell& newCell			= accessCell(newCellId);
			const Cell& srcCell		= getCell(cellId);
			for (int i = 0 ; i < nbSrcPolys ; i++)
			{
				PolygonId polyId = srcCell.getBoundaryPolygonId(i);
				newCell.setBoundaryPolygonId(i, polyId);
				Polygon& poly		= m_mesh->accessPolygon(polyId);
				// This will fail for the two new polys. These are initialized at the end of this method
				poly.setNegCellId( ( poly.getNegCellId() == cellId.value() )	? newCellId.value() : poly.getNegCellId() );
				poly.setPosCellId( ( poly.getPosCellId() == cellId.value() )	? newCellId.value() : poly.getPosCellId() );
			}
		}

		if ( numPolysIn )
		{
			insideCellIdOut = ( newCellId.isValid() ) ? newCellId : cellId;
		}
		else
		{
			HK_ASSERT(0x7cd305d6, !numPolysIn && numPolysOut);
			outsideCellIdOut = ( newCellId.isValid() ) ? newCellId : cellId;
		}
	}
}

//
//	Updates the connectivity information for a given polygon after a split

void hkcdConvexCellsTree3D::updateConnectivity(hkcdPlanarSolid* solid, CellId cellId, PolygonId splitPolygonId, PolygonId insidePolyId, PolygonId outsidePolyId, PlaneId splitPlaneId)
{
	HK_ASSERT(0xb37a653d, solid);

	// Update the neighboring cells to maintain manifoldness
	// NOTE: the parent/children relationships are lost after this step !!!
	Polygon& splitPolygon		= m_mesh->accessPolygon(splitPolygonId);
	// Get the neighboring cell
	const CellId n1Id			= CellId(splitPolygon.getNegCellId());
	const CellId n2Id			= CellId(splitPolygon.getPosCellId());
	HK_ASSERT(0x3d6e5a6f, n1Id == cellId || n2Id == cellId);
	CellId nId					= ( n1Id == cellId ) ? n2Id : n1Id;
	if ( nId.isValid() )
	{
		// Rebuild the cell since we have no way to update it
		const Cell& nCell		= accessCell(nId);
		const int nbOldPloys	= m_cells->getNumBoundaryPolygons(nId);
		hkArray<PolygonId> polyIds;		polyIds.reserve(nbOldPloys + 1);
		for (int i = 0 ; i < nbOldPloys ; i++)
		{
			const PolygonId pId = nCell.getBoundaryPolygonId(i);
			if ( pId != splitPolygonId )
			{
				polyIds.pushBack(pId);
			}
		}
		polyIds.pushBack(insidePolyId);
		polyIds.pushBack(outsidePolyId);
		HK_ASSERT(0x12af45de, polyIds.getSize() == nbOldPloys + 1);

		const CellId newCellId	= m_cells->allocCell(polyIds.getSize());
		Cell& newCell			= accessCell(newCellId);
		for (int k = polyIds.getSize() - 1; k >= 0; k--)
		{
			newCell.setBoundaryPolygonId(k, polyIds[k]);
			Polygon& poly		= m_mesh->accessPolygon(polyIds[k]);
			// This will fail for the two new polys. These are initialized at the end of this method
			poly.setNegCellId( ( poly.getNegCellId() == nId.value() )	? newCellId.value() : poly.getNegCellId() );
			poly.setPosCellId( ( poly.getPosCellId() == nId.value() )	? newCellId.value() : poly.getPosCellId() );
		}
	
		// Update the link bsp node <-> cell ids for the replaced cell.
		NodeId rNodeId		= NodeId(getCell(nId).getUserData());
		Node& rNode			= solid->accessNode(rNodeId);
		rNode.m_data = newCellId.value();
		accessCell(newCellId).setUserData(rNodeId.value());

		// Reparent and copy properties
		const Cell& oldCell	= getCell(nId);
		if ( oldCell.isSolid() ) newCell.setSolid();
		if ( oldCell.isEmpty() ) newCell.setEmpty();
		newCell.setLeaf(oldCell.isLeaf());

		// Delete the old cell
		m_cells->freeCell(nId);	
		nId					= newCellId;
	}

	// Update neighboring info
	Polygon& inPoly		= m_mesh->accessPolygon(insidePolyId);
	Polygon& outPoly	= m_mesh->accessPolygon(outsidePolyId);
	const bool neighborIsOnPositiveSide = ( n1Id == cellId );
	inPoly.setNegCellId(( neighborIsOnPositiveSide ) ? cellId.value() : nId.valueUnchecked());
	inPoly.setPosCellId(( neighborIsOnPositiveSide ) ? nId.valueUnchecked() : cellId.value());
	outPoly.setNegCellId(( neighborIsOnPositiveSide ) ? cellId.value() : nId.valueUnchecked());
	outPoly.setPosCellId(( neighborIsOnPositiveSide ) ? nId.valueUnchecked() : cellId.value());
	splitPolygon.setMaterialId(POLY_SURFACE_INVALID);
}

//
//	Removes all cells not marked as cellLabel

void hkcdConvexCellsTree3D::removeCellsOfType(CellLabel cellLabel)
{
	// Create two arrays, one with the polygon Ids we want to keep and the other with the polygon Ids we want to delete!
	hkArray<PolygonId> polysToKeep, polysToDelete;
	for (CellId cellId = m_cells->getFirstCellId(); cellId.isValid(); cellId = m_cells->getNextCellId(cellId))
	{
		const Cell& cell			= getCell(cellId);
		hkArray<PolygonId>& polys	= ( cell.getLabel() == cellLabel ) ? polysToDelete : polysToKeep;
		const int numBounds			= m_cells->getNumBoundaryPolygons(cellId);
		PolygonId* ptr				= polys.expandBy(numBounds);

		for (int k = numBounds - 1; k >= 0; k--)
		{
			ptr[k] = cell.getBoundaryPolygonId(k);
		}
	}

	// Sort both arrays
	int numPolysToKeep		= polysToKeep.getSize();		hkSort(reinterpret_cast<PolygonId::Type*>(polysToKeep.begin()), numPolysToKeep);
	int numPolysToDelete	= polysToDelete.getSize();		hkSort(reinterpret_cast<PolygonId::Type*>(polysToDelete.begin()), numPolysToDelete);

	// Remove duplicates
	numPolysToKeep		= hkAlgorithm::removeDuplicatesFromSortedList(polysToKeep.begin(), numPolysToKeep);			polysToKeep.setSize(numPolysToKeep);
	numPolysToDelete	= hkAlgorithm::removeDuplicatesFromSortedList(polysToDelete.begin(), numPolysToDelete);		polysToDelete.setSize(numPolysToDelete);

	// Remove polysToKeep from polysToDelete
	for (int iKeep = numPolysToKeep - 1,  iDelete = numPolysToDelete - 1; (iKeep >= 0) && (iDelete >= 0); )
	{
		const PolygonId polyA = polysToKeep[iKeep];
		const PolygonId polyB = polysToDelete[iDelete];
		if ( polyA > polyB )
		{
			iKeep--;	// Poly A is not marked for deletion
		}
		else if ( polyA == polyB )
		{
			polysToDelete.removeAt(iDelete);
			iKeep--;	iDelete--;	// Poly B is marked for deletion but also used
		}
		else
		{
			iDelete--;	// Poly B is marked for deletion and not used
		}
	}

	// Delete all polys
	m_mesh->removePolygons(polysToDelete);

	// We can finally delete all the cells!
	for (CellId cellId = m_cells->getLastCellId(); cellId.isValid();)
	{
		const Cell& cell		= getCell(cellId);
		const CellId prevCellId = m_cells->getPrevCellId(cellId);

		if ( cell.getLabel() == cellLabel )
		{
			m_cells->freeCell(cellId);
		}
		cellId = prevCellId;
	}
}

//
//	Utility functions

namespace hkcdConvexCellsTreeImpl
{
	//
	//	Compares 2 hkIntVectors by x, y, z in that order

	HK_FORCE_INLINE hkBool32 HK_CALL vectorLess(hkIntVectorParameter vA, hkIntVectorParameter vB)
	{
		const hkVector4Comparison cmpL = vA.compareLessThanS32(vB);
		const hkVector4Comparison cmpE = vA.compareEqualS32(vB);

		const int code = ((cmpL.getMask() << 2) & 0x1C) | (cmpE.getMask() & 3);
		return (0xFAF8FAF0 >> code) & 1;
	}

	HK_FORCE_INLINE hkBool32 HK_CALL vectorEq(hkIntVectorParameter vA, hkIntVectorParameter vB)
	{
		return vA.compareEqualS32(vB).horizontalAnd<3>().anyIsSet();
	}
}

//
//	Computes the approximate position of the vertices of the given convex cell

void hkcdConvexCellsTree3D::computeCellVertices(CellId cellId, hkArray<hkVector4>& verticesPos, bool useFastConversion) const
{
	const Cell& cell	= getCell(cellId);
	const int numFaces	= m_cells->getNumBoundaryPolygons(cellId);

	// Get polygons and gather all vertices, we'll have a lot of them shared.
	// V + F - E = 2
	hkInplaceArray<hkIntVector, 128> vtxIds;
	for (int fi = 0; fi < numFaces; fi++)
	{
		// Get the face
		const PolygonId faceId		= cell.getBoundaryPolygonId(fi);
		const Polygon& face			= m_mesh->getPolygon(faceId);
		const int numEdges			= m_mesh->getNumBoundaryPlanes(faceId);
		const hkUint32 supportId	= face.getSupportPlaneId().value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);

		// Add the vertices
		hkUint32 prevPlaneId	= face.getBoundaryPlaneId(numEdges - 1).value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);
		hkIntVector* vbPtr		= vtxIds.expandBy(numEdges);
		for (int crt = 0; crt < numEdges; crt++)
		{
			const hkUint32 crtPlaneId	= face.getBoundaryPlaneId(crt).value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);
			hkIntVector v;				v.set(supportId, prevPlaneId, crtPlaneId, 0);
			vbPtr[crt].setSortS32<3, HK_SORT_ASCENDING>(v);
			prevPlaneId = crtPlaneId;
		}
	}

	// Sort the vertices by their ids and remove duplicates
	hkSort(vtxIds.begin(), vtxIds.getSize(), hkcdConvexCellsTreeImpl::vectorLess);
	const int numVerts = hkAlgorithm::removeDuplicatesFromSortedList(vtxIds.begin(), vtxIds.getSize(), hkcdConvexCellsTreeImpl::vectorEq);
	vtxIds.setSize(numVerts);

	// Approximate the vertices' position
	verticesPos.setSize(numVerts);
	Plane planes[3];
	const PlanesCollection* planesCol = m_mesh->getPlanesCollection();

	hkDouble64 doubleBuff[4];	hkFloat32 realBuff[4];
	hkVector4 offset					= planesCol->getPositionOffset();
	hkVector4d offsetD;					offset.store<4, HK_IO_NATIVE_ALIGNED>(doubleBuff);	offsetD.load<4, HK_IO_NATIVE_ALIGNED>(doubleBuff);
	hkSimdReal scale					= planesCol->getPositionScale();
	hkSimdDouble64 scaleD;				scaleD.setFromFloat(scale.getReal());
	if ( useFastConversion )
	{
		scaleD.setReciprocal(scaleD);
	}

	for (int vi = numVerts - 1; vi >= 0; vi--)
	{
		// Estimate vertex
		hkIntVector vtx = vtxIds[vi];
		planesCol->getPlane(PlaneId(vtx.getComponent<0>()), planes[0]);
		planesCol->getPlane(PlaneId(vtx.getComponent<1>()), planes[1]);
		planesCol->getPlane(PlaneId(vtx.getComponent<2>()), planes[2]);

		hkVector4& vPos = verticesPos[vi];

		// Convert and add vertex
		if ( useFastConversion )
		{
			hkVector4d fvd;
			hkcdPlanarGeometryPredicates::approximateIntersectionFast(planes, fvd);
			fvd.mul(scaleD);	fvd.add(offsetD);	fvd.store<4, HK_IO_NATIVE_ALIGNED>(realBuff);
			vPos.load<4, HK_IO_NATIVE_ALIGNED>(realBuff);
		}
		else
		{
			hkcdPlanarGeometryPredicates::approximateIntersection(planes, vtx);
			planesCol->convertFixedPosition(vtx, vPos);
		}
	}
}

//
//	Collects all the leaf cells

void hkcdConvexCellsTree3D::collectLeafCells(hkArray<CellId>& cellIdsOut) const
{
	for (CellId cellId = m_cells->getFirstCellId(); cellId.isValid(); cellId = m_cells->getNextCellId(cellId))
	{
		// Get the cell and its children
		const Cell& cell = getCell(cellId);
		if ( !cell.isLeaf() )
		{
			continue;
		}

		cellIdsOut.pushBack(cellId);
	}
}

//
//	Collects all the cells marked as solid

void hkcdConvexCellsTree3D::collectSolidCells(hkArray<CellId>& cellIdsOut) const
{
	for (CellId cellId = m_cells->getFirstCellId(); cellId.isValid(); cellId = m_cells->getNextCellId(cellId))
	{
		// Get the cell and its children
		const Cell& cell = getCell(cellId);
		if ( !cell.isSolid() || !cell.isLeaf() )
		{
			continue;
		}

		cellIdsOut.pushBack(cellId);
	}
}

//
//	Extracts a solid planar geometry from a subset of selected cells
//	EXPECTS connectivity!

hkcdPlanarSolid* hkcdConvexCellsTree3D::buildSolidFromSubsetOfCells(const hkArray<CellId>& cellIdsIn)
{
	// Get all the boundary polygons from the cells
	hkArray<PolygonId> boundaryPolygonIds;
	hkRefPtr<hkcdPlanarGeometry> newGeom;
	if ( m_buildCellConnectivity )
	{
		newGeom.setAndDontIncrementRefCount(new hkcdPlanarGeometry(m_mesh->accessPlanesCollection(), 0, m_mesh->accessDebugger()));
		extractBoundaryPolygonsFromCellIds(cellIdsIn, *newGeom, boundaryPolygonIds);
	}
	else
	{
		newGeom = m_mesh;
		getUniquePolygonIdsFromCellIds(cellIdsIn, boundaryPolygonIds);
	}

	// Compute the corresponding bsp tree
	// If no polygon, no boundary is present, don't change the tree
	if ( boundaryPolygonIds.getSize() )
	{
		hkcdPlanarSolid* newSolidGeom = new hkcdPlanarSolid(newGeom->getPlanesCollection(), 0, newGeom->accessDebugger());

		// Collect all unique planes used by the polygons
		hkArray<PlaneId> planeIds;
		newGeom->getAllPolygonsPlanes(boundaryPolygonIds, planeIds, false);

		// Build the tree
		hkPseudoRandomGenerator rng(13);
		newSolidGeom->buildTree(*newGeom, rng, planeIds, boundaryPolygonIds, false, HK_NULL);

		return newSolidGeom;
	}

	return HK_NULL;
}

//
//	Returns a list polygons on the boundary of the solid, given a set of cells.
//	EXPECTS connectivity!!

void hkcdConvexCellsTree3D::extractBoundaryPolygonsFromCellIds(const hkArray<CellId>& cellIdsIn, hkcdPlanarGeometry& geomOut, hkArray<PolygonId>& boundaryPolygonIdsOut)
{
	// Get all the potential poly ids
	hkArray<PolygonId> polyToCheckIds;
	getUniquePolygonIdsFromCellIds(cellIdsIn, polyToCheckIds);
	boundaryPolygonIdsOut.reserve(polyToCheckIds.getSize()/2);

	// For each candidate poly, check the two neighboring cell state to decide
	for (int i = polyToCheckIds.getSize() - 1 ; i >=0 ; i--)
	{
		const PolygonId polyId		= polyToCheckIds[i];
		const Polygon& poly			= m_mesh->getPolygon(polyId);
		CellId posCellId			= CellId(poly.getPosCellId());
		CellId negCellId			= CellId(poly.getNegCellId());
		if ( !posCellId.isValid() || !negCellId.isValid() )		continue;

		const Cell& posCell			= getCell(posCellId);
		const Cell& negCell			= getCell(negCellId);
		HK_ASSERT(0xae2654aa, !posCell.isUnknown() && !negCell.isUnknown());

		// a poly is on the boundary iif on one side we have a solid cell and on the other side an empty cell
		if ( ((posCell.isSolid() && negCell.isEmpty()) || (posCell.isEmpty() && negCell.isSolid())) )
		{

			// Boundary polygon found. copy it into the output geometry
			PlaneId supportId		= poly.getSupportPlaneId();
			bool flipPoly			= !posCell.isSolid();
			supportId = ( flipPoly ) ? hkcdPlanarGeometryPrimitives::getOppositePlaneId(supportId) : supportId;

			// Set the bounding planes
			const int numBounds			= m_mesh->getNumBoundaryPlanes(polyId);
			const PolygonId newPolyId	= geomOut.addPolygon(supportId, 0, numBounds);
			{
				Polygon& newPoly		= geomOut.accessPolygon(newPolyId);

				if ( flipPoly )
				{
					for (int bi = numBounds - 1; bi >= 0; bi--)
					{
						const PlaneId boundId = poly.getBoundaryPlaneId(numBounds - 1 - bi);
						newPoly.setBoundaryPlaneId(bi, boundId);
					}
				}
				else
				{
					for (int bi = 0 ; bi < numBounds ; bi++)
					{
						const PlaneId boundId = poly.getBoundaryPlaneId(bi);
						newPoly.setBoundaryPlaneId(bi, boundId);
					}
				}
			}

			boundaryPolygonIdsOut.pushBack(newPolyId);
		}
	}
}

//
//	Return a list of unique polygon ids from a set of cells

void hkcdConvexCellsTree3D::getUniquePolygonIdsFromCellIds(const hkArray<CellId>& cellIdsIn, hkArray<PolygonId>& polygonIdsOut)
{
	// First, count the number of polygon to allocate
	int nbMaxPolys = 0;	
	for (int i = cellIdsIn.getSize() - 1 ; i >=0 ; i--)
	{		
		const int numPolys	= m_cells->getNumBoundaryPolygons(cellIdsIn[i]);
		nbMaxPolys			+= numPolys;
	}

	// Allocate and retrieve the polygons
	polygonIdsOut.reserve(nbMaxPolys);
	for (int i = cellIdsIn.getSize() - 1 ; i >=0 ; i--)
	{		
		const int numPolys	= m_cells->getNumBoundaryPolygons(cellIdsIn[i]);
		const Cell& cell	= getCell(cellIdsIn[i]);
		for (int pi = 0; pi < numPolys; pi++)
		{			
			const PolygonId polyId	= cell.getBoundaryPolygonId(pi);
			Polygon& poly			= m_mesh->accessPolygon(polyId);

			// When the convex cell tree is built with connectivity, polygon are shared between cells
			if ( !(poly.getMaterialId() & POLY_VISITED_FLAG) )
			{
				polygonIdsOut.pushBack(polyId);
				poly.setMaterialId(poly.getMaterialId() | POLY_VISITED_FLAG);
			}
		}
	}

	for (int i = cellIdsIn.getSize() - 1 ; i >=0 ; i--)
	{	
		const int numPolys	= m_cells->getNumBoundaryPolygons(cellIdsIn[i]);
		const Cell& cell	= getCell(cellIdsIn[i]);
		// Make sure the flip flag is not set on any support plane
		for (int pi = 0; pi < numPolys; pi++)
		{			
			const PolygonId polyId	= cell.getBoundaryPolygonId(pi);
			Polygon& poly			= m_mesh->accessPolygon(polyId);
			poly.setMaterialId(poly.getMaterialId() & (~POLY_VISITED_FLAG));
		}
	}
}

//
//	From a set of polygons of the original geometry, mark all the boundary of all the cells with an intersection test

void hkcdConvexCellsTree3D::markBoundaryCells(const hkcdPlanarGeometry& originalGeom, const hkArray<PolygonId>& originalBoundaryPolygonIds, hkcdPlanarEntityDebugger* debugger)
{
	// First of all, build a necessary acceleration structure: a map from Plane Id => Polygon Ids
	hkArray< hkArray<PolygonId> > planeIdToPolyIds;
	const PlanesCollection* planesCol = m_mesh->getPlanesCollection();
	planeIdToPolyIds.setSize(planesCol->getNumPlanes());

	// Add all the original polygons into the array
	for (int i = originalBoundaryPolygonIds.getSize() - 1 ; i >=0 ; i--)
	{
		int planeIdx = originalGeom.getPolygon(originalBoundaryPolygonIds[i]).getSupportPlaneId().value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);
		planeIdToPolyIds[planeIdx].pushBack(originalBoundaryPolygonIds[i]);
	}

	// Collect all the leaf cells
	hkArray<CellId> leafIds;
	collectLeafCells(leafIds);

	// Get all the polygon ids from the cells
	hkArray<PolygonId> polyToCheckIds;
	getUniquePolygonIdsFromCellIds(leafIds, polyToCheckIds);

	// For each poly, check if belongs to the boundary or not
	int nbIntersectionTests = 0;
	for (int i = polyToCheckIds.getSize() - 1 ; i >=0 ; i--)
	{
		// Get the plane Id corresponding to this cell boundary
		const PolygonId polyId		= polyToCheckIds[i];
		Polygon& poly				= m_mesh->accessPolygon(polyId);
		if ( poly.getMaterialId() == POLY_SURFACE_INVALID ) continue;  // Cannot be a boundary, invalid already (e.g. bounding box planes)
		PlaneId planeId				= poly.getSupportPlaneId();
		const int planeIdx			= planeId.value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);

		// Get the polygons to check for this plane
		hkArray<PolygonId>& candidateOriginalpolyIds = planeIdToPolyIds[planeIdx];
		for (int j = candidateOriginalpolyIds.getSize() - 1 ; j >= 0 ; j--)
		{
			// Check for intersection between the polygons
			const PolygonId candPolyId		= candidateOriginalpolyIds[j];
			const PlaneId candPlaneId		= originalGeom.getPolygon(candPolyId).getSupportPlaneId();
			const bool sameOrientation		= hkcdPlanarGeometryPrimitives::sameOrientationPlaneIds(planeId, candPlaneId);
			if (  sameOrientation && (poly.getMaterialId() & POLY_SURFACE_DIRECT) )		continue;
			if ( !sameOrientation && (poly.getMaterialId() & POLY_SURFACE_INDIRECT) )	continue;

			nbIntersectionTests++;
			if ( hkcdPlanarGeometry::check2dIntersection(originalGeom, candPolyId, *m_mesh, polyId) )
			{
				// Update poly info
				poly.setMaterialId(poly.getMaterialId() | (( sameOrientation ) ? POLY_SURFACE_DIRECT : POLY_SURFACE_INDIRECT) );
			}
		}
	}
}

//
//	Return a set of cell ids representing disconnected islands of solid regions

void hkcdConvexCellsTree3D::computeSolidRegionIslands(hkArray< hkArray<CellId> >& islands)
{
	// Get all leaf cells and init them
	hkArray<CellId> leafIds;	
	collectLeafCells(leafIds);
	const int numLeaves = leafIds.getSize();
	for (int i = numLeaves - 1 ; i >= 0 ; i--)
	{
		const CellId cellId = leafIds[i];
		Cell& cell = accessCell(cellId);
		cell.setVisited(false);
	}

	hkArray<CellId> cellQueue(numLeaves);
	CellId startCellId;
	islands.setSize(0);
	islands.reserve(8);

	do 
	{
		startCellId = CellId::invalid();

		// Find a starting point to flood a solid island
		for (int i = numLeaves - 1 ; i >= 0 ; i--) 
		{
			const CellId cellId = leafIds[i];
			Cell& cell			= accessCell(cellId);
			if ( !cell.isVisited() && cell.isSolid() )
			{
				startCellId = cellId;
				break;
			}
		}

		// Flood the island
		if ( startCellId.isValid() )
		{
			int queueHead, queueTail;
			queueHead = queueTail = 0;
			cellQueue[queueTail++]					= startCellId;
			accessCell(startCellId).setVisited(true);
			hkArray<CellId>& islandCellIds			= islands.expandOne();
			while ( queueHead < queueTail )
			{
				// Get the current cell
				const CellId cellId = cellQueue[queueHead++];
				Cell& cell			= accessCell(cellId);
				islandCellIds.pushBack(cellId);

				// Look at all its neighbors
				const int numPolys	= m_cells->getNumBoundaryPolygons(cellId);
				for (int pi = 0; pi < numPolys; pi++)
				{
					const PolygonId polyId	= cell.getBoundaryPolygonId(pi);
					const Polygon& poly		= m_mesh->getPolygon(polyId);

					// Get the neighbor cell id
					const hkUint32 nCellIdx	= (poly.getNegCellId() == cellId.value()) ? poly.getPosCellId() : poly.getNegCellId();
					const CellId nCellId	(nCellIdx);

					if ( nCellId.isValid() )
					{						
						Cell& nCell = accessCell(nCellId);

						// Add the neighbor to queue 
						if ( !nCell.isVisited() && nCell.isSolid() )
						{
							cellQueue[queueTail++] = nCellId;
							nCell.setVisited(true);
						}
					}
				}
			}
		}
	} while ( startCellId.isValid() );
}

//
//	Finds an empty cell

hkcdConvexCellsTree3D::CellId hkcdConvexCellsTree3D::findOutputCell()
{	
	for (CellId cellId = m_cells->getFirstCellId(); cellId.isValid(); cellId = m_cells->getNextCellId(cellId))
	{
		// Check if leaf
		const Cell& cell = getCell(cellId);
		if ( cell.isLeaf() )
		{
			// Check if it has a boundary belonging to the bounding box
			const int numPolys	= m_cells->getNumBoundaryPolygons(cellId);
			for (int pi = 0; pi < numPolys; pi++)
			{
				const PolygonId polyId	= cell.getBoundaryPolygonId(pi);
				const Polygon& poly		= m_mesh->getPolygon(polyId);
				if ( poly.getNegCellId() == CellId::invalid().valueUnchecked() || poly.getPosCellId() == CellId::invalid().valueUnchecked() )
				{
					return cellId;
				}
			}
		}
	}

	return CellId::invalid();
}

//
//	Infers in/out labels by flood filling thanks to the boundary properties computed on each polygon of each cell

void hkcdConvexCellsTree3D::inferCellsLabels(hkcdPlanarEntityDebugger* debugger)
{
	// Get all cells
	hkArray<CellId> leafIds;	
	collectLeafCells(leafIds);
	for (int i = leafIds.getSize() - 1 ; i >= 0 ; i--) { accessCell(leafIds[i]).setVisited(false); accessCell(leafIds[i]).setUserData(0); }
	hkArray<CellId> cellQueue(leafIds.getSize());

	hkUint32 currentLevel	= 1;			// even number is outside, odd number inside (0 is invalid level)
	int nbLabeledRegions	= 0;
	bool outLevel;
	int queueHead, queueTail;

	// Start flood/peeling alg
	CellId nextStartingCellId = findOutputCell();
	HK_ASSERT(0xfd12a35e, nextStartingCellId.isValid());
	accessCell(nextStartingCellId).setUserData(currentLevel);

	// Loop over the "peeling" levels, usually small loop number (2 for standard filled geometry)
	while ( nextStartingCellId.isValid() )					
	{
		currentLevel							= getCell(nextStartingCellId).getUserData();
		outLevel								= ( currentLevel % 2 == 1 );
		queueHead = queueTail					= 0;
		cellQueue[queueTail++]					= nextStartingCellId;
		accessCell(nextStartingCellId).setVisited(true);		

		// flood the in/out information 
		while ( queueHead < queueTail )
		{
			// Get the current cell
			const CellId cellId = cellQueue[queueHead++];
			Cell& cell			= accessCell(cellId);
			cell.setLabel(outLevel ? hkcdConvexCellsCollection::CELL_EMPTY : hkcdConvexCellsCollection::CELL_SOLID);
			cell.setUserData(currentLevel);
			nbLabeledRegions++;

			// Look at all its neighbors
			const int numPolys	= m_cells->getNumBoundaryPolygons(cellId);
			for (int pi = 0; pi < numPolys; pi++)
			{
				const PolygonId polyId	= cell.getBoundaryPolygonId(pi);
				const Polygon& poly		= m_mesh->getPolygon(polyId);

				// Get the neigbhor cell id
				const CellId nCellId	= CellId(( poly.getNegCellId() == cellId.value() ) ? poly.getPosCellId() : poly.getNegCellId());				
				if ( nCellId.isValid() && !getCell(nCellId).isVisited() )
				{
					Cell& nCell = accessCell(nCellId);
					// If the poly represent a boundary (stored in material of poly), we cannot go through!
					if ( canGoFromCellThroughPoly(cellId, polyId, outLevel) )
					{
						// Add the neighbor to list 
						cellQueue[queueTail++] = nCellId;
						nCell.setVisited(true);
					}
					else
					{
						// We may have found a inside cell candidate for the next peeling stage
						if ( nCell.getUserData() == 0 )  nCell.setUserData(currentLevel + 1);
					}
				}
			}
		}

		// If we don't have a starting cell, but not all cells are labeled, find one
		nextStartingCellId						= CellId::invalid();
		if ( nbLabeledRegions < leafIds.getSize() )
		{
			hkUint32 minLevelFound	= 0xffffffff;
			for (int i = leafIds.getSize() - 1 ; i >= 0 ; i--) 
			{
				const CellId cellId = leafIds[i];
				Cell& cell			= accessCell(cellId);				
				if ( !cell.isVisited() && cell.getUserData() > 0 && cell.getUserData() < minLevelFound )
				{
					nextStartingCellId	= cellId;
					minLevelFound		= cell.getUserData();
				}
			}
		}
	}

	HK_ASSERT(0xe987a65c, nbLabeledRegions == leafIds.getSize());
}

//
//	Updates the state of each cell by copying the label of the node of the provided solid planar geometry

void hkcdConvexCellsTree3D::relabelCellsFromSolid(const hkcdPlanarSolid* solid)
{
	for (CellId cellId = m_cells->getFirstCellId(); cellId.isValid(); cellId = m_cells->getNextCellId(cellId))
	{
		Cell& cell = accessCell(cellId);
		if ( cell.isLeaf() )
		{
			// Get the corresponding node
			NodeId nId = NodeId(cell.getUserData());
			if  ( nId.isValid() )
			{
				const Node& node = solid->getNode(nId);
				HK_ASSERT(0x3b6a4de7, node.m_typeAndFlags ==  hkcdPlanarSolid::NODE_TYPE_IN || node.m_typeAndFlags ==  hkcdPlanarSolid::NODE_TYPE_OUT);
				cell.setLabel(( node.m_typeAndFlags ==  hkcdPlanarSolid::NODE_TYPE_IN ) ? hkcdConvexCellsCollection::CELL_SOLID : hkcdConvexCellsCollection::CELL_EMPTY);
			}
		}
	}
}

//
//	Re-assign the labels of a bsp tree using the label store in the convex cell tree.
//	EXPECTS connectivity!

void hkcdConvexCellsTree3D::reassignSolidGeomLabels(const hkcdPlanarGeometry& originalGeom, hkcdPlanarSolid* solid, const hkArray<PolygonId>& originalBoundaryPolygonIds, hkcdPlanarEntityDebugger* debugger)
{
	HK_ASSERT(0xb37a654c, solid);

	// infer all the labels
	markBoundaryCells(originalGeom, originalBoundaryPolygonIds, debugger);
	inferCellsLabels(debugger);

	// Copy cell labels into the tree
	const hkcdPlanarSolid::NodeStorage* nodes = solid->getNodes();
	for (int ni = nodes->getCapacity() - 1; ni >= 0; ni--)
	{
		const NodeId nodeId(ni);
		Node& bspNode = solid->accessNode(nodeId);

		if ( bspNode.isAllocated() && ((bspNode.m_typeAndFlags == hkcdPlanarSolid::NODE_TYPE_IN) || (bspNode.m_typeAndFlags == hkcdPlanarSolid::NODE_TYPE_OUT)) )
		{
			// Get its associated cell
			const CellId cellId(bspNode.m_data);
			if ( !cellId.isValid() )
			{
				bspNode.m_typeAndFlags = hkcdPlanarSolid::NODE_TYPE_UNKNOWN;
				continue;
			}
			Cell& cell = accessCell(cellId);
			HK_ASSERT(0x3d76c3d9, !cell.isUnknown());
			cell.setUserData(nodeId.value());
			bspNode.m_typeAndFlags = cell.isSolid() ? hkcdPlanarSolid::NODE_TYPE_IN : hkcdPlanarSolid::NODE_TYPE_OUT;
		}
	}
}

//
//	Clip a set of input polygons in an input geometry using the boundary polygons of the convex cell tree
/*
void hkcdConvexCellsTree::clipPolygonsWithBoundary(	const hkcdPlanarSolid* solid, hkcdPlanarGeometry& inputGeom,
													const hkArray<PolygonId>& inputPolygons, hkArray<PolygonId>& onBoundaryPolyIdsOut, hkArray<PolygonId>& remainingpolyIdsOut)
{
	HK_ASSERT(0xb37a658d, m_buildCellConnectivity && m_solidMesh);

	// First, get the boundary polygons
	hkArray<CellId> solidCellIds;			collectSolidCells(solidCellIds);
	hkArray<PolygonId> boundaryPolyIds;		extractBoundaryPolygonsFromCellIds(solid, solidCellIds, inputGeom, boundaryPolyIds);

	// Regroup boundary polys by plane
	hkArray< hkArray<PolygonId> > boundaryPlaneIdToPolyIds;
	boundaryPlaneIdToPolyIds.setSize(inputGeom.getPlaneCollection()->getNumPlanes());
	for (int i = boundaryPolyIds.getSize() - 1 ; i >=0 ; i--)
	{
		int planeIdx = inputGeom.getPolygon(boundaryPolyIds[i]).getSupportPlaneId().value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);
		boundaryPlaneIdToPolyIds[planeIdx].pushBack(boundaryPolyIds[i]);
	}

	// Regroup input polys by plane
	hkArray< hkArray<PolygonId> > inputPlaneIdToPolyIds;
	inputPlaneIdToPolyIds.setSize(inputGeom.getPlaneCollection()->getNumPlanes());
	for (int i = inputPolygons.getSize() - 1 ; i >=0 ; i--)
	{
		int planeIdx = inputGeom.getPolygon(inputPolygons[i]).getSupportPlaneId().value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);
		inputPlaneIdToPolyIds[planeIdx].pushBack(inputPolygons[i]);
	}

	// Loop over the planes
	for (int pidx = boundaryPlaneIdToPolyIds.getSize() - 1 ; pidx >= 0 ; pidx--)
	{

		// Treat the two faces of the plane separately
		for (int f = 0 ; f <= 1 ; f++)
		{

			const hkArray<PolygonId>& boundaryPolyIdsOnThisFace		= boundaryPlaneIdToPolyIds[pidx];
			const hkArray<PolygonId>& inputPolyIdsOnThisFace		= inputPlaneIdToPolyIds[pidx];

			if ( inputPolyIdsOnThisFace.getSize() == 0 ) continue;
			if ( boundaryPolyIdsOnThisFace.getSize() == 0 )
			{
				remainingpolyIdsOut.append(inputPolyIdsOnThisFace);
				continue;
			}

			// Get the actual (flip or not flipped) plane id
			const Polygon& polyP	= inputGeom.getPolygon(inputPolyIdsOnThisFace[0]);
			PlaneId planeIdf		= PlaneId(polyP.getSupportPlaneId().value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG));
			planeIdf				= ( f == 0 ) ? planeIdf : PlaneId(planeIdf.value() | hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);

			// Get all the corresponding polygons
			hkArray<PolygonId> boundaryPolyIdsOnThisPlane;
			for (int pi = boundaryPolyIdsOnThisFace.getSize() - 1 ; pi >= 0 ; pi--)
			{
				const Polygon& polyP	= inputGeom.getPolygon(boundaryPolyIdsOnThisFace[pi]);
				const PlaneId planeIdc	= polyP.getSupportPlaneId();
				if ( hkcdPlanarGeometryPrimitives::sameOrientationPlaneIds(planeIdc, planeIdf) ) boundaryPolyIdsOnThisPlane.pushBack(boundaryPolyIdsOnThisFace[pi]);
			}			
			hkArray<PolygonId> inputPolyIdsOnThisPlane;
			for (int pi = inputPolyIdsOnThisFace.getSize() - 1 ; pi >= 0 ; pi--)
			{
				const Polygon& polyP	= inputGeom.getPolygon(inputPolyIdsOnThisFace[pi]);
				const PlaneId planeIdc	= polyP.getSupportPlaneId();
				if ( hkcdPlanarGeometryPrimitives::sameOrientationPlaneIds(planeIdc, planeIdf) ) inputPolyIdsOnThisPlane.pushBack(inputPolyIdsOnThisFace[pi]);
			}
			const int nbBoundaryPolysOnThisPlane					= boundaryPolyIdsOnThisPlane.getSize();
			const int nbInputPolysOnThisPlane						= inputPolyIdsOnThisPlane.getSize();

			// If no boundary poly, then add all the input poly to the remaining list
			if ( nbBoundaryPolysOnThisPlane == 0 )
			{
				remainingpolyIdsOut.append(inputPolyIdsOnThisPlane);
				continue;
			}

			// No input polys case
			if ( nbInputPolysOnThisPlane == 0 ) continue;

			// Extract the 2D BSP tree out of the face
			// In that goal, first gather the boundary planes of all the poly
			hkArray<PlaneId> edgePlaneIds;
			for (int p = boundaryPolyIdsOnThisPlane.getSize() - 1 ; p >= 0 ; p --)
			{
				const PolygonId polyId			= boundaryPolyIdsOnThisPlane[p];
				const Polygon& poly				= inputGeom.getPolygon(polyId);
				const int numBPlanes			= inputGeom.getNumBoundaryPlanes(polyId);
				PlaneId* planeIdPtr				= edgePlaneIds.expandBy(numBPlanes);
				for (int b = 0 ; b < numBPlanes ; b++)
				{
					planeIdPtr[b]				= PlaneId(poly.getBoundaryPlaneId(b).value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG));
				}
			}

			// Sort all the planes, and remove duplicates
			hkSort(edgePlaneIds.begin(), edgePlaneIds.getSize());
			edgePlaneIds.setSize(hkAlgorithm::removeDuplicatesFromSortedList(edgePlaneIds.begin(), edgePlaneIds.getSize()));

			// Build a BSP out of this surface delimitation
			hkcdPlanarSolid faceBSP(&inputGeom);
			hkPseudoRandomGenerator rng(13);
			hkSort(edgePlaneIds.begin(), edgePlaneIds.getSize());
			faceBSP.buildTree2D(inputGeom, rng, edgePlaneIds, boundaryPolyIdsOnThisPlane, HK_NULL);
			faceBSP.simplifyFromBoundaries();
			hkArray<PolygonId> insidePolyIds, bPolyIds, outsidePolyIds;

			// Classify the input polys with it
			faceBSP.classifyPolygons(inputGeom, inputPolyIdsOnThisPlane, insidePolyIds, bPolyIds, outsidePolyIds);

			// Output results (2D BSP, boundary result is ignored)
			onBoundaryPolyIdsOut.append(insidePolyIds);
			remainingpolyIdsOut.append(outsidePolyIds);
		}
	}
}*/

//
//	Converts a single cell to geometry

void hkcdConvexCellsTree3D::extractCellGeometry(CellId cellId, hkGeometry& cellGeomOut) const
{
	hkArray<int> polyIb;
	const Cell& cell = getCell(cellId);
	hkFindUniquePositionsUtil vtxWelder;
	const int numPolys	= m_cells->getNumBoundaryPolygons(cellId);
	const PlanesCollection* planesCol = m_mesh->accessPlanesCollection();

	for (int pi = 0; pi < numPolys; pi++)
	{
		// Each polygon is convex
		const PolygonId polyId	= cell.getBoundaryPolygonId(pi);
		const Polygon& poly		= m_mesh->getPolygon(polyId);

		// Compute all vertices
		const int numPolyVerts = m_mesh->getNumBoundaryPlanes(polyId);
		polyIb.setSize(numPolyVerts);

		Plane planes[3];	planesCol->getPlane(poly.getSupportPlaneId(), planes[0]);
		for (int prev = numPolyVerts - 1, crt = 0; crt < numPolyVerts; prev = crt, crt++)
		{
			planesCol->getPlane(poly.getBoundaryPlaneId(prev), planes[1]);
			planesCol->getPlane(poly.getBoundaryPlaneId(crt), planes[2]);

			hkIntVector iv;		hkcdPlanarGeometryPredicates::approximateIntersection(planes, iv);
			hkVector4 fv;		planesCol->convertFixedPosition(iv, fv);
			polyIb[crt]			= vtxWelder.addPosition(fv);
		}

		// Triangulate. The polygon is convex
		int ixMin = 0;
		for (int i = 1 ; i < numPolyVerts ; i++)
		{
			if ( polyIb[i] < polyIb[ixMin] ) ixMin = i;
		}
		int ixP1 = ixMin + 1;
		if ( ixP1 == numPolyVerts ) ixP1 = 0;
		int ixP2 = ixP1 + 1;
		if ( ixP2 == numPolyVerts ) ixP2 = 0;
		for (int k = 2; k < numPolyVerts; k++)
		{
			cellGeomOut.m_triangles.expandOne().set(polyIb[ixMin], polyIb[ixP1], polyIb[ixP2], cellId.value());
			ixP1++;
			if ( ixP1 == numPolyVerts ) ixP1 = 0;
			ixP2 = ixP1 + 1;
			if ( ixP2 == numPolyVerts ) ixP2 = 0;
		}
	}

	cellGeomOut.m_vertices.swap(vtxWelder.m_positions);
}

void hkcdConvexCellsTree3D::debugPrint() const
{
	for (CellId cellId = m_cells->getFirstCellId(); cellId.isValid(); cellId = m_cells->getNextCellId(cellId))
	{
		// Get the cell and its children
		const Cell& cell = getCell(cellId);
	
		hkStringBuf str;
		str.printf("Cell id %d --------------------------------------", cellId.value());
		HK_REPORT(str);

		const int numPolys	= m_cells->getNumBoundaryPolygons(cellId);
		str.printf("num polys is %d", numPolys);
		HK_REPORT(str);

		for (int pi = 0; pi < numPolys; pi++)
		{
			// Each polygon is convex
			const PolygonId polyId	= cell.getBoundaryPolygonId(pi);
			const Polygon& poly		= m_mesh->getPolygon(polyId);

			// Compute all vertices
			const int numPolyVerts = m_mesh->getNumBoundaryPlanes(polyId);

			str.printf("POLY %d has %d vertices", pi, numPolyVerts);
			HK_REPORT(str);

			
			int planeIds[3];	planeIds[0] = poly.getSupportPlaneId().value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);
			for (int prev = numPolyVerts - 1, crt = 0; crt < numPolyVerts; prev = crt, crt++)
			{
				planeIds[1] = poly.getBoundaryPlaneId(prev).value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);
				planeIds[2] = poly.getBoundaryPlaneId(crt).value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);

				str.printf("vertex %d, %d, %d", planeIds[0], planeIds[1], planeIds[2]);
				HK_REPORT(str);

			}

		}
	}
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
