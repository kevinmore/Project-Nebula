/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Geometry/Collide/hkcdCollide.h>
#include <Geometry/Collide/DataStructures/Planar/ConvexCellsTree/hkcdConvexCellsTree2D.h>

//
//	Constructor

hkcdConvexCellsTree2D::hkcdConvexCellsTree2D(hkcdPlanarGeometry* geom, PlaneId supportPlaneId)
:	hkcdConvexCellsTree<hkcdPlanarEntity::Polygon, hkcdPlanarEntity::PolygonId, hkcdPlanarGeometry>(geom)
,	m_supportPlaneId(supportPlaneId)
{
	m_cells.setAndDontIncrementRefCount(new hkcdPlanarGeometry());

	// Share the planes collection with the "cells" collection
	m_cells->setPlanesCollection(geom->accessPlanesCollection(), HK_NULL);
}

//
//	Creates a box cell that encloses the entire "known" space

hkcdConvexCellsTree2D::CellId hkcdConvexCellsTree2D::createBoundaryCell()
{
	const PlanesCollection* geomPlanes = m_mesh->getPlanesCollection();

	// Get the largest component of the support plane normal
	Plane supportPlane;		geomPlanes->getPlane(m_supportPlaneId, supportPlane);
	hkInt64Vector4 iN;		supportPlane.getExactNormal(iN);
	hkInt64Vector4 absN;	absN.setAbs(iN);
	const int axisZ			= absN.getIndexOfMaxComponent<3>();
	const int signZ			= (iN.getComponent(axisZ) >> 63L) & 1;

	// Determine bounds
	static const PlanesCollection::Bounds polyBounds[PlanesCollection::NUM_BOUNDS][4] = 
	{
		{ PlanesCollection::BOUND_NEG_Y, PlanesCollection::BOUND_NEG_Z, PlanesCollection::BOUND_POS_Y, PlanesCollection::BOUND_POS_Z },
		{ PlanesCollection::BOUND_NEG_Y, PlanesCollection::BOUND_POS_Z, PlanesCollection::BOUND_POS_Y, PlanesCollection::BOUND_NEG_Z },
		{ PlanesCollection::BOUND_NEG_Z, PlanesCollection::BOUND_NEG_X, PlanesCollection::BOUND_POS_Z, PlanesCollection::BOUND_POS_X },
		{ PlanesCollection::BOUND_NEG_Z, PlanesCollection::BOUND_POS_X, PlanesCollection::BOUND_POS_Z, PlanesCollection::BOUND_NEG_X },
		{ PlanesCollection::BOUND_NEG_Y, PlanesCollection::BOUND_POS_X, PlanesCollection::BOUND_POS_Y, PlanesCollection::BOUND_NEG_X },
		{ PlanesCollection::BOUND_NEG_Y, PlanesCollection::BOUND_NEG_X, PlanesCollection::BOUND_POS_Y, PlanesCollection::BOUND_POS_X },
	};
	const PlanesCollection::Bounds selBound = (PlanesCollection::Bounds)((axisZ << 1) + signZ);
	const PlaneId* selBounds = reinterpret_cast<const PlaneId*>(polyBounds[selBound]);
	
	// Create the cell
	CellId cellId	= m_cells->addPolygon(m_supportPlaneId, 0, 4);
	Cell& cell		= m_cells->accessPolygon(cellId);
	cell.setBoundaryPlaneId(0, selBounds[0]);
	cell.setBoundaryPlaneId(1, selBounds[1]);
	cell.setBoundaryPlaneId(2, selBounds[2]);
	cell.setBoundaryPlaneId(3, selBounds[3]);

	// Returns the newly created polygon
	return cellId;
}

//
//	Build a convex cell tree out of a solid bsp tree

void hkcdConvexCellsTree2D::buildFromSolid(hkcdPlanarSolid* solid)
{
	// Set-up boundary cell
	{
		CellId boundaryCellId = createBoundaryCell();
		solid->accessNode(solid->getRootNodeId()).m_data = boundaryCellId.value();
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
				nodeStack.pushBack(node.m_left);
			}

			// Recurse on right
			{					
				Node& rightChild	= solid->accessNode(node.m_right);	
				rightChild.m_data	= outCell.valueUnchecked();
				nodeStack.pushBack(node.m_right);
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
