/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Geometry/Collide/hkcdCollide.h>
#include <Geometry/Collide/DataStructures/Planar/ConvexCellsTree/hkcdConvexCellsTree3D.h>

#include <Common/Base/Algorithm/Sort/hkSort.h>

//
//	Constructor

hkcdPlanarSolid::hkcdPlanarSolid(const PlanesCollection* planesCollection, int initialNodeCapacity, hkcdPlanarEntityDebugger* debugger)
:	hkcdPlanarEntity(debugger)
,	m_planes(planesCollection)
,	m_rootNodeId(NodeId::invalid())
{
	m_nodes.setAndDontIncrementRefCount(new NodeStorage());

	// Pre-allocate the nodes array if required
	if ( initialNodeCapacity )
	{
		m_nodes->grow(initialNodeCapacity);
	}
}

/// Constructor from collection pointers
hkcdPlanarSolid::hkcdPlanarSolid(NodeStorage* nodeStorage, NodeId rootNodeId, const PlanesCollection* planesCollection, hkcdPlanarEntityDebugger* debugger)
:	hkcdPlanarEntity(debugger)
,	m_nodes(nodeStorage)
,	m_planes(planesCollection)
,	m_rootNodeId(rootNodeId)
{}

//
//	Copy constructor

hkcdPlanarSolid::hkcdPlanarSolid(const hkcdPlanarSolid& other)
:	hkcdPlanarEntity(other)
,	m_planes(other.m_planes)
,	m_rootNodeId(other.m_rootNodeId)
{
	m_nodes.setAndDontIncrementRefCount(new NodeStorage());
	m_nodes->copy(*other.m_nodes);
}

//
//	Destructor

hkcdPlanarSolid::~hkcdPlanarSolid()
{
	m_planes = HK_NULL;
}

//
//	Clears the tree

void hkcdPlanarSolid::clear()
{
	m_nodes->clear();
	m_rootNodeId = NodeId::invalid();
}

//
//	Builds a convex solid from an array of planes. The planes are assumed to bound a closed convex region, there are no checks to validate the assumption!

void hkcdPlanarSolid::buildConvex(const PlaneId* HK_RESTRICT planesIn, int numPlanes)
{
	if ( numPlanes == 0 )
	{
		return;
	}

	NodeId nodeId					= m_nodes->allocate();
	accessNode(nodeId).m_parent		= NodeId::invalid();
	m_rootNodeId					= nodeId;

	for (int k = 0; k < numPlanes; k++)
	{
		Node& node			= accessNode(nodeId);

		// Set it up
		node.m_data			= 0;
		node.m_typeAndFlags	= NODE_TYPE_INTERNAL;
		node.m_planeId		= planesIn[k];
		const NodeId inNode	= createInsideNode(nodeId);
		const NodeId ouNode	= createOutsideNode(nodeId);
		accessNode(nodeId).m_left		= inNode;
		accessNode(nodeId).m_right		= ouNode;
		nodeId				= getNode(nodeId).m_left;
	}
}

//
//	Sets a new planes collection. If the plane remapping table is non-null, the plane Ids on all nodes will be re-set as well (i.e. to match the plane Ids in the new collection)

void hkcdPlanarSolid::setPlanesCollection(const PlanesCollection* newPlanes, int* HK_RESTRICT planeRemapTable, bool addMissingPlanes)
{
	// Remap all plane Ids if necessary
	if ( planeRemapTable && m_planes && newPlanes && (m_planes != newPlanes) )
	{
		for (int ni = m_nodes->getCapacity() - 1; ni >= 0; ni--)
		{
			const NodeId nodeId		(ni);
			Node& node				= accessNode(nodeId);
			const PlaneId oldPlaneId = node.m_planeId;

			if ( node.isAllocated() && oldPlaneId.isValid() )
			{
				const int oldPlaneIdx		= oldPlaneId.value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);
				int newPlaneIdx				= planeRemapTable[oldPlaneIdx] & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);
				Plane oldPlane;				m_planes->getPlane(oldPlaneId, oldPlane);
				// We may want to add the plane now
				if ( addMissingPlanes && !PlaneId(newPlaneIdx).isValid() )
				{
					newPlaneIdx						= ((PlanesCollection*)newPlanes)->addPlane(oldPlane).value();
					planeRemapTable[oldPlaneIdx]	= newPlaneIdx;
				}
				Plane newPlane;				newPlanes->getPlane(PlaneId(newPlaneIdx), newPlane);
				const PlaneId newPlaneId	(newPlaneIdx | (hkcdPlanarGeometryPredicates::sameOrientation(oldPlane, newPlane) ? 0 : hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG));

				node.m_planeId = newPlaneId;
			}
		}
	}

	m_planes = newPlanes;
}

//
//	Shift all plane ids

void hkcdPlanarSolid::shiftPlaneIds(int offsetValue)
{
	for (int ni = m_nodes->getCapacity() - 1; ni >= 0; ni--)
	{
		const NodeId nodeId			(ni);
		Node& node					= accessNode(nodeId);
		const PlaneId oldPlaneId	= node.m_planeId;
		if ( node.isAllocated() && oldPlaneId.isValid() )
		{
			const int oldPlaneIdx		= oldPlaneId.value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);
			if ( oldPlaneIdx >= hkcdPlanarGeometryPlanesCollection::NUM_BOUNDS )
			{
				const PlaneId newPlaneId	( oldPlaneId.value() + offsetValue );
				node.m_planeId = newPlaneId;
			}
		}
	}
}

//
//	Collects all the plane Ids used by the Bsp tree

void hkcdPlanarSolid::collectUsedPlaneIds(hkBitField& usedPlaneIdsOut) const
{
	for (int ni = m_nodes->getCapacity() - 1; ni >= 0; ni--)
	{
		const NodeId nodeId		(ni);
		const Node& node		= getNode(nodeId);
		const PlaneId oldPlaneId = node.m_planeId;

		if ( node.isAllocated() && oldPlaneId.isValid() )
		{
			const int planeIdx	= oldPlaneId.value() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG);
			usedPlaneIdsOut.set(planeIdx);
		}
	}

	// Always add the world boundary planes
	for (int k = hkcdPlanarGeometryPlanesCollection::NUM_BOUNDS - 1; k >= 0; k--)
	{
		usedPlaneIdsOut.set(k);
	}
}

//
//	Simplifies the tree by collapsing nodes with identical labels

hkBool32 hkcdPlanarSolid::collapseIdenticalLabels()
{
	bool hasCollapsed;
	int numCollapses = -1;
	do
	{
		hasCollapsed = false;
		numCollapses++;

		for (int ni = m_nodes->getCapacity() - 1; ni >= 0; ni--)
		{
			const NodeId nodeId	(ni);
			Node& startNode		= accessNode(nodeId);

			if ( startNode.isAllocated() && (startNode.m_typeAndFlags == NODE_TYPE_INTERNAL) )
			{
				// If the node has both leaves equal, we can collapse it
				NodeId crtNodeId = nodeId;
				while ( canCollapse(crtNodeId) )
				{
					// We can collapse this node
					Node& crtNode			= accessNode(crtNodeId);
					
					// Take the label if its children
					crtNode.m_typeAndFlags	= getNodeLabel(crtNode.m_left);

					// Release children
					m_nodes->release(getNode(crtNodeId).m_left);
					m_nodes->release(getNode(crtNodeId).m_right);

					// Invalidate children
					accessNode(crtNodeId).m_left  = NodeId::invalid();
					accessNode(crtNodeId).m_right = NodeId::invalid();

					// Switch to parent to propagate collapse
					crtNodeId = getNode(crtNodeId).m_parent;
					hasCollapsed = true;
				}
			}
		}
	} while ( hasCollapsed );

	return numCollapses;
}

//
//	Optimizes the storage, by moving all unallocated nodes at the end and releasing unused memory. This will
//	modify the Ids of the nodes!

void hkcdPlanarSolid::optimizeStorage()
{
	const int maxNumNodes = m_nodes->getCapacity();

	hkArray<Node>	newNodes;		newNodes.reserve(maxNumNodes);
	hkArray<NodeId>	newFromOldId;	newFromOldId.setSize(maxNumNodes, NodeId::invalid());

	int newNodeId = 0;
	for (int k = 0; k < maxNumNodes; k++)
	{
		const NodeId oldNodeId(k);
		const Node& node = getNode(oldNodeId);

		if ( node.isAllocated() )
		{
			newNodes.pushBack(node);
			newFromOldId[k] = NodeId(newNodeId);
			newNodeId++;
		}
	}

	// Update ids
	for (int k = 0; k < newNodes.getSize(); k++)
	{
		Node& node = newNodes[k];

		if ( node.m_left.isValid() )	{	node.m_left		= newFromOldId[node.m_left.value()];	}
		if ( node.m_right.isValid() )	{	node.m_right	= newFromOldId[node.m_right.value()];	}
		if ( node.m_parent.isValid() )	{	node.m_parent	= newFromOldId[node.m_parent.value()];	}
	}

	// Update root and make sure the special nodes have not changed!
	if ( m_rootNodeId.isValid() )	{	m_rootNodeId = newFromOldId[m_rootNodeId.value()];	}

	// Finally, swap the new array with the old
	m_nodes->swapStorage(newNodes);
	m_nodes->compact();
}

//
//	Utility functions

namespace hkcdBspImpl
{
	// Type shortcuts
	typedef hkcdPlanarGeometry::PlaneId			PlaneId;
	typedef hkcdPlanarGeometry::PolygonId		PolygonId;
	typedef hkcdPlanarSolid::NodeId		NodeId;

	/// Stack entry
	struct StackEntry
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY, hkcdBspImpl::StackEntry);

		PlaneId* m_planeIds;			///< The array of plane Ids that can be used to classify the polygons
		PolygonId* m_polygonIds;		///< The array of polygon Ids that still need to be classified
		int m_numPlaneIds;				///< The number of plane Ids
		int m_numPolygonIds;			///< The number of polygon Ids
		int m_isLeftChild;				///< True if this child is the left child of the parent node
		NodeId m_parentNodeId;			///< The parent node that pushed this entry
	};

	/// The stack
	class Stack
	{
		public:

			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY, hkcdBspImpl::Stack);
			enum { LEFT_CHILD_FLAG	= 0x80000000, };

		public:

			HK_FORCE_INLINE void pop(StackEntry& entryOut)
			{
				const int n = m_storage.getSize();
				entryOut.m_isLeftChild		= m_storage[n - 2] & LEFT_CHILD_FLAG;
				entryOut.m_parentNodeId		= NodeId(m_storage[n - 1]);
				entryOut.m_numPolygonIds	= m_storage[n - 2] & (~LEFT_CHILD_FLAG);
				entryOut.m_numPlaneIds		= m_storage[n - 3];
				entryOut.m_polygonIds		= reinterpret_cast<PolygonId*>(&m_storage[n - 3 - entryOut.m_numPolygonIds]);
				entryOut.m_planeIds			= reinterpret_cast<PlaneId*>(&m_storage[n - 3 - entryOut.m_numPolygonIds - entryOut.m_numPlaneIds]);
				m_storage.setSize(n - 3 - entryOut.m_numPolygonIds - entryOut.m_numPlaneIds);
			}

			HK_FORCE_INLINE void push(const StackEntry& entryIn)
			{
				const int n = m_storage.getSize() + entryIn.m_numPlaneIds + entryIn.m_numPolygonIds + 3;
				m_storage.setSize(n);
				hkString::memCpy4(&m_storage[n - 3 - entryIn.m_numPolygonIds - entryIn.m_numPlaneIds], entryIn.m_planeIds, entryIn.m_numPlaneIds);
				hkString::memCpy4(&m_storage[n - 3 - entryIn.m_numPolygonIds], entryIn.m_polygonIds, entryIn.m_numPolygonIds);
				m_storage[n - 3]	= entryIn.m_numPlaneIds;
				m_storage[n - 2]	= entryIn.m_numPolygonIds | (entryIn.m_isLeftChild ? (int)LEFT_CHILD_FLAG : 0);
				m_storage[n - 1]	= entryIn.m_parentNodeId.valueUnchecked();
			}

			HK_FORCE_INLINE bool isEmpty() const
			{
				return !m_storage.getSize();
			}

		protected:

			hkArray<int> m_storage;	///< The stack storage
	};
}

//
//	Selects a splitting plane from the given list of polygons

int hkcdPlanarSolid::pickSplittingPlane(const hkcdPlanarGeometry& geom, hkPseudoRandomGenerator& rng,
												const PlaneId* HK_RESTRICT planeIds, int numPlanes,
												const PolygonId* HK_RESTRICT polygonIds, int numPolygons)
{
	HK_ASSERT(0xde15ab23, numPlanes > 0);

	// Initialize our best estimate for the splitting plane
	int bestCost		= -0x7FFFFFFF;
	int bestPlaneIdx	= -1;

	// Try a fixed number of random planes
	for (int crtTry = 0; crtTry < 5; crtTry++)	
	{
		const int pi				= rng.getRand32() % numPlanes;
		const PlaneId splitPlaneId	= planeIds[pi];

		// Clear statistics
		int numInFront	= 0;
		int numBehind	= 0;
		int numSplit	= 0;

		// Test all other polygons against the current splitting plane
		for (int pj = 0; pj < numPolygons; pj++)
		{
			if ( pj != pi )
			{
				const hkcdPlanarGeometryPredicates::Orientation ori = geom.approxClassify(polygonIds[pj], splitPlaneId);
				switch ( ori )
				{
				case hkcdPlanarGeometryPredicates::BEHIND:		numBehind++;	break;
				case hkcdPlanarGeometryPredicates::IN_FRONT_OF:	numInFront++;	break;
				case hkcdPlanarGeometryPredicates::INTERSECT:	numSplit++;		break;
				default:	break;
				}
			}
		}

		// Compute heuristic: h(splitPlane) = front - wSplit * split, wSplit = 8
		const int heuristic = numInFront - (numSplit << 3);
		if ( heuristic > bestCost )
		{
			bestCost = heuristic;
			bestPlaneIdx = pi;
		}
	}

	// Return our best estimate
	HK_ASSERT(0x31536947, bestPlaneIdx >= 0);
	return bestPlaneIdx;
}

//
//	Recursively builds the tree for the given set of planes

void hkcdPlanarSolid::buildTree(hkcdPlanarGeometry& polySoup, hkPseudoRandomGenerator& rng,
								hkArray<PlaneId>& srcPlaneIds, const hkArray<PolygonId>& srcPolygonIds, bool useBoundaryPlanes,
								hkcdPlanarEntityDebugger* debugger)
{
	// Type shortcuts
	typedef hkcdBspImpl::Stack		Stack;
	typedef hkcdBspImpl::StackEntry	StackEntry;

	// Preallocate the working memory
	const int numSrcPlanes	= srcPlaneIds.getSize();
	const int numSrcPolys	= srcPolygonIds.getSize();
	hkArray<PolygonId> frontPolyIds;	frontPolyIds.reserve(numSrcPolys);
	hkArray<PolygonId> backPolyIds;		backPolyIds.reserve(numSrcPolys);
	hkArray<PlaneId> frontPlaneIds;		frontPlaneIds.reserve(numSrcPlanes);
	hkArray<PlaneId> backPlaneIds;		backPlaneIds.reserve(numSrcPlanes);
	hkArray<PlaneId> tempPlaneIds;		tempPlaneIds.reserve(numSrcPlanes);

	// Push all original data on the stack
	Stack stack;
	{
		StackEntry entry;
		entry.m_planeIds		= const_cast<PlaneId*>(srcPlaneIds.begin());
		entry.m_polygonIds		= const_cast<PolygonId*>(srcPolygonIds.begin());
		entry.m_numPlaneIds		= numSrcPlanes;
		entry.m_numPolygonIds	= numSrcPolys;
		entry.m_parentNodeId	= NodeId::invalid();
		entry.m_isLeftChild		= false;
		stack.push(entry);
	}

	// While the stack is not empty, pop and process each entry
	while ( !stack.isEmpty() )
	{
		// Pop entry
		StackEntry entry;
		stack.pop(entry);
		HK_ASSERT(0x75217846, entry.m_numPolygonIds && entry.m_numPlaneIds);

		// Pick a splitting plane
		const int splitPlaneIdx		= pickSplittingPlane(polySoup, rng, entry.m_planeIds, entry.m_numPlaneIds, entry.m_polygonIds, entry.m_numPolygonIds);
		const PlaneId splitPlaneId	= entry.m_planeIds[splitPlaneIdx];
		for (int k = splitPlaneIdx + 1; k < entry.m_numPlaneIds; k++)
		{
			entry.m_planeIds[k - 1] = entry.m_planeIds[k];
		}
		entry.m_numPlaneIds--;
		HK_ASSERT(0x57c5b1ea, splitPlaneId.isValid());

		// Classify polygons w.r.t. the splitting plane
		frontPolyIds.setSize(0);	backPolyIds.setSize(0);
		int numSameCoplanar = 0, numOppositeCoplanar = 0;
		for (int k = 0; k < entry.m_numPolygonIds; k++)
		{
			const PolygonId polyId = entry.m_polygonIds[k];			
			const hkcdPlanarGeometryPredicates::Orientation ori = polySoup.classify(polyId, splitPlaneId);

			switch ( ori )
			{
			case hkcdPlanarGeometryPredicates::IN_FRONT_OF:	frontPolyIds.pushBack(polyId);	break;
			case hkcdPlanarGeometryPredicates::BEHIND:		backPolyIds.pushBack(polyId);	break;

			case hkcdPlanarGeometryPredicates::INTERSECT:
				{
					// We need to split the polygon with the plane
					PolygonId splitInside, splitOutside;
					polySoup.split(polyId, splitPlaneId, splitInside, splitOutside);
					HK_ASSERT(0x4a9c399c, splitInside.isValid() && splitOutside.isValid());
					backPolyIds.pushBack(splitInside);
					frontPolyIds.pushBack(splitOutside);
				}
				break;

			default:	// On plane
				{
					const PlaneId polySupportId = polySoup.getPolygon(polyId).getSupportPlaneId();
					if ( hkcdPlanarGeometryPrimitives::sameOrientationPlaneIds(polySupportId, splitPlaneId) )	{	numSameCoplanar++;		}
					else																						{	numOppositeCoplanar++;	}
				}
				break;
			}
		}

		// Allocate a new node
		NodeId nodeId = m_nodes->allocate();
		{
			Node& node			= accessNode(nodeId);
			node.m_left			= NodeId::invalid();
			node.m_right		= NodeId::invalid();
			node.m_typeAndFlags	= NODE_TYPE_INTERNAL;
			node.m_planeId		= splitPlaneId;
			node.m_parent		= entry.m_parentNodeId;

			if ( entry.m_parentNodeId.isValid() )
			{
				Node& parent = accessNode(entry.m_parentNodeId);
				if ( entry.m_isLeftChild )	{	parent.m_left	= nodeId;	}
				else						{	parent.m_right	= nodeId;	}
			}
			else
			{
				m_rootNodeId = nodeId;
			}
		}

		// Gather the left & right planes
		if ( backPolyIds.getSize() )
		{
			tempPlaneIds.setSize(0);				polySoup.getAllPolygonsPlanes(backPolyIds, tempPlaneIds, useBoundaryPlanes);
			backPlaneIds.setSize(numSrcPlanes);		backPlaneIds.setSize(hkAlgorithm::intersectionOfSortedLists(entry.m_planeIds, entry.m_numPlaneIds, tempPlaneIds.begin(), tempPlaneIds.getSize(), backPlaneIds.begin()));
			HK_ASSERT(0xfde56432, backPlaneIds.getSize());
		}
		if ( frontPolyIds.getSize() )
		{
			tempPlaneIds.setSize(0);				polySoup.getAllPolygonsPlanes(frontPolyIds, tempPlaneIds, useBoundaryPlanes);
			frontPlaneIds.setSize(numSrcPlanes);	frontPlaneIds.setSize(hkAlgorithm::intersectionOfSortedLists(entry.m_planeIds, entry.m_numPlaneIds, tempPlaneIds.begin(), tempPlaneIds.getSize(), frontPlaneIds.begin()));
			HK_ASSERT(0xfde56432, frontPlaneIds.getSize());
		}

		// Recurse on left
		if ( backPolyIds.getSize() )
		{
			StackEntry leftEntry;
			leftEntry.m_isLeftChild		= true;
			leftEntry.m_polygonIds		= backPolyIds.begin();
			leftEntry.m_numPolygonIds	= backPolyIds.getSize();
			leftEntry.m_planeIds		= backPlaneIds.begin();
			leftEntry.m_numPlaneIds		= backPlaneIds.getSize();
			leftEntry.m_parentNodeId	= nodeId;
			stack.push(leftEntry);			
		}
		else
		{
			const NodeId leafNode			= ( numSameCoplanar ) ? createInsideNode(nodeId) : createOutsideNode(nodeId);
			accessNode(nodeId).m_left		= leafNode;
		}

		// Recurse on right
		if ( frontPolyIds.getSize() )
		{
			StackEntry rightEntry;
			rightEntry.m_isLeftChild	= false;
			rightEntry.m_polygonIds		= frontPolyIds.begin();
			rightEntry.m_numPolygonIds	= frontPolyIds.getSize();
			rightEntry.m_planeIds		= frontPlaneIds.begin();
			rightEntry.m_numPlaneIds	= frontPlaneIds.getSize();
			rightEntry.m_parentNodeId	= nodeId;
			stack.push(rightEntry);
		}
		else
		{
			const NodeId leafNode			= ( numSameCoplanar ) ? createOutsideNode(nodeId) : createInsideNode(nodeId);
			accessNode(nodeId).m_right		= leafNode;
		}
	}
}

//
//	Builds the tree for the given set of planes, considering the plane as delimiting a flat surface

void hkcdPlanarSolid::buildTree2D(hkcdPlanarGeometry& polySoup, hkPseudoRandomGenerator& rng, hkArray<PlaneId>& srcPlaneIds, const hkArray<PolygonId>& srcPolygonIds, hkcdPlanarEntityDebugger* debugger)
{
	// Type shortcuts
	typedef hkcdBspImpl::Stack		Stack;
	typedef hkcdBspImpl::StackEntry	StackEntry;

	// Preallocate the working memory
	const int numSrcPlanes	= srcPlaneIds.getSize();
	const int numSrcPolys	= srcPolygonIds.getSize();
	hkArray<PolygonId> frontPolyIds;	frontPolyIds.reserve(numSrcPolys);
	hkArray<PolygonId> backPolyIds;		backPolyIds.reserve(numSrcPolys);
	hkArray<PlaneId> frontPlaneIds;		frontPlaneIds.reserve(numSrcPlanes);
	hkArray<PlaneId> backPlaneIds;		backPlaneIds.reserve(numSrcPlanes);

	// Push all original data on the stack
	Stack stack;
	{
		StackEntry entry;
		entry.m_planeIds		= const_cast<PlaneId*>(srcPlaneIds.begin());
		entry.m_polygonIds		= const_cast<PolygonId*>(srcPolygonIds.begin());
		entry.m_numPlaneIds		= numSrcPlanes;
		entry.m_numPolygonIds	= numSrcPolys;
		entry.m_parentNodeId	= NodeId::invalid();
		entry.m_isLeftChild		= false;
		stack.push(entry);
	}

	// While the stack is not empty, pop and process each entry
	while ( !stack.isEmpty() )
	{
		// Pop entry
		StackEntry entry;
		stack.pop(entry);
		HK_ASSERT(0x75217846, entry.m_numPolygonIds && entry.m_numPlaneIds);

		// Pick a splitting plane
		const int splitPlaneIdx		= pickSplittingPlane(polySoup, rng, entry.m_planeIds, entry.m_numPlaneIds, entry.m_polygonIds, entry.m_numPolygonIds);
		const PlaneId splitPlaneId	= entry.m_planeIds[splitPlaneIdx];
		for (int k = splitPlaneIdx + 1; k < entry.m_numPlaneIds; k++)
		{
			entry.m_planeIds[k - 1] = entry.m_planeIds[k];
		}
		entry.m_numPlaneIds--;
		HK_ASSERT(0x57c5b1ea, splitPlaneId.isValid());

		// Classify polygons w.r.t. the splitting plane
		frontPolyIds.setSize(0);	backPolyIds.setSize(0);
		int numSameCoplanar = 0, numOppositeCoplanar = 0;
		for (int k = 0; k < entry.m_numPolygonIds; k++)
		{
			const PolygonId polyId = entry.m_polygonIds[k];
			const hkcdPlanarGeometryPredicates::Orientation ori = polySoup.classify(polyId, splitPlaneId);

			switch ( ori )
			{
			case hkcdPlanarGeometryPredicates::IN_FRONT_OF:	frontPolyIds.pushBack(polyId);	break;
			case hkcdPlanarGeometryPredicates::BEHIND:		backPolyIds.pushBack(polyId);	break;

			case hkcdPlanarGeometryPredicates::INTERSECT:
				{
					// We need to split the polygon with the plane
					PolygonId splitInside, splitOutside;
					polySoup.split(polyId, splitPlaneId, splitInside, splitOutside);
					HK_ASSERT(0x4a9c399c, splitInside.isValid() && splitOutside.isValid());
					backPolyIds.pushBack(splitInside);
					frontPolyIds.pushBack(splitOutside);
				}
				break;

			default:	// On plane
				{
					const PlaneId polySupportId = polySoup.getPolygon(polyId).getSupportPlaneId();
					if ( hkcdPlanarGeometryPrimitives::sameOrientationPlaneIds(polySupportId, splitPlaneId) )	{	numSameCoplanar++;		}
					else																						{	numOppositeCoplanar++;	}
				}
				break;
			}
		}

		// Allocate a new node
		NodeId nodeId = m_nodes->allocate();
		{
			Node& node			= accessNode(nodeId);
			node.m_left			= NodeId::invalid();
			node.m_right		= NodeId::invalid();
			node.m_typeAndFlags	= NODE_TYPE_INTERNAL;
			node.m_planeId		= splitPlaneId;
			node.m_parent		= entry.m_parentNodeId;

			if ( entry.m_parentNodeId.isValid() )
			{
				Node& parent = accessNode(entry.m_parentNodeId);
				if ( entry.m_isLeftChild )	{	parent.m_left	= nodeId;	}
				else						{	parent.m_right	= nodeId;	}
			}
			else
			{
				m_rootNodeId = nodeId;
			}
		}

		// Gather the left & right planes
		backPlaneIds.setSize(0);
		if ( backPolyIds.getSize() )
		{
			// Polygon planes might not match the BSP planes, just pass all!
			backPlaneIds.append(entry.m_planeIds, entry.m_numPlaneIds);
		}
		frontPlaneIds.setSize(0);
		if ( frontPolyIds.getSize() )
		{
			// Polygon planes might not match the BSP planes, just pass all!
			frontPlaneIds.append(entry.m_planeIds, entry.m_numPlaneIds);
		}

		// Recurse on left
		if ( backPlaneIds.getSize() )
		{
			StackEntry leftEntry;
			leftEntry.m_isLeftChild		= true;
			leftEntry.m_polygonIds		= backPolyIds.begin();
			leftEntry.m_numPolygonIds	= backPolyIds.getSize();
			leftEntry.m_planeIds		= backPlaneIds.begin();
			leftEntry.m_numPlaneIds		= backPlaneIds.getSize();
			leftEntry.m_parentNodeId	= nodeId;
			stack.push(leftEntry);			
		}
		else
		{
			const NodeId leafNode			= ( backPolyIds.getSize() ) ? createInsideNode(nodeId) : createOutsideNode(nodeId);
			accessNode(nodeId).m_left		= leafNode;
		}

		// Recurse on right
		if ( frontPlaneIds.getSize() )
		{
			StackEntry rightEntry;
			rightEntry.m_isLeftChild	= false;
			rightEntry.m_polygonIds		= frontPolyIds.begin();
			rightEntry.m_numPolygonIds	= frontPolyIds.getSize();
			rightEntry.m_planeIds		= frontPlaneIds.begin();
			rightEntry.m_numPlaneIds	= frontPlaneIds.getSize();
			rightEntry.m_parentNodeId	= nodeId;
			stack.push(rightEntry);
		}
		else
		{
			const NodeId leafNode			= ( frontPolyIds.getSize() ) ? createInsideNode(nodeId) : createOutsideNode(nodeId);
			accessNode(nodeId).m_right		= leafNode;
		}
	}
}

//
//	Collapses any nodes still marked as unknown

hkBool32 hkcdPlanarSolid::collapseUnknownLabels()
{
	int numCollapses = 0;

	bool found;
	do
	{
		found = false;

		for (int ni = m_nodes->getCapacity() - 1; ni >= 0; ni--)
		{
			const NodeId nodeId(ni);
			Node& bspNode = accessNode(nodeId);

			if ( bspNode.isAllocated() && (bspNode.m_typeAndFlags == NODE_TYPE_INTERNAL) )
			{
				const Node& nodeLeftChild = getNode(bspNode.m_left);
				const Node& nodeRightChild = getNode(bspNode.m_right);

				if ( (nodeLeftChild.m_typeAndFlags == NODE_TYPE_UNKNOWN) || (nodeRightChild.m_typeAndFlags == NODE_TYPE_UNKNOWN) )
				{
					// We can remove this node and link its parent directly to the right child
					const NodeId knownChildNodeId = (nodeLeftChild.m_typeAndFlags == NODE_TYPE_UNKNOWN) ? bspNode.m_right : bspNode.m_left;
					Node& knownChild = accessNode(knownChildNodeId);

					if ( bspNode.m_parent.isValid() )
					{
						Node& parentNode = accessNode(bspNode.m_parent);
						if ( parentNode.m_left == nodeId )	{															parentNode.m_left	= knownChildNodeId;	}
						else								{	HK_ASSERT(0xd0536f1, parentNode.m_right == nodeId);		parentNode.m_right	= knownChildNodeId;	}
					}
					else
					{
						// This is the root node, we can set the root directly to the right child
						m_rootNodeId = knownChildNodeId;		
					}

					// Set the parent
					if ( knownChild.isAllocated() )
					{
						HK_ASSERT(0x606c5428, knownChild.m_parent == nodeId);
						knownChild.m_parent = bspNode.m_parent;
					}

					// We may want to remove the unknown child
					const NodeId unknownChildNodeId = (nodeLeftChild.m_typeAndFlags == NODE_TYPE_UNKNOWN) ? bspNode.m_left : bspNode.m_right;
					m_nodes->release(unknownChildNodeId);

					m_nodes->release(nodeId);
					numCollapses++;
					found = true;
				}
			}
		}
	} while ( found );

	return numCollapses;
}

//
//	Utility functions for the iterative implementation of classifyPolygons. The original recursive algorithm is taken from
//	Thibault's phd thesis.

namespace hkcdBspImpl
{
	/// Polygon category
	enum PolyId
	{
		POLYS_IN	= 0,
		POLYS_ON	= 1,
		POLYS_OUT	= 2,
	};

	/// Job, i.e. classifyPolygons stack entry
	struct Job
	{
		public:

			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY, hkcdBspImpl::Job);
		
		public:

			/// Sets the job as a classify job. It takes one array as input and produces 3 arrays as output, one for each inside / outside / on boundary polygons.
			HK_FORCE_INLINE void setClassify(NodeId nodeId, hkcdPlanarSolid::ArrayId inPolys, const hkcdPlanarSolid::ArrayId (&outPolys)[3])
			{
				HK_ASSERT(0x2df7ea52, nodeId.isValid());
				m_nodeId				= nodeId;
				input()					= inPolys;
				outputs<POLYS_IN>()		= outPolys[POLYS_IN];
				outputs<POLYS_ON>()		= outPolys[POLYS_ON];
				outputs<POLYS_OUT>()	= outPolys[POLYS_OUT];
			}

			/// Sets the job as a merge job. It takes 3 arrays as input and produces one merged array as output.
			HK_FORCE_INLINE void setMerge(hkcdPlanarSolid::ArrayId inPolysA, hkcdPlanarSolid::ArrayId inPolysB, hkcdPlanarSolid::ArrayId inPolysC, hkcdPlanarSolid::ArrayId outPolys)
			{
				m_nodeId			= NodeId::invalid();
				output()			= outPolys;
				inputs<POLYS_IN>()	= inPolysA;
				inputs<POLYS_ON>()	= inPolysB;
				inputs<POLYS_OUT>()	= inPolysC;
			}

		public:

			// Accessors for a classify job
			HK_FORCE_INLINE hkcdPlanarSolid::ArrayId& input()	{	return m_arrays[0];		}
			template <PolyId T>
			HK_FORCE_INLINE hkcdPlanarSolid::ArrayId& outputs()	{	return m_arrays[1 + T];	}

			// Accessors for a merge job
			template <PolyId T>
			HK_FORCE_INLINE hkcdPlanarSolid::ArrayId& inputs()	{	return m_arrays[T];		}
			HK_FORCE_INLINE hkcdPlanarSolid::ArrayId& output()	{	return m_arrays[3];		}

		public:

			NodeId m_nodeId;						///< The Bsp tree node Id.
			hkcdPlanarSolid::ArrayId m_arrays[4];	///< The inputs / outputs.
	};

	struct MatPoly
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY, hkcdBspImpl::MatPoly);

		HK_FORCE_INLINE static hkBool32 HK_CALL less(const MatPoly& a, const MatPoly& b) { return a.m_matId < b.m_matId; }

		hkUint32 m_matId;
		PolygonId m_polyId;		
	};
}

//
//	Subtracts the solid from the given polygon and returns a list of polygons outside / on the boundary of the solid

void hkcdPlanarSolid::classifyPolygons(	hkcdPlanarGeometry& polySoup, NodeId rootNodeId, const hkArray<PolygonId>& origPolygonsIn,
										hkArray<PolygonId>& insidePolygonsOut, hkArray<PolygonId>& boundaryPolygonsOut, hkArray<PolygonId>& outsidePolygonsOut, ArrayMgr& polyStorage) const
{
	typedef hkcdBspImpl::Job			Job;

#if HK_CONFIG_MONITORS == HK_CONFIG_MONITORS_ENABLED
	hkMonitorStream& timerStream = hkMonitorStream::getInstance();
#endif	

	// Early-out if no polygons provided
	if ( !origPolygonsIn.getSize() )
	{
		insidePolygonsOut.setSize(0);
		outsidePolygonsOut.setSize(0);
		boundaryPolygonsOut.setSize(0);
		return;
	}

	HK_TIMER_BEGIN2(timerStream, "Inits", HK_NULL);
	hkArray<Job> jobStack;
	jobStack.reserve(64);

	// Allocate outputs and push root node on the stack
	const ArrayId aidPolys[3]	= { polyStorage.allocArraySlot(), polyStorage.allocArraySlot(), polyStorage.allocArraySlot() };
	const ArrayId nullPolys		= polyStorage.allocArraySlot();
	{
		Job& job = jobStack.expandOne();
		job.setClassify(rootNodeId, polyStorage.allocArraySlot(), aidPolys);
		polyStorage.allocArrayStorage(job.input(), origPolygonsIn);
	}
	HK_TIMER_END()

	hkArray<PolygonId> tempPolys[3];
	while ( jobStack.getSize() )
	{
		// Pop last job on the stack
		Job job;
		{
			const int jobIdx = jobStack.getSize() - 1;
			job = jobStack[jobIdx];
			jobStack.removeAt(jobIdx);
		}

		// Process it
		if ( job.m_nodeId.isValid() )
		{
			// CLASSIFY job. Get current node
			const NodeId nodeId			= job.m_nodeId;
			const Node& bspNode			= (*m_nodes)[nodeId];
			const ArrayId jobInput		= job.input();
			const ArraySlot inputPolys	= polyStorage.getArraySlot(jobInput);

			switch ( bspNode.m_typeAndFlags )
			{
			case NODE_TYPE_OUT:	{	polyStorage.getArraySlot(job.outputs<hkcdBspImpl::POLYS_OUT>())	= inputPolys;	}	continue;
			case NODE_TYPE_IN:	{	polyStorage.getArraySlot(job.outputs<hkcdBspImpl::POLYS_IN>())	= inputPolys;	}	continue;
			default:	break;	// Internal node, recurse!
			}

			// Classify our polygons w.r.t. the node splitting plane
			const PlaneId splitPlaneId	= bspNode.m_planeId;
			tempPolys[hkcdBspImpl::POLYS_IN].setSize(0);
			tempPolys[hkcdBspImpl::POLYS_ON].setSize(0);
			tempPolys[hkcdBspImpl::POLYS_OUT].setSize(0);

			HK_TIMER_BEGIN2(timerStream, "Predicates", HK_NULL);
			for (int k = (int)inputPolys.m_size - 1; k >= 0; k--)
			{
				const PolygonId polyId	= polyStorage.getPolygonId(jobInput, k);
				const Orientation ori	= polySoup.classify(polyId, splitPlaneId);

				switch ( ori )
				{
				case hkcdPlanarGeometryPredicates::IN_FRONT_OF:	tempPolys[hkcdBspImpl::POLYS_OUT].pushBack(polyId);	break;
				case hkcdPlanarGeometryPredicates::BEHIND:		tempPolys[hkcdBspImpl::POLYS_IN].pushBack(polyId);	break;
				case hkcdPlanarGeometryPredicates::INTERSECT:
					{
						// We need to split the polygon into 2 parts
						PolygonId splitInside, splitOutside;
						polySoup.split(polyId, splitPlaneId, splitInside, splitOutside);
						HK_ASSERT(0x4a9bf46d, splitInside.isValid() && splitOutside.isValid());

						tempPolys[hkcdBspImpl::POLYS_IN].pushBack(splitInside);
						tempPolys[hkcdBspImpl::POLYS_OUT].pushBack(splitOutside);
					}
					break;
				default:	
					{
						HK_ASSERT(0x53da54ac, ori == hkcdPlanarGeometryPredicates::ON_PLANE);
						tempPolys[hkcdBspImpl::POLYS_ON].pushBack(polyId);
					}
					break;
				}
			}
			HK_TIMER_END();

			// Free input
			polyStorage.freeArray(jobInput);

			// Allocate output arrays
			const ArrayId polys_L[3]		= { polyStorage.allocArraySlot(), polyStorage.allocArraySlot(), polyStorage.allocArraySlot() };
			const ArrayId polys_R[3]		= { polyStorage.allocArraySlot(), polyStorage.allocArraySlot(), polyStorage.allocArraySlot() };
			const ArrayId coplanar_L[3]		= { polyStorage.allocArraySlot(), polyStorage.allocArraySlot(), polyStorage.allocArraySlot() };
			const ArrayId coplanar_InL[3]	= { polyStorage.allocArraySlot(), polyStorage.allocArraySlot(), polyStorage.allocArraySlot() };
			const ArrayId coplanar_OutL[3]	= { polyStorage.allocArraySlot(), polyStorage.allocArraySlot(), polyStorage.allocArraySlot() };

			// Merge all into the job outputs
			if ( inputPolys.m_size )
			{
				// Merge coplanar polygons
				const ArrayId tempMergedArrayA = polyStorage.allocArraySlot();
				const ArrayId tempMergedArrayB = polyStorage.allocArraySlot();
				Job* jobs = jobStack.expandBy(5);

				jobs[0].setMerge(polys_L[hkcdBspImpl::POLYS_IN], polys_R[hkcdBspImpl::POLYS_IN], coplanar_InL[hkcdBspImpl::POLYS_IN], job.outputs<hkcdBspImpl::POLYS_IN>());
				jobs[1].setMerge(polys_L[hkcdBspImpl::POLYS_OUT], polys_R[hkcdBspImpl::POLYS_OUT], coplanar_OutL[hkcdBspImpl::POLYS_OUT], job.outputs<hkcdBspImpl::POLYS_OUT>());
				jobs[2].setMerge(tempMergedArrayA, tempMergedArrayB, nullPolys, job.outputs<hkcdBspImpl::POLYS_ON>());
				jobs[3].setMerge(polys_L[hkcdBspImpl::POLYS_ON], polys_R[hkcdBspImpl::POLYS_ON], coplanar_InL[hkcdBspImpl::POLYS_OUT], tempMergedArrayA);
				jobs[4].setMerge(coplanar_InL[hkcdBspImpl::POLYS_ON], coplanar_OutL[hkcdBspImpl::POLYS_ON], coplanar_OutL[hkcdBspImpl::POLYS_IN], tempMergedArrayB);
			}

			// Recurse on left
			if ( tempPolys[hkcdBspImpl::POLYS_IN].getSize() )
			{
				Job& jLeft = jobStack.expandOne();
				jLeft.setClassify(bspNode.m_left, polyStorage.allocArraySlot(), polys_L);
				polyStorage.allocArrayStorage(jLeft.input(), tempPolys[hkcdBspImpl::POLYS_IN]);
			}

			// Recurse on right
			if ( tempPolys[hkcdBspImpl::POLYS_OUT].getSize() )
			{
				Job& jRight = jobStack.expandOne();
				jRight.setClassify(bspNode.m_right, polyStorage.allocArraySlot(), polys_R);
				polyStorage.allocArrayStorage(jRight.input(), tempPolys[hkcdBspImpl::POLYS_OUT]);
			}

			// Recurse on coplanar
			if ( tempPolys[hkcdBspImpl::POLYS_ON].getSize() )
			{
				Job* jobs = jobStack.expandBy(3);
				jobs[0].setClassify(bspNode.m_right, coplanar_L[hkcdBspImpl::POLYS_IN], coplanar_InL);
				jobs[1].setClassify(bspNode.m_right, coplanar_L[hkcdBspImpl::POLYS_OUT], coplanar_OutL);
				jobs[2].setClassify(bspNode.m_left, polyStorage.allocArraySlot(), coplanar_L);
				polyStorage.allocArrayStorage(jobs[2].input(), tempPolys[hkcdBspImpl::POLYS_ON]);
			}
		}
		else
		{
			// Merge
			const ArrayId srcIdA	= job.inputs<hkcdBspImpl::POLYS_IN>();
			const ArrayId srcIdB	= job.inputs<hkcdBspImpl::POLYS_ON>();
			const ArrayId srcIdC	= job.inputs<hkcdBspImpl::POLYS_OUT>();

			const int numPolysA		= polyStorage.getArraySlot(srcIdA).m_size;
			const int numPolysB		= polyStorage.getArraySlot(srcIdB).m_size;
			const int numPolysC		= polyStorage.getArraySlot(srcIdC).m_size;
			const int numMerged		= numPolysA + numPolysB + numPolysC;

			hkArray<PolygonId>& mergedPolys = tempPolys[0];
			mergedPolys.setSize(numMerged);
			if ( numMerged )
			{
				PolygonId* HK_RESTRICT ptr = &mergedPolys[0];
				for (int si = numPolysA - 1; si >= 0; si--)	{	ptr[si]	= polyStorage.getPolygonId(srcIdA, si);	}	ptr += numPolysA;
				for (int si = numPolysB - 1; si >= 0; si--)	{	ptr[si] = polyStorage.getPolygonId(srcIdB, si);	}	ptr += numPolysB;
				for (int si = numPolysC - 1; si >= 0; si--)	{	ptr[si] = polyStorage.getPolygonId(srcIdC, si);}
			}

			// Free inputs
			if ( srcIdA != nullPolys )	{	polyStorage.freeArray(srcIdA);	}
			if ( srcIdB != nullPolys )	{	polyStorage.freeArray(srcIdB);	}
			if ( srcIdC != nullPolys )	{	polyStorage.freeArray(srcIdC);	}

			// Write output
			if ( numMerged )
			{
				polyStorage.allocArrayStorage(job.output(), mergedPolys);
			}
		}
	}

	// Copy results to the caller arrays
	{
		const int numPolysIn	= polyStorage.getArraySlot(aidPolys[hkcdBspImpl::POLYS_IN]).m_size;
		const int numPolysOut	= polyStorage.getArraySlot(aidPolys[hkcdBspImpl::POLYS_OUT]).m_size;
		const int numPolysOn	= polyStorage.getArraySlot(aidPolys[hkcdBspImpl::POLYS_ON]).m_size;

		insidePolygonsOut.setSize(numPolysIn);
		outsidePolygonsOut.setSize(numPolysOut);
		boundaryPolygonsOut.setSize(numPolysOn);

		for (int si = numPolysIn - 1; si >= 0; si--)	{	insidePolygonsOut[si]	= polyStorage.getPolygonId(aidPolys[hkcdBspImpl::POLYS_IN], si);	}
		for (int si = numPolysOut - 1; si >= 0; si--)	{	outsidePolygonsOut[si]	= polyStorage.getPolygonId(aidPolys[hkcdBspImpl::POLYS_OUT], si);	}
		for (int si = numPolysOn - 1; si >= 0; si--)	{	boundaryPolygonsOut[si]	= polyStorage.getPolygonId(aidPolys[hkcdBspImpl::POLYS_ON], si);	}
	}

	// Try simplifying the results by replacing cut polygons with the original ones when possible
	HK_TIMER_BEGIN2(timerStream, "PolysRestore", HK_NULL);
	{
		typedef hkcdBspImpl::MatPoly	MatPoly;

		// Collect all material ids from the original geometry
		const int numOrigPolys = origPolygonsIn.getSize();
		hkArray<MatPoly> origMatToPoly;
		const int NB_TABLES = 3;
		PolygonId* inTables[NB_TABLES];
		MatPoly* outTables[NB_TABLES];
		hkArray<PolygonId>* outArrays[NB_TABLES];

		int inTableSizes[NB_TABLES];
		origMatToPoly.setSize(numOrigPolys);
		hkArray<MatPoly> inMatToPoly;						hkArray<MatPoly> outMatToPoly;					hkArray<MatPoly> boundMatToPoly;
		inTables[0] = boundaryPolygonsOut.begin();			inTables[1] = insidePolygonsOut.begin();		inTables[2] = outsidePolygonsOut.begin(); 
		inTableSizes[0] = boundaryPolygonsOut.getSize();	inTableSizes[1] = insidePolygonsOut.getSize();	inTableSizes[2] = outsidePolygonsOut.getSize();
		boundMatToPoly.setSize(inTableSizes[0]);			inMatToPoly.setSize(inTableSizes[1]);			outMatToPoly.setSize(inTableSizes[2]);	
		outTables[0] = boundMatToPoly.begin();				outTables[1] = inMatToPoly.begin();				outTables[2] = outMatToPoly.begin();
		outArrays[0] = &boundaryPolygonsOut;				outArrays[1] = &insidePolygonsOut;				outArrays[2] = &outsidePolygonsOut;

		// Fill all the MatPoly structures
		for (int k = origPolygonsIn.getSize() - 1 ; k >= 0 ; k--)
		{
			PolygonId pId = origPolygonsIn[k];
			const Polygon& poly = polySoup.getPolygon(pId);
			origMatToPoly[k].m_polyId = pId;
			origMatToPoly[k].m_matId = poly.getMaterialId();
		}

		for (int i = 0 ; i < NB_TABLES ; i++)
		{
			for (int k = 0 ; k < inTableSizes[i] ; k++)
			{
				outTables[i][k].m_polyId	= (*outArrays[i])[k];
				const Polygon& poly		= polySoup.getPolygon(outTables[i][k].m_polyId);
				outTables[i][k].m_matId	= poly.getMaterialId();
			}
		}

		boundaryPolygonsOut.clear();
		insidePolygonsOut.clear();
		outsidePolygonsOut.clear();

		// Sort all the results by material, and initialize variables for sweeping
		hkSort(origMatToPoly.begin(), origMatToPoly.getSize(), MatPoly::less);
		int cursorPos[NB_TABLES];
		int cursorOriginal = 0;
		bool tableFinished[NB_TABLES];
		bool sweepingFinished = true;
		int maxTableSize = 0;
		for (int i = 0 ; i < NB_TABLES ; i++)
		{
			hkSort(outTables[i], inTableSizes[i], MatPoly::less);
			cursorPos[i]		= 0;
			tableFinished[i]	= cursorPos[i] >= inTableSizes[i];
			sweepingFinished	= sweepingFinished && tableFinished[i];
			maxTableSize		= hkMath::max2(maxTableSize, inTableSizes[i]);
		}

		// Sweep the three arrays and try to replace group of polygons of the same material with original ones
		while ( !sweepingFinished )
		{
			// Get the smallest value at cursor index
			int ixMin = -1;
			for (int t = 0 ; t < NB_TABLES ; t++)
			{
				if ( !tableFinished[t] )
				{
					ixMin = t;
					break;
				}
			}

			for (int t = 0 ; t < NB_TABLES ; t++)
			{
				ixMin = ( !tableFinished[t] && (outTables[t][cursorPos[t]].m_matId < outTables[ixMin][cursorPos[ixMin]].m_matId) ) ? t : ixMin;
			}
			const hkUint32 cMat = outTables[ixMin][cursorPos[ixMin]].m_matId;

			// Move the cursor of the original array to reach the material
			while ( origMatToPoly[cursorOriginal].m_matId != cMat )
			{
				cursorOriginal++;
			}

			// See if the corresponding material is material isolated in only one part		
			bool isolated		= true;
			for (int t = 0 ; t < NB_TABLES ; t++)
			{
				if ( tableFinished[t] )
				{
					continue;
				}

				if ( (t != ixMin) && (cMat == outTables[t][cursorPos[t]].m_matId) )
				{
					// found another array containing same material: material is not isolated
					isolated = false;
					break;
				}
			}

			if ( isolated )
			{
				// isolated material, all the cut polygons can be replaced by the original ones!
				while ( (cursorOriginal < numOrigPolys) && (origMatToPoly[cursorOriginal].m_matId == cMat) ) 
				{
					outArrays[ixMin]->pushBack(origMatToPoly[cursorOriginal].m_polyId);
					cursorOriginal++;
				}
			}

			// Update all cursors and finish state
			sweepingFinished = true;
			for (int t = 0 ; t < NB_TABLES ; t++)
			{
				if ( tableFinished[t] )
				{
					continue;
				}

				while ( (cursorPos[t] < inTableSizes[t]) && (outTables[t][cursorPos[t]].m_matId == cMat) ) 
				{
					if ( !isolated )
					{
						outArrays[t]->pushBack(outTables[t][cursorPos[t]].m_polyId);
					}
					cursorPos[t]++;
				}
				tableFinished[t]	= cursorPos[t] >= inTableSizes[t];
				sweepingFinished	= sweepingFinished && tableFinished[t];
			}
		}
	}
	HK_TIMER_END();
}

//
//	Special case of the classifies, where only the inside OR boundary polys are needed

void hkcdPlanarSolid::classifyInsideOrBoundaryPolygons(hkcdPlanarGeometry& polySoup, const hkArray<PolygonId>& polygonsIn, hkArray<PolygonId>& insideOrBoundPolyIds, ArrayMgr* arrayMgr)
{
	hkArray< hkArray<PlaneId> > solidCellsPlaneIds;

	// First, get all the solid nodes
	int biggestArrayId = -1;
	int numPlanesinBiggestArray = -1;
	for (int k = m_nodes->getCapacity() - 1; k >= 0; k--)
	{
		NodeId nodeId(k);
		Node* node = &accessNode(nodeId);
		if ( node->isAllocated() )
		{
			if ( node->m_typeAndFlags == NODE_TYPE_IN )
			{
				hkArray<PlaneId>& solidCellPlaneIds = solidCellsPlaneIds.expandOne();
				// We got one, collect all the planes that form the convex solid cell		
				int nbPlanes = 0;
				while ( node->m_parent.isValid() )
				{
					node = &accessNode(node->m_parent);
					nbPlanes++;
				}
				if ( nbPlanes > numPlanesinBiggestArray )
				{
					numPlanesinBiggestArray	= nbPlanes;
					biggestArrayId			= solidCellsPlaneIds.getSize() - 1;
				}
				solidCellPlaneIds.reserve(nbPlanes);
				node = &accessNode(nodeId);
				NodeId prevNodeId = nodeId;
				while ( node->m_parent.isValid() )
				{
					const NodeId parentId	= node->m_parent;
					node					= &accessNode(parentId);
					if ( node->m_left == prevNodeId )
					{
						solidCellPlaneIds.pushBack(node->m_planeId);						
					}
					else
					{
						solidCellPlaneIds.pushBack(hkcdPlanarGeometryPrimitives::getOppositePlaneId(node->m_planeId));
					}
					prevNodeId = parentId;
				}
			}
		}
	}

	const int numSolidCells = solidCellsPlaneIds.getSize();
	if ( numSolidCells == 0 )
	{
		return;
	}

	// Compute a common plane path
	if ( numSolidCells > 1 )
	{
		hkArray<PlaneId> commonPlanePath;
		commonPlanePath.append(solidCellsPlaneIds[biggestArrayId]);
		for (int i = 0 ; i < numSolidCells ; i++)
		{
			if ( i == biggestArrayId )		continue;
			// Find the common planes (starting from the end)
			hkArray<PlaneId>& currPlaneArray = solidCellsPlaneIds[i];
			int numCommonPlanes = 0;
			for (int p = currPlaneArray.getSize() - 1, pRef = commonPlanePath.getSize() - 1 ; p >= 0 && pRef >= 0 ; p--, pRef--)
			{
				if ( currPlaneArray[p].value() == commonPlanePath[p].value() )
				{
					numCommonPlanes++;
				}
				else
				{
					break;
				}
			}
			commonPlanePath.setSize(numCommonPlanes);
		}
		solidCellsPlaneIds.pushBack(commonPlanePath);
	}

	// Classify all the polys against the common plane path, then against all solid cells
	hkArray<PolygonId> polysOnCommonPath, currInsidePolyIds, nextInsidePolyIds;
	insideOrBoundPolyIds.setSize(0);
	insideOrBoundPolyIds.reserve(polygonsIn.getSize()*2);
	nextInsidePolyIds.reserve(polygonsIn.getSize()*2);
	currInsidePolyIds.append(polygonsIn);

	for (int sc = solidCellsPlaneIds.getSize() - 1 ; sc >= 0 ; sc--)
	{
		hkArray<PlaneId>& currPlaneArray = solidCellsPlaneIds[sc];
		if ( sc != solidCellsPlaneIds.getSize() - 1 )
		{
			currInsidePolyIds.setSize(0);
			currInsidePolyIds.append(polysOnCommonPath);
		}
		for (int p = currPlaneArray.getSize() - 1 ; p >= 0 ; p--)
		{
			for (int polyIdx = currInsidePolyIds.getSize() - 1 ; polyIdx >= 0 ; polyIdx--)
			{
				const PolygonId polyId	= currInsidePolyIds[polyIdx];
				const Orientation ori	= polySoup.classify(polyId, currPlaneArray[p]);

				switch ( ori )
				{
				case hkcdPlanarGeometryPredicates::IN_FRONT_OF:	break;
				case hkcdPlanarGeometryPredicates::BEHIND:		nextInsidePolyIds.pushBack(polyId);	break;
				case hkcdPlanarGeometryPredicates::INTERSECT:
					{
						// We need to split the polygon into 2 parts
						PolygonId splitInside, splitOutside;
						polySoup.split(polyId, currPlaneArray[p], splitInside, splitOutside);
						HK_ASSERT(0x4a9bf47e, splitInside.isValid() && splitOutside.isValid());

						nextInsidePolyIds.pushBack(splitInside);
					}
					break;
				default:	
					{
						HK_ASSERT(0x53da54bd, ori == hkcdPlanarGeometryPredicates::ON_PLANE);
						nextInsidePolyIds.pushBack(polyId);		// Add on boundary as well
					}
					break;
				}
			}

			// Swap poly ids
			currInsidePolyIds.swap(nextInsidePolyIds);
			nextInsidePolyIds.setSize(0);
		}

		// Add the polys to the results
		if ( sc != solidCellsPlaneIds.getSize() - 1 || solidCellsPlaneIds.getSize() == 1 )
		{
			insideOrBoundPolyIds.append(currInsidePolyIds);
		}
		else
		{
			polysOnCommonPath.append(currInsidePolyIds);
		}
	}

}

//
//	Computes a set of polygons that cover the boundary of the solid

void hkcdPlanarSolid::computeBoundary(hkcdConvexCellsTree3D* cellGraph, hkcdPlanarGeometry& boundaryGeomOut, hkcdPlanarEntityDebugger* debugger)
{
	// Create a convex cells tree to manage the cuts
	typedef hkcdConvexCellsTree3D::CellId		CellId;

	boundaryGeomOut.setPlanesCollection(const_cast<PlanesCollection*>(m_planes.val()), HK_NULL);
	
	if ( cellGraph->hasManifoldCells() )
	{
		// Extract the boundary
		hkArray<CellId> solidCellIds;
		cellGraph->collectSolidCells(solidCellIds);
		hkArray<PolygonId> boundaryPolygonsIds;
		cellGraph->extractBoundaryPolygonsFromCellIds(solidCellIds, boundaryGeomOut, boundaryPolygonsIds);
	}
	else	
	{		
		typedef hkcdConvexCellsTree3D::Cell		Cell;
		// Get the solid cells
		hkArray<CellId> solidCellIds;
		cellGraph->collectSolidCells(solidCellIds);

		// Collect polygons on the solid cells' boundaries
		hkArray<PolygonId> cellPolys;
		cellGraph->getUniquePolygonIdsFromCellIds(solidCellIds, cellPolys);

		// Classify all cells' polygons against the BSP tree
		{
			hkArray<PolygonId> insidePolys, boundaryPolys, outsidePolys;
			hkArray<PolygonId> geomOutPolys;
			boundaryGeomOut.appendGeometryPolygons(*cellGraph->getGeometry(), cellPolys, false, geomOutPolys);
			classifyPolygons(boundaryGeomOut, geomOutPolys, insidePolys, boundaryPolys, outsidePolys);
			cellPolys.swap(boundaryPolys);
		}

		// Remove all other polygons from the mesh
		hkSort(cellPolys.begin(), cellPolys.getSize());
		boundaryGeomOut.keepPolygons(cellPolys);
	}
}

//
//	Returns the maximum depth of the tree

int hkcdPlanarSolid::computeMaxDepth() const
{
	int maxDepth = 0;
	for (int k = m_nodes->getCapacity() - 1; k >= 0; k--)
	{
		NodeId leafId			(k);
		const Node& leafNode	= getNode(leafId);

		if ( leafNode.isAllocated() )
		{
			if ( (leafNode.m_typeAndFlags != NODE_TYPE_INTERNAL) ||
				!leafNode.m_left.isValid() || (getNode(leafNode.m_left).m_typeAndFlags == NODE_TYPE_INTERNAL) ||
				!leafNode.m_right.isValid() || (getNode(leafNode.m_right).m_typeAndFlags == NODE_TYPE_INTERNAL) )
			{
				continue;
			}

			// This is a leaf, compute depth
			int depth = 0;
			while ( leafId.isValid() )
			{
				leafId = getNode(leafId).m_parent;
				depth++;
			}
			maxDepth = hkMath::max2(depth, maxDepth);
		}
	}

	return maxDepth;
}

//
//	Computes the number of leaf nodes

int hkcdPlanarSolid::computeNumLeafNodes() const
{
	int sum = 0;
	for (int ni = m_nodes->getCapacity() - 1; ni >= 0; ni--)
	{
		const NodeId nodeId(ni);
		const Node& bspNode = getNode(nodeId);
		if ( bspNode.isAllocated() && (bspNode.m_typeAndFlags != NODE_TYPE_INTERNAL) )
		{
			sum++;
		}
	}

	return sum;
}

//
//	Computes the number of nodes with the specified label
	
int hkcdPlanarSolid::computeNumNodesWithLabel(hkUint32 label) const
{
	int sum = 0;
	for (int ni = m_nodes->getCapacity() - 1; ni >= 0; ni--)
	{
		const NodeId nodeId(ni);
		const Node& bspNode = getNode(nodeId);
		if ( bspNode.isAllocated() && (bspNode.m_typeAndFlags == label) )
		{
			sum++;
		}
	}

	return sum;
}

//
//	Debug. Prints all the tree data

void hkcdPlanarSolid::dbgPrint() const
{
	hkStringBuf strb;

	for (int ni = 0; ni < m_nodes->getCapacity(); ni++)
	{
		const NodeId nodeId	(ni);
		const Node& node	= getNode(nodeId);
		const int planeIdx	= node.m_planeId.isValid() ? ((node.m_planeId.valueUnchecked() & (~hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG)) - 6) : node.m_planeId.valueUnchecked();
		const PlaneId pid	= node.m_planeId.isValid() ? PlaneId((node.m_planeId.valueUnchecked() & hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_FLAG) | planeIdx) : node.m_planeId;
	
		strb.printf("Node %d. Parent %d, Left %d, Right %d, PlaneId = %d (%d), Data = %d, Type = 0x%08X", 
			ni, node.m_parent.valueUnchecked(), node.m_left.valueUnchecked(), node.m_right.valueUnchecked(),
			pid.valueUnchecked(), planeIdx,
			node.m_data, node.m_typeAndFlags);
		HK_REPORT(strb);
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
