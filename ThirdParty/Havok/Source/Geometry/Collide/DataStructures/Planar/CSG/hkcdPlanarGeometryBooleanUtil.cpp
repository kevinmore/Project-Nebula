/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Geometry/Collide/hkcdCollide.h>
#include <Geometry/Collide/DataStructures/Planar/CSG/hkcdPlanarGeometryBooleanUtil.h>
#include <Geometry/Collide/DataStructures/Planar/Utils/hkcdPlanarGeometryWeldUtil.h>
#include <Geometry/Collide/DataStructures/Planar/ConvexCellsTree/hkcdConvexCellsTree3D.h>
#include <Geometry/Collide/DataStructures/Planar/ConvexCellsTree/hkcdConvexCellsTree2D.h>

#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/Base/Container/String/hkStringBuf.h>

//
//	Debug flags

#define ENABLE_TEXT_DEBUG	(0)

//
//	Types

typedef hkcdPlanarGeometryPlanesCollection	PlanesCollection;

//
//	Implementation

struct hkcdBspBooleanImpl : hkcdPlanarGeometryBooleanUtil::BooleanState
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY, hkcdBspBooleanImpl);

		// Types
		typedef hkcdPlanarGeometry::Plane					Plane;
		typedef hkcdPlanarGeometry::PlaneId					PlaneId;
		typedef hkcdPlanarGeometry::Polygon					Polygon;
		typedef hkcdPlanarGeometry::PolygonId				PolygonId;
		typedef hkcdPlanarSolid::Node						Node;
		typedef hkcdPlanarSolid::NodeId						NodeId;
		typedef hkcdPlanarSolid::NodeTypes					NodeType;
		typedef hkcdConvexCellsTree3D::Cell					Cell;
		typedef hkcdConvexCellsTree3D::CellId				CellId;
		typedef hkcdPlanarGeometryBooleanUtil::Operation	Operation;
		typedef hkcdPlanarGeometryPredicates::Orientation	Orientation;
		typedef hkcdPlanarCsgOperand						CsgOperand;

	public:

		/// Stack entry, for the iterative implementation of the BSP tree merge
		struct StackEntry
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_GEOMETRY, hkcdBspBooleanImpl::StackEntry);

			/// Flags
			enum
			{
				IDX_TREE_A_BIT		= hkcdPlanarGeometryPrimitives::FLIPPED_PLANE_BIT,	///< The index in m_trees of the tree acting as treeA.
				IS_RIGHT_CHILD_BIT	= IDX_TREE_A_BIT + 1,								///< The bit is set if the stack entry is the right child of m_parentNodeId A / B.
				COMPUTE_RESULT_A	= IS_RIGHT_CHILD_BIT + 1,							///< The bit is set if m_result[0] should be computed
				COMPUTE_RESULT_B	= COMPUTE_RESULT_A + 1,								///< The bit is set if m_result[1] should be computed
			};

			/// Sets the stack entry
			HK_FORCE_INLINE void set(	NodeId nodeIdA, NodeId nodeIdB,
										NodeId parentNodeIdA, NodeId parentNodeIdB,
										int idxTreeA, int childIdx, int computeResultA, int computeResultB, CellId regionId)
			{
				m_nodeIdA		= nodeIdA.valueUnchecked();
				m_nodeIdB		= nodeIdB.valueUnchecked();
				m_parentNodeIdA	= parentNodeIdA.valueUnchecked();
				m_parentNodeIdB	= parentNodeIdB.valueUnchecked();
				m_regionId		= regionId.valueUnchecked();
				m_flags =	(idxTreeA << IDX_TREE_A_BIT)		|
							(childIdx << IS_RIGHT_CHILD_BIT)	|
							(computeResultA << COMPUTE_RESULT_A)|
							(computeResultB << COMPUTE_RESULT_B);
			}

			HK_FORCE_INLINE NodeId getNodeIdA()			const	{	return NodeId(m_nodeIdA);		}
			HK_FORCE_INLINE NodeId getNodeIdB()			const	{	return NodeId(m_nodeIdB);		}
			HK_FORCE_INLINE NodeId getParentNodeIdA()	const	{	return NodeId(m_parentNodeIdA);	}
			HK_FORCE_INLINE NodeId getParentNodeIdB()	const	{	return NodeId(m_parentNodeIdB);	}
			HK_FORCE_INLINE CellId getRegionId()		const	{	return CellId(m_regionId & hkcdConvexCellsCollection::PAYLOAD_MASK);	}
			HK_FORCE_INLINE int getIdxTreeA()			const	{	return (m_flags >> IDX_TREE_A_BIT) & 1;		}
			HK_FORCE_INLINE int getChildIdx()			const	{	return (m_flags >> IS_RIGHT_CHILD_BIT) & 1;	}
			HK_FORCE_INLINE int isEnabledResultA()		const	{	return (m_flags >> COMPUTE_RESULT_A) & 1;	}
			HK_FORCE_INLINE int isEnabledResultB()		const	{	return (m_flags >> COMPUTE_RESULT_B) & 1;	}

		protected:

			hkUint32 m_nodeIdA;
			hkUint32 m_nodeIdB;
			hkUint32 m_parentNodeIdA;
			hkUint32 m_parentNodeIdB;
			hkUint32 m_regionId;
			hkUint32 m_flags;
		};

	public:

		// Constructor
		HK_FORCE_INLINE hkcdBspBooleanImpl()
		:	m_operandToMerge(HK_NULL)
		,	m_regions(HK_NULL)
		,	m_regions2d(HK_NULL)
		{
			m_result[0] = HK_NULL;
			m_result[1] = HK_NULL;
		}

		// Dummy assignment operator, to silence the compiler
		HK_FORCE_INLINE void operator=(const hkcdBspBooleanImpl&) {}

		// -------------------------------------------------------------------------------------------------------------
		// 3D merge
		// -------------------------------------------------------------------------------------------------------------
		
		/// Initializes the boolean
		void setOperands(const CsgOperand* treeA, const CsgOperand* treeB, int neededTrees, Operation op, bool manifoldCells, hkcdPlanarGeometryBooleanUtil::PolyCollManager* polysCollManager, hkcdPlanarGeometryBooleanUtil::CellCollManager* cellsCollManager);
		void setOperatorsForInPlaceMerge(CsgOperand* inputOutputTree, hkcdPlanarGeometry& commonGeom);

		// Starts merging the trees
		void merge(Operation op, bool computeOp, bool computeDualOp, bool simplifyResults = true);

		/// Merge a new tree into the existing one this merge perform an in-place operation as it will modify the input provided in setOperatorsForInPlaceMerge.
		HK_FORCE_INLINE void mergeInPlace(const CsgOperand* operandToMergeWith, Operation op, bool collapse = true);

		/// Remove all the regions added to the working planar during the merges
		void finalizeInPlaceMerge(CsgOperand& operandInOut, bool collapse = false);

		// -------------------------------------------------------------------------------------------------------------
		// 2D merge
		// -------------------------------------------------------------------------------------------------------------
		
		/// Initializes the boolean 2D
		void setOperatorsForInPlaceMerge2d(CsgOperand* inputOutputTree, hkcdPlanarGeometry& commonGeom, PlaneId supportPlaneId);

		/// Merges the input tree with the current result tree. WARNING: unlike its 3D counterpart, 
		HK_FORCE_INLINE void mergeInPlace2d(const CsgOperand* operandToMergeWith, Operation op, bool collapse = true);

		/// Remove all the regions added to the working planar during the merges
		void finalizeInPlaceMerge2d(CsgOperand& operandInOut, bool collapse = false);

	protected:

		/// In place merge template version used by 2D and 3D merge.
		template <typename T>
		void mergeInPlace(T* regions, const CsgOperand* operandToMergeWith, Operation op, bool collapse);

		// Evaluates an operation between leaf Bsp nodes
		HK_FORCE_INLINE NodeType eval(NodeType labelA, Operation op, NodeType labelB)
		{
			// Build code. Each operand has 4 combinations, both operands have 16 combinations. We need to return the result as 2 bits.
			const int labelCode = ((labelA << 2) | labelB) << 1;
			HK_ASSERT(0x7afe2e58, (labelA < 3) && (labelB < 3) && (labelCode < 32) && (op != hkcdPlanarGeometryBooleanUtil::OP_DIFFERENCE));

			const hkUint32 opCode	= (op == hkcdPlanarGeometryBooleanUtil::OP_UNION) ? 0xFFE4D5C7: 0xFFEAE4E3;
			const NodeType ret		= (NodeType)((opCode >> labelCode) & 3);
			return ret;
		}

		/// Recursively merges the given trees
		void doMerge(const CsgOperand* treeToMergeWith, Operation op, CsgOperand* result, CsgOperand* dualResult);

		/// Return a corresponding node Id from a given label
		HK_FORCE_INLINE NodeId nodeIdFromLabel(hkcdPlanarSolid* solid, hkcdPlanarSolid::NodeTypes label);

		/// Initialize a stack with the leaves of the given tree
		void initializeStackWithLeavesForDualOp(CsgOperand* resultTree, CsgOperand* dualResultTree, NodeId treeBRootNodeId, hkArray<StackEntry>& stack);
		void initializeStackWithLeaves(const hkcdPlanarSolid* tree, hkArray<StackEntry>& stack);

	public:

		hkRefPtr<const CsgOperand> m_operandToMerge;	///< The trees
		hkcdConvexCellsTree3D* m_regions;				///< The cell graph, used to track the feasible regions
		hkcdConvexCellsTree2D* m_regions2d;				///< The 2d cell graph, used to track the feasible regions
		bool m_swapInputTrees;

		/// The resulting Bsp trees. In the case of UNION, only m_result[0] is evaluated, while for INTERSECTION, both m_result[0] and m_result[1] are evaluated.
		/// In the latter case, m_result[0] = INTERSECTION and m_result[1] = DIFFERENCE, as the resulting trees are generally identical albeit with flipped labels.
		hkRefPtr<CsgOperand> m_result[2];
};

//
//	Merge the input tree in the results.

void hkcdBspBooleanImpl::doMerge(const CsgOperand* operandToMergeWith, Operation op, CsgOperand* result, CsgOperand* dualResult)
{
	hkArray<StackEntry> stack;
	const NodeType flipLabel[]		= { hkcdPlanarSolid::NODE_TYPE_INTERNAL, hkcdPlanarSolid::NODE_TYPE_OUT,
										hkcdPlanarSolid::NODE_TYPE_IN, hkcdPlanarSolid::NODE_TYPE_UNKNOWN };

	// Copy tree A into results
	hkcdPlanarSolid* solids[2]	= {	HK_NULL, HK_NULL };
	hkcdConvexCellsTree3D* regions[2]		= {	HK_NULL, HK_NULL };
	if ( result )		{	solids[0] = result->accessSolid();		regions[0] = result->accessRegions();		}
	if ( dualResult )	{	solids[1] = dualResult->accessSolid();	regions[1] = dualResult->accessRegions();	}

	// Push the tree leaves on the stack
	const hkcdPlanarSolid* treeToMergeWith = operandToMergeWith->getSolid();
	initializeStackWithLeavesForDualOp(result, dualResult, treeToMergeWith->getRootNodeId(), stack);

	// Iterate
	while ( stack.getSize() )
	{
		// Pop the first element on the stack
		const StackEntry entry = stack[0];
		stack.removeAt(0);
		NodeId nodeResId[2];
		nodeResId[0]			= entry.getNodeIdA();
		nodeResId[1]			= entry.getNodeIdB();
		NodeId nodeToMergeId	= entry.getParentNodeIdA();
		const Node& nodeToMerge	= treeToMergeWith->getNode(nodeToMergeId);		

		// Treat result (0), then dual result (1)
		bool resEnable[2];
		for (int r = 0 ; r < 2 ; r++)
		{
			resEnable[r] = false;
			NodeId nodeRId = nodeResId[r];
			if ( !solids[r] || !nodeRId.isValid() )
			{
				continue;
			}

			// If the region is empty, return the special NULL node (we'll reuse the unknown node for that!)
			Node& nodeRes				= solids[r]->accessNode(nodeRId);
			const CellId regionId		= CellId(nodeRes.m_data);
			if ( !regionId.isValid() )
			{
				nodeRes.m_typeAndFlags = hkcdPlanarSolid::NODE_TYPE_UNKNOWN;
				continue;
			}

			// Get node labels
			NodeType labels[2];
			labels[0]		= solids[r]->getNodeLabel(nodeRId);
			labels[1]		= treeToMergeWith->getNodeLabel(nodeToMergeId);

			// Check if it's worth to split the region represented by the leaf cell of A
			if ( labels[0] == hkcdPlanarSolid::NODE_TYPE_IN || labels[0] == hkcdPlanarSolid::NODE_TYPE_OUT )
			{
				const NodeType In_op_B	= eval(hkcdPlanarSolid::NODE_TYPE_IN , op, labels[0]);
				const NodeType Out_op_B	= eval(hkcdPlanarSolid::NODE_TYPE_OUT, op, labels[0]);
				if ( In_op_B == Out_op_B )
				{
					continue;
				}
			}

			// If the node of the tree B is a leaf, we can terminate!
			if ( labels[1] == hkcdPlanarSolid::NODE_TYPE_IN || labels[1] == hkcdPlanarSolid::NODE_TYPE_OUT )
			{
				// Evaluate the operation between labels of A and B, and add the leaf			
				nodeRes.m_typeAndFlags = ( r == 0 || m_swapInputTrees ) ? labels[1] : flipLabel[labels[1]];
				if ( regions[r] )
				{
					Cell& cell = regions[r]->accessCell(regionId);
					cell.setLabel(( nodeRes.m_typeAndFlags == hkcdPlanarSolid::NODE_TYPE_IN ) ? hkcdConvexCellsCollection::CELL_SOLID : hkcdConvexCellsCollection::CELL_EMPTY);
					cell.setLeaf(true);
				}
				continue;
			}

			resEnable[r] = true;
		}

		// If both res and dual res are out.. continue
		if ( !resEnable[0] && !resEnable[1] ) continue;

		// Main recursion. Get splitting plane from the current node of B
		const PlaneId splitPlaneId				= nodeToMerge.m_planeId;

		CellId insideRegionId, outsideRegionId;
		for (int r = 0 ; r <= 1 ; r++)
		{
			if ( !resEnable[r] ) continue;

			NodeId nodeRId = nodeResId[r];
			Node& nodeRes				= solids[r]->accessNode(nodeRId);
			const CellId regionId		= CellId(nodeRes.m_data);

			// Split current region with the splitting plane		
			if ( regions[r] )
			{
				regions[r]->splitCell(solids[r], regionId, splitPlaneId, insideRegionId, outsideRegionId);
			}

			// Replace the current result node of tree the node of tree B and allocate its children
			nodeRes.m_planeId		= splitPlaneId;
			nodeRes.m_typeAndFlags	= hkcdPlanarSolid::NODE_TYPE_INTERNAL;

			// Recurse on both subtrees of A
			{
				NodeId newChildId0	= NodeId::invalid();
				newChildId0			= solids[r]->createNode(PlaneId::invalid(), NodeId::invalid(), NodeId::invalid());
				Node& node0			= solids[r]->accessNode(newChildId0);
				node0.m_parent		= nodeRId;
				node0.m_data		= insideRegionId.valueUnchecked();
				if ( insideRegionId.isValid() && regions[r] ) regions[r]->accessCell(insideRegionId).setUserData(newChildId0.value());
				node0.m_typeAndFlags = hkcdPlanarSolid::NODE_TYPE_UNKNOWN;

				NodeId newChildId1	= NodeId::invalid();
				newChildId1			= solids[r]->createNode(PlaneId::invalid(), NodeId::invalid(), NodeId::invalid());
				Node& node1			= solids[r]->accessNode(newChildId1);
				node1.m_parent		= nodeRId;
				node1.m_data		= outsideRegionId.valueUnchecked();
				if ( outsideRegionId.isValid() && regions[r] ) regions[r]->accessCell(outsideRegionId).setUserData(newChildId1.value());
				node1.m_typeAndFlags = hkcdPlanarSolid::NODE_TYPE_UNKNOWN;

				Node& node			= solids[r]->accessNode(nodeRId);		// Pointers may have changed due to realocaation of nodes
				node.setChildId(0, newChildId0);
				node.setChildId(1, newChildId1);
			}
		}

		// Set up the stack for the next recursion step
		StackEntry* se = stack.expandBy(2);
		se[0].set(	resEnable[0] ? solids[0]->getNode(nodeResId[0]).m_left  : NodeId::invalid(), 
					resEnable[1] ? solids[1]->getNode(nodeResId[1]).m_left  : NodeId::invalid(), nodeToMerge.m_left , 
					NodeId(0), 0, 0, 0, 0, CellId::invalid());
		se[1].set(	resEnable[0] ? solids[0]->getNode(nodeResId[0]).m_right : NodeId::invalid(), 
					resEnable[1] ? solids[1]->getNode(nodeResId[1]).m_right : NodeId::invalid(), nodeToMerge.m_right,
					NodeId(0), 0, 0, 0, 0, CellId::invalid());
	}

	// Last step. If the inputs are not swapped, copy the cells of the result into the dual result
	if ( !m_swapInputTrees && result && dualResult )
	{
		hkcdConvexCellsTree3D* clone = new hkcdConvexCellsTree3D(*regions[0]);
		dualResult->setRegions(clone);
		clone->relabelCellsFromSolid(solids[1]);
		clone->removeReference();
	}
}

void hkcdBspBooleanImpl::initializeStackWithLeavesForDualOp(CsgOperand* result, CsgOperand* dualResult, NodeId treeBRootNodeId, hkArray<StackEntry>& stack)
{
	const NodeType flipLabel[]		= { hkcdPlanarSolid::NODE_TYPE_INTERNAL, hkcdPlanarSolid::NODE_TYPE_OUT,
										hkcdPlanarSolid::NODE_TYPE_IN, hkcdPlanarSolid::NODE_TYPE_UNKNOWN };

	hkcdPlanarSolid* solids[2]	= {	HK_NULL, HK_NULL };
	hkcdConvexCellsTree3D* regions[2]		= {	HK_NULL, HK_NULL };
	if ( result )		{	solids[0] = result->accessSolid();		regions[0] = result->accessRegions();		}
	if ( dualResult )	{	solids[1] = dualResult->accessSolid();	regions[1] = dualResult->accessRegions();	}

	NodeId nodeId[2];
	for (int r = 0 ; r < 2 ; r++)
	{
		nodeId[r] = ( solids[r] ) ? solids[r]->getRootNodeId() : NodeId::invalid();
	}

	// Put the root on the stack
	{
		StackEntry& entry	= stack.expandOne();
		entry.set(nodeId[0], nodeId[1], treeBRootNodeId, NodeId::invalid(), 0, 0, 1, 1, CellId::invalid());
	}

	// Recursively adds all the leaves
	int head = 1;
	int queue = 0;
	while ( queue < head )
	{
		const StackEntry& entry = stack[queue];

		nodeId[0]		= entry.getNodeIdA();
		nodeId[1]		= entry.getNodeIdB();
		HK_ASSERT(0xa357ed4f, nodeId[0].isValid() || nodeId[1].isValid());
		const Node& node	= ( nodeId[0].isValid() ) ? solids[0]->getNode(nodeId[0]) : solids[1]->getNode(nodeId[1]);

		if ( node.m_typeAndFlags == hkcdPlanarSolid::NODE_TYPE_INTERNAL )
		{
			// Remove the current node from the stack: it's not a leaf
			stack.removeAt(queue);		

			// Add the two children as potential leaves
			StackEntry& leftEntry = stack.expandOne();		
			leftEntry.set(	( nodeId[0].isValid() ) ? solids[0]->getNode(nodeId[0]).m_left  : nodeId[0],
							( nodeId[1].isValid() ) ? solids[1]->getNode(nodeId[1]).m_left  : nodeId[1],
							treeBRootNodeId, NodeId::invalid(), 0, 0, 0, 0, CellId::invalid());			

			//const Node& rightNode	= treeA->getNode(node.m_right);
			StackEntry& rightEntry = stack.expandOne();
			rightEntry.set(	( nodeId[0].isValid() ) ? solids[0]->getNode(nodeId[0]).m_right : nodeId[0],
							( nodeId[1].isValid() ) ? solids[1]->getNode(nodeId[1]).m_right : nodeId[1],
							treeBRootNodeId, NodeId::invalid(), 0, 0, 0, 0, CellId::invalid());			

			head++;
		}
		else
		{
			// We may want to flip the label for the dual result
			if ( m_swapInputTrees && nodeId[1].isValid() )
			{
				Node& nodeToFlip			= solids[1]->accessNode(nodeId[1]);
				if ( nodeToFlip.m_typeAndFlags == hkcdPlanarSolid::NODE_TYPE_IN || nodeToFlip.m_typeAndFlags == hkcdPlanarSolid::NODE_TYPE_OUT )
				{
					nodeToFlip.m_typeAndFlags	= flipLabel[nodeToFlip.m_typeAndFlags];
					const CellId cellToFlipId	= CellId(nodeToFlip.m_data);
					if ( cellToFlipId.isValid() )
					{
						Cell& cellToFlip			= regions[1]->accessCell(CellId(nodeToFlip.m_data));
						bool isSolid				= cellToFlip.isSolid();
						if ( isSolid ) cellToFlip.setEmpty(); else cellToFlip.setSolid();
					}
				}
			}

			queue++;
		}
	}
}

//
//	Initialize a stack with the leaves of the given tree

void hkcdBspBooleanImpl::initializeStackWithLeaves(const hkcdPlanarSolid* tree, hkArray<StackEntry>& stack)
{
	// Initialize the stack
	{
		NodeId nodeId = tree->getRootNodeId();
		if ( !nodeId.isValid() )
		{
			return;
		}
	
		// Put the root on the stack
		const Node& node	= tree->getNode(nodeId);
		StackEntry& entry	= stack.expandOne();
		entry.set(nodeId, NodeId::invalid(), NodeId::invalid(), NodeId::invalid(), 0, 0, 0, 0, CellId(node.m_data));
	}

	// Recursively adds all the leaves
	int head = 1;
	int queue = 0;
	while ( queue < head )
	{
		const StackEntry& entry = stack[queue];

		NodeId nodeId		= entry.getNodeIdA();
		const Node& node	= tree->getNode(nodeId);
		if ( node.m_typeAndFlags == hkcdPlanarSolid::NODE_TYPE_INTERNAL )
		{
			// Remove the current node from the stack: it's not a leaf
			stack.removeAt(queue);			

			// Add the two children as potential leaves
			const Node& leftNode	= tree->getNode(node.m_left);
			StackEntry& leftEntry = stack.expandOne();
			leftEntry.set(node.m_left, NodeId::invalid(), NodeId::invalid(), NodeId::invalid(), 0, 0, 0, 0, CellId(leftNode.m_data));			

			const Node& rightNode	= tree->getNode(node.m_right);
			StackEntry& rightEntry = stack.expandOne();
			rightEntry.set(node.m_right, NodeId::invalid(), NodeId::invalid(), NodeId::invalid(), 0, 0, 0, 0, CellId(rightNode.m_data));	

			head++;
		}
		else
		{
			queue++;
		}
	}
}

//
//	3D in-place merge

HK_FORCE_INLINE void hkcdBspBooleanImpl::mergeInPlace(const CsgOperand* operandToMergeWith, Operation op, bool collapse)
{
	mergeInPlace<hkcdConvexCellsTree3D>(m_regions, operandToMergeWith, op, collapse);
}

//
//	2D in-place merge

HK_FORCE_INLINE void hkcdBspBooleanImpl::mergeInPlace2d(const CsgOperand* operandToMergeWith, Operation op, bool collapse)
{
	mergeInPlace<hkcdConvexCellsTree2D>(m_regions2d, operandToMergeWith, op, collapse);
}

//
//	Merges the input tree with the current result tree
//	This merge performs an in-place operation as it will modify the input provided in setOperatorsForMerge.

template <typename T>
void hkcdBspBooleanImpl::mergeInPlace(T* regions, const CsgOperand* operandToMergeWith, Operation op, bool collapse)
{
	// tree A: result tree, in place operations (m_result[0])
	// tree B: input tree (m_trees[0])
	hkcdPlanarSolid* treeA			= m_result[0]->accessSolid();
	const hkcdPlanarSolid* treeB	= operandToMergeWith->getSolid();

	// Compute tree depths
	const int treeDepthA		= treeA->computeMaxDepth();
	const int treeDepthB		= treeB->computeMaxDepth();
	const int approxStackSize	= (treeDepthA + treeDepthB) << 1;
	hkArray<StackEntry> stack;	stack.reserve(approxStackSize);

	// Push the tree leaves on the stack	
	initializeStackWithLeaves(treeA, stack);

	// Iterate
	while ( stack.getSize() )
	{
		// Pop the first element on the stack
		const StackEntry entry = stack[0];
		stack.removeAt(0);
		NodeId nodeIdA	= entry.getNodeIdA();		// Id in m_result[0]
		NodeId nodeIdB	= entry.getNodeIdB();		// Id in m_trees[0]
		if ( !nodeIdB.isValid() )
		{
			nodeIdB = treeB->getRootNodeId();
		}

		// If the region is empty, return the special NULL node (we'll reuse the unknown node for that!)
		Node& nodeA								= treeA->accessNode(nodeIdA);
		const typename T::CellId regionId		= typename T::CellId(nodeA.m_data);

		if ( !regionId.isValid() )
		{
			nodeA.m_typeAndFlags = hkcdPlanarSolid::NODE_TYPE_UNKNOWN;
			continue;
		}

		// Get node labels
		NodeType labels[2];
		labels[0]		= treeA->getNodeLabel(nodeIdA);
		labels[1]		= treeB->getNodeLabel(nodeIdB);

		// If the node of the tree B is a leaf, we can terminate!
		if ( labels[1] == hkcdPlanarSolid::NODE_TYPE_IN || labels[1] == hkcdPlanarSolid::NODE_TYPE_OUT )
		{
			if ( labels[0] != hkcdPlanarSolid::NODE_TYPE_UNKNOWN )
			{
				// Evaluate the operation between labels of A and B, and add the leaf for the next merge
				nodeA.m_typeAndFlags = eval(labels[0], op, labels[1]);
			}
			else
			{
				// Take the label of the node of B
				nodeA.m_typeAndFlags = labels[1];
			}
			continue;
		}

		// Check if it's worth to split the region represented by the leaf cell of A
		if ( labels[0] == hkcdPlanarSolid::NODE_TYPE_IN || labels[0] == hkcdPlanarSolid::NODE_TYPE_OUT )
		{
			const NodeType In_op_B	= eval(hkcdPlanarSolid::NODE_TYPE_IN, op, labels[0]);
			const NodeType Out_op_B	= eval(hkcdPlanarSolid::NODE_TYPE_OUT, op, labels[0]);
			if ( In_op_B == Out_op_B )
			{
				continue;
			}
		}

		// Main recursion. Get splitting plane from the current node of B
		const Node& nodeB						= treeB->getNode(nodeIdB);
		const PlaneId splitPlaneId				= nodeB.m_planeId;

		// Split current region with the splitting plane
		typename T::CellId insideRegionId, outsideRegionId;
		regions->splitCell(HK_NULL, regionId, splitPlaneId, insideRegionId, outsideRegionId);	

		// Replace the current node of tree A by the node of tree B and allocate its children
		nodeA.m_planeId			= splitPlaneId;
		nodeA.m_typeAndFlags	= hkcdPlanarSolid::NODE_TYPE_INTERNAL;

		// Recurse on both subtrees of A
		{
			NodeId newChildId0	= NodeId::invalid();
			newChildId0			= treeA->createNode(PlaneId::invalid(), NodeId::invalid(), NodeId::invalid());
			Node& node0			= treeA->accessNode(newChildId0);
			node0.m_parent		= nodeIdA;
			node0.m_data		= insideRegionId.valueUnchecked();
			node0.m_typeAndFlags = hkcdPlanarSolid::NODE_TYPE_UNKNOWN;

			NodeId newChildId1	= NodeId::invalid();
			newChildId1			= treeA->createNode(PlaneId::invalid(), NodeId::invalid(), NodeId::invalid());
			Node& node1			= treeA->accessNode(newChildId1);
			node1.m_parent		= nodeIdA;
			node1.m_data		= outsideRegionId.valueUnchecked();
			node1.m_typeAndFlags = hkcdPlanarSolid::NODE_TYPE_UNKNOWN;

			Node& nA			= treeA->accessNode(nodeIdA);		// Pointers may have changed due to realocaation of nodes
			const Node& nB		= treeB->getNode(nodeIdB);
			nA.setChildId(0, newChildId0);
			nA.setChildId(1, newChildId1);

			// All the 0 values represent fields used by the merge3d, but are unused for the in place merging
			StackEntry* se = stack.expandBy(2);
			se[0].set(nA.m_left, nB.m_left, NodeId(0), NodeId(0), 0, 0, 0, 0, CellId(insideRegionId.valueUnchecked()));
			se[1].set(nA.m_right, nB.m_right, NodeId(0), NodeId(0), 0, 0, 0, 0, CellId(outsideRegionId.valueUnchecked()));
		}
	}

	// Simplify the trees to avoid exponential growth during stacked merging
	if ( collapse )
	{
		treeA->collapseUnknownLabels();
		treeA->collapseIdenticalLabels();
		treeA->optimizeStorage();
	}	
}

//
//	Starts merging the trees

void hkcdBspBooleanImpl::merge(Operation op, bool computeOp, bool computeDualOp, bool simplifyResults)
{
	HK_TIMER_BEGIN("Merge", HK_NULL);
	// Performs the merge
	doMerge(m_operandToMerge, op, m_result[0], m_result[1]);
	HK_TIMER_END();

	// Treat results
	HK_TIMER_BEGIN("SimplifyOperands", HK_NULL);
	for (int k = 0; k < 2; k++)
	{
		CsgOperand* result = m_result[k];

		// Optimize tree
		if ( result && result->getSolid() )
		{
			hkcdPlanarSolid* solid = result->accessSolid();

			if ( solid->isValid() )
			{			
				solid->collapseUnknownLabels();
				solid->collapseIdenticalLabels();
				solid->optimizeStorage();

				if ( simplifyResults && solid->isValid() ) 
				{
					result->simplifyFromBoundaries();
				}

				// Remove all regions that are no longer referenced by BSP nodes
				hkcdConvexCellsTree3D* regions = result->accessRegions();
				if ( regions && solid->isValid() )
				{	
					hkcdConvexCellsCollection* cells = regions->accessCells();
					const hkcdPlanarSolid::NodeStorage& nodes = *solid->getNodes();

					// Mark all cells as unvisited
					for (CellId cellId = cells->getFirstCellId(); cellId.isValid(); cellId = cells->getNextCellId(cellId))
					{
						Cell& cell = cells->accessCell(cellId);
						cell.setVisited(false);
					}

					// Visit all cells referenced by the BSP tree
					for (int ni = nodes.getCapacity() - 1; ni >= 0; ni--)
					{
						const NodeId nodeId	(ni);
						const Node& node	= nodes[nodeId];
						const CellId cellId	(node.m_data);
						HK_ASSERT(0x2a69e7dc, !node.isAllocated() || cellId.isValid());

						if ( node.isAllocated() && cellId.isValid() )
						{
							Cell& cell = cells->accessCell(cellId);
							cell.setVisited(true);
							if ( node.m_typeAndFlags == hkcdPlanarSolid::NODE_TYPE_IN || node.m_typeAndFlags == hkcdPlanarSolid::NODE_TYPE_OUT )
							{
								cell.setLeaf(true);
								cell.setLabel(( node.m_typeAndFlags == hkcdPlanarSolid::NODE_TYPE_IN ) ? hkcdConvexCellsCollection::CELL_SOLID : hkcdConvexCellsCollection::CELL_EMPTY);
							}
						}
					}

					// Delete all cells that are not referenced (i.e. visited)
					for (CellId cellId = cells->getFirstCellId(); cellId.isValid(); )
					{
						Cell& cell = cells->accessCell(cellId);
						const CellId idToDelete = cellId;
						cellId = cells->getNextCellId(cellId);

						if ( !cell.isVisited() )
						{
							cells->freeCell(idToDelete);
						}
					}
				}
			}
		}
	}
	HK_TIMER_END();
}

//
//	Remove all the regions added to the working planar during the merges

void hkcdBspBooleanImpl::finalizeInPlaceMerge(CsgOperand& operandInOut, bool collapse)
{
	hkcdPlanarSolid* result = m_result[0]->accessSolid();

	if ( collapse )
	{
		result->collapseUnknownLabels();
		result->collapseIdenticalLabels();
		result->optimizeStorage();
	}

	// Remove regions
	if ( m_regions )
	{
		m_regions->removeCellsOfType(hkcdConvexCellsCollection::CELL_UNKNOWN);
		m_regions->removeReference();
		m_regions = HK_NULL;
	}

	// Return
	operandInOut = *m_result[0];
}

//
//	Remove all the regions added to the working planar during the merges

void hkcdBspBooleanImpl::finalizeInPlaceMerge2d(CsgOperand& operandInOut, bool collapse) 
{
	hkcdPlanarSolid* result = m_result[0]->accessSolid();

	if ( collapse )
	{
		result->collapseUnknownLabels();
		result->collapseIdenticalLabels();
		result->optimizeStorage();
	}

	// Remove regions
	if ( m_regions2d )
	{
		m_regions2d->accessCells()->accessPolygons().clear();
		delete m_regions2d;
		m_regions2d = HK_NULL;
	}

	// Return
	operandInOut = *m_result[0];
}

//
//	Initializes the boolean

void hkcdBspBooleanImpl::setOperands(const CsgOperand* operandA, const CsgOperand* operandB, int neededTrees, Operation op, bool manifoldCells, hkcdPlanarGeometryBooleanUtil::PolyCollManager* polysCollManager, hkcdPlanarGeometryBooleanUtil::CellCollManager* cellsCollManager)
{
#if ( ENABLE_TEXT_DEBUG )
	{
		HK_REPORT("hkcdBspBooleanImpl::setOperands");
		HK_REPORT("Operand A");
		operandA->getSolid()->dbgPrint();
		HK_REPORT("Operand B");
		operandB->getSolid()->dbgPrint();
	}
#endif

	HK_TIMER_BEGIN("SetOperands", HK_NULL);

	const bool lightCopy = ( polysCollManager && cellsCollManager );

	HK_ASSERT(0x728e434, operandA && operandA->getSolid() && operandA->getSolid()->isValid());
	HK_ASSERT(0x728e434, operandB && operandB->getSolid() && operandB->getSolid()->isValid());
	const hkUint32 label			= ( op == hkcdPlanarGeometryBooleanUtil::OP_UNION ) ? hkcdPlanarSolid::NODE_TYPE_OUT : hkcdPlanarSolid::NODE_TYPE_IN;
	const int nbMinOpsIfNotSwapped	= operandA->getSolid()->computeNumNodesWithLabel(label);
	if ( neededTrees & 2 )
	{
		m_swapInputTrees			= operandB->getSolid()->computeNumLeafNodes() < nbMinOpsIfNotSwapped;
	}
	else
	{
		const int nbMinOpsBIfNotSwapped	= operandB->getSolid()->computeNumNodesWithLabel(label);
		if ( nbMinOpsIfNotSwapped != nbMinOpsBIfNotSwapped )
		{
			m_swapInputTrees			= nbMinOpsBIfNotSwapped < nbMinOpsIfNotSwapped;
		}
		else
		{
			m_swapInputTrees			= operandB->getSolid()->computeNumLeafNodes() < operandA->getSolid()->computeNumLeafNodes();
		}
	}

	// Choose an operand to initialize the results, and the "operand to be merged with"
	hkRefPtr<CsgOperand> newOperandA = HK_NULL, newOperandB = HK_NULL, result;
	m_operandToMerge = m_swapInputTrees ? operandA : operandB;

	// If the trees have different geometries, merge them now into the new one
	const PlanesCollection* planesColA = operandA->getSolid()->getPlanesCollection();
	const PlanesCollection* planesColB = operandB->getSolid()->getPlanesCollection();
	if ( planesColA != planesColB )
	{
		newOperandA.setAndDontIncrementRefCount(new CsgOperand());
		if ( lightCopy )
		{
			newOperandA->copyData(*operandA, (hkcdPlanarGeometryPlanesCollection*)operandA->getPlanesCollection(), polysCollManager, cellsCollManager, !m_swapInputTrees);
		}
		else
		{
			newOperandA->copyData(*operandA, true, !m_swapInputTrees);
		}

		newOperandB.setAndDontIncrementRefCount(new CsgOperand());
		if ( lightCopy )
		{
			newOperandB->copyData(*operandB, (hkcdPlanarGeometryPlanesCollection*)operandB->getPlanesCollection(), polysCollManager, cellsCollManager, m_swapInputTrees);
		}
		else
		{
			newOperandB->copyData(*operandB, true,  m_swapInputTrees);
		}

		result = m_swapInputTrees ? newOperandB : newOperandA;
		
		CsgOperand* opToMerge = m_swapInputTrees ? newOperandA : newOperandB;
		m_operandToMerge = opToMerge;
		
		// Must build a new geometry, shared between the two trees		
		// Create the merged collection
		hkArray<int> remapTable;
		const int numPlanesA = planesColA->getNumPlanes();
		PlanesCollection* planesColAB = PlanesCollection::createMergedCollection(planesColA, planesColB, &remapTable);
		
		// Re-index the planes inside the new trees
		newOperandA->setPlanesCollection(planesColAB, &remapTable[0]);
		newOperandB->setPlanesCollection(planesColAB, &remapTable[numPlanesA]);
		planesColAB->removeReference();
	}
	else
	{
		result.setAndDontIncrementRefCount(new CsgOperand());
		if ( lightCopy )
		{
			result->copyData(m_swapInputTrees ? *operandB : *operandA, (hkcdPlanarGeometryPlanesCollection*)operandA->getPlanesCollection(), polysCollManager, cellsCollManager, true);
		}
		else
		{
			result->copyData(m_swapInputTrees ? *operandB : *operandA, true, true);
		}
	}

	m_result[0] = HK_NULL;
	m_result[1] = HK_NULL;

	// Compute the convex cell trees of the results if it is still not there...
	result->getOrCreateConvexCellTree(manifoldCells, true);

	// Initialize the results
	if ( neededTrees & 1 )
	{
		m_result[0] = result;
	}
	if ( neededTrees & 2 )
	{
		if ( m_result[0] )
		{
			m_result[1].setAndDontIncrementRefCount(new CsgOperand());
			if ( lightCopy )
			{
				m_result[1]->copyData(*m_result[0], m_result[0]->accessPlanesCollection(), polysCollManager, cellsCollManager, true);
			}
			else
			{
				m_result[1]->copyData(*m_result[0], true, true);
			}
		}
		else
		{
			m_result[1] = result;
		}
	}

	HK_TIMER_END();
}

//
//	Initializes the boolean

void hkcdBspBooleanImpl::setOperatorsForInPlaceMerge(CsgOperand* inputOutputOperand, hkcdPlanarGeometry& commonGeom)
{
	// The input tree is assumed to be empty !
	HK_ASSERT(0xfa6841d3, inputOutputOperand && inputOutputOperand->getSolid() && !inputOutputOperand->getSolid()->isValid());
	hkcdPlanarSolid* inputOutputTree = inputOutputOperand->accessSolid();
	m_result[0] = inputOutputOperand;

	// Create the first region in the given geometry
	m_regions = new hkcdConvexCellsTree3D(&commonGeom);
	CellId firstRegionId = m_regions->createBoundaryCell();

	// Creates a dummy root node associated to the first region to enable further recursions
	NodeId newNodeId	= inputOutputTree->createNode(PlaneId::invalid(), NodeId::invalid(), NodeId::invalid());
	Node& node			= inputOutputTree->accessNode(newNodeId);
	node.m_parent		= NodeId::invalid();
	node.m_data			= firstRegionId.valueUnchecked();
	node.m_typeAndFlags = hkcdPlanarSolid::NODE_TYPE_UNKNOWN;
	inputOutputTree->setRootNode(newNodeId);	
}

//
//	Initializes the boolean (this version assumes that the two trees already share the same geometry)

void hkcdBspBooleanImpl::setOperatorsForInPlaceMerge2d(CsgOperand* inputOutputOperand, hkcdPlanarGeometry& commonGeom, PlaneId supportPlaneId)
{
	// The input tree is assumed to be empty !
	HK_ASSERT(0xfa6841d3, inputOutputOperand && inputOutputOperand->getSolid() && !inputOutputOperand->getSolid()->isValid());
	hkcdPlanarSolid* inputOutputTree = inputOutputOperand->accessSolid();
	m_result[0] = inputOutputOperand;

	// Create the first region in the given geometry
	m_regions2d = new hkcdConvexCellsTree2D(&commonGeom, supportPlaneId);
	const PolygonId firstRegionId = m_regions2d->createBoundaryCell();

	// Creates a dummy root node associated to the first region to enable further recursions
	NodeId newNodeId	= inputOutputTree->createNode(PlaneId::invalid(), NodeId::invalid(), NodeId::invalid());
	Node& node			= inputOutputTree->accessNode(newNodeId);
	node.m_parent		= NodeId::invalid();
	node.m_data			= firstRegionId.valueUnchecked();
	node.m_typeAndFlags = hkcdPlanarSolid::NODE_TYPE_UNKNOWN;
	inputOutputTree->setRootNode(newNodeId);
}

//
//	Initialize an in-place boolean operation.

hkcdPlanarGeometryBooleanUtil::BooleanState* HK_CALL hkcdPlanarGeometryBooleanUtil::setOperatorsForInPlaceMerge(hkcdPlanarCsgOperand* inputOutputOperand, hkcdPlanarGeometry& commonGeom)
{
	hkcdBspBooleanImpl* impl = new hkcdBspBooleanImpl();
	impl->setOperatorsForInPlaceMerge(inputOutputOperand, commonGeom);
	return impl;
}

//
//	Merge in-place. Use this instead of "compute" whenever you need to stack several boolean operation on the same BSP tree.

void HK_CALL hkcdPlanarGeometryBooleanUtil::mergeInPlace(BooleanState* bState, const hkcdPlanarCsgOperand* operandToMergeWith, Operation op, bool collapse)
{
	hkcdBspBooleanImpl* impl = (hkcdBspBooleanImpl*)bState;
	impl->mergeInPlace(operandToMergeWith, op, collapse);
}

//
//	Finalize th in-place merge.

void HK_CALL hkcdPlanarGeometryBooleanUtil::finalizeInPlaceMerge(BooleanState* bState, hkcdPlanarCsgOperand* operandInOut, bool collapse)
{
	hkcdBspBooleanImpl* impl = (hkcdBspBooleanImpl*)bState;
	impl->finalizeInPlaceMerge(*operandInOut, collapse);
	delete impl;
}

//
//	Computes A op B

void HK_CALL hkcdPlanarGeometryBooleanUtil::compute(const hkcdPlanarCsgOperand* operandA, Operation op, const hkcdPlanarCsgOperand* operandB,
													hkRefPtr<const hkcdPlanarCsgOperand>* resultOut, hkRefPtr<const hkcdPlanarCsgOperand>* dualResultOut, bool manifoldCells, bool dynamicSplit,
													PolyCollManager* polysCollManager, CellCollManager* cellsCollManager)
{
	// If we are required to perform a difference, mark tree B as flipped and do an intersection
	bool computeOp		= resultOut != HK_NULL;
	bool computeDualOp	= dualResultOut != HK_NULL;

	int resultIdx = 0;
	if ( op == OP_DIFFERENCE )
	{
		resultIdx = 1;
		op = OP_INTERSECTION;
		hkAlgorithm::swap(computeOp, computeDualOp);
	}

	// Handle null inputs
	const bool validTreeA = operandA && operandA->getSolid() && operandA->getSolid()->isValid();
	const bool validTreeB = operandB && operandB->getSolid() && operandB->getSolid()->isValid();
	if ( !validTreeA || !validTreeB )
	{
		// Either one of:
		//		Intersection with at least one null operand, empty intersection!
		//		Union, with solidA == NULL, return solidB
		//		Union, with solidA != NULL, return solidA
		if ( resultOut )
		{
			*resultOut = (op == OP_INTERSECTION) ? HK_NULL : (operandA ? operandA : operandB);
		}
		return;
	}

	// Start the algorithm
	const int neededTrees = (resultOut ? (1 << resultIdx) : 0) | (dualResultOut ? (1 << (1 - resultIdx)) : 0);
	hkcdBspBooleanImpl impl;

	impl.setOperands(operandA, operandB, neededTrees, op, manifoldCells, ( dynamicSplit ) ? polysCollManager : HK_NULL, ( dynamicSplit ) ? cellsCollManager : HK_NULL);
	impl.merge(op, computeOp, computeDualOp, !dynamicSplit);

	// Return the result of the Boolean operation
	if ( resultOut )
	{
		*resultOut = impl.m_result[resultIdx];
	}
	if ( dualResultOut )
	{
		*dualResultOut = impl.m_result[1 - resultIdx];
	}
}

//
//	Projects the given coplanar polygons on the boundary of (A op B) and adds the result to the mesh
//	This version builds a 2-d BSP tree out of the polygon of A, and classify the polygon of B w.r.t. this BSP tree

void HK_CALL hkcdPlanarGeometryBooleanUtil::addCoplanarPolygonsToMesh(hkcdPlanarGeometry& workingGeom, const hkArray<PolygonId>& coplanarPolysIn, int numPolysOfA, hkArray<PolygonId>& meshPolysOut)
{
	typedef hkcdPlanarCsgOperand						CsgOperand;

	const int numTotPolys = coplanarPolysIn.getSize();
	const int numPolysOfB = numTotPolys - numPolysOfA;

	// Early out: no polygon to merge because they all come from either A or B
	if ( !numPolysOfA || !numPolysOfB )
	{
		meshPolysOut.append(coplanarPolysIn);
		return;
	}

	// If the number of polygons of B is low compared to the number of A, call the O(n squared) algorithm
	const hkReal ratioA = (hkReal)numPolysOfA / (hkReal)numTotPolys;
	const hkReal ratioB = (hkReal)numPolysOfB / (hkReal)numTotPolys;
	if ( (numTotPolys < 200) || ((ratioA < 0.15f || ratioB < 0.15f) && (numTotPolys < 10000)) )			// Empirical values based on timings measurements
	{
		addCoplanarPolygonsToMeshPairedIntersectionTest(workingGeom, coplanarPolysIn, numPolysOfA, meshPolysOut);
		return;
	}

	// Copy all the polygon of B into a working array
	int nbPolyBToCut = 0;
	hkArray<hkcdPlanarGeometry::PolygonId> polyBToCutIds;
	polyBToCutIds.setSize(numPolysOfB);
	for (int i = 0 ; i < numPolysOfB ; i++)
	{
		polyBToCutIds[nbPolyBToCut] = coplanarPolysIn[i + numPolysOfA];
		nbPolyBToCut++;
	}

	// Build the surfaces of A and B by merging all their polygons
	CsgOperand surfaceA;
	{
		hkcdPlanarSolid* newSolid = new hkcdPlanarSolid(workingGeom.getPlanesCollection(), 0, workingGeom.accessDebugger());
		surfaceA.setGeometry(&workingGeom);
		surfaceA.setSolid(newSolid);
		newSolid->removeReference();
	}

	hkcdBspBooleanImpl boolOp;
	boolOp.setOperatorsForInPlaceMerge2d(&surfaceA, workingGeom, workingGeom.getPolygon(coplanarPolysIn[0]).getSupportPlaneId());

	for (int ipolyId = 0 ; ipolyId < numPolysOfA ; ipolyId++)
	{
		const hkcdPlanarGeometry::PolygonId polyId = coplanarPolysIn[ipolyId];

		// Build a small bsp tree out of the polygon
		hkcdPlanarSolid surfaceTmp(workingGeom.getPlanesCollection());
		CsgOperand tmpOperand;
		tmpOperand.setGeometry(&workingGeom);
		tmpOperand.setSolid(&surfaceTmp);
		
		const int numBounds = workingGeom.getNumBoundaryPlanes(polyId);
		hkArray<PlaneId> planesIds;
		for (int b = 0 ; b < numBounds ; b++)
		{
			planesIds.pushBack(workingGeom.getPolygon(polyId).getBoundaryPlaneId(b));
		}

		surfaceTmp.buildConvex(&planesIds[0], planesIds.getSize());

		// Merge it with the main tree
		boolOp.mergeInPlace2d(&tmpOperand, hkcdPlanarGeometryBooleanUtil::OP_UNION);

	}

	boolOp.finalizeInPlaceMerge2d(surfaceA);

	// Split all polygons of B with the surface of A
	hkArray<PolygonId> insidePolys;
	hkArray<PolygonId> outsidePolys;
	hkArray<PolygonId> boundPolys;
	surfaceA.getSolid()->classifyPolygons(workingGeom, polyBToCutIds, insidePolys, boundPolys, outsidePolys);

	// Gather the polygons of A and B
	if ( numPolysOfA > 0 )				meshPolysOut.append(&coplanarPolysIn[0], numPolysOfA);
	if ( outsidePolys.getSize() > 0 )	meshPolysOut.append(&outsidePolys[0], outsidePolys.getSize());
}

//
//	Projects the given coplanar polygons on the boundary of (A op B) and adds the result to the mesh
//	This version check intersection between polygons of A against polygons of B (simple algorithm with square complexity)

void HK_CALL hkcdPlanarGeometryBooleanUtil::addCoplanarPolygonsToMeshPairedIntersectionTest(hkcdPlanarGeometry& workingGeom, const hkArray<PolygonId>& coplanarPolysIn, int numPolysOfA, hkArray<PolygonId>& meshPolysOut)
{
	typedef hkcdPlanarCsgOperand						CsgOperand;

	// Early out: no polygon to merge because they all come from either A or B
	const int numPolysOfB = coplanarPolysIn.getSize() - numPolysOfA;
	if ( !numPolysOfA || !numPolysOfB )
	{
		meshPolysOut.append(coplanarPolysIn);
		return;
	}

	// Copy all the polygon of B into a working array
	hkArray<PolygonId> removeListIds, polyBToCutIds;
	polyBToCutIds.reserve(numPolysOfB * 3);
	polyBToCutIds.append(&coplanarPolysIn[numPolysOfA], numPolysOfB);

	// Split all polygons of A in the same way as the construction of the BSP tree
	// except that this time, the BUONDARY planes are used as splitting planes instead of the support plane, since we want to cut the surface and not the volume
	for (int ipolyAId = 0 ; ipolyAId < numPolysOfA ; ipolyAId++)
	{
		const PolygonId polyAId = coplanarPolysIn[ipolyAId];

		for (int ipolyBId = 0 ; ipolyBId < polyBToCutIds.getSize() ; ipolyBId++)
		{
			// Check for potential intersection between poly A and B
			PolygonId polyBId			= polyBToCutIds[ipolyBId];			
			const hkBool32 intersect	= workingGeom.check2dIntersection(polyAId, polyBId);

			// If the two polygon intersect, split polygon of B with each boundary plane of A
			if ( intersect )
			{
				const int numBoundsA	= workingGeom.getNumBoundaryPlanes(polyAId);
				PolygonId lastPid		= polyBId;

				for (int b = 0; b < numBoundsA; b++)
				{
					const PlaneId splitPlaneId		= workingGeom.getPolygon(polyAId).getBoundaryPlaneId(b);
					const hkcdPlanarGeometryPredicates::Orientation orientation	= workingGeom.classify(polyBId, splitPlaneId);

					if ( orientation == hkcdPlanarGeometryPredicates::INTERSECT )
					{
						PolygonId polyIn, polyOut;
						workingGeom.split(polyBId, splitPlaneId, polyIn, polyOut);

						// Add the outside polygon to the list
						polyBToCutIds.pushBack(polyOut);

						lastPid = polyIn;
						removeListIds.pushBack(polyBId);
						polyBId = polyIn;
					}
				}

				// insidePid contains the overlapping part, add it to the remove list
				if ( lastPid.isValid() )
				{
					removeListIds.pushBack(lastPid);
					polyBToCutIds.removeAt(ipolyBId);
					ipolyBId--;
				}
			}
		}
	}

	// remove the unwanted polygons
	workingGeom.removePolygons(removeListIds);	

	// Gather the polygons of A and B
	meshPolysOut.append(coplanarPolysIn.begin(), numPolysOfA);
	meshPolysOut.append(polyBToCutIds.begin(), polyBToCutIds.getSize());
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
