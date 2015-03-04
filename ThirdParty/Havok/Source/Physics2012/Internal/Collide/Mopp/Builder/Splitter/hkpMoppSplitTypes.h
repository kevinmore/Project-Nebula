/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

//
// Havok Memory Optimised Partial Polytope Binary Tree Node
// Each node represents a splitting plane of the original space into two half-spaces
// The left branch will always represent the half-space in front of the splitting plane
// (i.e. the plane normal is pointing INTO the half-space).
// Similarily, the right branch will always represent the half-space behind the splitting
// plane (i.e. the plane normal is pointing AWAY from the half-space).
// Note: If the pointer to the MOPP primitive is non-NULL, the branch represents a leaf, i.e.
//		 a terminal branch.
//

#ifndef HK_COLLIDE2_MOPP_SPLIT_TYPES_H
#define HK_COLLIDE2_MOPP_SPLIT_TYPES_H

#include <Physics2012/Internal/Collide/Mopp/Code/hkpMoppCode.h>


enum hkpMoppMeshType
{
	HK_MOPP_MT_LANDSCAPE,
	HK_MOPP_MT_INDOOR
};

typedef hkUint32 hkpMoppPrimitiveId;
	//: an id, representing a primitive

//
// structure to hold a single extent of an object
//
struct hkpMoppExtent 
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MOPP, hkpMoppExtent );

	hkReal m_min;
	hkReal m_max;
};

struct hkpMoppCompilerPrimitive 
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MOPP, hkpMoppCompilerPrimitive );

	hkpMoppPrimitiveId		        m_primitiveID;		// ID
	hkpMoppPrimitiveId				m_primitiveID2;		// an extra primitive id which can be used for Mediator implementations
	hkpMoppExtent					m_extent;			// set by the project functions, varys depending on direction
	inline hkBool operator <(const hkpMoppCompilerPrimitive& b) const { return this->m_extent.m_min < b.m_extent.m_min; }
	hkpMoppPrimitiveId		        m_origPrimitiveID;	// Original ID if reindexed while building chunks
};

//
// structure to hold a splitting plane and its associated cost
//
struct hkpMoppSplittingPlaneDirection 
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MOPP, hkpMoppSplittingPlaneDirection );

	hkVector4	m_direction;
	hkReal		m_cost;
	int			m_index;
};



class hkpMoppAssemblerData
{
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MOPP, hkpMoppAssemblerData );

		enum hkpCutCommandType
		{
			HK_MOPP_NO_CUT,
			HK_MOPP_CUT8,
			HK_MOPP_CUT24
		};

		enum hkpRescaleCommandType
		{
			HK_MOPP_NO_RESCALE,
			HK_MOPP_RESCALE
		};

		void init()
		{
			m_isAssembled = 0;
			m_scaleIsValid = 0;
			m_rescaleCommandType = HK_MOPP_NO_RESCALE;
			m_cutCommandType[0] = HK_MOPP_NO_CUT;
			m_cutCommandType[1] = HK_MOPP_NO_CUT;
			m_cutCommandType[2] = HK_MOPP_NO_CUT;
			m_chunkId = -1;
			m_assembledAddress = -1;
		}

	public:
		hkBool					m_isAssembled;
		hkBool					m_scaleIsValid;
		hkEnum<hkpRescaleCommandType,hkInt8>	m_rescaleCommandType;	// the number of bits a rescale will do
		hkEnum<hkpCutCommandType,hkInt8>		m_cutCommandType[3];

		int						m_chunkId;				// this is the chunk id of the root node of a chunk (or -1 otherwise)

		int						m_minCutPlanePosition[3];		// that defines the area which a bottom up node requires to be cut off
		int						m_maxCutPlanePosition[3];

		int						m_assembledAddress;		// during assembly, we note which nodes are finished and their address (-1 otherwise)


};

//#define MOPP_DEBUG_COSTS


class hkpMoppTreeNode
{
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MOPP, hkpMoppTreeNode );

	// 
	// some public classes
	//
	struct hkpMopp3DOPExtents 
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MOPP, hkpMoppTreeNode::hkpMopp3DOPExtents );
		hkpMoppExtent m_extent[3];
	};


public:
	class hkpMoppTreeInternalNode*		m_parent;
	hkBool					m_isTerminal;
	int						m_numPrimitives;			// the number of terminals in this subtree

	hkpMopp3DOPExtents		m_extents;					// the extent of the space for this node

	hkpMoppPrimitiveId	    m_minPrimitiveId;			// pointer to the primitive with the smallest ID for that node
	hkpMoppPrimitiveId	    m_maxPrimitiveId;			// pointer to the primitive with the largest ID for that node

	// the number of properties in use
	int						m_numProperties;									
	// like the primitive IDs - these user IDs need to be re-offset
	hkpPrimitiveProperty		m_minPropertyValue[hkpMoppCode::MAX_PRIMITIVE_PROPERTIES];	
	hkpPrimitiveProperty		m_maxPropertyValue[hkpMoppCode::MAX_PRIMITIVE_PROPERTIES];

	// data which can be used by the assembler
	hkpMoppAssemblerData		m_assemblerData;

	/// safe cast of this node to hkpMoppTreeInternalNode
	inline class hkpMoppTreeInternalNode* toNode();

	/// safe cast of this node to hkMoppTree
	inline class hkpMoppTreeTerminal* toTerminal();

	inline void init();
};

class hkpMoppTreeInternalNode* hkpMoppTreeNode::toNode()
{
	HK_ASSERT(0x72b4d18a,  !m_isTerminal );
	return reinterpret_cast<hkpMoppTreeInternalNode*>(this);
}


class hkpMoppTreeTerminal* hkpMoppTreeNode::toTerminal()
{
	HK_ASSERT(0x3f08f90a,  m_isTerminal );
	return reinterpret_cast<hkpMoppTreeTerminal*>(this);
}

void hkpMoppTreeNode::init()
{
	m_assemblerData.init();
}





class hkpMoppTreeTerminal: public hkpMoppTreeNode
{
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MOPP, hkpMoppTreeTerminal );

	// pointer to the primitive list on left if bottom of the tree, otherwise NULL
	hkpMoppCompilerPrimitive* m_primitive;
};


class hkpMoppBasicNode : public hkpMoppTreeNode 
{
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MOPP, hkpMoppBasicNode );

	//
	//	public classes
	//
	struct hkpMoppCostInfo 
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MOPP, hkpMoppBasicNode::hkpMoppCostInfo );

		hkReal	m_splitCost;			 
		hkReal	m_planeRightPositionCost;
		hkReal    m_primitiveIdSpread;
		hkReal	m_planeLeftPositionCost;	
		hkReal	m_numUnBalancedCost;	 
		hkReal	m_planeDistanceCost;
		hkReal	m_absoluteMin;
		hkReal	m_absoluteMax;
		hkReal	m_directionCost;
	};
	
public:
	const hkpMoppSplittingPlaneDirection*	m_plane;
	hkReal									m_planeRightPosition;	
	hkReal									m_planeLeftPosition;	
	hkReal									m_bestOverallCost;			// stores the overal best cost

#ifdef MOPP_DEBUG_COSTS
	void printDebugInfo();
	hkpMoppCostInfo			m_costInfo;
#endif
	const hkVector4& getPlaneNormal() const { return m_plane->m_direction; }
};

class hkpMoppTreeInternalNode : public hkpMoppBasicNode 
{
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MOPP, hkpMoppTreeInternalNode );

	hkpMoppTreeNode*			m_leftBranch;		// half space left of the  splitting plane
	hkpMoppTreeNode*			m_rightBranch;		// half space right of the splitting plane
	int						m_numPrimitives;
};

#endif // HK_COLLIDE2_MOPP_SPLIT_TYPES_H

/*
 * Havok SDK - Product file, BUILD(#20130912)
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
