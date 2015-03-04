/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

// hkpMoppDefaultSplitter definition

#ifndef HK_COLLIDE2_MOPP_SPLITTER_H
#define HK_COLLIDE2_MOPP_SPLITTER_H

#include <Physics2012/Internal/Collide/Mopp/Builder/Splitter/hkpMoppSplitTypes.h>

//
// Havok Memory Optimised Partial Polytope Tree
//

// forward definition
class hkpMoppSplitParams; 


class hkpMoppNodeMgr : public hkReferencedObject
{
public:
HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE);
	virtual void releaseNode( class hkpMoppTreeNode *nodeToRelease ) = 0;
	virtual int getFreeNodes() = 0;
};


class hkpMoppSplitter: public hkpMoppNodeMgr
{
public:
HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE);
	//
	// some public classes
	//

	struct hkpMoppScratchArea 
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MOPP, hkpMoppSplitter::hkpMoppScratchArea );

		hkpMoppCompilerPrimitive* m_primitives;
		hkpMoppTreeInternalNode*				m_nodes;
		hkpMoppTreeTerminal*	m_terminals;
	};

public:
	//
	// some public classes
	//
 
	/// parameters to the MOPP compile call
	struct hkpMoppSplitParams
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MOPP, hkpMoppSplitter::hkpMoppSplitParams );

		// set the essential parameters and initialize the rest with reasonable default values
		hkpMoppSplitParams( hkpMoppMeshType meshType = HK_MOPP_MT_LANDSCAPE );

		// the maximum error we allow the system to operate
		hkReal m_tolerance;				

		int	m_maxPrimitiveSplits;			// maximum number of split primitives in tree
		int	m_maxPrimitiveSplitsPerNode;	// maximum number of split primitives per node
		int	m_minRangeMaxListCheck;			// minimum number of elements which is checked in the max list 
		int	m_checkAllEveryN;				// all elements in the max list will be checked every N iterations

		// Flag that indicates whether 'interleaved building' is enabled or disabled.
		// For more information on 'interleaved building' see the respective parameter in hkpMoppCompilerInput.
		hkBool m_interleavedBuildingEnabled;
	};


public:
	
	hkpMoppSplitter() {}

	virtual	~hkpMoppSplitter() {}
	
	virtual hkpMoppTreeNode* buildTree(class hkpMoppMediator*, class hkpMoppCostFunction*, class hkpMoppAssembler*, const hkpMoppSplitParams&, hkpMoppScratchArea&) = 0;
		// recursively build the tree
};


#endif // HK_COLLIDE2_MOPP_SPLITTER_H

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
