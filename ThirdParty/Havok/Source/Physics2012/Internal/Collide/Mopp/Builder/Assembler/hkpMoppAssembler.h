/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

//
// Havok Memory Optimised Partial Polytope Assembler
// This class generates the binary BV code for the VM to execute
//

#ifndef HK_COLLIDE2_MOPP_ASSEMBLER_H
#define HK_COLLIDE2_MOPP_ASSEMBLER_H

struct hkpMoppSplittingPlaneDirection;
class hkpMoppTreeNode;

#include <Physics2012/Internal/Collide/Mopp/Code/hkpMoppCode.h>

class hkpMoppAssembler : public hkReferencedObject
{
	public:
		//
		// some public classes
		//
		struct hkpMoppAssemblerParams
		{
			public:
				HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MOPP, hkpMoppAssembler::hkpMoppAssemblerParams );

			hkpMoppAssemblerParams() :
											m_relativeFitToleranceOfInternalNodes(0.5f),
											m_absoluteFitToleranceOfInternalNodes(0.2f),
											m_absoluteFitToleranceOfTriangles (1.0f),
											m_groupLevels(5)
											{
												m_absoluteFitToleranceOfAxisAlignedTriangles.set( 0.2f, 0.2f, 0.05f );
												m_interleavedBuildingEnabled = true;
											}
			
			/// The maximum relative size of the unused space
			float m_relativeFitToleranceOfInternalNodes;

			/// The minimum width of a chopped off slice
			float m_absoluteFitToleranceOfInternalNodes;

			/// The tightness of the MOPP on a terminal level.
			/// The MOPP compiler tries to create a virtual proxy AABB node around each terminal
			/// where the distance between this proxy node and the hkReal AABB node is
			/// at most m_absoluteFitToleranceOfTriangles.
			float m_absoluteFitToleranceOfTriangles;

			/// The tightness for flat triangles for a given direction
			hkVector4 m_absoluteFitToleranceOfAxisAlignedTriangles;

			/// In order to optimize cache utilizations for the virtual machines
			/// the assembler should organize the tree accordingly:
			/// A node X and all nodes N in the subtree of X with a maximum pathlengths
			/// of m_groupLevels between X and N should be assembled into one continuous
			/// piece of memory.
			/// Note: to achieve best performance, the following formula should be true:
			/// (2^m_groupLevels) ~ cacheLineSizeOfCPU
			/// e.g., for PIII 2^5 ~ 32
			int m_groupLevels;

			/// Flag that indicates whether 'interleaved building' is enabled or disabled.
			///
			/// For more information on 'interleaved building' see the respective parameter in hkpMoppCompilerInput.
			hkBool m_interleavedBuildingEnabled;
		};

public:
	HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_MOPP);

	hkpMoppAssembler(){}

	virtual ~hkpMoppAssembler(){}

	/// Returns the number of splitting plane directions, the assembler can handle.
	virtual int   getNumSplittingPlaneDirections() const = 0;

	/// Returns a pointer to a static table to an [] of possible splitting planes
	/// including a cost for each plane.
	virtual const hkpMoppSplittingPlaneDirection* getSplittingPlaneDirections() const = 0;

	/// Assembles the tree into BV machine code.
	///
	/// Try to assemble a partial tree (the tree might not be complete).
	/// Once a node is fully assembled, the assembler should call
	/// hkpMoppNodeMgr::releaseNode(node) to tell the splitter, that a node can be reused.
	/// At least minNodesToAssemble must be assembled.
	/// The result of this assemble() call is implementation specific.
	virtual void assemble(hkpMoppTreeNode* rootNode, class hkpMoppNodeMgr* mgr, int minNodesToAssemble) = 0;	

	/// Gets the scale information for the tree.
	virtual void getScaleInfo( hkpMoppTreeNode* rootNode, hkpMoppCode::CodeInfo* scaleInfoOut  ) = 0;
};

#endif // HK_COLLIDE2_MOPP_ASSEMBLER_H

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
