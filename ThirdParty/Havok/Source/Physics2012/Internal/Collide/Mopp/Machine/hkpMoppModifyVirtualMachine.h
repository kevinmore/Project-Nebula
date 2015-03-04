/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HK_COLLIDE2_MOPP_MODIFY_VIRTUAL_MACHINE_H
#define HK_COLLIDE2_MOPP_MODIFY_VIRTUAL_MACHINE_H

#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppObbVirtualMachine.h>
#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppModifier.h>

class hkpMoppModifyVirtualMachine : public hkpMoppObbVirtualMachine 
{
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MOPP, hkpMoppModifyVirtualMachine );

		// standard constructor
		inline hkpMoppModifyVirtualMachine(){}
		// standard destructor
		inline ~hkpMoppModifyVirtualMachine(){}

			/// Read the hkMoppModifyVirtualMachine_queryAabb documentation
		void queryAabb( const hkpMoppCode* code, const hkAabb& aabb, hkpMoppModifier* modifierOut );

	protected:
		HK_FORCE_INLINE void addHit(unsigned int id, const unsigned int properties[hkpMoppCode::MAX_PRIMITIVE_PROPERTIES], hkUlong chunkId=0);

			/// returns true if this node should be removed
		hkBool queryModicationPointsRecursive	(const hkpMoppObbVirtualMachineQuery* query, const unsigned char* commands, int chunkId);

	protected:
		hkBool			m_tempLastShouldTerminalBeRemoved;
		hkpMoppModifier* m_modifier;

		// If set (-1) this mask ensures that the chunkId is stored in the shape key
		hkPadSpu<int>	m_reindexingMask;
};

#endif // HK_COLLIDE2_MOPP_MODIFY_VIRTUAL_MACHINE_H

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
