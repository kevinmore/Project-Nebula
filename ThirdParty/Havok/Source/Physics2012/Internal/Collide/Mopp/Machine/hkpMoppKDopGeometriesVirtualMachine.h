/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HK_COLLIDE2_MOPP_MOPP_AABB_VIRTUAL_MACHINE_H
#define HK_COLLIDE2_MOPP_MOPP_AABB_VIRTUAL_MACHINE_H

#include <Physics2012/Internal/Collide/Mopp/Code/hkpMoppCode.h>

#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppVirtualMachine.h>
#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppObbVirtualMachine.h>
#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppEarlyExitObbVirtualMachine.h>
#include <Physics2012/Internal/Collide/Mopp/Machine/hkp26Dop.h>
#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppMachine.h>

struct hkpMoppInfo;

class hkpMoppKDopGeometriesVirtualMachine : public hkpMoppVirtualMachine 
{
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MOPP, hkpMoppKDopGeometriesVirtualMachine );

	struct hkpMoppKDopGeometriesVirtualMachineQuery
	{
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MOPP, hkpMoppKDopGeometriesVirtualMachine::hkpMoppKDopGeometriesVirtualMachineQuery );

		//the offset of the all previous scales are accumulated here. Do not change order.
		int m_offset_x;		
		int m_offset_y;
		int m_offset_z;
		//the shifts from all previous scale commands are accumulated here
		int m_shift;	
		
		// the current offset for the primitives
		unsigned int m_primitiveOffset;  
	
		// props
		unsigned int m_properties[hkpMoppCode::MAX_PRIMITIVE_PROPERTIES];
	};

public:

	// standard constructor
	hkpMoppKDopGeometriesVirtualMachine();										
	// standard destructor
	~hkpMoppKDopGeometriesVirtualMachine();										

	void queryMopp( const hkpMoppCode* code, const hkpMoppKDopQuery &queryInput, hkpMoppInfo* kDopGeometries );

protected:

	hkReal		m_ItoFScale;	// int to float space scale 
	hkVector4	m_offset;

	unsigned int m_terminaloffset;
	hkArray<unsigned int> m_visitedTerminals;

	hkpMoppInfo* m_kDopGeometries;

	hkp26Dop		m_kdop;
	int			m_level;
	hkpMoppKDopQuery m_queryObject;

	hkBool m_hitFound, m_earlyExit;

	void queryMoppKDopGeometriesRecurse( const hkpMoppKDopGeometriesVirtualMachineQuery* query, const unsigned char* PC);

	void addHit( unsigned int id, const unsigned int properties[hkpMoppCode::MAX_PRIMITIVE_PROPERTIES] );

	void pushKDop( hkBool isTerminal = false, hkpShapeKey id = hkpShapeKey(-1) );
	void popKDop();
};

#endif // HK_COLLIDE2_MOPP_MOPP_AABB_VIRTUAL_MACHINE_H

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
