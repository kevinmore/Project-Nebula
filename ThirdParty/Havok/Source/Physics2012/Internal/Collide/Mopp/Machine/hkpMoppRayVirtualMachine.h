/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HK_COLLIDE2_MOPP_RAY_VIRTUAL_MACHINE_H
#define HK_COLLIDE2_MOPP_RAY_VIRTUAL_MACHINE_H

// Virtual Machine command definitions
#include <Physics2012/Internal/Collide/Mopp/Code/hkpMoppCode.h>
#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppVirtualMachine.h>

struct hkpMoppRayVirtualMachineQuery 
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MOPP, hkpMoppRayVirtualMachineQuery );

	int m_startx;
	int m_starty;
	int m_startz;
	int m_radius;

	int m_endx;
	int m_endy;
	int m_endz;
	int m_padding;

	//the offset of the all previous scales are accumulated here
	int m_offset_x;		
	int m_offset_y;
	int m_offset_z;
	//the shifts from all previous scale commands are accumulated here
	int m_shift;		
	// the current offset for the primitives
	unsigned int m_primitiveOffset;  
	unsigned int m_properties[hkpMoppCode::MAX_PRIMITIVE_PROPERTIES];
};

class hkpMoppRayVirtualMachine: public hkpMoppVirtualMachine 
{
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MOPP, hkpMoppRayVirtualMachine );

		inline hkpMoppRayVirtualMachine();
		inline ~hkpMoppRayVirtualMachine();

		// returns the number of hits
		void queryRay(const hkpMoppCode *code, const hkVector4 &start, const hkVector4 &end, float radius, hkArray<hkpMoppPrimitiveInfo> *primitives_out);		

	////////////////////////////////////////////////////////////////
	//
	// THE REMAINDER OF THIS FILE IS FOR INTERNAL USE
	//
	//////////////////////////////////////////////////////////////// 

	protected:

		HK_ALIGN16( hkpMoppFixedPoint m_startx );
		hkpMoppFixedPoint m_starty;
		hkpMoppFixedPoint m_startz;
		hkpMoppFixedPoint m_radius;

		hkpMoppFixedPoint m_endx;
		hkpMoppFixedPoint m_endy;
		hkpMoppFixedPoint m_endz;
		hkpMoppFixedPoint m_padding;

		void queryRayOnTree	( const hkpMoppRayVirtualMachineQuery *query, const unsigned char *commands);
};

#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppRayVirtualMachine.inl>

#endif // HK_COLLIDE2_MOPP_RAY_VIRTUAL_MACHINE_H

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
