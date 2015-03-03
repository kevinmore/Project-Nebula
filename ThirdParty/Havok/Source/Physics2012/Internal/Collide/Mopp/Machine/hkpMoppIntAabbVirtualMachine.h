/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HK_COLLIDE2_MOPP_INT_AABB_VIRTUAL_MACHINE_H
#define HK_COLLIDE2_MOPP_INT_AABB_VIRTUAL_MACHINE_H

#include <Physics2012/Internal/Collide/Mopp/Code/hkpMoppCode.h>
#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppVirtualMachine.h>

// 20 
typedef int hkpMoppFixedPoint;  

struct hkpMoppIntAabbVirtualMachineQuery
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MOPP, hkpMoppIntAabbVirtualMachineQuery );

	//since the box is converted to axis-aligned, we need only a min and max value
	int m_xHi;		
	//for each of the major axes
	int m_yHi;		
	int m_zHi;
	int m_HiPadding;

	int m_xLo;
	int m_yLo;
	int m_zLo;
	int m_LoPadding;

	//the offset of the all previous scales are accumulated here
	int m_offset_x;		
	int m_offset_y;
	int m_offset_z;
	// the current offset for the primitives
	unsigned int m_primitiveOffset;  

	//the shifts from all previous scale commands are accumulated here
	int m_shift;		

	unsigned int m_properties[hkpMoppCode::MAX_PRIMITIVE_PROPERTIES];

	
};

class hkpMoppIntAabbVirtualMachine : public hkpMoppVirtualMachine 
{
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MOPP, hkpMoppIntAabbVirtualMachine );

		// standard constructor
		inline hkpMoppIntAabbVirtualMachine();						
		// standard destructor
		inline ~hkpMoppIntAabbVirtualMachine();						

		void queryAabb(const hkpMoppCode* code, const hkInt32* aabb, hkArray<hkpMoppPrimitiveInfo>* primitives_out);

	////////////////////////////////////////////////////////////////
	//
	// THE REMAINDER OF THIS FILE IS FOR INTERNAL USE
	//
	//////////////////////////////////////////////////////////////// 

	protected:
		HK_ALIGN16( hkpMoppFixedPoint m_xHi );
		hkpMoppFixedPoint	m_yHi;
		hkpMoppFixedPoint	m_zHi;
		hkpMoppFixedPoint	m_HiPadding;

		hkpMoppFixedPoint	m_xLo;
		hkpMoppFixedPoint	m_yLo;
		hkpMoppFixedPoint	m_zLo;
		hkpMoppFixedPoint	m_LoPadding;


		//the information about the byte tree
		const hkpMoppCode*			m_code;

		void queryAabbOnTree	(const hkpMoppIntAabbVirtualMachineQuery* query, const unsigned char* commands);

		HK_FORCE_INLINE void generateQueryFromAabb(const hkVector4& aabbMin, const hkVector4& aabbMax, hkpMoppIntAabbVirtualMachineQuery& query);
};

#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppIntAabbVirtualMachine.inl>

#endif // HK_COLLIDE2_MOPP_INT_AABB_VIRTUAL_MACHINE_H

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
