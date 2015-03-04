/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HK_COLLIDE2_MOPP_OBB_VIRTUAL_MACHINE_H
#define HK_COLLIDE2_MOPP_OBB_VIRTUAL_MACHINE_H

#include <Physics2012/Internal/Collide/Mopp/Code/hkpMoppCode.h>
#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppVirtualMachine.h>

#if defined(HK_PLATFORM_SPU)
#include <Common/Base/Spu/Dma/Manager/hkSpuDmaManager.h>
#endif

// 20 
typedef int hkpMoppFixedPoint;  

struct hkpMoppObbVirtualMachineQuery
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MOPP, hkpMoppObbVirtualMachineQuery );

	//since the box is converted to axis-aligned, we need only a min and max value
#if !defined(HK_PLATFORM_SPU)
	HK_ALIGN16(int m_xHi);
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
#else
	//for each of the major axes
	hkPadSpu<int> m_xHi;		
	hkPadSpu<int> m_yHi;		
	hkPadSpu<int> m_zHi;

	hkPadSpu<int> m_xLo;
	hkPadSpu<int> m_yLo;
	hkPadSpu<int> m_zLo;

	//the offset of the all previous scales are accumulated here
	hkPadSpu<int> m_offset_x;		
	hkPadSpu<int> m_offset_y;
	hkPadSpu<int> m_offset_z;

	// the current offset for the primitives
	hkPadSpu<unsigned int> m_primitiveOffset;  

	//the shifts from all previous scale commands are accumulated here
	hkPadSpu<int> m_shift;		
#endif

	unsigned int m_properties[hkpMoppCode::MAX_PRIMITIVE_PROPERTIES];
	
};

class hkpMoppObbVirtualMachine : public hkpMoppVirtualMachine 
{
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MOPP, hkpMoppObbVirtualMachine );

		// standard constructor
		inline hkpMoppObbVirtualMachine();						
		// standard destructor
		inline ~hkpMoppObbVirtualMachine();						

#if !defined(HK_PLATFORM_SPU)
		typedef hkArray<hkpMoppPrimitiveInfo>* hkpPrimitiveOutputArray;
		
		void queryObb(const hkpMoppCode* code, const hkTransform& BvToWorld, const hkVector4& extent, const float radius, hkArray<hkpMoppPrimitiveInfo>* primitives_out);
#else
		typedef hkpMoppPrimitiveInfo* hkpPrimitiveOutputArray;
#endif

#if !defined(HK_PLATFORM_SPU)
		void queryAabb(const hkpMoppCode* code, const hkAabb& aabb, hkpPrimitiveOutputArray primitives_out);
#else
		int queryAabbWithMaxCapacity(const hkpMoppCode* code, const hkAabb& aabb, hkpPrimitiveOutputArray primitives_out, int primitives_out_capacity);
#endif


	////////////////////////////////////////////////////////////////
	//
	// THE REMAINDER OF THIS FILE IS FOR INTERNAL USE
	//
	//////////////////////////////////////////////////////////////// 

	protected:
#if !defined(HK_PLATFORM_SPU)
		HK_ALIGN16( hkpMoppFixedPoint m_xHi );	// for ps2
		hkpMoppFixedPoint	m_yHi;
		hkpMoppFixedPoint	m_zHi;
		hkpMoppFixedPoint	m_HiPadding;

		hkpMoppFixedPoint	m_xLo;
		hkpMoppFixedPoint	m_yLo;
		hkpMoppFixedPoint	m_zLo;
		hkpMoppFixedPoint	m_LoPadding;
#else
		hkPadSpu<hkpMoppFixedPoint>  m_xHi;	
		hkPadSpu<hkpMoppFixedPoint>	m_yHi;
		hkPadSpu<hkpMoppFixedPoint>	m_zHi;

		hkPadSpu<hkpMoppFixedPoint>	m_xLo;
		hkPadSpu<hkpMoppFixedPoint>	m_yLo;
		hkPadSpu<hkpMoppFixedPoint>	m_zLo;
#endif
		//the information about the byte tree
		hkPadSpu<const hkpMoppCode*>	m_code;

		// If set (-1) this mask ensures that the chunkId is stored in the shape key
		hkPadSpu<int>				m_reindexingMask;

	public:

#if defined(HK_PLATFORM_SPU)
		hkPadSpu<int> m_dmaGroup;

		hkPadSpu<const hkUint8*> m_originalBaseAddress;
#endif

	protected:

		void queryAabbOnTree	(const hkpMoppObbVirtualMachineQuery* query, const unsigned char* commands, int chunkOffset);
#if !defined(HK_PLATFORM_SPU)
		HK_FORCE_INLINE void generateQueryFromNode(const hkVector4& extent, const hkTransform& BvToWorld, const float radius, hkpMoppObbVirtualMachineQuery& query);
#endif
		HK_FORCE_INLINE void generateQueryFromAabb(const hkVector4& aabbMin, const hkVector4& aabbMax, hkpMoppObbVirtualMachineQuery& query);

};

#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppObbVirtualMachine.inl>

#endif // HK_COLLIDE2_MOPP_OBB_VIRTUAL_MACHINE_H

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
