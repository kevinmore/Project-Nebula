/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HK_COLLIDE2_MOPP_LONG_RAY_VIRTUAL_MACHINE_H
#define HK_COLLIDE2_MOPP_LONG_RAY_VIRTUAL_MACHINE_H

// Virtual Machine command definitions
#include <Physics2012/Internal/Collide/Mopp/Code/hkpMoppCode.h>
#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppVirtualMachine.h>
#include <Physics2012/Collide/Shape/hkpShape.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastInput.h>

#if defined(HK_PLATFORM_SPU)
#include <Common/Base/Spu/Dma/Manager/hkSpuDmaManager.h>
#include <Physics2012/Collide/Query/Multithreaded/Spu/hkpSpuConfig.h>
#endif

// Read detailed comment in the cpp class
class hkpMoppLongRayVirtualMachine : public hkpMoppVirtualMachine 
{
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MOPP, hkpMoppLongRayVirtualMachine );

		inline hkpMoppLongRayVirtualMachine(){}
		inline ~hkpMoppLongRayVirtualMachine(){}

		// data driven query
		hkBool queryLongRay(const class HK_SHAPE_CONTAINER* collection, const hkpMoppCode* code, const hkpShapeRayCastInput& input, hkpShapeRayCastOutput& rayResult);

		// collector based raycast
		void queryLongRay(const class HK_SHAPE_CONTAINER* collection, const hkpMoppCode* code, const hkpShapeRayCastInput& input, const hkpCdBody& body, hkpRayHitCollector& collector);

	////////////////////////////////////////////////////////////////
	//
	// THE REMAINDER OF THIS FILE IS FOR INTERNAL USE
	//
	//////////////////////////////////////////////////////////////// 

	protected:

		//we will use the information here to go from int to float space
		HK_PAD_ON_SPU( const hkpMoppCode* )			m_code;
		HK_PAD_ON_SPU( float )						m_ItoFScale;
		
		//used for ray intersection test
		//the original ray (must be in primitive space)
		hkpShapeRayCastInput m_ray;

		//whether a hkReal hit has been discovered already
		HK_PAD_ON_SPU( bool )					m_hitFound;		
		HK_PAD_ON_SPU( hkReal )					m_earlyOutHitFraction;

		// either one of those is set

		// for data driven
		HK_PAD_ON_SPU( hkpShapeRayCastOutput* )		m_rayResultPtr;

		// for callback driven
		HK_PAD_ON_SPU( hkpRayHitCollector* )		m_collector;
		HK_PAD_ON_SPU( const hkpCdBody* )			m_body;

		HK_PAD_ON_SPU( const HK_SHAPE_CONTAINER* )	m_collection;

		// If set (-1) this mask ensures that the chunkId is stored in the shape key
		HK_PAD_ON_SPU( int )						m_reindexingMask;

#if defined(HK_PLATFORM_SPU)
		HK_PAD_ON_SPU( const hkUint8* )			m_originalBaseAddress;
#endif

	protected:

		struct QueryInt;
		struct QueryFloat;

		void queryRayOnTree	( const QueryInt* query, const unsigned char* commands, QueryFloat* const fQuery, int chunkOffset);

			// only add a hit if it definitely is a hit
#ifndef HK_ARCH_ARM
		HK_FORCE_INLINE 
#endif 
		void addHit(unsigned int id, const unsigned int properties[hkpMoppCode::MAX_PRIMITIVE_PROPERTIES]);

		HK_FORCE_INLINE void queryLongRaySub(const hkpMoppCode* code,   const hkpShapeRayCastInput& input );
};


#endif // HK_COLLIDE2_MOPP_LONG_RAY_VIRTUAL_MACHINE_H

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
