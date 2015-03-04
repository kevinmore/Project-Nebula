/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HK_COLLIDE2_MOPP_RAY_BUNDLE_VIRTUAL_MACHINE_H
#define HK_COLLIDE2_MOPP_RAY_BUNDLE_VIRTUAL_MACHINE_H

// Virtual Machine command definitions
#include <Physics2012/Internal/Collide/Mopp/Code/hkpMoppCode.h>
#include <Physics2012/Internal/Collide/Mopp/Machine/hkpMoppVirtualMachine.h>
#include <Physics2012/Collide/Shape/hkpShape.h>
//#include <Physics2012/Collide/Shape/Query/hkpShapeRayBundleCastInput.h>

#if defined(HK_PLATFORM_SPU)
#include <Common/Base/Spu/Dma/Manager/hkSpuDmaManager.h>
#include <Physics2012/Collide/Query/Multithreaded/Spu/hkpSpuConfig.h>
#endif

struct RayPointBundle
{
	hkVector4 m_vec[3];
};

// Read detailed comment in the cpp class
class hkpMoppRayBundleVirtualMachine : public hkpMoppVirtualMachine 
{
	public:
		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_MOPP, hkpMoppRayBundleVirtualMachine );

		inline hkpMoppRayBundleVirtualMachine(){}
		inline ~hkpMoppRayBundleVirtualMachine(){}

		// data driven query
		hkVector4Comparison queryRayBundle(const class HK_SHAPE_CONTAINER* collection, const hkpMoppCode* code, const hkpShapeRayBundleCastInput& input, hkpShapeRayBundleCastOutput& rayResults, hkVector4ComparisonParameter mask);

		// collector based raycast
		//void queryRayBundle(const class HK_SHAPE_CONTAINER* collection, const hkpMoppCode* code, const hkpShapeRayCastInput& input, const hkpCdBody& body, hkpRayHitCollector& collector);


	////////////////////////////////////////////////////////////////
	//
	// THE REMAINDER OF THIS FILE IS FOR INTERNAL USE
	//
	//////////////////////////////////////////////////////////////// 

	protected:

		//we will use the information here to go from int to float space
		HK_PAD_ON_SPU( const hkpMoppCode* )			m_code;
		hkVector4					m_ItoFScale; // all components equal
		
		//used for ray intersection test
		//the original ray (must be in primitive space)
		const hkpShapeRayBundleCastInput* m_rays;

		// Cache the bundle versions. We need these to interpolate to the hitpoint when we get a narrowphase hit.
		RayPointBundle m_to;
		RayPointBundle m_from;
		
		//whether a hit for each ray has been discovered already
		hkVector4Comparison		m_hitsFound;		
		hkVector4				m_earlyOutHitFraction; 

		// either one of those is set

		// for data driven
		// assumed to point to an array of 4 hkpShapeRayCastOutputs
		HK_PAD_ON_SPU( hkpShapeRayBundleCastOutput* )		m_rayResultPtr;

		// for callback driven
		// assumed to point to an array of 4 hkpRayHitCollectors
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
		struct QueryBundle;

		void queryRayOnTree	( const QueryInt* query, const unsigned char* commands, QueryBundle* const fQuery, int chunkId);

			// only add a hit if it definitely is a hit
#ifndef HK_ARCH_ARM
		HK_FORCE_INLINE 
#endif 
		void addHit(unsigned int id, const unsigned int properties[hkpMoppCode::MAX_PRIMITIVE_PROPERTIES], QueryBundle* fQuery);

		HK_FORCE_INLINE void queryRayBundleSub(const hkpMoppCode* code,   const hkpShapeRayBundleCastInput& input, hkVector4ComparisonParameter mask );
};


#endif // HK_COLLIDE2_MOPP_RAY_BUNDLE_VIRTUAL_MACHINE_H

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
