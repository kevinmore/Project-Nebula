/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_QUERY_SHARED_DATA_H
#define HKNP_QUERY_SHARED_DATA_H

#include <Physics/Physics/Collide/Shape/Convex/Triangle/hknpTriangleShape.h>

class hknpQuerySharedData
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpCollideSharedData );

	public:

#if !defined(HK_PLATFORM_SPU)
		hknpQuerySharedData()
		{
			m_collisionFilter = HK_NULL;
			m_collisionFilterType = hknpCollisionFilter::ALWAYS_HIT_FILTER;
			m_shapeTagCodec = HK_NULL;
			m_shapeTagCodecType = hknpShapeTagCodec::NULL_CODEC;
			m_pCollisionQueryDispatcher = HK_NULL;
		}

		void initializeFromWorld(hknpWorld* pWorld)
		{
			m_collisionFilter = pWorld->m_modifierManager->getCollisionFilter();
			m_collisionFilterType = pWorld->m_modifierManager->getCollisionFilter()->m_type;
			m_shapeTagCodec = pWorld->getShapeTagCodec();
			m_shapeTagCodecType = pWorld->getShapeTagCodec()->m_type;
			m_pCollisionQueryDispatcher = pWorld->m_collisionQueryDispatcher;
		}
#endif

		hkPadSpu<hknpCollisionFilter*>						m_collisionFilter;
		hkEnum<hknpCollisionFilter::FilterType, hkUint8>	m_collisionFilterType;
		hkPadSpu<const hknpShapeTagCodec*>					m_shapeTagCodec;
		hkEnum<hknpShapeTagCodec::CodecType, hkUint8>		m_shapeTagCodecType;
		//hkPadSpu<hknpModifierFlags>						m_globalModifierFlags;
		hknpCollisionQueryDispatcherBase*					m_pCollisionQueryDispatcher;
		hknpInplaceTriangleShape							m_referenceTriangleShape;
};


#endif // HKNP_QUERY_SHARED_DATA_H

/*
 * Havok SDK - Base file, BUILD(#20130912)
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
