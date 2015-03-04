/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Physics2012/Collide/Query/CastUtil/hkpWorldLinearCaster.h>
#include <Physics2012/Collide/Query/CastUtil/hkpLinearCastInput.h>
#include <Physics2012/Collide/Shape/Query/hkpShapeRayCastInput.h>
#include <Physics2012/Collide/Filter/hkpCollisionFilter.h>
#include <Physics2012/Collide/BroadPhase/hkpBroadPhase.h>


hkReal hkpWorldLinearCaster::addBroadPhaseHandle( const hkpBroadPhaseHandle* broadPhaseHandle, int castIndex )
{
#if !defined(HK_PLATFORM_SPU)
	const hkpTypedBroadPhaseHandle* tp = static_cast<const hkpTypedBroadPhaseHandle*>( broadPhaseHandle );
	const hkpCollidable* collB = reinterpret_cast<const hkpCollidable*>( tp->getOwner() );

	const hkpShape* shapeB = collB->getShape();

	if ( (!shapeB) || (m_collidableA == collB) )
	{
		return m_castCollector->getEarlyOutDistance();
	}


	// AABB phantoms do not have shapes
	if ( m_filter->isCollisionEnabled( *m_collidableA, *collB ) )
	{
		hkpShapeType typeB = shapeB->getType();
		hkpCollisionDispatcher::LinearCastFunc linearCastFunc = m_shapeInput.m_dispatcher->getLinearCastFunc( m_typeA, typeB );
		linearCastFunc( *m_collidableA, *collB, m_shapeInput, *m_castCollector, m_startPointCollector ); 
	}
	return m_castCollector->getEarlyOutDistance();
#else
	return 0.0f;
#endif
}

void hkpWorldLinearCaster::linearCast( const hkpBroadPhase& broadphase, const hkpCollidable* collA,
									  const hkpLinearCastInput& input, const hkpCollidableCollidableFilter* filter,
									  const hkpCollisionInput& collInput, hkpCollisionAgentConfig* config, 
									  hkpBroadPhaseAabbCache* m_broadPhaseCache,
									  hkpCdPointCollector& castCollector, hkpCdPointCollector* startPointCollector )
{
	HK_ASSERT2(0x4e6207e1,  filter, "You need to specify a valid filter");
	HK_ASSERT2(0x1bd63818,  castCollector.getEarlyOutDistance() >= 1.0f, "Your collector has not been reset");

	m_castCollector = &castCollector;
	m_startPointCollector = startPointCollector;
	m_input = &input;
	m_collidableA = collA;
	m_filter = filter;
	m_typeA = collA->getShape()->getType();

	hkpCollisionInput& ip = m_shapeInput;
	ip = collInput;
	m_shapeInput.m_config = config;

	hkpBroadPhase::hkpCastAabbInput ci;
	{
		hkVector4 path; path.setSub( input.m_to, collA->getTransform().getTranslation() );
		m_shapeInput.setPathAndTolerance( path, input.m_startPointTolerance );
		hkAabb aabb;
		collA->getShape()->getAabb( collA->getTransform(), input.m_startPointTolerance, aabb );
		
		ci.m_from.setInterpolate( aabb.m_min, aabb.m_max, hkSimdReal_Inv2 );
		ci.m_to.setAdd( ci.m_from, path );
		ci.m_halfExtents.setSub( aabb.m_max, aabb.m_min );
		ci.m_halfExtents.mul( hkSimdReal_Inv2 );
		ci.m_aabbCacheInfo = m_broadPhaseCache;
		m_shapeInput.m_maxExtraPenetration = input.m_maxExtraPenetration;
	}
	broadphase.castAabb( ci, *this );
}

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
