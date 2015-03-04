/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Dynamics/World/hknpWorld.h>

#ifdef HK_PLATFORM_SPU
#include <Physics/Internal/Collide/BroadPhase/Hybrid/hknpHybridBroadPhase.h>
#endif

void hknpWorld::castRay( const hknpRayCastQuery& query, hknpCollisionQueryCollector* collector ) const
{
	HK_TIME_CODE_BLOCK( "WorldCastRay", HK_NULL );

	if( query.m_filter == HK_NULL )
	{
		hknpRayCastQuery* queryRw = const_cast<hknpRayCastQuery*>(&query);
		queryRw->m_filter = m_modifierManager->getCollisionQueryFilter();
	}

#ifdef HK_PLATFORM_SPU
	HK_ALIGN16(hkUint8) broadphaseBuffer[sizeof(hknpHybridBroadPhase)];

	hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion(broadphaseBuffer, m_broadPhase, sizeof(hknpHybridBroadPhase), hkSpuDmaManager::READ_COPY);

	hknpHybridBroadPhase* pBroadphase = (hknpHybridBroadPhase*)broadphaseBuffer;

	pBroadphase->hknpHybridBroadPhase::castRay( query, m_bodyManager.getBodyBuffer(), *this, m_intSpaceUtil, collector );

	hkSpuDmaManager::performFinalChecks(m_broadPhase, broadphaseBuffer, sizeof(hknpHybridBroadPhase));
#else
	m_broadPhase->castRay( query, m_bodyManager.getBodyBuffer(), *this, m_intSpaceUtil, collector );
#endif
}

void hknpWorld::castShape(
	const hknpShapeCastQuery& query, const hkRotation& queryShapeOrientation,
	hknpCollisionQueryCollector* collector) const
{
	HK_TIME_CODE_BLOCK( "WorldCastShape", HK_NULL );
	HK_ASSERT2( 0xaf137dd4, queryShapeOrientation.isOk(), "You must provide a valid hkRotation in queryShapeOrientation." );

	if( query.m_filter == HK_NULL )
	{
		hknpShapeCastQuery* queryRw = const_cast<hknpShapeCastQuery*>(&query);
		queryRw->m_filter = m_modifierManager->getCollisionQueryFilter();
	}

#ifdef HK_PLATFORM_SPU
	HK_ALIGN16(hkUint8) broadphaseBuffer[sizeof(hknpHybridBroadPhase)];

	hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion(broadphaseBuffer, m_broadPhase, sizeof(hknpHybridBroadPhase), hkSpuDmaManager::READ_COPY);

	hknpHybridBroadPhase* pBroadphase = (hknpHybridBroadPhase*)broadphaseBuffer;

	pBroadphase->hknpHybridBroadPhase::castShape(query, queryShapeOrientation, m_bodyManager.getBodyBuffer(), *this, m_intSpaceUtil, collector );

	hkSpuDmaManager::performFinalChecks(m_broadPhase, broadphaseBuffer, sizeof(hknpHybridBroadPhase));
#else
	m_broadPhase->castShape( query, queryShapeOrientation, m_bodyManager.getBodyBuffer(), *this, m_intSpaceUtil, collector );
#endif
}

void hknpWorld::getClosestPoints(
	const hknpClosestPointsQuery& query, const hkTransform& queryShapeTransform,
	hknpCollisionQueryCollector* collector) const
{
	HK_TIME_CODE_BLOCK( "WorldGetClosestPoints", HK_NULL );
	HK_ASSERT2( 0xaf137ddd, queryShapeTransform.isOk(), "You must provide a valid hkTransform in queryShapeTransform.");

	if( query.m_filter == HK_NULL )
	{
		hknpClosestPointsQuery* queryRw = const_cast<hknpClosestPointsQuery*>(&query);
		queryRw->m_filter = m_modifierManager->getCollisionQueryFilter();
	}

#ifdef HK_PLATFORM_SPU
	HK_ALIGN16(hkUint8) broadphaseBuffer[sizeof(hknpHybridBroadPhase)];

	hkSpuDmaManager::getFromMainMemoryAndWaitForCompletion(broadphaseBuffer, m_broadPhase, sizeof(hknpHybridBroadPhase), hkSpuDmaManager::READ_COPY);

	hknpHybridBroadPhase* pBroadphase = (hknpHybridBroadPhase*)broadphaseBuffer;

	pBroadphase->hknpHybridBroadPhase::getClosestPoints(query, queryShapeTransform, m_bodyManager.getBodyBuffer(), *this, m_intSpaceUtil, collector );

	hkSpuDmaManager::performFinalChecks(m_broadPhase, broadphaseBuffer, sizeof(hknpHybridBroadPhase));
#else
	m_broadPhase->getClosestPoints( query, queryShapeTransform, m_bodyManager.getBodyBuffer(), *this, m_intSpaceUtil, collector );
#endif
}

#ifndef HK_PLATFORM_SPU

void hknpWorld::queryAabb( const hkAabb& aabb, hkArray<hknpBodyId>& hits ) const
{
	HK_TIME_CODE_BLOCK( "WorldQueryAabb", HK_NULL );

	HK_ALIGN16(hkAabb16) aabbInt;
	m_intSpaceUtil.convertAabb(aabb, aabbInt);

	m_broadPhase->queryAabb( aabbInt, m_bodyManager.getBodyBuffer(), hits );
}

void hknpWorld::queryAabb( const hknpAabbQuery& query, hknpCollisionQueryCollector* collector ) const
{
	HK_TIME_CODE_BLOCK( "WorldQueryAabb", HK_NULL );

	if( query.m_filter == HK_NULL )
	{
		hknpAabbQuery* queryRw = const_cast<hknpAabbQuery*>(&query);
		queryRw->m_filter = m_modifierManager->getCollisionQueryFilter();
	}

	m_broadPhase->queryAabb( query, m_bodyManager.getBodyBuffer(), *this, m_intSpaceUtil, collector );
}

#endif

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
