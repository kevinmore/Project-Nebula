/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Collide/Modifier/ManifoldEventCreator/hknpManifoldEventCreator.h>

#include <Physics/Physics/Collide/hknpCdBody.h>
#include <Physics/Physics/Collide/NarrowPhase/Cache/hknpCollisionCache.h>
#include <Physics/Physics/Collide/Event/hknpCollideEvents.h>
#include <Physics/Physics/Dynamics/Simulation/hknpSimulationThreadContext.h>
#include <Physics/Physics/Collide/NarrowPhase/hknpManifold.h>


void hknpManifoldEventCreator::manifoldCreatedCallback(
	const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
	const hknpCdBody& cdBodyA, const hknpCdBody& cdBodyB,
	ManifoldCreatedCallbackInput* HK_RESTRICT info
	)
{
#if !defined(HK_PLATFORM_SPU)
	hknpManifoldCollisionCache* cache = info->m_collisionCache;
#else
	hknpManifoldCollisionCache* cache = info->m_collisionCacheInMainMemory.val();
#endif

	hknpManifoldStatusEvent event(
		cdBodyA.m_body->m_id, cdBodyA.m_shapeKey,
		cdBodyB.m_body->m_id, cdBodyB.m_shapeKey,
		cache, hknpManifoldStatusEvent::MANIFOLD_CREATED
		);

	event.m_filterBits = hknpCommandDispatchType::AS_SOON_AS_POSSIBLE;
	tl.execCommand(event);
}


void hknpManifoldEventCreator::manifoldProcessCallback(
	const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
	const hknpCdBody& cdBodyA, const hknpCdBody& cdBodyB,
	hknpManifold* HK_RESTRICT manifold
	)
{
	hknpManifoldProcessedEvent event(
		cdBodyA.m_body->m_id, cdBodyA.m_shapeKey,
		cdBodyB.m_body->m_id, cdBodyB.m_shapeKey
		);

#if !defined(HK_PLATFORM_SPU)
	event.m_manifoldCache = manifold->m_collisionCache;
#else
	event.m_manifoldCache = manifold->m_collisionCacheInMainMemory.val();
#endif
	manifold->checkConsistency();
	event.m_numContactPoints = hkUint8(manifold->m_numPoints);
	event.m_filterBits = hknpCommandDispatchType::AS_SOON_AS_POSSIBLE;
	event.m_isNewManifold = manifold->m_isNewManifold;
	event.m_manifold = *manifold;
	tl.execCommand(event);
}


void hknpManifoldEventCreator::manifoldDestroyedCallback(
	const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
	const hknpCdBodyBase& cdBodyA, const hknpCdBodyBase& cdBodyB,
	hknpManifoldCollisionCache* HK_RESTRICT cache, hknpCdCacheDestructReason::Enum reason
	)
{
	HK_ON_SPU( cache = HK_NULL; )

	hknpManifoldStatusEvent event(
		cdBodyA.m_body->m_id, cdBodyA.m_shapeKey,
		cdBodyB.m_body->m_id, cdBodyB.m_shapeKey,
		HK_NULL/*cache*/, hknpManifoldStatusEvent::MANIFOLD_DESTROYED	// the cache will be deleted after calling this modifier
		);

	event.m_filterBits = hknpCommandDispatchType::AS_SOON_AS_POSSIBLE;
	tl.execCommand(event);
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
