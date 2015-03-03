/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Physics/Physics/Dynamics/World/Deactivation/CdCacheFilter/hknpDeactiveCdCacheFilter.h>

#include <Physics/Physics/Collide/NarrowPhase/Cache/hknpCollisionCache.h>
#include <Physics/Physics/Dynamics/World/CacheManager/hknpCollisionCacheManager.h>


void hknpDeactiveCdCacheFilter::deactivateCaches(
	const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData, const hkArray<hknpBodyId>& deactivatedBodies,
	hknpCdCacheReader& inactiveCdCacheReader, hknpCdCacheStream& inactiveChildCdCacheStreamIn,
	hknpCdCacheWriter& inactiveCdCacheWriter, hknpCdCacheWriter& inactiveChildCdCacheWriter,
	hkArray<hknpBodyIdPair>& deactivatedDeletedCachesOut )
{
	hknpWorld* world = tl.m_world;

	if ( m_deleteInactiveCvxCaches && m_deleteInactiveMeshCaches )
	{
		hkAabb16* HK_RESTRICT aabbs = world->m_bodyManager.accessPreviousAabbs().begin();
		for (int i=0; i < deactivatedBodies.getSize(); i++)
		{
			hknpBodyId id = deactivatedBodies[i];
			aabbs[ id.value() ].setEmptyKeyUnchanged();
		}
		return;
	}

	for (const hknpCollisionCache* cdCache = inactiveCdCacheReader.access(); cdCache; cdCache = inactiveCdCacheReader.advanceAndAccessNext( cdCache ))
	{
		bool deleteCache = false;
		if ( cdCache->m_type == hknpCollisionCacheType::CONVEX_CONVEX )
		{
			deleteCache = m_deleteInactiveCvxCaches;
		}
		else
		{
			deleteCache = m_deleteInactiveMeshCaches;
		}

		if ( deleteCache )
		{
			hknpBodyIdPair& pair = deactivatedDeletedCachesOut.expandOne();
			pair.m_bodyA  = cdCache->m_bodyA;
			pair.m_bodyB  = cdCache->m_bodyB;
			const hknpBody* bodyA = &world->getSimulatedBody(cdCache->m_bodyA);
			const hknpBody* bodyB = &world->getSimulatedBody(cdCache->m_bodyB);

			hknpMotion* motions = tl.m_world->m_motionManager.accessMotionBuffer();

			hknpCdBodyBase cdBodyA;
			cdBodyA.m_body			= bodyA;
			cdBodyA.m_rootShape		= bodyA->m_shape;
			cdBodyA.m_motion		= &motions[bodyA->m_motionId.value()];
			cdBodyA.m_material		= &(tl.m_materials[bodyA->m_materialId.value()]);
			cdBodyA.m_shapeKey		= HKNP_INVALID_SHAPE_KEY;

			hknpCdBodyBase cdBodyB;
			cdBodyB.m_body			= bodyB;
			cdBodyB.m_rootShape		= bodyB->m_shape;
			cdBodyB.m_motion		= &motions[bodyB->m_motionId.value()];
			cdBodyB.m_material		= &(tl.m_materials[bodyB->m_materialId.value()]);
			cdBodyB.m_shapeKey		= HKNP_INVALID_SHAPE_KEY;

			cdCache->getLeafShapeKeys( &cdBodyA.m_shapeKey, &cdBodyB.m_shapeKey );

			const_cast<hknpCollisionCache*>(cdCache)->destruct( tl, sharedData, &inactiveChildCdCacheStreamIn, HK_NULL /*streamPpu*/, &cdBodyA, &cdBodyB, hknpCdCacheDestructReason::CACHE_DEACTIVATED );
		}
		else
		{
			hknpCollisionCache *dstCache = inactiveCdCacheWriter.reserve(cdCache->getSizeInBytes());
			hkString::memCpy16NonEmpty(dstCache, cdCache, cdCache->getSizeInBytes()>>4);
			dstCache->moveAndConsumeChildCaches( tl, &inactiveChildCdCacheStreamIn, HK_NULL, &inactiveChildCdCacheWriter );
			inactiveCdCacheWriter.advance( dstCache->getSizeInBytes() );
		}
	}
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
