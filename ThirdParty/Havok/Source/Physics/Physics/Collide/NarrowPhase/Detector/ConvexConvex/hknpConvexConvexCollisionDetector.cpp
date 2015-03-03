/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Physics/Physics/Collide/NarrowPhase/Detector/ConvexConvex/hknpConvexConvexCollisionDetector.h>

#if defined(HK_PLATFORM_HAS_SPU)
#	include <Common/Base/Monitor/Spu/hkSpuMonitorCache.h>
#endif

#include <Physics/Physics/Collide/NarrowPhase/Cache/hknpCollisionCache.h>


void hknpConvexConvexCollisionDetector::destructCollisionCache(
		 const hknpSimulationThreadContext& tl, const hknpModifierSharedData& sharedData,
		 hknpCollisionCache* cacheToDestruct,
		 hknpCdCacheStream* childCdCacheStream, hknpCdCacheStream* childCdCacheStreamPpu,
		 hknpCdBodyBase* HK_RESTRICT cdBodyA, hknpCdBodyBase* HK_RESTRICT cdBodyB,
		 hknpCdCacheDestructReason::Enum reason )
{
	hknpConvexConvexCollisionCache* HK_RESTRICT cvxCache = static_cast<hknpConvexConvexCollisionCache*>(cacheToDestruct);
	if ( cvxCache->hasManifoldData() )
	{
		hknpManifoldCollisionCache* manifoldCache = (hknpManifoldCollisionCache*)const_cast<hknpConvexConvexCollisionCache*>(cvxCache);
		manifoldCache->_fireManifoldDestroyed(tl, sharedData, *cdBodyA, *cdBodyB, reason);
	}
}

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
