/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_COLLIDE_UTIL_H
#define HKNP_COLLIDE_UTIL_H

#include <Physics/Physics/Collide/hknpCollideSharedData.h>
#include <Physics/Physics/Dynamics/World/Grid/hknpGrid.h>

class hknpMxJacobianSorter;
class hknpCdPairWriter;
class hknpLiveJacobianInfoWriter;
class hknpCdCacheConsumer;
class hknpCdCacheWriter;


class hknpCollidePipeline
{
	protected:

		enum DmaChannels
		{
			DMA_ID_MESH_GET_MOTIONS = 9,
			DMA_ID_MESH_GET_SHAPES = 10,
		};

	public:

		/// Merge and collide two streams (of collision pairs) into a single stream.
		static void HK_CALL mergeAndCollide2Streams(
			const hknpSimulationThreadContext& tl, const hknpInternalCollideSharedData& sharedData,
			int currentLinkIndex,

			hknpCdCacheConsumer& cdCacheReader,  hknpCdCacheStream& childCdCacheStreamIn,  hknpCdCacheStream* childCdCacheStreamInOnPpu,
			hknpCdCacheConsumer* cdCacheReader2, hknpCdCacheStream* childCdCacheStreamIn2, hknpCdCacheStream* childCdCacheStreamIn2OnPpu,

			// output
			hknpCdCacheWriter& cdCacheWriter,			hknpCdCacheWriter& childCdCacheWriter,
			hknpCdCacheWriter* inactiveCdCacheWriter,   hknpCdCacheWriter* inactiveChildCdCacheWriter,  //optional
			hknpCdCacheWriter* crossGridCdCacheWriter,  hknpCdCacheWriter* crossGridChildCdCacheWriter, //optional

			hknpCdPairWriter& activePairWriter, hknpLiveJacobianInfoWriter* liveJacInfoWriter,
			hknpMxJacobianSorter* HK_RESTRICT jacMovingMxSorter, hknpMxJacobianSorter* HK_RESTRICT fixedJacMxSorter );
};


#endif // HKNP_COLLIDE_UTIL_H

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
