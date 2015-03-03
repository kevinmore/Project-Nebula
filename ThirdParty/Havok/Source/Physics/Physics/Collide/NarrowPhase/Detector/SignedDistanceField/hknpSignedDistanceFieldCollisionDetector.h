/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

//HK_REFLECTION_PARSER_EXCLUDE_FILE

#ifndef HKNP_SIGNED_DISTANCE_FIELD_COLLISION_DETECTOR_H
#define HKNP_SIGNED_DISTANCE_FIELD_COLLISION_DETECTOR_H

#include <Physics/Physics/Collide/NarrowPhase/Detector/hknpCollisionDetector.h>


///
class hknpSignedDistanceFieldCollisionDetector : public hknpCompositeCollisionDetector
{
	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpSignedDistanceFieldCollisionDetector );

		// hknpCompositeCollisionDetector implementation.
		virtual void collideWithChildren(
			const hknpSimulationThreadContext& tl, const hknpInternalCollideSharedData& sharedData,
			hknpCdBody* HK_RESTRICT cdBodyA, const hknpShape* shapeA,
			hknpCdBody* HK_RESTRICT cdBodyB, const hknpShape* shapeB,

			hknpCompositeCollisionCache* compositeCdCache,
			hknpCdCacheStream* childCdCacheStream, hknpCdCacheStream* childCdCacheStreamPpu, // needed by the consumer
			hknpCdCacheWriter* childCdCacheWriter, hknpMxJacobianSorter* jacMxSorter,
			hknpLiveJacobianInfoWriter* liveJacInfoWriter ) HK_OVERRIDE;

		// Note destruct uses the convex convex version
};


#endif // HKNP_SIGNED_DISTANCE_FIELD_COLLISION_DETECTOR_H

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
