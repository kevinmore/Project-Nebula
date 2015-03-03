/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_CONVEX_COMPOSITE_COLLISION_DETECTOR_H
#define HKNP_CONVEX_COMPOSITE_COLLISION_DETECTOR_H

#include <Physics/Physics/Collide/NarrowPhase/Detector/hknpCollisionDetector.h>
#include <Common/Base/Types/Geometry/Aabb/hkAabbUtil.h>

class hknpCdCacheConsumer;


///
class hknpConvexCompositeCollisionDetector : public hknpCompositeCollisionDetector
{
	public:

		/// Get the expanded AABB of (convex) body A in (composite) body B space.
		static HK_FORCE_INLINE void HK_CALL buildExpandedLocalSpaceAabb(
			const hknpInternalCollideSharedData &sharedData,
			const hknpCdBody& cdBodyA, const hknpCdBody& cdBodyB,
			hkAabb* localSpaceAabbBodyAOut );

		/// Query a composite shape for overlapping shape keys and sort them.
		static int HK_CALL queryAabbWithNmp(
			hknpCollisionQueryContext* queryContext,
			const hknpAabbQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
			const hknpShape& targetShape, const hknpShapeQueryInfo& targetShapeInfo,
			const hknpQueryFilterData& targetShapeFilterData,
			hkArray<hknpShapeKey>* hitLeafs, hknpQueryAabbNmp* HK_RESTRICT nmpInOut );

		/// Perform collision detection for a convex body against a set of shape keys of a composite body.
		static void HK_CALL collideConvexWithCompositeKeys(
			const hknpSimulationThreadContext& tl, const hknpInternalCollideSharedData& sharedData,
			const hknpCdBody& cdBodyA, hknpCdBody* HK_RESTRICT cdBodyB,
			const hknpShapeKey* keys, int keysStriding, int numKeys,
			hknpCdCacheConsumer* HK_RESTRICT srcChildCdCacheConsumer, hknpCdCacheWriter* HK_RESTRICT childCdCacheWriter,
			hknpManifold* HK_RESTRICT manifoldBuffer,
			hknpMxJacobianSorter* HK_RESTRICT jacMxSorter, hknpLiveJacobianInfoWriter* HK_RESTRICT liveJacInfoWriter );

	public:

		HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpConvexCompositeCollisionDetector );

		// hknpCompositeCollisionDetector implementation.
		virtual void collideWithChildren(
			const hknpSimulationThreadContext& tl, const hknpInternalCollideSharedData& sharedData,
			hknpCdBody* HK_RESTRICT cdBodyA, const hknpShape* shapeA,
			hknpCdBody* HK_RESTRICT cdBodyB, const hknpShape* shapeB,
			hknpCompositeCollisionCache* compositeCdCache,
			hknpCdCacheStream* childCdCacheStream, hknpCdCacheStream* childCdCacheStreamPpu,	// needed by the consumer
			hknpCdCacheWriter* childCdCacheWriter, hknpMxJacobianSorter* jacMxSorter,
			hknpLiveJacobianInfoWriter* liveJacInfoWriter ) HK_OVERRIDE;
};


#include <Physics/Physics/Collide/NarrowPhase/Detector/ConvexComposite/hknpConvexCompositeCollisionDetector.inl>

#endif // HKNP_CONVEX_COMPOSITE_COLLISION_DETECTOR_H

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
