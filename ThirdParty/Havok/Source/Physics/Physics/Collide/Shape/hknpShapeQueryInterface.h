/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_SHAPE_QUERY_INTERFACE_H
#define HKNP_SHAPE_QUERY_INTERFACE_H

#include <Physics/Physics/Collide/Query/hknpCollisionQuery.h>
#include <Physics/Physics/Collide/Query/hknpCollisionQueryDispatcher.h>
#include <Physics/Physics/Collide/Query/hknpQueryAabbNmpUtil.h>


/// A collection of shape level spatial query methods.
class hknpShapeQueryInterface
{
	public:

		// =============================================================================================================
		//
		// CAST RAY
		//
		// =============================================================================================================

		/// Cast a ray against a \a targetShape.
		/// The ray has to be specified in target shape space.
		/// Use your \a collector of choice to return any hit(s).
		/// The results in the collector will be in world space.
		/// No filtering or shape tag decoding supported by this method!
		static HK_FORCE_INLINE void castRay(
			const hknpRayCastQuery& query,
			const hknpShape& targetShape, const hkTransform& targetShapeTransform,
			hknpCollisionQueryCollector* collector );

		/// Cast a ray against a \a targetShape.
		/// The ray has to be specified in target shape space.
		/// Use your \a collector of choice to return any hit(s).
		/// The results in the collector will be in world space.
		/// This method supports filtering and shape tag decoding.
		static HK_FORCE_INLINE void castRay(
			hknpCollisionQueryContext* queryContext,
			const hknpRayCastQuery& query,
			const hknpShape& targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			hknpCollisionQueryCollector* collector );

		// =====================================================================================================================
		//
		// CAST SHAPE
		//
		// =====================================================================================================================

		/// Cast a shape against \a targetShape.
		/// The \a query's underlying hkcdRay (i.e. its origin and direction) has to be specified in target shape space.
		/// Use your \a collector of choice to return any hit(s).
		/// The results in the collector will be in 'world space' (i.e. the common space defined by targetShapeToWorld and
		/// queryShapeOrientationInWorld).
		/// This method supports shape tag decoding.
		/// No filtering supported by this method!
		static HK_FORCE_INLINE void castShape(
			hknpCollisionQueryContext* queryContext,
			const hknpShapeCastQuery& query, const hkRotation& queryShapeOrientationInWorld,
			const hknpShape& targetShape, const hkTransform& targetShapeToWorld,
			hknpCollisionQueryCollector* collector );

		/// Cast a shape against \a targetShape.
		/// The \a query's underlying hkcdRay (i.e. its origin and direction) has to be specified in target shape space.
		/// Use your \a collector of choice to return any hit(s).
		/// The results in the collector will be in 'world space' (i.e. the common space defined by targetShapeInfo's and
		/// queryShapeInfo's m_shapeToWorld member).
		/// This method supports filtering and shape tag decoding.
		static HK_FORCE_INLINE void castShape(
			hknpCollisionQueryContext* queryContext,
			const hknpShapeCastQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
			const hknpShape& targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			hknpCollisionQueryCollector* collector);

		// =============================================================================================================
		//
		// GET CLOSEST POINTS
		//
		// =============================================================================================================

		/// Enumerate the closest distance between a query shape (specified in \a query) and \a targetShape
		/// (individually for each query/target shape combination.)
		/// Both transforms are expected to be in the same, shared reference space (named 'world' for simplicity reasons).
		/// Use your \a collector of choice to return any hit(s).
		/// No filtering or shape tag decoding supported by this method!
		static HK_FORCE_INLINE void getClosestPoints(
			hknpCollisionQueryContext* queryContext,
			const hknpClosestPointsQuery& query, const hkTransform& queryShapeToWorld,
			const hknpShape& targetShape, const hkTransform& targetShapeToWorld,
			hknpCollisionQueryCollector* collector );

		/// Enumerate the closest distance between a query shape (specified in \a query) and \a targetShape
		/// (individually for each query/target shape combination.)
		/// Use your \a collector of choice to return any hit(s).
		/// This method supports filtering and shape tag decoding.
		static HK_FORCE_INLINE void getClosestPoints(
			hknpCollisionQueryContext* queryContext,
			const hknpClosestPointsQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
			const hknpShape& targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			hknpCollisionQueryCollector* collector );

		// =============================================================================================================
		//
		// QUERY AABB
		//
		// =============================================================================================================

		/// Enumerate enabled shape keys for the given \a targetShape within the given AABB (target shape space).
		/// Use your \a collector of choice to return the wanted hits. Note though that for each hit passed to the
		/// collector the hknpCollisionResult::m_fraction is set to HK_REAL_MAX.
		/// No filtering or shape tag decoding supported by this method!
		static HK_FORCE_INLINE void queryAabb(
			const hknpAabbQuery& query,
			const hknpShape& targetShape,
			hknpCollisionQueryCollector* collector );

		/// Enumerate enabled shape keys for the given \a targetShape within the given AABB (target shape space).
		/// This method supports filtering and shape tag decoding.
		static HK_FORCE_INLINE void queryAabb(
			hknpCollisionQueryContext* queryContext,
			const hknpAabbQuery& query,
			const hknpShape& targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			hkArray<hknpShapeKey>* hits );

		/// Enumerate enabled shape keys for the given \a targetShape within the given AABB (target shape space).
		/// Use your \a collector of choice to return the wanted hits. Note though that for each hit passed to the
		/// collector the hknpCollisionResult::m_fraction is set to HK_REAL_MAX.
		/// This method supports filtering and shape tag decoding.
		static HK_FORCE_INLINE void queryAabb(
			hknpCollisionQueryContext* queryContext,
			const hknpAabbQuery& query,
			const hknpShape& targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			hknpCollisionQueryCollector* collector );
};

#include <Physics/Physics/Collide/Shape/hknpShapeQueryInterface.inl>

#endif // HKNP_SHAPE_QUERY_INTERFACE_H

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
