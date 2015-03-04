/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


// Helper function
HK_FORCE_INLINE void hknpShapeQueryInterface_setEOHF(hkSimdRealParameter maxFractionOrDistance, hkSimdReal* collectorEOHF)
{
	// The "Early Out Hit Fraction" can only ever be higher than the query's maximum fraction/distance if the collector
	// has been reset. If the collector is being re-used and has hits already stored inside, then the EOHF by design
	// has to be equal to or lower than the query's maximum fraction/distance and we don't want to reset it.
	if (*collectorEOHF > maxFractionOrDistance)
	{
		// As this seems to be an unused collector, set its EOHF to the query's maximum fraction/distance.
		*collectorEOHF = maxFractionOrDistance;
	}
}


// =====================================================================================================================
//
// CAST RAY
//
// =====================================================================================================================

HK_FORCE_INLINE void hknpShapeQueryInterface::castRay(
	const hknpRayCastQuery& query,
	const hknpShape& targetShape, const hkTransform& targetShapeTransform,
	hknpCollisionQueryCollector* collector)
{
	HK_TIME_CODE_BLOCK( "CastRay", HK_NULL );

	HK_ASSERT2(0xafead11d, query.m_filter == HK_NULL, "Filtering is not supported by this castRay() variant.");

	hknpInplaceTriangleShape targetTriangle( 0.0f );
	hknpCollisionQueryContext queryContext( HK_NULL, targetTriangle.getTriangleShape() );

	hknpQueryFilterData targetShapeFilterData;

	hknpShapeQueryInfo targetShapeInfo;
	targetShapeInfo.m_shapeToWorld = &targetShapeTransform;

	hknpShapeQueryInterface_setEOHF(query.getFraction(), &collector->m_earlyOutHitFraction);

	targetShape.castRayImpl(&queryContext, query, targetShapeFilterData, targetShapeInfo, collector);
}

HK_FORCE_INLINE void hknpShapeQueryInterface::castRay(
	hknpCollisionQueryContext* queryContext,
	const hknpRayCastQuery& query,
	const hknpShape& targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
	hknpCollisionQueryCollector* collector)
{
	HK_TIME_CODE_BLOCK( "CastRay", HK_NULL );

	HK_ASSERT2(0xafead114, targetShapeInfo.m_shapeToWorld != HK_NULL,
		"You must provide a valid world transform for the target shape.");
	HK_ASSERT2(0xafead11a, query.m_filter == HK_NULL || queryContext->m_shapeTagCodec != HK_NULL,
		"You must provide an hknpShapeTagCodec if you enable filtering for this query.");

	hknpShapeQueryInterface_setEOHF(query.getFraction(), &collector->m_earlyOutHitFraction);

	targetShape.castRayImpl(queryContext, query, targetShapeFilterData, targetShapeInfo, collector);
}


// =====================================================================================================================
//
// CAST SHAPE
//
// =====================================================================================================================

HK_FORCE_INLINE void hknpShapeQueryInterface::castShape(
	hknpCollisionQueryContext* queryContext,
	const hknpShapeCastQuery& query, const hkRotation& queryShapeOrientationInWorld,
	const hknpShape& targetShape, const hkTransform& targetShapeToWorld,
	hknpCollisionQueryCollector* collector)
{
	HK_TIME_CODE_BLOCK( "CastShape", HK_NULL );

	HK_ASSERT2(0xafead11c, query.m_filter == HK_NULL, "Filtering is not supported by this castShape() variant.");

	// As query.m_origin is in target space we need to transform it back to world space first.
	hkTransform queryShapeTransform;
	queryShapeTransform.getTranslation()._setTransformedPos(targetShapeToWorld, query.m_origin);
	queryShapeTransform.getRotation() = queryShapeOrientationInWorld;

	hknpShapeQueryInfo queryShapeInfo;
	queryShapeInfo.m_shapeToWorld = &queryShapeTransform;

	hknpQueryFilterData targetShapeFilterData;

	hknpShapeQueryInfo targetShapeInfo;
	targetShapeInfo.m_shapeToWorld = &targetShapeToWorld;

	hknpShapeQueryInterface_setEOHF(query.getFraction(), &collector->m_earlyOutHitFraction);

	hkTransform queryToTarget;
	queryToTarget._setMulInverseMul( *targetShapeInfo.m_shapeToWorld, *queryShapeInfo.m_shapeToWorld );

	queryContext->m_dispatcher->castShape(
		queryContext, query, queryShapeInfo,
		&targetShape, targetShapeFilterData, targetShapeInfo,
		queryToTarget, false, collector );
}

HK_FORCE_INLINE void hknpShapeQueryInterface::castShape(
	hknpCollisionQueryContext* queryContext,
	const hknpShapeCastQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
	const hknpShape& targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
	hknpCollisionQueryCollector* collector)
{
	HK_TIME_CODE_BLOCK( "CastShape", HK_NULL );

	HK_ASSERT2(0xafead115, queryShapeInfo.m_shapeToWorld != HK_NULL,
		"You have to provide a valid world transform for the query shape.");
	HK_ASSERT2(0xafead116, targetShapeInfo.m_shapeToWorld != HK_NULL,
		"You have to provide a valid world transform for the target shape.");
	HK_ASSERT2(0xafead11b, query.m_filter == HK_NULL || queryContext->m_shapeTagCodec != HK_NULL,
		"You have to provide an hknpShapeTagCodec if you enable filtering for this query.");

	// As query.m_origin is in target space we need to transform it back to world space before updating the query's
	// full world transform.
	hkTransform queryShapeToWorldFullTransform = *queryShapeInfo.m_shapeToWorld;
	queryShapeToWorldFullTransform.getTranslation()._setTransformedPos(*targetShapeInfo.m_shapeToWorld, query.m_origin);

	hknpShapeQueryInfo queryShapeInfoWithFullTransform(&queryShapeInfo);
	queryShapeInfoWithFullTransform.m_shapeToWorld = &queryShapeToWorldFullTransform;

	hknpShapeQueryInterface_setEOHF(query.getFraction(), &collector->m_earlyOutHitFraction);

	hkTransform queryToTarget;
	queryToTarget._setMulInverseMul( *targetShapeInfo.m_shapeToWorld, *queryShapeInfo.m_shapeToWorld );

	queryContext->m_dispatcher->castShape(
		queryContext, query, queryShapeInfoWithFullTransform,
		&targetShape, targetShapeFilterData, targetShapeInfo,
		queryToTarget, false, collector );
}


// =====================================================================================================================
//
// GET CLOSEST POINTS
//
// =====================================================================================================================

HK_FORCE_INLINE void hknpShapeQueryInterface::getClosestPoints(
	hknpCollisionQueryContext* queryContext,
	const hknpClosestPointsQuery& query, const hkTransform& queryShapeToWorld,
	const hknpShape& targetShape, const hkTransform& targetShapeToWorld,
	hknpCollisionQueryCollector* collector)
{
	HK_TIME_CODE_BLOCK( "GetClosestPoints", HK_NULL );

	HK_ASSERT2(0xafead11e, query.m_filter == HK_NULL, "Filtering is not supported by this getClosestPoints() variant.");

	hknpShapeQueryInfo queryShapeInfo;
	queryShapeInfo.m_shapeToWorld = &queryShapeToWorld;
	queryShapeInfo.m_shapeConvexRadius = query.m_shape->m_convexRadius;

	hknpQueryFilterData targetShapeFilterData;

	hknpShapeQueryInfo targetShapeInfo;
	targetShapeInfo.m_shapeToWorld = &targetShapeToWorld;
	targetShapeInfo.m_shapeConvexRadius = targetShape.m_convexRadius;

	getClosestPoints(queryContext, query, queryShapeInfo, targetShape, targetShapeFilterData, targetShapeInfo, collector);
}

HK_FORCE_INLINE void hknpShapeQueryInterface::getClosestPoints(
	hknpCollisionQueryContext* queryContext,
	const hknpClosestPointsQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
	const hknpShape& targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
	hknpCollisionQueryCollector* collector)
{
	HK_TIME_CODE_BLOCK( "GetClosestPoints", HK_NULL );

	HK_ASSERT2(0xafead112, queryShapeInfo.m_shapeToWorld != HK_NULL,
		"You have to provide a valid world transform for the query shape.");
	HK_ASSERT2(0xafead113, targetShapeInfo.m_shapeToWorld != HK_NULL,
		"You have to provide a valid world transform for the target shape.");
	HK_ASSERT2(0xafead119, query.m_filter == HK_NULL || queryContext->m_shapeTagCodec != HK_NULL,
		"You have to provide an hknpShapeTagCodec if you enable filtering for this query.");

	hknpShapeQueryInfo queryShapeInfoWithConvexRadius(&queryShapeInfo);
	queryShapeInfoWithConvexRadius.m_shapeConvexRadius = query.m_shape->m_convexRadius;

	hknpShapeQueryInfo targetShapeInfoWithConvexRadius(&targetShapeInfo);
	targetShapeInfoWithConvexRadius.m_shapeConvexRadius = targetShape.m_convexRadius;

	hkTransform queryToTarget;
	queryToTarget._setMulInverseMul(*targetShapeInfo.m_shapeToWorld, *queryShapeInfo.m_shapeToWorld);

	hknpShapeQueryInterface_setEOHF(hkSimdReal::fromFloat(query.m_maxDistance), &collector->m_earlyOutHitFraction);

	HK_ASSERT2( 0x74362871, queryContext->m_dispatcher, "Please provide a collision query dispatcher!" );
	queryContext->m_dispatcher->getClosestPoints(
		queryContext, query, queryShapeInfoWithConvexRadius,
		&targetShape, targetShapeFilterData, targetShapeInfoWithConvexRadius,
		queryToTarget, false, collector );
}


// =====================================================================================================================
//
// QUERY AABB
//
// =====================================================================================================================

HK_FORCE_INLINE void hknpShapeQueryInterface::queryAabb(
	const hknpAabbQuery& query,
	const hknpShape& targetShape,
	hknpCollisionQueryCollector* collector)
{
	HK_TIME_CODE_BLOCK( "QueryAabb", HK_NULL );

	HK_ASSERT2(0xafead11e, query.m_filter == HK_NULL, "Filtering is not supported by this queryAabb() variant.");

	hknpCollisionQueryContext queryContext(HK_NULL, HK_NULL);
	hknpShapeQueryInfo queryShapeInfo;
	hknpQueryFilterData targetShapeFilterData;
	hknpShapeQueryInfo targetShapeInfo;

	targetShape.queryAabbImpl(&queryContext, query, queryShapeInfo, targetShapeFilterData, targetShapeInfo, collector, HK_NULL);
}

HK_FORCE_INLINE void hknpShapeQueryInterface::queryAabb(
	hknpCollisionQueryContext* queryContext,
	const hknpAabbQuery& query,
	const hknpShape& targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
	hkArray<hknpShapeKey>* hits)
{
	HK_TIME_CODE_BLOCK( "QueryAabb", HK_NULL );

	HK_ASSERT2(0xafead117, query.m_filter == HK_NULL || queryContext->m_shapeTagCodec != HK_NULL,
		"You have to provide an hknpShapeTagCodec if you enable filtering for this query.");

	hknpShapeQueryInfo queryShapeInfo;

	targetShape.queryAabbImpl(queryContext, query, queryShapeInfo, targetShapeFilterData, targetShapeInfo, hits, HK_NULL);
}

HK_FORCE_INLINE void hknpShapeQueryInterface::queryAabb(
	hknpCollisionQueryContext* queryContext,
	const hknpAabbQuery& query,
	const hknpShape& targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
	hknpCollisionQueryCollector* collector)
{
	HK_TIME_CODE_BLOCK( "QueryAabb", HK_NULL );

	HK_ASSERT2(0xafead118, query.m_filter == HK_NULL || queryContext->m_shapeTagCodec != HK_NULL,
		"You have to provide an hknpShapeTagCodec if you enable filtering for this query.");

	hknpShapeQueryInfo queryShapeInfo;

	targetShape.queryAabbImpl(queryContext, query, queryShapeInfo, targetShapeFilterData, targetShapeInfo, collector, HK_NULL);
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
