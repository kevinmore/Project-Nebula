/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


// =====================================================================================================================
// hknpCollisionQueryContext
// =====================================================================================================================

HK_FORCE_INLINE hknpCollisionQueryContext::hknpCollisionQueryContext(
	hknpTriangleShape* queryTriangle, hknpTriangleShape* targetTriangle )
:	m_dispatcher(HK_NULL),
	m_shapeTagCodec(HK_NULL),
	m_queryTriangle(queryTriangle),
	m_targetTriangle(targetTriangle),
	m_externallyAllocatedTriangles(true)
{

}

hknpCollisionQueryContext::~hknpCollisionQueryContext()
{
	if (!m_externallyAllocatedTriangles)
	{
		HK_ON_CPU( m_queryTriangle->removeReference()  );
		HK_ON_CPU( m_targetTriangle->removeReference() );
	}
}


// =====================================================================================================================
// hknpQueryFilterData
// =====================================================================================================================

HK_FORCE_INLINE hknpQueryFilterData::hknpQueryFilterData()
{
	m_materialId          = hknpMaterialId::invalid();
	m_collisionFilterInfo = 0;
	m_userData            = 0;
}

HK_FORCE_INLINE hknpQueryFilterData::hknpQueryFilterData(const hknpBody& body)
{
	setFromBody(body);
}

HK_FORCE_INLINE void hknpQueryFilterData::setFromBody(const hknpBody& body)
{
	m_materialId          = body.m_materialId;
	m_collisionFilterInfo = body.m_collisionFilterInfo;
	m_userData            = body.m_userData;
}


// =====================================================================================================================
// hknpShapeQueryInfo
// =====================================================================================================================

HK_FORCE_INLINE hknpShapeQueryInfo::hknpShapeQueryInfo()
{
	m_body			= HK_NULL;
	m_rootShape		= HK_NULL;
	m_parentShape	= HK_NULL;
	m_shapeToWorld	= HK_NULL;
	m_shapeKeyMask	= HK_NULL;
	m_shapeIsScaled	= false;
	m_shapeScale	. setAll( 1.0f );
	m_shapeScaleOffset.setZero();
	m_shapeConvexRadius = 0.0f;
}

HK_FORCE_INLINE hknpShapeQueryInfo::hknpShapeQueryInfo(const hknpShapeQueryInfo* source)
{
	setFromInfo(*source);
}

HK_FORCE_INLINE	void hknpShapeQueryInfo::setFromInfo(const hknpShapeQueryInfo& source)
{
	// Can't use direct assignment of source here as source's copy operator is private.
	m_body			= source.m_body;
	m_rootShape		= source.m_rootShape;
	m_parentShape	= source.m_parentShape;
	m_shapeKeyPath	= source.m_shapeKeyPath;
	m_shapeToWorld	= source.m_shapeToWorld;
	m_shapeKeyMask	= HK_NULL;
	m_shapeIsScaled	= source.m_shapeIsScaled;
	m_shapeScale	= source.m_shapeScale;
	m_shapeScaleOffset	= source.m_shapeScaleOffset;
	m_shapeConvexRadius	= source.m_shapeConvexRadius;
}

HK_FORCE_INLINE	void hknpShapeQueryInfo::setFromBody( const hknpBody& body )
{
	m_body			= &body;
	m_rootShape		= body.m_shape;
	m_parentShape	= HK_NULL;
	m_shapeKeyPath	. reset();
	m_shapeToWorld	= &body.getTransform();
	m_shapeKeyMask	= HK_NULL;
	m_shapeIsScaled	= false;
	m_shapeScale	. setAll( 1.0f );
	m_shapeScaleOffset.setZero();
	m_shapeConvexRadius = body.m_shape->m_convexRadius;
}


// =====================================================================================================================
// hknpAabbQuery
// =====================================================================================================================

HK_FORCE_INLINE	hknpAabbQuery::hknpAabbQuery()
{
	init();
}

HK_FORCE_INLINE	hknpAabbQuery::hknpAabbQuery( const hkAabb& aabb )
{
	init();
	m_aabb = aabb;
}

void hknpAabbQuery::init()
{
	m_aabb.setEmpty();
	m_filter = HK_NULL;
}

HK_FORCE_INLINE void hknpAabbQueryUtil::addHit(
	const hknpBody* HK_RESTRICT queryBody, const hknpQueryFilterData& queryShapeFilterData,
	const hknpBody* HK_RESTRICT targetBody, hknpShapeKey targetShapeKey, const hknpQueryFilterData& targetShapeFilterData,
	hkArray<hknpShapeKey>* results )
{
#if defined(HK_PLATFORM_SPU)
	if( results->getSize() < results->getCapacity() )
	{
		results->pushBackUnchecked(targetShapeKey);
	}
#else
	results->pushBack(targetShapeKey);
#endif
}

HK_FORCE_INLINE void hknpAabbQueryUtil::addHit(
	const hknpBody* HK_RESTRICT queryBody,
	const hknpBody* HK_RESTRICT targetBody, hknpShapeKey targetShapeKey,
	hkArray<hknpShapeKey>* results )
{
#if defined(HK_PLATFORM_SPU)
	if( results->getSize() < results->getCapacity() )
	{
		results->pushBackUnchecked(targetShapeKey);
	}
#else
	results->pushBack(targetShapeKey);
#endif
}

HK_FORCE_INLINE void hknpAabbQueryUtil::addHit(
	const hknpBody* HK_RESTRICT queryBody, const hknpQueryFilterData& queryShapeFilterData,
	const hknpBody* HK_RESTRICT targetBody, hknpShapeKey targetShapeKey, const hknpQueryFilterData& targetShapeFilterData,
	hknpCollisionQueryCollector* collector )
{
	hknpCollisionResult result;
	result.m_queryType				= hknpCollisionQueryType::QUERY_AABB;
	result.m_queryBodyInfo.m_bodyId	= ( queryBody ? queryBody->m_id : hknpBodyId::INVALID );
	result.m_queryBodyInfo.m_shapeMaterialId = queryShapeFilterData.m_materialId;
	result.m_queryBodyInfo.m_shapeCollisionFilterInfo = queryShapeFilterData.m_collisionFilterInfo;
	result.m_queryBodyInfo.m_shapeUserData = queryShapeFilterData.m_userData;
	result.m_hitBodyInfo.m_bodyId	= ( targetBody ? targetBody->m_id : hknpBodyId::INVALID );
	result.m_hitBodyInfo.m_shapeKey	= targetShapeKey;
	result.m_hitBodyInfo.m_shapeMaterialId = targetShapeFilterData.m_materialId;
	result.m_hitBodyInfo.m_shapeCollisionFilterInfo = targetShapeFilterData.m_collisionFilterInfo;
	result.m_hitBodyInfo.m_shapeUserData = targetShapeFilterData.m_userData;

	collector->addHit( result );
}

HK_FORCE_INLINE void hknpAabbQueryUtil::addHit(
	const hknpBody* HK_RESTRICT queryBody,
	const hknpBody* HK_RESTRICT targetBody, hknpShapeKey targetShapeKey,
	hknpCollisionQueryCollector* collector )
{
	hknpCollisionResult result;
	result.m_queryType				= hknpCollisionQueryType::QUERY_AABB;
	result.m_queryBodyInfo.m_bodyId	= ( queryBody ? queryBody->m_id : hknpBodyId::INVALID );
	result.m_hitBodyInfo.m_bodyId	= ( targetBody ? targetBody->m_id : hknpBodyId::INVALID );
	result.m_hitBodyInfo.m_shapeKey	= targetShapeKey;

	collector->addHit( result );
}


// =====================================================================================================================
// hknpClosestPointsQuery
// =====================================================================================================================

HK_FORCE_INLINE	void hknpClosestPointsQuery::init()
{
	m_shape			= HK_NULL;
	m_body			= HK_NULL;
	m_maxDistance	= HK_REAL_MAX;
	m_filter		= HK_NULL;
}

HK_FORCE_INLINE	hknpClosestPointsQuery::hknpClosestPointsQuery()
{
	init();
}

HK_FORCE_INLINE	hknpClosestPointsQuery::hknpClosestPointsQuery(	const hknpShape& shape,	hkReal maxDistance )
{
	init();
	m_maxDistance	= maxDistance;
	m_shape			= &shape;
}

HK_FORCE_INLINE	hknpClosestPointsQuery::hknpClosestPointsQuery(	const hknpBody& body, hkReal maxDistance )
{
	init();
	m_maxDistance	= maxDistance;
	m_shape			= body.m_shape;
	m_body			= &body;
	m_filterData	. setFromBody(body);
}


// =====================================================================================================================
// hknpRayCastQuery
// =====================================================================================================================

HK_FORCE_INLINE	void hknpRayCastQuery::init()
{
	m_filter		= HK_NULL;
	m_flags			= hkcdRayQueryFlags::NO_FLAGS;
}

HK_FORCE_INLINE	hknpRayCastQuery::hknpRayCastQuery()
:	hkcdRay()
{
	init();
}

HK_FORCE_INLINE	hknpRayCastQuery::hknpRayCastQuery( hkVector4Parameter start, hkVector4Parameter end )
:	hkcdRay()
{
	init();
	setStartEnd( start, end );
}

HK_FORCE_INLINE	hknpRayCastQuery::hknpRayCastQuery( hkVector4Parameter start, hkVector4Parameter direction, hkSimdRealParameter length )
{
	init();
	setStartDirectionLength( start, direction, length );
}

HK_FORCE_INLINE	void hknpRayCastQuery::setStartEnd( hkVector4Parameter start, hkVector4Parameter end )
{
	setEndPoints( start, end );
	HK_WARN_ONCE_ON_DEBUG_IF( getFraction().getReal() > HK_REAL_HIGH, 0xaf12acd1, "Cast distance is too long to guarantee correct results." );
}

HK_FORCE_INLINE	void hknpRayCastQuery::setStartDirectionLength(
	hkVector4Parameter start, hkVector4Parameter direction, hkSimdRealParameter length )
{
	setOriginDirection( start, direction, length );
	HK_WARN_ONCE_ON_DEBUG_IF( getFraction().getReal() > HK_REAL_HIGH, 0xaf12acd2, "Cast distance is too long to guarantee correct results." );
}


// =====================================================================================================================
// hknpShapeCastQuery
// =====================================================================================================================

HK_FORCE_INLINE	hknpShapeCastQuery::hknpShapeCastQuery()
:	hknpRayCastQuery()
{
	m_shape        = HK_NULL;
	m_body         = HK_NULL;
	m_accuracy     = HKNP_DEFAULT_SHAPE_CAST_ACCURACY;
}

HK_FORCE_INLINE	hknpShapeCastQuery::hknpShapeCastQuery(
	const hknpShape& shape, hkVector4Parameter castStartPosition,
	hkVector4Parameter castDirection, hkSimdRealParameter maximumDistance )
:	hknpRayCastQuery( castStartPosition, castDirection, maximumDistance )
{
	m_shape         = &shape;
	m_body          = HK_NULL;
	m_accuracy      = HKNP_DEFAULT_SHAPE_CAST_ACCURACY;
}

HK_FORCE_INLINE	hknpShapeCastQuery::hknpShapeCastQuery(
	const hknpBody& body,
	hkVector4Parameter castDirection, hkSimdRealParameter maximumDistance )
:	hknpRayCastQuery( body.getTransform().getTranslation(), castDirection, maximumDistance )
{
	m_filterData	. setFromBody(body);
	m_shape			= body.m_shape;
	m_body			= &body;
	m_accuracy		= HKNP_DEFAULT_SHAPE_CAST_ACCURACY;
}


// =====================================================================================================================
// hknpCollisionResult
// =====================================================================================================================

HK_FORCE_INLINE	hknpCollisionResult::hknpCollisionResult()
{
	clear();
}

HK_FORCE_INLINE hkBool hknpCollisionResult::operator<( const hknpCollisionResult& b ) const
{
	return m_fraction < b.m_fraction;
}

HK_FORCE_INLINE	void hknpCollisionResult::clear()
{
	m_queryType									= hknpCollisionQueryType::UNDEFINED;

	m_position									. setZero();
	m_normal									. setZero();
	m_fraction									= HK_REAL_MAX;

	m_queryBodyInfo.m_bodyId					= hknpBodyId::invalid();
	m_queryBodyInfo.m_shapeKey					= HKNP_INVALID_SHAPE_KEY;
	m_queryBodyInfo.m_shapeMaterialId			= hknpMaterialId::invalid();
	m_queryBodyInfo.m_shapeCollisionFilterInfo	= 0;
	m_queryBodyInfo.m_shapeUserData				= 0;

	m_hitBodyInfo.m_bodyId						= hknpBodyId::invalid();
	m_hitBodyInfo.m_shapeKey					= HKNP_INVALID_SHAPE_KEY;
	m_hitBodyInfo.m_shapeMaterialId				= hknpMaterialId::invalid();
	m_hitBodyInfo.m_shapeCollisionFilterInfo	= 0;
	m_hitBodyInfo.m_shapeUserData				= 0;
}

HK_FORCE_INLINE const hknpAabbQueryResult& hknpCollisionResult::asAabb() const
{
	HK_ASSERT2( 0xaf13de20, m_queryType == hknpCollisionQueryType::QUERY_AABB, "Collision result is not a QUERY AABB result." );
	return (const hknpAabbQueryResult&)*this;
}

HK_FORCE_INLINE const hknpClosestPointsQueryResult& hknpCollisionResult::asClosestPoints() const
{
	HK_ASSERT2( 0xaf13de21, m_queryType == hknpCollisionQueryType::GET_CLOSEST_POINTS, "Collision result is not a GET CLOSEST POINTS result." );
	return (const hknpClosestPointsQueryResult&)*this;
}

HK_FORCE_INLINE const hknpRayCastQueryResult& hknpCollisionResult::asRayCast() const
{
	HK_ASSERT2( 0xaf13de22, m_queryType == hknpCollisionQueryType::RAY_CAST, "Collision result is not a RAY CAST result." );
	return (const hknpRayCastQueryResult&)*this;
}

HK_FORCE_INLINE const hknpShapeCastQueryResult& hknpCollisionResult::asShapeCast() const
{
	HK_ASSERT2( 0xaf13de23, m_queryType == hknpCollisionQueryType::SHAPE_CAST, "Collision result is not a SHAPE CAST result." );
	return (const hknpShapeCastQueryResult&)*this;
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
