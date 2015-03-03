/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


// =====================================================================================================================
// hknpRayCastQueryResult
// =====================================================================================================================

HK_FORCE_INLINE const hkVector4& hknpRayCastQueryResult::getPosition() const
{
	return m_position;
}

HK_FORCE_INLINE const hkVector4& hknpRayCastQueryResult::getSurfaceNormal() const
{
	return m_normal;
}

HK_FORCE_INLINE hkReal hknpRayCastQueryResult::getFraction( const hknpRayCastQuery& query ) const
{
	return m_fraction / query.getFraction().getReal();
}

HK_FORCE_INLINE hkReal hknpRayCastQueryResult::getDistance( const hknpRayCastQuery& query ) const
{
	return query.getDirection().length<3>().getReal() * m_fraction;
}

HK_FORCE_INLINE hkReal hknpRayCastQueryResult::getFractionRaw() const
{
	return m_fraction;
}

HK_FORCE_INLINE hknpBodyId hknpRayCastQueryResult::getBodyId() const
{
	return m_hitBodyInfo.m_bodyId;
}

HK_FORCE_INLINE hknpShapeKey hknpRayCastQueryResult::getShapeKey() const
{
	return m_hitBodyInfo.m_shapeKey;
}

HK_FORCE_INLINE hknpMaterialId hknpRayCastQueryResult::getShapeMaterialId() const
{
	return m_hitBodyInfo.m_shapeMaterialId;
}

HK_FORCE_INLINE hkUint32 hknpRayCastQueryResult::getShapeCollisionFilterInfo() const
{
	return m_hitBodyInfo.m_shapeCollisionFilterInfo;
}

HK_FORCE_INLINE hkUint64 hknpRayCastQueryResult::getShapeUserData() const
{
	return m_hitBodyInfo.m_shapeUserData;
}

HK_FORCE_INLINE hkBool32 hknpRayCastQueryResult::isInnerHit() const
{
	return hkcdRayCastResult::isInsideHit( m_hitResult );
}


// =====================================================================================================================
// hknpShapeCastQueryResult
// =====================================================================================================================

HK_FORCE_INLINE const hkVector4& hknpShapeCastQueryResult::getContactPosition() const
{
	return m_position;
}

HK_FORCE_INLINE const hkVector4& hknpShapeCastQueryResult::getHitShapeContactNormal() const
{
	return m_normal;
}

HK_FORCE_INLINE void hknpShapeCastQueryResult::getQueryShapeContactNormal( hkVector4* normal ) const
{
	normal->setNeg<3>( m_normal );
}

HK_FORCE_INLINE hkReal hknpShapeCastQueryResult::getFraction( const hknpShapeCastQuery& query ) const
{
	return m_fraction / query.getFraction().getReal();
}

HK_FORCE_INLINE hkReal hknpShapeCastQueryResult::getDistance( const hknpShapeCastQuery& query ) const
{
	return query.getDirection().length<3>().getReal() * m_fraction;
}

HK_FORCE_INLINE hkReal hknpShapeCastQueryResult::getFractionRaw() const
{
	return m_fraction;
}

HK_FORCE_INLINE hknpBodyId hknpShapeCastQueryResult::getQueryBodyId() const
{
	return m_queryBodyInfo.m_bodyId;
}

HK_FORCE_INLINE hknpShapeKey hknpShapeCastQueryResult::getQueryShapeKey() const
{
	return m_queryBodyInfo.m_shapeKey;
}

HK_FORCE_INLINE hknpMaterialId hknpShapeCastQueryResult::getQueryShapeMaterialId() const
{
	return m_queryBodyInfo.m_shapeMaterialId;
}

HK_FORCE_INLINE hkUint32 hknpShapeCastQueryResult::getQueryShapeCollisionFilterInfo() const
{
	return m_queryBodyInfo.m_shapeCollisionFilterInfo;
}

HK_FORCE_INLINE hkUint64 hknpShapeCastQueryResult::getQueryShapeUserData() const
{
	return m_queryBodyInfo.m_shapeUserData;
}

HK_FORCE_INLINE hknpBodyId hknpShapeCastQueryResult::getHitBodyId() const
{
	return m_hitBodyInfo.m_bodyId;
}

HK_FORCE_INLINE hknpShapeKey hknpShapeCastQueryResult::getHitShapeKey() const
{
	return m_hitBodyInfo.m_shapeKey;
}

HK_FORCE_INLINE hknpMaterialId hknpShapeCastQueryResult::getHitShapeMaterialId() const
{
	return m_hitBodyInfo.m_shapeMaterialId;
}

HK_FORCE_INLINE hkUint32 hknpShapeCastQueryResult::getHitShapeCollisionFilterInfo() const
{
	return m_hitBodyInfo.m_shapeCollisionFilterInfo;
}

HK_FORCE_INLINE hkUint64 hknpShapeCastQueryResult::getHitShapeUserData() const
{
	return m_hitBodyInfo.m_shapeUserData;
}


// =====================================================================================================================
// hknpClosestPointsQueryResult
// =====================================================================================================================

HK_FORCE_INLINE const hkVector4& hknpClosestPointsQueryResult::getClosestPointOnHitShape() const
{
	return m_position;
}

HK_FORCE_INLINE void hknpClosestPointsQueryResult::getClosestPointOnQueryShape( hkVector4* point ) const
{
	point->setAddMul( m_position, m_normal, hkSimdReal::fromFloat( m_fraction ) );
}

HK_FORCE_INLINE const hkVector4& hknpClosestPointsQueryResult::getSeparatingDirection() const
{
	return m_normal;
}

HK_FORCE_INLINE hkReal hknpClosestPointsQueryResult::getDistance() const
{
	return m_fraction;
}

HK_FORCE_INLINE hknpBodyId hknpClosestPointsQueryResult::getQueryBodyId() const
{
	return m_queryBodyInfo.m_bodyId;
}

HK_FORCE_INLINE hknpShapeKey hknpClosestPointsQueryResult::getQueryShapeKey() const
{
	return m_queryBodyInfo.m_shapeKey;
}

HK_FORCE_INLINE hknpMaterialId hknpClosestPointsQueryResult::getQueryShapeMaterialId() const
{
	return m_queryBodyInfo.m_shapeMaterialId;
}

HK_FORCE_INLINE hkUint32 hknpClosestPointsQueryResult::getQueryShapeCollisionFilterInfo() const
{
	return m_queryBodyInfo.m_shapeCollisionFilterInfo;
}

HK_FORCE_INLINE hkUint64 hknpClosestPointsQueryResult::getQueryShapeUserData() const
{
	return m_queryBodyInfo.m_shapeUserData;
}

HK_FORCE_INLINE hknpBodyId hknpClosestPointsQueryResult::getHitBodyId() const
{
	return m_hitBodyInfo.m_bodyId;
}

HK_FORCE_INLINE hknpShapeKey hknpClosestPointsQueryResult::getHitShapeKey() const
{
	return m_hitBodyInfo.m_shapeKey;
}

HK_FORCE_INLINE hknpMaterialId hknpClosestPointsQueryResult::getHitShapeMaterialId() const
{
	return m_hitBodyInfo.m_shapeMaterialId;
}

HK_FORCE_INLINE hkUint32 hknpClosestPointsQueryResult::getHitShapeCollisionFilterInfo() const
{
	return m_hitBodyInfo.m_shapeCollisionFilterInfo;
}

HK_FORCE_INLINE hkUint64 hknpClosestPointsQueryResult::getHitShapeUserData() const
{
	return m_hitBodyInfo.m_shapeUserData;
}


// =====================================================================================================================
// hknpAabbQueryResult
// =====================================================================================================================

HK_FORCE_INLINE hknpBodyId hknpAabbQueryResult::getBodyId() const
{
	return m_hitBodyInfo.m_bodyId;
}

HK_FORCE_INLINE hknpShapeKey hknpAabbQueryResult::getShapeKey() const
{
	return m_hitBodyInfo.m_shapeKey;
}

HK_FORCE_INLINE hknpMaterialId hknpAabbQueryResult::getShapeMaterialId() const
{
	return m_hitBodyInfo.m_shapeMaterialId;
}

HK_FORCE_INLINE hkUint32 hknpAabbQueryResult::getShapeCollisionFilterInfo() const
{
	return m_hitBodyInfo.m_shapeCollisionFilterInfo;
}

HK_FORCE_INLINE hkUint64 hknpAabbQueryResult::getShapeUserData() const
{
	return m_hitBodyInfo.m_shapeUserData;
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
