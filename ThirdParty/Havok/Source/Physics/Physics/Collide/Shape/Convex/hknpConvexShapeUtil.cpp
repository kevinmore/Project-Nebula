/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Physics/Physics/Collide/Shape/Convex/hknpConvexShapeUtil.h>

#include <Geometry/Internal/Algorithms/Gsk/hkcdGsk.h>
#include <Physics/Physics/Collide/Query/hknpCollisionQuery.h>
#include <Common/Base/Container/BitField/hkBitField.h>


HK_ALIGN16( const hkUint32 hknpConvexShapeUtil::s_curIndices[4] ) = {0,1,2,3};


int HK_CALL hknpConvexShapeUtil::getNumberOfUniqueVertices( const hknpConvexShape* shape )
{
	const hkcdVertex* vertices = shape->getVertices();
	int numVertices = shape->getNumberOfVertices();

	// Compute the number of unique vertex IDs
	hkLocalBitField found(numVertices, hkBitFieldValue::ZERO);
	int numUniqueVertices = 0;
	for (int i = 0; i < numVertices; i++)
	{
		int id = vertices[i].getInt24W();
		if (!found.get(id))
		{
			found.set(id);
			numUniqueVertices++;
		}
	}

	return numUniqueVertices;
}


bool HK_CALL hknpConvexShapeUtil::getClosestPoints(
	const hknpConvexShape* queryCvx, const hknpConvexShape* targetCvx,
	const hkTransform& queryToTarget,
	hkSimdReal* distance, hkVector4* HK_RESTRICT normal, hkVector4* HK_RESTRICT pointOnTarget )
{
	return getClosestPoints(
		queryCvx->getVertices(), queryCvx->getNumberOfVertices(), queryCvx->m_convexRadius,
		targetCvx->getVertices(), targetCvx->hknpConvexShape::getNumberOfVertices(), targetCvx->m_convexRadius,
		queryToTarget, distance, normal, pointOnTarget );
}


bool HK_CALL hknpConvexShapeUtil::getClosestPoints(
	const hknpConvexShape* queryCvx, const hknpQueryFilterData& queryShapeFilterData, const hknpShapeQueryInfo& queryShapeInfo,
	const hknpConvexShape* targetCvx, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
	const hkTransform& queryToTarget,
	hknpCollisionQueryCollector* collector )
{
	hkSimdReal distance = collector->getEarlyOutHitFraction();
	hkVector4 targetToQueryNormal;
	hkVector4 pointOnTarget;

	const bool ret = hknpConvexShapeUtil::getClosestPoints(
		queryCvx->getVertices(), queryCvx->hknpConvexShape::getNumberOfVertices(), queryCvx->m_convexRadius,
		targetCvx->getVertices(), targetCvx->getNumberOfVertices(), targetCvx->m_convexRadius,
		queryToTarget,
		&distance, &targetToQueryNormal, &pointOnTarget );

	if ( ret )
	{
		hknpCollisionResult result;

		result.m_queryType	= hknpCollisionQueryType::GET_CLOSEST_POINTS;

		result.m_fraction	= distance.getReal();
		result.m_position	. _setTransformedPos( *targetShapeInfo.m_shapeToWorld, pointOnTarget );
		result.m_normal		. _setRotatedDir( targetShapeInfo.m_shapeToWorld->getRotation(), targetToQueryNormal );

		result.m_queryBodyInfo.m_bodyId						= ( queryShapeInfo.m_body ? queryShapeInfo.m_body->m_id : hknpBodyId::INVALID );
		result.m_queryBodyInfo.m_shapeKey					= queryShapeInfo.m_shapeKeyPath.getKey();
		result.m_queryBodyInfo.m_shapeMaterialId			= queryShapeFilterData.m_materialId;
		result.m_queryBodyInfo.m_shapeCollisionFilterInfo	= queryShapeFilterData.m_collisionFilterInfo;
		result.m_queryBodyInfo.m_shapeUserData				= queryShapeFilterData.m_userData;

		result.m_hitBodyInfo.m_bodyId						= ( targetShapeInfo.m_body ? targetShapeInfo.m_body->m_id : hknpBodyId::INVALID );
		result.m_hitBodyInfo.m_shapeKey						= targetShapeInfo.m_shapeKeyPath.getKey();
		result.m_hitBodyInfo.m_shapeMaterialId				= targetShapeFilterData.m_materialId;
		result.m_hitBodyInfo.m_shapeCollisionFilterInfo		= targetShapeFilterData.m_collisionFilterInfo;
		result.m_hitBodyInfo.m_shapeUserData				= targetShapeFilterData.m_userData;

		collector->addHit( result );

		return true;
	}

	return false;
}


bool HK_CALL hknpConvexShapeUtil::getClosestPointsUsingConvexHull(
	const hknpShape* queryShape, const hknpQueryFilterData& queryShapeFilterData, const hknpShapeQueryInfo& queryShapeInfo,
	const hknpShape* targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
	const hkTransform& queryToTarget,
	hknpCollisionQueryCollector* collector )
{
	int numVertsQuery = queryShape->getNumberOfSupportVertices();
	hkcdVertex* vertsBufferQuery = hkAllocateStack<hkcdVertex>( numVertsQuery, "calcDistanceSuppVertBufferQuery" );
	const hkcdVertex* vertsQuery = queryShape->getSupportVertices( vertsBufferQuery, numVertsQuery );

	int numVertsTarget = targetShape->getNumberOfSupportVertices();
	hkcdVertex* vertsBufferTarget = hkAllocateStack<hkcdVertex>( numVertsTarget, "calcDistanceSuppVertBufferTarget" );
	const hkcdVertex* vertsTarget = targetShape->getSupportVertices( vertsBufferTarget, numVertsTarget );

	hkcdGsk::GetClosestPointInput	input; input.m_aTb = queryToTarget;
	hkcdGsk::Cache					cache; cache.init();
	hkcdGsk::GetClosestPointOutput	output;
	hkcdGsk::GetClosestPointStatus	status = hkcdGsk::getClosestPoint(vertsTarget, numVertsTarget, vertsQuery, numVertsQuery, input, &cache, output );
	hkSimdReal						radii; radii.setFromFloat(queryShape->m_convexRadius + targetShape->m_convexRadius);
	hkSimdReal						currentDistance = output.getDistance() - radii;
	hkDeallocateStack( vertsBufferTarget, numVertsTarget );
	hkDeallocateStack( vertsBufferQuery, numVertsQuery );

	if ( (status <= hkcdGsk::STATUS_OK_FLAG) && currentDistance.isLess( collector->getEarlyOutHitFraction() ) )
	{
		hkSimdReal targetShapeRadius; targetShapeRadius.setFromFloat( targetShape->m_convexRadius );
		hkVector4 pointOnTarget; pointOnTarget.setSubMul( output.m_pointAinA, output.m_normalInA, targetShapeRadius );
		hkVector4 targetToQueryNormal; targetToQueryNormal.setNeg<4>( output.m_normalInA );

		hknpCollisionResult result;

		result.m_queryType	= hknpCollisionQueryType::GET_CLOSEST_POINTS;

		result.m_fraction	= currentDistance.getReal();
		result.m_position	. _setTransformedPos( *targetShapeInfo.m_shapeToWorld, pointOnTarget );
		result.m_normal		. _setRotatedDir( targetShapeInfo.m_shapeToWorld->getRotation(), targetToQueryNormal );

		result.m_queryBodyInfo.m_bodyId						= ( queryShapeInfo.m_body ? queryShapeInfo.m_body->m_id : hknpBodyId::INVALID );
		result.m_queryBodyInfo.m_shapeKey					= queryShapeInfo.m_shapeKeyPath.getKey();
		result.m_queryBodyInfo.m_shapeMaterialId			= queryShapeFilterData.m_materialId;
		result.m_queryBodyInfo.m_shapeCollisionFilterInfo	= queryShapeFilterData.m_collisionFilterInfo;
		result.m_queryBodyInfo.m_shapeUserData				= queryShapeFilterData.m_userData;

		result.m_hitBodyInfo.m_bodyId						= ( targetShapeInfo.m_body ? targetShapeInfo.m_body->m_id : hknpBodyId::INVALID );
		result.m_hitBodyInfo.m_shapeKey						= targetShapeInfo.m_shapeKeyPath.getKey();
		result.m_hitBodyInfo.m_shapeMaterialId				= targetShapeFilterData.m_materialId;
		result.m_hitBodyInfo.m_shapeCollisionFilterInfo		= targetShapeFilterData.m_collisionFilterInfo;
		result.m_hitBodyInfo.m_shapeUserData				= targetShapeFilterData.m_userData;

		collector->addHit( result );

		return true;
	}

	return false;
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
