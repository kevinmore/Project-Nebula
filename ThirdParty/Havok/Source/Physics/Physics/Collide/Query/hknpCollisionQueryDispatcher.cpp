/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Physics/Physics/Collide/Query/hknpCollisionQueryDispatcher.h>

#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/Base/Math/Matrix/hkTransformUtil.h>

#include <Physics/Physics/Collide/Shape/Composite/Compound/Dynamic/hknpDynamicCompoundShape.h>
#include <Physics/Physics/Collide/Shape/Composite/Compound/Static/hknpStaticCompoundShape.h>
#include <Physics/Physics/Collide/Shape/Composite/HeightField/hknpHeightFieldShape.h>
#include <Physics/Physics/Collide/Shape/Composite/Masked/hknpMaskedCompositeShape.h>
#include <Physics/Physics/Collide/Shape/Composite/Mesh/Compressed/hknpCompressedMeshShape.h>
#include <Physics/Physics/Collide/Shape/Composite/Mesh/Compressed/hknpCompressedMeshShapeUtil.h>
#include <Physics/Physics/Collide/Shape/Composite/Mesh/Extern/hknpExternMeshShape.h>
#include <Physics/Physics/Collide/Shape/Convex/hknpConvexShapeUtil.h>
#include <Physics/Physics/Collide/Shape/Composite/Mesh/Extern/hknpExternMeshShape.h>
#include <Physics/Physics/Collide/Shape/Composite/Mesh/Extern/hknpExternMeshShapeUtil.h>
#include <Physics/Physics/Collide/Query/Collector/hknpFlippedShapeCastQueryCollector.h>
#include <Physics/Physics/Collide/Query/Collector/hknpFlippedGetClosestPointsQueryCollector.h>

/// Compute the set of closest points between two shapes.
void hknpCollisionQueryDispatcherBase::getClosestPoints(
	hknpCollisionQueryContext* queryContext,
	const hknpClosestPointsQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
	const hknpShape *targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
	const hkTransform& queryToTarget, bool queryAndTargetSwapped,
	hknpCollisionQueryCollector* collector ) const
{
	queryContext->m_dispatcher = this;
	_dispatchQuery<hknpCollisionQueryType::GET_CLOSEST_POINTS>(
		m_closestPointsDispatchTable, queryContext, query, queryShapeInfo, targetShape, targetShapeFilterData, targetShapeInfo, queryToTarget, queryAndTargetSwapped, collector );
}

/// Cast one shape against another shape.
void hknpCollisionQueryDispatcherBase::castShape(
	hknpCollisionQueryContext* queryContext,
	const hknpShapeCastQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
	const hknpShape *targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
	const hkTransform& queryToTarget, bool queryAndTargetSwapped,
	hknpCollisionQueryCollector* collector ) const
{
	queryContext->m_dispatcher = this;
	_dispatchQuery<hknpCollisionQueryType::SHAPE_CAST>(
		m_shapeCastDispatchTable, queryContext, query, queryShapeInfo, targetShape, targetShapeFilterData, targetShapeInfo, queryToTarget, queryAndTargetSwapped, collector );
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
