/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Common/Base/Math/Matrix/hkTransformUtil.h>

#include <Physics/Physics/Collide/Shape/TagCodec/hknpShapeTagCodec.h>
#include <Physics/Physics/Collide/Filter/hknpCollisionFilter.h>
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

namespace
{
	HK_FORCE_INLINE hknpCollisionQueryDispatcherBase::BaseType _getDefaultBaseType( hknpShapeType::Enum shapeType )
	{
		switch(shapeType)
		{
		case hknpShapeType::CONVEX					:
		case hknpShapeType::CONVEX_POLYTOPE			:
		case hknpShapeType::SPHERE					:
		case hknpShapeType::CAPSULE					:
		case hknpShapeType::TRIANGLE				:
			return hknpCollisionQueryDispatcherBase::CONVEX;
		case hknpShapeType::COMPRESSED_MESH			:
			return hknpCollisionQueryDispatcherBase::COMPRESSED_MESH;
		case hknpShapeType::EXTERN_MESH				:
			return hknpCollisionQueryDispatcherBase::EXTERN_MESH;
		case hknpShapeType::STATIC_COMPOUND			:
			return hknpCollisionQueryDispatcherBase::STATIC_COMPOUND;
		case hknpShapeType::DYNAMIC_COMPOUND		:
			return hknpCollisionQueryDispatcherBase::DYNAMIC_COMPOUND;
		case hknpShapeType::HEIGHT_FIELD			:
		case hknpShapeType::COMPRESSED_HEIGHT_FIELD	:
			return hknpCollisionQueryDispatcherBase::HEIGHT_FIELD;
		case hknpShapeType::USER_0					:
		case hknpShapeType::USER_1					:
		case hknpShapeType::USER_2					:
		case hknpShapeType::USER_3					:
			return hknpCollisionQueryDispatcherBase::USER;
		case hknpShapeType::MASKED_COMPOSITE		:
			return hknpCollisionQueryDispatcherBase::MASKED_COMPOSITE;
		case hknpShapeType::SCALED_CONVEX			:
			return hknpCollisionQueryDispatcherBase::SCALED_CONVEX;
		default:
			return hknpCollisionQueryDispatcherBase::NUM_BASE_TYPES;
		}
	}

	// This function checks the collision filters between two shapes
	template<hknpCollisionQueryType::Enum QUERY_TYPE, typename QUERY>
	HK_FORCE_INLINE bool _isCollisionEnabled( const QUERY& query, const hknpShapeQueryInfo& queryShapeInfo,
		const hknpShape *targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo, bool queryAndTargetSwapped )
	{
		if ( query.m_filter )
		{
			hknpCollisionFilter::FilterInput shapeFilterInputA;
			shapeFilterInputA.m_filterData		= query.m_filterData;
			shapeFilterInputA.m_body			= queryShapeInfo.m_body;
			shapeFilterInputA.m_rootShape		= queryShapeInfo.m_rootShape;
			shapeFilterInputA.m_parentShape		= queryShapeInfo.m_parentShape;
			shapeFilterInputA.m_shapeKey		= queryShapeInfo.m_shapeKeyPath.getKey();
			shapeFilterInputA.m_shape			= query.m_shape;

			hknpCollisionFilter::FilterInput shapeFilterInputB;
			shapeFilterInputB.m_filterData		= targetShapeFilterData;
			shapeFilterInputB.m_body			= targetShapeInfo.m_body;
			shapeFilterInputB.m_rootShape		= targetShapeInfo.m_rootShape;
			shapeFilterInputB.m_parentShape		= targetShapeInfo.m_parentShape;
			shapeFilterInputB.m_shapeKey		= targetShapeInfo.m_shapeKeyPath.getKey();
			shapeFilterInputB.m_shape			= targetShape;

			if ( !query.m_filter->isCollisionEnabled( QUERY_TYPE, !queryAndTargetSwapped, shapeFilterInputA, shapeFilterInputB ) )
			{
				return false;
			}
		}

		return true;
	}

	// This function dispatches the corresponding function pointer based on the query
	template <hknpCollisionQueryType::Enum QUERY_TYPE, typename QUERY, typename FUNCTION_TYPE>
	HK_FORCE_INLINE void _dispatchQuery(
		const hknpCollisionQueryDispatcherBase::DispatchTable<FUNCTION_TYPE> dispatchTable,
		hknpCollisionQueryContext* queryContext,
		const QUERY& query, const hknpShapeQueryInfo& queryShapeInfo,
		const hknpShape *targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
		const hkTransform& queryToTarget, bool queryAndTargetSwapped,
		hknpCollisionQueryCollector* collector )
	{
		if ( _isCollisionEnabled<QUERY_TYPE>(query, queryShapeInfo, targetShape, targetShapeFilterData, targetShapeInfo, queryAndTargetSwapped) )
		{
			hknpCollisionQueryDispatcherBase::BaseType queryShapeType = queryContext->m_dispatcher->getBaseShapeType(query.m_shape->getType());
			hknpCollisionQueryDispatcherBase::BaseType targetShapeType = queryContext->m_dispatcher->getBaseShapeType(targetShape->getType());

			FUNCTION_TYPE func = dispatchTable.getFunction(queryShapeType, targetShapeType);
			func(queryContext, query, queryShapeInfo, targetShape, targetShapeFilterData, targetShapeInfo, queryToTarget, queryAndTargetSwapped, collector);
		}
	}

	// This function is unimplemented
	template <typename QUERY>
	void notImplemented(
		hknpCollisionQueryContext* queryContext,
		const QUERY& query, const hknpShapeQueryInfo &queryShapeInfo,
		const hknpShape* targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo &targetShapeInfo,
		const hkTransform& queryToTarget, bool queryAndTargetSwapped,
		hknpCollisionQueryCollector* collector )
	{
		HK_ASSERT3( 0xaf1ee133, false, "Collision query function not implemented for shape types " << (int)query.m_shape->getType() << " vs " << (int)targetShape->getType() );
	}

	// The shape to unwrap when 'resolving' mask composite shapes, LOD shapes or scaled convex shapes.
	enum UnwrapMode
	{
		UNWRAP_QUERY,
		UNWRAP_TARGET,
		UNWRAP_BOTH,
	};

	
	HK_FORCE_INLINE void _unwrapMaskedCompositeShape(const hknpShape* &shape, hknpShapeQueryInfo* shapeInfo)
	{
		const hknpMaskedCompositeShape* maskedCompositeQueryShape = (const hknpMaskedCompositeShape*)shape;
		shapeInfo->m_shapeKeyMask = maskedCompositeQueryShape->m_mask;
		shape = maskedCompositeQueryShape->m_shape;
	}

	// This function unwraps a masked composite shape
	template <UnwrapMode UNWRAP_MODE, hknpCollisionQueryType::Enum QUERY_TYPE, typename FUNCTION_TYPE, typename QUERY>
	HK_FORCE_INLINE void unwrapMaskedCompositeShape(
		hknpCollisionQueryContext* queryContext,
		const QUERY& query, const hknpShapeQueryInfo& queryShapeInfo,
		const hknpShape* targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
		const hkTransform& queryToTarget, bool queryAndTargetSwapped,
		hknpCollisionQueryCollector* collector )
	{
		QUERY updatedQuery = query;
		hknpShapeQueryInfo updatedQueryShapeInfo(&queryShapeInfo);
		hknpShapeQueryInfo updatedTargetShapeInfo(&targetShapeInfo);

		if ((UNWRAP_MODE == UNWRAP_QUERY) || (UNWRAP_MODE == UNWRAP_BOTH))
		{
			_unwrapMaskedCompositeShape(updatedQuery.m_shape.ref(), &updatedQueryShapeInfo);
		}

		if ((UNWRAP_MODE == UNWRAP_TARGET) || (UNWRAP_MODE == UNWRAP_BOTH))
		{
			_unwrapMaskedCompositeShape(targetShape, &updatedTargetShapeInfo);
		}

		const hknpCollisionQueryDispatcherBase::DispatchTable<FUNCTION_TYPE>* dispatchtable = HK_NULL;
		if (QUERY_TYPE == hknpCollisionQueryType::GET_CLOSEST_POINTS)
		{
			dispatchtable = (const hknpCollisionQueryDispatcherBase::DispatchTable<FUNCTION_TYPE>*)&queryContext->m_dispatcher->m_closestPointsDispatchTable;
		}
		else if (QUERY_TYPE == hknpCollisionQueryType::SHAPE_CAST)
		{
			dispatchtable = (const hknpCollisionQueryDispatcherBase::DispatchTable<FUNCTION_TYPE>*)&queryContext->m_dispatcher->m_shapeCastDispatchTable;
		}
		HK_ASSERT2(0x4faac327, dispatchtable, "Unsupported query type!");

		_dispatchQuery<QUERY_TYPE>(
			*dispatchtable, queryContext, updatedQuery, updatedQueryShapeInfo, targetShape, targetShapeFilterData, updatedTargetShapeInfo, queryToTarget, queryAndTargetSwapped, collector );
	}

	// This function unwraps a scaled convex shape
	template <UnwrapMode UNWRAP_MODE, hknpCollisionQueryType::Enum QUERY_TYPE, typename FUNCTION_TYPE, typename QUERY>
	HK_FORCE_INLINE void unwrapScaledConvexShape(
		hknpCollisionQueryContext* queryContext,
		const QUERY& query, const hknpShapeQueryInfo& queryShapeInfo,
		const hknpShape* targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
		const hkTransform& queryToTarget, bool queryAndTargetSwapped,
		hknpCollisionQueryCollector* collector )
	{
		QUERY updatedQuery = query;
		hknpShapeQueryInfo queryShapeInfoWithScale(&queryShapeInfo);
		hknpShapeQueryInfo targetShapeInfoWithScale(&targetShapeInfo);

		if ((UNWRAP_MODE == UNWRAP_QUERY) || (UNWRAP_MODE == UNWRAP_BOTH))
		{
			const hknpScaledConvexShape* scaledQueryShape = static_cast<const hknpScaledConvexShape*>(updatedQuery.m_shape.val());
			updatedQuery.m_shape = scaledQueryShape->getChildShape();
			const hkVector4& scale = scaledQueryShape->getScale();
			queryShapeInfoWithScale.m_shapeConvexRadius = scaledQueryShape->m_convexRadius;
			if ( !scale.allEqual<3>(hkVector4::getConstant<HK_QUADREAL_1>(), hkSimdReal_Eps) )
			{
				queryShapeInfoWithScale.m_shapeIsScaled = true;
				queryShapeInfoWithScale.m_shapeScale = scale;
				queryShapeInfoWithScale.m_shapeScaleOffset = scaledQueryShape->getTranslation();
			}
		}

		if ((UNWRAP_MODE == UNWRAP_TARGET) || (UNWRAP_MODE == UNWRAP_BOTH))
		{
			const hknpScaledConvexShape* scaledTargetShape = static_cast<const hknpScaledConvexShape*>(targetShape);
			targetShape = scaledTargetShape->getChildShape();
			const hkVector4& scale = scaledTargetShape->getScale();
			targetShapeInfoWithScale.m_shapeConvexRadius = scaledTargetShape->m_convexRadius;
			if ( !scale.allEqual<3>(hkVector4::getConstant<HK_QUADREAL_1>(), hkSimdReal_Eps) )
			{
				targetShapeInfoWithScale.m_shapeIsScaled = true;
				targetShapeInfoWithScale.m_shapeScale = scale;
				targetShapeInfoWithScale.m_shapeScaleOffset = scaledTargetShape->getTranslation();
			}
		}

		const hknpCollisionQueryDispatcherBase::DispatchTable<FUNCTION_TYPE>* dispatchtable = HK_NULL;
		if (QUERY_TYPE == hknpCollisionQueryType::GET_CLOSEST_POINTS)
		{
			dispatchtable = (const hknpCollisionQueryDispatcherBase::DispatchTable<FUNCTION_TYPE>*)&queryContext->m_dispatcher->m_closestPointsDispatchTable;
		}
		else if (QUERY_TYPE == hknpCollisionQueryType::SHAPE_CAST)
		{
			dispatchtable = (const hknpCollisionQueryDispatcherBase::DispatchTable<FUNCTION_TYPE>*)&queryContext->m_dispatcher->m_shapeCastDispatchTable;
		}
		HK_ASSERT2(0x4faac329, dispatchtable, "Unsupported query type!");

		_dispatchQuery<QUERY_TYPE>(
			*dispatchtable, queryContext, updatedQuery, queryShapeInfoWithScale, targetShape, targetShapeFilterData, targetShapeInfoWithScale, queryToTarget, queryAndTargetSwapped, collector );
	}

	namespace hknpClosestPointFuncs
	{
		// This function flips a closest point query
		template<hknpCollisionQueryDispatcherBase::ClosestPointsFunc func>
		void getClosestPointsFlipped(
			hknpCollisionQueryContext* queryContext,
			const hknpClosestPointsQuery& query, const hknpShapeQueryInfo &queryShapeInfo,
			const hknpShape* targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo &targetShapeInfo,
			const hkTransform& queryToTarget, bool queryAndTargetSwapped,
			hknpCollisionQueryCollector* collector )
		{
			hkTransform	targetToQuery; targetToQuery._setInverse( queryToTarget );

			hknpClosestPointsQuery flippedQuery	= query;
			flippedQuery.m_shape				= targetShape;
			flippedQuery.m_filterData			= targetShapeFilterData;
			hknpFlippedGetClosestPointsQueryCollector flippedCollector(collector);

			// We need to switch query and target triangles as the query context is non-const!
			hkAlgorithm::swap( queryContext->m_queryTriangle, queryContext->m_targetTriangle );

			func(queryContext, flippedQuery, targetShapeInfo, query.m_shape, query.m_filterData, queryShapeInfo, targetToQuery, !queryAndTargetSwapped, &flippedCollector);

			// Switch temp triangles back.
			hkAlgorithm::swap( queryContext->m_queryTriangle, queryContext->m_targetTriangle );
		}

		// This function collects the closest points between two convex shapes
		void convexVsConvex(
			hknpCollisionQueryContext* queryContext,
			const hknpClosestPointsQuery& query, const hknpShapeQueryInfo &queryShapeInfo,
			const hknpShape* targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo &targetShapeInfo,
			const hkTransform& queryToTarget, bool queryAndTargetSwapped,
			hknpCollisionQueryCollector* collector )
		{
			const hknpConvexShape* convexQueryShape = query.m_shape->asConvexShape();
			const hknpConvexShape* convexTargetShape = targetShape->asConvexShape();
			hknpConvexShapeUtil::getClosestPointsWithScale( convexQueryShape->getVertices(), convexQueryShape->getNumberOfVertices(), query.m_filterData, queryShapeInfo, convexTargetShape->getVertices(), convexTargetShape->getNumberOfVertices(), targetShapeFilterData, targetShapeInfo, queryToTarget, collector );
		}

		// This function collects the closest points from a compressed mesh to a convex shape

		void convexVsCompressedMesh(
			hknpCollisionQueryContext* queryContext,
			const hknpClosestPointsQuery& query, const hknpShapeQueryInfo &queryShapeInfo,
			const hknpShape* targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo &targetShapeInfo,
			const hkTransform& queryToTarget, bool queryAndTargetSwapped,
			hknpCollisionQueryCollector* collector )
		{
			const hknpCompressedMeshShape*	targetCms = static_cast<const hknpCompressedMeshShape*>(targetShape);

			hknpCompressedMeshShapeUtil::getClosestPointsToConvex( queryContext, query, queryShapeInfo, targetCms, targetShapeFilterData, targetShapeInfo, queryToTarget, queryAndTargetSwapped, collector );
		}

		// This function collects the closest points from an extern mesh to a convex shape
		void convexVsExternMesh(
			hknpCollisionQueryContext* queryContext,
			const hknpClosestPointsQuery& query, const hknpShapeQueryInfo &queryShapeInfo,
			const hknpShape* targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo &targetShapeInfo,
			const hkTransform& queryToTarget, bool queryAndTargetSwapped,
			hknpCollisionQueryCollector* collector )
		{
			const hknpExternMeshShape*	targetCms = static_cast<const hknpExternMeshShape*>(targetShape);

			hknpExternMeshShapeUtil::getClosestPointsToConvex( queryContext, query, queryShapeInfo, targetCms, targetShapeFilterData, targetShapeInfo, queryToTarget, queryAndTargetSwapped, collector );
		}

		// This function queries the closest point from a height field to a convex shape
		void convexVsHeightField(
			hknpCollisionQueryContext* queryContext,
			const hknpClosestPointsQuery& query, const hknpShapeQueryInfo &queryShapeInfo,
			const hknpShape* targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo &targetShapeInfo,
			const hkTransform& queryToTarget, bool queryAndTargetSwapped,
			hknpCollisionQueryCollector* collector )
		{
			const hknpHeightFieldShape*	targetShf = static_cast<const hknpHeightFieldShape*>(targetShape);

			hknpHeightFieldShape::getClosestPointsImpl( queryContext, query, queryShapeInfo, targetShf, targetShapeFilterData, targetShapeInfo, queryToTarget, queryAndTargetSwapped, collector );
		}

		// This function collects the closest points from a static compound shape to a convex shape
		void convexVsStaticCompound(
			hknpCollisionQueryContext* queryContext,
			const hknpClosestPointsQuery& query, const hknpShapeQueryInfo &queryShapeInfo,
			const hknpShape* targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo &targetShapeInfo,
			const hkTransform& queryToTarget, bool queryAndTargetSwapped,
			hknpCollisionQueryCollector* collector )
		{
			const hknpStaticCompoundShape*	targetScs = static_cast<const hknpStaticCompoundShape*>(targetShape);

			hknpStaticCompoundShape::getClosestPointsImpl( queryContext, query, queryShapeInfo, targetScs, targetShapeFilterData, targetShapeInfo, queryToTarget, queryAndTargetSwapped, collector );
		}

		// This function collects the closest points from a dynamic compound shape to a convex shape
		void convexVsDynamicCompound(
			hknpCollisionQueryContext* queryContext,
			const hknpClosestPointsQuery& query, const hknpShapeQueryInfo &queryShapeInfo,
			const hknpShape* targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo &targetShapeInfo,
			const hkTransform& queryToTarget, bool queryAndTargetSwapped,
			hknpCollisionQueryCollector* collector )
		{
			const hknpDynamicCompoundShape*	targetDcs = static_cast<const hknpDynamicCompoundShape*>(targetShape);

			hknpDynamicCompoundShape::getClosestPointsImpl( queryContext, query, queryShapeInfo, targetDcs, targetShapeFilterData, targetShapeInfo, queryToTarget, queryAndTargetSwapped, collector );
		}

		// This function collects the closest points between two compressed meshes
		void compressedMeshVsCompressedMesh(
			hknpCollisionQueryContext* queryContext,
			const hknpClosestPointsQuery& query, const hknpShapeQueryInfo &queryShapeInfo,
			const hknpShape* targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo &targetShapeInfo,
			const hkTransform& queryToTarget, bool queryAndTargetSwapped,
			hknpCollisionQueryCollector* collector )
		{
			const hknpCompressedMeshShape* targetCms = static_cast<const hknpCompressedMeshShape*>(targetShape);

			hknpCompressedMeshShapeUtil::getClosestPointsToCompressedMesh( queryContext, query, queryShapeInfo, targetCms, targetShapeFilterData, targetShapeInfo, queryToTarget, queryAndTargetSwapped, collector );
		}

		// This function collects the closest points from a compressed mesh to a height field
		void heightFieldVsCompressedMesh(
			hknpCollisionQueryContext* queryContext,
			const hknpClosestPointsQuery& query, const hknpShapeQueryInfo &queryShapeInfo,
			const hknpShape* targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo &targetShapeInfo,
			const hkTransform& queryToTarget, bool queryAndTargetSwapped,
			hknpCollisionQueryCollector* collector )
		{
			const hknpCompressedMeshShape* targetCms = static_cast<const hknpCompressedMeshShape*>(targetShape);

			hknpCompressedMeshShapeUtil::getClosestPointsToHeightfield( queryContext, query, queryShapeInfo, targetCms, targetShapeFilterData, targetShapeInfo, queryToTarget, queryAndTargetSwapped, collector );
		}

		// This function collects the closest points between two heightfields.
		void heightFieldVsHeightfield(
			hknpCollisionQueryContext* queryContext,
			const hknpClosestPointsQuery& query, const hknpShapeQueryInfo &queryShapeInfo,
			const hknpShape* targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo &targetShapeInfo,
			const hkTransform& queryToTarget, bool queryAndTargetSwapped,
			hknpCollisionQueryCollector* collector )
		{
			HK_ASSERT2( 0xaf133461, targetShape->getType() == hknpShapeType::HEIGHT_FIELD || targetShape->getType() == hknpShapeType::COMPRESSED_HEIGHT_FIELD, "Target shape has to be a heightfield.");

			const hknpHeightFieldShape* targetHfs = static_cast<const hknpHeightFieldShape*>(targetShape);

			hknpHeightFieldShape::getClosestPointsToHeightfieldImpl( queryContext, query, queryShapeInfo, targetHfs, targetShapeFilterData, targetShapeInfo, queryToTarget, queryAndTargetSwapped, collector );
		}

		// This function parses an extern mesh shape and queries its children against an arbitrary target shape.
		void externMeshVsShape(
			hknpCollisionQueryContext* queryContext,
			const hknpClosestPointsQuery& query, const hknpShapeQueryInfo &queryShapeInfo,
			const hknpShape* targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo &targetShapeInfo,
			const hkTransform& queryToTarget, bool queryAndTargetSwapped,
			hknpCollisionQueryCollector* collector )
		{
			const hknpExternMeshShape* queryEms = static_cast<const hknpExternMeshShape*>( query.m_shape.val() );
			const hknpExternMeshShape::Mesh* queryEmsSource = queryEms->getMesh();
			for ( int i = 0; i < queryEmsSource->getNumTriangles(); i++ )
			{
				hkVector4 triangleVertices[3];
				queryEmsSource->getTriangleVertices( i, triangleVertices );

				hknpTriangleShape* queryTriangleShape = queryContext->m_queryTriangle;
				queryTriangleShape->setVertices( triangleVertices[0], triangleVertices[1], triangleVertices[2] );
				queryTriangleShape->m_convexRadius = targetShape->m_convexRadius;

				hknpShapeQueryInfo triangleShapeInfo( &queryShapeInfo );
				triangleShapeInfo.m_shapeKeyPath	. appendSubKey( i, queryEms->getNumShapeKeyBits() );

				hknpClosestPointsQuery childQuery = query;
				childQuery.m_shape = queryTriangleShape;

				if ( queryContext->m_shapeTagCodec )
				{
					hknpShapeTag queryEmsTriangleShapeTag = queryEmsSource->getTriangleShapeTag( i );

					hknpShapeTagCodec::Context targetShapeTagContext;
					targetShapeTagContext.m_queryType			= hknpCollisionQueryType::GET_CLOSEST_POINTS;
					targetShapeTagContext.m_body				= queryShapeInfo.m_body;
					targetShapeTagContext.m_rootShape			= queryShapeInfo.m_rootShape;
					targetShapeTagContext.m_parentShape			= queryEms;
					targetShapeTagContext.m_shapeKey			= triangleShapeInfo.m_shapeKeyPath.getKey();
					targetShapeTagContext.m_shape				= queryTriangleShape;
					targetShapeTagContext.m_partnerBody			= targetShapeInfo.m_body;
					targetShapeTagContext.m_partnerRootShape	= targetShapeInfo.m_rootShape;
					targetShapeTagContext.m_partnerShapeKey		= targetShapeInfo.m_shapeKeyPath.getKey();
					targetShapeTagContext.m_partnerShape		= targetShape;

					queryContext->m_shapeTagCodec->decode(
						queryEmsTriangleShapeTag, &targetShapeTagContext,
						&childQuery.m_filterData.m_collisionFilterInfo.ref(),
						&childQuery.m_filterData.m_materialId,
						&childQuery.m_filterData.m_userData.ref());
				}

				HK_ASSERT2(0x1c83e692, queryContext->m_dispatcher, "Uninitialized dispatcher, please make sure you are not calling this function directly!");

				queryContext->m_dispatcher->getClosestPoints(
					queryContext,
					childQuery, triangleShapeInfo,
					targetShape, targetShapeFilterData, targetShapeInfo,
					queryToTarget, queryAndTargetSwapped,
					collector );
			}
		}

		// This function parses a compound shape and queries its children against a target shape
		void compoundVsShape(
			hknpCollisionQueryContext* queryContext,
			const hknpClosestPointsQuery& query, const hknpShapeQueryInfo &queryShapeInfo,
			const hknpShape* targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo &targetShapeInfo,
			const hkTransform& queryToTarget, bool queryAndTargetSwapped,
			hknpCollisionQueryCollector* collector )
		{
			const hknpCompoundShape* queryCmp = static_cast<const hknpCompoundShape*>(query.m_shape.val());
			for ( hknpShapeInstanceIterator iter = queryCmp->getShapeInstanceIterator(); iter.isValid(); iter.next() )
			{
				const hknpShapeInstance& queryChild = iter.getValue();

				const hkTransform& queryChildToCompound = queryChild.getTransform();
				hkTransform queryChildToTarget; queryChildToTarget.setMul( queryToTarget, queryChildToCompound );

				hkTransform	queryChildToWorld;
				hkTransformUtil::_mulTransformTransform( *queryShapeInfo.m_shapeToWorld, queryChildToCompound, &queryChildToWorld );

				const hknpShape* childShape = queryChild.getShape();

				hknpShapeQueryInfo instanceShapeInfo(&queryShapeInfo);
				instanceShapeInfo.m_shapeKeyPath	. appendSubKey( iter.getIndex().value(), queryCmp->getNumShapeKeyBits() );
				instanceShapeInfo.m_shapeToWorld	= &queryChildToWorld;
				instanceShapeInfo.m_shapeConvexRadius = childShape->m_convexRadius;

				hkVector4 scale = queryChild.getScale();
				if ( !scale.allEqual<3>(hkVector4::getConstant<HK_QUADREAL_1>(), hkSimdReal_Eps) )
				{
					HK_ASSERT2( 0xaf14e234, !queryShapeInfo.m_shapeIsScaled, "Scaling is only supported on the lowest level of a shape hierarchy. The previous scale will now be overridden." );
					instanceShapeInfo.m_shapeIsScaled = true;
					instanceShapeInfo.m_shapeScale = scale;
					instanceShapeInfo.m_shapeScaleOffset.setZero();
				}

				hknpClosestPointsQuery childQuery = query;
				childQuery.m_shape = childShape;

				if ( queryContext->m_shapeTagCodec )
				{
					hknpShapeTagCodec::Context targetShapeTagContext;
					targetShapeTagContext.m_queryType			= hknpCollisionQueryType::GET_CLOSEST_POINTS;
					targetShapeTagContext.m_body				= queryShapeInfo.m_body;
					targetShapeTagContext.m_rootShape			= queryShapeInfo.m_rootShape;
					targetShapeTagContext.m_parentShape			= queryCmp;
					targetShapeTagContext.m_shapeKey			= instanceShapeInfo.m_shapeKeyPath.getKey();
					targetShapeTagContext.m_shape				= childShape;
					targetShapeTagContext.m_partnerBody			= targetShapeInfo.m_body;
					targetShapeTagContext.m_partnerRootShape	= targetShapeInfo.m_rootShape;
					targetShapeTagContext.m_partnerShapeKey		= targetShapeInfo.m_shapeKeyPath.getKey();
					targetShapeTagContext.m_partnerShape		= targetShape;

					queryContext->m_shapeTagCodec->decode(
						queryChild.getShapeTag(), &targetShapeTagContext,
						&childQuery.m_filterData.m_collisionFilterInfo.ref(),
						&childQuery.m_filterData.m_materialId,
						&childQuery.m_filterData.m_userData.ref());
				}

				HK_ASSERT2(0x1c83e692, queryContext->m_dispatcher, "Uninitialized dispatcher, please make sure you are not calling this function directly!");
				queryContext->m_dispatcher->getClosestPoints( queryContext, childQuery, instanceShapeInfo, targetShape, targetShapeFilterData, targetShapeInfo, queryChildToTarget, queryAndTargetSwapped, collector );
			}
		}

		// This function collects the closest points between two compound shapes
		void compoundVsCompound(
			hknpCollisionQueryContext* queryContext,
			const hknpClosestPointsQuery& query, const hknpShapeQueryInfo &queryShapeInfo,
			const hknpShape* targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo &targetShapeInfo,
			const hkTransform& queryToTarget, bool queryAndTargetSwapped,
			hknpCollisionQueryCollector* collector )
		{
			if ( static_cast<const hknpCompoundShape*>(query.m_shape.val())->getNumShapeKeyBits() > static_cast<const hknpCompoundShape*>(targetShape)->getNumShapeKeyBits() )
			{
				getClosestPointsFlipped<compoundVsShape>(queryContext, query, queryShapeInfo, targetShape, targetShapeFilterData, targetShapeInfo, queryToTarget, queryAndTargetSwapped, collector);
			}
			else
			{
				compoundVsShape(queryContext, query, queryShapeInfo, targetShape, targetShapeFilterData, targetShapeInfo, queryToTarget, queryAndTargetSwapped, collector);
			}
		}

	}	// hknpClosestPointFuncs namespace

	namespace hknpCastShapeFuncs
	{
		// This function flips a shape cast query
		template<hknpCollisionQueryDispatcherBase::ShapeCastFunc func>
		void shapeCastFlipped(
			hknpCollisionQueryContext* queryContext,
			const hknpShapeCastQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
			const hknpShape* targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			const hkTransform& queryToTarget, bool queryAndTargetSwapped,
			hknpCollisionQueryCollector* collector )
		{
			hkVector4 flippedCastDirection;
			{
				flippedCastDirection._setRotatedInverseDir( queryToTarget.getRotation(), query.getDirection() );
				flippedCastDirection.setNeg<3>( flippedCastDirection );
			}

			hknpShapeCastQuery flippedQuery	= query;
			flippedQuery.m_filterData		= targetShapeFilterData;
			flippedQuery.m_direction		= flippedCastDirection;
			flippedQuery.m_body				= targetShapeInfo.m_body;
			flippedQuery.m_shape			= targetShape;

			hknpQueryFilterData	flippedTargetShapeFilterData = query.m_filterData;

			// We need to switch query and target triangles as the query context is non-const and inner calls
			// to castShapeImpl() could otherwise overwrite the input triangle that got us here.
			hkAlgorithm::swap( queryContext->m_queryTriangle, queryContext->m_targetTriangle );

			// This flip-back collector simply reverts the flipping of query and target and forwards the results to the
			// actual collector.
			hkVector4 castDirectionWS; castDirectionWS.setRotatedDir( targetShapeInfo.m_shapeToWorld->getRotation(), query.getDirection() );
			hknpFlippedShapeCastQueryCollector flipBackCollector( castDirectionWS, collector );

			func( queryContext, flippedQuery, targetShapeInfo, query.m_shape, flippedTargetShapeFilterData, queryShapeInfo, queryToTarget, !queryAndTargetSwapped, &flipBackCollector );

			// Switch temp triangles back.
			hkAlgorithm::swap( queryContext->m_queryTriangle, queryContext->m_targetTriangle );
		}

		// This function templates the cast shape implementation of each shape class
		template<typename SHAPE_TYPE>
		void castShapeFunc(
			hknpCollisionQueryContext* queryContext,
			const hknpShapeCastQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
			const hknpShape* targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			const hkTransform& queryToTarget, bool queryAndTargetSwapped,
			hknpCollisionQueryCollector* collector )
		{
			SHAPE_TYPE::castShapeImpl(queryContext, query, queryShapeInfo, targetShape, targetShapeFilterData, targetShapeInfo, queryAndTargetSwapped, collector);
		}

		// This function casts a convex shape against another convex shape
		// Note that at this stage only the *direction* of the query's underlying ray is used. The *origin* is taken
		// from the relative transform between query and target shape.
		void castShapeConvex(
			hknpCollisionQueryContext* queryContext,
			const hknpShapeCastQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
			const hknpShape* targetShape, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
			const hkTransform& queryToTarget, bool queryAndTargetSwapped,
			hknpCollisionQueryCollector* collector )
		{
			const hknpConvexShape* queryShape = static_cast<const hknpConvexShape*>( query.m_shape.val() );
			const hknpConvexShape* targetCvxShape = static_cast<const hknpConvexShape*>( targetShape );

			hkTransform	updatedQueryToTarget = queryToTarget;
			bool		hasHit = false;
			hkSimdReal	castFraction; castFraction.setZero();
			hkSimdReal	distance = hkSimdReal_Max;
			hkVector4	normal;	normal.setZero();
			hkVector4	point;	point.setZero();

			int numQueryShapeVertices = queryShape->getNumberOfVertices();
			int numTargetShapeVertices = targetCvxShape->getNumberOfVertices();

			const hkcdVertex* queryShapeVertices = queryShape->getVertices();
			const hkcdVertex* targetShapeVertices = targetCvxShape->getVertices();

			hkReal queryShapeConvexRadius = queryShape->m_convexRadius;
			hkReal targetShapeConvexRadius = targetCvxShape->m_convexRadius;

			// If the query shape has a scale factor, we need to bake the scale into the shape's vertices before calling GSK.
			if ( queryShapeInfo.m_shapeIsScaled )
			{
				hkcdVertex* queryShapeVerticesBuffer = hkAllocateStack<hkcdVertex>(numQueryShapeVertices, "castConvexShapeScaledQueryShapeVerticesBuffer");
				for (int i = 0; i < numQueryShapeVertices; i++)
				{
					const hkcdVertex& originalVertex = queryShape->getVertex(i);
					hkVector4 scaledVertex;
					scaledVertex.setMul(originalVertex, queryShapeInfo.m_shapeScale);
					scaledVertex.add(queryShapeInfo.m_shapeScaleOffset);
					queryShapeVerticesBuffer[i].setXYZ_W( scaledVertex, originalVertex );
				}
				queryShapeVertices = queryShapeVerticesBuffer;
			}

			// If the target shape has a scale factor, we need to bake the scale into the shape's vertices before calling GSK.
			if ( targetShapeInfo.m_shapeIsScaled )
			{
				hkcdVertex* targetShapeVerticesBuffer = hkAllocateStack<hkcdVertex>(numTargetShapeVertices, "castConvexShapeScaledTargetShapeVerticesBuffer");
				for (int i = 0; i < numTargetShapeVertices; i++)
				{
					const hkcdVertex& originalVertex = targetCvxShape->getVertex(i);
					hkVector4 scaledVertex;
					scaledVertex.setMul(originalVertex, targetShapeInfo.m_shapeScale);
					scaledVertex.add(targetShapeInfo.m_shapeScaleOffset);
					targetShapeVerticesBuffer[i].setXYZ_W( scaledVertex, originalVertex );
				}
				targetShapeVertices = targetShapeVerticesBuffer;
			}
			int maxIterations = 256;
			for (int iterations = 0; iterations < maxIterations; ++iterations)
			{
				if ( !hknpConvexShapeUtil::getClosestPoints(
					queryShapeVertices, numQueryShapeVertices, queryShapeConvexRadius,
					targetShapeVertices, numTargetShapeVertices, targetShapeConvexRadius,
					updatedQueryToTarget,
					&distance, &normal, &point ) )
				{
					// Abort once the query shape is moving 'away' from the target shape (i.e. the distance is increasing again).
					break;
				}

				if ( distance.getReal() <= query.m_accuracy )
				{
					// We have a valid hit.
					hasHit = true;
					break;
				}

				hkSimdReal projCastDir = query.getDirection().dot<3>( normal );

				if ( projCastDir.isGreaterEqualZero() )
				{
					// Abort once the query shape has moved 'past' the target shape (or if it is already 'past' even
					// before the first iteration.)
					break;
				}

				hkSimdReal newFraction = castFraction - distance/projCastDir;

				if ( newFraction.isLessEqual( collector->getEarlyOutHitFraction() ) )
				{
					castFraction = newFraction;
					updatedQueryToTarget.getTranslation().setAddMul( queryToTarget.getTranslation(), query.getDirection(), castFraction );
				}
				else
				{
					// Abort once the minimum distance we have to cover to reach a (potential) hit with the current
					// target shape is already larger than the distance to the currently closest hit's (shape) position.
					break;
				}
			}

			if ( targetShapeInfo.m_shapeIsScaled )
			{
				hkDeallocateStack<hkcdVertex>(const_cast<hkcdVertex*>(targetShapeVertices), numTargetShapeVertices);
			}
			if ( queryShapeInfo.m_shapeIsScaled )
			{
				hkDeallocateStack<hkcdVertex>(const_cast<hkcdVertex*>(queryShapeVertices), numQueryShapeVertices);
			}

			if ( hasHit )
			{
				hknpCollisionResult hit;
				{
					hit.m_queryType									= hknpCollisionQueryType::SHAPE_CAST;

					hit.m_position									. _setTransformedPos( *targetShapeInfo.m_shapeToWorld, point );
					hit.m_normal									. _setRotatedDir( targetShapeInfo.m_shapeToWorld->getRotation(), normal );
					hit.m_fraction									= castFraction.getReal();

					hit.m_queryBodyInfo.m_bodyId					= ( queryShapeInfo.m_body ? queryShapeInfo.m_body->m_id : hknpBodyId::INVALID );
					hit.m_queryBodyInfo.m_shapeKey					= queryShapeInfo.m_shapeKeyPath.getKey();
					hit.m_queryBodyInfo.m_shapeMaterialId			= query.m_filterData.m_materialId;
					hit.m_queryBodyInfo.m_shapeCollisionFilterInfo	= query.m_filterData.m_collisionFilterInfo;
					hit.m_queryBodyInfo.m_shapeUserData				= query.m_filterData.m_userData;

					hit.m_hitBodyInfo.m_bodyId						= ( targetShapeInfo.m_body ? targetShapeInfo.m_body->m_id : hknpBodyId::INVALID );
					hit.m_hitBodyInfo.m_shapeKey					= targetShapeInfo.m_shapeKeyPath.getKey();
					hit.m_hitBodyInfo.m_shapeMaterialId				= targetShapeFilterData.m_materialId;
					hit.m_hitBodyInfo.m_shapeCollisionFilterInfo	= targetShapeFilterData.m_collisionFilterInfo;
					hit.m_hitBodyInfo.m_shapeUserData				= targetShapeFilterData.m_userData;
				}

				collector->addHit( hit );
			}
		}
	}	// hknpCastShapeFuncs namespace

	// this function unwraps shapes with hierarchy
	template <hknpCollisionQueryType::Enum QUERY_TYPE, typename FUNCTION_TYPE>
	HK_FORCE_INLINE void _initializeUnwrapFunctions(hknpCollisionQueryDispatcherBase::ShapeMask shapeMask, hknpCollisionQueryDispatcherBase::DispatchTable<FUNCTION_TYPE> &dispatchTable)
	{
		// Mask Composite Shape
		for (int i = 0; i < hknpCollisionQueryDispatcherBase::NUM_BASE_TYPES; i++)
		{
			dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::MASKED_COMPOSITE, hknpCollisionQueryDispatcherBase::BaseType(i),
				unwrapMaskedCompositeShape<UNWRAP_QUERY, QUERY_TYPE, FUNCTION_TYPE>, shapeMask);
		}
		for (int i = 0; i < hknpCollisionQueryDispatcherBase::NUM_BASE_TYPES; i++)
		{
			dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::BaseType(i), hknpCollisionQueryDispatcherBase::MASKED_COMPOSITE,
				unwrapMaskedCompositeShape<UNWRAP_TARGET, QUERY_TYPE, FUNCTION_TYPE>, shapeMask);
		}
		dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::MASKED_COMPOSITE, hknpCollisionQueryDispatcherBase::MASKED_COMPOSITE,
			unwrapMaskedCompositeShape<UNWRAP_BOTH, QUERY_TYPE, FUNCTION_TYPE>, shapeMask);

		// Scaled Convex Shape
		for (int i = 0; i < hknpCollisionQueryDispatcherBase::NUM_BASE_TYPES; i++)
		{
			dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::SCALED_CONVEX, hknpCollisionQueryDispatcherBase::BaseType(i),
				unwrapScaledConvexShape<UNWRAP_QUERY, QUERY_TYPE, FUNCTION_TYPE>, shapeMask);
		}
		for (int i = 0; i < hknpCollisionQueryDispatcherBase::NUM_BASE_TYPES; i++)
		{
			dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::BaseType(i), hknpCollisionQueryDispatcherBase::SCALED_CONVEX,
				unwrapScaledConvexShape<UNWRAP_TARGET, QUERY_TYPE, FUNCTION_TYPE>, shapeMask);
		}
		dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::SCALED_CONVEX, hknpCollisionQueryDispatcherBase::SCALED_CONVEX,
			unwrapScaledConvexShape<UNWRAP_BOTH, QUERY_TYPE, FUNCTION_TYPE>, shapeMask);
	}

	// This function sets the get closest points dispatch table to default implementation
	HK_FORCE_INLINE void _initializeGetClosestPoints(hknpCollisionQueryDispatcherBase::ShapeMask shapeMask, hknpCollisionQueryDispatcherBase::DispatchTable<hknpCollisionQueryDispatcherBase::ClosestPointsFunc> &dispatchTable)
	{
		
		// Default to unimplemented function
		for (int i = 0; i < hknpCollisionQueryDispatcherBase::NUM_BASE_TYPES; i++)
		{
			for (int j = 0; j < hknpCollisionQueryDispatcherBase::NUM_BASE_TYPES; j++)
			{
				dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::BaseType(i), hknpCollisionQueryDispatcherBase::BaseType(j), notImplemented, hknpCollisionQueryDispatcherBase::defaultShapeMask);
			}
		}

		// Convex to convex
		dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::CONVEX, hknpCollisionQueryDispatcherBase::CONVEX, hknpClosestPointFuncs::convexVsConvex, shapeMask);
		// Convex to mesh
		dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::CONVEX, hknpCollisionQueryDispatcherBase::COMPRESSED_MESH, hknpClosestPointFuncs::convexVsCompressedMesh, shapeMask);
		dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::COMPRESSED_MESH, hknpCollisionQueryDispatcherBase::CONVEX, hknpClosestPointFuncs::getClosestPointsFlipped<hknpClosestPointFuncs::convexVsCompressedMesh>, shapeMask);
		dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::CONVEX, hknpCollisionQueryDispatcherBase::EXTERN_MESH, hknpClosestPointFuncs::convexVsExternMesh, shapeMask);
		dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::EXTERN_MESH, hknpCollisionQueryDispatcherBase::CONVEX, hknpClosestPointFuncs::getClosestPointsFlipped<hknpClosestPointFuncs::convexVsExternMesh>, shapeMask);

		// Convex to compound
		dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::CONVEX, hknpCollisionQueryDispatcherBase::STATIC_COMPOUND, hknpClosestPointFuncs::convexVsStaticCompound, shapeMask);
		dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::STATIC_COMPOUND, hknpCollisionQueryDispatcherBase::CONVEX, hknpClosestPointFuncs::getClosestPointsFlipped<hknpClosestPointFuncs::convexVsStaticCompound>, shapeMask);
		dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::CONVEX, hknpCollisionQueryDispatcherBase::DYNAMIC_COMPOUND, hknpClosestPointFuncs::convexVsDynamicCompound, shapeMask);
		dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::DYNAMIC_COMPOUND, hknpCollisionQueryDispatcherBase::CONVEX, hknpClosestPointFuncs::getClosestPointsFlipped<hknpClosestPointFuncs::convexVsDynamicCompound>, shapeMask);
		// Convex to height field
		dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::CONVEX, hknpCollisionQueryDispatcherBase::HEIGHT_FIELD, hknpClosestPointFuncs::convexVsHeightField, shapeMask);
		dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::HEIGHT_FIELD, hknpCollisionQueryDispatcherBase::CONVEX, hknpClosestPointFuncs::getClosestPointsFlipped<hknpClosestPointFuncs::convexVsHeightField>, shapeMask);

		// Compressed mesh to compressed mesh
		dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::COMPRESSED_MESH, hknpCollisionQueryDispatcherBase::COMPRESSED_MESH, hknpClosestPointFuncs::compressedMeshVsCompressedMesh, shapeMask);

		// Height field to compressed mesh
		dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::HEIGHT_FIELD, hknpCollisionQueryDispatcherBase::COMPRESSED_MESH, hknpClosestPointFuncs::heightFieldVsCompressedMesh, shapeMask);
		dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::COMPRESSED_MESH, hknpCollisionQueryDispatcherBase::HEIGHT_FIELD, hknpClosestPointFuncs::getClosestPointsFlipped<hknpClosestPointFuncs::heightFieldVsCompressedMesh>, shapeMask);
		// Height field vs Height field.
		dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::HEIGHT_FIELD, hknpCollisionQueryDispatcherBase::HEIGHT_FIELD, hknpClosestPointFuncs::heightFieldVsHeightfield, shapeMask);

		// Extern Mesh vs Extern Mesh.
		dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::EXTERN_MESH, hknpCollisionQueryDispatcherBase::EXTERN_MESH, hknpClosestPointFuncs::externMeshVsShape, shapeMask);
		// Extern Mesh vs Compressed Mesh.
		dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::EXTERN_MESH, hknpCollisionQueryDispatcherBase::COMPRESSED_MESH, hknpClosestPointFuncs::externMeshVsShape, shapeMask);
		dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::COMPRESSED_MESH, hknpCollisionQueryDispatcherBase::EXTERN_MESH, hknpClosestPointFuncs::getClosestPointsFlipped<hknpClosestPointFuncs::externMeshVsShape>, shapeMask);
		// Extern Mesh vs Heightfield.
		dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::EXTERN_MESH, hknpCollisionQueryDispatcherBase::HEIGHT_FIELD, hknpClosestPointFuncs::externMeshVsShape, shapeMask);
		dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::HEIGHT_FIELD, hknpCollisionQueryDispatcherBase::EXTERN_MESH, hknpClosestPointFuncs::getClosestPointsFlipped<hknpClosestPointFuncs::externMeshVsShape>, shapeMask);

		// Static Compound to mesh
		dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::STATIC_COMPOUND, hknpCollisionQueryDispatcherBase::COMPRESSED_MESH, hknpClosestPointFuncs::compoundVsShape, shapeMask);
		dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::COMPRESSED_MESH, hknpCollisionQueryDispatcherBase::STATIC_COMPOUND, hknpClosestPointFuncs::getClosestPointsFlipped<hknpClosestPointFuncs::compoundVsShape>, shapeMask);
		dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::STATIC_COMPOUND, hknpCollisionQueryDispatcherBase::EXTERN_MESH, hknpClosestPointFuncs::compoundVsShape, shapeMask);
		dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::EXTERN_MESH, hknpCollisionQueryDispatcherBase::STATIC_COMPOUND, hknpClosestPointFuncs::getClosestPointsFlipped<hknpClosestPointFuncs::compoundVsShape>, shapeMask);
		// Static Compound to height field
		dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::STATIC_COMPOUND, hknpCollisionQueryDispatcherBase::HEIGHT_FIELD, hknpClosestPointFuncs::compoundVsShape, shapeMask);
		dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::HEIGHT_FIELD, hknpCollisionQueryDispatcherBase::STATIC_COMPOUND, hknpClosestPointFuncs::getClosestPointsFlipped<hknpClosestPointFuncs::compoundVsShape>, shapeMask);

		// Static Compound to compound
		dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::STATIC_COMPOUND, hknpCollisionQueryDispatcherBase::STATIC_COMPOUND, hknpClosestPointFuncs::compoundVsCompound, shapeMask);
		dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::STATIC_COMPOUND, hknpCollisionQueryDispatcherBase::DYNAMIC_COMPOUND, hknpClosestPointFuncs::compoundVsCompound, shapeMask);

		// Dynamic Compound to mesh
		dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::DYNAMIC_COMPOUND, hknpCollisionQueryDispatcherBase::COMPRESSED_MESH, hknpClosestPointFuncs::compoundVsShape, shapeMask);
		dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::COMPRESSED_MESH, hknpCollisionQueryDispatcherBase::DYNAMIC_COMPOUND, hknpClosestPointFuncs::getClosestPointsFlipped<hknpClosestPointFuncs::compoundVsShape>, shapeMask);
		dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::DYNAMIC_COMPOUND, hknpCollisionQueryDispatcherBase::EXTERN_MESH, hknpClosestPointFuncs::compoundVsShape, shapeMask);
		dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::EXTERN_MESH, hknpCollisionQueryDispatcherBase::DYNAMIC_COMPOUND, hknpClosestPointFuncs::getClosestPointsFlipped<hknpClosestPointFuncs::compoundVsShape>, shapeMask);
		// Dynamic Compound to compound
		dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::DYNAMIC_COMPOUND, hknpCollisionQueryDispatcherBase::STATIC_COMPOUND, hknpClosestPointFuncs::compoundVsCompound, shapeMask);
		dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::DYNAMIC_COMPOUND, hknpCollisionQueryDispatcherBase::DYNAMIC_COMPOUND, hknpClosestPointFuncs::compoundVsCompound, shapeMask);
		// Dynamic Compound to height field
		dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::DYNAMIC_COMPOUND, hknpCollisionQueryDispatcherBase::HEIGHT_FIELD, hknpClosestPointFuncs::compoundVsShape, shapeMask);
		dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::HEIGHT_FIELD, hknpCollisionQueryDispatcherBase::DYNAMIC_COMPOUND, hknpClosestPointFuncs::getClosestPointsFlipped<hknpClosestPointFuncs::compoundVsShape>, shapeMask);

		// Wrapped Shape
		_initializeUnwrapFunctions<hknpCollisionQueryType::GET_CLOSEST_POINTS>(shapeMask, dispatchTable);
	}

	// This function sets the shape cast dispatch table to default implementation
	HK_FORCE_INLINE void _initializeShapeCast(hknpCollisionQueryDispatcherBase::ShapeMask shapeMask, hknpCollisionQueryDispatcherBase::DispatchTable<hknpCollisionQueryDispatcherBase::ShapeCastFunc> &dispatchTable)
	{
		
		hknpCollisionQueryDispatcherBase::ShapeCastFunc func = notImplemented;
		hknpCollisionQueryDispatcherBase::ShapeCastFunc flippedFunc = notImplemented;

		// Default to unimplemented function
		for (int i = 0; i < hknpCollisionQueryDispatcherBase::NUM_BASE_TYPES; i++)
		{
			for (int j = 0; j < hknpCollisionQueryDispatcherBase::NUM_BASE_TYPES; j++)
			{
				dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::BaseType(i), hknpCollisionQueryDispatcherBase::BaseType(j), func, hknpCollisionQueryDispatcherBase::defaultShapeMask);
			}
		}

		// Convex to convex case
		func = hknpCastShapeFuncs::castShapeConvex;
		dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::CONVEX, hknpCollisionQueryDispatcherBase::CONVEX, func, shapeMask);

		// Height field case
		if(shapeMask & (1<<hknpCollisionQueryDispatcherBase::HEIGHT_FIELD))
		{
			func = hknpCastShapeFuncs::castShapeFunc<hknpHeightFieldShape>;
			flippedFunc = hknpCastShapeFuncs::shapeCastFlipped< hknpCastShapeFuncs::castShapeFunc<hknpHeightFieldShape> >;
			for (int i = 0; i < hknpCollisionQueryDispatcherBase::USER; i++)
			{
				if (hknpCollisionQueryDispatcherBase::BaseType(i) == hknpCollisionQueryDispatcherBase::CONVEX)
				{
					dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::HEIGHT_FIELD, hknpCollisionQueryDispatcherBase::CONVEX, flippedFunc, shapeMask);
				}
				dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::BaseType(i), hknpCollisionQueryDispatcherBase::HEIGHT_FIELD, func, shapeMask);
			}
		}

		// Some shape to compressed mesh shape, reflecting convex shape
		if(shapeMask & (1<<hknpCollisionQueryDispatcherBase::COMPRESSED_MESH))
		{
			func = hknpCastShapeFuncs::castShapeFunc<hknpCompressedMeshShape>;
			flippedFunc = hknpCastShapeFuncs::shapeCastFlipped< hknpCastShapeFuncs::castShapeFunc<hknpCompressedMeshShape> >;
			for (int i = 0; i < hknpCollisionQueryDispatcherBase::USER; i++)
			{
				if (hknpCollisionQueryDispatcherBase::BaseType(i) == hknpCollisionQueryDispatcherBase::CONVEX)
				{
					dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::COMPRESSED_MESH, hknpCollisionQueryDispatcherBase::CONVEX, flippedFunc, shapeMask);
				}
				dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::BaseType(i), hknpCollisionQueryDispatcherBase::COMPRESSED_MESH, func, shapeMask);
			}
		}

		// Some shape to extern mesh shape, reflecting convex shape
		if(shapeMask & (1<<hknpCollisionQueryDispatcherBase::EXTERN_MESH))
		{
			func = hknpCastShapeFuncs::castShapeFunc<hknpExternMeshShape>;
			flippedFunc = hknpCastShapeFuncs::shapeCastFlipped<hknpCastShapeFuncs::castShapeFunc<hknpExternMeshShape> >;
			for (int i = 0; i < hknpCollisionQueryDispatcherBase::USER; i++)
			{
				if (hknpCollisionQueryDispatcherBase::BaseType(i) == hknpCollisionQueryDispatcherBase::CONVEX)
				{
					dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::EXTERN_MESH, hknpCollisionQueryDispatcherBase::CONVEX, flippedFunc, shapeMask);
				}
				dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::BaseType(i), hknpCollisionQueryDispatcherBase::EXTERN_MESH, func, shapeMask);
			}
		}

		// Some shape to static compound shape, reflecting convex shape
		if(shapeMask & (1<<hknpCollisionQueryDispatcherBase::STATIC_COMPOUND))
		{
			func = hknpCastShapeFuncs::castShapeFunc<hknpStaticCompoundShape>;
			flippedFunc = hknpCastShapeFuncs::shapeCastFlipped<hknpCastShapeFuncs::castShapeFunc<hknpStaticCompoundShape> >;
			for (int i = 0; i < hknpCollisionQueryDispatcherBase::USER; i++)
			{
				if (hknpCollisionQueryDispatcherBase::BaseType(i) == hknpCollisionQueryDispatcherBase::CONVEX)
				{
					dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::STATIC_COMPOUND, hknpCollisionQueryDispatcherBase::CONVEX, flippedFunc, shapeMask);
				}
				dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::BaseType(i), hknpCollisionQueryDispatcherBase::STATIC_COMPOUND, func, shapeMask);
			}
		}

		// Some shape to static compound shape, reflecting convex shape
		if(shapeMask & (1<<hknpCollisionQueryDispatcherBase::DYNAMIC_COMPOUND))
		{
			func = hknpCastShapeFuncs::castShapeFunc<hknpDynamicCompoundShape>;
			flippedFunc = hknpCastShapeFuncs::shapeCastFlipped<hknpCastShapeFuncs::castShapeFunc<hknpDynamicCompoundShape> >;
			for (int i = 0; i < hknpCollisionQueryDispatcherBase::USER; i++)
			{
				if (hknpCollisionQueryDispatcherBase::BaseType(i) == hknpCollisionQueryDispatcherBase::CONVEX)
				{
					dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::DYNAMIC_COMPOUND, hknpCollisionQueryDispatcherBase::CONVEX, flippedFunc, shapeMask);
				}
				dispatchTable.setFunction(hknpCollisionQueryDispatcherBase::BaseType(i), hknpCollisionQueryDispatcherBase::DYNAMIC_COMPOUND, func, shapeMask);
			}
		}

		// Wrapped Shape
		_initializeUnwrapFunctions<hknpCollisionQueryType::SHAPE_CAST>(shapeMask, dispatchTable);
	}

}	// anonymous namespace


HK_FORCE_INLINE hknpCollisionQueryDispatcherBase::hknpCollisionQueryDispatcherBase()
{
	// Initialize the dispatcher to default values
	for (int i = 0; i < hknpShapeType::NUM_SHAPE_TYPES; i++)
	{
		hknpShapeType::Enum shapeType = hknpShapeType::Enum(i);
		setBaseShapeType( shapeType, _getDefaultBaseType(shapeType) );
	}
}

HK_FORCE_INLINE hknpCollisionQueryDispatcherBase::~hknpCollisionQueryDispatcherBase()
{
}

HK_FORCE_INLINE hknpGetClosestPointsQueryDispatcher::hknpGetClosestPointsQueryDispatcher(ShapeMask shapeMask)
	:	hknpCollisionQueryDispatcherBase()
{
	_initializeGetClosestPoints( shapeMask, m_closestPointsDispatchTable );
}

HK_FORCE_INLINE hknpShapeCastQueryDispatcher::hknpShapeCastQueryDispatcher(ShapeMask shapeMask)
	:	hknpCollisionQueryDispatcherBase()
{
	_initializeShapeCast( shapeMask, m_shapeCastDispatchTable );
}

HK_FORCE_INLINE hknpCollisionQueryDispatcher::hknpCollisionQueryDispatcher(ShapeMask shapeMask)
	:	hknpCollisionQueryDispatcherBase()
{
	_initializeGetClosestPoints( shapeMask, m_closestPointsDispatchTable );
	_initializeShapeCast( shapeMask, m_shapeCastDispatchTable );
}

HK_FORCE_INLINE void hknpCollisionQueryDispatcherBase::setBaseShapeType(
	const hknpShapeType::Enum shapeType, const hknpCollisionQueryDispatcherBase::BaseType dispatchType )
{
	m_baseTypeMap[shapeType] = dispatchType;
}

HK_FORCE_INLINE hknpCollisionQueryDispatcherBase::BaseType hknpCollisionQueryDispatcherBase::getBaseShapeType(
	const hknpShapeType::Enum shapeType ) const
{
	return m_baseTypeMap[shapeType];
}

template<typename FUNCTION_TYPE>
HK_FORCE_INLINE void hknpCollisionQueryDispatcherBase::DispatchTable<FUNCTION_TYPE>::setFunction(
	const BaseType queryType, const BaseType targetType, FUNCTION_TYPE func, ShapeMask shapeMask)
{
	HK_ASSERT( 0x3d6d8ad0, queryType < NUM_BASE_TYPES );
	HK_ASSERT( 0x733f2859, targetType < NUM_BASE_TYPES );
	HK_ASSERT2( 0x4f05952d, func, "Cannot use a null query function" );

	ShapeMask mask = (1<<queryType) | (1<<targetType);

	if((shapeMask & mask) == mask)
	{
		m_dispatchTable[queryType][targetType] = func;
	}
}

template<typename FUNCTION_TYPE>
HK_FORCE_INLINE FUNCTION_TYPE hknpCollisionQueryDispatcherBase::DispatchTable<FUNCTION_TYPE>::getFunction(
	const BaseType queryType, const BaseType targetType ) const
{
	HK_ASSERT( 0x3d6d8ad1, queryType < NUM_BASE_TYPES );
	HK_ASSERT( 0x733f2858, targetType < NUM_BASE_TYPES );
	return m_dispatchTable[queryType][targetType];
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
