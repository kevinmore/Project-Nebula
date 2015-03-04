/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Collide/Shape/Convex/Scaled/hknpScaledConvexShape.h>

#include <Common/Base/Types/Geometry/hkGeometry.h>
#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>

#include <Geometry/Internal/Algorithms/RayCast/hkcdRayCastConvex.h>

#include <Physics/Physics/Collide/Query/hknpQueryAabbNmpUtil.h>
#include <Physics/Physics/Collide/Shape/hknpShapeCollector.h>
#include <Physics/Physics/Collide/Shape/hknpShapeUtil.h>
#include <Physics/Physics/Collide/Shape/Convex/hknpConvexShapeUtil.h>
#include <Physics/Physics/Dynamics/Motion/hknpMotionUtil.h>


namespace
{
	
	class hknpScaledConvexShapeKeyIterator : public hknpShapeKeyIterator
	{
		public:

			HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
			virtual void next() HK_OVERRIDE {}

			HK_FORCE_INLINE hknpScaledConvexShapeKeyIterator( const hknpShape& shape, const hknpShapeKeyMask* mask )
				: hknpShapeKeyIterator( shape, mask ) {}
	};

	// This collector handles the scaling and transforming of the query result before forwarding the hit to
	// the original collector.
	class hknpScaledConvexShapeScaleAndTransformCollector : public hknpCollisionQueryCollector
	{
		public:

			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS, hknpScaledConvexShapeScaleAndTransformCollector );

			HK_FORCE_INLINE hknpScaledConvexShapeScaleAndTransformCollector(
				const hkTransform* transform, hkVector4Parameter translation,
				hkVector4Parameter scale, hkVector4Parameter invScale,
				hknpCollisionQueryCollector* childCollector )
			{
				m_transform				= transform;
				m_translation			= translation;
				m_scale					= scale;
				m_invScale				= invScale;
				m_childCollector		= childCollector;
				m_earlyOutHitFraction	= m_childCollector->getEarlyOutHitFraction();
			}

			//
			// hknpCollisionQueryCollector implementation
			//

			HK_FORCE_INLINE virtual void reset()
			{
				return m_childCollector->reset();
			}

			HK_FORCE_INLINE virtual bool hasHit() const
			{
				return m_childCollector->hasHit();
			}

			HK_FORCE_INLINE virtual int getNumHits() const
			{
				return m_childCollector->getNumHits();
			}

			HK_FORCE_INLINE virtual const hknpCollisionResult* getHits() const
			{
				return m_childCollector->getHits();
			}

			virtual void addHit( const hknpCollisionResult& flippedHit ) HK_OVERRIDE
			{
				hkSimdReal fraction = hkSimdReal::fromFloat( flippedHit.m_fraction );

				hknpCollisionResult hit = flippedHit;

				hit.m_position.mul( m_scale );
				hit.m_position.add( m_translation );
				hit.m_position._setTransformedPos( *m_transform, hit.m_position );

				hit.m_normal.mul( m_invScale );
				hit.m_normal._setRotatedDir( m_transform->getRotation(), hit.m_normal);
				hit.m_normal.normalize<3,HK_ACC_23_BIT,HK_SQRT_IGNORE>();

				m_childCollector->addHit( hit );

				// Update our local EOHF every time it might potentially have changed in the child collector.
				m_earlyOutHitFraction = m_childCollector->getEarlyOutHitFraction();
			}

		protected:

			hkVector4						m_translation;
			hkVector4						m_scale;
			hkVector4						m_invScale;
			const hkTransform*				m_transform;
			hknpCollisionQueryCollector*	m_childCollector;
	};

}	// anonymous namespace


#if !defined(HK_PLATFORM_SPU)

hknpScaledConvexShapeBase::hknpScaledConvexShapeBase(
	const hknpConvexShape* childShape, hkVector4Parameter scale, hknpShape::ScaleMode mode )
	: hknpShape( hknpCollisionDispatchType::CONVEX )
{
	init( childShape, scale, mode );
}

hknpScaledConvexShapeBase::hknpScaledConvexShapeBase( class hkFinishLoadedObjectFlag flag )
:	hknpShape(flag)
{
}

#endif

int hknpScaledConvexShapeBase::calcSize() const
{
	return sizeof(hknpScaledConvexShapeBase);
}

void hknpScaledConvexShapeBase::calcAabb( const hkTransform& transform, hkAabb& aabbOut ) const
{
	// Compute the scaled child shape AABB
	hkVector4 scaledAabbCenter;
	hkVector4 scaledAabbHalfExtents;
	{
		hkAabb childAabb;
		getChildShape()->calcAabbNoRadius( hkTransform::getIdentity(), childAabb );

		// Transform AABB out of child space
		childAabb.getCenter( scaledAabbCenter );
		scaledAabbCenter.setMul( scaledAabbCenter, m_scale );
		scaledAabbCenter.add( m_translation );

		childAabb.getHalfExtents( scaledAabbHalfExtents );
		scaledAabbHalfExtents.setMul( scaledAabbHalfExtents, m_scale );
	}

	// Calculate the AABB
	hkAabbUtil::calcAabb( transform, scaledAabbHalfExtents, scaledAabbCenter, aabbOut );

	// Expand by radius
	hkSimdReal radius; radius.load<1>(&m_convexRadius);
	aabbOut.expandBy( radius );
}

#if !defined(HK_PLATFORM_SPU)

hkRefNew<hknpShapeKeyIterator> hknpScaledConvexShapeBase::createShapeKeyIterator( const hknpShapeKeyMask* mask ) const
{
	return hkRefNew<hknpShapeKeyIterator>( new hknpScaledConvexShapeKeyIterator( *this, mask ) );
}

#endif

void hknpScaledConvexShapeBase::getLeafShape( hknpShapeKey key, hknpShapeCollector* collector ) const
{
	HK_ASSERT2( 0x1D3469AB, key == HKNP_INVALID_SHAPE_KEY, "Shape key unsupported" );
	collector->checkForReset();
	collector->m_shapeOut = this;
}

HK_DISABLE_OPTIMIZATION_VS2008_X64
void hknpScaledConvexShapeBase::getSupportingVertex(
	hkVector4Parameter direction, hkcdVertex* const HK_RESTRICT svOut ) const
{
	// Apply scaling to support direction (invariant to translation, so no translation here)
	hkVector4 localDir = direction; localDir.mul(m_scale);
	getChildShape()->hknpConvexShape::getSupportingVertex(localDir,svOut);

	// Transform the returned vertex (must be translated)
	hkVector4 localPoint;
	localPoint.setAddMul( m_translation, m_scale,*svOut);
	svOut->setXYZ(localPoint);
}
HK_RESTORE_OPTIMIZATION_VS2008_X64

int hknpScaledConvexShapeBase::getFaceVertices(
	int faceIndex, hkVector4& planeOut, hkcdVertex* HK_RESTRICT vertexBuffer ) const
{
	const int ret = getChildShape()->getFaceVertices(faceIndex, planeOut, vertexBuffer);

	// Scale vertices, preserve id in w component
	for( int i=0; i<hknpConvexPolytopeShape::MAX_NUM_VERTICES_PER_FACE; ++i )
	{
		hkVector4 temp;
		temp.setAddMul( m_translation, m_scale, vertexBuffer[i] );
		vertexBuffer[i].setXYZ( temp );
	}

	// Transform face plane and return
	hkVector4Util::scalePlane( m_scale, planeOut );
	hkVector4Util::translatePlane( m_translation, planeOut );
	return ret;
}

void hknpScaledConvexShapeBase::getFaceInfo( const int index, hkVector4& planeOut, int& minAngleOut ) const
{
	getChildShape()->getFaceInfo(index, planeOut, minAngleOut);
	hkVector4Util::scalePlane(m_scale, planeOut);
	hkVector4Util::translatePlane(m_translation, planeOut);

	// Convert min angle to real, bound, and convert back again
	
	const hkReal minAngleReal = ((hkReal)minAngleOut - 0.5f)*(HK_REAL_PI/510.0f);
	minAngleOut = hknpMotionUtil::convertAngleToAngularTIM( boundMinAngle(minAngleReal));
}

hkReal hknpScaledConvexShapeBase::calcMinAngleBetweenFaces() const
{
	return boundMinAngle( getChildShape()->calcMinAngleBetweenFaces());
}

int hknpScaledConvexShapeBase::getNumberOfSupportVertices() const
{
	return getChildShape()->getNumberOfVertices();
}

const hkcdVertex* hknpScaledConvexShapeBase::getSupportVertices(
	hkcdVertex* const HK_RESTRICT vertexBuffer, int bufferSize ) const
{
	const hkcdVertex* verts = getChildShape()->getVertices();

	// Transform vertices, preserve id in w component
	const int N = getChildShape()->hknpConvexShape::getNumberOfVertices();
	HK_ASSERT2( 0xaf14e120, bufferSize >= N, "vertexBuffer is too small." );
	for (int i=0; i<N; ++i )
	{
		hkVector4 h = verts[i];
		hkVector4 temp;
		temp.setAddMul(m_translation, m_scale, h);
		vertexBuffer[i].setXYZ_W(temp,h);
	}
	return vertexBuffer;
}

void hknpScaledConvexShapeBase::convertVertexIdsToVertices(
	const hkUint8* ids, int numVerts, hkcdVertex* verticesOut ) const
{
	getChildShape()->hknpConvexShape::convertVertexIdsToVertices(ids,numVerts,verticesOut);

	// Transform them, preserving the W component (contains the vertex ids)
	
	for (int i = 0; i < numVerts; ++i)
	{
		hkVector4 transformedVertex;
		transformedVertex.setAddMul(m_translation, m_scale, verticesOut[i]);
		verticesOut[i].setXYZ(transformedVertex);
	}
}

void hknpScaledConvexShapeBase::queryAabbImpl(
	hknpCollisionQueryContext* queryContext,
	const hknpAabbQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
	const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
	hkArray<hknpShapeKey>* hits, hknpQueryAabbNmp* nmpInOut) const
{
	hkAabb targetAabb;
	calcAabb(hkTransform::getIdentity(), targetAabb);

	// Query AABB is assumed to be in target space.
	if (!targetAabb.overlaps(query.m_aabb))
	{
		if (nmpInOut)
		{
			hknpQueryAabbNmpUtil::checkOverlapWithNmp(query.m_aabb, targetAabb, nmpInOut);
		}
		return;
	}

	if (query.m_filter)
	{
		hknpCollisionFilter::FilterInput shapeFilterInputA;
		shapeFilterInputA.m_filterData		= query.m_filterData;
		shapeFilterInputA.m_body			= queryShapeInfo.m_body;
		shapeFilterInputA.m_rootShape		= queryShapeInfo.m_rootShape;
		shapeFilterInputA.m_parentShape		= queryShapeInfo.m_parentShape;
		shapeFilterInputA.m_shapeKey		= queryShapeInfo.m_shapeKeyPath.getKey();
		shapeFilterInputA.m_shape			= HK_NULL;

		hknpCollisionFilter::FilterInput shapeFilterInputB;
		shapeFilterInputB.m_filterData		= targetShapeFilterData;
		shapeFilterInputB.m_body			= targetShapeInfo.m_body;
		shapeFilterInputB.m_rootShape		= targetShapeInfo.m_rootShape;
		shapeFilterInputB.m_parentShape		= targetShapeInfo.m_parentShape;
		shapeFilterInputB.m_shapeKey		= targetShapeInfo.m_shapeKeyPath.getKey();
		shapeFilterInputB.m_shape			= this;

		if (!query.m_filter->isCollisionEnabled(hknpCollisionQueryType::QUERY_AABB, true, shapeFilterInputA, shapeFilterInputB))
		{
			return;
		}
	}

	hknpAabbQueryUtil::addHit(
		queryShapeInfo.m_body, query.m_filterData,
		targetShapeInfo.m_body, targetShapeInfo.m_shapeKeyPath.getKey(), targetShapeFilterData,
		hits );
}

void hknpScaledConvexShapeBase::queryAabbImpl(
	hknpCollisionQueryContext* queryContext,
	const hknpAabbQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
	const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
	hknpCollisionQueryCollector* collector, hknpQueryAabbNmp* nmpInOut) const
{
	hkAabb targetAabb;
	calcAabb(hkTransform::getIdentity(), targetAabb);

	// Query AABB is assumed to be in target space.
	if (!targetAabb.overlaps(query.m_aabb))
	{
		if (nmpInOut)
		{
			hknpQueryAabbNmpUtil::checkOverlapWithNmp(query.m_aabb, targetAabb, nmpInOut);
		}
		return;
	}

	if (query.m_filter)
	{
		hknpCollisionFilter::FilterInput shapeFilterInputA;
		shapeFilterInputA.m_filterData		= query.m_filterData;
		shapeFilterInputA.m_body			= queryShapeInfo.m_body;
		shapeFilterInputA.m_rootShape		= queryShapeInfo.m_rootShape;
		shapeFilterInputA.m_parentShape		= queryShapeInfo.m_parentShape;
		shapeFilterInputA.m_shapeKey		= queryShapeInfo.m_shapeKeyPath.getKey();
		shapeFilterInputA.m_shape			= HK_NULL;

		hknpCollisionFilter::FilterInput shapeFilterInputB;
		shapeFilterInputB.m_filterData		= targetShapeFilterData;
		shapeFilterInputB.m_body			= targetShapeInfo.m_body;
		shapeFilterInputB.m_rootShape		= targetShapeInfo.m_rootShape;
		shapeFilterInputB.m_parentShape		= targetShapeInfo.m_parentShape;
		shapeFilterInputB.m_shapeKey		= targetShapeInfo.m_shapeKeyPath.getKey();
		shapeFilterInputB.m_shape			= this;

		if (!query.m_filter->isCollisionEnabled(hknpCollisionQueryType::QUERY_AABB, true, shapeFilterInputA, shapeFilterInputB))
		{
			return;
		}
	}

	hknpAabbQueryUtil::addHit(
		queryShapeInfo.m_body, query.m_filterData,
		targetShapeInfo.m_body, targetShapeInfo.m_shapeKeyPath.getKey(), targetShapeFilterData,
		collector );
}

void hknpScaledConvexShapeBase::castRayImpl(
	hknpCollisionQueryContext* queryContext,
	const hknpRayCastQuery& query,
	const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
	hknpCollisionQueryCollector* collector ) const
{
	const hknpConvexPolytopeShape* childShapeAsPolytope = getChildShape()->asConvexPolytopeShape();
	HK_ASSERT2(0xe1e949f0, childShapeAsPolytope, "Child shape must be a hknpConvexPolytopeShape.");

	if ( HK_VERY_LIKELY( m_convexRadius > 0.0f ) )
	{
		hkSimdReal	fraction = collector->getEarlyOutHitFraction();
		hkVector4	normal; normal.setZero();
		const hkVector4* const planes = childShapeAsPolytope->getPlanes();
		const int numPlanes	= childShapeAsPolytope->getNumFaces();

		// Scale and translate planes, apply convex radius.
		hkInplaceArray<hkVector4,64> shiftPlanes;
		shiftPlanes.setSize( numPlanes );
		{
			hkVector4 offset; offset.setZero(); offset(3) = m_convexRadius;
			for(int i=0; i<numPlanes; ++i)
			{
				shiftPlanes[i] = planes[i];
				hkVector4Util::scalePlane( m_scale, shiftPlanes[i]);
				hkVector4Util::translatePlane( m_translation, shiftPlanes[i]);
				shiftPlanes[i].setSub(shiftPlanes[i], offset);
			}
		}

		// Cast ray against transformed planes
		if ( hkcdRayCastConvex( query, shiftPlanes.begin(), numPlanes, &fraction, &normal, query.m_flags ) )
		{
			hknpCollisionResult result;
			{
				result.m_queryType								= hknpCollisionQueryType::RAY_CAST;
				result.m_normal._setRotatedDir( targetShapeInfo.m_shapeToWorld->getRotation(), normal );
				hkVector4 impact; impact.setAddMul( query.m_origin, query.getDirection(), fraction );
				result.m_position._setTransformedPos( *targetShapeInfo.m_shapeToWorld, impact );
				fraction.store<1>(&result.m_fraction.ref());
				result.m_queryBodyInfo.m_shapeMaterialId			= query.m_filterData.m_materialId;
				result.m_queryBodyInfo.m_shapeCollisionFilterInfo	= query.m_filterData.m_collisionFilterInfo;
				result.m_queryBodyInfo.m_shapeUserData				= query.m_filterData.m_userData;
				result.m_hitBodyInfo.m_shapeKey					= targetShapeInfo.m_shapeKeyPath.getKey();
				result.m_hitBodyInfo.m_shapeMaterialId			= targetShapeFilterData.m_materialId;
				result.m_hitBodyInfo.m_shapeCollisionFilterInfo	= targetShapeFilterData.m_collisionFilterInfo;
				result.m_hitBodyInfo.m_shapeUserData			= targetShapeFilterData.m_userData;
				result.m_hitBodyInfo.m_bodyId					= ( targetShapeInfo.m_body ? targetShapeInfo.m_body->m_id : hknpBodyId::INVALID );
			}

			collector->addHit( result );
		}
	}
	else
	{
		hkVector4 invScale; invScale.setReciprocal<HK_ACC_23_BIT,HK_DIV_IGNORE>(m_scale);

		// Transform input ray into child space (incl. target shape scaling.)
		hknpRayCastQuery childRayQuery = query;
		{
			hkVector4 origin = query.m_origin;
			hkVector4 direction = query.m_direction;
			hkSimdReal fraction = collector->getEarlyOutHitFraction();

			// Get ray to inverse scaled space (assumes that ray is already in shape space)
			origin.sub(m_translation);
			origin.mul(invScale);
			direction.sub(m_translation);
			direction.mul(invScale);

			// As we scale the direction without normalizing it again, we *don't* need to scale the fraction.
			childRayQuery.setOriginDirection(origin,direction,fraction);
		}

		hknpShapeQueryInfo targetChildShapeInfo(&targetShapeInfo);
		{
			// We will take care of the world transform, so strip it when passing to child shape
			targetChildShapeInfo.m_shapeToWorld = &hkTransform::getIdentity();
		}

		hknpScaledConvexShapeScaleAndTransformCollector scaleAndTransformCollector(
			targetShapeInfo.m_shapeToWorld, m_translation,
			m_scale, invScale,
			collector );

		childShapeAsPolytope->castRayImpl(
			queryContext, childRayQuery, targetShapeFilterData, targetChildShapeInfo, &scaleAndTransformCollector );
	}
}

int hknpScaledConvexShapeBase::getSupportingFace(
	hkVector4Parameter surfacePoint, const hkcdGsk::Cache* gskCache, bool useB,
	hkVector4& planeOut, int& minAngleOut, hkUint32 &prevFaceId ) const
{
	// Transform the inputs into child shape space
	hkVector4 surfacePointLocal;
	hkVector4 planeOutLocal;
	{
		hkVector4 invScale;
		invScale.setReciprocal<HK_ACC_23_BIT,HK_DIV_IGNORE>( m_scale );

		surfacePointLocal.setSub( surfacePoint, m_translation );
		surfacePointLocal.mul( invScale );

		planeOutLocal = planeOut;
		hkVector4Util::scalePlane( invScale, planeOutLocal );
	}

	// Query child shape
	const int faceIndex = getChildShape()->getSupportingFace(
		surfacePointLocal, gskCache, useB, planeOutLocal, minAngleOut, prevFaceId );

	// Transform the output plane
	hkVector4Util::scalePlane( m_scale, planeOutLocal );
	hkVector4Util::translatePlane( m_translation, planeOutLocal );
	planeOut = planeOutLocal;

	minAngleOut = 0;	
	return faceIndex;
}

int hknpScaledConvexShapeBase::getNumberOfFaces() const
{
	return getChildShape()->getNumberOfFaces();
}

#if !defined(HK_PLATFORM_SPU)

void hknpScaledConvexShapeBase::buildMassProperties(
	const MassConfig& massConfig, hkDiagonalizedMassProperties& massPropertiesOut ) const
{
	
	hkLocalBuffer<hkcdVertex> verticesBuffer( getNumberOfSupportVertices() );
	const hkcdVertex* vertices = getSupportVertices( verticesBuffer.begin(), getNumberOfSupportVertices() );

	const int numUniqueVertices = hknpConvexShapeUtil::getNumberOfUniqueVertices( getChildShape() );
	HK_ASSERT( 0xb75590a1, numUniqueVertices <= getNumberOfSupportVertices() );

	if( massConfig.m_quality == hknpShape::MassConfig::QUALITY_HIGH )
	{
		// Calculate accurate mass properties
		hknpConvexShapeUtil::buildHullMassProperties(
			massConfig, vertices, numUniqueVertices, m_convexRadius, massPropertiesOut );
	}
	else
	{
		// Approximate by AABB mass properties
		hkAabb aabb;
		aabb.setEmpty();
		for( int i=0; i<numUniqueVertices; ++i )
		{
			aabb.includePoint( verticesBuffer[i] );
		}
		aabb.expandBy( hkSimdReal::fromFloat(m_convexRadius) );
		hknpShapeUtil::buildAabbMassProperties( massConfig, aabb, massPropertiesOut );
	}
}

hkResult hknpScaledConvexShapeBase::buildSurfaceGeometry(
	const BuildSurfaceGeometryConfig& config, hkGeometry* geometryOut ) const
{
	return hknpShapeUtil::createConvexHullGeometry(*this, config.m_radiusMode, geometryOut);
}

hknpScaledConvexShape::hknpScaledConvexShape(
	const hknpConvexShape* childShape, hkVector4Parameter scale, hknpShape::ScaleMode mode )
:	hknpScaledConvexShapeBase( childShape, scale, mode )
{
	getChildShape()->addReference();
}

hknpScaledConvexShape::hknpScaledConvexShape( class hkFinishLoadedObjectFlag flag )
:	hknpScaledConvexShapeBase(flag)
{
}

#endif

hknpScaledConvexShape::~hknpScaledConvexShape()
{
	getChildShape()->removeReference();
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
