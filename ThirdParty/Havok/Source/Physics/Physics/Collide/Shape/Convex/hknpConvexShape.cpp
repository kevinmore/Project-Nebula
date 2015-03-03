/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Collide/Shape/Convex/hknpConvexShape.h>

#include <Common/Base/Types/Geometry/Aabb/hkAabbUtil.h>
#include <Common/Internal/GeometryProcessing/CollisionGeometryOptimizer/hkgpCgo.h>

#if !defined(HK_PLATFORM_SPU)
#	include <Common/Internal/GeometryProcessing/hkGeometryProcessing.h>
#	include <Common/Internal/GeometryProcessing/ConvexHull/hkgpConvexHull.h>
#	include <Common/GeometryUtilities/Inertia/hkInertiaTensorComputer.h>
#endif

#include <Geometry/Internal/Algorithms/Gsk/hkcdGsk.h>
#include <Geometry/Internal/Algorithms/RayCast/hkcdRayCastTriangle.h>
#include <Geometry/Internal/Algorithms/RayCast/hkcdRayCastConvex.h>
#include <Geometry/Internal/Algorithms/RayCast/hkcdRayCastSphere.h>
#include <Geometry/Internal/Algorithms/RayCast/hkcdRayCastCapsule.h>
#include <Geometry/Internal/Algorithms/Distance/hkcdDistancePointConvex.h>

#include <Physics/Physics/Collide/Query/hknpCollisionQuery.h>
#include <Physics/Physics/Collide/Query/hknpQueryAabbNmpUtil.h>
#include <Physics/Physics/Collide/Query/Collector/hknpCollisionQueryCollector.h>

#include <Physics/Physics/Collide/Shape/hknpShapeCollector.h>
#include <Physics/Physics/Collide/Shape/Convex/Sphere/hknpSphereShape.h>
#include <Physics/Physics/Collide/Shape/Convex/Capsule/hknpCapsuleShape.h>
#include <Physics/Physics/Collide/Shape/hknpShapeUtil.h>
#include <Physics/Physics/Collide/Shape/Convex/hknpConvexShapeUtil.h>

HK_ALIGN16(const hkUint32 hknpConvexShape::s_curIndices [4]) = {0,1,2,3};


namespace
{
	
	class hknpConvexShapeKeyIterator : public hknpShapeKeyIterator
	{
	public:

		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );
		virtual void next() HK_OVERRIDE {}

		HK_FORCE_INLINE hknpConvexShapeKeyIterator( const hknpShape& shape, const hknpShapeKeyMask* mask )
			: hknpShapeKeyIterator( shape, mask ) {}
	};
}	// anonymous namespace


#ifndef HK_PLATFORM_SPU

hknpConvexShape::hknpConvexShape( hkFinishLoadedObjectFlag flag )
	:	hknpShape(flag)
{
	if( flag.m_finishing )
	{
		m_flags.orWith( IS_CONVEX_SHAPE );
	}
}

#endif

hknpConvexShape::BuildConfig::BuildConfig()
{
	m_buildFaces				=	true;
	m_buildMassProperties		=	true;
	m_massConfig				=	hknpShape::MassConfig::fromDensity( 1.0f );
	m_shrinkByRadius			=	true;
	m_featurePreservationFactor	=	0.0f;
	m_maxNumVertices			=	MAX_NUM_VERTICES;
	m_extraTransform			=	HK_NULL;
	m_simplifyExpansionDistance	=	0.0f;
	//m_sizeOfBaseClass			=	HKNP_CONVEX_SHAPE_BASE_SIZE;
	m_sizeOfBaseClass			=	HKNP_CONVEX_POLYTOPE_BASE_SIZE;
}

#if !defined(HK_PLATFORM_SPU)

hkAabb hknpConvexShape::calcAabbNoRadius()
{
	const int						MXSIZE = 4;
	hkAabb							aabb; aabb.setEmpty();
	const hkVector4* HK_RESTRICT	verts = getVertices();
	for (int i =0; i < getNumberOfVertices(); verts += MXSIZE, i+= MXSIZE)
	{
		hkMxVector<MXSIZE> vertsMx; vertsMx.moveLoad(verts);
		hkMxUNROLL_4(aabb.includePoint(vertsMx.getVector<hkMxI>()));
	}
	return aabb;
}

#endif

void hknpConvexShape::calcAabb( const hkTransform& transform, hkAabb& aabbOut ) const
{
	calcAabbNoRadius( transform, aabbOut);

	// Expand AABB by expansion radius
	hkSimdReal radius; radius.load<1>( &m_convexRadius );
	aabbOut.expandBy( radius );
}

int hknpConvexShape::calcSize() const
{
	return calcConvexShapeSize( getNumberOfVertices() );
}

#if !defined(HK_PLATFORM_SPU)

hkResult hknpConvexShape::buildSurfaceGeometry( const BuildSurfaceGeometryConfig& config, hkGeometry* geometryOut) const
{
	return hknpShapeUtil::createConvexHullGeometry( *this, config.m_radiusMode, geometryOut );
}

#endif


void hknpConvexShape::queryAabbImpl(
	hknpCollisionQueryContext* queryContext,
	const hknpAabbQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
	const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
	hkArray<hknpShapeKey>* hits, hknpQueryAabbNmp* nmpInOut ) const
{
	hkAabb targetAabb;
	calcAabb(hkTransform::getIdentity(), targetAabb);

	// Query AABB is assumed to be in target space.
	if( !targetAabb.overlaps(query.m_aabb) )
	{
		if( nmpInOut )
		{
			hknpQueryAabbNmpUtil::checkOverlapWithNmp( query.m_aabb, targetAabb, nmpInOut );
		}
		return;
	}

	// Perform filtering
	if (query.m_filter)
	{
		hknpCollisionFilter::FilterInput shapeFilterInputA;
		{
			shapeFilterInputA.m_filterData		= query.m_filterData;
			shapeFilterInputA.m_body			= queryShapeInfo.m_body;
			shapeFilterInputA.m_rootShape		= queryShapeInfo.m_rootShape;
			shapeFilterInputA.m_parentShape		= queryShapeInfo.m_parentShape;
			shapeFilterInputA.m_shapeKey		= queryShapeInfo.m_shapeKeyPath.getKey();
			shapeFilterInputA.m_shape			= HK_NULL;
		}

		hknpCollisionFilter::FilterInput shapeFilterInputB;
		{
			shapeFilterInputB.m_filterData		= targetShapeFilterData;
			shapeFilterInputB.m_body			= targetShapeInfo.m_body;
			shapeFilterInputB.m_rootShape		= targetShapeInfo.m_rootShape;
			shapeFilterInputB.m_parentShape		= targetShapeInfo.m_parentShape;
			shapeFilterInputB.m_shapeKey		= targetShapeInfo.m_shapeKeyPath.getKey();
			shapeFilterInputB.m_shape			= this;
		}

		if( !query.m_filter->isCollisionEnabled(
			hknpCollisionQueryType::QUERY_AABB, true, shapeFilterInputA, shapeFilterInputB ) )
		{
			return;
		}
	}

	hknpAabbQueryUtil::addHit(
		queryShapeInfo.m_body, query.m_filterData,
		targetShapeInfo.m_body, targetShapeInfo.m_shapeKeyPath.getKey(), targetShapeFilterData,
		hits );
}


void hknpConvexShape::queryAabbImpl(
	hknpCollisionQueryContext* queryContext,
	const hknpAabbQuery& query, const hknpShapeQueryInfo& queryShapeInfo,
	const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
	hknpCollisionQueryCollector* collector, hknpQueryAabbNmp* nmpInOut ) const
{
	hkAabb targetAabb;
	calcAabb(hkTransform::getIdentity(), targetAabb);

	// Query AABB is assumed to be in target space.
	if( !targetAabb.overlaps(query.m_aabb) )
	{
		if( nmpInOut )
		{
			hknpQueryAabbNmpUtil::checkOverlapWithNmp( query.m_aabb, targetAabb, nmpInOut );
		}
		return;
	}

	// Perform filtering
	if (query.m_filter)
	{
		hknpCollisionFilter::FilterInput shapeFilterInputA;
		{
			shapeFilterInputA.m_filterData		= query.m_filterData;
			shapeFilterInputA.m_body			= queryShapeInfo.m_body;
			shapeFilterInputA.m_rootShape		= queryShapeInfo.m_rootShape;
			shapeFilterInputA.m_parentShape		= queryShapeInfo.m_parentShape;
			shapeFilterInputA.m_shapeKey		= queryShapeInfo.m_shapeKeyPath.getKey();
			shapeFilterInputA.m_shape			= HK_NULL;
		}

		hknpCollisionFilter::FilterInput shapeFilterInputB;
		{
			shapeFilterInputB.m_filterData		= targetShapeFilterData;
			shapeFilterInputB.m_body			= targetShapeInfo.m_body;
			shapeFilterInputB.m_rootShape		= targetShapeInfo.m_rootShape;
			shapeFilterInputB.m_parentShape		= targetShapeInfo.m_parentShape;
			shapeFilterInputB.m_shapeKey		= targetShapeInfo.m_shapeKeyPath.getKey();
			shapeFilterInputB.m_shape			= this;
		}

		if( !query.m_filter->isCollisionEnabled(
			hknpCollisionQueryType::QUERY_AABB, true, shapeFilterInputA, shapeFilterInputB ) )
		{
			return;
		}
	}

	hknpAabbQueryUtil::addHit(
		queryShapeInfo.m_body, query.m_filterData,
		targetShapeInfo.m_body, targetShapeInfo.m_shapeKeyPath.getKey(), targetShapeFilterData,
		collector );
}


void hknpConvexShape::castRayImpl(
	hknpCollisionQueryContext* queryContext,
	const hknpRayCastQuery& query,
	const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
	hknpCollisionQueryCollector* collector ) const
{
	hkSimdReal fraction = collector->getEarlyOutHitFraction();
	hkVector4 normal; normal.setZero();
	hkInt32 hitResult = 0;

	
	
	switch( getType() )
	{
		// Sphere.
	case hknpShapeType::SPHERE:
		{
			hkSimdReal radius; radius.setFromFloat( m_convexRadius );
			hkVector4 posAndRadius; posAndRadius.setXYZ_W( getVertices()[0], radius );
			hitResult = hkcdRayCastSphere( query, posAndRadius, &fraction, &normal, query.m_flags );
		}
		break;

		// Capsule.
	case hknpShapeType::CAPSULE:
		{
			const hknpCapsuleShape*	capsule = static_cast<const hknpCapsuleShape*>(this);
			hkSimdReal radius; radius.setFromFloat( m_convexRadius );
			hkcdRayCastCapsuleHitType unused;
			hitResult = hkcdRayCastCapsule( query, capsule->m_a, capsule->m_b, radius, &fraction, &normal, &unused, query.m_flags );
		}
		break;

		// Triangle.
	case hknpShapeType::TRIANGLE:
		{
			
			const hknpTriangleShape* triangle = static_cast<const hknpTriangleShape*>(this);
			hitResult = hkcdRayTriangleIntersect( query, getVertex(0), getVertex(1), getVertex(2), &fraction, &normal );
			if ( !hitResult && triangle->isQuad() )
			{
				hitResult = hkcdRayTriangleIntersect( query, getVertex(0), getVertex(2), getVertex(3), &fraction, &normal );
			}
		}
		break;

		// Convex with face information
	case hknpShapeType::CONVEX_POLYTOPE:
		{
			const hknpConvexPolytopeShape* cs = static_cast<const hknpConvexPolytopeShape*>(this);
			if( cs->getNumFaces() )
			{
				const hkVector4*				planes		=	cs->getPlanes();
				const int						numPlanes	=	cs->getNumFaces();
				hkInplaceArray<hkVector4,64>	shiftPlanes;
				if( m_convexRadius >= 0.0f )
				{
					hkVector4 offset; offset.setZero(); offset(3) = m_convexRadius;
					shiftPlanes.setSize(numPlanes);
					for(int i=0; i<numPlanes; ++i)
					{
						shiftPlanes[i].setSub( planes[i], offset );
					}
					planes = shiftPlanes.begin();
				}

				hitResult = hkcdRayCastConvex( query, planes, numPlanes, &fraction, &normal, query.m_flags );
			}
		}
		break;

		// General convex
	default:
		{
			HK_TIMER_BEGIN( "ConvexShapeCastRay(GSK)", HK_NULL );

			// Initialize ray-cast input
			hkcdGsk::RayCastInput gskInput( m_convexRadius );
			gskInput.m_from = query.m_origin;
			gskInput.m_direction = query.getDirection();
			gskInput.m_direction.mul( query.getFraction() );

			hkcdGsk::RayCastOutput gskOutput;
			gskOutput.m_fractionInOut = hkSimdReal_1; 

			// Run the ray-cast
			hitResult = hkcdGsk::rayCast( getVertices(), getNumberOfVertices(), gskInput, gskOutput );
			if( hitResult )
			{
				normal = gskOutput.m_normalOut;
				fraction = gskOutput.m_fractionInOut * query.getFraction();
			}

			HK_TIMER_END();
		}
		break;
	}

	if( hitResult )
	{
		hkVector4 impact;
		impact.setAddMul( query.m_origin, query.getDirection(), fraction );

		hknpCollisionResult result;
		result.m_queryType								= hknpCollisionQueryType::RAY_CAST;
		result.m_normal									. _setRotatedDir( targetShapeInfo.m_shapeToWorld->getRotation(), normal );
		result.m_position								. _setTransformedPos( *targetShapeInfo.m_shapeToWorld, impact );
		fraction.store<1>( &result.m_fraction.ref() );
		result.m_queryBodyInfo.m_shapeMaterialId			= query.m_filterData.m_materialId;
		result.m_queryBodyInfo.m_shapeCollisionFilterInfo	= query.m_filterData.m_collisionFilterInfo;
		result.m_queryBodyInfo.m_shapeUserData				= query.m_filterData.m_userData;
		result.m_hitBodyInfo.m_shapeKey					= targetShapeInfo.m_shapeKeyPath.getKey();
		result.m_hitBodyInfo.m_shapeMaterialId			= targetShapeFilterData.m_materialId;
		result.m_hitBodyInfo.m_shapeCollisionFilterInfo	= targetShapeFilterData.m_collisionFilterInfo;
		result.m_hitBodyInfo.m_shapeUserData			= targetShapeFilterData.m_userData;
		result.m_hitBodyInfo.m_bodyId					= ( targetShapeInfo.m_body ? targetShapeInfo.m_body->m_id : hknpBodyId::INVALID );
		result.m_hitResult								= hitResult;

		collector->addHit( result );
	}
}

#if !defined(HK_PLATFORM_SPU)

hknpConvexShape* hknpConvexShape::createFromVertices( const hkStridedVertices& verticesIn, hkReal radius, const BuildConfig& config0 )
{
	BuildConfig config = config0;

	// If there is an extra transform, bake it and recurse
	if( config.m_extraTransform != HK_NULL )
	{
		hkArray<hkVector4> transformed( verticesIn.getSize() );
		for( int i = 0; i < verticesIn.getSize(); ++i )
		{
			hkVector4& vertex = transformed[i];
			verticesIn.getVertex( i, vertex );
			vertex.setTransformedPos( *config.m_extraTransform, vertex );
		}

		config.m_extraTransform = HK_NULL;
		hkStridedVertices transformedVertices( transformed );
		return createFromVertices( transformedVertices, radius, config );
	}

	// If we want to store faces (which is the default BuildConfig setting), we create a hknpConvexPolytopeShape instead.
	if( config.m_buildFaces )
	{
		hknpConvexShape* shape = HK_NULL;
		switch( verticesIn.m_numVertices )
		{
		case 1:
			// A sphere
			{
				hkVector4 center; verticesIn.getVertex(0, center);
				shape = hknpSphereShape::createSphereShape( center, radius );
				if( shape && config.m_buildMassProperties )
				{
					shape->setMassProperties( config.m_massConfig );
				}
			}
			break;

		case 2:
			// A capsule
			{
				hkVector4 posA; verticesIn.getVertex(0, posA);
				hkVector4 posB; verticesIn.getVertex(1, posB);
				shape = hknpCapsuleShape::createCapsuleShape( posA, posB, radius );
				if( shape && config.m_buildMassProperties )
				{
					shape->setMassProperties( config.m_massConfig );
				}
			}
			break;

		case 3:
			// A triangle
			{
				hkVector4 a; verticesIn.getVertex(0, a);
				hkVector4 b; verticesIn.getVertex(1, b);
				hkVector4 c; verticesIn.getVertex(2, c);
				shape = hknpTriangleShape::createTriangleShape( a, b, c, radius );
				if( shape && config.m_buildMassProperties )
				{
					shape->setMassProperties( config.m_massConfig );
				}
			}
			break;

		default:
			// 4 is the minimum number of points required to span a 3D space
			{
				shape = hknpConvexPolytopeShape::createFromVerticesInternal( verticesIn, radius, config );
			}
			break;
		}

		if( shape )
		{
			return shape;
		}

		HK_WARN( 0x5B234D88, "Couldn't build convex polytope shape. Falling back to faceless convex shape." );
	}

	// Don't create or store face information
	config.m_sizeOfBaseClass = HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, sizeof(hknpConvexShape));

	const int numVertices = HK_NEXT_MULTIPLE_OF(4, verticesIn.m_numVertices);
	hknpConvexShape* shape;
	{
		int bufferSizeOut;
		void* buffer = allocateConvexShape(numVertices, config.m_sizeOfBaseClass, bufferSizeOut);
		shape = new (buffer) hknpConvexShape(numVertices, radius, config.m_sizeOfBaseClass);
		shape->m_memSizeAndFlags = (hkUint16)bufferSizeOut;

		if( HK_VERY_UNLIKELY( verticesIn.m_numVertices == 1 ) )	
		{
			shape->m_flags.orWith( USE_SINGLE_POINT_MANIFOLD );
		}
	}

	// Fill in vertices
	
	{
		hkVector4* vertices = shape->getVertices();
		const hkReal* vertexIn = verticesIn.m_vertices;
		int index = 0;
		for (; index < verticesIn.m_numVertices; ++index, vertexIn = hkAddByteOffsetConst(vertexIn, verticesIn.m_striding))
		{
			vertices[index].load<3, HK_IO_NATIVE_ALIGNED>(vertexIn);
			vertices[index].setInt24W(index);
		}

		// Repeat the last vertex to pad to a multiple of 4
		while (index & 3)
		{
			vertices[index++] = vertices[verticesIn.m_numVertices - 1];
		}
	}

	// Build and attach mass properties
	if( config.m_buildMassProperties )
	{
		shape->setMassProperties( config.m_massConfig );
	}

	return shape;
}

hknpConvexShape* HK_CALL hknpConvexShape::createFromIndexedVertices(
	const hkVector4* HK_RESTRICT vertexBuffer, const hkUint16* HK_RESTRICT indexBuffer, int numVertices, hkReal radius, const BuildConfig& config0 )
{
	// If we want to store faces (which is the default BuildConfig setting), we create a hknpConvexPolytopeShape instead.
	if( config0.m_buildFaces )
	{
		// Not supported!
		HK_ASSERT2(0x2b249f5d, false, "This feature is currently not supported!");
		return HK_NULL;
	}

	// Don't create or store face information
	BuildConfig config = config0;
	config.m_sizeOfBaseClass	= HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, sizeof(hknpConvexShape));
	const int numPaddedVertices	= HK_NEXT_MULTIPLE_OF(4, numVertices);

	// Allocate shape
	hknpConvexShape* shape;
	{
		int bufferSizeOut;
		void* buffer = allocateConvexShape(numPaddedVertices, config.m_sizeOfBaseClass, bufferSizeOut);
		shape = new (buffer) hknpConvexShape(numPaddedVertices, radius, config.m_sizeOfBaseClass);
		shape->m_memSizeAndFlags = (hkUint16)bufferSizeOut;
	}

	// Get output data
	hkVector4* vtxPtrOut = shape->getVertices();

	// Copy vertices
	for( int i = 0; i < numVertices; i++ )
	{
		hkVector4 v = vertexBuffer[indexBuffer[i]];
		v.setInt24W(i);
		vtxPtrOut[i] = v;
	}
	{
		hkVector4 v = vertexBuffer[indexBuffer[numVertices - 1]];
		v.setInt24W(numVertices - 1);

		for (int i = numVertices; i < numPaddedVertices; i++)
		{
			vtxPtrOut[i] = v;
		}
	}

	if( config.m_buildMassProperties )
	{
		shape->setMassProperties( config.m_massConfig );
	}

	return shape;
}

hknpConvexShape* HK_CALL hknpConvexShape::createFromHalfExtents(
	hkVector4Parameter halfExtent, hkReal radius, const BuildConfig& config )
{
	hkAabb aabb;
	aabb.m_min.setNeg<4>( halfExtent );
	aabb.m_max = halfExtent;
	return createFromAabb( aabb, radius, config );
}

hknpConvexShape* HK_CALL hknpConvexShape::createFromAabb( const hkAabb& aabbIn, hkReal radius, const BuildConfig& config )
{
	// Calculate a safely shrunk AABB
	hkAabb aabb;
	{
		hkVector4 extents;
		extents.setSub( aabbIn.m_max, aabbIn.m_min );

		hkVector4 shrinkVec;
		if( config.m_shrinkByRadius )
		{
			// work out the maximum shrinking for the input AABB (don't shrink more than 50% of the AABB)
			shrinkVec.setAll( radius );
			hkSimdReal maxShrink;
			{
				maxShrink = extents.horizontalMin<3>();
				maxShrink.setMax( hkSimdReal_0, maxShrink * hkSimdReal_Inv4 );	// quarter of diameter = half radius
			}
			hkVector4 maxShrinkV; maxShrinkV.setAll( maxShrink );
			shrinkVec.setMin( shrinkVec, maxShrinkV );
		}
		else
		{
			shrinkVec.setZero();
		}

		aabb.m_min.setAdd( aabbIn.m_min, shrinkVec );
		aabb.m_max.setSub( aabbIn.m_max, shrinkVec );
		aabb.m_max.setMax( aabb.m_min, aabb.m_max ); // to ensure that shrinking does not generate invalid AABBs

		// Ensure that none of the extents is zero
		{
			hkVector4Comparison extentIsZero = extents.equalZero();
			HK_WARN_ON_DEBUG_IF( extentIsZero.anyIsSet<hkVector4ComparisonMask::MASK_XYZ>(), 0xf034defd,
				"Creating a box having one extents smaller than the convex radius. This extents will be set to epsilon." );

			hkVector4 epsilon = hkVector4::getConstant(HK_QUADREAL_EPS);
			epsilon.zeroIfFalse( extentIsZero );

			aabb.m_max.add( epsilon );
			aabb.m_min.sub( epsilon );
		}
	}

	// Create the shape
	hknpConvexShape* shape;
	{
		hkVector4 verts[8];
		hkAabbUtil::get8Vertices( aabb, verts );

		BuildConfig conf = config;
		conf.m_shrinkByRadius = false;		// we pre-shrunk the AABB
		conf.m_buildMassProperties = false;	// we will build our own below

		shape = createFromVertices( hkStridedVertices(verts, 8), radius, conf );

		/// Boxes build with face info can safely set hknpShape::USE_NORMAL_TO_FIND_SUPPORT_PLANE
		if (conf.m_buildFaces)
		{
			shape->m_flags.orWith( hknpShape::USE_NORMAL_TO_FIND_SUPPORT_PLANE );
		}
	}

	// Build mass properties
	if( config.m_buildMassProperties )
	{
		hkDiagonalizedMassProperties massProperties;
		{
			hkVector4 halfExtents;
			aabb.getHalfExtents( halfExtents );
			hkVector4 radiusVec; radiusVec.setAll( radius );
			halfExtents.add( radiusVec );

			hkInertiaTensorComputer::computeBoxVolumeMassPropertiesDiagonalized(
				halfExtents, 1.0f, massProperties.m_inertiaTensor, massProperties.m_volume );
			massProperties.m_mass = config.m_massConfig.calcMassFromVolume( massProperties.m_volume );
			massProperties.m_inertiaTensor.mul( hkSimdReal::fromFloat( massProperties.m_mass * config.m_massConfig.m_inertiaFactor ) );
			massProperties.m_centerOfMass.setInterpolate( aabb.m_min, aabb.m_max, hkSimdReal_Inv2 );
			if( !config.m_extraTransform )
			{
				massProperties.m_majorAxisSpace.setIdentity();
			}
			else
			{
				massProperties.m_centerOfMass.setTransformedPos( *config.m_extraTransform, massProperties.m_centerOfMass );
				massProperties.m_majorAxisSpace.set( config.m_extraTransform->getRotation() );
			}
		}

		shape->setMassProperties( massProperties );
	}

	return shape;
}

hknpConvexShape* HK_CALL hknpConvexShape::createFromCylinder(
	hkVector4Parameter halfExtent, int numVertices, hkReal radius, const BuildConfig& config )
{
	// Create shrunken vertices
	hkArray<hkVector4> vertices;
	HK_ASSERT2( 0xaf14e213, numVertices >= 8, "You should allow for at least 8 vertices." );
	HK_ASSERT2( 0xaf14e211, (numVertices & 0x1) == 0, "You should allow for an even number of vertices." );
	vertices.reserve( numVertices );
	{
		int numVerts = numVertices / 2;
		hkReal halfHeight = halfExtent(1);
		for ( int vi=0; vi<numVerts; vi++ )
		{
			hkReal x = (halfExtent(0)-radius) * hkMath::sin( (2*HK_REAL_PI*vi) / (hkReal)numVerts );
			hkReal z = (halfExtent(2)-radius) * hkMath::cos( (2*HK_REAL_PI*vi) / (hkReal)numVerts );
			vertices.expandByUnchecked(1)->set( x, -halfHeight+radius, z );
			vertices.expandByUnchecked(1)->set( x, +halfHeight-radius, z );
		}
	}

	
	BuildConfig conf = config;
	conf.m_shrinkByRadius = false;	// we pre-shrunk the vertices
	hknpConvexShape* shape = createFromVertices( hkStridedVertices( vertices.begin(), vertices.getSize() ), radius, conf );

	return shape;
}

void hknpConvexShape::buildMassProperties(
	const MassConfig& massConfig, hkDiagonalizedMassProperties& massPropertiesOut ) const
{
	// Assumes the unique vertices occupy the first numVert elements of the vertex array
	const int numUniqueVertices = hknpConvexShapeUtil::getNumberOfUniqueVertices( this );
	hkResult result = hknpConvexShapeUtil::buildHullMassProperties(
		massConfig, &getVertex(0), numUniqueVertices, m_convexRadius, massPropertiesOut );
	if( result == HK_FAILURE )
	{
		// Fall back to AABB approximation.
		hknpShape::buildMassProperties( massConfig, massPropertiesOut );
	}
}

#endif	// !HK_PLATFORM_SPU

#if !defined(HK_PLATFORM_SPU)

hkRefNew<hknpShapeKeyIterator> hknpConvexShape::createShapeKeyIterator( const hknpShapeKeyMask* mask ) const
{
	return hkRefNew<hknpShapeKeyIterator>( new hknpConvexShapeKeyIterator( *this, mask ) );
}

#else

hknpShapeKeyIterator* hknpConvexShape::createShapeKeyIterator( hkUint8* buffer, int bufferSize, const hknpShapeKeyMask* mask ) const
{
	HK_ON_DEBUG( int iteratorSize = HK_NEXT_MULTIPLE_OF(16, sizeof(hknpConvexShapeKeyIterator)); )
	HK_ASSERT( 0xaf1fe143, iteratorSize < bufferSize );
	hknpConvexShapeKeyIterator* it = new (buffer) hknpConvexShapeKeyIterator( *this, mask );
	return it;
}

#endif

#if !defined(HK_PLATFORM_SPU)
void hknpConvexShape::getAllShapeKeys( const hknpShapeKeyPath& shapeKeyPath, const hknpShapeKeyMask* mask, hkFixedCapacityArray<hknpShapeKeyPath>* keyPathsOut ) const
#else
void hknpConvexShape::getAllShapeKeys( const hknpShapeKeyPath& shapeKeyPath, const hknpShapeKeyMask* mask, hkUint8* shapeBuffer, int shapeBufferSize, hkFixedCapacityArray<hknpShapeKeyPath>* keyPathsOut ) const
#endif
{
	if( !mask || mask->isShapeKeyEnabled( shapeKeyPath.getKey() ) )
	{
		keyPathsOut->pushBack( shapeKeyPath );
	}
}

void hknpConvexShape::getLeafShape( hknpShapeKey key, hknpShapeCollector* collector ) const
{
	HK_ASSERT2( 0x1D3469AB, key == HKNP_INVALID_SHAPE_KEY, "Shape key unsupported" );
	collector->checkForReset();
	collector->m_shapeOut = this;
}

int hknpConvexShape::getNumberOfSupportVertices() const
{
	return getNumberOfVertices();
}

const hkcdVertex* hknpConvexShape::getSupportVertices( hkcdVertex* vertexBuffer, int bufferSize ) const
{
	return getVertices();
}

// Reference implementation of getSupportingVertex, not the best performance on XBOX, still pretty good
void hknpConvexShape::getSupportingVertexRef( hkVector4Parameter direction, hkcdVertex& svOut ) const
{
	const hkVector4* HK_RESTRICT fv = getVertices();
	hkVector4 bestDot; hkVector4Util::dot3_4vs4( direction, fv[0], direction, fv[1], direction, fv[2], direction, fv[3], bestDot );
	fv += 4;

	hkIntVector curIndices; curIndices.m_quad = *(const hkQuadUint*)s_curIndices;
	hkIntVector stepIndices; stepIndices.splatImmediate32<4>();
	hkIntVector bestIndices = curIndices;

	// get max dots four at a time
	for (int i = hknpConvexShape::getNumberOfVertices() - 4; i>0; i-=4 )
	{
		curIndices.setAddU32( curIndices, stepIndices );
		hkVector4 curDot;
		hkVector4Util::dot3_4vs4( direction, fv[0], direction, fv[1], direction, fv[2], direction, fv[3], curDot );
		hkVector4Comparison comp = bestDot.less( curDot );
		bestDot.setSelect( comp, curDot, bestDot );
		bestIndices.setSelect( comp, curIndices, bestIndices );
		fv+=4;
	}

	int bestIndex = bestIndices.getComponentAtVectorMax(bestDot);

	// extract vertex
	(hkVector4&)svOut = getVertices()[bestIndex];
}

int hknpConvexShape::getFaceVertices( const int faceId, hkVector4& planeOut, hkcdVertex* const HK_RESTRICT verticesOut ) const
{
	HK_ASSERT2(0xf077e3da, ((faceId >> 24) & 0xff) < getNumberOfVertices(),	"Invalid vertex index");
	HK_ASSERT2(0xf077e3da, ((faceId >> 16) & 0xff) < getNumberOfVertices(),	"Invalid vertex index");
	HK_ASSERT2(0xf077e3da, ((faceId >>  8) & 0xff) < getNumberOfVertices(),	"Invalid vertex index");
	HK_ASSERT2(0xf077e3da, ((faceId      ) & 0xff) < getNumberOfVertices(),	"Invalid vertex index");

	int dummy;
	getFaceInfo(faceId, planeOut, dummy);

	verticesOut[0] = m_vertices[(faceId >> 24) & 0xff];
	verticesOut[1] = m_vertices[(faceId >> 16) & 0xff];
	verticesOut[2] = m_vertices[(faceId >> 8) & 0xff];
	verticesOut[3] = m_vertices[(faceId) & 0xff];

	return 4;
}

/*
 * Havok SDK - Product file, BUILD(#20130912)
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
