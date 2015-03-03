/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Common/Base/Algorithm/PseudoRandom/hkPseudoRandomGenerator.h>
#include <Common/Base/Types/Geometry/hkGeometry.h>
#include <Common/Base/UnitTest/hkUnitTest.h>

#include <Physics/Physics/Collide/Query/Collector/hknpAllHitsCollector.h>
#include <Physics/Physics/Collide/Query/Collector/hknpClosestHitCollector.h>
#include <Physics/Physics/Collide/Shape/hknpShapeQueryInterface.h>
#include <Physics/Physics/Collide/Shape/Composite/Compound/Static/hknpStaticCompoundShape.h>
#include <Physics/Physics/Collide/Shape/Composite/HeightField/Compressed/hknpCompressedHeightFieldShape.h>
#include <Physics/Physics/Collide/Shape/Composite/Masked/hknpMaskedCompositeShape.h>
#include <Physics/Physics/Collide/Shape/Composite/Mesh/Compressed/hknpCompressedMeshShape.h>
#include <Physics/Physics/Collide/Shape/Composite/Mesh/Extern/hknpExternMeshShape.h>
#include <Physics/Physics/Collide/Shape/Convex/Capsule/hknpCapsuleShape.h>
#include <Physics/Physics/Collide/Shape/Convex/Scaled/hknpScaledConvexShape.h>
#include <Physics/Physics/Collide/Shape/Convex/Sphere/hknpSphereShape.h>


#define VERTEX_0 -0.5f, 0.0f,  0.5f
#define VERTEX_1  0.5f, 0.0f,  0.5f
#define VERTEX_2  0.5f, 0.0f, -0.5f
#define VERTEX_3 -0.5f, 0.0f, -0.5f


#define TEST_FILTER_CALLBACK_PARAMETERS_QUERY_SHAPE_FILTER_INFO 1
#define TEST_FILTER_CALLBACK_PARAMETERS_TARGET_SHAPE_FILTER_INFO 2

namespace NpSpatialQueryUnitTest
{
	struct ExternMeshShapeGeometrySource : public hknpExternMeshShape::Mesh
	{
		HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_DEMO );

		ExternMeshShapeGeometrySource( int filterInfo )
		{
			m_filterInfo = filterInfo;
		}

		HK_FORCE_INLINE int getNumTriangles() const
		{
			return 1;
		}

		HK_FORCE_INLINE void getTriangleVertices(int index, hkVector4* verticesOut) const
		{
			verticesOut[0].set( VERTEX_0 );
			verticesOut[1].set( VERTEX_2 );
			verticesOut[2].set( VERTEX_3 );
		}

		HK_FORCE_INLINE hknpShapeTag getTriangleShapeTag(int index) const
		{
			return (hknpShapeTag)m_filterInfo;
		}

		int m_filterInfo;
	};

	class ShapeTagCodec : public hknpShapeTagCodec
	{
		public:

			HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );

			ShapeTagCodec() : hknpShapeTagCodec(USER_CODEC) {}

			static hknpShapeTag HK_CALL encode( hkUint32 collisionFilterInfo )
			{
				return (hknpShapeTag)collisionFilterInfo;
			}

			virtual void decode(hknpShapeTag shapeTag, const Context* context, hkUint32* collisionFilterInfo,
				hknpMaterialId* materialId, hkUint64* userData) const HK_OVERRIDE
			{
				*collisionFilterInfo = (int)shapeTag;
			}
	};

	class CollisionFilter : public hknpCollisionFilter
	{
		public:

			HK_DECLARE_CLASS_ALLOCATOR( HK_MEMORY_CLASS_PHYSICS );

			CollisionFilter()
				:	hknpCollisionFilter( USER_FILTER )
			{
			}

			virtual int filterBodyPairs(
				const hknpSimulationThreadContext& context,
				hknpBodyIdPair* pairs, int numPairs ) const HK_OVERRIDE
			{
				// Won't get called.
				return numPairs;
			}

			virtual bool isCollisionEnabled(
				hknpCollisionQueryType::Enum queryType,
				hknpBroadPhaseLayerIndex layerIndex ) const HK_OVERRIDE
			{
				// Won't get called.
				return true;
			}

			virtual bool isCollisionEnabled(
				hknpCollisionQueryType::Enum queryType,
				hknpBodyId bodyIdA,
				hknpBodyId bodyIdB ) const HK_OVERRIDE
			{
				// Won't get called.
				return true;
			}

			virtual bool isCollisionEnabled(
				hknpCollisionQueryType::Enum queryType,
				const hknpQueryFilterData& queryFilterData,
				const hknpBody& body ) const HK_OVERRIDE
			{
				// Won't get called.
				return true;
			}

			virtual bool isCollisionEnabled(
				hknpCollisionQueryType::Enum queryType,
				bool targetShapeIsB,
				const FilterInput& shapeInputA,
				const FilterInput& shapeInputB ) const HK_OVERRIDE
			{
				const FilterInput& queryInput = getQueryShapeInput( targetShapeIsB, shapeInputA, shapeInputB );
				const FilterInput& targetInput = getTargetShapeInput( targetShapeIsB, shapeInputA, shapeInputB );
				HK_TEST( queryInput.m_filterData.m_collisionFilterInfo == TEST_FILTER_CALLBACK_PARAMETERS_QUERY_SHAPE_FILTER_INFO );
				HK_TEST( targetInput.m_filterData.m_collisionFilterInfo == TEST_FILTER_CALLBACK_PARAMETERS_TARGET_SHAPE_FILTER_INFO );
				return true;
			}
	};

	struct ShapeTypes
	{
		enum Enum
		{
			SPHERE,
			CAPSULE,
			CONVEX_POLYTOPE,
			SCALED_CONVEX,
			COMPRESSED_MESH,
			EXTERN_MESH,
			HEIGHT_FIELD,
			COMPRESSED_MESH_WITH_CONVEX_SHAPES,
			COMPRESSED_MESH_WITH_SCALED_CONVEX_SHAPES,
			STATIC_COMPOUND,
			STATIC_COMPOUND_WITH_SCALED_CHILD,
			STATIC_COMPOUND_RECURSIVE,
			STATIC_COMPOUND_RECURSIVE_WITH_SCALED_LEAF,
		};
	};

	struct ShapeVariants
	{
		ShapeVariants( ShapeTypes::Enum type )
		:	m_type(type)
		{
		}

		ShapeTypes::Enum m_type;
	};

	ShapeVariants shapeTypes[] =
	{
		ShapeVariants( ShapeTypes::SPHERE ),
		ShapeVariants( ShapeTypes::CAPSULE ),
		ShapeVariants( ShapeTypes::CONVEX_POLYTOPE ),
		ShapeVariants( ShapeTypes::SCALED_CONVEX ),
		ShapeVariants( ShapeTypes::COMPRESSED_MESH ),
		ShapeVariants( ShapeTypes::EXTERN_MESH ),
		ShapeVariants( ShapeTypes::HEIGHT_FIELD ),
		ShapeVariants( ShapeTypes::COMPRESSED_MESH_WITH_CONVEX_SHAPES ),
		ShapeVariants( ShapeTypes::COMPRESSED_MESH_WITH_SCALED_CONVEX_SHAPES ),
		ShapeVariants( ShapeTypes::STATIC_COMPOUND ),
		ShapeVariants( ShapeTypes::STATIC_COMPOUND_WITH_SCALED_CHILD ),
		ShapeVariants( ShapeTypes::STATIC_COMPOUND_RECURSIVE ),
		ShapeVariants( ShapeTypes::STATIC_COMPOUND_RECURSIVE_WITH_SCALED_LEAF ),
	};



	hknpConvexShape* createSphereShape()
	{
		hkReal radius = 1.0f;
		return hknpSphereShape::createSphereShape( hkVector4::getZero(), radius );
	}

	hknpConvexShape* createCapsuleShape()
	{
		hkReal halfLength = 0.5f;
		hkVector4 a; a.set(0, halfLength, 0);
		hkVector4 b; b.set(0, -a(1), 0);
		return hknpCapsuleShape::createCapsuleShape( a, b, halfLength*0.3f );
	}

	hknpConvexShape* createConvexShape()
	{
		hkPseudoRandomGenerator rng( 20110613 );
		hkSimdReal scale = hkSimdReal_1;
		hkArray<hkVector4> vertices; vertices.setSize(64);
		for ( int i=0; i<vertices.getSize(); ++i )
		{
			rng.getRandomVector11( vertices[i] );
			vertices[i].mul(scale);
		}
		return hknpConvexShape::createFromVertices(vertices);
	}

	hknpShape* createScaledConvexShape()
	{
		hkVector4 scale; scale.set(1,2,3);
		hknpConvexShape* childShape = createConvexShape();
		hknpShape* shape = new hknpScaledConvexShape( childShape, scale );
		childShape->removeReference();
		return shape;
	}

	hknpShape* createCompressedMeshShapeWithTriangles( int filterInfo )
	{
		hkVector4 v0; v0.set( VERTEX_0 );
		hkVector4 v1; v1.set( VERTEX_1 );
		hkVector4 v2; v2.set( VERTEX_2 );
		hkVector4 v3; v3.set( VERTEX_3 );

		hkGeometry geometry;
		geometry.m_vertices.pushBack( v0 );
		geometry.m_vertices.pushBack( v1 );
		geometry.m_vertices.pushBack( v2 );
		geometry.m_vertices.pushBack( v3 );

		hkGeometry::Triangle& triangle0 = geometry.m_triangles.expandOne();
		triangle0.m_a = 0;
		triangle0.m_b = 2;
		triangle0.m_c = 3;
		triangle0.m_material = filterInfo;

		hkGeometry::Triangle& triangle1 = geometry.m_triangles.expandOne();
		triangle1.m_a = 0;
		triangle1.m_b = 1;
		triangle1.m_c = 2;
		triangle1.m_material = filterInfo;

		hknpDefaultCompressedMeshShapeCinfo meshInfo( &geometry );
		hknpShape* shape = new hknpCompressedMeshShape( meshInfo );

		return shape;
	}

	hknpShape* createExternMeshShapeWithTriangles( ExternMeshShapeGeometrySource* emsSource )
	{
		hknpShape* shape = new hknpExternMeshShape( emsSource );
		return shape;
	}

	hknpShape* createHeightFieldShape( int filterInfo )
	{
		hknpHeightFieldShapeCinfo cInfo;
		cInfo.m_dispatchType = hknpCollisionDispatchType::COMPOSITE;
		cInfo.m_xRes = 2;
		cInfo.m_zRes = 2;
		cInfo.m_scale.set( 1,1,1 );
		cInfo.m_minMaxTreeCoarseness = 1;

		int numSamples = cInfo.m_xRes * cInfo.m_zRes;
		hkArray<hkUint16> samples; samples.reserve( numSamples );
		hkArray<hknpShapeTag> shapeTags; shapeTags.reserve( numSamples );
		for ( int z = 0; z < cInfo.m_zRes; z++ )
		{
			for ( int x = 0; x < cInfo.m_xRes; x++ )
			{
				samples.pushBackUnchecked( static_cast<hkUint16>(0) );
				shapeTags.pushBackUnchecked( ShapeTagCodec::encode( filterInfo ) );
			}
		}
		hknpCompressedHeightFieldShape* fhShape = new hknpCompressedHeightFieldShape( cInfo, samples, 0.0f, 0.001f, &shapeTags );
		fhShape->buildMinMaxTree();
		return fhShape;
	}

	hknpShape* createCompressedMeshShapeWithConvexShape( int filterInfo )
	{
		hknpShape* childShape = createConvexShape();
		hkTransform childToRoot; childToRoot.setIdentity();

		hknpShapeInstance instance;
		instance.setShape( childShape );
		instance.setTransform( childToRoot );
		instance.setShapeTag( ShapeTagCodec::encode( filterInfo ) );

		childShape->removeReference();

		hknpDefaultCompressedMeshShapeCinfo meshInfo( HK_NULL, &instance, 1 );
		hknpShape* shape = new hknpCompressedMeshShape( meshInfo );
		return shape;
	}

	hknpShape* createCompressedMeshShapeWithScaledConvexShape( int filterInfo )
	{
		hknpShape* childShape = createConvexShape();
		hkTransform childToRoot; childToRoot.setIdentity();

		hkVector4 scale; scale.set( 0.9f, 0.8f, 0.7f );

		hknpShapeInstance instance;
		instance.setShape( childShape );
		instance.setTransform( childToRoot );
		instance.setShapeTag( ShapeTagCodec::encode( filterInfo ) );
		instance.setScale( scale );

		childShape->removeReference();

		hknpDefaultCompressedMeshShapeCinfo meshInfo( HK_NULL, &instance, 1 );
		hknpShape* shape = new hknpCompressedMeshShape( meshInfo );
		return shape;
	}

	hknpShape* createStaticCompoundShape( int filterInfo )
	{
		hknpShape* childShape = createConvexShape();
		hkTransform childToRoot; childToRoot.setIdentity();

		hknpShapeInstance instance;
		instance.setShape( childShape );
		instance.setTransform( childToRoot );
		instance.setShapeTag( ShapeTagCodec::encode( filterInfo ) );

		childShape->removeReference();

		hknpShape* shape = new hknpStaticCompoundShape( &instance, 1 );
		return shape;
	}

	hknpShape* createStaticCompoundShapeWithScaledChild( int filterInfo )
	{
		hknpShape* childShape = createConvexShape();
		hkTransform childToRoot; childToRoot.setIdentity();

		hkVector4 scale; scale.set( 0.9f, 0.8f, 0.7f );

		hknpShapeInstance instance;
		instance.setShape( childShape );
		instance.setTransform( childToRoot );
		instance.setShapeTag( ShapeTagCodec::encode( filterInfo ) );
		instance.setScale( scale );

		childShape->removeReference();

		hknpShape* shape = new hknpStaticCompoundShape( &instance, 1 );
		return shape;
	}

	hknpShape* createRecursiveStaticCompoundShape( int filterInfo )
	{
		hknpShape* childShape = createStaticCompoundShape( filterInfo );
		hkTransform childToRoot; childToRoot.setIdentity();

		hknpShapeInstance instance;
		instance.setShape( childShape );
		instance.setTransform( childToRoot );
		instance.setShapeTag( ShapeTagCodec::encode( filterInfo ) );

		childShape->removeReference();

		hknpShape* shape = new hknpStaticCompoundShape( &instance, 1 );
		return shape;
	}

	hknpShape* createRecursiveStaticCompoundShapeWithScaledLeaf( int filterInfo )
	{
		hknpShape* childShape = createStaticCompoundShapeWithScaledChild( filterInfo );
		hkTransform childToRoot; childToRoot.setIdentity();

		hknpShapeInstance instance;
		instance.setShape( childShape );
		instance.setTransform( childToRoot );
		instance.setShapeTag( ShapeTagCodec::encode( filterInfo ) );

		childShape->removeReference();

		hknpShape* shape = new hknpStaticCompoundShape( &instance, 1 );
		return shape;
	}

	hknpShape* createShape( int index, int rootFilterInfo, ExternMeshShapeGeometrySource* emsSource )
	{
		hknpShape* shape = HK_NULL;

		switch ( shapeTypes[index].m_type )
		{
			case ShapeTypes::SPHERE:
			{
				shape = createSphereShape();
				break;
			}
			case ShapeTypes::CAPSULE:
			{
				shape = createCapsuleShape();
				break;
			}
			case ShapeTypes::CONVEX_POLYTOPE:
			{
				shape = createConvexShape();
				break;
			}
			case ShapeTypes::SCALED_CONVEX:
			{
				shape = createScaledConvexShape();
				break;
			}
			case ShapeTypes::COMPRESSED_MESH:
			{
				shape = createCompressedMeshShapeWithTriangles( rootFilterInfo );
				break;
			}
			case ShapeTypes::EXTERN_MESH:
			{
				shape = createExternMeshShapeWithTriangles( emsSource );
				break;
			}
			case ShapeTypes::HEIGHT_FIELD:
			{
				shape = createHeightFieldShape( rootFilterInfo );
				break;
			}
			case ShapeTypes::COMPRESSED_MESH_WITH_CONVEX_SHAPES:
			{
				shape = createCompressedMeshShapeWithConvexShape( rootFilterInfo );
				break;
			}
			case ShapeTypes::COMPRESSED_MESH_WITH_SCALED_CONVEX_SHAPES:
			{
				shape = createCompressedMeshShapeWithScaledConvexShape( rootFilterInfo );
				break;
			}
			case ShapeTypes::STATIC_COMPOUND:
			{
				shape = createStaticCompoundShape( rootFilterInfo );
				break;
			}
			case ShapeTypes::STATIC_COMPOUND_WITH_SCALED_CHILD:
			{
				shape = createStaticCompoundShapeWithScaledChild( rootFilterInfo );
				break;
			}
			case ShapeTypes::STATIC_COMPOUND_RECURSIVE:
			{
				shape = createRecursiveStaticCompoundShape( rootFilterInfo );
				break;
			}
			case ShapeTypes::STATIC_COMPOUND_RECURSIVE_WITH_SCALED_LEAF:
			{
				shape = createRecursiveStaticCompoundShapeWithScaledLeaf( rootFilterInfo );
				break;
			}
			default:
			{
				HK_TEST( false );
			}
		}

		HK_TEST( shape != HK_NULL );

		return shape;
	}

	void testAabbFilterCallbackParameters( int queryFilterInfo, const hknpShape& targetShape, int targetRootFilterInfo )
	{
		hkTransform targetShapeToWorld; targetShapeToWorld.setIdentity();

		hknpCollisionQueryContext queryContext( HK_NULL, HK_NULL );
		ShapeTagCodec shapeTagCodec;
		queryContext.m_shapeTagCodec = &shapeTagCodec;

		CollisionFilter filter;

		hknpAabbQuery query;
		query.m_aabb.m_min.set( -10, -10, -10 );
		query.m_aabb.m_max.set(  10,  10,  10 );
		query.m_filter = &filter;
		query.m_filterData.m_collisionFilterInfo = queryFilterInfo;

		hknpQueryFilterData targetFilterData;
		targetFilterData.m_collisionFilterInfo = targetRootFilterInfo;

		hknpShapeQueryInfo targetShapeInfo;
		targetShapeInfo.m_rootShape		= &targetShape;
		targetShapeInfo.m_shapeToWorld	= &targetShapeToWorld;

		hknpAllHitsCollector collector;

		hknpShapeQueryInterface::queryAabb( &queryContext, query, targetShape, targetFilterData, targetShapeInfo, &collector );

		HK_ASSERT( 0xaf1e2431, collector.hasHit() );
	}

	void testClosestPointsFilterCallbackParameters( const hknpShape& queryShape, int queryRootFilterInfo, const hknpShape& targetShape, int targetRootFilterInfo )
	{
		hkTransform queryShapeToWorld; queryShapeToWorld.setIdentity(); queryShapeToWorld.getTranslation().set( -5, 0, 0 );
		hkTransform targetShapeToWorld; targetShapeToWorld.setIdentity();

		hknpCollisionQueryDispatcher dispatcher;

		hknpCollisionQueryContext queryContext;
		ShapeTagCodec shapeTagCodec;
		queryContext.m_shapeTagCodec = &shapeTagCodec;
		queryContext.m_dispatcher = &dispatcher;

		CollisionFilter filter;

		hknpClosestPointsQuery query( queryShape, 10.0f );
		query.m_filter = &filter;
		query.m_filterData.m_collisionFilterInfo = queryRootFilterInfo;

		hknpShapeQueryInfo queryShapeInfo;
		queryShapeInfo.m_rootShape		= &queryShape;
		queryShapeInfo.m_shapeToWorld	= &queryShapeToWorld;

		hknpQueryFilterData targetFilterData;
		targetFilterData.m_collisionFilterInfo = targetRootFilterInfo;

		hknpShapeQueryInfo targetShapeInfo;
		targetShapeInfo.m_rootShape		= &targetShape;
		targetShapeInfo.m_shapeToWorld	= &targetShapeToWorld;

		hknpClosestHitCollector collector;
		hknpShapeQueryInterface::getClosestPoints( &queryContext, query, queryShapeInfo, targetShape, targetFilterData, targetShapeInfo, &collector );

		HK_ASSERT( 0xaf1e2432, collector.hasHit() );
	}

	void testRayCastFilterCallbackParameters( int queryFilterInfo, const hknpShape& targetShape, int targetRootFilterInfo )
	{
		hkTransform targetShapeToWorld; targetShapeToWorld.setIdentity();

		hknpInplaceTriangleShape targetTriangle( 0.0f );
		hknpCollisionQueryContext queryContext( HK_NULL, targetTriangle.getTriangleShape() );
		ShapeTagCodec shapeTagCodec;
		queryContext.m_shapeTagCodec = &shapeTagCodec;

		hkVector4 castOrigin; castOrigin.set( 0, 5, 0 );
		hkVector4 castDirection = hkVector4::getConstant<HK_QUADREAL_0100>(); castDirection.mul( hkSimdReal_Minus1 );

		CollisionFilter filter;

		hknpRayCastQuery query( castOrigin, castDirection, hkSimdReal::fromFloat(10) );
		query.m_flags = hkcdRayQueryFlags::ENABLE_INSIDE_HITS;
		query.m_filter = &filter;
		query.m_filterData.m_collisionFilterInfo = queryFilterInfo;

		hknpQueryFilterData targetFilterData;
		targetFilterData.m_collisionFilterInfo = targetRootFilterInfo;

		hknpShapeQueryInfo targetShapeInfo;
		targetShapeInfo.m_rootShape		= &targetShape;
		targetShapeInfo.m_shapeToWorld	= &targetShapeToWorld;

		hknpClosestHitCollector collector;
		hknpShapeQueryInterface::castRay( &queryContext, query, targetShape, targetFilterData, targetShapeInfo, &collector );

		HK_ASSERT( 0xaf1e2433, collector.hasHit() );
	}

	void testShapeCastFilterCallbackParameters( const hknpShape& queryShape, int queryRootFilterInfo, const hknpShape& targetShape, int targetRootFilterInfo )
	{
		hkTransform queryShapeToWorld; queryShapeToWorld.setIdentity(); queryShapeToWorld.getTranslation().set( 0, 5, 0 );
		hkTransform targetShapeToWorld; targetShapeToWorld.setIdentity();

		hknpCollisionQueryDispatcher dispatcher;

		hknpCollisionQueryContext queryContext;
		ShapeTagCodec shapeTagCodec;
		queryContext.m_shapeTagCodec = &shapeTagCodec;
		queryContext.m_dispatcher = &dispatcher;

		hkVector4 castOriginTargetSpace;
		castOriginTargetSpace._setTransformedInversePos( targetShapeToWorld, queryShapeToWorld.getTranslation() );

		hkVector4 castDirection = hkVector4::getConstant<HK_QUADREAL_0100>(); castDirection.mul( hkSimdReal_Minus1 );
		hkVector4 castDirectionTargetSpace;
		castDirectionTargetSpace._setRotatedInverseDir( targetShapeToWorld.getRotation(), castDirection );

		CollisionFilter filter;

		hknpShapeCastQuery query( queryShape, castOriginTargetSpace, castDirectionTargetSpace, hkSimdReal::fromFloat(10) );
		query.m_filter = &filter;
		query.m_filterData.m_collisionFilterInfo = queryRootFilterInfo;

		hknpShapeQueryInfo queryShapeInfo;
		queryShapeInfo.m_rootShape		= &queryShape;
		queryShapeInfo.m_shapeToWorld	= &queryShapeToWorld;

		hknpQueryFilterData targetFilterData;
		targetFilterData.m_collisionFilterInfo = targetRootFilterInfo;

		hknpShapeQueryInfo targetShapeInfo;
		targetShapeInfo.m_rootShape		= &targetShape;
		targetShapeInfo.m_shapeToWorld	= &targetShapeToWorld;

		hknpClosestHitCollector collector;
		hknpShapeQueryInterface::castShape( &queryContext, query, queryShapeInfo, targetShape, targetFilterData, targetShapeInfo, &collector );

		HK_ASSERT( 0xaf1e2434, collector.hasHit() );
	}

	void testFilterCallbackParameters()
	{
		int numShapeTypes = sizeof(shapeTypes) / sizeof(ShapeVariants);

		int queryShapeFilterInfo = TEST_FILTER_CALLBACK_PARAMETERS_QUERY_SHAPE_FILTER_INFO;
		int targetShapeFilterInfo = TEST_FILTER_CALLBACK_PARAMETERS_TARGET_SHAPE_FILTER_INFO;

		ExternMeshShapeGeometrySource externQueryMesh( queryShapeFilterInfo );
		ExternMeshShapeGeometrySource externTargetMesh( targetShapeFilterInfo );

		for ( int ti = 0; ti < numShapeTypes; ti++ )
		{
			hknpShape* targetShape = createShape( ti, targetShapeFilterInfo, &externTargetMesh );

			if ( targetShape )
			{
				NpSpatialQueryUnitTest::testAabbFilterCallbackParameters( queryShapeFilterInfo, *targetShape, targetShapeFilterInfo );
				NpSpatialQueryUnitTest::testRayCastFilterCallbackParameters( queryShapeFilterInfo, *targetShape, targetShapeFilterInfo );

				for ( int qi = 0; qi < numShapeTypes; qi++ )
				{
					hknpShape* queryShape = createShape( qi, queryShapeFilterInfo, &externQueryMesh );

					if ( queryShape )
					{
						NpSpatialQueryUnitTest::testClosestPointsFilterCallbackParameters( *queryShape, queryShapeFilterInfo, *targetShape, targetShapeFilterInfo );
						NpSpatialQueryUnitTest::testShapeCastFilterCallbackParameters( *queryShape, queryShapeFilterInfo, *targetShape, targetShapeFilterInfo );

						queryShape->removeReference();
					}
				}

				targetShape->removeReference();
			}
		}
	}

	void testAabbAssociatedFilterData( hkUint64 userData, hkUint32 filterInfo, hknpMaterialId materialId, const hknpShape& targetShape )
	{
		ShapeTagCodec shapeTagCodec;

		hknpCollisionQueryContext queryContext;
		queryContext.m_shapeTagCodec = &shapeTagCodec;

		hknpAabbQuery query;
		query.m_aabb.m_min.set( -10, -10, -10 );
		query.m_aabb.m_max.set(  10,  10,  10 );
		query.m_filterData.m_userData = userData;
		query.m_filterData.m_collisionFilterInfo = filterInfo;
		query.m_filterData.m_materialId = materialId;

		hknpQueryFilterData targetShapeFilterData;

		hknpShapeQueryInfo targetShapeInfo;

		hknpAllHitsCollector collector;

		hknpShapeQueryInterface::queryAabb( &queryContext, query, targetShape, targetShapeFilterData, targetShapeInfo, &collector );

		HK_ASSERT( 0xaf1e2435, collector.hasHit() );
		HK_TEST( collector.getHits()[0].m_queryBodyInfo.m_shapeUserData == userData );
		HK_TEST( collector.getHits()[0].m_queryBodyInfo.m_shapeCollisionFilterInfo == filterInfo );
		HK_TEST( collector.getHits()[0].m_queryBodyInfo.m_shapeMaterialId.value() == materialId.value() );
	}

	void testRayCastAssociatedFilterData( hkUint64 userData, hkUint32 filterInfo, hknpMaterialId materialId, const hknpShape& targetShape )
	{
		hkTransform targetShapeToWorld; targetShapeToWorld.setIdentity();

		hkVector4 castOrigin; castOrigin.set( 0, 5, 0 );
		hkVector4 castDirection = hkVector4::getConstant<HK_QUADREAL_0100>(); castDirection.mul( hkSimdReal_Minus1 );

		hknpRayCastQuery query( castOrigin, castDirection, hkSimdReal::fromFloat(10) );
		query.m_flags = hkcdRayQueryFlags::ENABLE_INSIDE_HITS;
		query.m_filterData.m_userData = userData;
		query.m_filterData.m_collisionFilterInfo = filterInfo;
		query.m_filterData.m_materialId = materialId;

		hknpClosestHitCollector collector;

		hknpShapeQueryInterface::castRay( query, targetShape, targetShapeToWorld, &collector );

		HK_ASSERT( 0xaf1e2436, collector.hasHit() );
		HK_TEST( collector.getHits()[0].m_queryBodyInfo.m_shapeUserData == userData );
		HK_TEST( collector.getHits()[0].m_queryBodyInfo.m_shapeCollisionFilterInfo == filterInfo );
		HK_TEST( collector.getHits()[0].m_queryBodyInfo.m_shapeMaterialId.value() == materialId.value() );
	}

	void testUnaryQueryAssociatedFilterData()
	{
		int numShapeTypes = sizeof(shapeTypes) / sizeof(ShapeVariants);

		hkUint64 associatedUserData = 3;
		hkUint32 associatedFilterInfo = 30;
		hknpMaterialId associatedMaterialId(300);

		int targetShapeFilterInfo = 0;

		ExternMeshShapeGeometrySource externTargetMesh( targetShapeFilterInfo );

		for ( int ti = 0; ti < numShapeTypes; ti++ )
		{
			hknpShape* targetShape = createShape( ti, targetShapeFilterInfo, &externTargetMesh );

			if ( targetShape )
			{
				NpSpatialQueryUnitTest::testAabbAssociatedFilterData( associatedUserData, associatedFilterInfo, associatedMaterialId, *targetShape );
				NpSpatialQueryUnitTest::testRayCastAssociatedFilterData( associatedUserData, associatedFilterInfo, associatedMaterialId, *targetShape );

				targetShape->removeReference();
			}
		}
	}
}


int NpSpatialQueryUnitTest_main()
{
	NpSpatialQueryUnitTest::testFilterCallbackParameters();

	NpSpatialQueryUnitTest::testUnaryQueryAssociatedFilterData();

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif

HK_TEST_REGISTER( NpSpatialQueryUnitTest_main, "Fast", "Physics/Test/UnitTest/Physics/", __FILE__ );

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
