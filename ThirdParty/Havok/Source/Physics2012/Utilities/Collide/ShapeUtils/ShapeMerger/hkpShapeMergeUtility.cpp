/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Physics2012/Utilities/Collide/ShapeUtils/ShapeMerger/hkpShapeMergeUtility.h>

// All the shape types used.
#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Collide/Shape/Deprecated/ConvexList/hkpConvexListShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexTranslate/hkpConvexTranslateShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexTransform/hkpConvexTransformShape.h>
#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/hkpMoppBvTreeShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/List/hkpListShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/SimpleMesh/hkpSimpleMeshShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/StorageExtendedMesh/hkpStorageExtendedMeshShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/Mesh/hkpMeshMaterial.h>

// Rigid body
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>


extern const hkClass hkpStorageExtendedMeshShapeMaterialClass;

namespace
{

struct ConvexShapeInfo
{
	hkTransform m_transform;
	hkpMaterial m_material;
	hkUint32 m_filterInfo;
	hkUlong m_bodyUserData;
	const hkpConvexShape* m_shape;
};


void _extractConvexShape( const hkpConvexShape* shape, const hkTransform& parentTransform, const hkpMaterial& material, hkUint32 filterInfo, hkUlong userData, hkArray< ConvexShapeInfo >& convexShapes );
void _extractConvexTranslateShape( const hkpConvexTranslateShape* shape, const hkTransform& parentTransform, const hkpMaterial& material, hkUint32 filterInfo, hkUlong userData, hkArray< ConvexShapeInfo >& convexShapes );
void _extractConvexTransformShape( const hkpConvexTransformShape* shape, const hkTransform& parentTransform, const hkpMaterial& material, hkUint32 filterInfo, hkUlong userData, hkArray< ConvexShapeInfo >& convexShapes );
void _extractConvexListShape( const hkpConvexListShape* shape, const hkTransform& parentTransform, const hkpMaterial& material, hkUint32 filterInfo, hkUlong userData, hkArray< ConvexShapeInfo >& convexShapes );
void _extractListShape( const hkpListShape* shape, const hkTransform& parentTransform, const hkpMaterial& material, hkUint32 filterInfo, hkUlong userData, hkArray< ConvexShapeInfo >& convexShapes, hkpStorageExtendedMeshShape* mesh );
void _extractMoppShape( const hkpMoppBvTreeShape* shape, const hkTransform& parentTransform, const hkpMaterial& material, hkUint32 filterInfo, hkUlong userData, hkArray< ConvexShapeInfo >& convexShapes, hkpStorageExtendedMeshShape* mesh );
void _extractSimpleMeshShape( const hkpSimpleMeshShape* shape, const hkTransform& parentTransform, const hkpMaterial& material, hkUint32 filterInfo, hkUlong userData, hkArray< ConvexShapeInfo >& convexShapes, hkpStorageExtendedMeshShape* mesh );
void _extractExtendedMeshShape( const hkpExtendedMeshShape* shape, const hkTransform& parentTransform, const hkpMaterial& material, hkUint32 filterInfo, hkUlong userData, hkArray< ConvexShapeInfo >& convexShapes, hkpStorageExtendedMeshShape* mesh );
void _extractShapes( const hkpShape* shape, const hkTransform& parentTransform, const hkpMaterial& material, hkUint32 filterInfo, hkUlong userData, hkArray< ConvexShapeInfo >& convexShapes, hkpStorageExtendedMeshShape* mesh );


HK_FORCE_INLINE bool isLess( hkVector4Parameter lhs, hkVector4Parameter rhs )
{
	const hkVector4Comparison eq = lhs.equal(rhs);
	const hkVector4Comparison ls = lhs.less(rhs);

	// Lexicographical compare of points 
	if ( ls.anyIsSet<hkVector4ComparisonMask::MASK_X>() )
	{
		return true;
	}

	if ( eq.anyIsSet<hkVector4ComparisonMask::MASK_X>() && ls.anyIsSet<hkVector4ComparisonMask::MASK_Y>() )
	{
		return true;
	}

	if ( eq.allAreSet<hkVector4ComparisonMask::MASK_XY>() && ls.anyIsSet<hkVector4ComparisonMask::MASK_Z>() )
	{
		return true;
	}

	return false;
}


HK_FORCE_INLINE hkBool operator<( const ConvexShapeInfo& lhs, const ConvexShapeInfo& rhs )
{
	return isLess( lhs.m_transform.getRotation().getColumn<0>() , rhs.m_transform.getRotation().getColumn<0>()) && 
		   isLess( lhs.m_transform.getRotation().getColumn<1>() , rhs.m_transform.getRotation().getColumn<1>()) && 
		   isLess( lhs.m_transform.getRotation().getColumn<2>() , rhs.m_transform.getRotation().getColumn<2>()) && 
		   isLess( lhs.m_transform.getTranslation() , rhs.m_transform.getTranslation());
}


HK_FORCE_INLINE hkBool32 isNot( hkVector4Parameter lhs, hkVector4Parameter rhs )
{
	const hkVector4Comparison notEq = lhs.notEqual(rhs);
	return notEq.anyIsSet<hkVector4ComparisonMask::MASK_XYZ>();
}


HK_FORCE_INLINE hkBool32 _hasDifferentTransform( const ConvexShapeInfo& lhs, const ConvexShapeInfo& rhs )
{
	return isNot(lhs.m_transform.getRotation().getColumn<0>(), rhs.m_transform.getRotation().getColumn<0>()) | 
		   isNot(lhs.m_transform.getRotation().getColumn<1>(), rhs.m_transform.getRotation().getColumn<1>()) | 
		   isNot(lhs.m_transform.getRotation().getColumn<2>(), rhs.m_transform.getRotation().getColumn<2>()) | 
		   isNot(lhs.m_transform.getTranslation(), rhs.m_transform.getTranslation());
}


HK_FORCE_INLINE hkUint16 _findMaterialIndex( const hkArray< hkpStorageExtendedMeshShape::Material >& materials, hkpStorageExtendedMeshShape::Material material )
{
	for ( int i = 0; i < materials.getSize(); ++i )
	{
		if ( materials[ i ].m_filterInfo == material.m_filterInfo && materials[ i ].m_friction == material.m_friction && materials[ i ].m_restitution == material.m_restitution )
		{
			return hkUint16( i );
		}
	}

	return hkUint16( materials.getSize() );
}


void _extractConvexShape( const hkpConvexShape* shape, const hkTransform& parentTransform, const hkpMaterial& material, hkUint32 filterInfo, hkUlong userData, hkArray< ConvexShapeInfo >& convexShapes )
{
	ConvexShapeInfo& info = convexShapes.expandOne();
	info.m_transform = parentTransform;
	info.m_material = material;
	info.m_filterInfo = filterInfo;
	info.m_bodyUserData = userData;
	info.m_shape = shape;
}


void _extractConvexTranslateShape( const hkpConvexTranslateShape* shape, const hkTransform& parentTransform, const hkpMaterial& material, hkUint32 filterInfo, hkUlong userData, hkArray< ConvexShapeInfo >& convexShapes )
{
	ConvexShapeInfo& info = convexShapes.expandOne();
	info.m_transform = parentTransform;
	info.m_material = material;
	info.m_filterInfo = filterInfo;
	info.m_bodyUserData = userData;
	info.m_shape = shape->getChildShape();

	hkTransform shapeTransform;
	shapeTransform.setIdentity();
	shapeTransform.setTranslation( shape->getTranslation() );

	info.m_transform.setMulEq( shapeTransform );
}


void _extractConvexTransformShape( const hkpConvexTransformShape* shape, const hkTransform& parentTransform, const hkpMaterial& material, hkUint32 filterInfo, hkUlong userData, hkArray< ConvexShapeInfo >& convexShapes )
{
	ConvexShapeInfo& info = convexShapes.expandOne();
	info.m_transform = parentTransform;
	info.m_material = material;
	info.m_filterInfo = filterInfo;
	info.m_bodyUserData = userData;
	info.m_shape = shape->getChildShape();

	hkTransform shapeTransform; shape->getTransform( &shapeTransform );
	info.m_transform.setMulEq( shapeTransform );
}


void _extractConvexListShape( const hkpConvexListShape* shape, const hkTransform& parentTransform, const hkpMaterial& material, hkUint32 filterInfo, hkUlong userData, hkArray< ConvexShapeInfo >& convexShapes, hkpStorageExtendedMeshShape* mesh )
{
	for ( hkpShapeKey key = shape->getFirstKey(); key != HK_INVALID_SHAPE_KEY; key = shape->getNextKey( key ) )
	{
		hkpShapeBuffer shapeBuffer;
		const hkpShape* childShape = shape->getChildShape( key, shapeBuffer );
		HK_ASSERT( 0xdd1a1aee, childShape != static_cast< void* >( shapeBuffer ) );

		_extractShapes( childShape, parentTransform, material, filterInfo, userData, convexShapes, mesh );
	}
}


void _extractListShape( const hkpListShape* shape, const hkTransform& parentTransform, const hkpMaterial& material, hkUint32 filterInfo, hkUlong userData, hkArray< ConvexShapeInfo >& convexShapes, hkpStorageExtendedMeshShape* mesh )
{
	for ( hkpShapeKey key = shape->getFirstKey(); key != HK_INVALID_SHAPE_KEY; key = shape->getNextKey( key ) )
	{
		hkpShapeBuffer shapeBuffer;
		const hkpShape* childShape = shape->getChildShape( key, shapeBuffer );
		HK_ASSERT( 0xdd1a1aee, childShape != static_cast< void* >( shapeBuffer ) );

		_extractShapes( childShape, parentTransform, material, filterInfo, userData, convexShapes, mesh );
	}
}


void _extractMoppShape( const hkpMoppBvTreeShape* shape, const hkTransform& parentTransform, const hkpMaterial& material, hkUint32 filterInfo, hkUlong userData, hkArray< ConvexShapeInfo >& convexShapes, hkpStorageExtendedMeshShape* mesh )
{
	const hkpShapeCollection* collection = shape->getShapeCollection();
	_extractShapes( collection, parentTransform, material, filterInfo, userData, convexShapes, mesh );
}


void _extractSimpleMeshShape( const hkpSimpleMeshShape* shape, const hkTransform& parentTransform, const hkpMaterial& material, hkUint32 filterInfo, hkUlong userData, hkArray< ConvexShapeInfo >& convexShapes, hkpStorageExtendedMeshShape* mesh )
{
	// Prepare transformed vertices
	hkLocalArray< hkVector4 > transformedVertices( shape->m_vertices.getSize() );
	transformedVertices.setSize( shape->m_vertices.getSize() );

	hkVector4Util::transformPoints( parentTransform, &shape->m_vertices[ 0 ], shape->m_vertices.getSize(), &transformedVertices[ 0 ] );
	


	// Prepare material
	hkUint16 indexBase = 0;

	hkpStorageExtendedMeshShape::Material meshMaterial;
	meshMaterial.m_filterInfo = filterInfo;
	meshMaterial.m_friction.setReal<true>(material.getFriction());
	meshMaterial.m_restitution.setReal<true>(material.getRestitution());
	meshMaterial.m_userData = userData;


	// Add new subpart
	hkpExtendedMeshShape::TrianglesSubpart subpart;
	subpart.m_numVertices = transformedVertices.getSize();
	subpart.m_vertexBase = &transformedVertices[ 0 ](0);
	subpart.m_vertexStriding = sizeof( hkVector4 );
	subpart.m_numTriangleShapes = shape->m_triangles.getSize();
	subpart.m_indexBase = &shape->m_triangles[ 0 ];
	subpart.m_indexStriding = sizeof( hkpSimpleMeshShape::Triangle );
	subpart.m_stridingType = hkpExtendedMeshShape::INDICES_INT32;
	subpart.setNumMaterials(1);
	subpart.m_materialBase = &meshMaterial;
	subpart.m_materialStriding = sizeof( hkpStorageExtendedMeshShape::Material );
	subpart.m_materialIndexBase = &indexBase;
	subpart.m_materialIndexStriding = sizeof( hkUint16 );
	subpart.setMaterialIndexStridingType(hkpExtendedMeshShape::MATERIAL_INDICES_INT16);
	subpart.m_userData = userData;

	mesh->addTrianglesSubpart( subpart );
}


void _extractExtendedMeshShape( const hkpExtendedMeshShape* shape, const hkTransform& parentTransform, const hkpMaterial& material, hkUint32 filterInfo, hkUlong userData, hkArray< ConvexShapeInfo >& convexShapes, hkpStorageExtendedMeshShape* mesh )
{
	for ( int i = 0; i < shape->getNumShapesSubparts(); ++i )
	{
		const hkpExtendedMeshShape::ShapesSubpart& subpart = shape->getShapesSubpartAt( i );

		hkTransform shapeTransform; shapeTransform.set( subpart.getRotation(), subpart.getTranslation() );

		hkTransform mergedTransform;
		mergedTransform.setMul( parentTransform, shapeTransform );

		for ( int k = 0; k < subpart.m_childShapes.getSize(); ++k )
		{
			const hkpShape* childShape = subpart.m_childShapes[ k ];
			_extractShapes( childShape, mergedTransform, material, filterInfo, userData, convexShapes, mesh );
		}
	}

	for ( int i = 0; i < shape->getNumTrianglesSubparts(); ++i )
	{
		hkpExtendedMeshShape::TrianglesSubpart subpart = shape->getTrianglesSubpartAt( i );

		const hkReal* vertexBase = subpart.m_vertexBase;
		hkLocalArray< hkVector4 > vertices( subpart.m_numVertices );
		vertices.setSize( subpart.m_numVertices );

		for( int k = 0; k < subpart.m_numVertices; ++k )
		{
			hkVector4 vertex; 
			vertex.load<3,HK_IO_NATIVE_ALIGNED>( vertexBase );

			vertices[ k ]._setTransformedPos( parentTransform, vertex );
			vertexBase = hkAddByteOffsetConst( vertexBase, subpart.m_vertexStriding );
		}

		subpart.m_vertexBase     = &vertices[ 0 ](0);
		subpart.m_vertexStriding = sizeof( hkVector4 );
		subpart.m_userData = userData;

		mesh->addTrianglesSubpart( subpart );
	}
}



void _extractShapes( const hkpShape* shape, const hkTransform& parentTransform, const hkpMaterial& material, hkUint32 filterInfo, hkUlong userData, hkArray< ConvexShapeInfo >& convexShapes, hkpStorageExtendedMeshShape* mesh )
{
	switch ( shape->getType() )
	{
	case hkcdShapeType::SPHERE:
	case hkcdShapeType::CAPSULE:
	case hkcdShapeType::CYLINDER:
	case hkcdShapeType::BOX:
	case hkcdShapeType::CONVEX_VERTICES:
		_extractConvexShape( static_cast< const hkpConvexShape* >( shape ), parentTransform, material, filterInfo, userData, convexShapes );
		break;

	case hkcdShapeType::CONVEX_TRANSLATE:
		_extractConvexTranslateShape( static_cast< const hkpConvexTranslateShape* >( shape ), parentTransform, material, filterInfo, userData, convexShapes );
		break;

	case hkcdShapeType::CONVEX_TRANSFORM:
		_extractConvexTransformShape( static_cast< const hkpConvexTransformShape* >( shape ), parentTransform, material, filterInfo, userData, convexShapes );
		break;

	case hkcdShapeType::CONVEX_LIST:
		_extractConvexListShape( static_cast< const hkpConvexListShape* >( shape ), parentTransform, material, filterInfo, userData, convexShapes, mesh );
		break;

	case hkcdShapeType::LIST:
		_extractListShape( static_cast< const hkpListShape* >( shape ), parentTransform, material, filterInfo, userData, convexShapes, mesh );
		break;

	case hkcdShapeType::MOPP:
		_extractMoppShape( static_cast< const hkpMoppBvTreeShape* >( shape ), parentTransform, material, filterInfo, userData, convexShapes, mesh );
		break;

	case hkcdShapeType::TRIANGLE_COLLECTION:
		_extractSimpleMeshShape( static_cast< const hkpSimpleMeshShape* >( shape ), parentTransform, material, filterInfo, userData, convexShapes, mesh );
		break;

	case hkcdShapeType::EXTENDED_MESH:
		_extractExtendedMeshShape( static_cast< const hkpExtendedMeshShape* >( shape ), parentTransform, material, filterInfo, userData, convexShapes, mesh );
		break;


	default:
		HK_ASSERT( 0xdd25aaee, !"Should never get here - invalid shape associated with rigid body" );
		break;

	}
}


}  // anonymous namespace


hkResult hkpShapeMergeUtility::mergeShapes( const hkArray< hkpRigidBody* >& bodies, hkpStorageExtendedMeshShape* outputMesh )
{
	HK_ASSERT ( 0xdd12eefa, bodies.getSize() != 0 && outputMesh != HK_NULL );
	

	// Collect convex shapes and copy triangle subparts over to output mesh immediately
	hkArray< ConvexShapeInfo > convexShapes;
	for ( int i = 0; i < bodies.getSize(); ++i )
	{
		hkpRigidBody* body = bodies[ i ];
		if ( !body->isFixed() )
		{
			HK_WARN( 0xf0ed343e, "Merging non fixed body, are you sure?");
		}

		_extractShapes( body->getCollidable()->getShape(), body->getTransform(), body->getMaterial(), body->getCollidable()->getCollisionFilterInfo(), body->getUserData(), convexShapes, outputMesh );
	}


	// Sort convex shapes to identify those which share the same transform
	if ( !convexShapes.isEmpty() )
	{
		hkSort( &convexShapes[ 0 ], convexShapes.getSize() );

		// Batch add shapes that share the same transform
		for ( int rangeBegin = 0, rangeEnd = 1; rangeEnd <= convexShapes.getSize(); ++rangeEnd )
		{
			if ( rangeEnd >= convexShapes.getSize() || _hasDifferentTransform( convexShapes[ rangeBegin ], convexShapes[ rangeEnd ] ) )
			{
				int rangeSize = rangeEnd - rangeBegin;
				hkLocalArray< const hkpConvexShape* > childShapes( rangeSize );
				hkLocalArray< hkpStorageExtendedMeshShape::Material > childMaterials( rangeSize );
				hkLocalArray< hkUint16 > childMaterialIndices( rangeSize );

				for ( int i = 0; i < rangeSize; ++i )
				{
					// Collect shapes
					childShapes.pushBack( convexShapes[ rangeBegin + i ].m_shape );

					// Add unique materials 
					hkpStorageExtendedMeshShape::Material meshMaterial;
					meshMaterial.m_filterInfo = convexShapes[ rangeBegin + i ].m_filterInfo;
					meshMaterial.m_friction.setReal<true>(convexShapes[ rangeBegin + i ].m_material.getFriction());
					meshMaterial.m_restitution.setReal<true>(convexShapes[ rangeBegin + i ].m_material.getRestitution());
					meshMaterial.m_userData = convexShapes[ rangeBegin + i ].m_bodyUserData;

					hkUint16 materialIndex = _findMaterialIndex( childMaterials, meshMaterial );
					childMaterialIndices.pushBack( materialIndex );

					if ( materialIndex == childMaterials.getSize() )
					{
						childMaterials.pushBack( meshMaterial );
					}
				}

				// Add batch
				hkpExtendedMeshShape::ShapesSubpart subpart( &childShapes[ 0 ], childShapes.getSize(), convexShapes[ rangeBegin ].m_transform );
				subpart.setNumMaterials(hkUint16( childMaterials.getSize() ));
				subpart.m_materialBase = &childMaterials[ 0 ];
				subpart.m_materialStriding = sizeof( hkpStorageExtendedMeshShape::Material );
				subpart.m_materialIndexBase = &childMaterialIndices[ 0 ];
				subpart.m_materialIndexStriding = sizeof( hkUint16 );
				subpart.setMaterialIndexStridingType(hkpExtendedMeshShape::MATERIAL_INDICES_INT16);

				outputMesh->addShapesSubpart( subpart );


				// Update range
				rangeBegin = rangeEnd;	
			}
		}

	}
	
	return HK_SUCCESS;	
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
