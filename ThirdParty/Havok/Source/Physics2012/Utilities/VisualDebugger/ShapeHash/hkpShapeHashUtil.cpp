/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Physics2012/Utilities/VisualDebugger/ShapeHash/hkpShapeHashUtil.h>

#include <Common/Base/System/Io/Writer/Crc/hkCrcStreamWriter.h>
#include <Common/Base/Container/String/hkStringBuf.h>

#include <Physics2012/Collide/Shape/Convex/Sphere/hkpSphereShape.h>
#include <Physics2012/Collide/Shape/Convex/Cylinder/hkpCylinderShape.h>
#include <Physics2012/Collide/Shape/Convex/Triangle/hkpTriangleShape.h>
#include <Physics2012/Collide/Shape/Convex/Box/hkpBoxShape.h>
#include <Physics2012/Collide/Shape/Convex/Capsule/hkpCapsuleShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexVertices/hkpConvexVerticesShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexTranslate/hkpConvexTranslateShape.h>
#include <Physics2012/Collide/Shape/Convex/ConvexTransform/hkpConvexTransformShape.h>
#include <Physics2012/Collide/Shape/Misc/Transform/hkpTransformShape.h>
#include <Physics2012/Collide/Shape/Misc/Bv/hkpBvShape.h>
#include <Physics2012/Collide/Shape/Misc/MultiRay/hkpMultiRayShape.h>
#include <Physics2012/Collide/Shape/Misc/PhantomCallback/hkpPhantomCallbackShape.h>
#include <Physics2012/Collide/Shape/HeightField/SampledHeightField/hkpSampledHeightFieldShape.h>
#include <Physics2012/Collide/Shape/HeightField/Plane/hkpPlaneShape.h>
#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/hkpMoppBvTreeShape.h>

#include <Common/Base/Reflection/hkClass.h>

void hkpShapeHashUtil::writeShape( const hkpShape* shape, Mode mode )
{
	hkpShapeType type = shape->getType();

	// The type value contributes to the hash.
	writeUint8((hkUint8)type);
	writeUint8((hkUint8)((hkcdShapeDispatchType::ShapeDispatchTypeEnum)shape->m_dispatchType));
	writeUint8(shape->m_bitsPerKey);
	writeUint8((hkUint8)((hkcdShapeInfoCodecType::ShapeInfoCodecTypeEnum)shape->m_shapeInfoCodecType));

	switch ( type )
	{
		case hkcdShapeType::SPHERE:
		{
			const hkpSphereShape *const sphereShape = static_cast<const hkpSphereShape*>( shape );

			writeSphereShape( sphereShape );

			break;
		}
		case hkcdShapeType::CYLINDER:
		{
			const hkpCylinderShape *const cylinderShape = static_cast<const hkpCylinderShape*>( shape );

			writeCylinderShape( cylinderShape, mode );

			break;
		}
		case hkcdShapeType::TRIANGLE:
		{
			const hkpTriangleShape *const triangleShape = static_cast<const hkpTriangleShape*>( shape );

			writeTriangleShape( triangleShape, mode );

			break;
		}
		case hkcdShapeType::BOX:
		{
			const hkpBoxShape *const boxShape = static_cast<const hkpBoxShape*>( shape );

			writeBoxShape( boxShape, mode );

			break;
		}
		case hkcdShapeType::CAPSULE:
		{
			const hkpCapsuleShape *const capsuleShape = static_cast<const hkpCapsuleShape*>( shape );

			writeCapsuleShape( capsuleShape );

			break;
		}
		case hkcdShapeType::CONVEX_VERTICES:
		{
			const hkpConvexVerticesShape *const convexVerticesShape = static_cast<const hkpConvexVerticesShape*>( shape );

			writeConvexVerticesShape( convexVerticesShape, mode );

			break;
		}
		case hkcdShapeType::CONVEX_TRANSLATE:
		{
			const hkpConvexTranslateShape *const convexTranslateShape = static_cast<const hkpConvexTranslateShape*>( shape );

			writeConvexTranslateShape( convexTranslateShape, mode );

			break;
		}
		case hkcdShapeType::CONVEX_TRANSFORM:
		{
			const hkpConvexTransformShape *const convexTransformShape = static_cast<const hkpConvexTransformShape*>( shape );

			writeConvexTransformShape( convexTransformShape, mode );

			break;
		}
		case hkcdShapeType::TRANSFORM:
		{
			const hkpTransformShape *const transformShape = static_cast<const hkpTransformShape*>( shape );

			writeTransformShape( transformShape, mode );

			break;
		}
		case hkcdShapeType::SAMPLED_HEIGHT_FIELD:
		{
			const hkpSampledHeightFieldShape *const heightField = static_cast<const hkpSampledHeightFieldShape*>( shape );

			writeSampledHeightFieldShape( heightField );

			break;
		}
		case hkcdShapeType::MULTI_RAY:
		{
			const hkpMultiRayShape *const multiRay = static_cast<const hkpMultiRayShape*>( shape );

			writeMultiRayShape( multiRay );

			break;
		}
		case hkcdShapeType::BV:
		{
			const hkpBvShape* bvShape = static_cast<const hkpBvShape*>(shape);
			
			writeBvShape( bvShape, mode );

			break;
		}
		case hkcdShapeType::MOPP:
		{
			const hkpMoppBvTreeShape *const mopp = static_cast<const hkpMoppBvTreeShape*>( shape );

			writeMoppBvTreeShape( mopp, mode );

			break;
		}

		case hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_COLLECTION:
		case hkcdShapeType::COLLECTION:
		case hkcdShapeType::LIST:
		case hkcdShapeType::EXTENDED_MESH:
		case hkcdShapeType::COMPRESSED_MESH:
		case hkcdShapeType::MULTI_SPHERE:
		case hkcdShapeType::CONVEX_LIST: // Deprecated.
		case hkcdShapeType::TRIANGLE_COLLECTION:
		{
			writeShapeContainer( shape->getContainer(), mode );

			break;
		}
		case hkcdShapeType::PHANTOM_CALLBACK:
		{
			const hkpPhantomCallbackShape *const phantomShape = static_cast<const hkpPhantomCallbackShape*>( shape );

			writePhantomCallbackShape( phantomShape );

			break;
		}
		case hkcdShapeType::PLANE:
		{
			const hkpPlaneShape *const plane = static_cast<const hkpPlaneShape*>( shape );

			writePlaneShape( plane );

			break;
		}

		case hkcdShapeType::TRI_SAMPLED_HEIGHT_FIELD_BV_TREE:
		case hkcdShapeType::BV_TREE:
		case hkcdShapeType::CONVEX:
		case hkcdShapeType::SPHERE_REP:
		case hkcdShapeType::CONVEX_PIECE:
		case hkcdShapeType::USER0:
		case hkcdShapeType::USER1:
		case hkcdShapeType::USER2:
		case hkcdShapeType::HEIGHT_FIELD:
		default:
		{
			const int numUserFunctions = hkpUserShapeHashUtil::getInstance().m_userShapeHashFunctions.getSize();

			m_hasHash = false;
			for (int i = 0; i < numUserFunctions; ++i )
			{
				if ( hkpUserShapeHashUtil::getInstance().m_userShapeHashFunctions[i].m_type == type )
				{
					m_hasHash = hkpUserShapeHashUtil::getInstance().m_userShapeHashFunctions[i].m_f( shape, *this );
					break;
				}
			}
		}
	}
}

void hkpShapeHashUtil::writeTransform( const hkTransform& transform )
{
	writeVector3( transform.getColumn<0>() );
	writeVector3( transform.getColumn<1>() );
	writeVector3( transform.getColumn<2>() );
	writeVector3( transform.getColumn<3>() );
}

void hkpShapeHashUtil::writeQsTransform( const hkQsTransform& transform )
{
	hkTransform t;
	transform.copyToTransformNoScale( t );
	writeTransform( t );
	writeVector4( transform.m_scale );
}

void hkpShapeHashUtil::writeSphereShape( const hkpSphereShape* shape )
{
	writeReal( shape->getRadius() );
}

void hkpShapeHashUtil::writeCylinderShape( const hkpCylinderShape* shape, Mode mode )
{
	writeVector3( shape->getVertex<0>() );
	writeVector3( shape->getVertex<1>() );
	writeReal( shape->getCylinderRadius() );
	if ( mode == USE_CONVEX_RADIUS )
	{
		writeReal( shape->getRadius() );
	}
}

void hkpShapeHashUtil::writeTriangleShape( const hkpTriangleShape* shape, Mode mode )
{
	writeVector3( shape->getVertex<0>() );
	writeVector3( shape->getVertex<1>() );
	writeVector3( shape->getVertex<2>() );
	if ( mode == USE_CONVEX_RADIUS )
	{
		writeReal( shape->getRadius() );
	}
}

void hkpShapeHashUtil::writeBoxShape( const hkpBoxShape* shape, Mode mode )
{
	writeVector3( shape->getHalfExtents() );
	if ( mode == USE_CONVEX_RADIUS )
	{
		writeReal( shape->getRadius() );
	}
}

void hkpShapeHashUtil::writeCapsuleShape( const hkpCapsuleShape* shape )
{
	writeVector3( shape->getVertex<0>() );
	writeVector3( shape->getVertex<1>() );
	writeReal( shape->getRadius() );
}

void hkpShapeHashUtil::writeConvexVerticesShape( const hkpConvexVerticesShape *shape, Mode mode )
{
	hkArray<hkVector4> vertices;
	{
		shape->getOriginalVertices( vertices );
	}
	const int numVertices = vertices.getSize();
	for ( int i = 0; i < numVertices; ++i )
	{
		writeVector3( vertices[i] );
	}
	if ( mode == USE_CONVEX_RADIUS )
	{
		writeReal( shape->getRadius() );
	}
}

void hkpShapeHashUtil::writeConvexTranslateShape( const hkpConvexTranslateShape* shape, Mode mode )
{
	writeVector3( shape->getTranslation() );
	writeShape( shape->getChildShape(), mode );
}

void hkpShapeHashUtil::writeConvexTransformShape( const hkpConvexTransformShape* shape, Mode mode )
{
	writeQsTransform( shape->getQsTransform() );
	writeVector4( shape->getExtraScale() );
	writeShape( shape->getChildShape(), mode );
}

void hkpShapeHashUtil::writeTransformShape( const hkpTransformShape* shape, Mode mode )
{
	writeTransform( shape->getTransform() );
	writeShape( shape->getChildShape(), mode );
}

void hkpShapeHashUtil::writeSampledHeightFieldShape( const hkpSampledHeightFieldShape* shape )
{
	const int xRez = shape->m_xRes;
	const int zRez = shape->m_zRes;

	writeUint32( xRez );
	writeUint32( zRez );
	writeVector3( shape->m_intToFloatScale );

	for ( int x = 0; x < xRez; ++x )
	{
		for ( int z = 0; z < zRez; ++z )
		{
			writeReal( shape->getHeightAt( x, z ) );
		}
	}
}

void hkpShapeHashUtil::writeShapeContainer( const hkpShapeContainer* container, Mode mode )
{
	HK_ALIGN16( hkpShapeBuffer buffer );
	for ( hkpShapeKey key = container->getFirstKey(); key != HK_INVALID_SHAPE_KEY; key = container->getNextKey( key ) )
	{
		writeShape( container->getChildShape( key, buffer ), mode );
	}
}

void hkpShapeHashUtil::writeMultiRayShape( const hkpMultiRayShape* shape )
{
	const hkArray<hkpMultiRayShape::Ray>& rays = shape->getRays();
	const int numRays = rays.getSize();
	for ( int i = 0; i < numRays; ++i )
	{
		writeVector3( rays[i].m_start );
		writeVector3( rays[i].m_end );
	}
}

void hkpShapeHashUtil::writeBvShape( const hkpBvShape* shape, Mode mode )
{
	writeShape( shape->getBoundingVolumeShape(), mode );
	writeShape( shape->getChildShape(), mode );
}

void hkpShapeHashUtil::writePhantomCallbackShape( const hkpPhantomCallbackShape* shape )
{
	hkAabb aabb;
	{
		hkTransform transform;
		{
			transform.setIdentity();
		}
		shape->getAabb( transform, 0.0f, aabb );

	}
	writeVector3( aabb.m_max );
	writeVector3( aabb.m_min );
}

void hkpShapeHashUtil::writePlaneShape( const hkpPlaneShape* shape )
{
	const hkVector4& plane = shape->getPlane();
	writeVector3( plane );
	writeReal( plane( 3 ) );
	writeVector3( shape->getAabbCenter() );
	writeVector3( shape->getAabbHalfExtents() );
}

void hkpShapeHashUtil::writeMoppBvTreeShape( const hkpMoppBvTreeShape* shape, Mode mode )
{
	return writeShape( shape->getChild(), mode );
}

HK_SINGLETON_IMPLEMENTATION(hkpUserShapeHashUtil);

void hkpUserShapeHashUtil::registerUserShapeHashFunction( WriteToHashFunction f, hkpShapeType type )
{
	const int numHashFunctions = m_userShapeHashFunctions.getSize();
	for (int i = 0; i < numHashFunctions; ++i )
	{
		if ( m_userShapeHashFunctions[i].m_type == type )
		{
			HK_WARN(0x7bbfa3c4, "You have registered two shape hash functions for type" << type << ". Do you have two different shapes with this type?");
			return;
		}
	}
	UserShapeHashFunctions b;
	b.m_f = f;
	b.m_type = type;

	m_userShapeHashFunctions.pushBack(b);
}

#ifdef HK_DEBUG
void hkpShapeHashUtil::assertShapesUpToDate()
{
// 	hkStringBuf strb;
// 	strb.printf("hkpSphereShape: 0x%08x",				hkpSphereShapeClass.getSignature());				HK_REPORT(strb);
// 	strb.printf("hkpCylinderShape: 0x%08x",				hkpCylinderShapeClass.getSignature());				HK_REPORT(strb);
// 	strb.printf("hkpTriangleShape: 0x%08x",				hkpTriangleShapeClass.getSignature());				HK_REPORT(strb);
// 	strb.printf("hkpBoxShape: 0x%08x",					hkpBoxShapeClass.getSignature());					HK_REPORT(strb);
// 	strb.printf("hkpCapsuleShape: 0x%08x",				hkpCapsuleShapeClass.getSignature());				HK_REPORT(strb);
// 	strb.printf("hkpConvexVerticesShape: 0x%08x",		hkpConvexVerticesShapeClass.getSignature());		HK_REPORT(strb);
// 	strb.printf("hkpConvexTranslateShape: 0x%08x",		hkpConvexTranslateShapeClass.getSignature());		HK_REPORT(strb);
// 	strb.printf("hkpConvexTransformShape: 0x%08x",		hkpConvexTransformShapeClass.getSignature());		HK_REPORT(strb);
// 	strb.printf("hkpTransformShape: 0x%08x",			hkpTransformShapeClass.getSignature());				HK_REPORT(strb);
// 	strb.printf("hkpSampledHeightFieldShape: 0x%08x",	hkpSampledHeightFieldShapeClass.getSignature());	HK_REPORT(strb);
// 	strb.printf("hkpMultiRayShape: 0x%08x",				hkpMultiRayShapeClass.getSignature());				HK_REPORT(strb);
// 	strb.printf("hkpPhantomCallbackShape: 0x%08x",		hkpPhantomCallbackShapeClass.getSignature());		HK_REPORT(strb);
// 	strb.printf("hkpPlaneShape: 0x%08x",				hkpPlaneShapeClass.getSignature());					HK_REPORT(strb);

	// If you trigger one of these asserts, you should check that the corresponding function above
	// is still appropriate for calculating the shape's hash. After you've confirmed that the
	// function is correct, you'll need to update the assert with the new signature.
	HK_ASSERT2( 0x39a1e781, hkpSphereShapeClass.getSignature() == 0x89b169a3, "Signature for hkpSphereShape has changed. Please update hkpShapeHashUtil." );
	HK_ASSERT2( 0x39a1e781, hkpCylinderShapeClass.getSignature() == 0x2f66d4c0, "Signature for hkpCylinderShape has changed. Please update hkpShapeHashUtil." );
	HK_ASSERT2( 0x39a1e781, hkpTriangleShapeClass.getSignature() == 0x3b005d5f, "Signature for hkpTriangleShape has changed. Please update hkpShapeHashUtil." );
	HK_ASSERT2( 0x39a1e781, hkpBoxShapeClass.getSignature() == 0x0c1112ea, "Signature for hkpBoxShape has changed. Please update hkpShapeHashUtil." );
	HK_ASSERT2( 0x39a1e781, hkpCapsuleShapeClass.getSignature() == 0xb58289ff, "Signature for hkpCapsuleShape has changed. Please update hkpShapeHashUtil." );
	HK_ASSERT2( 0x39a1e781, hkpConvexVerticesShapeClass.getSignature() == 0x5bb14288, "Signature for hkpConvexVerticesShape has changed. Please update hkpShapeHashUtil." );
	HK_ASSERT2( 0x39a1e781, hkpConvexTranslateShapeClass.getSignature() == 0x3b729b87, "Signature for hkpConvexTranslateShape has changed. Please update hkpShapeHashUtil." );
	HK_ASSERT2( 0x39a1e781, hkpConvexTransformShapeClass.getSignature() == 0x8ba4efab, "Signature for hkpConvexTransformShape has changed. Please update hkpShapeHashUtil." );
	HK_ASSERT2( 0x39a1e781, hkpTransformShapeClass.getSignature() == 0x5ccbb8ee, "Signature for hkpTransformShape has changed. Please update hkpShapeHashUtil." );
	HK_ASSERT2( 0x39a1e781, hkpSampledHeightFieldShapeClass.getSignature() == 0x18882f3e, "Signature for hkpSampledHeightFieldShape has changed. Please update hkpShapeHashUtil." );
	HK_ASSERT2( 0x39a1e781, hkpMultiRayShapeClass.getSignature() == 0x35af0ac5, "Signature for hkpMultiRayShape has changed. Please update hkpShapeHashUtil." );
	HK_ASSERT2( 0x39a1e781, hkpPhantomCallbackShapeClass.getSignature() == 0xa09e5162, "Signature for hkpPhantomCallbackShape has changed. Please update hkpShapeHashUtil." );
	HK_ASSERT2( 0x39a1e781, hkpPlaneShapeClass.getSignature() == 0x80b812cb, "Signature for hkpPlaneShape has changed. Please update hkpShapeHashUtil." );
}
#endif

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
