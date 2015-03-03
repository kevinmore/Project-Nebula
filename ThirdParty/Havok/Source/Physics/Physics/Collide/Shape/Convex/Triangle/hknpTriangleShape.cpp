/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Collide/Shape/Convex/Triangle/hknpTriangleShape.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>
#include <Physics/Physics/Collide/Shape/Convex/hknpConvexShapeUtil.h>

HK_COMPILE_TIME_ASSERT( sizeof(hknpTriangleShape) == sizeof(hknpConvexPolytopeShape) );
HK_COMPILE_TIME_ASSERT( hknpInplaceTriangleShape::BUFFER_SIZE % 16 == 0 );

#if defined(HK_PLATFORM_SPU)
hknpTriangleShape* hknpTriangleShape::s_referenceTriangleShape = HK_NULL;
#endif

#if !defined(HK_PLATFORM_SPU)

hknpTriangleShape::hknpTriangleShape( hkFinishLoadedObjectFlag flag )
	:	hknpConvexPolytopeShape(flag)
{}

hknpInplaceTriangleShape::hknpInplaceTriangleShape( bool dummy )
{
	// Make sure the buffer is the right size
	HK_ON_DEBUG( const int sizeOfBaseClass = HK_NEXT_MULTIPLE_OF( HK_REAL_ALIGNMENT, sizeof(hknpTriangleShape) ) );
	HK_ON_DEBUG( const int memSize = hknpConvexPolytopeShape::calcConvexPolytopeShapeSize( 4, 4, 8, sizeOfBaseClass ) );
	HK_ASSERT( 0x76c86a23, memSize <= BUFFER_SIZE );
	HK_WARN_ONCE_ON_DEBUG_IF( memSize != BUFFER_SIZE, 0x76c86a24, "hknpInplaceTriangleShape buffer is bigger than necessary" );

	// Allocate in place
	hknpTriangleShape* triangleShape = new (&m_buffer[0]) hknpTriangleShape( HKNP_SHAPE_DEFAULT_CONVEX_RADIUS );
	triangleShape->m_memSizeAndFlags = (hkUint16)BUFFER_SIZE;

	//
	// Initialize topography
	//

	hknpConvexPolytopeShape::Face* faces = triangleShape->getFaces();
	for( int i = 0; i < 4; i++ )
	{
		faces[i].m_firstIndex = 0;
		faces[i].m_numIndices = 4;
		faces[i].m_minHalfAngle = 127;
	}
	faces[1].m_firstIndex = 4;

	hknpConvexPolytopeShape::VertexIndex* indices = triangleShape->getIndices();
	indices[0] = 0;
	indices[1] = 1;
	indices[2] = 2;
	indices[3] = 3;

	indices[4] = 2;
	indices[5] = 1;
	indices[6] = 0;
	indices[7] = 3;
}

int hknpTriangleShape::calcSize() const
{
	return hknpInplaceTriangleShape::BUFFER_SIZE;
}

void hknpTriangleShape::buildMassProperties(
	const MassConfig& massConfig, hkDiagonalizedMassProperties& massPropertiesOut ) const
{
	hkResult result;
	if( isQuad() )
	{
		// Assume planar
		result = hknpConvexShapeUtil::buildQuadMassProperties( massConfig, &getVertex(0), m_convexRadius, massPropertiesOut );
	}
	else
	{
		result = hknpConvexShapeUtil::buildTriangleMassProperties( massConfig, &getVertex(0), m_convexRadius, massPropertiesOut );
	}

	if( result == HK_FAILURE )
	{
		// Fall back to AABB approximation.
		hknpShape::buildMassProperties( massConfig, massPropertiesOut );
	}
}

hknpTriangleShape* HK_CALL hknpTriangleShape::createEmptyTriangleShape( hkReal radius )
{
	// Copy the reference triangle into newly allocated memory.
	hknpTriangleShape* triangle = (hknpTriangleShape*)hkMemoryRouter::getInstance().heap().blockAlloc( hknpInplaceTriangleShape::BUFFER_SIZE );
	HK_MEMORY_TRACKER_ON_NEW_REFOBJECT( hknpInplaceTriangleShape::BUFFER_SIZE, triangle );
	hkString::memCpy16NonEmpty( triangle, getReferenceTriangleShape(), hknpInplaceTriangleShape::BUFFER_SIZE >> 4 );
	triangle->m_convexRadius = radius;
	return triangle;
}

#endif


void hknpTriangleShape::setVertices(
	hkVector4Parameter a,
	hkVector4Parameter b,
	hkVector4Parameter c )
{
	_setVertices( a, b, c );
}

void hknpTriangleShape::setVertices(
	hkVector4Parameter a,
	hkVector4Parameter b,
	hkVector4Parameter c,
	hkVector4Parameter d )
{
	_setVertices( a, b, c, d );
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
