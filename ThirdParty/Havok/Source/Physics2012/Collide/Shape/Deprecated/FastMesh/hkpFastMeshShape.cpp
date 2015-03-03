/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Physics2012/Collide/Shape/Deprecated/FastMesh/hkpFastMeshShape.h>
#include <Physics2012/Collide/Shape/Convex/Triangle/hkpTriangleShape.h>



hkpFastMeshShape::hkpFastMeshShape( hkReal radius, int numBitsForSubpartIndex )
: hkpMeshShape(radius, numBitsForSubpartIndex)
{
	HK_WARN(0x32285b9a,"Use of hkpFastMeshShape is deprecated. Please use hkpExtendedMeshShape or some other shape.");
}

#if !defined(HK_PLATFORM_SPU)

hkpFastMeshShape::hkpFastMeshShape( hkFinishLoadedObjectFlag flag ) : hkpMeshShape(flag)
{
	if( flag.m_finishing )
	{
		setType(HKCD_SHAPE_TYPE_FROM_CLASS(hkpFastMeshShape));
	}
}

#endif


const hkpShape* hkpFastMeshShape::getChildShape( hkpShapeKey key, hkpShapeBuffer& buffer ) const
{
	HK_WARN_ONCE(0x32285b9a,"Use of hkpFastMeshShape is deprecated. Please use hkpExtendedMeshShape or some other shape.");

	HK_ASSERT2(0x6810be4e,  m_subparts.getSize() == 1, "hkpFastMeshShape only works with one subpart" );

	// Extract triangle index and sub-part index
	const hkUint32 triangleIndex = key;

	// Grab a handle to the sub-part
	const Subpart& part = m_subparts[ 0 ];

	HK_ASSERT2(0x27c96ec3,  part.m_stridingType == INDICES_INT16, "hkpFastMeshShape only works with INDICES_INT16");
	HK_ASSERT2(0x3e4da054,  (part.m_vertexStriding & 15) == 0, "hkpFastMeshShape only works with vertex striding of multiple of 16");
	HK_ASSERT2(0x21ca33fc,  (hkUlong(part.m_vertexBase) & 15) == 0, "hkpFastMeshShape only works with aligned vertices");
	
	// The three triangle vertices as float pointers to be filled in
	const hkUint16* triangle = hkAddByteOffsetConst<hkUint16>( (const hkUint16*)part.m_indexBase, part.m_indexStriding * triangleIndex );
	
	HK_ASSERT2(0x978f756, part.m_flipAlternateTriangles == 0 || part.m_flipAlternateTriangles == 1, "m_flipAlternateTriangles must equal 0 or 1");
	
	// m_flipAlternateTriangles is 1 if flip is enabled, 0 is disabled
	int triangleWindingFlip = triangleIndex & part.m_flipAlternateTriangles;

	// Grab the vertices
	const hkVector4* base = reinterpret_cast<const hkVector4*>( part.m_vertexBase );
	const hkVector4* vf0 = hkAddByteOffsetConst<hkVector4>( base, part.m_vertexStriding * triangle[0] );
	const hkVector4* vf1 = hkAddByteOffsetConst<hkVector4>( base, part.m_vertexStriding * triangle[1 + triangleWindingFlip] );
	const hkVector4* vf2 = hkAddByteOffsetConst<hkVector4>( base, part.m_vertexStriding * triangle[1 + (1 ^ triangleWindingFlip) ] );
	

	hkUint16 weldingInfo = (m_weldingInfo.getSize() == 0) ? 0 : m_weldingInfo[ part.m_triangleOffset + triangleIndex ];

	HK_ASSERT(0x51b0bd8f,  sizeof( hkpTriangleShape ) <= HK_SHAPE_BUFFER_SIZE );
	hkpTriangleShape *triangleShape = new( buffer ) hkpTriangleShape( m_radius, weldingInfo, m_weldingType );

	triangleShape->getVertex(0).setMul( *vf0, m_scaling );
	triangleShape->getVertex(1).setMul( *vf1, m_scaling );
	triangleShape->getVertex(2).setMul( *vf2, m_scaling );

	return triangleShape;
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
