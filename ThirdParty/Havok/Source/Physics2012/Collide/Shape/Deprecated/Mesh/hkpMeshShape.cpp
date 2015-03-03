/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Common/Base/Types/Geometry/hkGeometry.h>

#include <Physics2012/Collide/Shape/Deprecated/Mesh/hkpMeshShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/Mesh/hkpMeshMaterial.h>
#include <Physics2012/Collide/Shape/Convex/Triangle/hkpTriangleShape.h>
#include <Physics2012/Collide/Shape/Compound/Collection/SimpleMesh/hkpSimpleMeshShape.h>
#include <Physics2012/Collide/Shape/Compound/Tree/Mopp/hkpMoppBvTreeShape.h>
#include <Physics2012/Collide/Util/hkpTriangleUtil.h>

#include <Physics2012/Collide/Util/Welding/hkpMeshWeldingUtility.h>

extern hkReal hkDefaultTriangleDegeneracyTolerance;

hkpMeshShape::hkpMeshShape( hkReal radius,int numBitsForSubpartIndex )
:	hkpShapeCollection( HKCD_SHAPE_TYPE_FROM_CLASS(hkpMeshShape), COLLECTION_MESH_SHAPE)
{
	HK_WARN(0x1791c2c5,"Use of hkpMeshShape is deprecated. Please use hkpExtendedMeshShape.");

	m_scaling.setXYZ( 1.0f );
	m_radius = radius;
	m_weldingType = hkpWeldingUtility::WELDING_TYPE_NONE;

	HK_ASSERT2(0x16aa7e0a, numBitsForSubpartIndex > 0 && numBitsForSubpartIndex < 32,\
		"cinfo.m_numBitsForSubpartIndex must be greater than zero and less than 32."\
		"See comment in construction info for details on how this parameter is used.");

	m_numBitsForSubpartIndex = numBitsForSubpartIndex;
}

hkpMeshShape::hkpMeshShape( hkFinishLoadedObjectFlag flag )
:	hkpShapeCollection(flag)
,	m_subparts(flag)
,	m_weldingInfo(flag)
{
	if( flag.m_finishing )
	{
		// 3.0 compatibility. m_materialIndexStridingType is loaded as binary zero
		// For 3.0 files material indices are always int8
		for( int i = 0; i < m_subparts.getSize(); ++i )
		{
			if( m_subparts[i].m_materialIndexStridingType == MATERIAL_INDICES_INVALID )
			{
				m_subparts[i].m_materialIndexStridingType = MATERIAL_INDICES_INT8;
			}
		}
		setType(HKCD_SHAPE_TYPE_FROM_CLASS(hkpMeshShape));
		m_collectionType = COLLECTION_MESH_SHAPE;
	}
}

void hkpMeshShape::setWeldingInfo( hkpShapeKey key, hkInt16 weldingInfo)
{
	// Extract triangle index and sub-part index
	const hkUint32 triangleIndex = key & ( ~0U >> m_numBitsForSubpartIndex );
	const hkUint32 subPartIndex = key >> ( 32 - m_numBitsForSubpartIndex );

	// Grab a handle to the sub-part
	const Subpart& part = m_subparts[ subPartIndex ];

	HK_ASSERT2(0xc2749a04, part.m_indexBase, "Invalid mesh shape. First subpart has no elements/triangles.");

	const int index = part.m_triangleOffset + triangleIndex;

	HK_ASSERT3(0xa94b20c8, index >= 0 && index < m_weldingInfo.getSize(), "hkpMeshShape does not have a triangle at index" << index);
	m_weldingInfo[index] = weldingInfo;
}

void hkpMeshShape::initWeldingInfo( hkpWeldingUtility::WeldingType weldingType )
{
	HK_ASSERT2(0x897654dd, m_subparts.getSize() != 0, "You must add subparts before calling computeWeldingInfo.");

	m_weldingType = weldingType;

	if (weldingType != hkpWeldingUtility::WELDING_TYPE_NONE )
	{
		HK_ON_DEBUG( if ( m_weldingInfo.getSize() != 0) HK_WARN(0x798d7651, "You are calling computeWeldingInfo more than once on a mesh."); )

		int totalSize = 0;
		for (int i = 0; i < m_subparts.getSize(); ++i)
		{
			m_subparts[i].m_triangleOffset = totalSize;
			totalSize += m_subparts[i].m_numTriangles;
		}
		m_weldingInfo.reserveExactly(totalSize);
		m_weldingInfo.setSize(totalSize, 0);
	}
	else
	{
		m_weldingInfo.clearAndDeallocate();
	}
}


hkpShapeKey hkpMeshShape::getFirstKey() const
{
	if ( m_subparts.getSize() == 0 )
	{
		return HK_INVALID_SHAPE_KEY;
	}

	hkpShapeBuffer buffer;
	const hkpShape* shape = getChildShape(0, buffer );
	const hkpTriangleShape* tri = static_cast<const hkpTriangleShape*>(shape);
	if ( hkpTriangleUtil::isDegenerate( tri->getVertex<0>(), tri->getVertex<1>(), tri->getVertex<2>(), hkDefaultTriangleDegeneracyTolerance ) == false )
	{
		return 0;
	}
	return getNextKey( 0 );
}

// Get the next child shape key.
hkpShapeKey hkpMeshShape::getNextKey( hkpShapeKey initialKey ) const
{
	hkpShapeBuffer buffer;

	unsigned subPart = initialKey  >> ( 32 - m_numBitsForSubpartIndex );
	int triIndex = initialKey  & ( ~0U >> m_numBitsForSubpartIndex );

	while (1)
	{
		if ( ++triIndex >= m_subparts[subPart].m_numTriangles )
		{
			if ( ++subPart >= unsigned(m_subparts.getSize()) )
			{
				return HK_INVALID_SHAPE_KEY;
			}
			triIndex = 0;
		}
		hkpShapeKey key = ( subPart << ( 32 - m_numBitsForSubpartIndex )) | triIndex;

		//
		//	check for valid triangle
		//

		const hkpShape* shape = getChildShape(key, buffer );

		const hkpTriangleShape* tri = static_cast<const hkpTriangleShape*>(shape);
		if ( hkpTriangleUtil::isDegenerate( tri->getVertex<0>(), tri->getVertex<1>(), tri->getVertex<2>(), hkDefaultTriangleDegeneracyTolerance ) == false )
		{
			return key;
		}
	}
}


const hkpShape* hkpMeshShape::getChildShape( hkpShapeKey key, hkpShapeBuffer& buffer ) const
{
	// Extract triangle index and sub-part index
	const hkUint32 triangleIndex = key & ( ~0U >> m_numBitsForSubpartIndex );
	const hkUint32 subPartIndex = key >> ( 32 - m_numBitsForSubpartIndex );

	//int triIndex = initialKey  & ( ~0L >> m_numBitsForSubpartIndex );
	//unsigned subPart = initialKey  >> ( 32 - m_numBitsForSubpartIndex );

	// Grab a handle to the sub-part
	const Subpart& part = m_subparts[ subPartIndex ];

	HK_ASSERT2(0xad45bb32, part.m_indexBase, "Invalid mesh shape. First subpart has no elements/triangles.");

	// The three triangle vertices as float pointers to be filled in
	const hkReal* vf0 = HK_NULL;
	const hkReal* vf1 = HK_NULL;
	const hkReal* vf2 = HK_NULL;

	HK_ASSERT2(0x978f756, part.m_flipAlternateTriangles == 0 || part.m_flipAlternateTriangles == 1, "m_flipAlternateTriangles must equal 0 or 1");

	// m_flipAlternateTriangles is 1 if flip is enabled, 0 is disabled
	int triangleWindingFlip = triangleIndex & part.m_flipAlternateTriangles;

	// Extract the triangle indicies and vertices
	if ( part.m_stridingType == INDICES_INT16 )
	{
		const hkUint16* triangle = hkAddByteOffsetConst<hkUint16>( (const hkUint16*)part.m_indexBase, part.m_indexStriding * triangleIndex );

		// Grab the vertices
		vf0 = hkAddByteOffsetConst<hkReal>( part.m_vertexBase, part.m_vertexStriding * triangle[0] );
		vf1 = hkAddByteOffsetConst<hkReal>( part.m_vertexBase, part.m_vertexStriding * triangle[1 + triangleWindingFlip] );
		vf2 = hkAddByteOffsetConst<hkReal>( part.m_vertexBase, part.m_vertexStriding * triangle[1 + (1 ^ triangleWindingFlip) ] );
	}
	else
	{
		const hkUint32* triangle = hkAddByteOffsetConst<hkUint32>( (const hkUint32*)part.m_indexBase, part.m_indexStriding * triangleIndex);

		// Grab the vertices
		vf0 = hkAddByteOffsetConst<hkReal>( part.m_vertexBase, part.m_vertexStriding * triangle[0] );
		vf1 = hkAddByteOffsetConst<hkReal>( part.m_vertexBase, part.m_vertexStriding * triangle[1 + triangleWindingFlip] );
		vf2 = hkAddByteOffsetConst<hkReal>( part.m_vertexBase, part.m_vertexStriding * triangle[1 + (1 ^ triangleWindingFlip)] );
	}

	// generate hkVector4s out of our vertices
	hkVector4 vertex0;
	hkVector4 vertex1;
	hkVector4 vertex2;

	vertex0.load<3,HK_IO_NATIVE_ALIGNED>(vf0);
	vertex1.load<3,HK_IO_NATIVE_ALIGNED>(vf1);
	vertex2.load<3,HK_IO_NATIVE_ALIGNED>(vf2);

	vertex0.zeroComponent<3>();
	vertex1.zeroComponent<3>();
	vertex2.zeroComponent<3>();

	vertex0.mul(m_scaling);
	vertex1.mul(m_scaling);
	vertex2.mul(m_scaling);

	hkUint16 weldingInfo = (m_weldingInfo.getSize() == 0) ? 0 : m_weldingInfo[ part.m_triangleOffset + triangleIndex ];

	HK_ASSERT(0x73f97fa7,  sizeof( hkpTriangleShape ) <= HK_SHAPE_BUFFER_SIZE );
	hkpTriangleShape *triangleShape = new( buffer ) hkpTriangleShape( m_radius, weldingInfo, m_weldingType );

	triangleShape->setVertex<0>( vertex0 );
	triangleShape->setVertex<1>( vertex1 );
	triangleShape->setVertex<2>( vertex2 );

	return triangleShape;
}

hkUint32 hkpMeshShape::getCollisionFilterInfo( hkpShapeKey key ) const
{
	const hkpMeshMaterial* material = getMeshMaterial(key);

	if (material)
	{
		return material->m_filterInfo;
	}
	else
	{
		return 0;
	}
}



static void HK_CALL meshShape_addToAabb(hkAabb& aabb, const hkTransform& localToWorld, const hkReal* v,const hkVector4& scaling)
{
	hkVector4 vLocal;
	vLocal.load<3,HK_IO_NATIVE_ALIGNED>(v);
	vLocal.zeroComponent<3>();
	vLocal.mul(scaling);

	hkVector4 vWorld; 
	vWorld._setTransformedPos( localToWorld, vLocal );

	aabb.m_min.setMin( aabb.m_min, vWorld );
	aabb.m_max.setMax( aabb.m_max, vWorld );
}



void hkpMeshShape::getAabb(const hkTransform& localToWorld, hkReal tolerance, hkAabb& out) const
{
	HK_WARN_ONCE(0x1791c2c5,"Use of hkpMeshShape is deprecated. Please use hkpExtendedMeshShape.");


	out.m_min = hkVector4::getConstant<HK_QUADREAL_MAX>();
	out.m_max = hkVector4::getConstant<HK_QUADREAL_MINUS_MAX>();


	for (int s = 0; s < m_subparts.getSize(); s++)
	{
		const Subpart& part = m_subparts[s];

		// as getAabb is the first thing to be called upon addition of the shape
		// to a world, we check in debug iof the ptrs are ok
		HK_ASSERT2(0x6541f816, part.m_indexBase, "No indices provided in a subpart of a hkpMeshShape." );
		HK_ASSERT2(0x6541f817, part.m_vertexBase, "No vertices provided in a subpart of a hkpMeshShape." );

		for (int v = 0; v < part.m_numTriangles; v++ )
		{
			const hkReal* vf0;
			const hkReal* vf1;
			const hkReal* vf2;

			if ( part.m_stridingType == INDICES_INT16)
			{
				const hkUint16* tri = hkAddByteOffsetConst<hkUint16>( (const hkUint16*)part.m_indexBase, part.m_indexStriding * v);
				vf0 = hkAddByteOffsetConst<hkReal>(part.m_vertexBase, part.m_vertexStriding * tri[0] );
				vf1 = hkAddByteOffsetConst<hkReal>(part.m_vertexBase, part.m_vertexStriding * tri[1] );
				vf2 = hkAddByteOffsetConst<hkReal>(part.m_vertexBase, part.m_vertexStriding * tri[2] );
			}
			else
			{
				const hkUint32* tri = hkAddByteOffsetConst<hkUint32>( (const hkUint32*)part.m_indexBase, part.m_indexStriding * v);

				vf0 = hkAddByteOffsetConst<hkReal>(part.m_vertexBase, part.m_vertexStriding * tri[0] );
				vf1 = hkAddByteOffsetConst<hkReal>(part.m_vertexBase, part.m_vertexStriding * tri[1] );
				vf2 = hkAddByteOffsetConst<hkReal>(part.m_vertexBase, part.m_vertexStriding * tri[2] );
			}
			meshShape_addToAabb(out, localToWorld, vf0,m_scaling);
			meshShape_addToAabb(out, localToWorld, vf1,m_scaling);
			meshShape_addToAabb(out, localToWorld, vf2,m_scaling);
		}
	}

	hkSimdReal tol4; tol4.setFromFloat( tolerance + m_radius );
	out.m_min.setSub( out.m_min,tol4 );
	out.m_max.setAdd( out.m_max,tol4 );

}

void hkpMeshShape::setScaling( const hkVector4& scaling )
{
	m_scaling = scaling;
}

void hkpMeshShape::assertSubpartValidity( const Subpart& part )
{
	HK_ASSERT2(0x68fb31d4, m_subparts.getSize() < ((1 << m_numBitsForSubpartIndex) - 1 ), "You are adding too many subparts for the mesh shape. "\
		"You can change the number of bits usable for the subpart index by changing the m_numBitsForSubpartIndex in the mesh construction info.");

	HK_ASSERT2(0x6541f716,  part.m_vertexBase, "Subpart vertex base pointer is not set or null.");
	HK_ASSERT2(0x426c5d43,  part.m_vertexStriding >= 4, "Subpart vertex striding is not set or invalid (less than 4 bytes stride).");
	HK_ASSERT2(0x2223ecab,  part.m_numVertices > 0, "Subpart num vertices is not set or negative.");
	HK_ASSERT2(0x5a93ebb6,  part.m_indexBase, "Subpart index base pointer is not set or null.");
	HK_ASSERT2(0x12131a31,  ((part.m_stridingType == INDICES_INT16) || (part.m_stridingType == INDICES_INT32)),
		"Subpart index type is not set or out of range (16 or 32 bit only).");
	HK_ASSERT2(0x492cb07c,  part.m_indexStriding >= 2,
		"Subpart index striding pointer is not set or invalid (less than 2 bytes stride).");
	HK_ASSERT2(0x53c3cd4f,  part.m_numTriangles > 0, "Subpart num triangles is not set or negative.");
	HK_ASSERT2(0xad5aae43,  part.m_materialIndexBase == HK_NULL || part.m_materialIndexStridingType == MATERIAL_INDICES_INT8 || part.m_materialIndexStridingType == MATERIAL_INDICES_INT16, "Subpart materialIndexStridingType is not set or out of range (8 or 16 bit only).");

	HK_ASSERT2(0x7b8c4c78,	part.m_numTriangles-1 < (1<<(32-m_numBitsForSubpartIndex)),
		"There are only 32 bits available to index the sub-part and triangle in a "
		"hkpMeshShape. This subpart has too many triangles, attempts to index a "
		"triangle could overflow the available bits. Try decreasing the number of "
		"bits reserved for the sub-part index.");
}

void hkpMeshShape::addSubpart( const Subpart& part )
{
	HK_ON_DEBUG( assertSubpartValidity(part); )
	HK_ASSERT2(0x09fe8645, m_weldingInfo.getSize() == 0, "You must add all subparts prior to building welding information" );


	Subpart& p = m_subparts.expandOne();
	p = part;

	// disable materials
	if ( p.m_materialIndexBase == HK_NULL)
	{
		p.m_numMaterials = 1;
		p.m_materialBase = reinterpret_cast<const hkpMeshMaterial*>( &hkVector4::getZero() );
		p.m_materialIndexBase = reinterpret_cast<const hkUint8*>( &hkVector4::getZero() );
	}
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
