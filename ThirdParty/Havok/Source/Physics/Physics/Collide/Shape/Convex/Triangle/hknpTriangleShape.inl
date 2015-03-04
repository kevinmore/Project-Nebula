/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#if !defined(HK_PLATFORM_SPU)

hknpTriangleShape* HK_CALL hknpTriangleShape::createTriangleShape(
	hkVector4Parameter a, hkVector4Parameter b, hkVector4Parameter c, hkReal radius)
{
	hknpTriangleShape* ts = createEmptyTriangleShape( radius );
	ts->setVertices( a,b,c );
	return ts;
}

HK_FORCE_INLINE hknpTriangleShape::hknpTriangleShape( hkReal radius )
	: hknpConvexPolytopeShape( 4, 4, 4, radius, HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, sizeof(hknpTriangleShape)) )
{
	m_flags.orWith( USE_NORMAL_TO_FIND_SUPPORT_PLANE );

	hknpConvexPolytopeShape::Face* faces = getFaces();
	for( int i = 0; i < 4; i++ )
	{
		faces[i].m_firstIndex = 0;
		faces[i].m_numIndices = 4;
		faces[i].m_minHalfAngle = 127;
	}

	hknpConvexPolytopeShape::VertexIndex* indices = getIndices();
	indices[0] = 0;
	indices[1] = 1;
	indices[2] = 2;
	indices[3] = 3;
}

#endif

HK_FORCE_INLINE void hknpTriangleShape::_setVertices(
	hkVector4Parameter a, hkVector4Parameter b, hkVector4Parameter c )
{
	hkVector4* HK_RESTRICT verts  = getVertices();
	hkVector4* HK_RESTRICT planes = getPlanes();

	hkVector4  normal; hkVector4Util::setNormalOfTriangle( normal, a, b, c );
	hkSimdReal invNormalLen = normal.lengthInverse<3>();
	normal.setXYZ_W( normal, -normal.dot<3>(a) );
	normal.mul( invNormalLen );

	// Fill vertices
	verts[0] = a;
	verts[1] = b;
	verts[2] = c;
	verts[3] = a;
	verts[0].setInt24W(0);
	verts[1].setInt24W(1);
	verts[2].setInt24W(2);
	verts[3].setInt24W(0);

	// Fill planes
	// NOTE: These planes are only used for collision detection. Ray casting is special cased.
	planes[0] = normal;
	planes[1].setNeg<4>( normal ); // This is required for backface collisions
	planes[2] = normal;
	planes[3] = normal;
}

HK_FORCE_INLINE void hknpTriangleShape::_setVertices(
	hkVector4Parameter a, hkVector4Parameter b, hkVector4Parameter c, hkVector4Parameter d )
{
	hkVector4* HK_RESTRICT verts  = getVertices();
	hkVector4* HK_RESTRICT planes = getPlanes();

	hkVector4  normal; hkVector4Util::setNormalOfTriangle( normal, a, b, c );
	hkSimdReal invNormalLen = normal.lengthInverse<3>();
	normal.setXYZ_W( normal, -normal.dot<3>(a) );
	normal.mul( invNormalLen );

	// Fill vertices
	verts[0] = a;
	verts[1] = b;
	verts[2] = c;
	verts[3] = d;
	verts[0].setInt24W(0);
	verts[1].setInt24W(1);
	verts[2].setInt24W(2);
	verts[3].setInt24W(3);

	// Fill planes
	// NOTE: These planes are only used for collision detection. Ray casting is special cased.
	planes[0] = normal;
	planes[1].setNeg<4>( normal ); // This is required for backface collisions
	planes[2] = normal;
	planes[3] = normal;
}

HK_FORCE_INLINE bool hknpTriangleShape::isQuad() const
{
	// Currently vertex 3's .w component is 3 for quad and 0 for a triangle
	const hkcdVertex& vtx = getVertex(3);
	return vtx.getInt24W() == 3;
}

#if defined(HK_PLATFORM_SPU)

HK_FORCE_INLINE const hknpTriangleShape* HK_CALL hknpTriangleShape::getReferenceTriangleShape()
{
	HK_ASSERT(0x2283a73f, s_referenceTriangleShape);
	return s_referenceTriangleShape;
}

#else

HK_FORCE_INLINE const hknpTriangleShape* HK_CALL hknpTriangleShape::getReferenceTriangleShape()
{
	static hknpInplaceTriangleShape instance(true);
	return instance.getTriangleShape();
}

#endif

HK_FORCE_INLINE hknpInplaceTriangleShape::hknpInplaceTriangleShape( hkReal radius )
{
	hkString::memCpy16NonEmpty( m_buffer, hknpTriangleShape::getReferenceTriangleShape(), BUFFER_SIZE >> 4 );
	getTriangleShape()->m_convexRadius = radius;
}

HK_FORCE_INLINE hknpTriangleShape* hknpInplaceTriangleShape::getTriangleShape()
{
	return (hknpTriangleShape*)(m_buffer);
}

HK_FORCE_INLINE void hknpTriangleShape::_convertToSingleTriangle()
{
	hkVector4* HK_RESTRICT verts  = getVertices();
	verts[3] = verts[0];
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
