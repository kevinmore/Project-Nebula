/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_CONVEX_POLYTOPE_SHAPE_INL
#define HKNP_CONVEX_POLYTOPE_SHAPE_INL

#if !defined(HK_PLATFORM_SPU)

HK_FORCE_INLINE hknpConvexPolytopeShape::hknpConvexPolytopeShape(
	int numVertices, int numFaces, int numIndices, hkReal radius, int sizeOfBaseClass )
	: hknpConvexShape(numVertices, radius, sizeOfBaseClass)
{
	init(numVertices, numFaces, numIndices, sizeOfBaseClass);
}

HK_FORCE_INLINE void* hknpConvexPolytopeShape::allocateConvexPolytopeShape(
	int numVertices, int numFaces, int numIndices, int sizeOfBaseClass, int& shapeSizeOut )
{
	HK_ASSERT(0xF973BE7F, (sizeOfBaseClass - HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, sizeOfBaseClass)) == 0 );

	// Compute memory footprint
	shapeSizeOut = calcConvexPolytopeShapeSize( numVertices, numFaces, numIndices, sizeOfBaseClass );
	HK_ASSERT2(0xf034e3de, (shapeSizeOut < 32767), "Shape too large to fit in memory");
	// There is a vertex index in the gsk cache (8 bit)
	// Because of padding, the nearest multiple of 4 before exceeding 255 is 252.
	HK_ASSERT2(0xF973BE7C, (numVertices&3) == 0, "Vertices must be padded");
	HK_ASSERT2(0x9FC800A4, (numVertices < 256), "Too many vertices");

	// Allocate
	void* block = hkMemoryRouter::getInstance().heap().blockAlloc(shapeSizeOut);
	HK_MEMORY_TRACKER_ON_NEW_REFOBJECT(shapeSizeOut, block);

	return block;
}

#endif

HK_FORCE_INLINE void hknpConvexPolytopeShape::init(int numVertices, int numFaces, int numIndices, int sizeOfBaseClass)
{
	m_flags.orWith( IS_CONVEX_POLYTOPE_SHAPE );

	if( numFaces < 256 )
	{
		m_flags.orWith( USE_SMALL_FACE_INDICES );
	}

	const int numPlanes = HK_NEXT_MULTIPLE_OF(4, numFaces);
	hkUint16 verticesOffset	= (hkUint16) HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, sizeOfBaseClass);
	hkUint16 planesOffset	= (hkUint16) HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, verticesOffset + sizeof(hkVector4) * numVertices);
	hkUint16 facesOffset	= (hkUint16) HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, planesOffset   + sizeof(hkVector4) * numPlanes);
	hkUint16 indicesOffset	= (hkUint16) HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, facesOffset    + sizeof(Face) * numFaces);

	// Compute stream offsets
	m_planes._setOffset(planesOffset - HK_OFFSET_OF(hknpConvexPolytopeShape, m_planes));
	m_planes._setSize((hkUint16) numPlanes);
	m_faces._setOffset(facesOffset - HK_OFFSET_OF(hknpConvexPolytopeShape, m_faces));
	m_faces._setSize((hkUint16) numFaces);
	m_indices._setOffset(indicesOffset - HK_OFFSET_OF(hknpConvexPolytopeShape, m_indices));
	m_indices._setSize((hkUint16) numIndices);
}

HK_FORCE_INLINE int hknpConvexPolytopeShape::calcConvexPolytopeShapeSize(
	int numVertices, int numFaces, int numIndices, int sizeofBaseClass )
{
	// Calculate the required size to store everything, assuming that the start of each relarray's data is aligned.
	const int numPlanes		= HK_NEXT_MULTIPLE_OF(4, numFaces);
	const int baseSize		= HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, sizeofBaseClass);
	const int vertsSize		= HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, ((numVertices >> 2) * sizeof(hkMatrix4)));
	const int planesSize	= HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, ((numPlanes >> 2) * sizeof(hkMatrix4)));
	const int facesSize		= HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, (numFaces * sizeof(Face)));
	const int indicesSize	= HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, (numIndices * sizeof(VertexIndex)));
	const int totalSize		= HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, baseSize + vertsSize + planesSize + facesSize + indicesSize);

	return totalSize;
}

HK_FORCE_INLINE void hknpConvexPolytopeShape::getPlanes( hkVector4* HK_RESTRICT planesOut ) const
{
	// copy plane equations
	const int N = m_planes.getSize();
	for( int i=0; i<N; ++i )
	{
		planesOut[i] = m_planes[i];
	}
}

HK_FORCE_INLINE void hknpConvexPolytopeShape::getFaceInfo( const int index, hkVector4& planeOut, int& minAngleOut ) const
{
	planeOut = m_planes[index];
	minAngleOut = m_faces[index].m_minHalfAngle;
}

HK_FORCE_INLINE int hknpConvexPolytopeShape::getSupportingFace(
	hkVector4Parameter surfacePointA, const hkcdGsk::Cache* gskCache, bool useB,
	hkVector4& planeOut, int& minAngleOut, hkUint32 &prevFaceId ) const
{
	// find supporting face and write vertices to sfOut
	const int facesA = hknpConvexPolytopeShape::getNumPlanes();
	const hkVector4* HK_RESTRICT planeA = m_planes.begin();
	const hkUint32 numFacesAminus4 = facesA-4;	// getting faces early to avoid LHS on XBOX

	hkVector4 pA; pA.setXYZ_W( surfacePointA, hkVector4::getConstant(HK_QUADREAL_1));

	// lets do the last 4 vertices first (to avoid padding issues)
	int a = numFacesAminus4;	// we already evaluated the first 4, we know that planes are padded, so shape is safe

	hkVector4	bestDotA; hkVector4Util::dot4_4vs4( pA, planeA[a+0], pA, planeA[a+1], pA, planeA[a+2], pA, planeA[a+3], bestDotA );
	hkIntVector	curIndices;		curIndices.m_quad = *(const hkQuadUint*)s_curIndices;
	hkIntVector	stepIndices;	stepIndices.splatImmediate32<4>();
	hkIntVector	bestIndicesA;	bestIndicesA.setAll( numFacesAminus4 );
	bestIndicesA.setAddU32( bestIndicesA, curIndices );	// these are the last indices

#if !defined(HK_PLATFORM_SPU)
	int maxNum = a;
	while ( maxNum > 0)
	{
		hkVector4 curDotA; hkVector4Util::dot4_4vs4( pA, planeA[0], pA, planeA[1], pA, planeA[2], pA, planeA[3], curDotA );

		hkVector4Comparison compA = bestDotA.less(curDotA);
		bestDotA.setSelect( compA, curDotA, bestDotA  );
		bestIndicesA.setSelect( compA, curIndices, bestIndicesA );
		planeA+=4;
		a -= 4;
		maxNum -= 4;
		curIndices.setAddU32( curIndices, stepIndices );
	}
#endif

	while ( a > 0 )
	{
		hkVector4 curDot; hkVector4Util::dot4_4vs4( pA, planeA[0], pA, planeA[1], pA, planeA[2], pA, planeA[3], curDot );
		planeA +=4; a-=4;
		hkVector4Comparison comp = bestDotA.less(curDot);
		bestDotA.setSelect( comp, curDot, bestDotA  );
		bestIndicesA.setSelect( comp, curIndices, bestIndicesA );
		curIndices.setAddU32( curIndices, stepIndices );
	}

	int best =  bestIndicesA.getComponentAtVectorMax(bestDotA);

	// Unfortunately, we cannot write this out before the loop above, because we do not know
	// the face index at that point. Having these writes after the loop means that the two
	// variables a likely to end up in memory and not in registers.
	planeOut = m_planes[best];
	minAngleOut = m_faces[best].m_minHalfAngle;

	return best;
}

HK_FORCE_INLINE int hknpConvexPolytopeShape::getSupportingFaceFromNormal(
	hkVector4Parameter normal, hkVector4& planeOut, int& minAngleOut, hkUint32 &prevFaceId ) const
{
	// find supporting face and write vertices to sfOut
	const int facesA = hknpConvexPolytopeShape::getNumPlanes();
	const hkVector4* HK_RESTRICT planeA = m_planes.begin();
	const hkUint32 numFacesAminus4 = facesA-4;	// getting faces early to avoid LHS on XBOX

	hkVector4 negNormal; negNormal.setNeg<4>(normal);

	// lets do the last 4 vertices first (to avoid padding issues)
	int a = numFacesAminus4;	// we already evaluated the first 4, we know that planes are padded, so shape is safe

	hkVector4	bestDotA; hkVector4Util::dot3_4vs4( negNormal, planeA[a+0], negNormal, planeA[a+1], negNormal, planeA[a+2], negNormal, planeA[a+3], bestDotA );
	hkIntVector	curIndices;		curIndices.m_quad = *(const hkQuadUint*)s_curIndices;
	hkIntVector	stepIndices;	stepIndices.splatImmediate32<4>();
	hkIntVector	bestIndicesA;	bestIndicesA.setAll( numFacesAminus4 );
	bestIndicesA.setAddU32( bestIndicesA, curIndices );	// these are the last indices

	while ( a > 0 )
	{
		hkVector4 curDot; hkVector4Util::dot3_4vs4( negNormal, planeA[0], negNormal, planeA[1], negNormal, planeA[2], negNormal, planeA[3], curDot );
		planeA +=4; a-=4;
		hkVector4Comparison comp = bestDotA.less(curDot);
		bestDotA.setSelect( comp, curDot, bestDotA  );
		bestIndicesA.setSelect( comp, curIndices, bestIndicesA );
		curIndices.setAddU32( curIndices, stepIndices );
	}

	int best = bestIndicesA.getComponentAtVectorMax(bestDotA);

	// Unfortunately, we cannot write this out before the loop above, because we do not know the face index at that point.
	// Having these writes after the loop means that the two variables are likely to end up in memory and not in registers.
	planeOut = m_planes[best];
	minAngleOut = m_faces[best].m_minHalfAngle;

	return best;
}

#endif // HKNP_CONVEX_POLYTOPE_SHAPE_INL

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
