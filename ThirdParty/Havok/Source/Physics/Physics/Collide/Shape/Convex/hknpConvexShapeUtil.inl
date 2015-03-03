/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/Inertia/hkInertiaTensorComputer.h>
#include <Common/Internal/GeometryProcessing/ConvexHull/hkgpConvexHull.h>
#include <Common/Base/Container/Array/hkArray.h>
#include <Geometry/Internal/Algorithms/Distance/hkcdDistancePointTriangle.h>
#include <Physics/Internal/Collide/Gsk/hknpGskUtil.h>

HK_DISABLE_OPTIMIZATION_VS2008_X64
void hknpConvexShapeUtil::getSupportingVertices2(
	const hkVector4* fvA, int numA, const hkVector4* fvB,  int numB,
	hkVector4Parameter directionA, hkVector4Parameter directionB,
	hkVector4* bestAOut, hkVector4* bestBOut )
{
	hk4xVector4 bestVertA;	bestVertA.moveLoad( fvA );
	hk4xVector4 bestVertB;	bestVertB.moveLoad( fvB );

	hk4xSingle directionAVec; directionAVec.setVector(directionA);
	hk4xSingle directionBVec; directionBVec.setVector(directionB);

	hk4xReal bestDotA; bestDotA.setDot<3>( directionAVec, bestVertA  );
	hk4xReal bestDotB; bestDotB.setDot<3>( directionBVec, bestVertB  );

	fvA += 4;
	fvB += 4;
	numA -= 4;
	numB -= 4;

#if !defined(HK_PLATFORM_SPU)
	// if we are fighting for codesize, don't do shape unrolled loop
	int maxNum = hkMath::min2( numA, numB );	// optimize me

	while ( maxNum > 0)
	{
		hk4xVector4 vertA;	vertA.moveLoad(fvA);
		hk4xVector4 vertB;	vertB.moveLoad(fvB);

		hk4xReal dotA; dotA.setDot<3>( directionAVec, vertA  );
		hk4xReal dotB; dotB.setDot<3>( directionBVec, vertB  );

		hk4xVector4 bA; bA.setBroadcast(bestDotA);
		hk4xVector4 bB; bB.setBroadcast(bestDotB);

		hk4xMask compA; dotA.greater( bA, compA );
		hk4xMask compB; dotB.greater( bB, compB );

		bestDotA.setSelect(  compA, dotA,	bestDotA   );
		bestVertA.setSelect( compA, vertA, bestVertA  );
		bestDotB.setSelect(  compB, dotB,  bestDotB   );
		bestVertB.setSelect( compB, vertB, bestVertB  );

		fvA+=4;
		fvB+=4;
		numA -= 4;
		numB -= 4;
		maxNum -= 4;
	}
#endif

	while( numA > 0 )
	{
		hk4xVector4 vertA;	vertA.moveLoad(fvA);
		hk4xReal dotA;   dotA.setDot<3>( directionAVec, vertA  );
		hk4xVector4 bA; bA.setBroadcast(bestDotA);
		hk4xMask compA; dotA.greater( bA, compA );

		bestDotA.setSelect ( compA, dotA,  bestDotA  );
		bestVertA.setSelect ( compA, vertA, bestVertA  );

		fvA+=4;
		numA -= 4;
	}

	while( numB > 0 )
	{
		hk4xVector4 vertB;	vertB.moveLoad(fvB);
		hk4xReal dotB;   dotB.setDot<3>( directionBVec, vertB  );
		hk4xVector4 bB; bB.setBroadcast(bestDotB);
		hk4xMask compB; dotB.greater( bB, compB );
		bestDotB.setSelect(  compB, dotB,    bestDotB   );
		bestVertB.setSelect( compB, vertB,   bestVertB  );

		fvB+=4;
		numB -= 4;
	}

	// binary combine a
	hkVector4 bestA;
	{
		hkVector4Comparison cmp1gt0 = bestDotA.getReal<1>().greater( bestDotA.getReal<0>() );
		hkVector4Comparison cmp3gt2 = bestDotA.getReal<3>().greater( bestDotA.getReal<2>() );

		hkSimdReal bestDotA01;  bestDotA01.setSelect( cmp1gt0, bestDotA.getReal<1>(),    bestDotA.getReal<0>() );
		hkVector4 bestVertA01; bestVertA01.setSelect( cmp1gt0, bestVertA.getVector<1>(), bestVertA.getVector<0>() );
		hkSimdReal bestDotA23;  bestDotA23.setSelect( cmp3gt2, bestDotA.getReal<3>(),    bestDotA.getReal<2>() );
		hkVector4 bestVertA23; bestVertA23.setSelect( cmp3gt2, bestVertA.getVector<3>(), bestVertA.getVector<2>() );

		hkVector4Comparison a01GTa23 = bestDotA01.greater(bestDotA23);
		bestA.setSelect( a01GTa23, bestVertA01,  bestVertA23 );
	}
	hkVector4 bestB;
	{
		hkVector4Comparison cmp1gt0 = bestDotB.getReal<1>().greater( bestDotB.getReal<0>() );
		hkVector4Comparison cmp3gt2 = bestDotB.getReal<3>().greater( bestDotB.getReal<2>() );

		hkSimdReal bestDotB01;  bestDotB01.setSelect( cmp1gt0, bestDotB. getReal<1>(),   bestDotB.getReal<0>() );
		hkVector4 bestVertB01; bestVertB01.setSelect( cmp1gt0, bestVertB.getVector<1>(), bestVertB.getVector<0>()   );
		hkSimdReal bestDotB23;  bestDotB23.setSelect( cmp3gt2, bestDotB. getReal<3>(),   bestDotB.getReal<2>() );
		hkVector4 bestVertB23; bestVertB23.setSelect( cmp3gt2, bestVertB.getVector<3>(), bestVertB.getVector<2>()   );

		hkVector4Comparison b01GTb23 = bestDotB01.greater(bestDotB23);
		bestB.setSelect( b01GTb23, bestVertB01, bestVertB23  );
	}
	*bestAOut = bestA;
	*bestBOut = bestB;
}
HK_RESTORE_OPTIMIZATION_VS2008_X64

HK_FORCE_INLINE void hknpConvexShapeUtil::getSupportingVertices(
	const hknpConvexShape& shapeA, hkVector4Parameter directionA,
	const hknpConvexShape& shapeB, const hkTransform& aTb,
	hkcdVertex* HK_RESTRICT vertexAinAOut,
	hkcdVertex* HK_RESTRICT vertexBinBOut,
	hkVector4* HK_RESTRICT vertexBinAOut)
{
	hkVector4 negDir;	 negDir.setNeg<4>(directionA);
	hkVector4 directionB; directionB._setRotatedInverseDir( aTb.getRotation(), negDir);

	const hkVector4* fvA = shapeA.getVertices();
	const hkVector4* fvB = shapeB.getVertices();

	int numA = shapeA.hknpConvexShape::getNumberOfVertices();
	int numB = shapeB.hknpConvexShape::getNumberOfVertices();

	getSupportingVertices2(fvA, numA, fvB, numB, directionA, directionB, (hkVector4*)vertexAinAOut, vertexBinBOut );
	vertexBinAOut->_setTransformedPos(aTb, *vertexBinBOut);

#if defined(HK_DEBUG)
	if (1)
	{
		hkcdVertex cA;		shapeA.getSupportingVertexRef( directionA, cA );
		hkcdVertex cB;		shapeB.getSupportingVertexRef( directionB, cB );

		HK_ASSERT(0x5a7aca40, directionA.dot<3>( cA ).getReal() <= HK_REAL_EPSILON + directionA.dot<3>( *vertexAinAOut ).getReal() );
		HK_ASSERT(0x1b6d4f65, directionB.dot<3>( cB ).getReal() <= HK_REAL_EPSILON + directionB.dot<3>( *vertexBinBOut ).getReal() );
	}
#endif
}


HK_FORCE_INLINE void hknpConvexShapeUtil::getSupportingFace(
	const hkVector4* planeA, const int facesA, hkVector4Parameter surfacePointA, int& faceAOut )
{
	const hkUint32 numFacesAminus4 = facesA-4;	// getting faces early to avoid LHS on XBOX
	HK_ASSERT2( 0xf0fc4cf1, facesA>=4 , "Your shape need to have at least 4 faces (padding is OK)");

	hkVector4 pA; pA.setXYZ_W( surfacePointA, hkVector4::getConstant(HK_QUADREAL_1));

	// lets do the last 4 vertices first (to avoid padding issues)

	int a = numFacesAminus4;	// we already evaluated the first 4, we know that planes are padded, so shape is safe

	hkVector4 bestDotA; hkVector4Util::dot4_4vs4( pA, planeA[a+0], pA, planeA[a+1], pA, planeA[a+2], pA, planeA[a+3], bestDotA );
	hkIntVector	curIndices;		curIndices.m_quad = (const hkQuadUint&)s_curIndices;
	hkIntVector	stepIndices;	stepIndices.splatImmediate32<4>();
	hkIntVector	bestIndicesA;	bestIndicesA.setAll( numFacesAminus4 );
	bestIndicesA.setAddU32( bestIndicesA, curIndices );	// these are the last indices

	while ( a > 0 )
	{
		hkVector4 curDot; hkVector4Util::dot4_4vs4( pA, planeA[0], pA, planeA[1], pA, planeA[2], pA, planeA[3], curDot );
		planeA += 4; a -= 4;
		hkVector4Comparison comp = bestDotA.less(curDot);
		bestDotA.setSelect( comp, curDot, bestDotA  );
		bestIndicesA.setSelect( comp, curIndices, bestIndicesA );
		curIndices.setAddU32( curIndices, stepIndices );
	}
	int faceIdA = bestIndicesA.getComponentAtVectorMax(bestDotA);
	faceAOut = faceIdA;
}


HK_FORCE_INLINE void hknpConvexShapeUtil::getSupportingFaces(
	const hkVector4* HK_RESTRICT planeA, const int facesA,
	const hkVector4* HK_RESTRICT planeB, const int facesB,
	hkVector4Parameter surfacePointA, hkVector4Parameter surfacePointB,
	int& faceAOut, int& faceBOut  )
{
	const hkUint32 numFacesAminus4 = facesA-4;	// getting faces early to avoid LHS on XBOX
	const hkUint32 numFacesBminus4 = facesB-4;
	HK_ASSERT2( 0xf0fc4cf1, facesA>=4 && facesB>=4, "Your shape need to have at least 4 faces (padding is OK)");

	hkVector4 pA; pA.setXYZ_W( surfacePointA, hkVector4::getConstant(HK_QUADREAL_1));
	hkVector4 pB; pB.setXYZ_W( surfacePointB, hkVector4::getConstant(HK_QUADREAL_1));

	// lets do the last 4 vertices first (to avoid padding issues)

	int a = numFacesAminus4;	// we already evaluated the first 4, we know that planes are padded, so shape is safe
	int b = numFacesBminus4;

	hkVector4 bestDotA; hkVector4Util::dot4_4vs4( pA, planeA[a+0], pA, planeA[a+1], pA, planeA[a+2], pA, planeA[a+3], bestDotA );
	hkVector4 bestDotB; hkVector4Util::dot4_4vs4( pB, planeB[b+0], pB, planeB[b+1], pB, planeB[b+2], pB, planeB[b+3], bestDotB );

	hkIntVector curIndices;		curIndices.m_quad = (const hkQuadUint&)s_curIndices;
	hkIntVector stepIndices;	stepIndices.splatImmediate32<4>();

	hkIntVector bestIndicesA;	bestIndicesA.setAll( numFacesAminus4 );
	hkIntVector bestIndicesB;	bestIndicesB.setAll( numFacesBminus4 );
	bestIndicesA.setAddU32( bestIndicesA, curIndices );	// these are the last indices
	bestIndicesB.setAddU32( bestIndicesB, curIndices );

#if !defined(HK_PLATFORM_SPU)
	int maxNum = hkMath::min2( a, b );	// optimize me
	while ( maxNum > 0)
	{
		hkVector4 curDotA; hkVector4Util::dot4_4vs4( pA, planeA[0], pA, planeA[1], pA, planeA[2], pA, planeA[3], curDotA );
		hkVector4 curDotB; hkVector4Util::dot4_4vs4( pB, planeB[0], pB, planeB[1], pB, planeB[2], pB, planeB[3], curDotB );

		hkVector4Comparison compA = bestDotA.less(curDotA);
		hkVector4Comparison compB = bestDotB.less(curDotB);
		bestDotA.setSelect( compA, curDotA, bestDotA  );
		bestDotB.setSelect( compB, curDotB, bestDotB );
		bestIndicesA.setSelect( compA, curIndices, bestIndicesA );
		bestIndicesB.setSelect( compB, curIndices, bestIndicesB );
		planeA+=4;
		planeB+=4;
		a -= 4;
		b -= 4;
		maxNum -= 4;
		curIndices.setAddU32( curIndices, stepIndices );
	}
#endif

	hkIntVector curIndicesB = curIndices;

	while ( a > 0 )
	{
		hkVector4 curDot; hkVector4Util::dot4_4vs4( pA, planeA[0], pA, planeA[1], pA, planeA[2], pA, planeA[3], curDot );
		planeA +=4; a-=4;
		hkVector4Comparison comp = bestDotA.less(curDot);
		bestDotA.setSelect( comp, curDot, bestDotA  );
		bestIndicesA.setSelect( comp, curIndices, bestIndicesA );
		curIndices.setAddU32( curIndices, stepIndices );
	}
	while ( b > 0 )
	{
		hkVector4 curDot; hkVector4Util::dot4_4vs4( pB, planeB[0], pB, planeB[1], pB, planeB[2], pB, planeB[3], curDot );
		planeB +=4; b-=4;

		hkVector4Comparison comp = bestDotB.less(curDot);
		bestDotB.setSelect( comp, curDot, bestDotB );
		bestIndicesB.setSelect( comp, curIndicesB, bestIndicesB );
		curIndicesB.setAddU32( curIndicesB, stepIndices );
	}

	int faceIdA = bestIndicesA.getComponentAtVectorMax(bestDotA);
	int faceIdB = bestIndicesB.getComponentAtVectorMax(bestDotB);

	faceAOut = faceIdA;
	faceBOut = faceIdB;
}


HK_FORCE_INLINE void hknpConvexShapeUtil::getFaceVertices(
	const hknpConvexPolytopeShape& shapeA, const int faceIndexA, hkVector4& planeOutA, int& verticesOutA, hkVector4* const HK_RESTRICT vertexBufferA,
	const hknpConvexPolytopeShape& shapeB, const int faceIndexB, hkVector4& planeOutB, int& verticesOutB, hkVector4* const HK_RESTRICT vertexBufferB )
{
	static const hknpConvexShape::VertexIndex mo = hknpConvexShape::VertexIndex(~0);
	static const hknpConvexShape::VertexIndex indexMask[] = {
		mo, mo, mo, mo, mo, mo, mo, mo, mo, mo, mo, mo, mo, mo, mo, mo,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	HK_COMPILE_TIME_ASSERT( sizeof(indexMask) == hknpConvexShape::MAX_NUM_VERTICES_PER_FACE * sizeof(hknpConvexShape::VertexIndex)*2 );

	// Get vertices from shape A
	const hkVector4* const HK_RESTRICT verticesA = shapeA.getVertices();
	const hknpConvexPolytopeShape::Face &faceA = shapeA.hknpConvexPolytopeShape::getFace( faceIndexA );
	const int numVerticesA = faceA.m_numIndices;
	const hknpConvexPolytopeShape::VertexIndex* indexMasksA = &indexMask[ hknpConvexPolytopeShape::MAX_NUM_VERTICES_PER_FACE - numVerticesA ];
	const hknpConvexPolytopeShape::VertexIndex* indicesA	= &shapeA.hknpConvexPolytopeShape::getIndices()[faceA.m_firstIndex];

	// Get vertices from shape B
	const hkVector4* const HK_RESTRICT verticesB = shapeB.getVertices();
	const hknpConvexPolytopeShape::Face &faceB = shapeB.hknpConvexPolytopeShape::getFace( faceIndexB );
	const int numVerticesB = faceB.m_numIndices;
	const hknpConvexPolytopeShape::VertexIndex* indexMasksB = &indexMask[ hknpConvexPolytopeShape::MAX_NUM_VERTICES_PER_FACE - numVerticesB ];
	const hknpConvexPolytopeShape::VertexIndex* indicesB	= &shapeB.hknpConvexPolytopeShape::getIndices()[faceB.m_firstIndex];

	// write plane equation
	planeOutA = shapeA.m_planes[faceIndexA];
	planeOutB = shapeB.m_planes[faceIndexB];

	// write out number of vertices
	verticesOutA = numVerticesA;
	verticesOutB = numVerticesB;

	// copy vertices
	{
		hkVector4 v0A; v0A.load<4>( &verticesA[indicesA[/*indexMasks[0]&*/(0)]](0) ); // a face has at least 2 edges, no need to check them
		hkVector4 v1A; v1A.load<4>( &verticesA[indicesA[/*indexMasks[1]&*/(1)]](0) );
		hkVector4 v2A; v2A.load<4>( &verticesA[indicesA[indexMasksA[2] & (2)]](0) );
		hkVector4 v3A; v3A.load<4>( &verticesA[indicesA[indexMasksA[3] & (3)]](0) );

		hkVector4 v0B; v0B.load<4>( &verticesB[indicesB[/*indexMasks[0]&*/(0)]](0) ); // a face has at least 2 edges, no need to check them
		hkVector4 v1B; v1B.load<4>( &verticesB[indicesB[/*indexMasks[1]&*/(1)]](0) );
		hkVector4 v2B; v2B.load<4>( &verticesB[indicesB[indexMasksB[2] & (2)]](0) );
		hkVector4 v3B; v3B.load<4>( &verticesB[indicesB[indexMasksB[3] & (3)]](0) );

		vertexBufferA[0] = v0A;
		vertexBufferA[1] = v1A;
		vertexBufferA[2] = v2A;
		vertexBufferA[3] = v3A;

		vertexBufferB[0] = v0B;
		vertexBufferB[1] = v1B;
		vertexBufferB[2] = v2B;
		vertexBufferB[3] = v3B;
	}

	if ( HK_VERY_UNLIKELY(numVerticesA>4 || numVerticesB>4 ) )
	{
		int i  = 4;
		do
		{
			hkVector4 v0A; v0A.load<4>( &verticesA[indicesA[indexMasksA[i]   & (i  )]](0) ); // a face has at least 2 edges, no need to check them
			hkVector4 v1A; v1A.load<4>( &verticesA[indicesA[indexMasksA[i+1] & (i+1)]](0) );
			hkVector4 v2A; v2A.load<4>( &verticesA[indicesA[indexMasksA[i+2] & (i+2)]](0) );
			hkVector4 v3A; v3A.load<4>( &verticesA[indicesA[indexMasksA[i+3] & (i+3)]](0) );

			hkVector4 v0B; v0B.load<4>( &verticesB[indicesB[indexMasksB[i]   & (i  )]](0) ); // a face has at least 2 edges, no need to check them
			hkVector4 v1B; v1B.load<4>( &verticesB[indicesB[indexMasksB[i+1] & (i+1)]](0) );
			hkVector4 v2B; v2B.load<4>( &verticesB[indicesB[indexMasksB[i+2] & (i+2)]](0) );
			hkVector4 v3B; v3B.load<4>( &verticesB[indicesB[indexMasksB[i+3] & (i+3)]](0) );

			vertexBufferA[i]   = v0A;
			vertexBufferA[i+1] = v1A;
			vertexBufferA[i+2] = v2A;
			vertexBufferA[i+3] = v3A;

			vertexBufferB[i]   = v0B;
			vertexBufferB[i+1] = v1B;
			vertexBufferB[i+2] = v2B;
			vertexBufferB[i+3] = v3B;

			i+=4;
		} while (i < numVerticesA || i < numVerticesB);
	}
}


HK_FORCE_INLINE void hknpConvexShapeUtil::getFaceVertices(
	const hknpConvexPolytopeShape& shapeA, const int faceIndexA, int& verticesOutA, hkVector4* const HK_RESTRICT vertexBufferA )
{
	static const hknpConvexShape::VertexIndex mo = hknpConvexShape::VertexIndex(~0);
	static const hknpConvexShape::VertexIndex indexMask[] = {
		mo, mo, mo, mo, mo, mo, mo, mo, mo, mo, mo, mo, mo, mo, mo, mo,
		0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
	HK_COMPILE_TIME_ASSERT( sizeof(indexMask) == hknpConvexShape::MAX_NUM_VERTICES_PER_FACE * sizeof(hknpConvexShape::VertexIndex)*2 );

	// Get vertices from shape A
	const hkVector4* const HK_RESTRICT verticesA = shapeA.getVertices();
	const hknpConvexPolytopeShape::Face &faceA = shapeA.hknpConvexPolytopeShape::getFace( faceIndexA );
	const int numVerticesA = faceA.m_numIndices;
	const hknpConvexPolytopeShape::VertexIndex* indexMasksA = &indexMask[ hknpConvexPolytopeShape::MAX_NUM_VERTICES_PER_FACE - numVerticesA ];
	const hknpConvexPolytopeShape::VertexIndex* indicesA	= &shapeA.hknpConvexPolytopeShape::getIndices()[faceA.m_firstIndex];

	// write out number of vertices
	verticesOutA = numVerticesA;

	// copy vertices
	{
		hkVector4 v0A; v0A.load<4>( &verticesA[indicesA[/*indexMasks[0]&*/(0)]](0) ); // a face has at least 2 edges, no need to check them
		hkVector4 v1A; v1A.load<4>( &verticesA[indicesA[/*indexMasks[1]&*/(1)]](0) );
		hkVector4 v2A; v2A.load<4>( &verticesA[indicesA[indexMasksA[2] & (2)]](0) );
		hkVector4 v3A; v3A.load<4>( &verticesA[indicesA[indexMasksA[3] & (3)]](0) );

		vertexBufferA[0] = v0A;
		vertexBufferA[1] = v1A;
		vertexBufferA[2] = v2A;
		vertexBufferA[3] = v3A;
	}

	if ( HK_VERY_UNLIKELY(numVerticesA>4 ) )
	{
		int i  = 4;
		do
		{
			hkVector4 v0A; v0A.load<4>( &verticesA[indicesA[indexMasksA[i]   & (i  )]](0) ); // a face has at least 2 edges, no need to check them
			hkVector4 v1A; v1A.load<4>( &verticesA[indicesA[indexMasksA[i+1] & (i+1)]](0) );
			hkVector4 v2A; v2A.load<4>( &verticesA[indicesA[indexMasksA[i+2] & (i+2)]](0) );
			hkVector4 v3A; v3A.load<4>( &verticesA[indicesA[indexMasksA[i+3] & (i+3)]](0) );

			vertexBufferA[i]   = v0A;
			vertexBufferA[i+1] = v1A;
			vertexBufferA[i+2] = v2A;
			vertexBufferA[i+3] = v3A;
			i+=4;
		} while (i < numVerticesA );
	}
}

HK_FORCE_INLINE bool hknpConvexShapeUtil::getClosestPoints(
	const hkcdVertex* queryVertices, int numQueryVertices, hkReal queryRadius,
	const hkcdVertex* targetVertices, int numTargetVertices, hkReal targetRadius,
	const hkTransform& queryToTarget,
	hkSimdReal* distance, hkVector4* HK_RESTRICT normal, hkVector4* HK_RESTRICT pointOnTarget)
{
	hkcdGsk::GetClosestPointInput input; input.m_aTb = queryToTarget;
	hkcdGsk::Cache cache; cache.init();
	hkcdGsk::GetClosestPointOutput output;
	hkcdGsk::GetClosestPointStatus status = hkcdGsk::getClosestPoint(
		targetVertices, numTargetVertices, queryVertices, numQueryVertices, input, &cache, output);

	hkSimdReal radii; radii.setFromFloat(queryRadius + targetRadius);
	hkSimdReal currentDistance = output.getDistance() - radii;
	if ((status <= hkcdGsk::STATUS_OK_FLAG) && currentDistance.isLess(*distance))
	{
		hkSimdReal targetRadiusSimd; targetRadiusSimd.setFromFloat(targetRadius);
		pointOnTarget->setSubMul(output.m_pointAinA, output.m_normalInA, targetRadiusSimd);
		normal->setNeg<4>(output.m_normalInA); // m_normalInA points from query (B) to target (A)
		*distance = currentDistance;
		return true;
	}
	return false;
}

HK_FORCE_INLINE bool hknpConvexShapeUtil::getClosestPointsWithScale(
	const hkcdVertex* queryShapeVertices, int numQueryShapeVertices, const hknpQueryFilterData& queryShapeFilterData, const hknpShapeQueryInfo& queryShapeInfo,
	const hkcdVertex* targetShapeVertices, int numTargetShapeVertices, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
	const hkTransform& queryToTarget,
	hknpCollisionQueryCollector* HK_RESTRICT collector )
{
	// If the query shape has a scale factor, we need to bake the scale into the shape's vertices before calling GSK.
	if ( queryShapeInfo.m_shapeIsScaled )
	{
		hkcdVertex* queryShapeVerticesBuffer = hkAllocateStack<hkcdVertex>(numQueryShapeVertices, "getClosestPointsWithScaleQueryShapeVerticesBuffer");
		for (int i = 0; i < numQueryShapeVertices; i++)
		{
			const hkcdVertex& originalVertex = queryShapeVertices[i];
			hkVector4 scaledVertex;
			scaledVertex.setMul(originalVertex, queryShapeInfo.m_shapeScale);
			scaledVertex.add(queryShapeInfo.m_shapeScaleOffset);
			queryShapeVerticesBuffer[i].setXYZ_W( scaledVertex, originalVertex );
		}
		queryShapeVertices = queryShapeVerticesBuffer;
	}

	// If the target shape has a scale factor, we need to bake the scale into the shape's vertices before calling GSK.
	if ( targetShapeInfo.m_shapeIsScaled )
	{
		hkcdVertex* targetShapeVerticesBuffer = hkAllocateStack<hkcdVertex>(numTargetShapeVertices, "getClosestPointsWithScaleTargetShapeVerticesBuffer");
		for (int i = 0; i < numTargetShapeVertices; i++)
		{
			const hkcdVertex& originalVertex = targetShapeVertices[i];
			hkVector4 scaledVertex;
			scaledVertex.setMul(originalVertex, targetShapeInfo.m_shapeScale);
			scaledVertex.add(targetShapeInfo.m_shapeScaleOffset);
			targetShapeVerticesBuffer[i].setXYZ_W( scaledVertex, originalVertex );
		}
		targetShapeVertices = targetShapeVerticesBuffer;
	}

	hkSimdReal distance = collector->getEarlyOutHitFraction();
	hkVector4 targetToQueryNormal; targetToQueryNormal.setZero();
	hkVector4 pointOnTarget; pointOnTarget.setZero();

	bool result = hknpConvexShapeUtil::getClosestPoints(
		queryShapeVertices, numQueryShapeVertices, queryShapeInfo.m_shapeConvexRadius,
		targetShapeVertices, numTargetShapeVertices, targetShapeInfo.m_shapeConvexRadius,
		queryToTarget,
		&distance, &targetToQueryNormal, &pointOnTarget);


	if ( targetShapeInfo.m_shapeIsScaled )
	{
		hkDeallocateStack(const_cast<hkcdVertex*>(targetShapeVertices), numTargetShapeVertices);
	}

	if ( queryShapeInfo.m_shapeIsScaled )
	{
		hkDeallocateStack(const_cast<hkcdVertex*>(queryShapeVertices), numQueryShapeVertices);
	}

	if ( result )
	{
		hknpCollisionResult collisionResult;

		collisionResult.m_queryType	= hknpCollisionQueryType::GET_CLOSEST_POINTS;

		collisionResult.m_fraction	= distance.getReal();
		collisionResult.m_position	. _setTransformedPos( *targetShapeInfo.m_shapeToWorld, pointOnTarget );
		collisionResult.m_normal	. _setRotatedDir( targetShapeInfo.m_shapeToWorld->getRotation(), targetToQueryNormal );

		collisionResult.m_queryBodyInfo.m_bodyId			= ( queryShapeInfo.m_body ? queryShapeInfo.m_body->m_id : hknpBodyId::INVALID );
		collisionResult.m_queryBodyInfo.m_shapeKey			= queryShapeInfo.m_shapeKeyPath.getKey();
		collisionResult.m_queryBodyInfo.m_shapeMaterialId	= queryShapeFilterData.m_materialId;
		collisionResult.m_queryBodyInfo.m_shapeCollisionFilterInfo = queryShapeFilterData.m_collisionFilterInfo;
		collisionResult.m_queryBodyInfo.m_shapeUserData		= queryShapeFilterData.m_userData;

		collisionResult.m_hitBodyInfo.m_bodyId				= ( targetShapeInfo.m_body ? targetShapeInfo.m_body->m_id : hknpBodyId::INVALID );
		collisionResult.m_hitBodyInfo.m_shapeKey			= targetShapeInfo.m_shapeKeyPath.getKey();
		collisionResult.m_hitBodyInfo.m_shapeMaterialId		= targetShapeFilterData.m_materialId;
		collisionResult.m_hitBodyInfo.m_shapeCollisionFilterInfo = targetShapeFilterData.m_collisionFilterInfo;
		collisionResult.m_hitBodyInfo.m_shapeUserData		= targetShapeFilterData.m_userData;

		collector->addHit( collisionResult );

		return true;
	}

	return false;
}

HK_FORCE_INLINE void closestPointOnTriangle(
	const hkcdVertex* triangleVertices, const hkVector4& positionInShape,
	const hknpQueryFilterData& queryShapeFilterData, const hknpShapeQueryInfo& queryShapeInfo,
	const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
	hknpCollisionQueryCollector* HK_RESTRICT collector )
{
	hkVector4 resultNormal;
	hkVector4 pointOnTarget;

	hkcdPointTriangleProject(
		positionInShape,
		triangleVertices[0], triangleVertices[1], triangleVertices[2],
		&pointOnTarget, &resultNormal );

	const hkSimdReal distanceSq = pointOnTarget.distanceToSquared(positionInShape);
	if( distanceSq < hkMath::square(collector->getEarlyOutHitFraction()) )
	{
		hknpCollisionResult hit;

		hit.m_queryType = hknpCollisionQueryType::GET_CLOSEST_POINTS;

		hit.m_fraction	= distanceSq.sqrt<HK_ACC_FULL,HK_SQRT_SET_ZERO>().getReal();
		hit.m_position	. _setTransformedPos( *targetShapeInfo.m_shapeToWorld, pointOnTarget );
		hit.m_normal	. _setRotatedDir( targetShapeInfo.m_shapeToWorld->getRotation(), resultNormal );

		hit.m_queryBodyInfo.m_bodyId					= ( queryShapeInfo.m_body ? queryShapeInfo.m_body->m_id : hknpBodyId::INVALID );
		hit.m_queryBodyInfo.m_shapeKey					= queryShapeInfo.m_shapeKeyPath.getKey();
		hit.m_queryBodyInfo.m_shapeMaterialId			= queryShapeFilterData.m_materialId;
		hit.m_queryBodyInfo.m_shapeCollisionFilterInfo	= queryShapeFilterData.m_collisionFilterInfo;
		hit.m_queryBodyInfo.m_shapeUserData				= queryShapeFilterData.m_userData;

		hit.m_hitBodyInfo.m_bodyId						= ( targetShapeInfo.m_body ? targetShapeInfo.m_body->m_id : hknpBodyId::INVALID );
		hit.m_hitBodyInfo.m_shapeKey					= targetShapeInfo.m_shapeKeyPath.getKey();
		hit.m_hitBodyInfo.m_shapeMaterialId				= targetShapeFilterData.m_materialId;
		hit.m_hitBodyInfo.m_shapeCollisionFilterInfo	= targetShapeFilterData.m_collisionFilterInfo;
		hit.m_hitBodyInfo.m_shapeUserData				= targetShapeFilterData.m_userData;

		collector->addHit(hit);
	}
}

HK_FORCE_INLINE bool hknpConvexShapeUtil::getClosestPointsToTriangleWithScale(
	const hkcdVertex& queryShapeVertex, const hknpQueryFilterData& queryShapeFilterData, const hknpShapeQueryInfo& queryShapeInfo,
	const hkcdVertex* targetShapeVertices, const hknpQueryFilterData& targetShapeFilterData, const hknpShapeQueryInfo& targetShapeInfo,
	const hkTransform& queryToTarget,
	hknpCollisionQueryCollector* HK_RESTRICT collector )
{
	// If the query shape has a scale factor, we need to bake the scale into the shape's vertices before calling GSK.
	hkVector4 scaledVertex = queryShapeVertex;
	if ( queryShapeInfo.m_shapeIsScaled )
	{
		scaledVertex.mul(queryShapeInfo.m_shapeScale);
		scaledVertex.add(queryShapeInfo.m_shapeScaleOffset);
		scaledVertex.setXYZ_W( scaledVertex, queryShapeVertex );
	}

	// If the target shape has a scale factor, we need to bake the scale into the shape's vertices before calling GSK.
	hkcdVertex scaledTargetVertices[3];
	if ( targetShapeInfo.m_shapeIsScaled )
	{
		for (int i = 0; i < 3; i++)
		{
			const hkcdVertex& originalVertex = targetShapeVertices[i];
			hkVector4 scaledTargetVertex;
			scaledTargetVertex.setMul(originalVertex, targetShapeInfo.m_shapeScale);
			scaledTargetVertex.add(targetShapeInfo.m_shapeScaleOffset);
			scaledTargetVertices[i].setXYZ_W( scaledTargetVertex, originalVertex );
		}
		targetShapeVertices = scaledTargetVertices;
	}


	hkVector4 positionInShape; positionInShape.setTransformedPos(queryToTarget, scaledVertex);

	closestPointOnTriangle( targetShapeVertices, positionInShape,
		queryShapeFilterData, queryShapeInfo, targetShapeFilterData, targetShapeInfo, collector);

	return false;
}

HK_FORCE_INLINE hkResult hknpConvexShapeUtil::buildSphereMassProperties(
	const hknpShape::MassConfig& massConfig, hkVector4Parameter center, hkReal radius,
	hkDiagonalizedMassProperties& massPropertiesOut )
{
	if( radius > 0.0f )
	{
		massPropertiesOut.m_volume = 4.0f/3.0f * HK_REAL_PI * radius * radius * radius;
		massPropertiesOut.m_mass = massConfig.calcMassFromVolume( massPropertiesOut.m_volume );
		massPropertiesOut.m_inertiaTensor.setAll( radius * radius * 0.4f * massPropertiesOut.m_mass * massConfig.m_inertiaFactor );
		massPropertiesOut.m_centerOfMass.setXYZ_0( center );
		massPropertiesOut.m_majorAxisSpace.setIdentity();

		return HK_SUCCESS;
	}
	return HK_FAILURE;
}

HK_FORCE_INLINE hkResult hknpConvexShapeUtil::buildCapsuleMassProperties(
	const hknpShape::MassConfig& massConfig, hkVector4Parameter axisStart, hkVector4Parameter axisEnd, hkReal radius,
	hkDiagonalizedMassProperties& massPropertiesOut )
{
	hkMassProperties mp;
	hkResult result = hkInertiaTensorComputer::computeCapsuleVolumeMassProperties( axisStart, axisEnd, radius, 1.0f, mp );
	if( result == HK_SUCCESS )
	{
		massPropertiesOut.pack( mp );
		hkReal mass = massConfig.calcMassFromVolume( massPropertiesOut.m_volume );
		massPropertiesOut.m_inertiaTensor.mul( hkSimdReal::fromFloat( mass*massConfig.m_inertiaFactor ) );
		massPropertiesOut.m_mass = mass;
	}
	return result;
}

HK_FORCE_INLINE hkResult hknpConvexShapeUtil::buildTriangleMassProperties(
	const hknpShape::MassConfig& massConfig, const hkVector4* vertices, hkReal radius,
	hkDiagonalizedMassProperties& massPropertiesOut )
{
	hkMassProperties mp;
	radius = hkMath::max2(radius, 0.0001);
	hkResult result = hkInertiaTensorComputer::computeTriangleSurfaceMassProperties(
		vertices[0], vertices[1], vertices[2], 1.0f, radius, mp);
	if (mp.m_volume == 0)
	{
		// We have a degenerated triangle
		return HK_FAILURE;
	}
	if( result == HK_SUCCESS )
	{
		massPropertiesOut.pack( mp );
		hkReal mass = massConfig.calcMassFromVolume( massPropertiesOut.m_volume );
		massPropertiesOut.m_inertiaTensor.mul( hkSimdReal::fromFloat( mass*massConfig.m_inertiaFactor ) );
		massPropertiesOut.m_mass = mass;
	}
	return result;
}

HK_FORCE_INLINE hkResult hknpConvexShapeUtil::buildQuadMassProperties(
	const hknpShape::MassConfig& massConfig, const hkVector4* vertices, hkReal radius,
	hkDiagonalizedMassProperties& massPropertiesOut )
{
	hkInplaceArray<hkMassElement, 2> massElements(2);
	hkResult result;

	result = hkInertiaTensorComputer::computeTriangleSurfaceMassProperties(
		vertices[0], vertices[1], vertices[2], 1.0f, radius, massElements[0].m_properties);
	if( result == HK_FAILURE )
	{
		return HK_FAILURE;
	}

	result = hkInertiaTensorComputer::computeTriangleSurfaceMassProperties(
		vertices[2], vertices[3], vertices[0], 1.0f, radius, massElements[1].m_properties);
	if( result == HK_FAILURE )
	{
		return HK_FAILURE;
	}

	hkMassProperties mp;
	hkInertiaTensorComputer::combineMassProperties( massElements, mp );
	massPropertiesOut.pack( mp );
	if (mp.m_volume == 0)
	{
		// We have a degenerated quad
		return HK_FAILURE;
	}

	return HK_SUCCESS;
}

HK_FORCE_INLINE hkResult hknpConvexShapeUtil::buildHullMassProperties(
	const hknpShape::MassConfig& massConfig, const hkVector4* vertices, const int numVertices, hkReal radius,
	hkDiagonalizedMassProperties& massPropertiesOut )
{
	HK_ASSERT( 0xe85a014, numVertices > 0 && radius >= 0.0f );
	switch( numVertices )
	{
	case 1:
		// Sphere
		return buildSphereMassProperties( massConfig, vertices[0], radius, massPropertiesOut );

	case 2:
		// Capsule
		return buildCapsuleMassProperties( massConfig, vertices[0], vertices[1], radius, massPropertiesOut );

	case 3:
		// Triangle
		return buildTriangleMassProperties( massConfig, vertices, radius, massPropertiesOut );

	default:
		// Hull
		{
			hkgpConvexHull hull;
			hkgpConvexHull::BuildConfig hullBuildConfig;
			hullBuildConfig.m_allowLowerDimensions = true;
			hullBuildConfig.m_buildMassProperties = true;

			// Build a hull from the vertices
			const int dim = hull.build( vertices, numVertices, hullBuildConfig );

			// Handle degenerated cases
			switch ( dim )
			{
			case 0:
				// Handle sphere with duplicated points
				return buildSphereMassProperties( massConfig, vertices[0], radius, massPropertiesOut );

			case 1:
				// Handle collinear capsule
				{
					hkAabb aabb; aabb.setEmpty();
					for (int i = 0; i < numVertices; i++)
					{
						aabb.includePoint(vertices[i]);
					}
					return buildCapsuleMassProperties( massConfig, aabb.m_max, aabb.m_min, radius, massPropertiesOut );
				}

			case 2:
				// Handle planes
				{
					// Build a quad
					if (numVertices == 4)
					{
						return hknpConvexShapeUtil::buildQuadMassProperties( massConfig, vertices, radius, massPropertiesOut );
					}
					// Find the best quad for the polygon. NOTE: this is an approximation!
					else
					{
						hkVector4 quad[4];
						hknpGskUtil::findBestQuadImpl(hkVector4::getZero(), numVertices, vertices, quad);
						return hknpConvexShapeUtil::buildQuadMassProperties( massConfig, quad, radius, massPropertiesOut );
					}
				}

			case 3:
				// Expand to include the radius
				if( radius > 0.0f )
				{
					hkgpConvexHull::AbsoluteScaleConfig scaleConfig;
					scaleConfig.m_method = hkgpConvexHull::AbsoluteScaleConfig::SKM_PLANES;
					hull.absoluteScale( radius, scaleConfig );
				}

				if( hull.hasValidMassProperties() )
				{
					// Pack the mass properties
					hkMassProperties massProperties;
					{
						massProperties.m_volume = hull.getVolume().getReal();
						massProperties.m_mass = massConfig.calcMassFromVolume( massProperties.m_volume );
						massProperties.m_centerOfMass = hull.getCenterOfMass();
						massProperties.m_inertiaTensor.setMul(
							hkSimdReal::fromFloat( massProperties.m_mass * massConfig.m_inertiaFactor ),
							hull.getWorldInertia() );
					}
					massPropertiesOut.pack( massProperties );
					return HK_SUCCESS;
				}
				return HK_FAILURE;

			default:
				return HK_FAILURE;
			}
		}
	}
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
