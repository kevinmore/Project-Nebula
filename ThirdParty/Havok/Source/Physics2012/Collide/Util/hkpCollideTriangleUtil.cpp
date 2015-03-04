/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>
#include <Physics2012/Collide/Util/hkpTriangleUtil.h>
#include <Physics2012/Collide/Util/hkpCollideTriangleUtil.h>
#include <Common/Base/Algorithm/Collide/LineSegment/hkLineSegmentUtil.h>
#include <Common/Base/Algorithm/Sort/hkSort.h>

void HK_CALL hkpCollideTriangleUtil::setupClosestPointTriangleCache( const hkVector4* tri, ClosestPointTriangleCache& cache )
{
	hkVector4 Q; Q.setSub(tri[0], tri[1]);      
	hkVector4 R; R.setSub(tri[2], tri[1]);

	
	









	const hkReal QQ = hkpTriangleUtil::dot3fullAcc(Q, Q);
	const hkReal RR = hkpTriangleUtil::dot3fullAcc(R, R);
	const hkReal QR = hkpTriangleUtil::dot3fullAcc(R, Q);

	volatile hkReal QQRR = QQ * RR;
	volatile hkReal QRQR = QR * QR;
	const hkReal Det = (QQRR - QRQR);


	HK_ASSERT2(0x761ec287,  Det != 0, "possible degenerate triangle encountered" );

	const hkReal invDet = hkReal(1) / Det;





	cache.m_QQ = QQ * invDet;
	cache.m_RR = RR * invDet;
	cache.m_QR = QR * invDet;

	hkVector4 triNormal; triNormal.setCross( Q, R );
	const hkSimdReal len2 = triNormal.lengthSquared<3>();
	HK_ASSERT2(0x1693884c,  len2.getReal() > 0, "Error: Not a valid triangle" );
	const hkSimdReal invLen = len2.sqrtInverse<HK_ACC_FULL, HK_SQRT_IGNORE>();
	invLen.store<1,HK_IO_NATIVE_ALIGNED>((hkReal*)&(cache.m_invTriNormal));
}


void hkpCollideTriangleUtil::setupPointTriangleDistanceCache( const hkVector4* tri, PointTriangleDistanceCache& cache )
{
	hkVector4 E0; E0.setSub(tri[2], tri[1]);      
	hkVector4 E1; E1.setSub(tri[0], tri[2]);
	hkVector4 E2; E2.setSub(tri[1], tri[0]);

	hkVector4 triNormal; triNormal.setCross( E0, E1 );

	hkVector4 dots, invSqrtDots;
	hkVector4Util::dot3_4vs4(E0, E0, E1, E1, E2, E2, triNormal, triNormal, dots);

	invSqrtDots.setSqrtInverse<HK_ACC_23_BIT,HK_SQRT_IGNORE>(dots);
	invSqrtDots.store<4,HK_IO_NATIVE_ALIGNED>(&cache.m_invEdgeLen[0]);
	

	const hkSimdReal nrmLen = invSqrtDots.getComponent<3>() * dots.getComponent<3>();
	nrmLen.store<1,HK_IO_NATIVE_ALIGNED>(&cache.m_normalLen);
}



// #if HK_SIMD_COMPARE_MASK_X == 1
//													 0,  x, y  xy, z,  xz, yz, xyz
const hkInt8 hkpCollideTriangleUtil::maskToIndex[] = { -1,  0, 1, 2-8, 2, 1-8, 0-8, -1 };
//													 xyzw, xyzW, xyZw, xyZW, xYzw, xYzW, xYZw, xYZW, Xyzw, XyzW, XyZw, XyZW, XYzw, XYzW, XYZw, XYZW
//const hkInt8 hkpCollideTriangleUtil::maskToIndex[] = {    -1,   -1,    2,   -1,    1,   -1,  0-8,   -1,    0,   -1,  1-8,   -1,  2-8,   -1,   -1,   -1 };
// #else
// #	error unknown mask
// #endif
const hkInt8 hkpCollideTriangleUtil::vertexToEdgeLut[] = { 2, 0, 1, 2, 0 };


#if !defined(HK_PLATFORM_SPU)
void HK_CALL hkpCollideTriangleUtil::calcTrianglePlaneDirections( const hkVector4* tri, const PointTriangleDistanceCache& cache, hkTransform& planeEquationsOut, hkVector4& normalOut )
{

    hkVector4 E0; E0.setSub(tri[2], tri[1]);      
    hkVector4 E1; E1.setSub(tri[0], tri[2]);
    hkVector4 E2; E2.setSub(tri[1], tri[0]);

	hkVector4 invE; invE.load<4,HK_IO_NATIVE_ALIGNED>(&cache.m_invEdgeLen[0]); 
	hkSimdReal cacheNLen; cacheNLen.load<1,HK_IO_NATIVE_ALIGNED>(&cache.m_normalLen);

	hkVector4 triNormal; triNormal.setCross( E0, E1 );
	triNormal.mul( invE.getComponent<3>() );
	normalOut = triNormal;

	E0.mul( invE.getComponent<0>() );
	E1.mul( invE.getComponent<1>() );
	E2.mul( invE.getComponent<2>() );
	
	hkVector4& m0 = planeEquationsOut.getColumn(0);
	hkVector4& m1 = planeEquationsOut.getColumn(1);
	hkVector4& m2 = planeEquationsOut.getColumn(2);
	hkVector4& m3 = planeEquationsOut.getColumn(3);

	m0.setCross( E0, triNormal );
	m1.setCross( E1, triNormal );
	m2.setCross( E2, triNormal );
	m3 = triNormal;

	HK_TRANSPOSE4(m0,m1,m2,m3);

	m3.setXYZ_W(hkVector4::getZero(), -invE.getComponent<0>() * cacheNLen);
}
#endif

#if defined(HK_PLATFORM_PS3_PPU) && defined(HK_COMPILER_SNC)
_Pragma("control %push O=0")
#endif
hkpCollideTriangleUtil::ClosestPointTriangleStatus HK_CALL hkpCollideTriangleUtil::closestPointTriangle( hkVector4Parameter position, const hkVector4* tri, const ClosestPointTriangleCache& cache, ClosestPointTriangleResult& result , hkpFeatureOutput* featureOutput )
{
	//
	// if allocated, featureOutput will be used for welding the closest point normals 
	//
	if(featureOutput)
	{
		featureOutput->numFeatures = 0;
	}

	hkVector4 relPos; relPos.setSub( tri[1], position );

    hkVector4 Q; Q.setSub(tri[0], tri[1]);      
    hkReal sq = relPos.dot<3>(Q).getReal();

    hkVector4 R; R.setSub(tri[2], tri[1]);
    hkReal sr = relPos.dot<3>(R).getReal();

    hkReal q = (sr * cache.m_QR - cache.m_RR * sq);
    hkReal r = (sq * cache.m_QR - cache.m_QQ * sr);

	// Make sure, we are really outside, before moving to the edge edge cases
	const hkReal relEps = hkReal(0.001f);

	hkVector4 proj;
	proj.set(	q + relEps,
				(hkReal(1) + relEps) - q - r,
				r + relEps,
				relEps );

	const hkVector4Comparison mask = proj.greaterZero();
		
	//
	//	Completely inside
	//
	if ( mask.allAreSet<hkVector4ComparisonMask::MASK_XYZ>() )
	{
		//const hkRotation& triMatrix = reinterpret_cast<const hkRotation&>(tri[0]);
		result.hitDirection.setCross( Q, R );
#if defined(HK_PLATFORM_SPU)
		hkSimdReal invTN; invTN.setFromFloat(cache.m_invTriNormal.val());
#else
		hkSimdReal invTN; invTN.setFromFloat(cache.m_invTriNormal);
#endif
		result.hitDirection.mul( invTN );
		hkSimdReal rDist = result.hitDirection.dot<3>( relPos );

		if ( rDist.isGreaterZero() )
		{
			result.hitDirection.setNeg<4>( result.hitDirection );
		}
		else
		{
			rDist = -rDist;
		}
		result.distance = rDist.getReal();

		if(featureOutput)
		{
			featureOutput->numFeatures = 3;
			featureOutput->featureIds[0] = (hkpVertexId)0;
			featureOutput->featureIds[1] = (hkpVertexId)1;
			featureOutput->featureIds[2] = (hkpVertexId)2;
		}

		return HIT_TRIANGLE_FACE;
	}

	// now do the three edges
	int index = hkpCollideTriangleUtil::maskToIndex[ mask.getMask<hkVector4ComparisonMask::MASK_XYZ>() ];

	// check for a single positive value, only one edge needed
	if ( index < 0 )
	{
		index += 8;
		HK_ASSERT2(0x53fe2ebc,  index >=0 && index < 3, "Degenerate case found in point - triangle collision detection algorithm. This can result from a degenerate input triangle" );
		
		hkLineSegmentUtil::ClosestPointLineSegResult cpls;
		int whereOnLine = hkLineSegmentUtil::closestPointLineSeg( position, tri[ vertexToEdgeLut[ index + 2 ] ], tri[ vertexToEdgeLut[ index ] ], cpls );
		
		if( featureOutput )
		{
			if( whereOnLine == hkLineSegmentUtil::CLSLS_POINTB_START )
			{			
				featureOutput->featureIds[0] = (hkpVertexId)vertexToEdgeLut[ index ];
				featureOutput->numFeatures = 1;
			}
			else if( whereOnLine == hkLineSegmentUtil::CLSLS_POINTB_END )
			{
				featureOutput->featureIds[0] = (hkpVertexId)vertexToEdgeLut[ index + 2 ];			
				featureOutput->numFeatures = 1;
			}
			else
			{
				featureOutput->featureIds[0] = (hkpVertexId)vertexToEdgeLut[ index ];
				featureOutput->featureIds[1] = (hkpVertexId)vertexToEdgeLut[ index + 2 ];			
				featureOutput->numFeatures = 2;
			}
		}		

		result.hitDirection.setSub( position, cpls.m_pointOnEdge ); 
		result.distance = result.hitDirection.normalizeWithLength<3>().getReal();
	}
	else
		// check two edges and search the closer one
	{
		HK_ASSERT(0x5a25e14d,  index >=0 && index < 3 );

		hkLineSegmentUtil::ClosestPointLineSegResult cplsA;
		int whereOnLineA = hkLineSegmentUtil::closestPointLineSeg( position, tri[ index ], tri[vertexToEdgeLut[ index + 2 ] ], cplsA );
	
		hkLineSegmentUtil::ClosestPointLineSegResult cplsB;
		int whereOnLineB = hkLineSegmentUtil::closestPointLineSeg( position, tri[ vertexToEdgeLut[ index ] ], tri[ index ], cplsB );

		hkVector4 t0; t0.setSub( position, cplsA.m_pointOnEdge ); 
		hkVector4 t1; t1.setSub( position, cplsB.m_pointOnEdge );
		const hkSimdReal distA = t0.lengthSquared<3>();
		const hkSimdReal distB = t1.lengthSquared<3>();

		if ( distA < distB )
		{
			const hkSimdReal inv = distA.sqrtInverse();

			if( featureOutput )
			{
				if( whereOnLineA == hkLineSegmentUtil::CLSLS_POINTB_START )
				{
					featureOutput->numFeatures = 1;
					featureOutput->featureIds[0] = (hkpVertexId)index;
				}
				else if( whereOnLineA == hkLineSegmentUtil::CLSLS_POINTB_END )
				{
					featureOutput->numFeatures = 1;
					featureOutput->featureIds[0] = (hkpVertexId)vertexToEdgeLut[ index + 2 ];
				}
				else
				{
					featureOutput->numFeatures = 2;
					featureOutput->featureIds[0] = (hkpVertexId)index;
					featureOutput->featureIds[1] = (hkpVertexId)vertexToEdgeLut[ index + 2 ];
				}			
			}			

			result.distance = (distA * inv).getReal();
			result.hitDirection.setMul( inv, t0 );
		}
		else
		{
			const hkSimdReal inv = distB.sqrtInverse();			
			
			if( featureOutput )
			{
				if( whereOnLineB == hkLineSegmentUtil::CLSLS_POINTB_START )
				{
					featureOutput->numFeatures = 1;
					featureOutput->featureIds[0] = (hkpVertexId)index;
				}
				else if( whereOnLineB == hkLineSegmentUtil::CLSLS_POINTB_END )
				{
					featureOutput->numFeatures = 1;
					featureOutput->featureIds[0] = (hkpVertexId)vertexToEdgeLut[ index ];
				}
				else
				{
					featureOutput->numFeatures = 2;
					featureOutput->featureIds[0] = (hkpVertexId)index;
					featureOutput->featureIds[1] = (hkpVertexId)vertexToEdgeLut[ index ];
				}
			}
			

			result.distance = (distB * inv).getReal();
			result.hitDirection.setMul( inv, t1 );
		}
	}

	return HIT_TRIANGLE_EDGE;
}

#if defined(HK_PLATFORM_PS3_PPU) && defined(HK_COMPILER_SNC)
_Pragma("control %pop O")
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
