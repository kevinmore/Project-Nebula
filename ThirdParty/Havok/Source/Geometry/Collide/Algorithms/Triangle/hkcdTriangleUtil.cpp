/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Geometry/Collide/hkcdCollide.h>
#include <Geometry/Collide/Algorithms/Triangle/hkcdTriangleUtil.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>

//
//	Returns true if the triangle is degenerate.
//	Degenerate is assumed to be:
//	     - it has very small area (cross product of edges all squared less than given tolerance).
//	     - it has a aspect ratio which will cause collision detection algorithms to fail.
//
//		a              First vertex.
//		b              Second vertex.
//		c              Third vertex.
//		tolerance	   Minimal acceptable area and squared edge length

hkBool32 HK_CALL hkcdTriangleUtil::isDegenerate(hkVector4Parameter a, hkVector4Parameter b, hkVector4Parameter c, hkSimdRealParameter tolerance)
{
	struct LocalUtilities
	{
		static HK_FORCE_INLINE hkReal dot3fpu(const hkVector4& aa, const hkVector4& bb)
		{
			volatile hkReal p0 = aa(0) * bb(0);
			volatile hkReal p1 = aa(1) * bb(1);
			volatile hkReal p2 = aa(2) * bb(2);
			return p0 + p1 + p2;
		}
	};
	// Small area check
	{
		hkVector4 edge1; edge1.setSub(a, b);
		hkVector4 edge2; edge2.setSub(a, c);
		hkVector4 cross; cross.setCross(edge1, edge2);

		hkVector4 edge1b; edge1b.setSub(b, a);
		hkVector4 edge2b; edge2b.setSub(b, c);
		hkVector4 crossb; crossb.setCross(edge1b, edge2b);

		hkVector4Comparison cmp0 = tolerance.greater(cross.lengthSquared<3>());
		hkVector4Comparison cmp1 = tolerance.greater(crossb.lengthSquared<3>());
		cmp0.setOr(cmp0, cmp1);
		if ( cmp0.allAreSet() )
		{ 
			return true;
		}
	}

	// Point triangle distance check
	{
		hkVector4 Q; Q.setSub(a, b);      
		hkVector4 R; R.setSub(c, b);

		// Use FPU math for better determinism across platforms
		const hkReal QQ = LocalUtilities::dot3fpu(Q, Q);
		const hkReal RR = LocalUtilities::dot3fpu(R, R);
		const hkReal QR = LocalUtilities::dot3fpu(R, Q);

		volatile hkReal QQRR = QQ * RR;
		volatile hkReal QRQR = QR * QR;
		hkReal Det = (QQRR - QRQR);

		return (Det == 0.0f);
	}
}


//
//	calculate the barycentric coordinates of a point projected onto a triangle.
//	Note: result 0 and 2 are always sign correct, result 1 is calculated as 1.0f - p0 - p2, this function is not epsilon safe.

void HK_CALL hkcdTriangleUtil::calcBarycentricCoordinates(hkVector4Parameter vP, hkVector4Parameter vA, hkVector4Parameter vB, hkVector4Parameter vC, hkVector4& result)
{
	// If the triangle is non-degenerate, we should be able to return proper barycentric coordinates
	{
		// Compute the triangle normal
		hkVector4 vN;
		hkcdTriangleUtil::calcNonUnitNormal(vA, vB, vC, vN);

		// Determine whether the point is inside the triangle
		hkVector4 vDots;
		{
			hkVector4 vPA;	vPA.setSub(vA, vP);
			hkVector4 vPB;	vPB.setSub(vB, vP);
			hkVector4 vPC;	vPC.setSub(vC, vP);

			hkVector4 nPAB, nPBC, nPCA;
			hkVector4Util::cross_3vs1(vPB, vPC, vPA, vN, nPAB, nPBC, nPCA);
			hkVector4Util::dot3_3vs3(vPB, nPBC, vPC, nPCA, vPA, nPAB, vDots);
		}

		// Barycentric coordinates need normalization.
		hkSimdReal barySum = vDots.horizontalAdd<3>();
		if ( barySum.isNotEqualZero() )
		{
			// We can normalize the barycentric coordinates and exit!
			hkSimdReal baryInvSum;
			baryInvSum.setReciprocal(barySum);
			result.setMul(vDots, baryInvSum);
			return;
		}
	}

	// The triangle is degenerate (i.e. a line segment).
	// Compute the barycentric coordinates for the point on the longest edge of the triangle.
	// If the triangle is a point, return zero.
	{
		hkVector4 vAB;	vAB.setSub(vB, vA);
		hkVector4 vBC;	vBC.setSub(vC, vB);
		hkVector4 vCA;	vCA.setSub(vA, vC);

		hkVector4 vAP;	vAP.setSub(vP, vA);
		hkVector4 vBP;	vBP.setSub(vP, vB);
		hkVector4 vCP;	vCP.setSub(vP, vC);

		// Compute edge lengths
		hkVector4 vEdgeLen, vSegLen;
		hkVector4Util::dot3_3vs3(vAP, vAB, vBP, vBC, vCP, vCA, vSegLen);
		hkVector4Util::dot3_3vs3(vAB, vAB, vBC, vBC, vCA, vCA, vEdgeLen);

		// Compute the index of the maximum edge length
		const hkSimdReal maxEdgeLen = vEdgeLen.horizontalMax<3>();
		int bestIndex = vEdgeLen.findComponent<3>(maxEdgeLen);

		// Prevent division by zero
		vEdgeLen.setMax(vEdgeLen, hkVector4::getConstant<HK_QUADREAL_EPS>());

		// Compute times of intersection
		hkVector4 vTois;
		vTois.setDiv(vSegLen, vEdgeLen);

		{
			const hkVector4 vOne = hkVector4::getConstant<HK_QUADREAL_1>();

			hkVector4 b0;	b0.setSub(vOne, vTois);	// (u0, u1, u2, *)
			hkVector4 b1 = vTois;					// (v0, v1, v2, *)
			hkVector4 b2;	b2.setZero();			// (0, 0, 0, 0)

			// b0 = (u0, v0, 0, *)
			// b1 = (u1, v1, 0, *)
			// b2 = (u2, v2, 0, *)
			HK_TRANSPOSE3(b0, b1, b2);
			b1.setPermutation<hkVectorPermutation::ZXYZ>(b1);	// b1 = (0, u1, v1, 0)
			b2.setPermutation<hkVectorPermutation::YZXZ>(b2);	// b2 = (v2, 0, u2, 0)

			HK_ALIGN_REAL(hkVector4) bary[3] = { b0, b1, b2 };
			result.setZero();
			result.setSelect(maxEdgeLen.greaterZero(), bary[bestIndex], result);
		}
	}
}

//
//	Previous version for computing barycentric coordinates, now deprecated.

void HK_CALL hkcdTriangleUtil::calcBarycentricCoordinatesDeprecated(hkVector4Parameter pos, hkVector4Parameter t0, hkVector4Parameter t1, hkVector4Parameter t2, hkVector4& result)
{
	hkVector4 R, Q;
	Q.setSub(t0, t1);      
	R.setSub(t2, t1);

	const hkReal QQ = Q.lengthSquared<3>().getReal();
	const hkReal RR = R.lengthSquared<3>().getReal();
	const hkReal QR = R.dot<3>(Q).getReal();

	const hkReal QQRR = QQ * RR;
	const hkReal QRQR = QR * QR;
	const hkReal Det = (QQRR - QRQR);

	if ( Det > 0.0f )
	{
		const hkReal invDet = 1.0f / Det;

		hkVector4 relPos; relPos.setSub( t1, pos );
		hkReal sq = relPos.dot<3>(Q).getReal();
		hkReal sr = relPos.dot<3>(R).getReal();

		hkReal q = (sr * QR - RR * sq);
		hkReal r = (sq * QR - QQ * sr);

		result(0) = invDet * q;
		result(1) = invDet * (Det - q - r) ;
		result(2) = invDet * r;
		return;
	}

	hkVector4 S;
	S.setSub( t0, t2 );
	const hkReal SS = S.lengthSquared<3>().getReal();

	if ( QQ >= RR )
	{
		if ( SS >= QQ )
		{
			result(1) = 0.0f;
			if ( SS > 0.0f )
			{
				hkVector4 relPos; relPos.setSub( pos, t2 );
				hkReal p = relPos.dot<3>(S).getReal();
				result(0) = p / SS;
				result(2) = 1.0f - result(0);
				return;
			}
			result(0) = 0.0f;
			result(2) = 0.0f;
			return;
		}
		else
		{
			hkVector4 relPos; relPos.setSub( pos, t1 );
			hkReal p  = relPos.dot<3>(Q).getReal();
			result(0) = p / QQ;
			result(1) = 1.0f - result(0);
			result(2) = 0.0f;
			return;
		}
	}
	else
	{
		if ( SS >= RR )
		{
			hkVector4 relPos; relPos.setSub( pos, t2 );
			hkReal p = relPos.dot<3>(S).getReal();
			result(0) = p / SS;
			result(1) = 0.0f;
			result(2) = 1.0f - result(0);
			return;
		}
		else
		{
			hkVector4 relPos; relPos.setSub( pos, t1 );
			hkReal p  = relPos.dot<3>(R).getReal();
			result(0) = 0.0f;
			result(2) = p / RR;
			result(1) = 1.0f - result(2);
			return;
		}
	}
}

//
//	Table for triangle plane intersect edges. Edges are stored as vertex indices, i.e. (v0 << 4) | v1
//	Indices are:
//		a = 0
//		b = 1
//		c = 2
//		iab = 3;
//		iac = 4;
//		ibc = 5;

static const hkUint8 s_planeIntersectEdgeTable[27] = 
{
	0xFF,	// Rule 0x000
	0x01,	// Rule 0x001 -> [a, b]
	0x01,	// Rule 0x002

	0x02,	// Rule 0x010 -> [a, c]
	0x00,	// Rule 0x011 -> [a, a]
	0x05,	// Rule 0x012 -> [a, ibc]

	0x02,	// Rule 0x020 -> [a, c]
	0x05,	// Rule 0x021 a, ibc
	0x00,	// Rule 0x022 a, a

	0x12,	// Rule 0x100 b, c
	0x11,	// Rule 0x101 b, b
	0x14,	// Rule 0x102 b, iac

	0x22,	// Rule 0x110 c, c
	0xFF,	// Rule 0x111 
	0x45,	// Rule 0x112 iac, ibc

	0x32,	// Rule 0x120 iab, c
	0x35,	// Rule 0x121 iab, ibc
	0x34,	// Rule 0x122 iab, iac

	0x12,	// Rule 0x200 b, c
	0x14,	// Rule 0x201 b, iac
	0x11,	// Rule 0x202 b, b

	0x32,	// Rule 0x210 iab, c
	0x34,	// Rule 0x211 iab, iac
	0x35,	// Rule 0x212 iab, ibc

	0x22,	// Rule 0x220 c, c
	0x45,	// Rule 0x221 iac, ibc
	0xFF,	// Rule 0x222 
};

//
//	Clip a triangle with a plane
//		a               First vertex.
//		b               Second vertex.
//		c               Third vertex.
//		plane		   Plane equation.
//		edgesOut		   Resulting intersection edges. Includes degenerate edges where both start and end points coincide
//		tolerance	   Intersection tolerance.

int HK_CALL hkcdTriangleUtil::clipWithPlane(hkVector4Parameter a, hkVector4Parameter b, hkVector4Parameter c,
											hkVector4Parameter plane, hkSimdRealParameter tolerance, hkVector4 edgesOut[6])
{
	// Compute signed distances from the plane to each triangle vertex
	hkVector4 d012;
	{
		hkSimdReal d0 = plane.dot4xyz1(a);
		hkSimdReal d1 = plane.dot4xyz1(b);
		hkSimdReal d2 = plane.dot4xyz1(c);

		d012.set(d0, d1, d2, d2);
	} 

	// Treat very small values as zero
	{
		hkVector4 absd012;	absd012.setAbs(d012);
		hkVector4 tol;		tol.setAll(tolerance);

		const hkVector4Comparison cmp = absd012.less(tol);
		d012.zeroIfTrue(cmp);
	}

	// Compute codes: 0 if d == 0, 1 if d < 0 and 2 if d > 0
	int rule;
	{
		const hkVector4Comparison lt0 = d012.lessZero();
		const hkVector4Comparison gt0 = d012.greaterZero();

		const int maskLt0 = lt0.getMask();
		const int maskGt0 = gt0.getMask();

		// Compute codes: 0 if d == 0, 1 if d < 0 and 2 if d > 0
		const int c0 = ((maskLt0 & hkVector4ComparisonMask::MASK_X) | ((maskGt0 & hkVector4ComparisonMask::MASK_X) << 1)) >> hkVector4Comparison::INDEX_X;
		const int c1 = ((maskLt0 & hkVector4ComparisonMask::MASK_Y) | ((maskGt0 & hkVector4ComparisonMask::MASK_Y) << 1)) >> hkVector4Comparison::INDEX_Y;
		const int c2 = ((maskLt0 & hkVector4ComparisonMask::MASK_Z) | ((maskGt0 & hkVector4ComparisonMask::MASK_Z) << 1)) >> hkVector4Comparison::INDEX_Z;

		// Compute rule code
		rule = c0 * 9 + c1 * 3 + c2;
	}

	//Handle special rules: 3 or 0 resulting edges
	if ( !rule )	//(0, 0, 0)
	{
		edgesOut[0] = a;		edgesOut[1] = b;
		edgesOut[2] = b;		edgesOut[3] = c;
		edgesOut[4] = c;		edgesOut[5] = a;
		return 3;	// 3 edges
	}

	if( (rule == 13) || (rule == 26) )	// (-, -, -) or (+, +, +)
		return 0;	// No intersection, all distances positive / negative

	//All other cases produce an edge
	// Compute interpolation factors
	//	factors = [d0 / (d0 - d2), d0 / (d0 - d1), d1 / (d1 - d2), *]
	hkVector4 factors;
	{
		hkVector4 d001, d212;
		d001.setPermutation<hkVectorPermutation::XXYY>(d012);
		d212.setPermutation<hkVectorPermutation::ZYZZ>(d012);

		hkVector4 diff;
		diff.setSub(d001, d212);

		// Prevent division by zero!
		diff.setSelect(diff.equalZero(), hkVector4::getConstant<HK_QUADREAL_EPS>(), diff);
		factors.setDiv(d001, diff);
	}

	// Compute interpolated values
	HK_ALIGN_REAL(hkVector4) vecs[6];
	vecs[0] = a;	vecs[1] = b;	vecs[2] = c;
	vecs[3].setInterpolate(a, b, factors.getComponent<1>());
	vecs[4].setInterpolate(a, c, factors.getComponent<0>());
	vecs[5].setInterpolate(b, c, factors.getComponent<2>());

	// Compute edge code, vertex indices, and output intersection edge
	{
		const hkUint8 edgeCode = s_planeIntersectEdgeTable[rule];
		const int idx0 = (edgeCode >> 4) & 0xF;
		const int idx1 = edgeCode & 0xF;
		edgesOut[0] = vecs[idx0];
		edgesOut[1] = vecs[idx1];
	}

	// Return 1 edge
	return 1;
}


hkBool32 hkcdTriangleUtil::checkForFlatConvexQuad( hkVector4Parameter a, hkVector4Parameter b, hkVector4Parameter c, hkVector4Parameter d, hkReal tolerance)
{
	hkVector4 normal; hkcdTriangleUtil::calcUnitNormal( a, b, c, normal );

	hkVector4 p30; p30.setSub( a, d);
	hkSimdReal outOfPlane; outOfPlane.setAbs( p30.dot<3>( normal ) );
	hkSimdReal eps; eps.setFromFloat(tolerance);

	if ( outOfPlane > eps )
	{
		return false;
	}
	
	// check for convex
	hkVector4 p01; p01.setSub( b, a );
	hkVector4 p12; p12.setSub( c, b );
	//hkVector4 p23; p23.setSub( d, c );
	hkVector4 en01; en01.setCross( normal, p01 );
	hkVector4 en12; en12.setCross( normal, p12 );
	en01.normalize<3,HK_ACC_12_BIT, HK_SQRT_IGNORE>();
	en12.normalize<3,HK_ACC_12_BIT, HK_SQRT_IGNORE>();

	hkVector4 p32; p32.setSub( c, d );
	hkSimdReal d013 = en01.dot<3>(p30);
	hkSimdReal d123 = en12.dot<3>(p32);
	hkSimdReal md; md.setMax( d013, d123 );
	return md.isLess( eps );
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
