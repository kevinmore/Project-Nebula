/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Geometry/Collide/hkcdCollide.h>
#include <Geometry/Collide/DataStructures/Planar/Predicates/hkcdPlanarGeometryPredicates.h>

//
//	Statistics

#ifdef HK_DEBUG

hkUint32 hkcdPlanarGeometryPredicates::m_numApproxCalls	= 0;
hkUint32 hkcdPlanarGeometryPredicates::m_numExactCalls	= 0;
hkUint32 hkcdPlanarGeometryPredicates::m_numApproxDet3	= 0;
hkUint32 hkcdPlanarGeometryPredicates::m_numApproxDet4	= 0;
hkUint32 hkcdPlanarGeometryPredicates::m_numExactDet3	= 0;
hkUint32 hkcdPlanarGeometryPredicates::m_numExactDet4	= 0;

#endif

//
//	Computes the orientation of the point (ptPlaneA, ptPlaneB, ptPlaneC) with respect to the planeD
//	Must evaluate the sign of:
//		| ax ay az |   | ax ay az aw | 
//		| bx by bz | * | bx by bz bw |
//		| cx cy cz |   | cx cy cz cw |
//					   | dx dy dz dw |

hkcdPlanarGeometryPredicates::Orientation HK_CALL hkcdPlanarGeometryPredicates::orientation(PlaneParam ptPlaneA, PlaneParam ptPlaneB, PlaneParam ptPlaneC, PlaneParam planeD)
{
	// Evaluate Det3
	hkUint32 retVal = 0;
	hkVector4dComparison det3LessZero, det3EqualZero;
	if ( !computeApproxDet3(ptPlaneA, ptPlaneB, ptPlaneC, det3LessZero, det3EqualZero) )
	{
		//retVal |= CACHEABLE;
		computeExactDet3(ptPlaneA, ptPlaneB, ptPlaneC, det3LessZero, det3EqualZero);
	}
// 	else
// 	{
// 		hkVector4Comparison det3lz, det3ez;
// 		computeExactDet3(ptPlaneA, ptPlaneB, ptPlaneC, det3lz, det3ez);
// 		HK_ASSERT(0x23bae805, (det3LessZero.getMask() == det3lz.getMask()) && (det3EqualZero.getMask() == det3ez.getMask()));
// 	}

	// Evaluate Det4
	hkVector4dComparison det4LessZero, det4EqualZero;
	if ( !computeApproxDet4(ptPlaneA, ptPlaneB, ptPlaneC, planeD, det4LessZero, det4EqualZero) )
	{
		//retVal |= CACHEABLE;
		computeExactDet4(ptPlaneA, ptPlaneB, ptPlaneC, planeD, det4LessZero, det4EqualZero);
	}
// 	else
// 	{
// 		hkVector4Comparison det4lz, det4ez;
// 		computeExactDet4(ptPlaneA, ptPlaneB, ptPlaneC, planeD, det4lz, det4ez);
// 		HK_ASSERT(0x117c898c, (det4LessZero.getMask() == det4lz.getMask()) && (det4EqualZero.getMask() == det4ez.getMask()));
// 	}

	// Check signs, return proper orientation
	retVal |= CACHEABLE;
	{
		const int sign = mulSigns(det3LessZero, det3EqualZero, det4LessZero, det4EqualZero);
		return (Orientation)(((sign < 0) ? BEHIND : ((sign > 0) ? IN_FRONT_OF : ON_PLANE)) | retVal);
	}
}

//
//	Computes the intersection determinants using exact arithmetic. Only the non-null determinants are computed

void HK_CALL hkcdPlanarGeometryPredicates::computeIntersectionDeterminants(const Plane (&planes)[3], hkSimdInt<256>* outDetX, hkSimdInt<256>* outDetY, hkSimdInt<256>* outDetZ, hkSimdInt<256>* outDetW)
{
	// Compute common stuff
	hkInt64Vector4 iNrm0;	planes[0].getExactNormal(iNrm0);
	hkInt64Vector4 iNrm1;	planes[1].getExactNormal(iNrm1);
	hkInt64Vector4 iNrm2;	planes[2].getExactNormal(iNrm2);
	hkInt128Vector4 nAB;	nAB.setCross(iNrm0, iNrm1);
	hkInt128Vector4 nBC;	nBC.setCross(iNrm1, iNrm2);
	hkInt128Vector4 nCA;	nCA.setCross(iNrm2, iNrm0);

	hkSimdInt<128> iOffset0;	planes[0].getExactOffset(iOffset0);
	hkSimdInt<128> iOffset1;	planes[1].getExactOffset(iOffset1);
	hkSimdInt<128> iOffset2;	planes[2].getExactOffset(iOffset2);

	if ( outDetX )
	{
		hkSimdInt<256> detX;
		detX.setMul(iOffset0, nBC.m_x);
		detX.addMul(iOffset1, nCA.m_x);
		detX.addMul(iOffset2, nAB.m_x);
		outDetX->setNeg(detX);
	}
	if ( outDetY )
	{
		hkSimdInt<256> detY;
		detY.setMul(iOffset0, nBC.m_y);
		detY.addMul(iOffset1, nCA.m_y);
		detY.addMul(iOffset2, nAB.m_y);
		outDetY->setNeg(detY);
	}
	if ( outDetZ )
	{
		hkSimdInt<256> detZ;
		detZ.setMul(iOffset0, nBC.m_z);
		detZ.addMul(iOffset1, nCA.m_z);
		detZ.addMul(iOffset2, nAB.m_z);
		outDetZ->setNeg(detZ);
	}
	if ( outDetW )
	{
		*outDetW = nBC.dot<3>(iNrm0);
	}
}

//
//	Returns true if the vertex defined by three planes (ptPlaneA, ptPlaneB, ptPlaneC) is coplanar to all the given test planes

hkBool32 HK_CALL hkcdPlanarGeometryPredicates::coplanar(PlaneParam planeA, PlaneParam planeB, PlaneParam planeC, const Plane* HK_RESTRICT testPlanes, int numTestPlanes)
{
	const hkSimdDouble64 tol = hkSimdDouble64::fromFloat(TOL_DET_4);

	// The point is collinear if
	//		Orientation(ptPlaneA, ptPlaneB, ptPlaneC, testPlane[0]) == ON_PLANE
	//		Orientation(ptPlaneA, ptPlaneB, ptPlaneC, testPlane[1]) == ON_PLANE...
	// The point is assumed to be valid, i.e. the determinant of planes A, B, C is non-null
	// In order for the point to be collinear, we only need to test the 4-determinants!
	{
		const hkVector4d fEqnA	= planeA.getApproxEquation();
		const hkVector4d fEqnB	= planeB.getApproxEquation();
		hkVector4d nAB;			nAB.setCross(fEqnA, fEqnB);
		hkVector4d nA;			nA.setMul(fEqnB.getComponent<3>(),	fEqnA);
								nA.subMul(fEqnA.getComponent<3>(),	fEqnB);

		for (int k = 0; k < numTestPlanes; k++)
		{
			const Plane& planeD = testPlanes[k];

			const hkVector4d fEqnC	= planeC.getApproxEquation();
			const hkVector4d fEqnD	= planeD.getApproxEquation();
			hkVector4d nCD;			nCD.setCross(fEqnC, fEqnD);
			hkVector4d nD;			nD.setMul(fEqnD.getComponent<3>(),	fEqnC);
									nD.subMul(fEqnC.getComponent<3>(),	fEqnD);

			// Compute determinant
			hkSimdDouble64 det;		det.setAdd(nAB.dot<3>(nD), nCD.dot<3>(nA));
									det.setAbs(det);

			// All should be zero. If not, definitely not collinear!
			HK_ON_DEBUG(m_numApproxCalls++);
			if ( det.greaterEqual(tol).anyIsSet() )
			{
				return false;
			}
		}
	}

	// Switch to exact arithmetic
	{
		hkInt64Vector4 iNrmA, iNrmB;
		hkSimdInt<128> iOffsetA, iOffsetB;
		planeA.getExactNormal(iNrmA);	planeA.getExactOffset(iOffsetA);
		planeB.getExactNormal(iNrmB);	planeB.getExactOffset(iOffsetB);
		hkInt128Vector4 nAB;	nAB.setCross(iNrmA, iNrmB);
		hkInt128Vector4 nA;		nA.setMul<3>(iOffsetB,	iNrmA);
								nA.subMul<3>(iOffsetA,	iNrmB);

		for (int k = 0; k < numTestPlanes; k++)
		{
			const Plane& planeD = testPlanes[k];

			hkInt64Vector4 iNrmC, iNrmD;
			hkSimdInt<128> iOffsetC, iOffsetD;
			planeC.getExactNormal(iNrmC);	planeC.getExactOffset(iOffsetC);
			planeD.getExactNormal(iNrmD);	planeD.getExactOffset(iOffsetD);
			hkInt128Vector4 nCD;	nCD.setCross(iNrmC, iNrmD);
			hkInt128Vector4 nD;		nD.setMul<3>(iOffsetD,	iNrmC);
									nD.subMul<3>(iOffsetC,	iNrmD);

			// Compute determinants
			hkSimdInt<256> detLeft		= nAB.dot<3>(nD);
			hkSimdInt<256> detRight;	detRight.setNeg(nCD.dot<3>(nA));
			HK_ON_DEBUG(m_numExactCalls++);
			if ( !detLeft.equal(detRight).anyIsSet() )
			{
				return false;
			}
		}
	}

	return true;
}

//
//	Computes an approximate direction for the edge resulting from the intersection of the two given planes

void HK_CALL hkcdPlanarGeometryPredicates::approximateEdgeDirection(PlaneParam planeA, PlaneParam planeB, hkIntVector& edgeDirectionOut)
{
	// Compute edge direction and maximum absolute component
	hkInt64Vector4 iNrmA;			planeA.getExactNormal(iNrmA);
	hkInt64Vector4 iNrmB;			planeB.getExactNormal(iNrmB);
	hkInt128Vector4 vEdgeDir;		vEdgeDir.setCross(iNrmA, iNrmB);
	hkInt128Vector4 vAbsDir;		vAbsDir.setAbs<3>(vEdgeDir);
	const int axisZ	= vAbsDir.getIndexOfMaxComponent<3>();

	// We want to divide the other two components by the maximum component so we get back two sub-unitary fractions.
	// We'll use the fractions to encode direction
	const int axisX = ((1 << axisZ) & 3);
	const int axisY = ((1 << axisX) & 3);

	// Get components
	hkSimdInt<128> dx = vAbsDir.getComponent(axisX);
	hkSimdInt<128> dy = vAbsDir.getComponent(axisY);
	hkSimdInt<128> dz = vAbsDir.getComponent(axisZ);

	// Reduce first by the greatest common divisor
	hkSimdInt<128> gcdxz;	gcdxz.setGreatestCommonDivisor(dx, dz);
	hkSimdInt<128> gcdyz;	gcdyz.setGreatestCommonDivisor(dy, dz);

	// Divide by the gcd, to get the irreducible directions
	hkInt128Vector4 vTop;	vTop.set(dx, dz, dy, dz);
	hkInt128Vector4 vBot;	vBot.set(gcdxz, gcdxz, gcdyz, gcdyz);
	vTop.setUnsignedDiv<3>(vTop, vBot);

	// We know the components of vEdgeDir are stored on 104 bits (including the sign bit), so we can shift them by 24 and get an approximate int value direction
	// We don't care about accuracy at this stage, as we are only required to return an approximate direction. The final tests will be done using the exact predicates anyway!
	const int numBitsShl = 128 - hkcdPlanarGeometryPrimitives::NumBitsPlanesIntersectionEdgeDir::NUM_BITS;
	dx.setShiftLeft<numBitsShl>(vTop.m_x);
	dy.setShiftLeft<numBitsShl>(vTop.m_z);
	dx.setUnsignedDiv(dx, vTop.m_y);
	dy.setUnsignedDiv(dy, vTop.m_w);

	hkIntVector vDir;
	vDir.setComponent(axisX, dx.getWord<0>());
	vDir.setComponent(axisY, dy.getWord<0>());
	vDir.setComponent(axisZ, (1 << numBitsShl));

	// Set proper signs. Since we divide by z, z should always end-up positive!
	hkVector4fComparison signsf;	signsf.setXor(vEdgeDir.lessZero(), vEdgeDir.getComponent(axisZ).lessZero());
#ifdef HK_REAL_IS_DOUBLE
	hkVector4Comparison signs;		signs.set(signsf.getMask());
#else
	const hkVector4Comparison signs	= signsf;
#endif

	edgeDirectionOut.setFlipSignS32(vDir, signs);
}

//
//	Returns true if the edge determined by the intersection of planes A and B is contained on plane C

hkBool32 HK_CALL hkcdPlanarGeometryPredicates::edgeOnPlane(PlaneParam edgePlaneA, PlaneParam edgePlaneB, PlaneParam planeC)
{
	// Planes A and B are assumed to intersect along the same line. For the line to lie on plane C, we need the rank of matrix
	//	| ax ay az aw |
	//	| bx by bz bw |
	//	| cx cy cz cw |
	// to be 2 < 3, so the determinants of all 3x3 minors must be zero

	// Compute using floating point first
	{
		hkVector4d fDets;	computeIntersectionDeterminants(edgePlaneA.getApproxEquation(), edgePlaneB.getApproxEquation(), planeC.getApproxEquation(), fDets);
							fDets.setAbs(fDets);
		hkVector4d fErr;	fErr.set(TOL_DET_3_WYZ, TOL_DET_3_XWZ, TOL_DET_3_XYW, TOL_DET_3_XYZ);
	
		HK_ON_DEBUG(m_numApproxCalls++);
		if ( fDets.greaterEqual(fErr).anyIsSet() )
		{
			// At least one of the minors is non-null, the edge is not on the plane
			return false;
		}
	}

	// We couldn't decide on the sign using floating point math, fall-back to fixed precision
	{
		const Plane planes[3] = { edgePlaneA, edgePlaneB, planeC };
		hkSimdInt<256> dets[4];
		computeIntersectionDeterminants(planes, &dets[0], &dets[1], &dets[2], &dets[3]);

		// Or all determinants together, we're only interested in all being zero
		hkSimdInt<256> d;	d.setOr(dets[0], dets[1]);
							d.setOr(d, dets[2]);
							d.setOr(d, dets[3]);

		return d.equalZero().anyIsSet();
	}
}

//
//	Computes the winding of the 3 given vertices w.r.t. the given triangle normal. The vertices are assumed to share the same support plane
//	Let a, b, c be the planes of a vertex P. We define:
//				| ax ay az |		    | aw ay az |			| ax aw az |			| ax ay aw |
//		dP =	| bx by bz |,	dPX = - | bw by bz |,	dPY = - | bx bw bz |,	dPZ = - | bx by bw |
//				| cx cy cz |			| cw cy cz |		    | cx cw cz |			| cx cy cw |
//
//	The coordinates of the point P are:
//		P = [dPX, dPY, dPZ] / detP
//	
//	Assume points are projected into the plane Z = 0, the coordinates of A, B, C are:
//		A = [dAX / dA, dAY / dA]
//		B = [dBX / dB, dBY / dB]
//		C = [dCX / dC, dCY / dC]

//	The edge BC is:
//		BC = [dCX / dC - dBX / dB, dCY / dC - dBY / dB]
//
//	And the outward normal nBC is:
//		nBC = [dBY / dB - dCY / dC, dCX / dC - dBX / dB]
//	We need to evaluate the equation of the 2D plane having normal nBC and passing through B, in A. If ABC is CCW, A should be outside BC, i.e. the eqn should be positive!
//		predicate	= sign[(A - B) nBC] = 
//					= sign[ (dAX / dA - dBX / dB) * (dBY / dB - dCY / dC) + (dAY / dA - dBY / dB) * (dCX / dC - dBX / dB) ]
//					= (dAX / dA - dBX / dB) * (dBY / dB - dCY / dC) > (dAY / dA - dBY / dB) * (dBX / dB - dCX / dC) ? 1 / 0 / -1
//
//	Let:
//		signAX = sign(dAX / dA - dBX / dB)
//		signAY = sign(dAY / dA - dBY / dB)
//		signBX = sign(dBX / dB - dCX / dC)
//		signBY = sign(dBY / dB - dCY / dC)
//
//	The predicate can be written:
//		predicate	= signAX * signBY * Abs[dAX / dA - dBX / dB] * Abs[dBY / dB - dCY / dC] > signAY * signBX * Abs[dAY / dA - dBY / dB] * Abs[dBX / dB - dCX / dC]
//					= (signAX * signBY) * Abs[dAX / dA - dBX / dB] / Abs[dAY / dA - dBY / dB] > (signAY * signBX) * Abs[dBX / dB - dCX / dC] / Abs[dBY / dB - dCY / dC]
//
//	So, we only need to compare the fractions:
//		Abs[(dB dAX - dA dBX) / (dB dAY - dA dBY)] > Abs[(dC dBX - dB dCX) / (dC dBY - dB dCY)]

hkcdPlanarGeometryPredicates::Winding HK_CALL hkcdPlanarGeometryPredicates::computeExactTriangleWinding(const Plane (&planesA)[3], const Plane (&planesB)[3], const Plane (&planesC)[3], const Plane& supportPlane, int axisX, int axisY, int axisZ)
{
	// Compute all determinants first
	hkSimdInt<256> dAX, dAY, dA;
	hkSimdInt<256> dBX, dBY, dB;
	hkSimdInt<256> dCX, dCY, dC;
	{
		hkSimdInt<256>* dets[3] = { HK_NULL };
		dets[axisX] = &dAX;	dets[axisY] = &dAY;		computeIntersectionDeterminants(planesA, dets[0], dets[1], dets[2], &dA);
		dets[axisX] = &dBX;	dets[axisY] = &dBY;		computeIntersectionDeterminants(planesB, dets[0], dets[1], dets[2], &dB);
		dets[axisX] = &dCX;	dets[axisY] = &dCY;		computeIntersectionDeterminants(planesC, dets[0], dets[1], dets[2], &dC);
	}

	// Compute the signs:
	//		signAX = sign(dAX / dA - dBX / dB)
	//		signAY = sign(dAY / dA - dBY / dB)
	//		signBX = sign(dBX / dB - dCX / dC)
	//		signBY = sign(dBY / dB - dCY / dC)
	const int signAX = hkSimdInt<256>::compareFractions(dAX, dA, dBX, dB);	// (Ax / A) > (Bx / B) ? +1 / 0 / -1
	const int signAY = hkSimdInt<256>::compareFractions(dAY, dA, dBY, dB);	// (Ay / A) > (By / B) ? +1 / 0 / -1
	const int signBX = hkSimdInt<256>::compareFractions(dBX, dB, dCX, dC);	// (Bx / B) > (Cx / C) ? +1 / 0 / -1
	const int signBY = hkSimdInt<256>::compareFractions(dBY, dB, dCY, dC);	// (By / B) > (Cy / C) ? +1 / 0 / -1
	
	// Compute left and right signs
	const int signLeft	= signAX * signBY;	// The sign of the predicate's Lhs
	const int signRight	= signAY * signBX;	// The sign of the predicate's Rhs

	// Analyse the signs before starting to evaluate the remainder of the predicate
	int flipResult = 0;
	{
		const int signR_EqZero		= 1 - (signRight & 1);		HK_ASSERT(0x5270855d, (signR_EqZero && !signRight) || (!signR_EqZero && signRight));
		const int signL_EqZero		= 1 - (signLeft & 1);		HK_ASSERT(0x7bb59b7a, (signR_EqZero && !signRight) || (!signR_EqZero && signRight));
		const int signR_LessZero	= (signRight >> 1) & 1;		HK_ASSERT(0x7577fede, (signR_LessZero && (signRight < 0)) || (!signR_LessZero && (signRight >= 0)));
		const int signL_LessZero	= (signLeft >> 1) & 1;		HK_ASSERT(0x476456fd, (signL_LessZero && (signLeft < 0)) || (!signL_LessZero && (signLeft >= 0)));
		const int signsMask			= (signR_EqZero << 3) | (signR_LessZero << 2) | (signL_EqZero << 1) | signL_LessZero;
		const int retCode			= (0x5252AE03 >> (signsMask << 1)) & 3;
		hkInt64Vector4 iNrm;		supportPlane.getExactNormal(iNrm);
		flipResult					= (iNrm.getComponent(axisZ) >> 63L) & 1;	// (nZ < 0) ? 1 : 0

		if ( retCode != 3 )
		{
			// Flip the winding based on the sign
			const Winding wind	= (Winding)(retCode - 1);
			return flipResult ? (Winding)-wind : wind;
		}

		HK_ASSERT(0xb126644, (signsMask == 0) || (signsMask == 5));
		flipResult = (signsMask == 5) ? (1 - flipResult) : flipResult;
	}

	// Compute the fraction in the predicate
	{
		// Abs[(dB dAX - dA dBX) / (dB dAY - dA dBY)] > Abs[(dC dBX - dB dCX) / (dC dBY - dB dCY)]
		hkSimdInt<512> topL;	topL.setMul(dB, dAX);	topL.subMul(dA, dBX);	topL.setAbs(topL);	// (B Ax - A Bx)
		hkSimdInt<512> botL;	botL.setMul(dB, dAY);	botL.subMul(dA, dBY);	botL.setAbs(botL);	// (B Ay - A By)
		hkSimdInt<512> topR;	topR.setMul(dC, dBX);	topR.subMul(dB, dCX);	topR.setAbs(topR);	// (C Bx - B Cx)
		hkSimdInt<512> botR;	botR.setMul(dC, dBY);	botR.subMul(dB, dCY);	botR.setAbs(botR);	// (C By - B Cy)

 		const Winding wind = (Winding)hkSimdInt<512>::compareFractions(topL, botL, topR, botR);
 		return flipResult ? (Winding)-wind : wind;
	}
}

namespace hkcdPlanarGeometryPredicatesImpl
{
	//
	//	Compares (a/b) w.r.t. (c/d). Only returns a valid result if the fractions are sufficiently far apart

	static HK_FORCE_INLINE hkBool32 HK_CALL approxCompareFractions(hkSimdDouble64Parameter origA, hkSimdDouble64Parameter origB, hkSimdDouble64Parameter origC, hkSimdDouble64Parameter origD, hkSimdDouble64Parameter errAC, hkSimdDouble64Parameter errBD, int& cmpOut)
	{
		hkVector4dComparison pmmp;	pmmp.set<hkVector4Comparison::MASK_YZ>();
		hkVector4d vErrAC;			vErrAC.setAll(errAC);						// [d, d, d, d]
									vErrAC.setFlipSign(vErrAC, pmmp);			// [d, -d, -d, d]
		hkVector4dComparison mppm;	mppm.set<hkVector4Comparison::MASK_XW>();
		hkVector4d vErrBD;			vErrBD.setAll(errBD);						// [e, e, e, e]
									vErrBD.setFlipSign(vErrBD, mppm);			// [-e, e, -e, e]
		const hkSimdDouble64 inv	= hkSimdDouble64::fromFloat(INV_2_POW_24);

		// Estimate the interval of (a/b) = [fMin, fMax]
		hkSimdDouble64 fMin, fMax;
		{
			hkVector4d vA;		vA.setAll(origA);	vA.add(vErrAC);
			hkVector4d vB;		vB.setAll(origB);	vB.add(vErrBD);
			hkVector4d vAB;		vAB.setDiv(vA, vB);
			hkVector4d vMax;	vMax = hkVector4d::getConstant<HK_QUADREAL_MAX>();
								vMax.setFlipSign(vMax, vA.lessZero());
								vAB.setSelect(vB.equalZero(), vMax, vAB);

			fMin = vAB.horizontalMin<4>();
			fMin.subMul(fMin, inv);		// Expand by the last bit of the mantissa
			fMax = vAB.horizontalMax<4>();
			fMax.addMul(fMax, inv);
		}

		// Estimate the interval of (c/d) = [gMin, gMax]
		hkSimdDouble64 gMin, gMax;
		{
			hkVector4d vC;		vC.setAll(origC);	vC.add(vErrAC);
			hkVector4d vD;		vD.setAll(origD);	vD.add(vErrBD);
			hkVector4d vCD;		vCD.setDiv(vC, vD);
			hkVector4d vMax;	vMax = hkVector4d::getConstant<HK_QUADREAL_MAX>();
								vMax.setFlipSign(vMax, vC.lessZero());
								vCD.setSelect(vD.equalZero(), vMax, vCD);

			gMin = vCD.horizontalMin<4>();
			gMin.subMul(gMin, inv);
			gMax = vCD.horizontalMax<4>();
			gMax.addMul(gMax, inv);
		}

		if ( fMin.greater(gMax).anyIsSet() )	{	cmpOut = 1;		return true;	}
		if ( gMin.greater(fMax).anyIsSet() )	{	cmpOut = -1;	return true;	}
		return false;
	}

	static HK_FORCE_INLINE int HK_CALL compareFinalFractions_exactWinding(const hkcdPlanarGeometryPredicates::Plane (&planesA)[3], const hkcdPlanarGeometryPredicates::Plane (&planesB)[3], const hkcdPlanarGeometryPredicates::Plane (&planesC)[3], int axisX, int axisY)
	{
		// Compute all determinants first
		hkSimdInt<256> dAX, dAY, dA;
		hkSimdInt<256> dBX, dBY, dB;
		hkSimdInt<256> dCX, dCY, dC;
		{
			hkSimdInt<256>* dets[3] = { HK_NULL };
			dets[axisX] = &dAX;	dets[axisY] = &dAY;		hkcdPlanarGeometryPredicates::computeIntersectionDeterminants(planesA, dets[0], dets[1], dets[2], &dA);
			dets[axisX] = &dBX;	dets[axisY] = &dBY;		hkcdPlanarGeometryPredicates::computeIntersectionDeterminants(planesB, dets[0], dets[1], dets[2], &dB);
			dets[axisX] = &dCX;	dets[axisY] = &dCY;		hkcdPlanarGeometryPredicates::computeIntersectionDeterminants(planesC, dets[0], dets[1], dets[2], &dC);
		}

		// Compute the fraction in the predicate
		{
			// Abs[(dB dAX - dA dBX) / (dB dAY - dA dBY)] > Abs[(dC dBX - dB dCX) / (dC dBY - dB dCY)]
			hkSimdInt<512> topL;	topL.setMul(dB, dAX);	topL.subMul(dA, dBX);	topL.setAbs(topL);	// (B Ax - A Bx)
			hkSimdInt<512> botL;	botL.setMul(dB, dAY);	botL.subMul(dA, dBY);	botL.setAbs(botL);	// (B Ay - A By)
			hkSimdInt<512> topR;	topR.setMul(dC, dBX);	topR.subMul(dB, dCX);	topR.setAbs(topR);	// (C Bx - B Cx)
			hkSimdInt<512> botR;	botR.setMul(dC, dBY);	botR.subMul(dB, dCY);	botR.setAbs(botR);	// (C By - B Cy)

			return hkSimdInt<512>::compareFractions(topL, botL, topR, botR);
		}
	}
}

//
//	Estimates the winding of the 3 given vertices w.r.t. the given triangle normal (floating-point).

hkcdPlanarGeometryPredicates::Winding HK_CALL hkcdPlanarGeometryPredicates::estimateTriangleWinding(const Plane (&planesA)[3], const Plane (&planesB)[3], const Plane (&planesC)[3], const Plane& supportPlane, int axisX, int axisY, int axisZ)
{	
	// Compute determinants
	hkVector4d detsX, detsY, detsW;
	{
		hkVector4d dets[3];
		computeIntersectionDeterminants(planesA[0].getApproxEquation(), planesA[1].getApproxEquation(), planesA[2].getApproxEquation(), dets[0]);	// [dAX, dAY, dAZ, dA]
		computeIntersectionDeterminants(planesB[0].getApproxEquation(), planesB[1].getApproxEquation(), planesB[2].getApproxEquation(), dets[1]);	// [dBX, dBY, dBZ, dB]
		computeIntersectionDeterminants(planesC[0].getApproxEquation(), planesC[1].getApproxEquation(), planesC[2].getApproxEquation(), dets[2]);	// [dCX, dCY, dCZ, dC]

		// Transpose
		detsW.setZero();
		HK_TRANSPOSE4d(dets[0], dets[1], dets[2], detsW);	// detsW = [dA, dB, dC, 0]
		detsX = dets[axisX];									// detsX = [dAX, dBX, dCX, 0]
		detsY = dets[axisY];									// detsY = [dAY, dBY, dCY, 0]

		// If we can't decide on the signs of dA, dB, dC, abandon!
		hkVector4d temp;		temp.setAbs(detsW);
		hkVector4d vTol;		vTol.setAll(TOL_DET_3_XYZ);
		if ( temp.less(vTol).horizontalOr<3>().anyIsSet() )
		{
			return WINDING_UNKNOWN;
		}
	}

	// Compute the signs:
	//		signAX = sign(dAX / dA - dBX / dB)
	//		signAY = sign(dAY / dA - dBY / dB)
	//		signBX = sign(dBX / dB - dCX / dC)
	//		signBY = sign(dBY / dB - dCY / dC)
	int signAX, signAY, signBX, signBY;
	const hkSimdDouble64 tolAC	= hkSimdDouble64::fromFloat(TOL_DET_3_XYW);
	const hkSimdDouble64 tolBD	= hkSimdDouble64::fromFloat(TOL_DET_3_XYZ);
	const hkSimdDouble64 dAX	= detsX.getComponent<0>(), dBX	= detsX.getComponent<1>(),	dCX	= detsX.getComponent<2>();
	const hkSimdDouble64 dAY	= detsY.getComponent<0>(), dBY	= detsY.getComponent<1>(),	dCY	= detsY.getComponent<2>();
	const hkSimdDouble64 dA		= detsW.getComponent<0>(), dB	= detsW.getComponent<1>(),	dC	= detsW.getComponent<2>();
		
	if (	!hkcdPlanarGeometryPredicatesImpl::approxCompareFractions(dAX, dA, dBX, dB, tolAC, tolBD, signAX) ||	// (Ax / A) > (Bx / B) ? +1 / 0 / -1
			!hkcdPlanarGeometryPredicatesImpl::approxCompareFractions(dAY, dA, dBY, dB, tolAC, tolBD, signAY) ||	// (Ay / A) > (By / B) ? +1 / 0 / -1
			!hkcdPlanarGeometryPredicatesImpl::approxCompareFractions(dBX, dB, dCX, dC, tolAC, tolBD, signBX) ||	// (Bx / B) > (Cx / C) ? +1 / 0 / -1
			!hkcdPlanarGeometryPredicatesImpl::approxCompareFractions(dBY, dB, dCY, dC, tolAC, tolBD, signBY) )		// (By / B) > (Cy / C) ? +1 / 0 / -1
	{
		return WINDING_UNKNOWN;
	}

	// Compute left and right signs
	const int signLeft	= signAX * signBY;	// The sign of the predicate's Lhs
	const int signRight	= signAY * signBX;	// The sign of the predicate's Rhs

	// Analyze the signs before starting to evaluate the remainder of the predicate
	int flipResult = 0;
	{
		const int signR_EqZero		= 1 - (signRight & 1);		HK_ASSERT(0x621737fa, (signR_EqZero && !signRight) || (!signR_EqZero && signRight));
		const int signL_EqZero		= 1 - (signLeft & 1);		HK_ASSERT(0x7cf9fa71, (signR_EqZero && !signRight) || (!signR_EqZero && signRight));
		const int signR_LessZero	= (signRight >> 1) & 1;		HK_ASSERT(0x71e69b5e, (signR_LessZero && (signRight < 0)) || (!signR_LessZero && (signRight >= 0)));
		const int signL_LessZero	= (signLeft >> 1) & 1;		HK_ASSERT(0x31e6685f, (signL_LessZero && (signLeft < 0)) || (!signL_LessZero && (signLeft >= 0)));
		const int signsMask			= (signR_EqZero << 3) | (signR_LessZero << 2) | (signL_EqZero << 1) | signL_LessZero;
		const int retCode			= (0x5252AE03 >> (signsMask << 1)) & 3;
		hkInt64Vector4 iNrm;		supportPlane.getExactNormal(iNrm);
		flipResult					= (iNrm.getComponent(axisZ) >> 63L) & 1;	// (nZ < 0) ? 1 : 0

		if ( retCode != 3 )
		{
			// Flip the winding based on the sign
			const Winding wind	= (Winding)(retCode - 1);
			return flipResult ? (Winding)-wind : wind;
		}

		HK_ASSERT(0x1c7cd802, (signsMask == 0) || (signsMask == 5));
		flipResult = (signsMask == 5) ? (1 - flipResult) : flipResult;
	}

	// Compute the fraction in the predicate
	{
		// Abs[(dB dAX - dA dBX) / (dB dAY - dA dBY)] > Abs[(dC dBX - dB dCX) / (dC dBY - dB dCY)]
		hkSimdDouble64 topL;	topL.setMul(dB, dAX);	topL.subMul(dA, dBX);	topL.setAbs(topL);	// (B Ax - A Bx)
		hkSimdDouble64 botL;	botL.setMul(dB, dAY);	botL.subMul(dA, dBY);	botL.setAbs(botL);	// (B Ay - A By)
		hkSimdDouble64 topR;	topR.setMul(dC, dBX);	topR.subMul(dB, dCX);	topR.setAbs(topR);	// (C Bx - B Cx)
		hkSimdDouble64 botR;	botR.setMul(dC, dBY);	botR.subMul(dB, dCY);	botR.setAbs(botR);	// (C By - B Cy)

		int iWind;
		const hkSimdDouble64 tol = hkSimdDouble64::fromFloat(457.0f);
		if ( !hkcdPlanarGeometryPredicatesImpl::approxCompareFractions(topL, botL, topR, botR, tol, tol, iWind) )
		{
			iWind = hkcdPlanarGeometryPredicatesImpl::compareFinalFractions_exactWinding(planesA, planesB, planesC, axisX, axisY);
		}
		const Winding wind = (Winding)iWind;
		return flipResult ? (Winding)-wind : wind;
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
