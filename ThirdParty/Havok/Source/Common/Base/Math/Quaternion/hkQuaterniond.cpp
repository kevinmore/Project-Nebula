/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>

void hkQuaterniond::setAxisAngle(hkVector4dParameter axis, hkSimdDouble64Parameter angle)
{
	HK_ON_DEBUG(hkSimdDouble64 al; al.setAbs( axis.length<3>() - hkSimdDouble64_1 ); )
	HK_MATH_ASSERT(0x34bd3b6e, al.getReal() < hkDouble64(0.01f), "Axis is not normalized in hkQuaterniond::setAxisAngle()");
	const hkSimdDouble64 halfAngle = hkSimdDouble64_Half * angle;
	hkSimdDouble64 s,c;
	hkVector4dUtil::sinCos(halfAngle,s,c);
	hkVector4d q; q.setMul(axis, s);
	m_vec.setXYZ_W(q, c);
}

void hkQuaterniond::setAxisAngle(hkVector4dParameter axis, hkDouble64 angle)
{
	setAxisAngle(axis,hkSimdDouble64::fromFloat(angle));
}

void hkQuaterniond::setAxisAngle_Approximate(hkVector4dParameter axis, hkSimdDouble64Parameter angle)
{
	HK_ON_DEBUG(hkSimdDouble64 al; al.setAbs( axis.length<3>() - hkSimdDouble64_1 ); )
	HK_MATH_ASSERT(0x34bd3b6e, al.getReal() < hkDouble64(0.01f), "Axis is not normalized in hkQuaterniond::setAxisAngle()");
	const hkSimdDouble64 halfAngle = hkSimdDouble64_Half * angle;
	hkSimdDouble64 s,c;
	hkVector4dUtil::sinCosApproximation(halfAngle,s,c);
	hkVector4d q; q.setMul(axis, s);
	m_vec.setXYZ_W(q, c);
}

void hkQuaterniond::setFromEulerAngles(hkSimdDouble64Parameter roll, hkSimdDouble64Parameter pitch, hkSimdDouble64Parameter yaw) 
{
#if defined(HK_PLATFORM_WIN32) && (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
	hkVector4d pprr; pprr.set(pitch,pitch,roll,roll);
	hkVector4d pprr2; pprr2.setMul(pprr, hkVector4d::getConstant<HK_QUADREAL_INV_2>());
	hkVector4d sc_pprr; hkVector4dUtil::sinCos(pprr2,sc_pprr);  // sin(p)cos(p)sin(r)cos(r)

	hkVector4d ccss_roll; ccss_roll.setPermutation<hkVectorPermutation::WWZZ>(sc_pprr);
	hkVector4d cscs_pitch; cscs_pitch.setPermutation<hkVectorPermutation::YXYX>(sc_pprr);
	hkVector4d m; m.setMul(ccss_roll, cscs_pitch); // c1c2, c1s2, s1c2, s1s2

	hkSimdDouble64 sin_y,cos_y; hkVector4dUtil::sinCos(yaw * hkSimdDouble64_Half,sin_y,cos_y); // sin(y)cos(y)
	hkVector4d n; n.setMul(m, sin_y); // c1c2s3, c1s2s3,  s1c2s3,  s1s2s3
	hkVector4dComparison mask; mask.set<hkVector4ComparisonMask::MASK_ZW>();
	n.setFlipSign(n, mask);          // c1c2s3, c1s2s3, -s1c2s3, -s1s2s3

	hkVector4d o; o.setMul(m, cos_y); // c1c2c3, c1s2c3, s1c2c3, s1s2c3
	hkVector4d s; s.setPermutation<hkVectorPermutation::WZYX>(o); // s1s2c3, s1c2c3, c1s2c3, c1c2c3

	m_vec.setAdd(n,s); // c1c2s3+s1s2c3, c1s2s3+s1c2c3, -s1c2s3+c1s2c3, -s1s2s3+c1c2c3
#else
	hkDouble64 roll2  = roll.getReal() * 0.5f;
	hkDouble64 pitch2 = pitch.getReal() * 0.5f;
	hkDouble64 yaw2   = yaw.getReal() * 0.5f;

	hkDouble64 c1 = hkMath::cos(roll2);
	hkDouble64 s1 = hkMath::sin(roll2);
	hkDouble64 c2 = hkMath::cos(pitch2);
	hkDouble64 s2 = hkMath::sin(pitch2);
	hkDouble64 c3 = hkMath::cos(yaw2);
	hkDouble64 s3 = hkMath::sin(yaw2);

	hkDouble64 c1c2 = c1*c2;
	hkDouble64 c1s2 = c1*s2;
	hkDouble64 s1s2 = s1*s2;
	hkDouble64 s1c2 = s1*c2;

	m_vec(0) = c1c2*s3 + s1s2*c3;
	m_vec(1) = s1c2*c3 + c1s2*s3;
	m_vec(2) = c1s2*c3 - s1c2*s3;
	m_vec(3) = c1c2*c3 - s1s2*s3;
#endif
}

void hkQuaterniond::setFromEulerAngles(hkDouble64 roll, hkDouble64 pitch, hkDouble64 yaw) 
{
	setFromEulerAngles(hkSimdDouble64::fromFloat(roll),hkSimdDouble64::fromFloat(pitch),hkSimdDouble64::fromFloat(yaw));
}

void hkQuaterniond::setFromEulerAngles_Approximate(hkSimdDouble64Parameter roll, hkSimdDouble64Parameter pitch, hkSimdDouble64Parameter yaw) 
{
#if defined(HK_PLATFORM_WIN32) && (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
	hkVector4d pprr; pprr.set(pitch,pitch,roll,roll);
	hkVector4d pprr2; pprr2.setMul(pprr, hkVector4d::getConstant<HK_QUADREAL_INV_2>());
	hkVector4d yy2; yy2.setAll(yaw * hkSimdDouble64_Half);

	hkVector4d sc_yy2;   hkVector4dUtil::sinCosApproximation(yy2,sc_yy2);     // sin(y)cos(y)
	hkVector4d sc_pprr2; hkVector4dUtil::sinCosApproximation(pprr2,sc_pprr2); // sin(p)cos(p)sin(r)cos(r)

	hkVector4d ccss_roll; ccss_roll.setPermutation<hkVectorPermutation::WWZZ>(sc_pprr2);
	hkVector4d cscs_pitch; cscs_pitch.setPermutation<hkVectorPermutation::YXYX>(sc_pprr2);
	hkVector4d m; m.setMul(ccss_roll, cscs_pitch); // c1c2, c1s2, s1c2, s1s2

	const hkSimdDouble64 sin_y = sc_yy2.getComponent<0>();
	hkVector4d n; n.setMul(m, sin_y); // c1c2s3, c1s2s3,  s1c2s3,  s1s2s3
	hkVector4dComparison mask; mask.set<hkVector4ComparisonMask::MASK_ZW>();
	n.setFlipSign(n, mask);          // c1c2s3, c1s2s3, -s1c2s3, -s1s2s3

	const hkSimdDouble64 cos_y = sc_yy2.getComponent<1>();
	hkVector4d o; o.setMul(m, cos_y); // c1c2c3, c1s2c3, s1c2c3, s1s2c3

	hkVector4d s; s.setPermutation<hkVectorPermutation::WZXY>(o); // s1s2c3, s1c2c3, c1c2c3, c1s2c3
	m_vec.setAdd(n,s); // c1c2s3+s1s2c3, c1s2s3+s1c2c3, -s1c2s3+c1c2c3, -s1s2s3+c1s2c3
#else
	setFromEulerAngles(roll, pitch, yaw);
#endif
}



hkBool32 hkQuaterniond::isOk(const hkDouble64 epsilon) const
{
	hkBool32 ok = m_vec.isOk<4>();
	const hkSimdDouble64 error = m_vec.lengthSquared<4>() - hkSimdDouble64::getConstant<HK_QUADREAL_1>();
	hkSimdDouble64 absErr; absErr.setAbs(error);
	hkSimdDouble64 tol; tol.setFromFloat(epsilon);
	return ok && absErr.isLess(tol);
}

void hkQuaterniond::set(const hkRotationd& r)
{
	_set(r);
}

void hkQuaterniond::setFlippedRotation(hkVector4dParameter from)
{
	hkVector4d vec;
	hkVector4dUtil::calculatePerpendicularVector(from, vec);
	vec.normalize<3>();
	vec.zeroComponent<3>();
	m_vec = vec;
}

//
//	Sets this = Slerp(q0, q1, t)
#define HK_ONE_MINUS_QUATERNIONd_DELTA hkDouble64(1.0f - 1e-3f)

HK_FORCE_INLINE void hkQuaterniond::setSlerp(hkQuaterniondParameter q0, hkQuaterniondParameter q1, hkSimdDouble64Parameter t)
#if defined(HK_COMPILER_GHS)
	
	
{
	hkDouble64 tReal = t.getReal();
	hkDouble64 qdelta = HK_ONE_MINUS_QUATERNIONd_DELTA;

	hkDouble64 oldCosTheta = q0.m_vec.dot<4>(q1.m_vec).getReal();
	hkDouble64 cosTheta = __FSEL(oldCosTheta, oldCosTheta, -oldCosTheta);

	hkDouble64 t0,t1;

	if (cosTheta < qdelta)
	{
		hkDouble64 theta = hkMath::acos(cosTheta);
		// use sqrtInv(1+c^2) instead of 1.0/sin(theta) 
		const hkDouble64 iSinTheta = hkMath::sqrtInverse( 1.0f - (cosTheta*cosTheta) );
		const hkDouble64 tTheta = tReal * theta;

		hkDouble64 s0 = hkMath::sin(theta-tTheta);
		hkDouble64 s1 = hkMath::sin(tTheta);

		t0 = s0 * iSinTheta;
		t1 = s1 * iSinTheta;
	}
	else
	{
		// If q0 is nearly the same as q1 we just linearly interpolate
		t0 = 1.0f - tReal;
		t1 = tReal;
	}	

	t1 = __FSEL(oldCosTheta, t1, -t1);

	hkVector4d slerp;
	hkSimdDouble64 t0Simd; t0Simd.setFromFloat(t0);
	hkSimdDouble64 t1Simd; t1Simd.setFromFloat(t1);
	slerp.setMul( t0Simd, q0.m_vec);
	slerp.addMul( t1Simd, q1.m_vec);
	slerp.normalize<4>();

	m_vec = slerp;
}
#else
{
	const hkSimdDouble64 one = hkSimdDouble64::getConstant<HK_QUADREAL_1>();
	hkSimdDouble64 qdelta; qdelta.setFromFloat(HK_ONE_MINUS_QUATERNIONd_DELTA);

	hkSimdDouble64 cosTheta = q0.m_vec.dot<4>(q1.m_vec);

	// If B is on the opposite hemisphere use -B instead of B
	const hkVector4dComparison cosThetaLessZero = cosTheta.lessZero();
	cosTheta.setFlipSign(cosTheta, cosThetaLessZero);

	hkSimdDouble64 t0,t1;

	if (cosTheta < qdelta)
	{
		hkSimdDouble64 theta; 
#if defined(HK_PLATFORM_WIN32) && (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED) && !defined(HK_ARCH_ARM)
		theta.m_real = hkMath::twoAcos(cosTheta.m_real);
#else
		theta.setFromFloat( hkMath::acos(cosTheta.getReal()) );
#endif

		// use sqrtInv(1+c^2) instead of 1.0/sin(theta) 
		const hkSimdDouble64 iSinTheta = ( one - (cosTheta*cosTheta) ).sqrtInverse<HK_ACC_23_BIT,HK_SQRT_IGNORE>();
		const hkSimdDouble64 tTheta = t * theta;

		hkSimdDouble64 s0,s1; 
#if defined(HK_PLATFORM_WIN32) && (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED) && !defined(HK_ARCH_ARM)
		hkVector4d ss; ss.set(theta-tTheta, tTheta, tTheta, tTheta);
		hkVector4d s01; 
		s01.m_quad.xy = hkMath::twoSin(ss.m_quad.xy);
		s0 = s01.getComponent<0>();
		s1 = s01.getComponent<1>();
#else
		s0.setFromFloat(hkMath::sin((theta-tTheta).getReal()));
		s1.setFromFloat(hkMath::sin(tTheta.getReal()));
#endif
		t0 = s0 * iSinTheta;
		t1 = s1 * iSinTheta;
	}
	else
	{
		// If q0 is nearly the same as q1 we just linearly interpolate
		t0 = one - t;
		t1 = t;
	}	

	t1.setFlipSign(t1, cosThetaLessZero);

	hkVector4d slerp;
	slerp.setMul( t0, q0.m_vec);
	slerp.addMul( t1, q1.m_vec);
	slerp.normalize<4>();

	m_vec = slerp;
}
#undef HK_ONE_MINUS_QUATERNIONd_DELTA
#endif

//
//	Calculates the barycentric interpolation between q0, q1, q2. The barycentric coordinates are l0, l1, l2, with l0 omitted (not used).

void hkQuaterniond::setBarycentric(const hkQuaterniond* HK_RESTRICT qVerts, hkVector4dParameter vLambda)
{
	// Triangle ABC has barycentric coords (l0, l1, l2)
	// Triangle BCA has barycentric coords (l1, l2, l0)
	// Triangle CAB has barycentric coords (l2, l0, l1)
	// We need to compute (l1 + l2, l2 + l0, l0 + l1) and pick the biggest absolute value
	// in order to get a stable inverse result
	hkVector4d l_120;	l_120.setPermutation<hkVectorPermutation::YZXW>(vLambda);	// [l1, l2, l0]
	hkVector4d l_201;	l_201.setPermutation<hkVectorPermutation::ZXYW>(vLambda);	// [l2, l0, l1]
	hkVector4d l;		l.setAdd(l_120, l_201);										// [l1 + l2, l2 + l0, l0 + l1]

	// Compute the most stable vertex permutation
	const int idxA		= l.getIndexOfMaxAbsComponent<3>();							// Either 0, 1, or 2, index of the largest absolute component in l
	const int idxB		= (1 << idxA) & 3;
	const int idxC		= (1 << idxB) & 3;
	HK_ASSERT(0x4890e973, (idxA >= 0) && (idxA <= 2) && (idxB >= 0) && (idxB <= 2) && (idxC >= 0) && (idxC <= 2));

	// Interpolate
	const hkSimdDouble64 l12	= l.getComponent(idxA);
	const hkSimdDouble64 l2		= l_201.getComponent(idxA);
	hkSimdDouble64 absL12;		absL12.setAbs(l12);
	hkSimdDouble64 invL12;		invL12.setReciprocal(l12);
	hkSimdDouble64 u;			u.setZero();
	u.setSelect(absL12.greater(hkSimdDouble64_Eps), l2 * invL12, u);

	const hkQuaterniond qA = qVerts[idxA];
	const hkQuaterniond qB = qVerts[idxB];
	const hkQuaterniond qC = qVerts[idxC];
	hkQuaterniond q0, q1;
	q0.setClosest(qA, qB);	q0.setSlerp(q0, qB, l12);
	q1.setClosest(qA, qC);	q1.setSlerp(q1, qC, l12);
	q0.setClosest(q0, q1);	setSlerp(q0, q1, u);
}

void hkQuaterniond::removeAxisComponent (hkVector4dParameter axis)
{
	// Rotate the desired axis 
	hkVector4d rotatedAxis;
	rotatedAxis._setRotatedDir(*this, axis);

	// Calculate the shortest rotation that would bring align both axis
	// Now, calculate the rotation required to reach that alignment
	// This is the component of the rotation perpendicular to the axis

	const hkSimdDouble64 dotProd = axis.dot<3>(rotatedAxis);
	const hkSimdDouble64 one = hkSimdDouble64::getConstant<HK_QUADREAL_1>();
	hkSimdDouble64 eps; eps.setFromFloat(hkDouble64(1e-3f));

	// Parallel
	if ( (dotProd-one) > -eps )
	{
		setIdentity();
		return;
	}

	// Opposite
	if ( (dotProd+one) < eps )
	{
		hkVector4d perpVector;
		hkVector4dUtil::calculatePerpendicularVector(axis, perpVector);
		perpVector.normalize<3>();

		m_vec = perpVector;       // axis * sin(PI/2)
		m_vec.zeroComponent<3>(); // cos(PI/2)
		return;
	}

	// else
	{
		hkSimdDouble64 rotationAngle;
#if defined(HK_PLATFORM_WIN32) && (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED) && !defined(HK_ARCH_ARM)
		rotationAngle.m_real = hkMath::twoAcos(dotProd.m_real);
#else
		rotationAngle.setFromFloat( hkMath::acos(dotProd.getReal()) );
#endif
		hkVector4d rotationAxis;
		rotationAxis.setCross(axis, rotatedAxis);
		rotationAxis.normalize<3>();

		setAxisAngle(rotationAxis, rotationAngle);
	}
}

void hkQuaterniond::decomposeRestAxis(hkVector4dParameter axis, hkQuaterniond& restOut, hkSimdDouble64& angleOut) const
{
	hkQuaterniond axisRot;
	{
		restOut = *this;
		restOut.removeAxisComponent(axis);

		// axisRot = inv(rest) * q
		axisRot.setInverseMul(restOut, *this);
		axisRot.normalize(); // workaround COM-2170 to avoid getAngle returning a NaN in some cases.
	}

	hkSimdDouble64 a; a.setFromFloat(axisRot.getAngle());

	const hkSimdDouble64 reverse = axisRot.getRealPart() * axisRot.getImag().dot<3>(axis);
	angleOut.setFlipSign(a, reverse);
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
