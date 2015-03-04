/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>

void hkQuaternionf::setAxisAngle(hkVector4fParameter axis, hkSimdFloat32Parameter angle)
{
	HK_ON_DEBUG(hkSimdFloat32 al; al.setAbs( axis.length<3>() - hkSimdFloat32_1 ); )
	HK_MATH_ASSERT(0x34bd3b6e, al.getReal() < hkFloat32(0.01f), "Axis is not normalized in hkQuaternionf::setAxisAngle()");
	const hkSimdFloat32 halfAngle = hkSimdFloat32_Half * angle;
	hkSimdFloat32 s,c;
	hkVector4fUtil::sinCos(halfAngle,s,c);
	hkVector4f q; q.setMul(axis, s);
	m_vec.setXYZ_W(q, c);
}

void hkQuaternionf::setAxisAngle(hkVector4fParameter axis, hkFloat32 angle)
{
	setAxisAngle(axis,hkSimdFloat32::fromFloat(angle));
}

void hkQuaternionf::setAxisAngle_Approximate(hkVector4fParameter axis, hkSimdFloat32Parameter angle)
{
	HK_ON_DEBUG(hkSimdFloat32 al; al.setAbs( axis.length<3>() - hkSimdFloat32_1 ); )
	HK_MATH_ASSERT(0x34bd3b6e, al.getReal() < hkFloat32(0.01f), "Axis is not normalized in hkQuaternionf::setAxisAngle()");
	const hkSimdFloat32 halfAngle = hkSimdFloat32_Half * angle;
	hkSimdFloat32 s,c;
	hkVector4fUtil::sinCosApproximation(halfAngle,s,c);
	hkVector4f q; q.setMul(axis, s);
	m_vec.setXYZ_W(q, c);
}

void hkQuaternionf::setFromEulerAngles(hkSimdFloat32Parameter roll, hkSimdFloat32Parameter pitch, hkSimdFloat32Parameter yaw) 
{
#if defined(HK_PLATFORM_WIN32) && (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
	hkVector4f pprr; pprr.set(pitch,pitch,roll,roll);
	hkVector4f pprr2; pprr2.setMul(pprr, hkVector4f::getConstant<HK_QUADREAL_INV_2>());
	hkVector4f sc_pprr; hkVector4fUtil::sinCos(pprr2,sc_pprr);  // sin(p)cos(p)sin(r)cos(r)

	hkVector4f ccss_roll; ccss_roll.setPermutation<hkVectorPermutation::WWZZ>(sc_pprr);
	hkVector4f cscs_pitch; cscs_pitch.setPermutation<hkVectorPermutation::YXYX>(sc_pprr);
	hkVector4f m; m.setMul(ccss_roll, cscs_pitch); // c1c2, c1s2, s1c2, s1s2

	hkSimdFloat32 sin_y,cos_y; hkVector4fUtil::sinCos(yaw * hkSimdFloat32_Half,sin_y,cos_y); // sin(y)cos(y)
	hkVector4f n; n.setMul(m, sin_y); // c1c2s3, c1s2s3,  s1c2s3,  s1s2s3
	hkVector4fComparison mask; mask.set<hkVector4ComparisonMask::MASK_ZW>();
	n.setFlipSign(n, mask);          // c1c2s3, c1s2s3, -s1c2s3, -s1s2s3

	hkVector4f o; o.setMul(m, cos_y); // c1c2c3, c1s2c3, s1c2c3, s1s2c3
	hkVector4f s; s.setPermutation<hkVectorPermutation::WZYX>(o); // s1s2c3, s1c2c3, c1s2c3, c1c2c3

	m_vec.setAdd(n,s); // c1c2s3+s1s2c3, c1s2s3+s1c2c3, -s1c2s3+c1s2c3, -s1s2s3+c1c2c3
#else
	hkFloat32 roll2  = roll.getReal() * 0.5f;
	hkFloat32 pitch2 = pitch.getReal() * 0.5f;
	hkFloat32 yaw2   = yaw.getReal() * 0.5f;

	hkFloat32 c1 = hkMath::cos(roll2);
	hkFloat32 s1 = hkMath::sin(roll2);
	hkFloat32 c2 = hkMath::cos(pitch2);
	hkFloat32 s2 = hkMath::sin(pitch2);
	hkFloat32 c3 = hkMath::cos(yaw2);
	hkFloat32 s3 = hkMath::sin(yaw2);

	hkFloat32 c1c2 = c1*c2;
	hkFloat32 c1s2 = c1*s2;
	hkFloat32 s1s2 = s1*s2;
	hkFloat32 s1c2 = s1*c2;

	m_vec(0) = c1c2*s3 + s1s2*c3;
	m_vec(1) = s1c2*c3 + c1s2*s3;
	m_vec(2) = c1s2*c3 - s1c2*s3;
	m_vec(3) = c1c2*c3 - s1s2*s3;
#endif
}

void hkQuaternionf::setFromEulerAngles(hkFloat32 roll, hkFloat32 pitch, hkFloat32 yaw) 
{
	setFromEulerAngles(hkSimdFloat32::fromFloat(roll),hkSimdFloat32::fromFloat(pitch),hkSimdFloat32::fromFloat(yaw));
}

void hkQuaternionf::setFromEulerAngles_Approximate(hkSimdFloat32Parameter roll, hkSimdFloat32Parameter pitch, hkSimdFloat32Parameter yaw) 
{
#if defined(HK_PLATFORM_WIN32) && (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
	hkVector4f pprr; pprr.set(pitch,pitch,roll,roll);
	hkVector4f pprr2; pprr2.setMul(pprr, hkVector4f::getConstant<HK_QUADREAL_INV_2>());
	hkVector4f yy2; yy2.setAll(yaw * hkSimdFloat32_Half);

	hkVector4f sc_yy2;   hkVector4fUtil::sinCosApproximation(yy2,sc_yy2);     // sin(y)cos(y)
	hkVector4f sc_pprr2; hkVector4fUtil::sinCosApproximation(pprr2,sc_pprr2); // sin(p)cos(p)sin(r)cos(r)

	hkVector4f ccss_roll; ccss_roll.setPermutation<hkVectorPermutation::WWZZ>(sc_pprr2);
	hkVector4f cscs_pitch; cscs_pitch.setPermutation<hkVectorPermutation::YXYX>(sc_pprr2);
	hkVector4f m; m.setMul(ccss_roll, cscs_pitch); // c1c2, c1s2, s1c2, s1s2

	const hkSimdFloat32 sin_y = sc_yy2.getComponent<0>();
	hkVector4f n; n.setMul(m, sin_y); // c1c2s3, c1s2s3,  s1c2s3,  s1s2s3
	hkVector4fComparison mask; mask.set<hkVector4ComparisonMask::MASK_ZW>();
	n.setFlipSign(n, mask);          // c1c2s3, c1s2s3, -s1c2s3, -s1s2s3

	const hkSimdFloat32 cos_y = sc_yy2.getComponent<1>();
	hkVector4f o; o.setMul(m, cos_y); // c1c2c3, c1s2c3, s1c2c3, s1s2c3

	hkVector4f s; s.setPermutation<hkVectorPermutation::WZXY>(o); // s1s2c3, s1c2c3, c1c2c3, c1s2c3
	m_vec.setAdd(n,s); // c1c2s3+s1s2c3, c1s2s3+s1c2c3, -s1c2s3+c1c2c3, -s1s2s3+c1s2c3
#else
	setFromEulerAngles(roll, pitch, yaw);
#endif
}



hkBool32 hkQuaternionf::isOk(const hkFloat32 epsilon) const
{
	hkBool32 ok = m_vec.isOk<4>();
	const hkSimdFloat32 error = m_vec.lengthSquared<4>() - hkSimdFloat32::getConstant<HK_QUADREAL_1>();
	hkSimdFloat32 absErr; absErr.setAbs(error);
	hkSimdFloat32 tol; tol.setFromFloat(epsilon);
	return ok && absErr.isLess(tol);
}

void hkQuaternionf::set(const hkRotationf& r)
{
	_set(r);
}

void hkQuaternionf::setFlippedRotation(hkVector4fParameter from)
{
	hkVector4f vec;
	hkVector4fUtil::calculatePerpendicularVector(from, vec);
	vec.normalize<3>();
	vec.zeroComponent<3>();
	m_vec = vec;
}

//
//	Sets this = Slerp(q0, q1, t)
#define HK_ONE_MINUS_QUATERNIONf_DELTA hkFloat32(1.0f - 1e-3f)

void hkQuaternionf::setSlerp(hkQuaternionfParameter q0, hkQuaternionfParameter q1, hkSimdFloat32Parameter t)
#if defined(HK_COMPILER_GHS)
{


	hkFloat32 tReal = t.getReal();
	hkFloat32 qdelta = HK_ONE_MINUS_QUATERNIONf_DELTA;

	hkFloat32 oldCosTheta = q0.m_vec.dot<4>(q1.m_vec).getReal();
	hkFloat32 cosTheta = __FSEL(oldCosTheta, oldCosTheta, -oldCosTheta);

	hkFloat32 t0,t1;

	if (cosTheta < qdelta)
	{
		hkFloat32 theta = hkMath::acos(cosTheta);
		// use sqrtInv(1+c^2) instead of 1.0/sin(theta) 
		const hkFloat32 iSinTheta = hkMath::sqrtInverse( 1.0f - (cosTheta*cosTheta) );
		const hkFloat32 tTheta = tReal * theta;

		hkFloat32 s0 = hkMath::sin(theta-tTheta);
		hkFloat32 s1 = hkMath::sin(tTheta);

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

	hkVector4f slerp;
	hkSimdFloat32 t0Simd; t0Simd.setFromFloat(t0);
	hkSimdFloat32 t1Simd; t1Simd.setFromFloat(t1);
	slerp.setMul( t0Simd, q0.m_vec);
	slerp.addMul( t1Simd, q1.m_vec);
	slerp.normalize<4>();

	m_vec = slerp;
}
#else
{
	const hkSimdFloat32 one = hkSimdFloat32::getConstant<HK_QUADREAL_1>();
	hkSimdFloat32 qdelta; qdelta.setFromFloat(HK_ONE_MINUS_QUATERNIONf_DELTA);

	hkSimdFloat32 cosTheta = q0.m_vec.dot<4>(q1.m_vec);

	// If B is on the opposite hemisphere use -B instead of B
	const hkVector4fComparison cosThetaLessZero = cosTheta.lessZero();
	cosTheta.setFlipSign(cosTheta, cosThetaLessZero);

	hkSimdFloat32 t0,t1;

	if (cosTheta < qdelta)
	{
		hkSimdFloat32 theta; 
#if defined(HK_PLATFORM_WIN32) && (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED) && !defined(HK_ARCH_ARM)
		theta.m_real = hkMath::quadAcos(cosTheta.m_real);
#else
		theta.setFromFloat( hkMath::acos(cosTheta.getReal()) );
#endif

		// use sqrtInv(1+c^2) instead of 1.0/sin(theta) 
		const hkSimdFloat32 iSinTheta = ( one - (cosTheta*cosTheta) ).sqrtInverse<HK_ACC_23_BIT,HK_SQRT_IGNORE>();
		const hkSimdFloat32 tTheta = t * theta;

		hkSimdFloat32 s0,s1; 
#if defined(HK_PLATFORM_WIN32) && (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED) && !defined(HK_ARCH_ARM)
		hkVector4f ss; ss.set(theta-tTheta, tTheta, tTheta, tTheta);
		hkVector4f s01; 
		s01.m_quad = hkMath::quadSin(ss.m_quad);
		s0 = s01.getComponent<0>();
		s1 = s01.getComponent<1>();
#else

		// Simplify the multiples of 2Pi, as they seem to cause issues on some platforms, i.e. sin(LargeNumber) = NaN
		hkSimdFloat32 nDivs;
		const hkSimdFloat32 angle0 = hkVector4Util::toRange(theta - tTheta, hkSimdFloat32_Pi, hkSimdFloat32_TwoPi, nDivs);
		const hkSimdFloat32 angle1 = hkVector4Util::toRange(tTheta, hkSimdFloat32_Pi, hkSimdFloat32_TwoPi, nDivs);

		s0.setFromFloat(hkMath::sin(angle0.getReal()));
		s1.setFromFloat(hkMath::sin(angle1.getReal()));
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

	hkVector4f slerp;
	slerp.setMul( t0, q0.m_vec);
	slerp.addMul( t1, q1.m_vec);
	slerp.normalize<4>();

	m_vec = slerp;
}
#undef HK_ONE_MINUS_QUATERNIONf_DELTA
#endif

//
//	Calculates the barycentric interpolation between q0, q1, q2. The barycentric coordinates are l0, l1, l2, with l0 omitted (not used).

void hkQuaternionf::setBarycentric(const hkQuaternionf* HK_RESTRICT qVerts, hkVector4fParameter vLambda)
{
	// Triangle ABC has barycentric coords (l0, l1, l2)
	// Triangle BCA has barycentric coords (l1, l2, l0)
	// Triangle CAB has barycentric coords (l2, l0, l1)
	// We need to compute (l1 + l2, l2 + l0, l0 + l1) and pick the biggest absolute value
	// in order to get a stable inverse result
	hkVector4f l_120;	l_120.setPermutation<hkVectorPermutation::YZXW>(vLambda);	// [l1, l2, l0]
	hkVector4f l_201;	l_201.setPermutation<hkVectorPermutation::ZXYW>(vLambda);	// [l2, l0, l1]
	hkVector4f l;		l.setAdd(l_120, l_201);										// [l1 + l2, l2 + l0, l0 + l1]

	// Compute the most stable vertex permutation
	const int idxA		= l.getIndexOfMaxAbsComponent<3>();							// Either 0, 1, or 2, index of the largest absolute component in l
	const int idxB		= (1 << idxA) & 3;
	const int idxC		= (1 << idxB) & 3;
	HK_ASSERT(0x4890e973, (idxA >= 0) && (idxA <= 2) && (idxB >= 0) && (idxB <= 2) && (idxC >= 0) && (idxC <= 2));

	// Interpolate
	const hkSimdFloat32 l12	= l.getComponent(idxA);
	const hkSimdFloat32 l2		= l_201.getComponent(idxA);
	hkSimdFloat32 absL12;		absL12.setAbs(l12);
	hkSimdFloat32 invL12;		invL12.setReciprocal(l12);
	hkSimdFloat32 u;			u.setZero();
	u.setSelect(absL12.greater(hkSimdFloat32_Eps), l2 * invL12, u);

	const hkQuaternionf qA = qVerts[idxA];
	const hkQuaternionf qB = qVerts[idxB];
	const hkQuaternionf qC = qVerts[idxC];
	hkQuaternionf q0, q1;
	q0.setClosest(qA, qB);	q0.setSlerp(q0, qB, l12);
	q1.setClosest(qA, qC);	q1.setSlerp(q1, qC, l12);
	q0.setClosest(q0, q1);	setSlerp(q0, q1, u);
}

void hkQuaternionf::removeAxisComponent (hkVector4fParameter axis)
{
	// Rotate the desired axis 
	hkVector4f rotatedAxis;
	rotatedAxis._setRotatedDir(*this, axis);

	// Calculate the shortest rotation that would bring align both axis
	// Now, calculate the rotation required to reach that alignment
	// This is the component of the rotation perpendicular to the axis

	const hkSimdFloat32 dotProd = axis.dot<3>(rotatedAxis);
	const hkSimdFloat32 one = hkSimdFloat32::getConstant<HK_QUADREAL_1>();
	hkSimdFloat32 eps; eps.setFromFloat(hkFloat32(1e-3f));

	// Parallel
	if ( (dotProd-one) > -eps )
	{
		setIdentity();
		return;
	}

	// Opposite
	if ( (dotProd+one) < eps )
	{
		hkVector4f perpVector;
		hkVector4fUtil::calculatePerpendicularVector(axis, perpVector);
		perpVector.normalize<3>();

		m_vec = perpVector;       // axis * sin(PI/2)
		m_vec.zeroComponent<3>(); // cos(PI/2)
		return;
	}

	// else
	{
		hkSimdFloat32 rotationAngle;
#if defined(HK_PLATFORM_WIN32) && (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED) && !defined(HK_ARCH_ARM)
		rotationAngle.m_real = hkMath::quadAcos(dotProd.m_real);
#else
		rotationAngle.setFromFloat( hkMath::acos(dotProd.getReal()) );
#endif
		hkVector4f rotationAxis;
		rotationAxis.setCross(axis, rotatedAxis);
		rotationAxis.normalize<3>();

		setAxisAngle(rotationAxis, rotationAngle);
	}
}

void hkQuaternionf::decomposeRestAxis(hkVector4fParameter axis, hkQuaternionf& restOut, hkSimdFloat32& angleOut) const
{
	hkQuaternionf axisRot;
	{
		restOut = *this;
		restOut.removeAxisComponent(axis);

		// axisRot = inv(rest) * q
		axisRot.setInverseMul(restOut, *this);
		axisRot.normalize(); // workaround COM-2170 to avoid getAngle returning a NaN in some cases.
	}

	hkSimdFloat32 a; a.setFromFloat(axisRot.getAngle());

	const hkSimdFloat32 reverse = axisRot.getRealPart() * axisRot.getImag().dot<3>(axis);
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
