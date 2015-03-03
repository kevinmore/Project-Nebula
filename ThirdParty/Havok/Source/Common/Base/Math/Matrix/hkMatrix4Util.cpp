/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Math/Matrix/hkMatrix4Util.h>

namespace __hkMatrix4UtilAlgo
{

#if HK_CONFIG_SIMD != HK_CONFIG_SIMD_ENABLED

hkResult invert(hkMatrix4f& m, hkSimdFloat32Parameter epsilon)
{
	hkFloat32 fA0 = m(0,0)*m(1,1) - m(0,1)*m(1,0);
	hkFloat32 fA1 = m(0,0)*m(1,2) - m(0,2)*m(1,0);
	hkFloat32 fA2 = m(0,0)*m(1,3) - m(0,3)*m(1,0);
	hkFloat32 fA3 = m(0,1)*m(1,2) - m(0,2)*m(1,1);
	hkFloat32 fA4 = m(0,1)*m(1,3) - m(0,3)*m(1,1);
	hkFloat32 fA5 = m(0,2)*m(1,3) - m(0,3)*m(1,2);
	hkFloat32 fB0 = m(2,0)*m(3,1) - m(2,1)*m(3,0);
	hkFloat32 fB1 = m(2,0)*m(3,2) - m(2,2)*m(3,0);
	hkFloat32 fB2 = m(2,0)*m(3,3) - m(2,3)*m(3,0);
	hkFloat32 fB3 = m(2,1)*m(3,2) - m(2,2)*m(3,1);
	hkFloat32 fB4 = m(2,1)*m(3,3) - m(2,3)*m(3,1);
	hkFloat32 fB5 = m(2,2)*m(3,3) - m(2,3)*m(3,2);

	hkFloat32 det = fA0*fB5 - fA1*fB4 + fA2*fB3 + fA3*fB2 - fA4*fB1 + fA5*fB0;

	if ( hkMath::fabs(det) <= epsilon.getReal() )
	{
		return HK_FAILURE;
	}

	hkSimdFloat32 invDet; invDet.setReciprocal<HK_ACC_FULL,HK_DIV_IGNORE>( hkSimdFloat32::fromFloat(det) );

	hkMatrix4f temp;
	temp(0,0) = + m(1,1)*fB5 - m(1,2)*fB4 + m(1,3)*fB3;
	temp(1,0) = - m(1,0)*fB5 + m(1,2)*fB2 - m(1,3)*fB1;
	temp(2,0) = + m(1,0)*fB4 - m(1,1)*fB2 + m(1,3)*fB0;
	temp(3,0) = - m(1,0)*fB3 + m(1,1)*fB1 - m(1,2)*fB0;
	temp(0,1) = - m(0,1)*fB5 + m(0,2)*fB4 - m(0,3)*fB3;
	temp(1,1) = + m(0,0)*fB5 - m(0,2)*fB2 + m(0,3)*fB1;
	temp(2,1) = - m(0,0)*fB4 + m(0,1)*fB2 - m(0,3)*fB0;
	temp(3,1) = + m(0,0)*fB3 - m(0,1)*fB1 + m(0,2)*fB0;
	temp(0,2) = + m(3,1)*fA5 - m(3,2)*fA4 + m(3,3)*fA3;
	temp(1,2) = - m(3,0)*fA5 + m(3,2)*fA2 - m(3,3)*fA1;
	temp(2,2) = + m(3,0)*fA4 - m(3,1)*fA2 + m(3,3)*fA0;
	temp(3,2) = - m(3,0)*fA3 + m(3,1)*fA1 - m(3,2)*fA0;
	temp(0,3) = - m(2,1)*fA5 + m(2,2)*fA4 - m(2,3)*fA3;
	temp(1,3) = + m(2,0)*fA5 - m(2,2)*fA2 + m(2,3)*fA1;
	temp(2,3) = - m(2,0)*fA4 + m(2,1)*fA2 - m(2,3)*fA0;
	temp(3,3) = + m(2,0)*fA3 - m(2,1)*fA1 + m(2,2)*fA0;

	m.setMul(invDet, temp);

	return HK_SUCCESS;
}

hkResult invert(hkMatrix4d& m, hkSimdDouble64Parameter epsilon)
{
	hkDouble64 fA0 = m(0,0)*m(1,1) - m(0,1)*m(1,0);
	hkDouble64 fA1 = m(0,0)*m(1,2) - m(0,2)*m(1,0);
	hkDouble64 fA2 = m(0,0)*m(1,3) - m(0,3)*m(1,0);
	hkDouble64 fA3 = m(0,1)*m(1,2) - m(0,2)*m(1,1);
	hkDouble64 fA4 = m(0,1)*m(1,3) - m(0,3)*m(1,1);
	hkDouble64 fA5 = m(0,2)*m(1,3) - m(0,3)*m(1,2);
	hkDouble64 fB0 = m(2,0)*m(3,1) - m(2,1)*m(3,0);
	hkDouble64 fB1 = m(2,0)*m(3,2) - m(2,2)*m(3,0);
	hkDouble64 fB2 = m(2,0)*m(3,3) - m(2,3)*m(3,0);
	hkDouble64 fB3 = m(2,1)*m(3,2) - m(2,2)*m(3,1);
	hkDouble64 fB4 = m(2,1)*m(3,3) - m(2,3)*m(3,1);
	hkDouble64 fB5 = m(2,2)*m(3,3) - m(2,3)*m(3,2);

	hkDouble64 det = fA0*fB5 - fA1*fB4 + fA2*fB3 + fA3*fB2 - fA4*fB1 + fA5*fB0;

	if ( hkMath::fabs(det) <= epsilon.getReal() )
	{
		return HK_FAILURE;
	}

	hkSimdDouble64 invDet; invDet.setReciprocal<HK_ACC_FULL,HK_DIV_IGNORE>( hkSimdDouble64::fromFloat(det) );

	hkMatrix4d temp;
	temp(0,0) = + m(1,1)*fB5 - m(1,2)*fB4 + m(1,3)*fB3;
	temp(1,0) = - m(1,0)*fB5 + m(1,2)*fB2 - m(1,3)*fB1;
	temp(2,0) = + m(1,0)*fB4 - m(1,1)*fB2 + m(1,3)*fB0;
	temp(3,0) = - m(1,0)*fB3 + m(1,1)*fB1 - m(1,2)*fB0;
	temp(0,1) = - m(0,1)*fB5 + m(0,2)*fB4 - m(0,3)*fB3;
	temp(1,1) = + m(0,0)*fB5 - m(0,2)*fB2 + m(0,3)*fB1;
	temp(2,1) = - m(0,0)*fB4 + m(0,1)*fB2 - m(0,3)*fB0;
	temp(3,1) = + m(0,0)*fB3 - m(0,1)*fB1 + m(0,2)*fB0;
	temp(0,2) = + m(3,1)*fA5 - m(3,2)*fA4 + m(3,3)*fA3;
	temp(1,2) = - m(3,0)*fA5 + m(3,2)*fA2 - m(3,3)*fA1;
	temp(2,2) = + m(3,0)*fA4 - m(3,1)*fA2 + m(3,3)*fA0;
	temp(3,2) = - m(3,0)*fA3 + m(3,1)*fA1 - m(3,2)*fA0;
	temp(0,3) = - m(2,1)*fA5 + m(2,2)*fA4 - m(2,3)*fA3;
	temp(1,3) = + m(2,0)*fA5 - m(2,2)*fA2 + m(2,3)*fA1;
	temp(2,3) = - m(2,0)*fA4 + m(2,1)*fA2 - m(2,3)*fA0;
	temp(3,3) = + m(2,0)*fA3 - m(2,1)*fA1 + m(2,2)*fA0;

	m.setMul(invDet, temp);

	return HK_SUCCESS;
}

hkResult setInverse(const hkMatrix4f& m, hkMatrix4f& out, hkSimdFloat32Parameter epsilon)
{
	out = m;
	return invert(out,epsilon);
}

hkResult setInverse(const hkMatrix4d& m, hkMatrix4d& out, hkSimdDouble64Parameter epsilon)
{
	out = m;
	return invert(out,epsilon);
}

#else

hkResult invert(hkMatrix4f& m, hkSimdFloat32Parameter epsilon)
{
	hkMatrix4f out;
	hkResult res = setInverse(m, out, epsilon);
	if (res == HK_SUCCESS)
	{
		m = out;
	}
	return res;
}

hkResult invert(hkMatrix4d& m, hkSimdDouble64Parameter epsilon)
{
	hkMatrix4d out;
	hkResult res = setInverse(m, out, epsilon);
	if (res == HK_SUCCESS)
	{
		m = out;
	}
	return res;
}

hkResult setInverse(const hkMatrix4f& m, hkMatrix4f& out, hkSimdFloat32Parameter epsilon)
{
	HK_ASSERT(0x2454a322, &out != &m);

	hkVector4f a, b, c;
	hkVector4f r1, r2, r3, tt, tt2;
	hkVector4f sum;
	hkVector4f m0, m1, m2, m3;
	hkVector4f t0, t1;

	// Calculating the minterms for the first line.

	hkVector4fComparison pnpn; pnpn.set<hkVector4ComparisonMask::MASK_XZ>();
	hkVector4fComparison npnp; npnp.set<hkVector4ComparisonMask::MASK_YW>();

	const hkVector4f* cols = &(m.getColumn<0>());

	// 
	tt = cols[3];
	tt2.setPermutation<hkVectorPermutation::YZWX>(cols[2]);
	c.setMul(tt2, tt);                 // V3' V4
	t0.setPermutation<hkVectorPermutation::ZWXY>(tt);
	a.setMul(tt2, t0);                 // V3' V4"
	t0.setPermutation<hkVectorPermutation::WXYZ>(tt);
	b.setMul(tt2, t0);                 // V3' V4^

	t0.setPermutation<hkVectorPermutation::YZWX>(a);
	t1.setPermutation<hkVectorPermutation::ZWXY>(c);
	r1.setSub(t0, t1);                 // V3" V4^ - V3^ V4"
	t0.setPermutation<hkVectorPermutation::ZWXY>(b);
	r2.setSub(t0, b);                  // V3^ V4' - V3' V4^
	t0.setPermutation<hkVectorPermutation::YZWX>(c);
	r3.setSub(a, t0);                  // V3' V4" - V3" V4'

	tt = cols[1];
	a.setPermutation<hkVectorPermutation::YZWX>(tt);
	sum.setMul(a, r1);
	b.setPermutation<hkVectorPermutation::ZWXY>(tt);
	sum.addMul(b, r2);
	c.setPermutation<hkVectorPermutation::WXYZ>(tt);
	sum.addMul(c, r3);

	hkVector4f detTmp; detTmp.setMul(sum, cols[0]);
	// Calculating the determinant.
	//hkSimdFloat32 det = detTmp.getSimdAt(0) - detTmp.getSimdAt(1) + detTmp.getSimdAt(2) - detTmp.getSimdAt(3);

	hkVector4f detTmp2; detTmp2.setFlipSign(detTmp, pnpn);
	hkSimdFloat32 det = detTmp2.horizontalAdd<4>();
	hkSimdFloat32 absDet; absDet.setAbs(det);

	// Check for determinant
	if (absDet.isLess(epsilon))
	{
		return HK_FAILURE;
	}

	m0.setFlipSign(sum, pnpn);

	// Calculating the minterms of the second line (using previous results).
	tt.setPermutation<hkVectorPermutation::YZWX>(cols[0]);
	sum.setMul(tt, r1);
	tt.setPermutation<hkVectorPermutation::YZWX>(tt);
	sum.addMul(tt, r2);
	tt.setPermutation<hkVectorPermutation::YZWX>(tt);
	sum.addMul(tt, r3);
	m1.setFlipSign(sum, npnp);

	// Calculating the minterms of the third line.
	tt.setPermutation<hkVectorPermutation::YZWX>(cols[0]);
	a.setMul(tt, b);                               // V1' V2"
	b.setMul(tt, c);                               // V1' V2^
	c.setMul(tt, cols[1]);                         // V1' V2

	t0.setPermutation<hkVectorPermutation::YZWX>(a);
	t1.setPermutation<hkVectorPermutation::ZWXY>(c);
	r1.setSub(t0, t1);                             // V1" V2^ - V1^ V2"
	t0.setPermutation<hkVectorPermutation::ZWXY>(b);
	r2.setSub(t0, b);                              // V1^ V2' - V1' V2^
	t0.setPermutation<hkVectorPermutation::YZWX>(c);
	r3.setSub(a, t0);                              // V1' V2" - V1" V2'

	tt.setPermutation<hkVectorPermutation::YZWX>(cols[3]);
	sum.setMul(tt, r1);
	tt.setPermutation<hkVectorPermutation::YZWX>(tt);
	sum.addMul(tt, r2);
	tt.setPermutation<hkVectorPermutation::YZWX>(tt);
	sum.addMul(tt, r3);
	m2.setFlipSign(sum, pnpn);

	// Recip det
	hkSimdFloat32 recipDet;
	recipDet.setReciprocal<HK_ACC_FULL,HK_DIV_IGNORE>(det);

	// Divide the first 12 minterms with the determinant.
	m0.mul(recipDet);
	m1.mul(recipDet);
	m2.mul(recipDet);

	// Calculate the minterms of the forth line and divide by the determinant.
	tt.setPermutation<hkVectorPermutation::YZWX>(cols[2]);
	sum.setMul(tt, r1);
	tt.setPermutation<hkVectorPermutation::YZWX>(tt);
	sum.addMul(tt, r2);
	tt.setPermutation<hkVectorPermutation::YZWX>(tt);
	sum.addMul(tt, r3);

	m3.setFlipSign(sum, npnp);
	m3.mul(recipDet);

	out.setCols(m0,m1,m2,m3);

	out.transpose();

	return HK_SUCCESS; 
}

hkResult setInverse(const hkMatrix4d& m, hkMatrix4d& out, hkSimdDouble64Parameter epsilon)
{
	HK_ASSERT(0x2454a322, &out != &m);

	hkVector4d a, b, c;
	hkVector4d r1, r2, r3, tt, tt2;
	hkVector4d sum;
	hkVector4d m0, m1, m2, m3;
	hkVector4d t0, t1;

	// Calculating the minterms for the first line.

	hkVector4dComparison pnpn; pnpn.set<hkVector4ComparisonMask::MASK_XZ>();
	hkVector4dComparison npnp; npnp.set<hkVector4ComparisonMask::MASK_YW>();

	const hkVector4d* cols = &(m.getColumn<0>());

	// 
	tt = cols[3];
	tt2.setPermutation<hkVectorPermutation::YZWX>(cols[2]);
	c.setMul(tt2, tt);                 // V3' V4
	t0.setPermutation<hkVectorPermutation::ZWXY>(tt);
	a.setMul(tt2, t0);                 // V3' V4"
	t0.setPermutation<hkVectorPermutation::WXYZ>(tt);
	b.setMul(tt2, t0);                 // V3' V4^

	t0.setPermutation<hkVectorPermutation::YZWX>(a);
	t1.setPermutation<hkVectorPermutation::ZWXY>(c);
	r1.setSub(t0, t1);                 // V3" V4^ - V3^ V4"
	t0.setPermutation<hkVectorPermutation::ZWXY>(b);
	r2.setSub(t0, b);                  // V3^ V4' - V3' V4^
	t0.setPermutation<hkVectorPermutation::YZWX>(c);
	r3.setSub(a, t0);                  // V3' V4" - V3" V4'

	tt = cols[1];
	a.setPermutation<hkVectorPermutation::YZWX>(tt);
	sum.setMul(a, r1);
	b.setPermutation<hkVectorPermutation::ZWXY>(tt);
	sum.addMul(b, r2);
	c.setPermutation<hkVectorPermutation::WXYZ>(tt);
	sum.addMul(c, r3);

	hkVector4d detTmp; detTmp.setMul(sum, cols[0]);
	// Calculating the determinant.
	//hkSimdDouble64 det = detTmp.getSimdAt(0) - detTmp.getSimdAt(1) + detTmp.getSimdAt(2) - detTmp.getSimdAt(3);

	hkVector4d detTmp2; detTmp2.setFlipSign(detTmp, pnpn);
	hkSimdDouble64 det = detTmp2.horizontalAdd<4>();
	hkSimdDouble64 absDet; absDet.setAbs(det);

	// Check for determinant
	if (absDet.isLess(epsilon))
	{
		return HK_FAILURE;
	}

	m0.setFlipSign(sum, pnpn);

	// Calculating the minterms of the second line (using previous results).
	tt.setPermutation<hkVectorPermutation::YZWX>(cols[0]);
	sum.setMul(tt, r1);
	tt.setPermutation<hkVectorPermutation::YZWX>(tt);
	sum.addMul(tt, r2);
	tt.setPermutation<hkVectorPermutation::YZWX>(tt);
	sum.addMul(tt, r3);
	m1.setFlipSign(sum, npnp);

	// Calculating the minterms of the third line.
	tt.setPermutation<hkVectorPermutation::YZWX>(cols[0]);
	a.setMul(tt, b);                               // V1' V2"
	b.setMul(tt, c);                               // V1' V2^
	c.setMul(tt, cols[1]);                         // V1' V2

	t0.setPermutation<hkVectorPermutation::YZWX>(a);
	t1.setPermutation<hkVectorPermutation::ZWXY>(c);
	r1.setSub(t0, t1);                             // V1" V2^ - V1^ V2"
	t0.setPermutation<hkVectorPermutation::ZWXY>(b);
	r2.setSub(t0, b);                              // V1^ V2' - V1' V2^
	t0.setPermutation<hkVectorPermutation::YZWX>(c);
	r3.setSub(a, t0);                              // V1' V2" - V1" V2'

	tt.setPermutation<hkVectorPermutation::YZWX>(cols[3]);
	sum.setMul(tt, r1);
	tt.setPermutation<hkVectorPermutation::YZWX>(tt);
	sum.addMul(tt, r2);
	tt.setPermutation<hkVectorPermutation::YZWX>(tt);
	sum.addMul(tt, r3);
	m2.setFlipSign(sum, pnpn);

	// Recip det
	hkSimdDouble64 recipDet;
	recipDet.setReciprocal<HK_ACC_FULL,HK_DIV_IGNORE>(det);

	// Divide the first 12 minterms with the determinant.
	m0.mul(recipDet);
	m1.mul(recipDet);
	m2.mul(recipDet);

	// Calculate the minterms of the forth line and divide by the determinant.
	tt.setPermutation<hkVectorPermutation::YZWX>(cols[2]);
	sum.setMul(tt, r1);
	tt.setPermutation<hkVectorPermutation::YZWX>(tt);
	sum.addMul(tt, r2);
	tt.setPermutation<hkVectorPermutation::YZWX>(tt);
	sum.addMul(tt, r3);

	m3.setFlipSign(sum, npnp);
	m3.mul(recipDet);

	out.setCols(m0,m1,m2,m3);

	out.transpose();

	return HK_SUCCESS; 
}

#endif

} // namespace

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
