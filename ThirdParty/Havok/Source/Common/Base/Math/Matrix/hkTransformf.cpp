/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Math/Vector/hkVector4Util.h>
#include <Common/Base/Math/Matrix/hkMatrix3Util.h>
#include <Common/Base/Math/Matrix/hkTransformUtil.h>

void hkTransformf::get4x4ColumnMajor(hkFloat32* HK_RESTRICT d) const
{
	const hkFloat32* HK_RESTRICT p = &m_rotation(0,0);
#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
	hkVector4f dv0, dv1, dv2, dv3;
	dv0.load<4>(p);
	dv1.load<4>(p+4);
	dv2.load<4>(p+8);
	dv3.load<4>(p+12);

	const hkSimdFloat32 one = hkSimdFloat32::getConstant<HK_QUADREAL_1>();

	dv0.zeroComponent<3>();
	dv1.zeroComponent<3>();
	dv2.zeroComponent<3>();
	dv3.setComponent<3>(one);

	dv0.store<4>(d);
	dv1.store<4>(d+4);
	dv2.store<4>(d+8);
	dv3.store<4>(d+12);
#else
	for (int i = 0; i<4; i++)
	{
		hkFloat32 a = hkFloat32(p[0]);
		hkFloat32 b = hkFloat32(p[1]);
		hkFloat32 c = hkFloat32(p[2]);

		d[0] = a;
		d[1] = b;
		d[2] = c;
		d[3] = 0.0f;
		d+= 4;
		p+= 4;
	}
	d[-1] = 1.0f;
#endif
}

void hkTransformf::get4x4ColumnMajor(hkDouble64* HK_RESTRICT d) const
{
	const hkFloat32* HK_RESTRICT p = &m_rotation(0,0);
#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
	hkVector4f dv0, dv1, dv2, dv3;
	dv0.load<4>(p);
	dv1.load<4>(p+4);
	dv2.load<4>(p+8);
	dv3.load<4>(p+12);

	const hkSimdFloat32 one = hkSimdFloat32::getConstant<HK_QUADREAL_1>();

	dv0.zeroComponent<3>();
	dv1.zeroComponent<3>();
	dv2.zeroComponent<3>();
	dv3.setComponent<3>(one);

	dv0.store<4>(d);
	dv1.store<4>(d+4);
	dv2.store<4>(d+8);
	dv3.store<4>(d+12);
#else
	for (int i = 0; i<4; i++)
	{
		hkDouble64 a = hkDouble64(p[0]);
		hkDouble64 b = hkDouble64(p[1]);
		hkDouble64 c = hkDouble64(p[2]);
		d[0] = a;
		d[1] = b;
		d[2] = c;
		d[3] = 0.0;
		d+= 4;
		p+= 4;
	}
	d[-1] = 1.0;
#endif
}

void hkTransformf::set4x4ColumnMajor(const hkFloat32* p)
{
	hkFloat32* HK_RESTRICT d = &m_rotation(0,0);
#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
	hkVector4f dv0, dv1, dv2, dv3;
	dv0.load<4>(p);
	dv1.load<4>(p+4);
	dv2.load<4>(p+8);
	dv3.load<4>(p+12);

	const hkSimdFloat32 one = hkSimdFloat32::getConstant<HK_QUADREAL_1>();

	dv0.zeroComponent<3>();
	dv1.zeroComponent<3>();
	dv2.zeroComponent<3>();
	dv3.setComponent<3>(one);

	dv0.store<4>(d);
	dv1.store<4>(d+4);
	dv2.store<4>(d+8);
	dv3.store<4>(d+12);
#else
	for (int i = 0; i<4; i++)
	{
		hkFloat32 d0 = hkFloat32(p[0]);
		hkFloat32 d1 = hkFloat32(p[1]);
		hkFloat32 d2 = hkFloat32(p[2]);
		d[0] = d0;
		d[1] = d1;
		d[2] = d2;
		d[3] = hkFloat32(0);
		d+= 4;
		p+= 4;
	}
	d[-1] = hkFloat32(1);
#endif
}

void hkTransformf::set4x4ColumnMajor(const hkDouble64* p)
{
	hkFloat32* HK_RESTRICT d = &m_rotation(0,0);
#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
	hkVector4f dv0, dv1, dv2, dv3;
	dv0.load<4>(p);
	dv1.load<4>(p+4);
	dv2.load<4>(p+8);
	dv3.load<4>(p+12);

	const hkSimdFloat32 one = hkSimdFloat32::getConstant<HK_QUADREAL_1>();

	dv0.zeroComponent<3>();
	dv1.zeroComponent<3>();
	dv2.zeroComponent<3>();
	dv3.setComponent<3>(one);

	dv0.store<4>(d);
	dv1.store<4>(d+4);
	dv2.store<4>(d+8);
	dv3.store<4>(d+12);
#else
	for (int i = 0; i<4; i++)
	{
		hkFloat32 d0 = hkFloat32(p[0]);
		hkFloat32 d1 = hkFloat32(p[1]);
		hkFloat32 d2 = hkFloat32(p[2]);
		d[0] = d0;
		d[1] = d1;
		d[2] = d2;
		d[3] = hkFloat32(0);
		d+= 4;
		p+= 4;
	}
	d[-1] = hkFloat32(1);
#endif
}

void hkTransformf::setInverse( const hkTransformf& t )
{
	_setInverse(t);
}


// aTc = aTb * bTc
void hkTransformf::setMul( const hkTransformf& aTb, const hkTransformf& bTc )
{
	hkTransformfUtil::_mulTransformTransform( aTb, bTc, this );
}

void hkTransformf::setMulEq( const hkTransformf& bTc )
{
	hkTransformf aTb = *this; // copy
	hkTransformfUtil::_mulTransformTransform( aTb, bTc, this );
}


// aTc = aTb * bTc
void hkTransformf::setMul( const hkQsTransformf& aTb, const hkTransformf& bTc )
{
	// Calculate the 3x3 matrices for rotation and scale
	hkRotationf rotMatrix; rotMatrix.set(aTb.getRotation());
	hkRotationf scaMatrix; hkMatrix3fUtil::_setDiagonal(aTb.getScale(), scaMatrix);

	// Calculate R*S
	hkRotationf rotSca; rotSca.setMul(rotMatrix, scaMatrix);

	// Construct transform
	{
		hkVector4f* HK_RESTRICT col = (hkVector4f* HK_RESTRICT)&m_rotation;
		col[0]._setRotatedDir(rotSca, bTc.m_rotation.getColumn<0>());
		col[1]._setRotatedDir(rotSca, bTc.m_rotation.getColumn<1>());
		col[2]._setRotatedDir(rotSca, bTc.m_rotation.getColumn<2>());
		m_translation._setRotatedDir(rotSca, bTc.m_translation);
	}

	
	m_translation.add( aTb.getTranslation() );

/*
	HK_ASSERT2(0x1ff88f0e, aTb.getRotation().isOk(), "hkQuaternionf not normalized/invalid!");

	hkVector4f col0, col1, col2;
	col0 = bTc.getRotation().getColumn(0);
	col1 = bTc.getRotation().getColumn(1);
	col2 = bTc.getRotation().getColumn(2);
	
	col0.mul(aTb.m_scale);
	col1.mul(aTb.m_scale);
	col2.mul(aTb.m_scale);

#if defined( HK_COMPILER_HAS_INTRINSICS_ALTIVEC )
	hkVector4f proda, prodb, prodc;
	{
		const hkVector4f q = aTb.getRotation().m_vec;
		hkVector4f xxxx, yyyy, zzzz;
		
		hkVector4f q2;	q2.setAdd(q,q);
		xxxx.setBroadcast(q2,0);
		yyyy.setBroadcast(q2,1);
		zzzz.setBroadcast(q2,2);

		hkVector4f yxwz, zwxy, wzyx;
		HK_VECTOR4f_PERM1(yxwz, q, HK_VECTOR4f_PERM1ARG(1,0,3,2) );
		HK_VECTOR4f_PERM1(zwxy, q, HK_VECTOR4f_PERM1ARG(2,3,0,1) );
		HK_VECTOR4f_PERM1(wzyx, q, HK_VECTOR4f_PERM1ARG(3,2,1,0) );

#if defined HK_PLATFORM_XBOX360
		static HK_ALIGN16( const hkUint32 nx[4] ) = { 0x80000000, 0, 0, 0 };
		static HK_ALIGN16( const hkUint32 ny[4] ) = { 0, 0x80000000, 0, 0 };
		static HK_ALIGN16( const hkUint32 nz[4] ) = { 0, 0, 0x80000000, 0 };
		yxwz = __vxor( yxwz, *(const hkQuadFloat32*)&ny );
		zwxy = __vxor( zwxy, *(const hkQuadFloat32*)&nx );
		wzyx = __vxor( wzyx, *(const hkQuadFloat32*)&nz );
#else
		static HK_ALIGN16( const vec_uint4 nx ) = { 0x80000000, 0, 0, 0 };
		static HK_ALIGN16( const vec_uint4 ny ) = { 0, 0x80000000, 0, 0 };
		static HK_ALIGN16( const vec_uint4 nz ) = { 0, 0, 0x80000000, 0 };
		yxwz = vec_xor( hkQuadFloat32(yxwz), (const hkQuadFloat32)ny );
		zwxy = vec_xor( hkQuadFloat32(zwxy), (const hkQuadFloat32)nx );
		wzyx = vec_xor( hkQuadFloat32(wzyx), (const hkQuadFloat32)nz );
#endif
		proda.setMul(zzzz,zwxy);
		proda.subMul(yyyy,yxwz);
		prodb.setMul(xxxx,yxwz);
		prodb.subMul(zzzz,wzyx);
		prodc.setMul(yyyy,wzyx);
		prodc.subMul(xxxx,zwxy);
	}
#elif defined HK_COMPILER_HAS_INTRINSICS_IA32 && (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
	hkVector4f proda, prodb, prodc;
	{
		
		const hkVector4f xyzw = aTb.getRotation().m_vec;
	
		hkVector4f xyzw2;
		xyzw2.setAdd( xyzw, xyzw );								
		hkVector4f tmp1, tmp2, tmp5, tmp6;
		HK_VECTOR4f_PERM1(tmp2, xyzw, HK_VECTOR4f_PERM1ARG(1,0,0,1));
		HK_VECTOR4f_PERM1(prodc, xyzw2, HK_VECTOR4f_PERM1ARG(1,1,2,2) );
		tmp2.mul(prodc);
		HK_VECTOR4f_PERM1(prodb, xyzw, HK_VECTOR4f_PERM1ARG(2,3,3,3) );
		HK_VECTOR4f_PERM1(tmp5, xyzw2, HK_VECTOR4f_PERM1ARG(2,2,1,0) );
		prodb.mul(tmp5);
		tmp1 = _mm_mul_ss( xyzw, xyzw2 );
		tmp6 = hkQuadReal0000;
		tmp6 = _mm_sub_ss( tmp6, tmp1 );							
		tmp6 = _mm_sub_ss( tmp6, tmp2 );							
		static HK_ALIGN16( const hkUint32 _negateMask0[4] ) = { 0, 0x80000000, 0x80000000, 0x80000000 };
		const hkQuadFloat32 negateMask0 = *(const hkQuadFloat32*)_negateMask0;
		tmp2 = _mm_xor_ps ( tmp2, negateMask0 );
		static HK_ALIGN16( const hkUint32 negateMask1[4] ) = { 0x80000000, 0, 0x80000000, 0x80000000 };
		prodb = _mm_xor_ps ( prodb, *(const hkQuadFloat32*)negateMask1 );
		prodc = prodb;												
		prodc.sub(tmp2);
		proda = prodc;
		tmp2 = _mm_move_ss( tmp2, tmp1 );							
		prodb = _mm_xor_ps ( prodb, negateMask0 );					
		prodb.sub(tmp2);
		HK_VECTOR4f_PERM1(prodb, prodb, HK_VECTOR4f_PERM1ARG(1,0,3,2) );
		prodc = _mm_movehl_ps( prodc, prodb );
		HK_VECTOR4f_SHUF(prodc,prodc,tmp6, HK_VECTOR4f_SHUFFLE(1,3,0,2) );
	}

	hkVector4f v0, v1, v2, sum;
	v0.setBroadcast(col0,0);
	v1.setBroadcast(col0,1);
	v2.setBroadcast(col0,2);
	sum.setAddMul(col0,proda,v0);
	sum.addMul(prodb,v1);
	sum.addMul(prodc,v2);
	m_rotation.getColumn(0).setXYZ0(sum);

	v0.setBroadcast(col1,0);
	v1.setBroadcast(col1,1);
	v2.setBroadcast(col1,2);
	sum.setAddMul(col1,proda,v0);
	sum.addMul(prodb,v1);
	sum.addMul(prodc,v2);
	m_rotation.getColumn(1).setXYZ0(sum);

	v0.setBroadcast(col2,0);
	v1.setBroadcast(col2,1);
	v2.setBroadcast(col2,2);
	sum.setAddMul(col2,proda,v0);
	sum.addMul(prodb,v1);
	sum.addMul(prodc,v2);
	m_rotation.getColumn(2).setXYZ0(sum);

	const hkVector4f v = bTc.getTranslation();
	v0.setBroadcast(v,0);
	v1.setBroadcast(v,1);
	v2.setBroadcast(v,2);
	sum.setAddMul(v,proda,v0);
	sum.addMul(prodb,v1);
	sum.addMul(prodc,v2);
	m_translation.setXYZW(sum,hkVector4f::getConstant(HK_QUADREAL_1));
#else
	m_rotation.getColumn(0).setRotatedDir(aTb.getRotation(),col0);
	m_rotation.getColumn(1).setRotatedDir(aTb.getRotation(),col1);
	m_rotation.getColumn(2).setRotatedDir(aTb.getRotation(),col2);
	m_translation.setRotatedDir(aTb.getRotation(),bTc.getTranslation());
#endif

	m_translation.add(aTb.m_translation);
*/
}

void hkTransformf::setMulInverseMul( const hkTransformf& bTa, const hkTransformf &bTc )
{
	HK_ASSERT( 0xf0345456, &bTa != this );	// check for aliasing
	hkVector4fUtil::rotateInversePoints( bTa.getRotation(), &bTc.getRotation().getColumn(0), 3, &this->getRotation().getColumn(0) );
	hkVector4f h; h.setSub(bTc.m_translation, bTa.m_translation );
	m_translation._setRotatedInverseDir( bTa.m_rotation, h);
}


bool hkTransformf::isOk() const
{
	return(	getTranslation().isOk<3>() && getRotation().isOk() );
}

void hkTransformf::setMulMulInverse( const hkTransformf &wTa, const hkTransformf &wTb )
{
	hkTransformfUtil::_computeMulInverse(wTa, wTb, *this);
}

bool hkTransformf::isApproximatelyEqual( const hkTransformf& t, hkFloat32 zero ) const
{
	hkSimdFloat32 sZ; sZ.setFromFloat(zero);
	return		m_rotation.isApproximatelyEqualSimd( t.getRotation(), sZ )
			&&	m_translation.allEqual<3>( t.getTranslation(), sZ );
}

bool hkTransformf::isApproximatelyEqualSimd( const hkTransformf& t, hkSimdFloat32Parameter sZ ) const
{
	return		m_rotation.isApproximatelyEqualSimd( t.getRotation(), sZ )
			&&	m_translation.allEqual<3>( t.getTranslation(), sZ );
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
