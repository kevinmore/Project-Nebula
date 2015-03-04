/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Math/Matrix/hkMatrix4Util.h>

bool hkMatrix4d::isOk() const
{
#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
	bool col0Ok = m_col0.isOk<4>();
	bool col1Ok = m_col1.isOk<4>();
	bool col2Ok = m_col2.isOk<4>();
	bool col3Ok = m_col3.isOk<4>();
	return col0Ok && col1Ok && col2Ok && col3Ok;
#else
	const hkDouble64* f = reinterpret_cast<const hkDouble64*>(this);
	for(int i=0; i<16; ++i)
	{
		if( hkMath::isFinite(f[i]) == false )
		{
			return false;
		}
	}
	return true;
#endif
}

hkResult hkMatrix4d::setInverse(const hkMatrix4d& m, hkSimdDouble64Parameter eps)
{
	return hkMatrix4dUtil::setInverse(m, *this, eps);
}

void hkMatrix4d::transpose()
{
	HK_TRANSPOSE4d(m_col0,m_col1,m_col2,m_col3);
}

void hkMatrix4d::add( const hkMatrix4d& a )
{
	m_col0.add( a.getColumn<0>() );
	m_col1.add( a.getColumn<1>() );
	m_col2.add( a.getColumn<2>() );
	m_col3.add( a.getColumn<3>() );
}

void hkMatrix4d::sub( const hkMatrix4d& a )
{
	m_col0.sub( a.getColumn<0>() );
	m_col1.sub( a.getColumn<1>() );
	m_col2.sub( a.getColumn<2>() );
	m_col3.sub( a.getColumn<3>() );
}

// aTc = aTb * bTc


hkBool32 hkMatrix4d::isAffineTransformation() const
{
	hkVector4d row3; getRow<3>(row3);
	
#if !defined(HK_DEBUG)
	return row3.allExactlyEqual<3>(hkVector4d::getConstant<HK_QUADREAL_0001>());
#else
	// same as above, but you can set breakpoints
	if (row3.allExactlyEqual<3>(hkVector4d::getConstant<HK_QUADREAL_0001>()))
	{
		return true;
	}
	return false;
#endif
}

void hkMatrix4d::get4x4RowMajor(hkFloat32* HK_RESTRICT d) const
{
#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
	hkVector4d c0 = m_col0;
	hkVector4d c1 = m_col1;
	hkVector4d c2 = m_col2;
	hkVector4d c3 = m_col3;

	HK_TRANSPOSE4d(c0,c1,c2,c3);

	c0.store<4>(d);
	c1.store<4>(d+4);
	c2.store<4>(d+8);
	c3.store<4>(d+12);
#else
	const hkDouble64* p = (const hkDouble64*)this;
	for (int i = 0; i<4; i++)
	{
		hkFloat32 d0 = hkFloat32(p[0]);
		hkFloat32 d4 = hkFloat32(p[1]);
		hkFloat32 d8 = hkFloat32(p[2]);
		hkFloat32 d12 = hkFloat32(p[3]);
		d[0] = d0;
		d[4] = d4;
		d[8] = d8;
		d[12] = d12;
		d+= 1;
		p+= 4;
	}
#endif
}

void hkMatrix4d::get4x4RowMajor(hkDouble64* HK_RESTRICT d) const
{
#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
	hkVector4d c0 = m_col0;
	hkVector4d c1 = m_col1;
	hkVector4d c2 = m_col2;
	hkVector4d c3 = m_col3;

	HK_TRANSPOSE4d(c0,c1,c2,c3);

	c0.store<4>(d);
	c1.store<4>(d+4);
	c2.store<4>(d+8);
	c3.store<4>(d+12);
#else
	const hkDouble64* p = (const hkDouble64*)this;
	for (int i = 0; i<4; i++)
	{
		hkDouble64 d0 = hkDouble64(p[0]);
		hkDouble64 d4 = hkDouble64(p[1]);
		hkDouble64 d8 = hkDouble64(p[2]);
		hkDouble64 d12 = hkDouble64(p[3]);
		d[0] = d0;
		d[4] = d4;
		d[8] = d8;
		d[12] = d12;
		d+= 1;
		p+= 4;
	}
#endif
}

void hkMatrix4d::set4x4RowMajor(const hkFloat32* p)
{
#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
	hkVector4d c0; c0.load<4>(p);
	hkVector4d c1; c1.load<4>(p+4);
	hkVector4d c2; c2.load<4>(p+8);
	hkVector4d c3; c3.load<4>(p+12);

	HK_TRANSPOSE4d(c0,c1,c2,c3);

	m_col0 = c0;
	m_col1 = c1;
	m_col2 = c2;
	m_col3 = c3;
#else
	hkDouble64* HK_RESTRICT d = (hkDouble64*)this;
	for (int i = 0; i<4; i++)
	{
		hkDouble64 d0 = hkDouble64(p[0]);
		hkDouble64 d1 = hkDouble64(p[4]);
		hkDouble64 d2 = hkDouble64(p[8]);
		hkDouble64 d3 = hkDouble64(p[12]);
		d[0] = d0;
		d[1] = d1;
		d[2] = d2;
		d[3] = d3;
		p+= 1;
		d+= 4;
	}
#endif
}

void hkMatrix4d::set4x4RowMajor(const hkDouble64* p)
{
#if (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED)
	hkVector4d c0; c0.load<4>(p);
	hkVector4d c1; c1.load<4>(p+4);
	hkVector4d c2; c2.load<4>(p+8);
	hkVector4d c3; c3.load<4>(p+12);

	HK_TRANSPOSE4d(c0,c1,c2,c3);

	m_col0 = c0;
	m_col1 = c1;
	m_col2 = c2;
	m_col3 = c3;
#else
	hkDouble64* HK_RESTRICT d = (hkDouble64*)this;
	for (int i = 0; i<4; i++)
	{
		hkDouble64 d0 = hkDouble64(p[0]);
		hkDouble64 d1 = hkDouble64(p[4]);
		hkDouble64 d2 = hkDouble64(p[8]);
		hkDouble64 d3 = hkDouble64(p[12]);
		d[0] = d0;
		d[1] = d1;
		d[2] = d2;
		d[3] = d3;
		p+= 1;
		d+= 4;
	}
#endif
}

void hkMatrix4d::get4x4ColumnMajor(hkFloat32* HK_RESTRICT d) const
{
	m_col0.store<4>(d);
	m_col1.store<4>(d+4);
	m_col2.store<4>(d+8);
	m_col3.store<4>(d+12);
}

void hkMatrix4d::set4x4ColumnMajor(const hkFloat32* p)
{
	m_col0.load<4>(p);
	m_col1.load<4>(p+4);
	m_col2.load<4>(p+8);
	m_col3.load<4>(p+12);
}

void hkMatrix4d::get4x4ColumnMajor(hkDouble64* HK_RESTRICT d) const
{
	m_col0.store<4>(d);
	m_col1.store<4>(d+4);
	m_col2.store<4>(d+8);
	m_col3.store<4>(d+12);
}

void hkMatrix4d::set4x4ColumnMajor(const hkDouble64* p)
{
	m_col0.load<4>(p);
	m_col1.load<4>(p+4);
	m_col2.load<4>(p+8);
	m_col3.load<4>(p+12);
}

//
//	Set the contents based on the given hkTransformd. Will set the bottom row to (0,0,0,1) in this hkMatrix4d as 
//	it is undefined in a hkTransformd (not used)

void hkMatrix4d::set(const hkQTransformd& qt)
{
	hkTransformd t;
	t.set(qt.getRotation(), qt.getTranslation());
	m_col0 = t.getColumn<0>();
	m_col1 = t.getColumn<1>();
	m_col2 = t.getColumn<2>();
	m_col3 = t.getTranslation();
	resetFourthRow();
}

void hkMatrix4d::set(const hkQsTransformd& qst)
{
	hkRotationd rotMatrix; rotMatrix.set (qst.m_rotation);

	// This is equivalent to rotMatrix*diag(scale), but with fewer intermediate steps.
	hkVector4d c0; c0.setMul( rotMatrix.getColumn<0>(), qst.m_scale.getComponent<0>() );
	hkVector4d c1; c1.setMul( rotMatrix.getColumn<1>(), qst.m_scale.getComponent<1>() );
	hkVector4d c2; c2.setMul( rotMatrix.getColumn<2>(), qst.m_scale.getComponent<2>() );

	// this sets the w components equivalent to how resetFourthRow() would.
	m_col0.setXYZ_0( c0 );
	m_col1.setXYZ_0( c1 );
	m_col2.setXYZ_0( c2 );
	m_col3.setXYZ_W( qst.m_translation, hkSimdDouble64_1 );
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
