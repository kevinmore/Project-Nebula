/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>

//
//	Sets this transform to be the inverse of the given transform qt.

void hkQTransformf::setInverse(const hkQTransformf& qt)
{
	_setInverse(qt);
}

//
//	Sets this transform to be the product of qt1 and qt2.  (this = qt1 * qt2)

void hkQTransformf::setMul(const hkQTransformf& qt1, const hkQTransformf& qt2)
{
	_setMul(qt1, qt2);
}

//
//	Sets this transform to be the product of qt1 and qt2.  (this = qt1 * qt2)

void hkQTransformf::setMul(const hkTransformf& t1, const hkQTransformf& qt2)
{
	// Get rotation part as a quaternion
	hkQuaternionf q1;
	q1.setAndNormalize(t1.getRotation());

	// Convert to QTransform and call the inlined variant
	hkQTransformf qt1; qt1.set(q1, t1.getTranslation());
	_setMul(qt1, qt2);
}

//
//	Sets this transform to be the product of t1 and t2.  (this = t1 * t2)

void hkQTransformf::setMul(const hkTransformf& t1, const hkTransformf& t2)
{
	// Get rotation part as a quaternion
	hkQuaternionf q1;	q1.setAndNormalize(t1.getRotation());
	hkQuaternionf q2;	q2.setAndNormalize(t2.getRotation());

	// Convert to QTransform and call the inlined variant
	hkQTransformf qt1; qt1.set(q1, t1.getTranslation());
	hkQTransformf qt2; qt2.set(q2, t2.getTranslation());
	_setMul(qt1, qt2);
}

//
//	Sets this transform to be the product of qt1 and t2.  (this = qt1 * t2)

void hkQTransformf::setMul(const hkQTransformf& qt1, const hkTransformf& t2)
{
	// Get rotation part as a quaternion
	hkQuaternionf q2;	q2.setAndNormalize(t2.getRotation());

	// Convert to QTransform and call the inlined variant
	hkQTransformf qt2; qt2.set(q2, t2.getTranslation());
	_setMul(qt1, qt2);
}

//
//	Sets this transform to be the product of the inverse of qt1 by qt2.  (this = qt1^-1 * qt2)

void hkQTransformf::setMulInverseMul(const hkQTransformf& qt1, const hkQTransformf& qt2)
{
	_setMulInverseMul(qt1, qt2);
}

//
//	Sets this transform to be the product of the inverse of t1 by qt2.  (this = t1^-1 * qt2)

void hkQTransformf::setMulInverseMul(const hkTransformf& t1, const hkQTransformf& qt2)
{
	// Get rotation part as a quaternion
	hkQuaternionf q1;	q1.setAndNormalize(t1.getRotation());

	// Convert to QTransform and call the inlined variant
	hkQTransformf qt1; qt1.set(q1, t1.getTranslation());
	_setMulInverseMul(qt1, qt2);
}

//
//	Sets this transform to be the product of the inverse of qt1 by t2.  (this = qt1^-1 * t2)

void hkQTransformf::setMulInverseMul(const hkQTransformf& qt1, const hkTransformf& t2)
{
	// Get rotation part as a quaternion
	hkQuaternionf q2;	q2.setAndNormalize(t2.getRotation());

	// Convert to QTransform and call the inlined variant
	hkQTransformf qt2; qt2.set(q2, t2.getTranslation());
	_setMulInverseMul(qt1, qt2);
}

//
//	Sets this transform to be the product of the inverse of t1 by t2.  (this = t1^-1 * t2)

void hkQTransformf::setMulInverseMul(const hkTransformf& t1, const hkTransformf& t2)
{
	// Get rotation part as a quaternion
	hkQuaternionf q1;	q1.setAndNormalize(t1.getRotation());
	hkQuaternionf q2;	q2.setAndNormalize(t2.getRotation());

	// Convert to QTransform and call the inlined variant
	hkQTransformf qt1; qt1.set(q1, t1.getTranslation());
	hkQTransformf qt2; qt2.set(q2, t2.getTranslation());
	_setMulInverseMul(qt1, qt2);
}

//
//	Sets this transform to be the product of qt1 and the inverse of qt2. (this = qt1 * qt2^-1)

void hkQTransformf::setMulMulInverse(const hkQTransformf &qt1, const hkQTransformf &qt2)
{
	_setMulMulInverse(qt1, qt2);
}

bool hkQTransformf::isApproximatelyEqual( const hkQTransformf& other, hkFloat32 epsilon ) const
{
	hkSimdFloat32 sEps; sEps.setFromFloat(epsilon);
	return _isApproximatelyEqual(other, sEps);
}

//
//	Sets this transform  to a linear interpolation of the transforms qtA and qtB. 
//	Quaternions are checked for polarity and the resulting rotation is normalized

void hkQTransformf::setInterpolate4(const hkQTransformf& qtA, const hkQTransformf& qtB, hkSimdFloat32Parameter t)
{
	// Make sure we interpolate in the same hemisphere
	const hkQuaternionf qa = qtA.getRotation();
	hkQuaternionf qb;
	qb.setClosest(qtB.getRotation(), qa);

	// n-lerp the rotation part
	m_rotation.m_vec.setInterpolate(qa.m_vec, qb.m_vec, t);
	m_rotation.normalize();

	// lerp the translation part
	m_translation.setInterpolate(qtA.getTranslation(), qtB.getTranslation(), t);
}

//
//	Global instance used by hkQTransformf::getIdentity()

HK_ALIGN_FLOAT( hkFloat32 hkQTransformfIdentity_storage[8] ) =
{
	0, 0, 0, 1, // rotation
	0, 0, 0, 0, // position
};

//
//	Checks for bad values (denormals or infinities)

bool hkQTransformf::isOk() const
{
	return m_translation.isOk<3>() && m_rotation.isOk();
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
