/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Math/SweptTransform/hkSweptTransformd.h>

void hkSweptTransformd::approxTransformAt( hkSimdDouble64Parameter time, hkTransformd& transformOut ) const
{
#if defined(HK_PLATFORM_PS3_SPU)
	// on SPU force outline
	const hkSimdDouble64 dt = getInterpolationValue(time);

	hkQuaterniond q;
	q.m_vec.setInterpolate( m_rotation0.m_vec, m_rotation1.m_vec, dt );
	q.normalize();

	transformOut.setRotation( q );
	transformOut.getTranslation().setInterpolate( m_centerOfMass0, m_centerOfMass1, dt);

	hkVector4d centerShift;
	centerShift.setRotatedDir( transformOut.getRotation(), m_centerOfMassLocal); // outline
	transformOut.getTranslation().sub( centerShift );
#else
	_approxTransformAt(time, transformOut );
#endif
}

void hkSweptTransformd::approxTransformAt( hkTime time, hkTransformd& transformOut ) const
{
#if defined(HK_PLATFORM_PS3_SPU)
	// on SPU force outline
	const hkSimdDouble64 dt = hkSimdDouble64::fromFloat(getInterpolationValue(time)); // late transition

	hkQuaterniond q;
	q.m_vec.setInterpolate( m_rotation0.m_vec, m_rotation1.m_vec, dt );
	q.normalize();

	transformOut.setRotation( q );
	transformOut.getTranslation().setInterpolate( m_centerOfMass0, m_centerOfMass1, dt);

	hkVector4d centerShift;
	centerShift.setRotatedDir( transformOut.getRotation(), m_centerOfMassLocal); // outline
	transformOut.getTranslation().sub( centerShift );
#else
	_approxTransformAt(time, transformOut );
#endif
}

void hkSweptTransformd::initSweptTransform( hkVector4dParameter position, hkQuaterniondParameter rotation )
{
	// need the 0 here?
	m_centerOfMass0.setXYZ_0( position ); // w base time
	m_centerOfMass1.setXYZ_0( position ); // invDeltaTime

	m_rotation0 = rotation;
	m_rotation1 = rotation;
	m_centerOfMassLocal.setZero();
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
