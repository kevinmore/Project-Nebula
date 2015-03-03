/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

inline const hkVector4& hknpSpringAction::getPositionAinA()
{
	return m_positionAinA;
}

inline const hkVector4& hknpSpringAction::getPositionBinB()
{
	return m_positionBinB;
}

inline void hknpSpringAction::setStrength(hkReal str)
{
  m_strength = str;
}

inline hkReal hknpSpringAction::getStrength()
{
	return m_strength;
}

inline void hknpSpringAction::setDamping(hkReal damp)
{
  m_damping = damp;
}

inline hkReal hknpSpringAction::getDamping()
{
	return m_damping;
}

inline void hknpSpringAction::setRestLength(hkReal restLength)
{
  m_restLength = restLength;
}

inline hkReal hknpSpringAction::getRestLength()
{
	return m_restLength;
}

inline void hknpSpringAction::setOnCompression(hkBool onCompression)
{
  m_onCompression = onCompression;
}

inline hkBool hknpSpringAction::getOnCompression()
{
	return m_onCompression;
}

inline void hknpSpringAction::setOnExtension(hkBool onExtension)
{
  m_onExtension = onExtension;
}

inline hkBool hknpSpringAction::getOnExtension()
{
	return m_onExtension;
}

inline const hkVector4& hknpSpringAction::getLastForce()
{
	return m_lastForce;
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
