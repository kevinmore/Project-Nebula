/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */


HK_FORCE_INLINE hknpCharacterSurfaceInfo::hknpCharacterSurfaceInfo()
	: m_isSurfaceDynamic(false)
	, m_supportedState(SUPPORTED)
	, m_surfaceDistanceExcess(0.0f)
{
	m_surfaceNormal.set( 0.0f, 0.0f, 1.0f, 0.0f);
	m_surfaceVelocity.set( 0.0f, 0.0f, 0.0f, 0.0f);
}

HK_FORCE_INLINE hknpCharacterSurfaceInfo::hknpCharacterSurfaceInfo(
	hkVector4Parameter up, hkVector4Parameter velocity, const SupportedState state, hkBool isDynamic )
	: m_isSurfaceDynamic(false)
	, m_supportedState(state)
	, m_surfaceDistanceExcess(0.0f)
	, m_surfaceNormal(up)
	, m_surfaceVelocity(velocity)
{

}

HK_FORCE_INLINE void hknpCharacterSurfaceInfo::set( const hknpCharacterSurfaceInfo& other )
{
	m_supportedState = other.m_supportedState;
	m_surfaceDistanceExcess = other.m_surfaceDistanceExcess;
	m_isSurfaceDynamic = other.m_isSurfaceDynamic;
	m_surfaceNormal = other.m_surfaceNormal;
	m_surfaceVelocity = other.m_surfaceVelocity;
}

HK_FORCE_INLINE hkBool hknpCharacterSurfaceInfo::isValid() const
{
	return ((m_supportedState == UNSUPPORTED) || (m_surfaceNormal.isOk<3>() && m_surfaceNormal.isNormalized<3>() &&
			m_surfaceVelocity.isOk<3>() && hkMath::isFinite(m_surfaceDistanceExcess)));
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
