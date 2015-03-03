/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/Dynamics/Motion/hknpMotionProperties.h>

HK_FORCE_INLINE void hknpDeactivationState::resetCounters(int random)
{
	m_frameCounter = random&3;
	if (m_numDeactivationChecks!=0xff)
	{
		m_numDeactivationChecks = 0;
#ifdef DEACTIVATION_DEBUG_ACTIVATION_REASON
		m_activationReason = ACTIVATIONREASON_NOT_FULLY_TESTED;
#endif
	}
}

HK_FORCE_INLINE int hknpDeactivationState::getNumInactiveFrames() const
{
	return m_numDeactivationChecks;
}

HK_FORCE_INLINE hkBool32 hknpDeactivationState::isDeactivationEnabled() const
{
	return m_numDeactivationChecks!=0xff;
}

HK_FORCE_INLINE hkUint32 hknpDeactivationState::isOkToDeactivateAsMask(const hknpMotionProperties& motionProperties) const
{
	int numActiveFramesLeft = int((motionProperties.m_numDeactivationFrequencyPasses<<2)-m_numDeactivationChecks); //when negative, disable (multiply by motionProperties.m_numQuadDeactivationFrames)
#ifdef DEACTIVATION_FREEFALL_DETECTION
	int keepActiveOffset = (int(m_activationFlags.get())<<8);//when any flag is set, do not deactivate
	hkUint32 result = hkUint32( (numActiveFramesLeft + keepActiveOffset) >> 31 );
#else
	hkUint32 result = hkUint32( numActiveFramesLeft >> 31 );
#endif
	return result;
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
