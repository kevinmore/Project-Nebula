/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Physics2012/Utilities/CharacterControl/CharacterProxy/Multithreaded/hkpCharacterProxyJobs.h>

#if defined (HK_PLATFORM_HAS_SPU)
#include <Physics2012/Collide/BroadPhase/hkpBroadPhase.h>
#endif

// =====================================================================================================================
// CHARACTER PROXY INTEGRATE
// =====================================================================================================================

// Size of hkpCharacterProxyIntegrateCommand has to be a multiple of 16.
#if !defined(HK_COMPILER_MWERKS) && (HK_NATIVE_ALIGNMENT==16)
HK_COMPILE_TIME_ASSERT( (sizeof(hkpCharacterProxyIntegrateCommand) & 0xf) == 0 );
#endif

#if !defined(HK_PLATFORM_SPU)
void hkpCharacterProxyIntegrateJob::setRunsOnSpuOrPpu()
{
#if defined (HK_PLATFORM_HAS_SPU)

	if( m_broadphase )
	{
		const hkpBroadPhase* implBp = m_broadphase->getCapabilityDelegate(hkpBroadPhase::SUPPORT_SPU_CHAR_PROXY_INT);
		if ( !implBp )
		{
			HK_WARN_ONCE(0xaf35e144, "This broadphase query is not supported on SPU. Moving job onto PPU. Performance loss likely.");
			m_jobSpuType = HK_JOB_SPU_TYPE_DISABLED;
			return;
		}

		m_broadphase = implBp;
	}

	if( m_createCdPointCollectorOnCpuFunc )
	{
		HK_WARN_ONCE(0xaf15e143, "There is an hkCreateAllCdPointCollectorOnCpuFunc in this job. The job has to therefore move to the PPU. Performance loss likely. A custom addCdPoint function may be implemented on SPU by calling hkpFixedBufferCdPointCollector::registerCustomAddCdPointFunction().");
		m_jobSpuType = HK_JOB_SPU_TYPE_DISABLED;
		return;
	}

	for (int commandIdx = 0; commandIdx < m_numCommands; commandIdx++)
	{
		const hkpCharacterProxyIntegrateCommand* command = &m_commandArray[commandIdx];
		if ( command->m_collidable->m_forceCollideOntoPpu != 0 )
		{
			HK_WARN_ONCE(0xaf15e143, "There's at least one shape in the job that is not supported on the SPU. The job has to therefore move to the PPU. Performance loss likely. Consider moving the unsupported shape(s) into a separate job.");
			m_jobSpuType = HK_JOB_SPU_TYPE_DISABLED;
			return;
		}

		if ( command->m_character->m_listeners.getSize() > 0 )
		{
			HK_WARN_ONCE(0xaf2647fe, "Callbacks from PPU-side listeners will not be fired SPU-side. The job has to therefore move to the PPU. Performance loss likely. Consider implementing callback code SPU-side using hkpSpuCharacterProxyUtil::registerCustomFunctions()." );
			m_jobSpuType = HK_JOB_SPU_TYPE_DISABLED;
			return;
		}
	}
	
	m_jobSpuType = HK_JOB_SPU_TYPE_ENABLED;
#endif
}
#endif

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
