/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>

#include <Physics2012/Utilities/CharacterControl/CharacterProxy/Multithreaded/Cpu/hkpCpuCharacterProxyIntegrateJob.h>
#include <Physics2012/Utilities/CharacterControl/CharacterProxy/Multithreaded/hkpCharacterProxyJobs.h>
#include <Physics2012/Utilities/CharacterControl/CharacterProxy/Multithreaded/Cpu/hkpCpuCharacterProxyUtil.h>
#include <Physics2012/Utilities/CharacterControl/CharacterProxy/Multithreaded/Cpu/hkpCpuCharacterProxyCollector.h>

hkJobQueue::JobStatus HK_CALL hkCpuCharacterProxyIntegrateJob( hkJobQueue& jobQueue, hkJobQueue::JobQueueEntry& nextJobOut )
{
	HK_TIMER_BEGIN("CharProxyIntegrate", HK_NULL);

	const hkpCharacterProxyIntegrateJob& characterProxyJob = reinterpret_cast<hkpCharacterProxyIntegrateJob&>( nextJobOut );

	hkpCharacterProxyIntegrateCommand* command = const_cast<hkpCharacterProxyIntegrateCommand*>( characterProxyJob.m_commandArray );
	
	hkpCpuCharacterProxyCollector *castCollector;
	hkpCpuCharacterProxyCollector *startCollector;
	{
		if( characterProxyJob.m_createCdPointCollectorOnCpuFunc )
		{
			castCollector = characterProxyJob.m_createCdPointCollectorOnCpuFunc();
			startCollector = characterProxyJob.m_createCdPointCollectorOnCpuFunc();		
		}
		else
		{
			castCollector = new hkpCpuCharacterProxyCollector();
			startCollector =  new hkpCpuCharacterProxyCollector();		
		}
	}	

	for (int i = 0; i < characterProxyJob.m_numCommands ; i++ )
	{
		// The collector will filter out collisions with the character's collidable which is set here
		castCollector->setCharactersCollidable( command->m_collidable );
		startCollector->setCharactersCollidable( command->m_collidable );

		hkStepInfo stepInfo;
		{
			stepInfo.m_deltaTime = characterProxyJob.m_deltaTime;
			stepInfo.m_invDeltaTime = characterProxyJob.m_invDeltaTime;	
		}

		command->m_character->integrateImplementation( stepInfo, characterProxyJob.m_worldGravity, command, *castCollector, *startCollector );

		command++;
	}

	delete startCollector;
	delete castCollector;

	HK_TIMER_END();

	return jobQueue.finishJobAndGetNextJob( &nextJobOut, nextJobOut, hkJobQueue::WAIT_FOR_NEXT_JOB );
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
