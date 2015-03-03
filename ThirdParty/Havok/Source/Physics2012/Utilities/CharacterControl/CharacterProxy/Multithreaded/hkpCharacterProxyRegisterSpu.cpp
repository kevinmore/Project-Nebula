/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Physics2012/Utilities/CharacterControl/CharacterProxy/Multithreaded/hkpCharacterProxyJobs.h>
#include <Physics2012/Utilities/CharacterControl/CharacterProxy/Multithreaded/hkpCharacterProxyJobQueueUtils.h>
#include <Physics2012/Utilities/CharacterControl/CharacterProxy/Multithreaded/Cpu/hkpCpuCharacterProxyIntegrateJob.h>

#if !defined (HK_PLATFORM_SPU)

static hkJobQueue::ProcessJobFunc s_characterProxyProcessFuncs[hkpCharacterProxyJob::CHARACTER_PROXY_JOB_END];

void hkpCharacterProxyJobQueueUtils::registerWithJobQueue( hkJobQueue* jobQueue )
{
#if defined(HK_PLATFORM_MULTI_THREAD) && (HK_CONFIG_THREAD == HK_CONFIG_MULTI_THREADED)
	hkJobQueue::hkJobHandlerFuncs jobHandlerFuncs;
	jobHandlerFuncs.m_popJobFunc = popCharacterProxyJob;
	jobHandlerFuncs.m_finishJobFunc = finishCharacterProxyJob;

	jobHandlerFuncs.initProcessJobFuncs( s_characterProxyProcessFuncs, HK_COUNT_OF(s_characterProxyProcessFuncs) ) ;

	jobHandlerFuncs.registerProcessJobFunc( hkpCharacterProxyJob::CHARACTER_PROXY_JOB_INTEGRATE,
		hkCpuCharacterProxyIntegrateJob);

	jobQueue->registerJobHandler( HK_JOB_TYPE_CHARACTER_PROXY, jobHandlerFuncs );

#if defined(HK_PLATFORM_HAS_SPU)

#if defined (HK_PLATFORM_PS3_PPU)
	extern char _binary_hkpSpursCharacterProxy_elf_start[];
	void* elf =	_binary_hkpSpursCharacterProxy_elf_start;
#else
	void* elf = (void*)HK_JOB_TYPE_CHARACTER_PROXY;
#endif // defined (HK_PLATFORM_PS3_PPU)
	jobQueue->registerSpuElf( HK_JOB_TYPE_CHARACTER_PROXY, elf );

#endif // defined(HK_PLATFORM_HAS_SPU)

#endif // defined(HK_PLATFORM_MULTI_THREAD) && (HK_CONFIG_THREAD == HK_CONFIG_MULTI_THREADED)

}
#endif // !defined (HK_PLATFORM_SPU)

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
