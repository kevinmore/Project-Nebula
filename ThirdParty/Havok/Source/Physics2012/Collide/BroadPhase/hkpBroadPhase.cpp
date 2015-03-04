/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Collide/hkpCollide.h>

#include <Common/Base/Thread/CriticalSection/hkCriticalSection.h>

#include <Physics2012/Collide/BroadPhase/hkpBroadPhase.h>


hkpBroadPhase::hkpBroadPhase(BroadPhaseType type, int size, int caps)
	:	m_type((hkUint16)type)
	,	m_size((hkUint16)size)
	,	m_caps((hkUint32)caps)
{
	m_criticalSection = HK_NULL;
	m_multiThreadCheck.disableChecks();
}

void hkpBroadPhase::enableMultiThreading(int spinCountForCriticalSection)
{
	if (!m_criticalSection)	
	{
		m_criticalSection = new hkCriticalSection(spinCountForCriticalSection);
		m_multiThreadCheck.enableChecks();
	}
}

hkpBroadPhase::~hkpBroadPhase()
{
	if ( m_criticalSection )
	{
		delete m_criticalSection;
		m_criticalSection = HK_NULL;
	}
}

void hkpBroadPhase::lockImplementation()
{
	m_criticalSection->enter();
	markForWrite();
}

void hkpBroadPhase::unlockImplementation()
{
	unmarkForWrite();
	m_criticalSection->leave();
}


// Set tree broad phase functions to null. Must be registered if functionality is required.
hkpBroadPhase::createSweepAndPruneBroadPhaseFunc hkpBroadPhase::s_createSweepAndPruneBroadPhaseFunction = HK_NULL;
hkpBroadPhase::createTreeBroadPhaseFunc hkpBroadPhase::s_createTreeBroadPhaseFunction = HK_NULL;
hkpBroadPhase::updateTreeBroadPhaseFunc hkpBroadPhase::s_updateTreeBroadPhaseFunction = HK_NULL;
hkpBroadPhase::updateTreeBroadPhaseFunc hkpBroadPhase::s_updateTreeBroadPhaseFunction32 = HK_NULL;

/*
 * Havok SDK - Product file, BUILD(#20130912)
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
