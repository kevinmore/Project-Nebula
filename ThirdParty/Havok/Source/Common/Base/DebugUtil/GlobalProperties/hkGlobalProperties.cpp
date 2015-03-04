/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/DebugUtil/GlobalProperties/hkGlobalProperties.h>
#include <Common/Base/Thread/CriticalSection/hkCriticalSection.h>

HK_SINGLETON_IMPLEMENTATION(hkGlobalProperties);

// Used to buffer ::set call before the singleton is properly initialized.
struct hkGlobalProperties_PendingEntry
{
	const void*	m_key;
	const void*	m_value;
};

static int								hkGlobalProperties_initialized = 0;
static int								hkGlobalProperties_numPending = 0;
static hkGlobalProperties_PendingEntry	hkGlobalProperties_pending[32];

//
hkGlobalProperties::hkGlobalProperties()
{	
	m_lock = new hkCriticalSection();
	m_data.reserve(1024);

	hkGlobalProperties_initialized = 1;
	for(int i=0; i<hkGlobalProperties_numPending; ++i)
	{
		set(hkGlobalProperties_pending[i].m_key, hkGlobalProperties_pending[i].m_value);
	}
	hkGlobalProperties_numPending = 0;	
}

//
hkGlobalProperties::~hkGlobalProperties()
{
	delete m_lock;
	hkGlobalProperties_initialized = 0;
}

//
void		hkGlobalProperties::set(const void* key, const void* value)
{
	if(hkGlobalProperties_initialized)
	{
		m_lock->enter();
		m_data.insert(key, value);
		m_lock->leave();
	}
	else
	{
		hkGlobalProperties_PendingEntry& entry = hkGlobalProperties_pending[hkGlobalProperties_numPending++];
		entry.m_key		=	key;
		entry.m_value	=	value;
	}
}

//
const void*	hkGlobalProperties::get(const void* key, const void* defaultValue)
{
	if(hkGlobalProperties_initialized)
	{
		m_lock->enter();
		const void* value = m_data.getWithDefault(key,defaultValue);
		m_lock->leave();
		return value;
	}
	else
	{
		return defaultValue;
	}
}

//
void		hkGlobalProperties::clear(const void* key)
{
	if(hkGlobalProperties_initialized)
	{
		m_lock->enter();
		m_data.remove(key);
		m_lock->leave();
	}	
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
