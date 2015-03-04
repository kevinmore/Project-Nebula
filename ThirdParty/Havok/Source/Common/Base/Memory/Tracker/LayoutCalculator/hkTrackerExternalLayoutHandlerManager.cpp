/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Tracker/LayoutCalculator/hkTrackerExternalLayoutHandlerManager.h>

#include <Common/Base/Memory/Tracker/LayoutCalculator/hkTrackerLayoutCalculator.h>

// This singleton is only created and registered if the memory tracker is being used.
#if defined(HK_MEMORY_TRACKER_ENABLE)
HK_SINGLETON_IMPLEMENTATION(hkTrackerExternalLayoutHandlerManager);
#endif

void hkTrackerExternalLayoutHandlerManager::addHandler(const char* name, hkTrackerLayoutHandler* handler)
{
	hkStorageStringMap<hkTrackerLayoutHandler*>::Iterator iter = m_handlers.findKey(name);
	if (m_handlers.isValid(iter))
	{
		hkTrackerLayoutHandler* oldHandler = m_handlers.getValue(iter);
		oldHandler->removeReference();
	}

	// Add it
	handler->addReference();
	m_handlers.insert(name, handler);
}

hkTrackerLayoutHandler* hkTrackerExternalLayoutHandlerManager::getHandler(const char* name) const
{
	return m_handlers.getWithDefault(name, HK_NULL);
}

void hkTrackerExternalLayoutHandlerManager::removeHandler(const char* name)
{
	hkStorageStringMap<hkTrackerLayoutHandler*>::Iterator iter = m_handlers.findKey(name);
	if (m_handlers.isValid(iter))
	{
		hkTrackerLayoutHandler* oldHandler = m_handlers.getValue(iter);
		oldHandler->removeReference();
		m_handlers.remove(iter);
	}
}

void hkTrackerExternalLayoutHandlerManager::addHandlersToLayoutCalculator(hkTrackerLayoutCalculator* layoutCalc)
{
	HK_ASSERT(0x6f436af3, layoutCalc);
	for( hkStorageStringMap<hkTrackerLayoutHandler*>::Iterator iter = m_handlers.getIterator();
		 m_handlers.isValid(iter); 
		 iter = m_handlers.getNext(iter) )
	{
		layoutCalc->addHandler(m_handlers.getKey(iter), m_handlers.getValue(iter));
	}	 
}

void hkTrackerExternalLayoutHandlerManager::clear()
{
	for( hkStorageStringMap<hkTrackerLayoutHandler*>::Iterator iter = m_handlers.getIterator(); 
		 m_handlers.isValid(iter); 
		 iter = m_handlers.getNext(iter))
	{
		hkTrackerLayoutHandler* handler = m_handlers.getValue(iter);
		handler->removeReference();
	}
	m_handlers.clear();
}

hkTrackerExternalLayoutHandlerManager::~hkTrackerExternalLayoutHandlerManager()
{
	clear();
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
