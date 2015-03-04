/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>

#include <Common/Base/Reflection/hkTypeInfo.h>

#if 0
#define TRACE(A) A
#else
#define TRACE(A)
#endif

void hkTypeInfo::finishLoadedObject(void* ptr, int finishFlag) const
{
	if (m_finishLoadedObjectFunction)
	{
		m_finishLoadedObjectFunction(ptr, finishFlag);
	}

#if defined(HK_MEMORY_TRACKER_ENABLE)
	if( hkMemoryTracker* tracker = hkMemoryTracker::getInstancePtr() )
	{
		if (isVirtual())
		{
			tracker->onNewReferencedObject(m_scopedName, getSize(), ptr);
		}
		else
		{
			tracker->onNewObject(m_scopedName, getSize(), ptr);
		}
	}
#endif
}

void hkTypeInfo::finishLoadedObjectWithoutTracker(void* ptr, int finishFlag) const
{
	if (m_finishLoadedObjectFunction)
	{
		m_finishLoadedObjectFunction(ptr, finishFlag);
	}
}

void hkTypeInfo::cleanupLoadedObject(void* ptr) const
{
	if (m_cleanupLoadedObjectFunction)
	{
		TRACE(printf("-dtor\t%s at %p...", getTypeName(), ptr));
		m_cleanupLoadedObjectFunction(ptr);
		TRACE(printf("done.\n"));
	}

#if defined(HK_MEMORY_TRACKER_ENABLE)
	if( hkMemoryTracker* tracker = hkMemoryTracker::getInstancePtr() )
	{
		if (isVirtual())
		{
			tracker->onDeleteReferencedObject(ptr);
		}
		else
		{
			tracker->onDeleteObject(ptr);
		}
	}
#endif
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
