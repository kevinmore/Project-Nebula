/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Tracker/Default/hkDefaultMemoryTracker.h>
#include <Common/Base/Memory/System/hkMemorySystem.h>
#include <Common/Base/Reflection/hkClass.h>
#include <Common/Base/Reflection/hkTypeInfo.h>
#include <Common/Base/System/StackTracer/hkStackTracer.h>
#include <Common/Base/Container/StringMap/hkCachedHashMap.cxx>

HK_REFLECTION_DEFINE_STUB_VIRTUAL(hkDefaultMemoryTracker, hkReferencedObject);

/* static */hkMemoryRouter::Allocator* hkDefaultMemoryTrackerAllocator::s_allocator;

static const char s_libraryName[] = "hkBase";

#define HK_TYPE_DEF(x) \
	{ \
		#x,  \
		s_libraryName, \
		hkUint16(sizeof(x)), \
		hkUint8(HK_ALIGN_OF(x)), \
		hkMemoryTracker::TypeDefinition::TYPE_BASIC \
	}

static const hkMemoryTracker::TypeDefinition s_defaultTypes[] = 
{
	HK_TYPE_DEF(signed char), HK_TYPE_DEF(hkInt64), HK_TYPE_DEF(hkInt32), HK_TYPE_DEF(hkInt16), HK_TYPE_DEF(hkInt8), HK_TYPE_DEF(hkUint64), HK_TYPE_DEF(hkUint32), HK_TYPE_DEF(hkUint16),
    HK_TYPE_DEF(hkUint8), HK_TYPE_DEF(hkReal), HK_TYPE_DEF(int), HK_TYPE_DEF(char), HK_TYPE_DEF(short), HK_TYPE_DEF(long), HK_TYPE_DEF(hk_size_t), HK_TYPE_DEF(hkTransform),
    HK_TYPE_DEF(hkVector4), HK_TYPE_DEF(hkRotation), HK_TYPE_DEF(hkMatrix3), HK_TYPE_DEF(hkMatrix4), HK_TYPE_DEF(hkHalf), 
    HK_TYPE_DEF(float), HK_TYPE_DEF(hkBool), HK_TYPE_DEF(hkUFloat8), HK_TYPE_DEF(hkUlong), HK_TYPE_DEF(hkLong), HK_TYPE_DEF(hk_size_t)
};

template class hkCachedHashMap<hkStringMapOperations, hkDefaultMemoryTrackerAllocator>;

// !!!!!!!!!!!!!!!!!!! hkDefaultMemoryTracker::ClassAlloc !!!!!!!!!!!!!!!!!!!!

hkBool hkDefaultMemoryTracker::ClassAlloc::operator==(const ClassAlloc& rhs) const
{
	if (m_typeName != rhs.m_typeName)
	{
		if (m_typeName == HK_NULL || rhs.m_typeName == HK_NULL)
		{
			return false;
		}

		if (hkString::strCmp(m_typeName, rhs.m_typeName) != 0)
		{
			return false;
		}
	}
	return m_size == rhs.m_size && 
			m_ptr == rhs.m_ptr &&
			m_flags == rhs.m_flags;
}

// !!!!!!!!!!!!!!!!!!!!!!!! hkDefaultMemoryTracker !!!!!!!!!!!!!!!!!!!!!!!!!!

hkDefaultMemoryTracker::hkDefaultMemoryTracker(hkMemoryAllocator* allocator):
	m_classAllocFreeList(sizeof(ClassAlloc), sizeof(void*), 4096, allocator)
{
	for (int i = 0; i < (int)HK_COUNT_OF(s_defaultTypes); i++)
	{
		const TypeDefinition& def = s_defaultTypes[i];
		addTypeDefinition(def);
	}

	m_assertOnRemove = HK_NULL;

	m_trackingEnabled = false;
}

static void HK_CALL _dumpStackTrace(const char* text, void* context)
{
	hkOstream& stream = *(hkOstream*)context;
	stream << text;
}

static void _dumpStackTrack(void* ptr, hkOstream& stream)
{
	// dump it... 
	hkMemorySystem* system = &hkMemorySystem::getInstance();
	hkUlong stackTrace[16];
	int stackSize = HK_COUNT_OF(stackTrace);

	hk_size_t allocSize;
	hkResult res = system->getAllocationCallStack(ptr, stackTrace, stackSize, allocSize);

	if (res == HK_SUCCESS)
	{
		hkStackTracer tracer;
		tracer.dumpStackTrace( stackTrace, stackSize, _dumpStackTrace, &stream);
	}
}

hkDefaultMemoryTracker::ClassAlloc* hkDefaultMemoryTracker::_addClassAlloc(const char* typeName, hk_size_t size, void* ptr, int flags)
{
	if (!ptr)
		return HK_NULL;

	m_criticalSection.enter();


#if 0 && defined(HK_DEBUG)
	{
		ClassAlloc* alloc = m_classAllocMap.getWithDefault(ptr, HK_NULL);

		if (alloc)
		{
			// Dump the previous allocation
			hkOfstream stream("trace.txt");
			_dumpStackTrack(alloc->m_ptr, stream);
		}
	}
#endif

	#if defined(HK_PLATFORM_PS3_PPU)
		m_classAllocMap.remove(ptr);
	#endif

#if defined(HK_DEBUG)
	if( ClassAlloc* alloc = m_classAllocMap.getWithDefault(ptr, HK_NULL) )
	{
		HK_ASSERT3(0x23423a4, false, "Memory tracker clash. There is a new object of type " << typeName << " at the same address as an object of type " << alloc->m_typeName );
	}
#endif
		
	//HK_ASSERT(0x23432432, size != 544);

	ClassAlloc* alloc = (ClassAlloc*)m_classAllocFreeList.alloc();
	alloc->m_typeName = typeName;
	alloc->m_size = size;
	alloc->m_ptr = ptr;
	alloc->m_flags = flags;

	m_classAllocMap.insert(ptr, alloc);
	if (m_trackingEnabled)
	{
		// If tracking is enabled add it to the created map.
		m_createdMap.insert(ptr, alloc);
	}
	
	m_criticalSection.leave();
	return alloc;
}

void hkDefaultMemoryTracker::_removeClassAlloc(void* ptr)
{
	if (!ptr)
		return;

	m_criticalSection.enter();

#if 0
	{
		int index = m_allocArray.indexOf(ptr);
		HK_ASSERT(0x2342a343, index >= 0);
		m_allocArray.removeAt(index);
	}
#endif

	{
		ClassAllocMapType::Iterator iter = m_classAllocMap.findKey(ptr);
		if (m_classAllocMap.isValid(iter))
		{
			ClassAlloc* alloc = m_classAllocMap.getValue(iter);
			m_classAllocFreeList.free(alloc);
			m_classAllocMap.remove(iter);

			if (alloc == m_assertOnRemove)
			{
				HK_ASSERT2(0x2323432, false, "Hit assert on remove");
				m_assertOnRemove = HK_NULL;
			}
		}
		else
		{
			#if !defined(HK_PLATFORM_PS3_PPU)
				HK_ASSERT2(0x23434234, false, "Pointer not found");
			#endif
		}
	}

	if (m_trackingEnabled)
	{
		// If its in the created map, then it was created since the history was last cleared
		// so we don't need to worry about it in the deleted map.
		// Else remove from the deleted map
		CreatedMap::Iterator iter = m_createdMap.findKey(ptr);
		if (m_createdMap.isValid(iter))
		{
			m_createdMap.remove(iter);
		}
		else
		{
			m_deletedMap.insert(ptr, 1);
		}
	}

	m_criticalSection.leave();
}

void hkDefaultMemoryTracker::clearTrackingHistory()
{
	m_deletedMap.clear();
	m_createdMap.clear();
}

const hkDefaultMemoryTracker::ClassAlloc* hkDefaultMemoryTracker::findClassAlloc(void* ptr) const
{
	m_criticalSection.enter();

	ClassAlloc* alloc = m_classAllocMap.getWithDefault(ptr, HK_NULL);

	m_criticalSection.leave();
	return alloc;
}

void hkDefaultMemoryTracker::setAssertRemoveAlloc(const ClassAlloc* alloc)
{
	m_criticalSection.enter();

	if (alloc)
	{
		HK_ASSERT(0x2554a35a, m_classAllocMap.getWithDefault(alloc->m_ptr, HK_NULL) == alloc);
	}
	m_assertOnRemove = alloc;

	m_criticalSection.leave();
}


void hkDefaultMemoryTracker::onNewReferencedObject(const char* typeName, hk_size_t size, void* ptr)
{
	_addClassAlloc(typeName, size, ptr, ClassAlloc::FLAG_REFERENCED_OBJECT);
}

void hkDefaultMemoryTracker::onDeleteReferencedObject(void* ptr)
{
	_removeClassAlloc(ptr);
}


void hkDefaultMemoryTracker::onNewObject(const char* typeName, hk_size_t size, void* ptr)
{
	_addClassAlloc(typeName, size, ptr, 0);
}

void hkDefaultMemoryTracker::onDeleteObject(void* ptr)
{
	_removeClassAlloc(ptr);
}

void hkDefaultMemoryTracker::onNewRaw(const char* name, hk_size_t size, void* ptr)
{
	_addClassAlloc(name, size, ptr, 0);
}

void hkDefaultMemoryTracker::onDeleteRaw(void* ptr)
{
	_removeClassAlloc(ptr);
}

void hkDefaultMemoryTracker::addTypeDefinition(const TypeDefinition& defIn)
{
    hkCriticalSectionLock lock(&m_criticalSection);

	if (findTypeDefinition(defIn.m_typeName))
	{
		// Its already added
		return;
	}

	m_nameTypeMap.insert(defIn.m_typeName, &defIn);
}

void hkDefaultMemoryTracker::clearTypeDefinitions()
{
	hkCriticalSectionLock lock(&m_criticalSection);
	m_nameTypeMap.reset();
}

const hkMemoryTracker::TypeDefinition* hkDefaultMemoryTracker::findTypeDefinition(const char* typeName)
{
	hkCriticalSectionLock lock(&m_criticalSection);
	return m_nameTypeMap.getWithDefault(typeName, HK_NULL);
}

hk_size_t hkDefaultMemoryTracker::getTypeDefinitions(const TypeDefinition** typeDefinitions)
{
	if (typeDefinitions)
	{
		NameTypeMap::Iterator iter = m_nameTypeMap.getIterator();
		for (; m_nameTypeMap.isValid(iter); iter = m_nameTypeMap.getNext(iter))
		{
			*typeDefinitions = m_nameTypeMap.getValue(iter);
			typeDefinitions++;
		}
	}

	return m_nameTypeMap.getSize();
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
