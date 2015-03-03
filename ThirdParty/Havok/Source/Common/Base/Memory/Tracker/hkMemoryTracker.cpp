/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Tracker/hkMemoryTracker.h>
#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>
#include <Common/Base/Container/SubString/hkSubString.h>
#include <Common/Base/Memory/System/hkMemorySystem.h>
#include <Common/Base/Memory/Tracker/Default/hkDefaultMemoryTracker.h>

class hkDummyMemoryTrackerImpl: public hkMemoryTracker
{
	public:
		virtual void onNewReferencedObject(const char* typeName, hk_size_t size, void* ptr) {}
		virtual void onDeleteReferencedObject(void* ptr) {}
		virtual void onNewObject(const char* typeName, hk_size_t size, void* ptr) {}
		virtual void onDeleteObject(void* ptr) {}
		virtual void onNewRaw(const char* name, hk_size_t size, void* ptr) {}
		virtual void onDeleteRaw(void* ptr) {}
		virtual void addTypeDefinition(const TypeDefinition& def) {}
		virtual const TypeDefinition* findTypeDefinition(const char* name) { return HK_NULL; }
		virtual hk_size_t getTypeDefinitions(const TypeDefinition** typeDefinitions) { return 0; }
		virtual void clearTypeDefinitions() {}
};

/* static */hkMemoryTracker* hkMemoryTracker::s_singleton = HK_NULL;

/* static */hkMemoryTracker& HK_CALL hkMemoryTracker::getInstance() 
{
	if (s_singleton)
	{
		return *s_singleton;
	}

	static hkDummyMemoryTrackerImpl dummy;
	return dummy; 
}

/* static */void HK_CALL hkMemoryTracker::setInstance(hkMemoryTracker* tracker) 
{ 
	s_singleton = tracker;
}

const hkMemoryTracker::TypeDefinition* hkMemoryTracker::findTypeDefinition(const hkSubString& name)
{
	hkLocalBuffer<char> buffer(name.length() + 1);
	hkString::strNcpy(buffer.begin(), name.m_start, name.length());
	buffer[name.length()] = 0;
	return findTypeDefinition(buffer.begin());
}

namespace // anonymous
{
	struct Allocation
	{
		void* m_start;							///< The start of the block
		hk_size_t m_size;						///< Total size of the allocation
	};

	typedef hkArray<Allocation, hkDefaultMemoryTrackerAllocator> AllocArrayType;
}

hkBool hkMemoryTracker::isBasicType(const hkSubString& name)
{
	const TypeDefinition* def = findTypeDefinition(name);
	if (!def)
	{
		return false;
	}

	return def->m_type == TypeDefinition::TYPE_BASIC;
}

const hkMemoryTracker::ClassDefinition* hkMemoryTracker::findClassDefinition(const hkSubString& name)
{
	const TypeDefinition* typeDef = findTypeDefinition(name);
	if (typeDef && typeDef->m_type == TypeDefinition::TYPE_CLASS)
	{
		return static_cast<const ClassDefinition*>(typeDef);
	}
	return HK_NULL;
}

const hkMemoryTracker::ClassDefinition* hkMemoryTracker::findClassDefinition(const char* name)
{
	const TypeDefinition* typeDef = findTypeDefinition(name);
	if (typeDef && typeDef->m_type == TypeDefinition::TYPE_CLASS)
	{
		return static_cast<const ClassDefinition*>(typeDef);
	}
	return HK_NULL;
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
