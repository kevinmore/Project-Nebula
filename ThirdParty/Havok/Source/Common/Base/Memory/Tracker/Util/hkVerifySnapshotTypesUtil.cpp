/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Tracker/Util/hkVerifySnapshotTypesUtil.h>
#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>
#include <Common/Base/Reflection/TypeTree/hkTrackerTypeTreeParser.h>
#include <Common/Base/Memory/Tracker/CurrentFunction/hkCurrentFunction.h>

HK_FORCE_INLINE hkBool _orderByType(const hkTrackerSnapshot::ClassAlloc& a, const hkTrackerSnapshot::ClassAlloc& b)
{
	return a.m_typeName < b.m_typeName;
}

static void HK_CALL hkVerifySnapshotTypesUtil_checkType(const hkTrackerSnapshot& snapshot, const char* typeName, const hkTrackerSnapshot::ClassAlloc* allocs, int numAllocs, hkTrackerLayoutCalculator& layoutCalc, hkArray<const hkTrackerSnapshot::ClassAlloc*>& invalidAllocs)
{
	typedef hkTrackerTypeTreeNode Node;
	typedef hkTrackerSnapshot::ClassAlloc ClassAlloc;

	hkTrackerTypeTreeCache& typeCache = *layoutCalc.getTypeCache();

	const Node* type = HK_NULL;

	if (typeName[0] == '!')
	{
		// Use the parser to get the rtti type
		type = hkTrackerTypeTreeParser::parseType(typeName + 1, typeCache);
	}
	else
	{
		// We need to parse this
		hkLocalBuffer<char> buffer(1024);
		// Extract into the buffer
		hkCurrentFunctionUtil::getClassName(typeName, buffer.begin());

		typeName = typeCache.newText(buffer.begin());

		// Parse it
		type = hkTrackerTypeTreeParser::parseType(typeName, typeCache);
	}
	if (type == HK_NULL)
	{
		return;
	}

	// Only deal with classes (named has to be determined to be a class)
	if (!(type->m_type == Node::TYPE_NAMED || 
		  type->m_type == Node::TYPE_CLASS ||
		  type->m_type == Node::TYPE_CLASS_TEMPLATE))
	{
		return;
	}

	hkMemoryTracker& tracker = hkMemoryTracker::getInstance();

	const hkMemoryTracker::ClassDefinition* classDef  = tracker.findClassDefinition(type->m_name);
	if (!classDef)
	{
		return;
	}

	// I need to find things I can check for validity
	const hkTrackerTypeLayout* layout = layoutCalc.getLayout(type);
	if (layout == HK_NULL || layout->m_fullScan )
	{
		return;
	}

	typedef hkTrackerTypeLayout::Member Member;
	hkArray<Member> members;
	// Find the members
	layoutCalc.calcMembers(type, allocs->m_size, members);
}

/* static */void HK_CALL hkVerifySnapshotTypesUtil::verifyTypes(hkTrackerSnapshot& snapshot, hkTrackerLayoutCalculator& layoutCalc)
{
	typedef hkTrackerSnapshot::ClassAlloc ClassAlloc;

	hkArrayBase<ClassAlloc>& allocs = snapshot.getClassAllocs();
	
	// Make names unique
	{
		typedef hkStringMap<const char*> Map;

		Map map;
		for (int i = 0; i < allocs.getSize(); i++)
		{
			ClassAlloc& alloc = allocs[i];

			if (alloc.m_typeName)
			{
				const char* typeName = map.getWithDefault(alloc.m_typeName, HK_NULL);
				if (typeName == HK_NULL)
				{
					typeName = alloc.m_typeName;
					map.insert(typeName, typeName);
				}
				alloc.m_typeName = typeName;
			}
		}
	}

	// Okay, have in order 
	hkSort(allocs.begin(), allocs.getSize(), _orderByType);
	
	{
		hkArray<const ClassAlloc*> invalidAllocs;
		{
			const ClassAlloc* start = allocs.begin();
			const ClassAlloc* end = allocs.end();

			while (start < end)
			{
				const char* typeName = start->m_typeName;
				const ClassAlloc* cur = start + 1;

				// Find a run of the same type
				while (cur < end && cur->m_typeName == typeName) cur++;

				if (typeName)
				{
					// Work out which of this type look invalid

					hkVerifySnapshotTypesUtil_checkType(snapshot, typeName, start, int(cur - start), layoutCalc, invalidAllocs);
				}

				// Next
				start = cur;
			}
		}

		if (invalidAllocs.getSize() > 0)
		{
			for (int i = invalidAllocs.getSize() - 1; i >= 0; i--)
			{
				const ClassAlloc* invalidAlloc = invalidAllocs[i];
				const int removeIndex = int(invalidAlloc - allocs.begin());
				allocs.removeAt(removeIndex);
			}

			snapshot.orderClassAllocs();
		}
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
