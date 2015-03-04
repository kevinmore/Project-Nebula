/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/UnitTest/hkUnitTest.h>

#include <Common/Base/Memory/Tracker/hkMemoryTracker.h>
#include <Common/Base/Reflection/TypeTree/hkTrackerTypeTreeCache.h>
#include <Common/Base/Reflection/TypeTree/hkTrackerTypeTreeParser.h>

#include <Common/Base/Fwd/hkcstdio.h>


int ParserTest_main()
{
	hkTrackerTypeTreeCache cache;
	hkMemoryTracker& memoryTracker = hkMemoryTracker::getInstance();
	hk_size_t numTypes = memoryTracker.getTypeDefinitions(HK_NULL);

	hkArray<const hkMemoryTracker::TypeDefinition*> types;
	types.setSize(int(numTypes));
	memoryTracker.getTypeDefinitions(types.begin());

	for (int i = 0; i < types.getSize(); i++)
	{
		// Look at the members
		const hkMemoryTracker::TypeDefinition* typeDef = types[i];

		if (typeDef->m_type == hkMemoryTracker::TypeDefinition::TYPE_CLASS)
		{
			const hkMemoryTracker::ClassDefinition* clsDef = static_cast<const hkMemoryTracker::ClassDefinition*>(typeDef);

			for (int j = 0; j < clsDef->m_numMembers; j++)
			{
				const hkMemoryTracker::Member& member = clsDef->m_members[j];
				const char* typeName = member.m_typeName;
				const hkTrackerTypeTreeNode* node;
				if (cache.getTypeExpressionTree(typeName, &node))
				{
					continue;
				}

				// Try parsing it 
				node = hkTrackerTypeTreeParser::parseNewType( hkSubString(typeName), cache);
				HK_TEST2(node != HK_NULL, "Couldn't parse type for " << typeName);
				
			}
		}
	}

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(ParserTest_main, "Fast", "Common/Test/UnitTest/Base/", __FILE__     );

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
