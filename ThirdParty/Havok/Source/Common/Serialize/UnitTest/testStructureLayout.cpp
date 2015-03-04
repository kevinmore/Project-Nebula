/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>
#include <Common/Base/Container/String/hkStringBuf.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Serialize/UnitTest/PlatformClassList.h>
#include <Common/Serialize/Util/hkBuiltinTypeRegistry.h>

static void testLayouts(const hkClass*const* origClasses, const hkClass*const* computedClasses, hkStructureLayout::LayoutRules rules, const char* testType)
{
	const hkClass* const* origp = origClasses;
	const hkClass* const* compp = computedClasses;
	while(*origp!= HK_NULL && *compp != HK_NULL)
	{
		const hkClass* orig = *origp;
		const hkClass* comp = *compp;

		if( orig->getFlags().get(hkClass::FLAGS_NOT_SERIALIZABLE) == 0 )
		{
			hkStringBuf desc;
			HK_TEST( orig->getNumMembers() == comp->getNumMembers() );
			for( int i = 0; i < orig->getNumMembers(); ++i )
			{
				int origOff = orig->getMember(i).getOffset();
				int compOff = comp->getMember(i).getOffset();
				
				if( origOff != compOff )
				{
					desc.printf("%s: %s %i %i %i %s [%d, %d, %d, %d]\n", orig->getName(), testType, i, origOff, compOff, orig->getMember(i).getName(), rules.m_bytesInPointer, rules.m_littleEndian, rules.m_reusePaddingOptimization, rules.m_emptyBaseClassOptimization);
				}
				HK_TEST2( origOff == compOff, desc.cString() );
			}
			hkStringBuf errMsg; errMsg.printf("%s size differs native vs computed %i %i [%i, %i, %i, %i]", orig->getName(), orig->getObjectSize(), comp->getObjectSize(), rules.m_bytesInPointer, rules.m_littleEndian, rules.m_reusePaddingOptimization, rules.m_emptyBaseClassOptimization );
			HK_TEST2( orig->getObjectSize() == comp->getObjectSize(), errMsg.cString() );
		}
		
		++origp;
		++compp;
	}
	HK_TEST( *origp == HK_NULL && *compp == HK_NULL );
}

static int testStructureLayout()
{
#ifdef HK_REAL_IS_DOUBLE
	HK_REPORT("This unit test is disabled in DP builds, as reflection system cannot distinguish 32 and 64 Bit floating point numbers.");
#else
	// compiled in are same as computed
	{
		PlatformClassList classes( hkBuiltinTypeRegistry::StaticLinkedClasses );		
		classes.computeOffsets( hkStructureLayout::HostLayoutRules );
		testLayouts( hkBuiltinTypeRegistry::StaticLinkedClasses, classes.m_copies.begin(), hkStructureLayout::HostLayoutRules, "native vs computed" );
	}
#endif
	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(testStructureLayout, "Fast", "Common/Test/UnitTest/Serialize/", __FILE__     );

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
