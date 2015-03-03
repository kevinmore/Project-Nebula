/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Serialize/hkSerialize.h>
#include <Common/Base/UnitTest/hkUnitTest.h>

#define DO_TEST 0

#if DO_TEST
#include <Behavior/Behavior/hkbBehavior.h> 
#include <Behavior/Behavior/Event/hkbEventPayload.h> 
#include <Behavior/Behavior/Generator/Clip/hkbClipGenerator.h> 
#endif

int tagfileTest()
{
#if DO_TEST
	{
		hkbRealEventPayload payload;
		hkbClipTrigger trigger;
		trigger.m_event.m_id = 11;
		trigger.m_event.setPayload( &payload );

		hkbClipTriggerArray triggerArray;
		triggerArray.m_triggers.pushBack( trigger );

		hkArray<char> buf;
		hkSerializeUtil::saveTagfile(&triggerArray, hkbClipTriggerArrayClass, hkOstream(buf).getStreamWriter(), true );

		hkRefPtr<hkResource> res; res.setAndDontIncrementRefCount( hkSerializeUtil::load(buf.begin(), buf.getSize()) );
		hkbClipTriggerArray* cta = res->getContents<hkbClipTriggerArray>();
		HK_TEST( cta->m_triggers.getSize() == 1);
		HK_TEST( cta->m_triggers[0].m_event.getId() == 11);
		HK_TEST( cta->m_triggers[0].m_event.getPayload() != HK_NULL);
	}
#endif

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(tagfileTest, "Fast", "Common/Test/UnitTest/Serialize/", __FILE__ );

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
