/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/UnitTest/hkUnitTest.h>

#include <Common/Base/Types/hkSignalSlots.h>

namespace UnitTest
{
	HK_DECLARE_SIGNAL(TestSignal, hkSignal1<int>);

	struct SimpleSlot
	{
		void onTestSignal(int i) { HK_TEST(i == 0xabcd); }
	};

	struct UnsubscribingSlot
	{
		UnsubscribingSlot(TestSignal& testSignal) : m_testSignal(testSignal) {}

		void onTestSignal(int i) 
		{
			HK_TEST(i == 0xabcd);
			m_testSignal.unsubscribeAll(this);
		}

		TestSignal& m_testSignal;
	};
}

int signalUnitTest_main()
{
	UnitTest::TestSignal testSignal;
	HK_TEST(testSignal.getSlots() == HK_NULL);
	HK_TEST(testSignal.getNumSubscriptions() == 0);

	{
		UnitTest::SimpleSlot sSlot;
		for (int i = 0; i < 10; ++i)
		{
			HK_SUBSCRIBE_TO_SIGNAL(testSignal, &sSlot, SimpleSlot);
		}
		HK_TEST(testSignal.getNumSubscriptions() == 10);

		testSignal.fire(0xabcd);
		HK_TEST(testSignal.getNumSubscriptions() == 10);

		testSignal.unsubscribeAll(&sSlot);
		HK_TEST(testSignal.getNumSubscriptions() == 0);
	}

	{
		UnitTest::SimpleSlot sSlot;
		UnitTest::UnsubscribingSlot uSlot(testSignal);
		for (int i = 0; i < 10; ++i)
		{
			HK_SUBSCRIBE_TO_SIGNAL(testSignal, &sSlot, SimpleSlot);
			HK_SUBSCRIBE_TO_SIGNAL(testSignal, &uSlot, UnsubscribingSlot);
		}

		for (int i = 0; i < 10; ++i)
		{
			testSignal.unsubscribe(&sSlot, &UnitTest::SimpleSlot::onTestSignal);
			HK_TEST(testSignal.getNumSubscriptions() == 20 - i - 1);
		}

		for (hkSlot* slot = testSignal.getSlots(); slot; slot = slot->getNext())
		{
			HK_TEST(!slot->hasNoSubscription());
			HK_TEST(slot->m_object == &uSlot);
		}

		for (int i = 0; i < 10; ++i)
		{
			testSignal.unsubscribe(&uSlot, &UnitTest::UnsubscribingSlot::onTestSignal);
			HK_TEST(testSignal.getNumSubscriptions() == 10 - i - 1);
		}
		
		HK_TEST(testSignal.getNumSubscriptions() == 0);
	}

	{
		UnitTest::SimpleSlot sSlot;
		UnitTest::UnsubscribingSlot uSlot(testSignal);
		for (int i = 0; i < 10; ++i)
		{
			HK_SUBSCRIBE_TO_SIGNAL(testSignal, &sSlot, SimpleSlot);
			HK_SUBSCRIBE_TO_SIGNAL(testSignal, &uSlot, UnsubscribingSlot);
		}

		testSignal.fire(0xabcd);
		HK_TEST(testSignal.getNumSubscriptions() == 10);

		for (hkSlot* slot = testSignal.getSlots(); slot; slot = slot->getNext())
		{
			HK_TEST(!slot->hasNoSubscription());
			HK_TEST(slot->m_object == &sSlot);
		}

		testSignal.unsubscribeAll(&sSlot);
		HK_TEST(testSignal.getNumSubscriptions() == 0);
	}

	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif

HK_TEST_REGISTER(signalUnitTest_main, "Fast", "Common/Test/UnitTest/Base/", __FILE__);

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
