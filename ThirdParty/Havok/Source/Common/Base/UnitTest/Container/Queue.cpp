/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/Container/Queue/hkQueue.h>

class QueueTestClass
{
public:
	int x;

	QueueTestClass(int y=0)
	{
		x = y;
	}

	int get()
	{
		return x;
	}
};

static void queue()
{
	// Testing Queue functionality
	const int element = 10;
	int data;

	// Testing default queue functionality.
	{
		hkQueue<int> que;
		HK_TEST(que.getCapacity() == 0);
		HK_TEST(que.getSize() == 0);
		HK_TEST(que.isEmpty());
	}

	// Testing enqueue()
	{
		hkQueue<int> que;
		que.enqueue(element);
		// Default capcity is 8
		HK_TEST(que.getCapacity() == 8);
		HK_TEST( que.getSize() == 1 );
		hkQueue<int> que1(2);
		for(int i = 0; i < 3; i++)
		{
			que1.enqueue(i);
		}
		HK_TEST(que1.getCapacity() == 4);
		HK_TEST(que1.getSize() == 3);
	}

	// Testing dequeue()
	{
		hkQueue<int> que;
		que.enqueue(element);
		que.dequeue(data);
		HK_TEST( element == data );
		HK_TEST(que.getSize() == 0);

		hkQueue<int> que1(3);
		for(int i = 0; i < 3; i++)
		{
			que1.enqueue(i);
		}
		for(int i = 0; i < 3; i++)
		{
			que1.dequeue(data);
			HK_TEST(data == i);
		}

		HK_TEST(que1.isEmpty());
	}

	// Testing dequeue() & enqueue() with QueueTest object.
	{
		QueueTestClass ob1(10);
		hkQueue<QueueTestClass> que;
		que.enqueue(ob1);
		QueueTestClass ob2(9);
		que.enqueue(ob2);
		QueueTestClass ob3(8);
		que.enqueue(ob3);

		HK_TEST(que.getSize() == 3);
		QueueTestClass ob4;
		que.dequeue(ob4);
		HK_TEST(ob4.get() == 10);
		que.dequeue(ob4);
		HK_TEST(ob4.get() == 9);
		que.dequeue(ob4);
		HK_TEST(ob4.get() == 8);
	}

	// Testing enqueueInFront() functionality.
	{
		hkQueue<int> que(10);
		int i;
		for(i = 0; i < 5; ++i)
		{
			que.enqueue(i);
		}

		for(i = 5; i < 10; ++i)
		{
			que.enqueueInFront(i);
		}

		HK_TEST( que.getSize()==10 );

		for(i = 9; i >= 5; --i)
		{
			que.dequeue(data);
			HK_TEST(data == i);
		}
		HK_TEST( que.getSize() == 5 );
		for(i = 0; i < 5; ++i)
		{
			que.dequeue(data);
			HK_TEST(data == i);
		}
		HK_TEST( que.getSize() == 0 );
	}

	// Testing setCapacity() functionality
	{
		hkQueue<int> que;
		que.setCapacity(3);
		que.enqueueInFront(element);
		HK_TEST(que.getCapacity() != 4);
		HK_TEST(que.getCapacity() == 3);
		for(int i = 100; i <= 300; i += 100)
		{
			que.enqueue(i);
		}
		// Capacity increased by enqueue function
		HK_TEST(que.getCapacity() == 6);
		int tmp;
		que.dequeue(tmp);
		HK_TEST( tmp == element);
		HK_TEST(que.getSize() == 3);
		for(int i = 100; i <= 300; i += 100 )
		{
			que.dequeue(tmp);
			HK_TEST(tmp == i);
		}
		HK_TEST(que.getSize() == 0);
		HK_TEST(que.getCapacity() == 6);
		que.setCapacity(4);
		HK_TEST(que.getCapacity() == 6);
	}

	// Testing clear() functionality
	{
		hkQueue<int> que(3);
		for(int i = 0; i < 3; i++)
		{
			que.enqueue(i);
		}
		que.clear();
		HK_TEST( que.getSize() == 0);
		HK_TEST(que.getCapacity() == 3);
		HK_TEST(que.isEmpty());
	}

	// Testing getElement() functionality
	{
		hkQueue<int> que(4);
		que.enqueue(-2);
		que.enqueue(-1);
		for(int i = 0; i < 10; i++)
		{
			que.enqueue(i);
			HK_TEST(que.getElement(2) == i);
			int tmp;
			que.dequeue(tmp);
		}
	}
}

int queue_main()
{
	queue();
	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(queue_main, "Fast", "Common/Test/UnitTest/Base/", __FILE__     );

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
