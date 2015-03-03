/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Thread/Thread/WinRT/hkWinRTThreadUtils.h>

using namespace Havok;
using namespace Windows::System::Threading;
using namespace Windows::Storage;
using namespace Concurrency;

task<ByteArray> Havok::readDataAsync(DataReader^ reader, unsigned int numBytes)
{
	task<unsigned int> loadTask( reader->LoadAsync(numBytes) );
		
	auto byteArrayTask = loadTask.then([reader] (unsigned int size) -> ByteArray 
	{
		auto a = ref new Platform::Array<byte>(size);
		reader->ReadBytes(a); 
		ByteArray ba = { a };
		return ba;
	});

	return byteArrayTask;
}


template <typename TASK >
static bool __SYNCWAITVOID( TASK& t )
{
	if ( Havok::isBlockingAllowed() )
	{
		try 
		{
			t.wait();
			return true;
		}
		catch (Platform::Exception^ e)
		{
			::OutputDebugString(e->Message->Data());
		}
		return false;
	}
	else // PPL will not let us use wait() in main UI (an ASTA thread). 
	{	
		// Tried a few things here. 
		//   Lightweight concurrency::CurrentScheduler not exposed in Metro so can't make own thread func here easily outside of the tasks
		//   ThreadPool::RunAsync ok, but if we pass in a task<>, the PPL can deadlock waiting for the task to shedule id we wait on the thread to finish (with a event say)
		
		HK_ASSERT(0x57491cee, "You can't call blocking functions from the main UI thread in WinRT Applications");
		return false;
	
	}
}

template <typename TASK, typename RET>
static bool __SYNCWAIT( TASK& t, RET& r )
{
	if ( Havok::isBlockingAllowed() )
	{
		try 
		{
			t.wait();
			r = t.get();
			return true;
		}
		catch (Platform::Exception^ e)
		{
			::OutputDebugString(e->Message->Data());
		}
		return false;
	}
	else // PPL will not let us use wait() in main UI (an ASTA thread). 
	{
		// See above func.

		HK_ASSERT(0x2e03bba2, "You can't call blocking functions from the main UI thread in WinRT Applications");
		return false;
	}

}

bool Havok::taskWaitVoid( Concurrency::task<void>& t )
{
	return __SYNCWAITVOID< Concurrency::task<void> >( t );
}

bool Havok::taskWaitUint( Concurrency::task<unsigned int>& t, unsigned int& res )
{
	return __SYNCWAIT< Concurrency::task<unsigned int>, unsigned int >( t, res );
}

bool Havok::taskWaitByteArray( Concurrency::task<ByteArray> t, Platform::Array<byte>^& res )
{
	Havok::ByteArray r;
	if ( __SYNCWAIT< Concurrency::task<ByteArray> >( t, r ) )
	{
		res = r.data;
		return true;
	}
	return false;
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
