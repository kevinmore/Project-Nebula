/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#pragma once

#ifndef __GCCXML__

using namespace Windows::Foundation;
using namespace Windows::Storage::Streams;
using namespace std;

#include <ppl.h>
#include <ppltasks.h>

namespace Havok
{
	inline bool isBlockingAllowed()
	{
		return !Concurrency::details::_Task_impl_base::_IsNonBlockingThread();
	}

    struct ByteArray { Platform::Array<byte>^ data; };

	Concurrency::task<ByteArray> readDataAsync(DataReader^ reader, unsigned int numBytes);
	
	// Some common waits we like to do
	bool taskWaitVoid( Concurrency::task<void>& t ); 
	bool taskWaitUint( Concurrency::task<unsigned int>& t, unsigned int& res );
	bool taskWaitByteArray( Concurrency::task<ByteArray> t, Platform::Array<byte>^& res );
}

#endif

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
