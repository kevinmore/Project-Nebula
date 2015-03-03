/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Fwd/hkcstring.h>
using namespace std;

HK_COMPILE_TIME_ASSERT( sizeof(void*)==sizeof(char*) );

// hack to force this to be out of line
hkResult HK_CALL hkArrayUtil::_reserve(hkMemoryAllocator& mem, void* array, int numElem, int sizeElem)
{
#if !defined(HK_PLATFORM_PS3_SPU)
	HK_ASSERT2(0x3b67c014, numElem >= 0, "Number of elements must not be negative");
	HK_ASSERT2(0x243bf8d1, sizeElem >= 0, "The size of an element must not be negative");

	typedef hkArray<char> hkAnyArray;
	hkAnyArray* self = reinterpret_cast< hkAnyArray* >(array);
	HK_ASSERT(0x673429dd, numElem >= self->getCapacity() );

	char* p = HK_NULL;
	int reqNumBytes = numElem * sizeElem;
	HK_ASSERT2(0xaf1eed13, (reqNumBytes & hkAnyArray::CAPACITY_MASK) == reqNumBytes, "The requested storage space exceeds the amount that can be stored in this array.");
	if( (self->m_capacityAndFlags & hkAnyArray::DONT_DEALLOCATE_FLAG) == 0)
	{
		p = static_cast<char*>( mem.bufRealloc( self->m_data, self->getCapacity()*sizeElem, reqNumBytes) );
	}
	else
	{
		p = static_cast<char*>(mem.bufAlloc(reqNumBytes));
		if(p)
		{
			memcpy(p, self->m_data, self->m_size*sizeElem);
		}
	}

	self->m_data = p;
	self->m_capacityAndFlags = reqNumBytes / sizeElem;
	
	return (p) ? HK_SUCCESS : HK_FAILURE;
#else
	HK_ERROR(0x3f898db4, "Can't resize on SPU!");
	return HK_FAILURE;
#endif
}


// hack to force this to be out of line
void hkArrayUtil::_reserveMore(hkMemoryAllocator& mem, void* array, int sizeElem)
{
#if !defined(HK_PLATFORM_SPU)
	HK_ASSERT2(0x5828d5cf, sizeElem >= 0, "The size of an element must not be negative");
	typedef hkArray<char> hkAnyArray;
	hkAnyArray* self = reinterpret_cast< hkAnyArray* >(array);
	int numElem = (self->m_size ? self->m_size*2 : 1);

	int reqNumBytes = numElem * sizeElem;
	if( (self->m_capacityAndFlags & hkAnyArray::DONT_DEALLOCATE_FLAG) == 0)
	{
		self->m_data = static_cast<char*>( mem.bufRealloc( self->m_data, self->getCapacity()*sizeElem, reqNumBytes ) );
	}
	else // mem is not from the mem manager
	{
		char* p = static_cast<char*>(mem.bufAlloc(reqNumBytes));
		memcpy(p, self->m_data, self->m_size*sizeElem);
		self->m_data = p;
	}
	
	self->m_capacityAndFlags = reqNumBytes / sizeElem;
#else
	HK_ASSERT2(0x5ccfc21e, false, "Can't resize on SPU!");
#endif
}

#if !defined(HK_PLATFORM_PS3_SPU)

// hack to force this to be out of line
void hkArrayUtil::_reduce(hkMemoryAllocator& mem, void* array, int sizeElem, char* inplaceMem, int requestedCapacity)
{
	typedef hkArray<char> hkAnyArray;
	hkAnyArray* self = reinterpret_cast< hkAnyArray* >(array);

	HK_ASSERT2(0x5828d5f0, sizeElem >= 0, "The size of an element must not be negative");
	HK_ASSERT2(0x5828d5f1, (self->m_capacityAndFlags & hkAnyArray::DONT_DEALLOCATE_FLAG) == 0, "Don't call _reduce if the array is already optimal or preallocated.");

	if ( (HK_NULL != inplaceMem) && (self->m_size < requestedCapacity) )
	{
		// this is the case of an hkInplaceArray
		memcpy(inplaceMem, self->m_data, self->m_size*sizeElem);
		mem.bufFree( self->m_data, self->getCapacity()*sizeElem);
		self->m_data = inplaceMem;
		self->m_capacityAndFlags = requestedCapacity | hkAnyArray::DONT_DEALLOCATE_FLAG;
	}
	else
	{
		HK_ASSERT2(0x5828d5f1, requestedCapacity >= self->m_size, "Can't reduce array");
		int reqNumBytes = requestedCapacity * sizeElem;
		self->m_data = static_cast<char*>(mem.bufRealloc(self->m_data, self->m_capacityAndFlags*sizeElem, reqNumBytes));
		self->m_capacityAndFlags = reqNumBytes /  sizeElem;
	}
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
