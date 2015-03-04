/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Container/hkContainerAllocators.h>
#include <Common/Base/Memory/Allocator/Malloc/hkMallocAllocator.h>

void* hkContainerTempAllocator::Allocator::blockAlloc( int numBytes ) { return hkMemoryRouter::getInstance().temp().blockAlloc(numBytes); }
void hkContainerTempAllocator::Allocator::blockFree( void* p, int numBytes ) { return hkMemoryRouter::getInstance().temp().blockFree(p, numBytes); }
void* hkContainerTempAllocator::Allocator::bufAlloc( int& reqNumBytesInOut ) { return hkMemoryRouter::getInstance().temp().bufAlloc(reqNumBytesInOut); }
void hkContainerTempAllocator::Allocator::bufFree( void* p, int numBytes ) { return hkMemoryRouter::getInstance().temp().bufFree(p,numBytes); }
void* hkContainerTempAllocator::Allocator::bufRealloc( void* pold, int oldNumBytes, int& reqNumBytesInOut ) { return hkMemoryRouter::getInstance().temp().bufRealloc(pold,oldNumBytes,reqNumBytesInOut); }
void hkContainerTempAllocator::Allocator::getMemoryStatistics( MemoryStatistics& u ) const { return hkMemoryRouter::getInstance().temp().getMemoryStatistics(u); }
int hkContainerTempAllocator::Allocator::getAllocatedSize(const void* obj, int nbytes) const { return hkMemoryRouter::getInstance().temp().getAllocatedSize(obj, nbytes); }
hkContainerTempAllocator::Allocator hkContainerTempAllocator::s_alloc;

void* hkContainerDebugAllocator::Allocator::blockAlloc( int numBytes ) { return hkMemoryRouter::getInstance().debug().blockAlloc(numBytes); }
void hkContainerDebugAllocator::Allocator::blockFree( void* p, int numBytes ) { return hkMemoryRouter::getInstance().debug().blockFree(p, numBytes); }
void* hkContainerDebugAllocator::Allocator::bufAlloc( int& reqNumBytesInOut ) { return hkMemoryRouter::getInstance().debug().bufAlloc(reqNumBytesInOut); }
void hkContainerDebugAllocator::Allocator::bufFree( void* p, int numBytes ) { return hkMemoryRouter::getInstance().debug().bufFree(p,numBytes); }
void* hkContainerDebugAllocator::Allocator::bufRealloc( void* pold, int oldNumBytes, int& reqNumBytesInOut ) { return hkMemoryRouter::getInstance().debug().bufRealloc(pold,oldNumBytes,reqNumBytesInOut); }
void hkContainerDebugAllocator::Allocator::getMemoryStatistics( MemoryStatistics& u ) const { return hkMemoryRouter::getInstance().debug().getMemoryStatistics(u); }
int hkContainerDebugAllocator::Allocator::getAllocatedSize(const void* obj, int nbytes) const { return hkMemoryRouter::getInstance().debug().getAllocatedSize(obj, nbytes); }
hkContainerDebugAllocator::Allocator hkContainerDebugAllocator::s_alloc;

void* hkContainerHeapAllocator::Allocator::blockAlloc( int numBytes ) { return hkMemoryRouter::getInstance().heap().blockAlloc(numBytes); }
void hkContainerHeapAllocator::Allocator::blockFree( void* p, int numBytes ) { return hkMemoryRouter::getInstance().heap().blockFree(p, numBytes); }
void* hkContainerHeapAllocator::Allocator::bufAlloc( int& reqNumBytesInOut ) { return hkMemoryRouter::getInstance().heap().bufAlloc(reqNumBytesInOut); }
void hkContainerHeapAllocator::Allocator::bufFree( void* p, int numBytes ) { return hkMemoryRouter::getInstance().heap().bufFree(p,numBytes); }
void* hkContainerHeapAllocator::Allocator::bufRealloc( void* pold, int oldNumBytes, int& reqNumBytesInOut ) { return hkMemoryRouter::getInstance().heap().bufRealloc(pold,oldNumBytes,reqNumBytesInOut); }
void hkContainerHeapAllocator::Allocator::getMemoryStatistics( MemoryStatistics& u ) const { return hkMemoryRouter::getInstance().heap().getMemoryStatistics(u); }
int hkContainerHeapAllocator::Allocator::getAllocatedSize(const void* obj, int nbytes) const { return hkMemoryRouter::getInstance().heap().getAllocatedSize(obj, nbytes); }
hkContainerHeapAllocator::Allocator hkContainerHeapAllocator::s_alloc;

hkMemoryAllocator& hkContainerDefaultMallocAllocator::get(const void*) { return *hkMallocAllocator::m_defaultMallocAllocator; }

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
