/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/UnitTest/hkUnitTest.h>


static void* custom_alloc( int numBytes ) { return HK_NULL; }
static void custom_free( void*, int numBytes ) { }

class CustomAllocator : public hkMemoryAllocator
{
	public:

		virtual void* blockAlloc( int numBytes ) HK_OVERRIDE { return custom_alloc(numBytes); }
		virtual void blockFree( void* p, int numBytes ) HK_OVERRIDE { return custom_free(p, numBytes); }
		virtual void getMemoryStatistics( MemoryStatistics& u ) const HK_OVERRIDE { }
		virtual int getAllocatedSize(const void* obj, int numBytes) const HK_OVERRIDE { return numBytes; }
};


int allocator_test_main()
{
	CustomAllocator c;
	return 0;
}

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif
HK_TEST_REGISTER(allocator_test_main, "Fast", "Common/Test/UnitTest/Base/", __FILE__);

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
