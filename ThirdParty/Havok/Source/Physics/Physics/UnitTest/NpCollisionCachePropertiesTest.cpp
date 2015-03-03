/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>

#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Physics/Physics/Collide/NarrowPhase/Cache/hknpCollisionCache.h>
#ifdef HK_COMPILER_GCC
// There is no class declaration macro for collision cache, intentionally, and GCC doesn't contain placement new operator
#include <new>
#endif

#if defined(HK_COMPILER_MWERKS)
#	pragma fullpath_file on
#endif

static const hkUint8 CANARY = 0xa5;

void basicTest(void* buffer)
{
	const float seed = 0.8f;
	float value = seed;

	hkString::memSet(buffer,0x0,HKNP_MAX_COLLISION_CACHE_SIZE);

	// Pseudo construct() function
	hknpManifoldCollisionCache* cache = new (buffer) hknpManifoldCollisionCache;
	cache->init();
	hkUint32 baseSize = sizeof(hknpManifoldCollisionCache);
	cache->m_propertyBufferOffset = hkUint8(baseSize);
	cache->m_sizeInQuads = hkUint8(HK_NEXT_MULTIPLE_OF(1<<4,baseSize)>>4);

	// Try storing 6 reals in the cache;
	cache->allocateProperty<float>(hknpManifoldCollisionCache::RESTITUTION_PROPERTY);
	value = hkMath::sin(value);
	*(cache->accessProperty<float>(hknpManifoldCollisionCache::RESTITUTION_PROPERTY)) = value;

	cache->allocateProperty<float>(hknpManifoldCollisionCache::SOFT_CONTACT_PROPERTY);
	value = hkMath::sin(value);
	*(cache->accessProperty<float>(hknpManifoldCollisionCache::SOFT_CONTACT_PROPERTY)) = value;

	cache->allocateProperty<float>(hknpManifoldCollisionCache::USER_PROPERTY_0);
	value = hkMath::sin(value);
	*(cache->accessProperty<float>(hknpManifoldCollisionCache::USER_PROPERTY_0)) = value;

	cache->allocateProperty<float>(hknpManifoldCollisionCache::USER_PROPERTY_1);
	value = hkMath::sin(value);
	*(cache->accessProperty<float>(hknpManifoldCollisionCache::USER_PROPERTY_1)) = value;

	cache->allocateProperty<float>(hknpManifoldCollisionCache::USER_PROPERTY_2);
	value = hkMath::sin(value);
	*(cache->accessProperty<float>(hknpManifoldCollisionCache::USER_PROPERTY_2)) = value;

	cache->allocateProperty<float>(hknpManifoldCollisionCache::USER_PROPERTY_3);
	value = hkMath::sin(value);
	*(cache->accessProperty<float>(hknpManifoldCollisionCache::USER_PROPERTY_3)) = value;



	HK_TEST(cache->getPropertyBufferSize() == 6 * sizeof(float));
	HK_TEST(hkUint32(cache->getSizeInBytes()) == HK_NEXT_MULTIPLE_OF(1<<4, baseSize + 6 * sizeof(float) ));

	// Test the values
	value = seed;
	value = hkMath::sin(value);
	HK_TEST(*(cache->accessProperty<float>(hknpManifoldCollisionCache::RESTITUTION_PROPERTY)) == value);

	value = hkMath::sin(value);
	HK_TEST(*(cache->accessProperty<float>(hknpManifoldCollisionCache::SOFT_CONTACT_PROPERTY)) == value);

	value = hkMath::sin(value);
	HK_TEST(*(cache->accessProperty<float>(hknpManifoldCollisionCache::USER_PROPERTY_0)) == value);

	value = hkMath::sin(value);
	HK_TEST(*(cache->accessProperty<float>(hknpManifoldCollisionCache::USER_PROPERTY_1)) == value);

	value = hkMath::sin(value);
	HK_TEST(*(cache->accessProperty<float>(hknpManifoldCollisionCache::USER_PROPERTY_2)) == value);

	value = hkMath::sin(value);
	HK_TEST(*(cache->accessProperty<float>(hknpManifoldCollisionCache::USER_PROPERTY_3)) == value);

	hkString::memSet(buffer,0x0,HKNP_MAX_COLLISION_CACHE_SIZE);
}


void testAlignment(void* ptr, hkUint32 align)
{
#ifndef HK_ALIGN_RELAX_CHECKS
	HK_TEST( (hkUlong(ptr) & (align-1)) == 0 );
#endif
}

// Test alignment of requests
void alignTest(void* buffer)
{
	hkString::memSet(buffer,0x0,HKNP_MAX_COLLISION_CACHE_SIZE);
	// Pseudo construct() function
	hknpManifoldCollisionCache* cache = new(buffer) hknpManifoldCollisionCache;
	cache->init();
	hkUint32 baseSize = sizeof(hknpManifoldCollisionCache);
	cache->m_propertyBufferOffset = hkUint8(baseSize);
	cache->m_sizeInQuads = hkUint8(HK_NEXT_MULTIPLE_OF(1<<4,baseSize)>>4);

	// Add a float 4 -bytes aligned (default);
	cache->allocateProperty<float>(hknpManifoldCollisionCache::USER_PROPERTY_3);
	testAlignment(cache->accessProperty<float>(hknpManifoldCollisionCache::USER_PROPERTY_3), 4);

	// Then ask for a 8-byte aligned slot
	cache->allocateProperty<float>(hknpManifoldCollisionCache::USER_PROPERTY_1, 8);
	// Check alignment (can't do test in HK_TEST because of multiple template parameters in macro) #c++sadness
	testAlignment(cache->accessProperty<float>(hknpManifoldCollisionCache::USER_PROPERTY_1), 8);

	// Add another float with 4-bytes align
	cache->allocateProperty<float>(hknpManifoldCollisionCache::USER_PROPERTY_2);
	testAlignment(cache->accessProperty<float>(hknpManifoldCollisionCache::USER_PROPERTY_2), 4);

	// And then a 16-bytes align vector4
	cache->allocateProperty<hkVector4f>(hknpManifoldCollisionCache::USER_PROPERTY_0, 16);
	testAlignment(cache->accessProperty<hkVector4f>(hknpManifoldCollisionCache::USER_PROPERTY_0), 16);

	hkString::memSet(buffer,0x0,HKNP_MAX_COLLISION_CACHE_SIZE);
}

// Just try to fill the buffer
void fillTest(void* buffer)
{
#ifndef HK_ALIGN_RELAX_CHECKS
	hkString::memSet(buffer,0x0,HKNP_MAX_COLLISION_CACHE_SIZE);
	// Pseudo construct() function
	hknpManifoldCollisionCache* cache = new(buffer) hknpManifoldCollisionCache;
	cache->init();
	hkUint32 baseSize = sizeof(hknpManifoldCollisionCache);
	cache->m_propertyBufferOffset = hkUint8(baseSize);
	cache->m_sizeInQuads = hkUint8(HK_NEXT_MULTIPLE_OF(1<<4,baseSize)>>4);

	const float seed = 0.7f;
	float value = seed;
	// 4 16-bytes align vector4
	cache->allocateProperty<hkVector4f>(hknpManifoldCollisionCache::USER_PROPERTY_2, 16);
	value = hkMath::sin(value);
	cache->accessProperty<hkVector4f>(hknpManifoldCollisionCache::USER_PROPERTY_2)->setAll(value);

	cache->allocateProperty<hkVector4f>(hknpManifoldCollisionCache::USER_PROPERTY_3, 16);
	value = hkMath::sin(value);
	cache->accessProperty<hkVector4f>(hknpManifoldCollisionCache::USER_PROPERTY_3)->setAll(value);

	cache->allocateProperty<hkVector4f>(hknpManifoldCollisionCache::USER_PROPERTY_1, 16);
	value = hkMath::sin(value);
	cache->accessProperty<hkVector4f>(hknpManifoldCollisionCache::USER_PROPERTY_1)->setAll(value);

	cache->allocateProperty<hkVector4f>(hknpManifoldCollisionCache::USER_PROPERTY_0, 16);
	value = hkMath::sin(value);
	cache->accessProperty<hkVector4f>(hknpManifoldCollisionCache::USER_PROPERTY_0)->setAll(value);



	HK_TEST(cache->getPropertyBufferSize() == 4 * sizeof(hkVector4f));
	HK_TEST(hkUint32(cache->getSizeInBytes()) == HK_NEXT_MULTIPLE_OF(1<<4, baseSize + 4 * sizeof(hkVector4f) ));


	// Check values
	hkBool32 isOk = false;
	value = seed;
	hkVector4f test; test.setZero();

	value = hkMath::sin(value);
	test.setAll(value);
	isOk = cache->accessProperty<hkVector4f>(hknpManifoldCollisionCache::USER_PROPERTY_2)->allExactlyEqual<4>(test);

	value = hkMath::sin(value);
	test.setAll(value);
	isOk = cache->accessProperty<hkVector4f>(hknpManifoldCollisionCache::USER_PROPERTY_3)->allExactlyEqual<4>(test);

	value = hkMath::sin(value);
	test.setAll(value);
	isOk = cache->accessProperty<hkVector4f>(hknpManifoldCollisionCache::USER_PROPERTY_1)->allExactlyEqual<4>(test);

	value = hkMath::sin(value);
	test.setAll(value);
	isOk = cache->accessProperty<hkVector4f>(hknpManifoldCollisionCache::USER_PROPERTY_0)->allExactlyEqual<4>(test);


	hkString::memSet(buffer,0x0,HKNP_MAX_COLLISION_CACHE_SIZE);
#endif
}

// Try to change the cache type and check the size etc is correctly adjusted.
void demoteTest(void *buffer)
{
	hkString::memSet(buffer,0x0,HKNP_MAX_COLLISION_CACHE_SIZE);
	// Pseudo construct() function

	hknpManifoldCollisionCache* cache = new(buffer) hknpManifoldCollisionCache;
	hknpConvexConvexCollisionCache* cacheBase = cache;
	cache->init();
	hkUint32 baseSize = sizeof(hknpManifoldCollisionCache);
	cache->m_propertyBufferOffset = hkUint8(baseSize);
	cache->m_sizeInQuads = hkUint8(HK_NEXT_MULTIPLE_OF(1<<4,baseSize)>>4);

	// Add some data
	cache->allocateProperty<hkVector4f>(hknpManifoldCollisionCache::USER_PROPERTY_0,16);
	HK_TEST( cache->getSizeInBytes() == HK_NEXT_MULTIPLE_OF(16,  sizeof(hknpManifoldCollisionCache) + sizeof(hkVector4f)));

	// Revert the cache to convex-convex
	cache->demoteToCvxCvxCache();

	HK_TEST(cacheBase->getSizeInBytes() == sizeof(hknpConvexConvexCollisionCache) );

	// Put it back to Manifold
	cacheBase->promoteTo<hknpManifoldCollisionCache>();
	baseSize = sizeof(hknpManifoldCollisionCache);
	cache->m_propertyBufferOffset = hkUint8(baseSize);
	cache->m_sizeInQuads = hkUint8(HK_NEXT_MULTIPLE_OF(1<<4,baseSize)>>4);

	// The two pointers are the same, hopefully...
	HK_ASSERT(0x0a3290b4, cacheBase == cache);

	// The properties buffer should be empty.
	HK_TEST(cache->getPropertyBufferSize() == 0);
	// Access to the property should fail (return HK_NULL)
	HK_TEST(cache->accessProperty<hkVector4f>(hknpManifoldCollisionCache::USER_PROPERTY_0) == HK_NULL);
}


int NpCollisionCacheProperties_main()
{
	// Buffer to store  the collision cache
	HK_ALIGN128(hkUint8 buffer[HKNP_MAX_COLLISION_CACHE_SIZE + 1]);
	buffer[HKNP_MAX_COLLISION_CACHE_SIZE] = CANARY;

	basicTest(buffer);
	alignTest(buffer);
	fillTest(buffer);
	demoteTest(buffer);

	// Check that we didn't write over the maximum allowed size;
	//HK_TEST(hkBool(buffer[HKNP_MAX_COLLISION_CACHE_SIZE] == hkUint8(CANARY)));
	HK_TEST(buffer[HKNP_MAX_COLLISION_CACHE_SIZE] == CANARY);
	return 0;
}


HK_TEST_REGISTER(NpCollisionCacheProperties_main, "Fast", "Physics/Test/UnitTest/Physics/", __FILE__);

/*
 * Havok SDK - Product file, BUILD(#20130912)
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
