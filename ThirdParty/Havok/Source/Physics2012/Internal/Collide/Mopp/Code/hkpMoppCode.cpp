/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Internal/hkpInternal.h>
#include <Physics2012/Internal/Collide/Mopp/Code/hkpMoppCode.h>


	// MOPP chunk size must be a power of 2
HK_COMPILE_TIME_ASSERT( (HK_MOPP_CHUNK_SIZE & (HK_MOPP_CHUNK_SIZE - 1)) == 0 );
HK_COMPILE_TIME_ASSERT( HK_MOPP_CHUNK_SIZE == 1<< HK_MOPP_LOG_CHUNK_SIZE );

#if defined(HK_PLATFORM_SPU)
hkpMoppCache* g_SpuMoppCache;
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
