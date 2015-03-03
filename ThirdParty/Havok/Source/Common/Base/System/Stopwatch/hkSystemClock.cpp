/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/System/Stopwatch/hkSystemClock.h>

#if defined(HK_PLATFORM_WIN32) || defined(HK_PLATFORM_XBOX) || defined(HK_PLATFORM_XBOX360)
#	include <Common/Base/System/Stopwatch/Impl/hkWindowsSystemClock.h>

#elif defined(HK_PLATFORM_LINUX)
#	include <Common/Base/System/Stopwatch/Impl/hkLinuxSystemClock.h>

#elif defined(HK_PLATFORM_MAC386) || defined(HK_PLATFORM_MACPPC) || defined(HK_PLATFORM_IOS)
#	include <Common/Base/System/Stopwatch/Impl/hkMacSystemClock.h>

#elif defined( HK_PLATFORM_PS3_PPU) || defined (HK_PLATFORM_PS3_SPU)
#   include <Common/Base/System/Stopwatch/Impl/hkPs3SystemClock.h>

#elif defined(HK_PLATFORM_WIIU)
#	include <Common/Base/System/Stopwatch/Impl/hkWiiuSystemClock.h>
	
#elif defined(HK_PLATFORM_GC)
#	include <Common/Base/System/Stopwatch/Impl/hkNgcSystemClock.h>

#elif defined(HK_PLATFORM_PSVITA)
#	include <Common/Base/System/Stopwatch/Impl/hkPsVitaSystemClock.h>

#elif defined(HK_PLATFORM_LRB)
#	include <Common/Base/System/Stopwatch/Impl/hkLrbSystemClock.h>

#elif defined(HK_PLATFORM_CTR)
#	include <Common/Base/System/Stopwatch/Impl/hkCtrSystemClock.h>

#elif defined(HK_PLATFORM_ANDROID) || defined(HK_PLATFORM_TIZEN)
#	include <Common/Base/System/Stopwatch/Impl/hkAndroidSystemClock.h>

#elif defined(HK_PLATFORM_PS4)
#	include <Common/Base/System/Stopwatch/Impl/hkPs4SystemClock.h>

#else
#  error ERROR: No SystemClock implementation available!

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
