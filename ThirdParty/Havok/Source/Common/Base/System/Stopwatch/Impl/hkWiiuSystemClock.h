/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#ifndef HK_WIIU_SYSTEMCLOCK_H
#define HK_WIIU_SYSTEMCLOCK_H
#include <cafe.h>

hkUint64 HK_CALL hkSystemClock::getTickCounter()
{
	hkUint64 upperBytes = ((hkUint64)__MFTBU()) << 32;
	hkUint64 lowerBytes = (hkUint64) __MFTB();
	return upperBytes | lowerBytes;
}

hkUint64 HK_CALL hkSystemClock::getTicksPerSecond()
{
	return static_cast<hkUint64>(OS_TIMER_CLOCK);
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
