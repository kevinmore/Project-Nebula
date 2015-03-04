/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Thread/CpuCache/hkCpuCache.h>
#include <Common/Base/System/Stopwatch/hkStopwatch.h>

hkCpuCache::hkCpuCache()
{
	Cinfo info;
	init(info);
	m_checkId = 0;
	m_frameIterations = 0;
	m_currentId = 0;
}

void hkCpuCache::init( const Cinfo& info  )
{
	for (int i = 0; i < MAX_CHECKS; i++)
	{
		m_names[i] = HK_NULL;
		m_checkCount[i] = 0;
		m_checkTicks[i] = 0;
	}
}

void hkCpuCache::startFrame()
{
	m_frameIterations++;
	m_checkId = (m_checkId+1) & (MAX_CHECKS-1);
}

void hkCpuCache::flushCache(  )
{
#if defined(HK_PLATFORM_WIN32) && !HK_PLATFORM_IS_CONSOLE
	int cacheSize = 8024*1024;
#else
	int cacheSize = 1024*1024;
#endif
	char *data = hkAllocateChunk<char>( cacheSize, 0 );
	// wipe cache
	{	 int x = 0; for (char* p = data; p < data+cacheSize; p+= 32) {x += p[0];	p[0] = hkUchar(x);} }
	{	 int x = 0; for (char* p = data; p < data+cacheSize; p+= 32) {x += p[0];	p[0] = hkUchar(x);} }
	hkDeallocateChunk( data, cacheSize, 0 );
}


void hkCpuCache::printStats()
{
	hkUint64 ticksPerSecond = hkStopwatch::getTicksPerSecond();
	hkUint64 clocksPerSecond = 3200000000ul;
	int clocksPerTick = int( clocksPerSecond/ticksPerSecond );
	HK_REPORT2(0xf0235456, "Cache Report for " << m_frameIterations << "frames");
	for (int i = 0; i < MAX_CHECKS; i++)
	{
		if ( m_checkCount[i] == 0)
		{
			continue;
		}
		int c = m_checkCount[i];
		int ticks = m_checkTicks[i];
		int clocks = ticks*clocksPerTick/c - 46;

		char buffer[1024];
		hkString::snprintf( buffer, 1024, "%-20s(%4i): %i cycles/check, %3f usecs total", m_names[i], c, clocks, clocks*1e6f/hkReal(clocksPerSecond));
		HK_REPORT2( 0xf0235456, buffer );
	}
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
