/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/UnitTest/hkUnitTest.h>
#include <Common/Base/Thread/JobQueue/hkJobQueue.h>



#if (HK_CONFIG_THREAD == HK_CONFIG_MULTI_THREADED)
	hkJobQueue* hkUnitTest::s_jobQueue = HK_NULL;
	hkJobThreadPool* hkUnitTest::s_jobThreadPool = HK_NULL;
#endif

hkUint32 hkUnitTest::randSeed = 'h'+'a'+'v'+'o'+'k';

hkReal HK_CALL hkUnitTest::rand01()
{
	const hkUint32 a = 1103515245;
	const hkUint32 c = 12345;
	const hkUint32 m = hkUint32(-1) >> 1;
	randSeed = (a * randSeed + c ) & m;
	return (hkReal(randSeed) / hkReal(m));
}

hkUint32 HK_CALL hkUnitTest::rand()
{
	const hkUint32 a = 1103515245;
	const hkUint32 c = 12345;
	const hkUint32 m = hkUint32(-1) >> 1;
	randSeed = (a * randSeed + c ) & m;
	return randSeed;
}

#ifndef HK_PLATFORM_SPU

static hkBool hkDefaultTestReportFunction(hkBool32 cond, const char* desc, const char* file, int line)
{
	if ( !cond )
	{
		HK_BREAKPOINT(0);	
	}
	return bool(cond);
}

hkTestReportFunctionType hkTestReportFunction = hkDefaultTestReportFunction;
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
