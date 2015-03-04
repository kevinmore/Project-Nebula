/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Thread/Semaphore/hkSemaphore.h>

// match the define in hkSemaphoreBusyWait.h:67
#if defined(HK_PLATFORM_PSVITA) || (defined(HK_PLATFORM_IOS) && !defined(HK_PLATFORM_IOS_SIM))

#include <Common/Base/Thread/Semaphore/hkSemaphoreBusyWait.h>
#include <Common/Base/Thread/Thread/Arm/hkArmThreadSync.h>
#include <Common/Base/Fwd/hkcstdio.h>

//#include <Common/Base/System/Android/hkAndroidCpuInfo.h>
//#include <android/log.h>

// Assumes main semaphore is normal Posix one etc

hkSemaphoreBusyWait::hkSemaphoreBusyWait( int initialCount, int /*maxCount*/ )
: m_value(initialCount)
{
//	bool supportLdrex = (HK_ANDROID_CPU_ARM_FEATURE_LDREX_STREX & hkAndroidGetCpuFeatures()) != 0;
}

hkSemaphoreBusyWait::~hkSemaphoreBusyWait()
{
    
}

void hkSemaphoreBusyWait::acquire()
{
    do 
	{
		int curValue= hk_ldrex( &m_value );
		if (curValue > 0)
		{
			--curValue;
			if ( 0 == hk_strex(curValue, &m_value) )
			{
				break; // success
			}
		}
		// otherwise spin
	} while (true);

}

void hkSemaphoreBusyWait::release(int count)
{
	int curValue;
	do 
	{
		curValue = hk_ldrex( &m_value );
		curValue += count;
	} while ( 0 != hk_strex(curValue, &m_value) );
}

void hkSemaphoreBusyWait::acquire(hkSemaphoreBusyWait* semaphore)
{
	semaphore->acquire();
}

void hkSemaphoreBusyWait::release(hkSemaphoreBusyWait* semaphore, int count)
{
	semaphore->release(count);
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
