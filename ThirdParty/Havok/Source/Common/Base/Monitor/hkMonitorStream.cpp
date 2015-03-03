/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Monitor/hkMonitorStream.h>

#if defined(HK_XBOX_USE_PERFLIB) && defined(HK_PLATFORM_XBOX360)
hkUlong g_hkXbox360PerfSampleRegAddr = 0x8FFF1230; // 0x30 = 6*8 == reg 6 == LHS cycles in a PB0T0 setup
#endif 

HK_THREAD_LOCAL( hkMonitorStream* ) hkMonitorStream__m_instance;

#if (HK_CONFIG_MONITORS == HK_CONFIG_MONITORS_ENABLED)

	#if !defined(HK_PLATFORM_PS3_SPU)

		#include <Common/Base/Memory/Router/hkMemoryRouter.h>

		void hkMonitorStream::init()
		{
			#if defined(HK_PLATFORM_SPU)
				static hkMonitorStream _instance;
				hkMonitorStream* instance = &_instance;
			#else
				hkMonitorStream* instance = hkMemDebugBlockAlloc<hkMonitorStream>(1);
				new (instance) hkMonitorStream();
			#endif

			HK_MEMORY_TRACKER_NEW_OBJECT(hkMonitorStream, instance);
			HK_THREAD_LOCAL_SET( hkMonitorStream__m_instance, instance );
			instance->m_isBufferAllocatedOnTheHeap = false;
			instance->m_start = HK_NULL;
			instance->m_capacity = HK_NULL;
			instance->m_end = HK_NULL;
			instance->m_capacityMinus16 = HK_NULL;
		}

		void hkMonitorStream::quit()
		{
			hkMonitorStream* s = HK_THREAD_LOCAL_GET( hkMonitorStream__m_instance );
			if ( getStart() && isBufferAllocatedOnTheHeap() )
			{
				HK_MEMORY_TRACKER_DELETE_RAW(getStart());
				hkMemDebugBufFree(getStart(), int(getCapacity()-getStart()));
			}
			
			HK_MEMORY_TRACKER_DELETE_OBJECT(s);
			s->~hkMonitorStream();
			hkMemDebugBlockFree(s,1);
			HK_THREAD_LOCAL_SET( hkMonitorStream__m_instance, HK_NULL ); // CK: cleanup, so we can find non-init situations faster
		}

		void HK_CALL hkMonitorStream::resize( int newSize )
		{
			if ( newSize == getCapacity() - getStart() )
			{
				return;
			}

			if (newSize > 0)
			{
				if ( getStart() && isBufferAllocatedOnTheHeap() )
				{
					HK_MEMORY_TRACKER_DELETE_RAW(getStart());
					hkMemDebugBufFree(getStart(), int(getCapacity()-getStart()));
				}

				m_isBufferAllocatedOnTheHeap = true;
				m_start = hkMemDebugBufAlloc<char>(newSize);
				m_end = m_start;
				m_capacity = m_start + newSize;
				m_capacityMinus16 = m_capacity - 32;
				HK_MEMORY_TRACKER_NEW_RAW("buffer_hkMonitorStream", m_start, newSize);
			}
			else
			{
				quit();
			}
		}
	#else
		void hkMonitorStream::init() 
		{
			static hkMonitorStream _instance;
			hkMonitorStream__m_instance = &_instance;
		}
	#endif

	void HK_CALL hkMonitorStream::setStaticBuffer( char* buffer, int bufferSize )
	{
	#if !defined(HK_PLATFORM_PS3_SPU)
		if ( isBufferAllocatedOnTheHeap() )
		{
			resize(0);
		}
	#endif

		m_isBufferAllocatedOnTheHeap = false;
		m_start = buffer ;
		m_end = buffer;
		m_capacity = m_start + bufferSize;
		m_capacityMinus16 = m_capacity - 32 ;
	}

	void HK_CALL hkMonitorStream::reset()
	{
		m_end = m_start;
#if defined(HK_DEBUG) && !defined(HK_PLATFORM_SPU)
		
#	ifdef HK_DEBUG_SLOW
		HK_MONITOR_ADD_VALUE( "Debug build. Use Release for profiling.", 0.0f, HK_MONITOR_TYPE_SINGLE );
#	else
		HK_MONITOR_ADD_VALUE( "Dev build. Use Release for profiling.", 0.0f, HK_MONITOR_TYPE_SINGLE );
#	endif
#endif
	}

#else // #if (HK_CONFIG_MONITORS == HK_CONFIG_MONITORS_DISABLED)

	void hkMonitorStream::init() 
	{
		// we need this so that hkMonitorStream::getInstance() will return a valid pointer
		static hkMonitorStream _instance;
		HK_THREAD_LOCAL_SET( hkMonitorStream__m_instance, &_instance );
	}

	void hkMonitorStream::quit()
	{
	}

	void HK_CALL hkMonitorStream::setStaticBuffer( char* buffer, int bufferSize )
	{
	}

	void HK_CALL hkMonitorStream::reset()
	{
	}

	void HK_CALL hkMonitorStream::resize( int newSize )
	{
	}

#endif

#if defined(HK_COMPILER_MWERKS)
#	pragma force_active on
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
