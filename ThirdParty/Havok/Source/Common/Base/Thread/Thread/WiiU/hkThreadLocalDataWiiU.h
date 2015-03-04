/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE
#ifndef HKBASE_HK_THREAD_LOCAL_DATA_WIIU_H
#define HKBASE_HK_THREAD_LOCAL_DATA_WIIU_H

#		include <cafe/os.h>

// No compiler TLS, and OSThread interface supports the grand total of 2 keys
// so just use one key to store our table of keys. 

#define HK_WIIU_MAX_TLS_VALUES 31
struct hkWiiuTlsPerThread
{
	void* m_values[HK_WIIU_MAX_TLS_VALUES];
};

struct hkWiiuTls
{
	hkUint32 m_keyUsed; // 0 (empty) + 31 allowed used 

	hkWiiuTls()
	{
		m_keyUsed = 0;
	}

	int newKey()
	{
		hkUint32 mask = 1;
		hkUint32 cur = m_keyUsed;
		for( int i = 0; i < HK_WIIU_MAX_TLS_VALUES; ++i )
		{
			if( (cur & mask) == 0 )
			{
				m_keyUsed |= mask;
				return i;
			}
			mask <<= 1;
		}
		HK_BREAKPOINT(0); // tls full
		return -1;
	}

	void delKey(int key)
	{
		hkUint32 mask = hkUint32(1) << key;
		m_keyUsed &= ~mask;
	}

	__attribute__((always_inline)) void* getValue(int key)
	{
		// Get this thread's list
		hkWiiuTlsPerThread* t = (hkWiiuTlsPerThread*)OSGetThreadSpecific(1);
		if (t==HK_NULL)
		{
			t = new hkWiiuTlsPerThread;
			for (int i=0; i < HK_WIIU_MAX_TLS_VALUES; ++i) 
			{ 
				t->m_values[i] = HK_NULL; 
			}
			OSSetThreadSpecific(1,t);
		}
		return t->m_values[key];
	}

	__attribute__((always_inline)) void setValue(int key, void* v)
	{
		// Get this thread's list
		hkWiiuTlsPerThread* t = (hkWiiuTlsPerThread*)OSGetThreadSpecific(1);
		if (t==HK_NULL)
		{
			t = new hkWiiuTlsPerThread;
			for (int i=0; i < HK_WIIU_MAX_TLS_VALUES; ++i) 
			{ 
				t->m_values[i] = HK_NULL; 
			}
			OSSetThreadSpecific(1,t);
		}

		t->m_values[key] = v;

		if (v == HK_NULL)
		{
			//check if all NULL now, if so no more tls for this thread
			for (int i=0; i < HK_WIIU_MAX_TLS_VALUES; ++i) 
			{ 
				if ( t->m_values[i] != HK_NULL)
					return;
			}

			delete t;
			OSSetThreadSpecific(1, HK_NULL);
		}
	}

	bool empty()
	{
		return m_keyUsed == 0;
	}

};

extern hkWiiuTls* g_hkWiiuTls;

template < typename T > 
class hkThreadLocalData
{
public:

	hkThreadLocalData()
	{
		//- All ctors from same thread (at startup etc) so thread safe here
		if (g_hkWiiuTls == HK_NULL)
		{
			// first in
			g_hkWiiuTls = new hkWiiuTls();
		}

		m_key = g_hkWiiuTls->newKey();
	}

	~hkThreadLocalData()
	{
		HK_ON_DEBUG( if (!g_hkWiiuTls) { HK_BREAKPOINT(0); } )
			g_hkWiiuTls->delKey(m_key);
		if (g_hkWiiuTls->empty())
		{
			// last out 
			delete g_hkWiiuTls;
			g_hkWiiuTls = HK_NULL;
		}
	}

	HK_FORCE_INLINE T getData() 
	{
		HK_ON_DEBUG( if (!g_hkWiiuTls) { HK_BREAKPOINT(0); } )
			return (T)g_hkWiiuTls->getValue(m_key);
	}

	HK_FORCE_INLINE void setData(T p) 
	{
		HK_ON_DEBUG( if (!g_hkWiiuTls) { HK_BREAKPOINT(0); } )
			g_hkWiiuTls->setValue(m_key, (void*) p);
	}

	int m_key;
};

#		define HK_THREAD_LOCAL(TYPE) hkThreadLocalData<TYPE>
#		define HK_THREAD_LOCAL_SET(var,value) var.setData(value)
#		define HK_THREAD_LOCAL_GET(var) var.getData()

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
