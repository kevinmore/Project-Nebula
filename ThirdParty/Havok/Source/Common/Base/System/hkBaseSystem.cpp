/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/System/hkBaseSystem.h>
#include <Common/Base/Config/hkConfigVersion.h>
#include <Common/Base/DebugUtil/MultiThreadCheck/hkMultiThreadCheck.h>
#include <Common/Internal/Misc/hkSystemDate.h>
#include <Common/Base/System/Io/FileSystem/hkNativeFileSystem.h>

#if defined ( HK_PLATFORM_SIM_SPU )
struct hkSingletonInitNode* hkSingletonInitList;
#endif

#if !defined (HK_PLATFORM_PS3_SPU)
#	include <Common/Base/System/Error/hkDefaultError.h>
#	include <Common/Base/Thread/CriticalSection/hkCriticalSection.h>
#endif

#if !defined (HK_PLATFORM_LRB)
#include <Common/Base/System/Io/Socket/hkSocket.h>
#endif

#if defined(HK_PLATFORM_PS3_PPU)
#	include <sys/ppu_thread.h>
#endif

#if defined(HK_PLATFORM_WIN32)
	#include <Common/Base/Fwd/hkwindows.h>
	typedef BOOL (__stdcall *tGetLogicalProcessorInformation)(OUT PSYSTEM_LOGICAL_PROCESSOR_INFORMATION Buffer,IN PDWORD ReturnLength);
#endif

#if defined(HK_PLATFORM_MAC386) || defined(HK_PLATFORM_MACPPC) || defined(HK_PLATFORM_IOS)
	#include <sys/param.h>
	#include <sys/sysctl.h>
#endif

#if defined(HK_PLATFORM_LINUX) || defined(HK_PLATFORM_TIZEN)
	#include <unistd.h>
#endif

#if defined(HK_PLATFORM_ANDROID)
	#include <Common/Base/System/Android/hkAndroidCpuInfo.h>
	#include <android/log.h>
	#include <unistd.h> // for sysconf
#endif

#include <Common/Base/KeyCode.h>

#if defined(HK_PLATFORM_GC) && !defined(HK_PLATFORM_WIIU)
#	define KEYCODE_ATTRIBUTES __attribute__((section(".sdata")))
#else
#	define KEYCODE_ATTRIBUTES
#endif


void HK_CALL hkGetHardwareInfo( hkHardwareInfo& info )
{
#if defined(HK_PLATFORM_XBOX360)
	info.m_numThreads = 6;
#elif defined(HK_PLATFORM_PS3_PPU) || defined(HK_PLATFORM_HAS_SPU)
	info.m_numThreads = 2;
#elif defined (HK_PLATFORM_LRB)
	info.m_numThreads = 4;
#elif defined(HK_PLATFORM_PSVITA)
	#if HK_CONFIG_THREAD==HK_CONFIG_MULTI_THREADED
		info.m_numThreads = 3;
	#else
		info.m_numThreads = 1;
	#endif
#elif defined(HK_PLATFORM_PS4)
	info.m_numThreads = 6; // allowed 0x3f as user mask == first 6 cores
#elif defined(HK_PLATFORM_WIIU)
	info.m_numThreads = HK_CONFIG_WIIU_NUM_THREADS;

#elif defined(HK_PLATFORM_DURANGO)
	info.m_numThreads = 6; // OS reserves 2

#elif defined(HK_PLATFORM_WIN32)

	// Use system info
	_SYSTEM_INFO lpSystemInfo;
	#if defined(HK_PLATFORM_WINRT)
		GetNativeSystemInfo(&lpSystemInfo);
	#else
		GetSystemInfo(&lpSystemInfo);
	#endif
		
	info.m_numThreads = hkMath::min2<int>(lpSystemInfo.dwNumberOfProcessors, HK_MAX_NUM_THREADS);

	#if !defined(HK_PLATFORM_WINRT) && !defined( HK_PLATFORM_DURANGO ) 
		// Load this dynamically as GetLogicalProcessorInformation is only available on
		// XP SP3, and later systems
		HINSTANCE hKernel32Dll;
		tGetLogicalProcessorInformation pGetLogicalProcessorInformation = HK_NULL;
		// Load kernel32.dll to see if GetLogicalProcessorInformation is available
		hKernel32Dll = LoadLibraryA( "kernel32.dll");

		if(hKernel32Dll)
		{
			pGetLogicalProcessorInformation = (tGetLogicalProcessorInformation) GetProcAddress(hKernel32Dll, "GetLogicalProcessorInformation");
		}
	
		if(pGetLogicalProcessorInformation)
		{
			// refine to dinstinguish hyperthreads and real cores
			DWORD length=0;

			pGetLogicalProcessorInformation( HK_NULL, &length );

			// Allocate on stack since heap may not be initialized yet - max 128 entries  = 3k
			char buffer[ 128 * sizeof( _SYSTEM_LOGICAL_PROCESSOR_INFORMATION ) ];
			if (length <= sizeof( buffer ))
			{
				pGetLogicalProcessorInformation( (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION)buffer, &length );
				PSYSTEM_LOGICAL_PROCESSOR_INFORMATION proc = (PSYSTEM_LOGICAL_PROCESSOR_INFORMATION)buffer;

				// Count real cores
				DWORD bufferOffset = 0;
				int cores = 0;
				while( bufferOffset < length )
				{
					if (proc->Relationship == RelationProcessorCore )
						cores++;
					bufferOffset += sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
					proc++;
				}
				info.m_numThreads = cores;
			}
		}

		// We don't need this anymore
		if(hKernel32Dll)
		{
			FreeLibrary(hKernel32Dll);
		}
	#else

		// Currently no way to query this on WinRT
		// Need to imp our own cpuid calls, but with Core i7 etc that is tricky as reports max num possibkle in packaghe so the code to query actual cores is more involved.
		if (info.m_numThreads > 6)
		{
			info.m_numThreads /= 2;
		}

	#endif

#elif defined(HK_PLATFORM_LINUX) || defined(HK_PLATFORM_TIZEN)
	info.m_numThreads = sysconf(_SC_NPROCESSORS_ONLN);
#elif defined(HK_PLATFORM_MAC386) || defined(HK_PLATFORM_MACPPC) || ( (HK_CONFIG_THREAD==HK_CONFIG_MULTI_THREADED) && defined(HK_PLATFORM_IOS) )
	info.m_numThreads = 1; 
	size_t size=sizeof(info.m_numThreads);
	sysctlbyname("hw.physicalcpu",&info.m_numThreads,&size,NULL,0);
#elif (HK_CONFIG_THREAD==HK_CONFIG_MULTI_THREADED) && defined(HK_PLATFORM_ANDROID)
	info.m_numThreads = hkAndroidGetCpuCount(); 
	static bool reported = false;
	if (!reported)
	{
		__android_log_print(ANDROID_LOG_INFO, "Havok", "Found %d Cpu Cores (sysconf says %d)", info.m_numThreads, sysconf(_SC_NPROCESSORS_ONLN)  );
		//reported = true;
	}
#else
	info.m_numThreads = 1;
#endif
}

#ifdef HK_DEBUG
#	include <Common/Base/Fwd/hkcstdio.h>
	using namespace std;
#endif

hkBool hkBaseSystemIsInitialized;
#if defined(HK_DEBUG)
	hkBool hkBaseSystemInitVerbose = true;
#else
	hkBool hkBaseSystemInitVerbose; /* = false; */
#endif

// We create a dummy singleton so that even if no other singleton
// is used, the singleton list should not be empty.  Therefore if
// the singleton list is empty, it means that the static constructors
// which register the singletons were not called. This could be
// caused by not calling mwInit() or possibly an incorrect linker file.
class hkDummySingleton : public hkReferencedObject, public hkSingleton<hkDummySingleton>
{
	public:

		hkDummySingleton() {}

		virtual void forceLinkage() { }
};

#ifdef HK_COMPILER_ARMCC // Error:  #2574: explicit specialization of member s_instance must precede its first use 
HK_SINGLETON_SPECIALIZATION_DECL(hkDummySingleton);
#endif

#ifndef HK_PLATFORM_CTR
#define PRINTF_FUNC printf
#else
#define PRINTF_FUNC nndbgDetailPrintf
#endif

#ifdef HK_DEBUG

extern "C" hkSystemTime HK_CALL hkGetSystemTime();

static void showKeycode( const char* keycode )
{
	if ( (keycode == HK_NULL) || (keycode[0] == '\0') )
	{
		return;
	}

	const int sz = 20;
	char product[sz];
	char expiry[sz];
	// Skip to ':'
	while (*keycode && (*keycode++ != ':') ) { }

	// Copy to and consume '.'
	{
		char* e = expiry; 
		while (*keycode && (e-expiry < sz-1) && (*keycode != '.') ) *e++ = *keycode++;
		*e = '\0';
	}

	// Copy to and consume '.'
	{
		char* p = product;
		keycode++;
		while (*keycode && (p-product < sz-1) && (*keycode != '.') ) *p++ = *keycode++;
		*p = '\0';
	}

	keycode++;

	PRINTF_FUNC(" %-16s : %s (expires %s)\n", product, keycode, expiry );

}
#endif // HK_DEBUG

static void HK_CALL showHavokBuild()
{
#if defined(HK_DEBUG)
	if( hkBaseSystemInitVerbose )
	{
		PRINTF_FUNC("------------------------------------------------------------------\n");
		PRINTF_FUNC(" Havok - Build (%d)\n", HAVOK_BUILD_NUMBER);
		PRINTF_FUNC(" Version %s\n", HAVOK_SDK_VERSION_NUM_STRING);
		PRINTF_FUNC(" Base system initialized.\n");
		PRINTF_FUNC("------------------------------------------------------------------\n");
		PRINTF_FUNC("Havok License Keys:\n");

		extern const char HK_PHYSICS_2012_KEYCODE[] KEYCODE_ATTRIBUTES;
		showKeycode(HK_PHYSICS_2012_KEYCODE);	

		extern const char HK_PHYSICS_KEYCODE[] KEYCODE_ATTRIBUTES;
		showKeycode(HK_PHYSICS_KEYCODE);

		extern const char HK_ANIMATION_KEYCODE[] KEYCODE_ATTRIBUTES;
		showKeycode(HK_ANIMATION_KEYCODE);	

		extern const char HK_BEHAVIOR_KEYCODE[] KEYCODE_ATTRIBUTES;
		showKeycode(HK_BEHAVIOR_KEYCODE);	

		extern const char HK_DESTRUCTION_2012_KEYCODE[] KEYCODE_ATTRIBUTES;
		showKeycode(HK_DESTRUCTION_2012_KEYCODE);

		extern const char HK_DESTRUCTION_KEYCODE[] KEYCODE_ATTRIBUTES;
		showKeycode(HK_DESTRUCTION_KEYCODE);

		extern const char HK_CLOTH_KEYCODE[] KEYCODE_ATTRIBUTES;
		showKeycode(HK_CLOTH_KEYCODE);	

		extern const char HK_AI_KEYCODE[] KEYCODE_ATTRIBUTES;
		showKeycode(HK_AI_KEYCODE);

		PRINTF_FUNC("------------------------------------------------------------------\n");
	}
#endif
}

void hkBaseSystem::initSingletons()
{
	// ptrToCur is a handle to the current node.
	hkSingletonInitNode** ptrToCur = &hkSingletonInitList;
	hkSingletonInitNode* cur = *ptrToCur;

	hkArray<hkSingletonInitNode*>::Temp again;

#	if defined(HK_DEBUG)
		if (cur == HK_NULL )
		{
			// *****************************************************************
			// The singletons list is empty - the static variable constructors have not been called.
			// If you are using Metrowerks, check that you have called mwInit()
			// prior to calling hkBaseSystem::init()
			// *****************************************************************

			PRINTF_FUNC("\n*****************************************************************\n");
			PRINTF_FUNC("Havok Error:\n");
			PRINTF_FUNC("The singletons list is empty - the static variable constructors have\n");
			PRINTF_FUNC("not been called. Check that the global constructor chain has been called.\n");
			PRINTF_FUNC("Hint: if you are using Metrowerks, check that you have called mwInit()\n");
			PRINTF_FUNC("prior to calling hkBaseSystem::init().\n");
			PRINTF_FUNC("*****************************************************************\n");


			HK_BREAKPOINT(0);
		}
#	endif

	while(cur)
	{
		if( (*cur->m_value == HK_NULL) && (cur->m_createFunc != HK_NULL) )
		{
			void* p = (*cur->m_createFunc)();
			if(p)
			{
				// constructed ok
				*cur->m_value = p;
				ptrToCur = &cur->m_next;
				cur = cur->m_next;
			}
			else
			{
				// If a singleton creation function returns HK_NULL it means
				// that the object was not ready to be constructed (perhaps
				// it depends on another singleton?)
				// Remove it from the list and save it for later
				again.pushBack(cur);
				cur = cur->m_next;
				(*ptrToCur)->m_next = HK_NULL;
				*ptrToCur = cur;
			}
		}
		else // skip already done nodes
		{
			ptrToCur = &cur->m_next;
			cur = cur->m_next;
		}
	}

	// Go through the deferred list.
	while( again.getSize() )
	{
		int origSize = again.getSize();
		for(int i = origSize-1; i>=0; --i)
		{
			cur = again[i];
			HK_ASSERT(0x491ec52a, cur->m_next == HK_NULL);

			void* p = (*cur->m_createFunc)();
			if(p)
			{
				// succeeded, put it back on the global list.
				*cur->m_value = p;
				*ptrToCur = cur;
				ptrToCur = &cur->m_next;
				again.removeAt(i);
			}
		}
		HK_ASSERT2(0x14db3060,  again.getSize() < origSize, "cycle detected in singleton construction");
	}
	// array of singleton nodes is now sorted by dependency.
}


void HK_CALL hkBaseSystem::quitSingletons()
{
	// destroy singletons in reverse order to creation.
	hkInplaceArray<hkSingletonInitNode*,128> nodes;
	hkSingletonInitNode* cur = hkSingletonInitList;
	while(cur)
	{
		if( (*cur->m_value != HK_NULL) && (cur->m_createFunc != HK_NULL) )
		{
			nodes.pushBack(cur);
		}
		cur = cur->m_next;
	}

	for(int i = nodes.getSize()-1; i >= 0; --i)
	{
		hkReferencedObject* obj = static_cast<hkReferencedObject*>(*nodes[i]->m_value);
		HK_ASSERT(0x3407ed4e, obj != HK_NULL);
		obj->removeReferenceLockUnchecked();
		*nodes[i]->m_value = HK_NULL;
	}
}
#if !defined(HK_PLATFORM_SPU)

#include <Common/Base/Config/hkProductFeatures.h>

//  Initialize Havok's subsystems.
hkResult HK_CALL hkBaseSystem::init(hkMemoryRouter* memoryRouter, hkErrorReportFunction errorReportFunction, void* errorReportObject)
{
	if(hkBaseSystemIsInitialized==false)
	{
		initThread( memoryRouter );

		hkReferencedObject::initializeLock();
		#if !defined(HK_PLATFORM_SPU) 
			hkFileSystem::replaceInstance( new hkNativeFileSystem() );
			hkError::replaceInstance( new hkDefaultError(errorReportFunction, errorReportObject) );
		#endif

		initSingletons();
		hkDummySingleton::getInstance().forceLinkage();

		#if !defined(HK_PLATFORM_SPU)
			// hkProductFeatures.cxx must be included in order to register product features
			hkProductFeatures::initialize();
		#endif

		hkBaseSystemIsInitialized = true;
#if !defined(HK_PLATFORM_LRB)
		showHavokBuild();
#endif
		HK_ON_DEBUG_MULTI_THREADING( hkMultiThreadCheck::staticInit(&memoryRouter->heap()));

		HK_CHECK_FLUSH_DENORMALS();
	}

	return HK_SUCCESS;
}

#ifdef HK_PLATFORM_WIIU
#include <cafe/os.h>
#endif


hkResult hkBaseSystem::initThread( hkMemoryRouter* memoryRouter )
{
#ifdef HK_PLATFORM_WIIU
	OSSetThreadSpecific(1,HK_NULL);
#endif

#if defined HK_PLATFORM_PS3_PPU && defined HK_PS3_NO_TLS
	sys_ppu_thread_t id;
	sys_ppu_thread_get_id(&id);
	g_hkPs3PrxTls->registerThread((hkUlong) id);
#endif
#ifndef HK_PLATFORM_ANDROID
	// Apparently Android doesn't initialize the thread-local variables to 0, so this might be non-null even the first time.
	if ( hkMemoryRouter::getInstancePtr() )
	{
		HK_ASSERT2(0x1740f0d2, false, "hkBaseSystem::initThread() was called after thread was already initialized. Don't call this on the main thread." );
	}
#endif

	hkMemoryRouter::replaceInstance( memoryRouter );
	hkMonitorStream::init();
	return HK_SUCCESS;
}

hkResult hkBaseSystem::quitThread()
{
	if( hkMonitorStream* m = hkMonitorStream::getInstancePtr() )
	{
		m->quit();
	}
	if( hkMemoryRouter* a = hkMemoryRouter::getInstancePtr())
	{
		a->replaceInstance( HK_NULL );
	}
	return HK_SUCCESS;
}


//  Quit the subsystems. It is safe to call multiple times.
hkResult HK_CALL hkBaseSystem::quit()
{
	hkResult res = HK_SUCCESS;	

	if(hkBaseSystemIsInitialized )
	{
		HK_ON_DEBUG_MULTI_THREADING( hkMultiThreadCheck::staticQuit() );

		hkReferencedObject::setLockMode( hkReferencedObject::LOCK_MODE_NONE );
		quitSingletons();

		#ifndef HK_PLATFORM_LRB
			// Be nice to the network and shut it down gracefully (if we used it and init'd it)
			if ( hkSocket::s_platformNetInitialized && hkSocket::s_platformNetQuit )
			{
				hkSocket::s_platformNetQuit();
				hkSocket::s_platformNetInitialized = false;
			}
		#endif

		// reverse order to init.
		#if !defined (HK_PLATFORM_SPU)
			hkError::replaceInstance( HK_NULL );
			hkFileSystem::replaceInstance( HK_NULL );
		#endif
		hkReferencedObject::deinitializeLock();

		quitThread();
		hkBaseSystemIsInitialized = false;
	}
	return res;
}
#endif
hkBool HK_CALL hkBaseSystem::isInitialized()
{
	return hkBaseSystemIsInitialized;
}


#if defined(HK_COMPILER_MWERKS)
#	pragma force_active on
#endif

HK_SINGLETON_IMPLEMENTATION(hkDummySingleton);

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
