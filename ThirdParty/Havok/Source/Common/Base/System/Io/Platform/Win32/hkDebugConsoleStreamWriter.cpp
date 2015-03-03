/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/System/Io/Platform/Win32/hkDebugConsoleStreamWriter.h>
#include <Common/Base/Fwd/hkcstdio.h>

#if defined(HK_PLATFORM_WIN32)
#	include <Common/Base/Fwd/hkwindows.h>
#elif defined(HK_PLATFORM_XBOX) || defined(HK_PLATFORM_XBOX360)
#	include <xtl.h>
#else
#	error debug console is only for win32 or xbox
#endif

int hkDebugConsoleStreamWriter::write(const void* buf, int nbytes)
{
	// careful about accessing buf[-1]
	if(nbytes!=0)
	{
		const char* cbuf = static_cast<const char*>(buf);
		if(cbuf[nbytes-1] == HK_NULL)
		{
			OutputDebugStringA( cbuf );
#if !defined(HK_PLATFORM_XBOX) && !defined(HK_PLATFORM_XBOX360) // as printf is routed through debug out anyway
			printf("%s", cbuf);
#endif
		}
		else
		{
			hkArray<char> wbuf(nbytes+1);
			hkArray<char>::copy( &wbuf[0], cbuf, nbytes);
			wbuf[nbytes] = '\0';
			OutputDebugStringA( &wbuf[0] );
#if !defined(HK_PLATFORM_XBOX) && !defined(HK_PLATFORM_XBOX360) // as printf is routed through debug out anyway
			printf("%s", &wbuf[0]);
#endif
		}
	}

	return nbytes;
}

hkBool hkDebugConsoleStreamWriter::isOk() const
{
	return true;
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
