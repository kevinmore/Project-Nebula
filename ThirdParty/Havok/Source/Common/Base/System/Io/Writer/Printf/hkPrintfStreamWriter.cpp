/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/System/Io/Writer/Printf/hkPrintfStreamWriter.h>
#include <Common/Base/Fwd/hkcstdio.h>

using namespace std;

int hkPrintfStreamWriter::write( const void* buf, int nbytes)
{
	// careful about accessing buf[-1]
	if( nbytes > 0 )
	{
		const char* cbuf = static_cast<const char*>(buf);
		if ( cbuf[nbytes-1] == HK_NULL)
		{
#ifdef HK_PLATFORM_CTR
			nndbgDetailPrintf("%s", static_cast<const char*>(buf));
#else
			printf("%s", static_cast<const char*>(buf));
#endif
		}
		else // need to null terminate
		{
			hkArray<char> wbuf(int(nbytes+1));
			hkArray<char>::copy( &wbuf[0], static_cast<const char*>(buf), int(nbytes));
			wbuf[int(nbytes)] = '\0';
#ifdef HK_PLATFORM_CTR
			nndbgDetailPrintf( "%s", &wbuf[0] ); 
#else
			printf( "%s", &wbuf[0] ); 
#endif
		}
	}
	return nbytes;
}

hkBool hkPrintfStreamWriter::isOk() const
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
