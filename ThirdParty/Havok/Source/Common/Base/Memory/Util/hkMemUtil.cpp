/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Memory/Util/hkMemUtil.h>
#include <Common/Base/Fwd/hkcstring.h>

using namespace std;

#if defined(HK_PLATFORM_WIIU)
namespace hkMemUtil
{


#if 0 && (CAFE_OS_SDK_VERSION >= 20900)
#define CHECK_OSBLOCKMOVE(dst, src, nbytes, flush) OSBlockMove(dst, src, nbytes, flush)
#else
#define CHECK_OSBLOCKMOVE(dst, src, nbytes, flush) if(dst != src) { OSBlockMove(dst, src, nbytes, flush); } else
#endif
	void HK_CALL memCpy(void* dst, const void* src, int nbytes)
	{
		CHECK_OSBLOCKMOVE(dst, src, nbytes, false);
	}

	void HK_CALL memSet(void* dst, const int c, int n)
	{
		OSBlockSet(dst,c,(size_t)n);
	}

	void HK_CALL memCpyBackwards(void* dst, const void* src, int nbytes)
	{
		CHECK_OSBLOCKMOVE(dst, src, nbytes, false);
	}

	void HK_CALL memMove(void* dst, const void* src, int nbytes)
	{
		CHECK_OSBLOCKMOVE(dst, src, nbytes, false);
	}
}
#else

template<unsigned int N>
static void HK_FORCE_INLINE memCopyBackwards(void* dst, const void* src, int nbytes)
{
	HK_ASSERT(0x341ec349, dst != src);
	HK_ASSERT(0x4543e434, (dst > src) || ((static_cast<char*>(dst)+nbytes) <= src) );
	typedef typename hkMemUtil::TypeFromAlign<N>::type CopyType;
	HK_ASSERT(0x70ffeb38, N == 1 || ( ((hkUlong)dst & (N-1)) == 0 && ((hkUlong)src & (N-1)) == 0 && (nbytes & (N-1)) == 0 ));
	unsigned int i,j;
	for( i = nbytes, j = nbytes/sizeof(CopyType)-1; i >= sizeof(CopyType); i -= sizeof(CopyType), --j )
	{
		static_cast<CopyType*>(dst)[j] = static_cast<const CopyType*>(src)[j];
	}
}

namespace hkMemUtil
{
	void HK_CALL memCpy(void* dst, const void* src, int nbytes)
	{
		memcpy(dst, src, nbytes);
	}

	void HK_CALL memSet(void* dst, const int c, int n)
	{
		memset(dst,c,(unsigned)n);
	}

	void HK_CALL memCpyBackwards(void* dst, const void* src, int nbytes)
	{
	#if defined(HK_PLATFORM_PSVITA)
		memmove(dst, src, nbytes);
	#else
		if( (nbytes & ~7) && (nbytes & 0x7) == 0 && ((hkUlong)dst & 0x7) == 0 && ((hkUlong)src & 0x7) == 0 )
		{
			memCopyBackwards<8>(dst, src, nbytes);
			dst = static_cast<char*>(dst) + (nbytes & ~7);
			src = static_cast<const char*>(src) + (nbytes & ~7);
			nbytes &= 7;
		}

		if( (nbytes & ~3) && (nbytes & 0x3) == 0 && ((hkUlong)dst & 0x3) == 0 && ((hkUlong)src & 0x3) == 0 )
		{
			memCopyBackwards<4>(dst, src, nbytes);
			dst = static_cast<char*>(dst) + (nbytes & ~3);
			src = static_cast<const char*>(src) + (nbytes & ~3);
			nbytes &= 3;
		}
		
		if( (nbytes & ~1) && (nbytes & 0x1) == 0 && ((hkUlong)dst & 0x1) == 0 && ((hkUlong)src & 0x1) == 0 )
		{
			memCopyBackwards<2>(dst, src, nbytes);
			dst = static_cast<char*>(dst) + (nbytes & ~1);
			src = static_cast<const char*>(src) + (nbytes & ~1);
			nbytes &= 1;
		}
		
		memCopyBackwards<1>(dst, src, nbytes);
	#endif
	}

	void HK_CALL memMove(void* dst, const void* src, int nbytes)
	{
		if( dst > src )
		{
			memCpyBackwards(dst, src, nbytes);
		}
		else if( dst < src )
		{
			memmove(dst, src, nbytes);
		}
	}
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
