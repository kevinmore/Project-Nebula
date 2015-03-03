/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Algorithm/Compression/hkBufferCompression.h>
#include <Common/Base/Algorithm/Compression/hkBufferCompressionInternal.h>

static HK_FORCE_INLINE hkResult decompressBuffer(const void* in_start, hk_size_t inlen, void*& out_start, hk_size_t outlen, bool checkinput)
{
	const hkUchar* in = static_cast<const hkUchar*>(in_start);
	hkUchar* out = static_cast<hkUchar*>(out_start);
	hkUchar* out_end = static_cast<hkUchar*>(out) + outlen;
	const hkUchar* in_end = static_cast<const hkUchar*>(in) + inlen;

	while (in < in_end)
	{
		int control = *in++;
		int backlen = control >> hkBufferCompression::LITCOUNT_BITS;
		int fieldB = (control & (hkBufferCompression::MAX_LITERAL_LEN - 1));
		if (backlen == 0)
		{
			// Literal run
			int litlen = fieldB + 1;
			if (checkinput)
			{
				if (in + litlen > in_end || out + litlen > out_end)
				{
					return HK_FAILURE;
				}
			}
			hkBufferCompression::copyout(out, in, litlen);
		}
		else
		{
			// Backreference
			int offset = (fieldB << 8) + (*in++) + 1;
			backlen += 2;
			if (backlen == 9)
			{
				// long-format backreference
				backlen += (*in++);
			}
			const hkUchar* ref = out - offset;
			if (checkinput)
			{
				if (out + backlen > out_end)
				{
					return HK_FAILURE;
				}
			}
			hkBufferCompression::copyout(out, ref, backlen);
		}
	}
	out_start = out;
	return HK_SUCCESS;
}

hkResult hkBufferCompression::decompressBufferChecked(const void* in, hk_size_t inlen, void*& out_start, hk_size_t outlen)
{
	return decompressBuffer(in, inlen, out_start, outlen, true);
}

void hkBufferCompression::decompressBufferFast(const void* in, hk_size_t inlen, void*& out_start)
{
	decompressBuffer(in, inlen, out_start, (hk_size_t)-1, false);
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
