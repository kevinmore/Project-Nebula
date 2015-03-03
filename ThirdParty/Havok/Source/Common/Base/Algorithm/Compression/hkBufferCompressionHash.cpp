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

namespace hkBufferCompression
{
	class Dictionary
	{
		public:
			Dictionary(const hkUchar* d) 
				: m_keys(hkMemTempBlockAlloc<int>(DICTSIZE)), 
				  m_values(hkMemTempBlockAlloc<const hkUchar*>(DICTSIZE)), 
				  m_data_end(d)
			{
				for (int i=0; i < DICTSIZE; i++)
				{
					m_keys[i] = EMPTY;
					m_values[i] = 0;
				}
			}
			~Dictionary()
			{
				hkMemTempBufFree(m_values, DICTSIZE);
				hkMemTempBufFree(m_keys, DICTSIZE);
			}


			void find(int key, const hkUchar* input, int& matchlen, int& offset, const hkUchar*& ptr)
			{
				int hpos = hashpos(key);
				int step = getstep(key);
				matchlen = 0;
				for (int iter=0; iter<MAXCHAIN; iter++)
				{
					if (m_keys[hpos] == EMPTY)
					{
						return; // not found
					}

					const hkUchar* p = m_values[hpos];
					int curroffset = (int)(input - p);
					HK_ASSERT(0x7c927846, curroffset > 0);
					if (curroffset <= MAX_MATCH_OFFSET)
					{
						int maxmatch = hkMath::min2( int(MAX_LONG_MATCH_LEN), int(m_data_end - input) );

						int mlen = 0;
						for (mlen = 0; mlen < maxmatch; mlen++)
						{
							if (p[mlen] != input[mlen]) break;
						}
						if (mlen > matchlen)
						{
							matchlen = mlen;
							ptr = p;
							offset = curroffset;
						}
					}

					hpos = (hpos + step) % DICTSIZE;
				}
			}

			void insert(int key, const hkUchar* input)
			{
				int hpos = hashpos(key);
				int step = getstep(key);
				for (int iter=0; iter<MAXCHAIN - 1; iter++)
				{
					if (m_keys[hpos] == EMPTY || input - m_values[hpos] > MAX_MATCH_OFFSET)
					{
						break;
					}
					hpos = (hpos + step) % DICTSIZE;
				}

				m_keys[hpos] = key;
				m_values[hpos] = input;
			}

		private:

			static inline int hashpos(int k)
			{
				unsigned int p = ((k >> 6) ^ (k + 25543) ^ (k << 13));
				return p % DICTSIZE;
			}
			static inline int getstep(int k)
			{
				return 1; // linear probing
			}

			int* m_keys;
			const hkUchar** m_values;
			const hkUchar* m_data_end;
			enum { EMPTY = -1 };
			enum { MAXCHAIN = 16 };
	};
}


hk_size_t hkBufferCompression::hashCompress(const void* in_param, hk_size_t inlen, void* outbuf_param, hk_size_t outlen)
{
		// in is the next byte to be read, out is the next byte to be written
	hkUchar* outbuf = static_cast<hkUchar*>(outbuf_param);
	const hkUchar* in = static_cast<const hkUchar*>(in_param);
	const hkUchar* in_end = in + inlen;
	CompressedOutput out(outbuf);

	Dictionary dict(in_end);

	while (in < in_end - 3 && out.bytesWritten() + 10 < outlen)
	{
		hkUchar curr = *in;
		int matchlen=0, offset=0, context=0;
		const hkUchar* ptr;
		int consumed;

		context = (in[0] << 16) + (in[1] << 8) + in[2];

		dict.find(context, in, 
			matchlen, offset, ptr);

		if (matchlen >= 3)
		{
			out.writeBackref(matchlen, offset);
			consumed = matchlen;
		}
		else
		{
			out.writeLiteral(curr);
			consumed = 1;
		}
		dict.insert(context, in);

		in += consumed;
	}
	// Handle the last 0-3 characters as literals
	while (in < in_end && out.bytesWritten() + 10 < outlen)
	{
		out.writeLiteral(*in);
		in++;
	}
	out.endLiteralRun();
	if (out.bytesWritten() + 10 >= outlen)
	{
		return 0; //output buffer overflow
	}
	else
	{
		return out.bytesWritten();
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
