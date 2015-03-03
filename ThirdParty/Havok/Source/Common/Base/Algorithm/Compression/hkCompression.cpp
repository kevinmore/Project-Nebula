/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>

#include <Common/Base/Algorithm/Compression/hkCompression.h>
#include <Common/Base/Algorithm/Compression/hkBufferCompression.h>

namespace hkCompression
{
	// Blocks start with a 16-byte header:
	//   <name>  <len>
	//   magic:    2   [03BC, little-endian]
	//	 version:  1   [currently always "1", anything less than 16 will be read by this code]
	//   compress: 1   [0 or 1 indicating compressed or uncompressed block]
	//   pivotpos: 4   [offset needed for decompression-in-place, unsigned little-endian]
	//   complen:  4   [length of compressed data in bytes, unsigned little-endian]
	//   uncomplen:4   [length of uncompressed data in bytes, unsigned little-endian]
	//
	class BlockHeader
	{
	public:

		// Constructor, added to prevent uninitialized members warning as error on PlayStation(R)3.
		HK_FORCE_INLINE BlockHeader()
		:	m_compressed_len(0)
		,	m_uncompressed_len(0)
		,	m_pivotpos(0)
		,	m_iscompressed(false)
		{}

		hkResult parse(const void* bufIn)
		{
			const hkUchar* buf = static_cast<const hkUchar*>(bufIn);
			int magic = buf[0] + (buf[1] << 8),
				version = buf[2],
				compressed = buf[3];

			if (!(magic == 0x03bc && version < 16 &&
				  (compressed == 0 || compressed == 1)))
			{
				return HK_FAILURE;
			}
			m_iscompressed = compressed == 1 ? true : false;

			m_pivotpos         = readUint32(buf+4);
			m_compressed_len   = readUint32(buf+8);
			m_uncompressed_len = readUint32(buf+12);

			if (m_compressed_len > m_uncompressed_len)
			{
				return HK_FAILURE;
			}

			if (!m_iscompressed && m_compressed_len != m_uncompressed_len)
			{
				return HK_FAILURE;
			}

			return HK_SUCCESS;
		}

		void write(hkUchar* buf)
		{
			HK_ASSERT(0x43562222, m_compressed_len <= m_uncompressed_len);
			buf[0] = 0xbc;
			buf[1] = 0x03;
			buf[2] = 1;
			buf[3] = m_iscompressed ? 1 : 0;
			writeUint32(buf+4, m_pivotpos);
			writeUint32(buf+8, m_compressed_len);
			writeUint32(buf+12, m_uncompressed_len);
		}
	private:
		static void writeUint32(hkUchar* buf, hkUint32 x)
		{
			buf[0] = (hkUchar)(x & 0xff); x>>=8;
			buf[1] = (hkUchar)(x & 0xff); x>>=8;
			buf[2] = (hkUchar)(x & 0xff); x>>=8;
			buf[3] = (hkUchar)(x & 0xff); x>>=8;
		}
		static hkUint32 readUint32(const hkUchar* buf)
		{
			return buf[0] + (buf[1] << 8) + (buf[2] << 16) + (buf[3] << 24);
		}
	public:

		
			/// Length of the compressed block, in bytes, not counting header
		hkUint32 m_compressed_len;
			/// Length of the block when uncompressed, in bytes
		hkUint32 m_uncompressed_len;
			/// Offset for in-place decompression (currently unimplemented)
		hkUint32 m_pivotpos;
			/// Whether this block is stored compressed or uncompressed
		bool m_iscompressed;
	};

	template <class T>
	static hk_size_t arrayDiff(const T* x, const T* y)
	{
		HK_ASSERT(0x4365ab3f, x >= y);
		return static_cast<hk_size_t>(x - y);
	}
}


hkCompression::Result hkCompression::compress(const void*& in_param, hk_size_t inlen, void*& out_param, hk_size_t outlen, hk_size_t bufsize)
{
	HK_ASSERT2(0x03bc4321, bufsize > 128, "compression buffer size too small");
	hkUchar* out = static_cast<hkUchar*>(out_param);
	hkUchar* out_end = out + outlen;
	const hkUchar* in = static_cast<const hkUchar*>(in_param);
	const hkUchar* in_end = in + inlen;
	// The maximum size of a compressed block (including header) will be "bufsize"
	while (arrayDiff(out_end, out) >= bufsize && in < in_end)
	{
		BlockHeader h;
		hkUchar* headerdata = out;
		out += BLOCK_HEADER_SIZE;
		hk_size_t uncompsz = hkMath::min2(arrayDiff(in_end, in), bufsize - BLOCK_HEADER_SIZE);
		hk_size_t compsz = hkBufferCompression::compressBuffer(in, uncompsz, out, uncompsz);
		if (compsz == 0 || compsz > uncompsz)
		{
			// Compression failed (the data got bigger and maybe didn't fit in the buffer)
			// Write the data uncompressed
			h.m_pivotpos = static_cast<hkUint32>(-1);
			h.m_compressed_len = (hkUint32)uncompsz;
			h.m_uncompressed_len = (hkUint32)uncompsz;
			h.m_iscompressed = false;
			h.write(headerdata);

			hkString::memCpy(out, in, (int)uncompsz);
			compsz = uncompsz;
		}
		else
		{
			// Compression succeeded
			h.m_pivotpos = static_cast<hkUint32>(-1);
			h.m_compressed_len = (hkUint32)compsz;
			h.m_uncompressed_len = (hkUint32)uncompsz;
			h.m_iscompressed = true;
			h.write(headerdata);
		}
		out += compsz;
		in += uncompsz;
	}

	in_param = in;
	out_param = out;

	if (in == in_end)
	{
		return COMP_NEEDINPUT;
	}
	else
	{
		return COMP_NEEDOUTPUT;
	}
}

static hkCompression::Result _decompress(const hkUchar*& in, hk_size_t inlen, hkUchar*& out, hk_size_t outlen, bool checkinput)
{
	using namespace hkCompression;

	hkUchar* out_end = out + outlen;
	const hkUchar* in_end = in + inlen;
	hkCompression::BlockHeader h; h.m_iscompressed = false; h.m_compressed_len = 0; h.m_uncompressed_len = 0;
	while (1)
	{
		if (in_end - in < BLOCK_HEADER_SIZE)
		{
			return COMP_NEEDINPUT;
		}
		if (h.parse(in) == HK_FAILURE)
		{
			return COMP_ERROR;
		}

		if (arrayDiff(in_end, (in + BLOCK_HEADER_SIZE)) < h.m_compressed_len)
		{
			return COMP_NEEDINPUT;
		}
		if (arrayDiff(out_end, out) < h.m_uncompressed_len)
		{
			return COMP_NEEDOUTPUT;
		}
		HK_ON_DEBUG(hkUint8* old_out = out);
		if (h.m_iscompressed)
		{
			if (checkinput)
			{
				void* out2 = out;
				hkResult res = hkBufferCompression::decompressBufferChecked(in + BLOCK_HEADER_SIZE, h.m_compressed_len, out2, arrayDiff(out_end, out));
				out = static_cast<hkUchar*>(out2);
				if (res == HK_FAILURE)
				{
					return COMP_ERROR;
				}
			}
			else
			{
				void* out2 = out;
				hkBufferCompression::decompressBufferFast(in + BLOCK_HEADER_SIZE, h.m_compressed_len, out2);
				out = static_cast<hkUchar*>(out2);
			}
		}
		else
		{
			HK_ASSERT(0xba87de21, h.m_compressed_len == h.m_uncompressed_len);
			hkString::memCpy(out, in + BLOCK_HEADER_SIZE, h.m_uncompressed_len);
			out += h.m_uncompressed_len;
		}
		HK_ON_DEBUG(HK_ASSERT(0x54363322, out == old_out + h.m_uncompressed_len));
		in += BLOCK_HEADER_SIZE + h.m_compressed_len;
	}
}

hkCompression::Result hkCompression::decompress(const void*& in_void, hk_size_t inlen, void*& out_void, hk_size_t outlen, bool checkinput)
{
	const hkUchar* in = static_cast<const hkUchar*>(in_void);
	hkUchar* out = static_cast<hkUchar*>(out_void);
	Result res = _decompress(in, inlen, out, outlen, checkinput);
	in_void = in;
	out_void = out;
	return res;
}


hkResult hkCompression::getDecompressionBufferSize(const void* in, hk_size_t& compressed_out, hk_size_t& uncompressed_out)
{
	BlockHeader h; h.m_uncompressed_len=0; h.m_compressed_len=0;
	if (h.parse(in) == HK_FAILURE) return HK_FAILURE;

	compressed_out = BLOCK_HEADER_SIZE + h.m_compressed_len;
	uncompressed_out = h.m_uncompressed_len;
	return HK_SUCCESS;
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
