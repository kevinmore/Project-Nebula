/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/System/Io/FileSystem/hkFileSystem.h>
#include <Common/Base/System/Io/IArchive/hkIArchive.h>
#include <Common/Base/System/Io/Reader/Memory/hkMemoryStreamReader.h>

hkIArchive::hkIArchive(hkStreamReader* sb, hkBool bs)
	:	m_streamReader(sb), m_byteSwap(bs)
{
}

hkIArchive::hkIArchive(const char* filename, hkBool bs)
	: m_byteSwap(bs)
{
	m_streamReader = hkFileSystem::getInstance().openReader(filename);
}

hkIArchive::hkIArchive(const void* mem, int memSize, hkBool byteswap)
	: m_byteSwap(byteswap)
{
	m_streamReader.setAndDontIncrementRefCount( new hkMemoryStreamReader(mem, memSize, hkMemoryStreamReader::MEMORY_INPLACE) );
}

hkIArchive::~hkIArchive()
{
}

static HK_FORCE_INLINE void byteswap(char& a, char& b)
{
	char t = a;
	a = b;
	b = t;
}

void hkIArchive::readArrayFloat32(hkDouble64* buf, int nelem)
{
	for (int i =0; i < nelem; i++)
	{
		hkFloat32 d = readFloat32();
		buf[i] = d;
	}
}


void hkIArchive::readArrayGeneric(void* array, int elemsize, int arraySize)
{
	int nreq = elemsize * arraySize;
	HK_ON_DEBUG( int nread = ) m_streamReader->read(array, nreq);
#ifdef HK_DEBUG
	if ( nreq != nread )
	{
		// Failed read: write whole array with 0xffs.
		for ( int i = 0; i < elemsize * arraySize; ++i )
		{
			hkUchar* dst = static_cast<hkUchar*>(array);
			dst[i] = 0xff;
		}
	}
	else
#endif // HK_DEBUG	
	if ( m_byteSwap )
	{
		char* dst = static_cast<char*>(array);
		switch( elemsize )
		{
			case 1:
			{
				break;
			}
			case 2:
			{
				for(int i = 0; i < arraySize; ++i)
				{
					byteswap(dst[0], dst[1]);
					dst += 2;
				}
				break;
			}
			case 4:
			{
				for(int i = 0; i < arraySize; ++i)
				{
					byteswap(dst[0], dst[3]);
					byteswap(dst[1], dst[2]);
					dst += 4;
				}
				break;
			}
			case 8:
			{
				for(int i = 0; i < arraySize; ++i)
				{
					byteswap(dst[0], dst[7]);
					byteswap(dst[1], dst[6]);
					byteswap(dst[2], dst[5]);
					byteswap(dst[3], dst[4]);
					dst += 8;
				}
				break;
			}
			default:
			{
				HK_ASSERT3(0x3d71710d, 0, "elemsize " << elemsize << " not handled.\n" \
					"elemsize must be a power of two and no greater than 8 (64 bits)");

			}
		}
	}
}

int hkIArchive::readRaw(void* buf, int nbytes)
{
	return m_streamReader->read(buf, nbytes);
}

hkBool hkIArchive::isOk() const
{
	return m_streamReader && m_streamReader->isOk();
}

hkStreamReader* hkIArchive::getStreamReader()
{
	return m_streamReader;
}

void hkIArchive::setStreamReader(hkStreamReader* newBuf)
{
	m_streamReader = newBuf;
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
