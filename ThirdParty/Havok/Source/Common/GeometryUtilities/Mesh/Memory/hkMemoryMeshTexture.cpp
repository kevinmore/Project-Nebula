/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>
#include <Common/GeometryUtilities/Mesh/Memory/hkMemoryMeshTexture.h>
#include <Common/Base/Reflection/hkClass.h>

hkMemoryMeshTexture::hkMemoryMeshTexture()
:m_format(hkMeshTexture::Unknown)
,m_hasMipMaps(false)
,m_filterMode(hkMeshTexture::ANISOTROPIC)
,m_usageHint(hkMeshTexture::UNKNOWN)
,m_textureCoordChannel(-1)
{

}

void hkMemoryMeshTexture::getData(hkUint8*& data, int& size, hkMemoryMeshTexture::Format& format)
{
	data = m_data.begin();
	size = m_data.getSize();
	format = m_format;
}

void hkMemoryMeshTexture::setData(hkUint8* data, int size, hkMemoryMeshTexture::Format format)
{
	m_format = format;

	// Copy raw formats locally, to have a consistent behaviour
	if ( format == RAW )
	{
		m_data.setSize(0);
		m_data.append(data, size);
	}
	else
	{
		m_data.setDataUserFree(data, size, size);
	}
}

hkMeshTexture::Format hkMemoryMeshTexture::getFormat() const
{
	return m_format;
}

const char* hkMemoryMeshTexture::getFilename() const
{
	return m_filename.cString();
}

void hkMemoryMeshTexture::setFilename( const char* filename )
{
	m_filename = filename;
}

bool hkMemoryMeshTexture::getHasMipMaps() const
{
	return m_hasMipMaps;
}

void hkMemoryMeshTexture::setHasMipMaps(bool hasMipMaps)
{
	m_hasMipMaps = hasMipMaps;
}

hkMemoryMeshTexture::FilterMode hkMemoryMeshTexture::getFilterMode() const
{
	return m_filterMode;
}

void hkMemoryMeshTexture::setFilterMode(hkMemoryMeshTexture::FilterMode filterMode)
{
	m_filterMode = filterMode;
}

hkMeshTexture::TextureUsageType hkMemoryMeshTexture::getUsageHint() const
{
	return m_usageHint;
}

void hkMemoryMeshTexture::setUsageHint( hkMeshTexture::TextureUsageType hint )
{
	m_usageHint = hint;
}

hkInt32 hkMemoryMeshTexture::getTextureCoordChannel() const
{
	return m_textureCoordChannel;
}

void hkMemoryMeshTexture::setTextureCoordChannel( hkInt32 channelIndex )
{
	m_textureCoordChannel = channelIndex;
}

//
//	hkMeshTexture implementation

hkMeshTexture::Sampler* hkMemoryMeshTexture::createSampler() const
{
	
	HK_ASSERT(0x36c5bc1c, false);
	return HK_NULL;
}

//
//	Tests whether two textures are equal

bool hkMemoryMeshTexture::equals(const hkMeshTexture* other) const
{
	if ( !other || !hkMemoryMeshTextureClass.equals(other->getClassType()) )
	{
		return false;
	}

	const hkMemoryMeshTexture* mmo = reinterpret_cast<const hkMemoryMeshTexture*>(other);
	if ( (m_format != mmo->m_format)					||
		 (m_hasMipMaps != mmo->m_hasMipMaps)			||
		 (m_filterMode != mmo->m_filterMode)			||
		 (m_usageHint != mmo->m_usageHint)				||
		 (m_data.getSize() != mmo->m_data.getSize())	||
		 (m_filename.compareTo(mmo->m_filename) != 0)	||
		 (m_textureCoordChannel != mmo->m_textureCoordChannel) )
	{
		return false;
	}
	
	// Finally, compare the data
	return hkString::memCmp(m_data.begin(), mmo->m_data.begin(), m_data.getSize()) == 0;
}

//
//	Returns the class type

const hkClass* hkMemoryMeshTexture::getClassType() const
{
	return &hkMemoryMeshTextureClass;
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
