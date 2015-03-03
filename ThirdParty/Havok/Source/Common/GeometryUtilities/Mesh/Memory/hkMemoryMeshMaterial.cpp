/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>
#include <Common/GeometryUtilities/Mesh/Memory/hkMemoryMeshMaterial.h>
#include <Common/Base/Reflection/hkClass.h>

#include <Common/GeometryUtilities/Mesh/Memory/hkMemoryMeshVertexBuffer.h>

hkMemoryMeshMaterial::hkMemoryMeshMaterial(const char* name)
:	hkMeshMaterial()
,	m_materialName(name)
,	m_userData(0)
,	m_tesselationFactor(0.0f)
,	m_displacementAmount(0.0f)
{
	m_diffuseColor = hkVector4::getConstant<HK_QUADREAL_1>();
	m_ambientColor = hkVector4::getConstant<HK_QUADREAL_0001>();
	m_specularColor = hkVector4::getConstant<HK_QUADREAL_0001>();
	m_emissiveColor.setZero();
}

hkMemoryMeshMaterial::hkMemoryMeshMaterial( hkFinishLoadedObjectFlag flag )
: hkMeshMaterial(flag)
, m_materialName(flag)
, m_textures(flag)
{	
}

hkResult hkMemoryMeshMaterial::createCompatibleVertexFormat(const hkVertexFormat& format, hkVertexFormat& compatibleFormat)
{
    // Memory system can deal with all formats
    compatibleFormat = format;
    return HK_SUCCESS;
}

hkMeshVertexBuffer* hkMemoryMeshMaterial::createCompatibleVertexBuffer(hkMeshVertexBuffer* buffer)
{
    // All are suitable so just return it
    buffer->addReference();
    return buffer;
}

bool hkMemoryMeshMaterial::isCompatible(const hkMeshVertexBuffer* buffer)
{
    return hkMemoryMeshVertexBufferClass.equals(buffer->getClassType());
}

void hkMemoryMeshMaterial::setName(const char* name)
{
	m_materialName = name;
}

const char* hkMemoryMeshMaterial::getName() const
{
	return m_materialName.cString();
}

int hkMemoryMeshMaterial::getNumTextures() const
{
	return m_textures.getSize();
}

hkMeshTexture* hkMemoryMeshMaterial::getTexture(int index) const
{
	return m_textures[index];
}

//
//	Reserves a texture slot for the given texture

int hkMemoryMeshMaterial::reserveTextureSlot(hkMeshTexture* newTex)
{
	const hkMeshTexture::TextureUsageType newHint = newTex->getUsageHint();
	int numTextures = m_textures.getSize();

	int ti = 0;
	for (; ti < numTextures; ti++)
	{
		const hkMeshTexture* myTex = m_textures[ti];
		const hkMeshTexture::TextureUsageType myHint = myTex->getUsageHint();
		if ( myHint > newHint )
		{
			break;	// Found slot!
		}
	}

	if ( ti >= numTextures )
	{
		// Append
		numTextures = ti + 1;
		m_textures.setSize(numTextures, HK_NULL);
	}
	else
	{
		// Insert
		m_textures.insertAt(ti, HK_NULL);
	}

	return ti;
}

void hkMemoryMeshMaterial::addTexture(hkMeshTexture* texture)
{
	const int slotIdx = reserveTextureSlot(texture);
	setTexture(slotIdx, texture);
}

//
//	Replaces the texture set on slot i

void hkMemoryMeshMaterial::setTexture(int i, hkMeshTexture* texture)
{
	m_textures[i] = texture;
}

void hkMemoryMeshMaterial::getColors( hkVector4& diffuse, hkVector4& ambient, hkVector4& specular, hkVector4& emissive ) const
{
	diffuse = m_diffuseColor;
	ambient = m_ambientColor;
	specular = m_specularColor;
	emissive = m_emissiveColor;
}

void hkMemoryMeshMaterial::setColors( const hkVector4& diffuse, const hkVector4& ambient, const hkVector4& specular, const hkVector4& emissive )
{
	m_diffuseColor = diffuse;
	m_ambientColor = ambient;
	m_specularColor = specular;
	m_emissiveColor = emissive;
}

//
//	Tests whether two materials are equal

bool hkMemoryMeshMaterial::equals(const hkMeshMaterial* other)
{
	// Should have the same class
	if ( !hkMemoryMeshMaterialClass.equals(other->getClassType()) )
	{
		return false;
	}
	
	const hkMemoryMeshMaterial* mmtl = reinterpret_cast<const hkMemoryMeshMaterial*>(other);

	// Compare displacements
	if ( (m_displacementAmount != other->getDisplacementAmount()) || (m_tesselationFactor != other->getTesselationFactor()) )
	{
		return false;
	}
	
	// Compare names
	if ( m_materialName.compareTo(mmtl->m_materialName) )
	{
		return false;
	}

	// Compare textures
	if ( m_textures.getSize() != mmtl->m_textures.getSize() )
	{
		return false;
	}
	for (int k = m_textures.getSize() - 1; k >= 0; k--)
	{
		// Dominants are more like vertex data, will be merged together. We can exclude them from the comparison
		if ( m_textures[k] && (m_textures[k]->getUsageHint() == hkMeshTexture::DOMINANTS) )
		{
			continue;
		}

		if ( !m_textures[k]->equals(mmtl->m_textures[k]) )
		{
			return false;
		}
	}

	// Compare colors
	hkVector4Comparison cmp0 = m_diffuseColor.equal(mmtl->m_diffuseColor);
	hkVector4Comparison cmp1 = m_ambientColor.equal(mmtl->m_ambientColor);
	hkVector4Comparison cmp2 = m_specularColor.equal(mmtl->m_specularColor);
	hkVector4Comparison cmp3 = m_emissiveColor.equal(mmtl->m_emissiveColor);
	cmp0.setAnd(cmp0, cmp1);
	cmp2.setAnd(cmp2, cmp3);
	cmp0.setAnd(cmp0, cmp2);
	return cmp0.allAreSet() ? true : false;
}

//
//	Assignment operator

hkMemoryMeshMaterial& hkMemoryMeshMaterial::operator=(const hkMemoryMeshMaterial& other)
{
	m_materialName	= other.m_materialName;
	m_diffuseColor	= other.m_diffuseColor;
	m_ambientColor	= other.m_ambientColor;
	m_specularColor	= other.m_specularColor;
	m_emissiveColor	= other.m_emissiveColor;
	m_displacementAmount	= other.m_displacementAmount;
	m_tesselationFactor		= other.m_tesselationFactor;

	m_textures.setSize(0);
	m_textures.append( other.m_textures );

	return *this;
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
