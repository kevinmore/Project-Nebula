/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>
#include <Common/GeometryUtilities/Mesh/hkMeshShape.h>
#include <Common/GeometryUtilities/Mesh/Utils/DisplacementMappingUtil/hkDisplacementMappingUtil.h>

//
//	Returns the displacement map set on the given material

hkMeshTexture* HK_CALL hkDisplacementMappingUtil::getDisplacementMap(const hkMeshMaterial* mtl)
{
	if ( mtl )
	{
		for (int ti = mtl->getNumTextures() - 1; ti >= 0; ti--)
		{
			hkMeshTexture* tex = mtl->getTexture(ti);
			if ( tex && (tex->getUsageHint() == hkMeshTexture::DISPLACEMENT) )
			{
				return tex;
			}
		}
	}

	// No map found
	return HK_NULL;
}

//
//	Returns the dominants map set on the given material

hkMeshTexture* HK_CALL hkDisplacementMappingUtil::getDominantsMap(const hkMeshMaterial* mtl)
{
	if ( mtl )
	{
		for (int ti = mtl->getNumTextures() - 1; ti >= 0; ti--)
		{
			hkMeshTexture* tex = mtl->getTexture(ti);
			if ( tex && (tex->getUsageHint() == hkMeshTexture::DOMINANTS) )
			{
				return tex;
			}
		}
	}

	// No map found
	return HK_NULL;
}

//
//	Sets the given dominants map on the given material

void HK_CALL hkDisplacementMappingUtil::setDominantsMap(hkMeshMaterial* mtl, hkMeshTexture* dominantsMap)
{
	if ( !dominantsMap )
	{
		return;
	}

	// See if the material already has a dominants map on
	for (int ti = mtl->getNumTextures() - 1; ti >= 0; ti--)
	{
		const hkMeshTexture* tex = mtl->getTexture(ti);
		if ( tex && (tex->getUsageHint() == hkMeshTexture::DOMINANTS) )
		{
			// The material has a dominants map already. Set the new one!
			mtl->setTexture(ti, dominantsMap);
			return;
		}
	}

	// The material does not have a dominants map set. Set it now!
	mtl->addTexture(dominantsMap);
}

//
//	Sets the I-th displacement map on the given material to the given texture

void HK_CALL hkDisplacementMappingUtil::setDisplacementMap(int i, hkMeshMaterial* mtl, hkMeshTexture* displacementMap)
{
	if ( !displacementMap )
	{
		return;
	}

	// Count the number of displacement maps already set
	const int numTextures	= mtl->getNumTextures();
	int firstMapIdx			= -1;

	// Compute the index of the first displacement map
	for (int ti = 0; ti < numTextures; ti++)
	{
		const hkMeshTexture* tex = mtl->getTexture(ti);
		if ( tex && (tex->getUsageHint() == hkMeshTexture::DISPLACEMENT) )
		{
			firstMapIdx = ti;
			break;
		}
	}

	// If no displacement maps were set, add ours now
	if ( firstMapIdx >= 0 )
	{
		// We have at least one displacement map.
		// See if we can find a slot for ours
		for (int ti = firstMapIdx; (ti < numTextures); ti++)
		{
			const hkMeshTexture* tex = mtl->getTexture(ti);
			if ( tex && (tex->getUsageHint() == hkMeshTexture::DISPLACEMENT) )
			{
				if ( i-- == 0 )
				{
					// Add here
					mtl->setTexture(ti, displacementMap);
					return;
				}
			}
		}
	}

	// We could not find a slot, append
	while ( i-- >= 0 )
	{
		mtl->addTexture(displacementMap);
	}	
}

//
//	Constructor

hkDisplacementMappingUtil::DominantsBuffer::DominantsBuffer()
:	m_data(HK_NULL)
,	m_texture(HK_NULL)
{}

//
//	Allocates a buffer for N dominants

void hkDisplacementMappingUtil::DominantsBuffer::alloc(int numDominants)
{
	// Allocate
	const int size	= sizeof(DominantInfo) * numDominants + sizeof(Descriptor);
	m_data			= hkAlignedAllocate<hkUint8>(16, size, HK_MEMORY_CLASS_SCENE_DATA);

	// Initialize descriptor
	Descriptor& d	= *reinterpret_cast<Descriptor*>(m_data);
	hkString::memSet(&d, 0, sizeof(Descriptor));
	d.m_offset		= sizeof(Descriptor);
	d.m_stride		= sizeof(DominantInfo);
	d.m_numElements	= numDominants;
}

//
//	Creates and returns the texture

hkMeshTexture* hkDisplacementMappingUtil::DominantsBuffer::realize(hkMeshSystem* meshSystem)
{
	if ( m_data && !m_texture )
	{
		hkMeshTexture* dMap = meshSystem->createTexture();
		hkStringBuf strb;	strb.printf("Dominants_0x%08X", (hkUlong)dMap);
		dMap->setUsageHint(hkMeshTexture::DOMINANTS);
		dMap->setFilename(strb.cString());
		dMap->setTextureCoordChannel(0);	// We don't care about the uv channel, it won't be used!

		Descriptor& d = *reinterpret_cast<Descriptor*>(m_data);
		dMap->setData(m_data, sizeof(Descriptor) + d.m_numElements * d.m_stride, hkMeshTexture::RAW);
		m_texture.setAndDontIncrementRefCount(dMap);
	}

	return m_texture;
}

//
//	Destructor

hkDisplacementMappingUtil::DominantsBuffer::~DominantsBuffer()
{
	if ( m_data )
	{
		hkAlignedDeallocate<hkUint8>(m_data);
		m_data = HK_NULL;
	}

	m_texture = HK_NULL;
}

//
//	Clones a material without preserving the name

hkMeshMaterial* HK_CALL hkDisplacementMappingUtil::duplicateMaterial(hkMeshSystem* meshSystem, const hkMeshMaterial* src)
{
	hkMeshMaterial* dst = meshSystem->cloneMaterial(src);
	hkStringBuf strb;	strb.printf("%s_0x%08X", src->getName(), (hkUlong)dst);
	dst->setName(strb.cString());
	return dst;
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
