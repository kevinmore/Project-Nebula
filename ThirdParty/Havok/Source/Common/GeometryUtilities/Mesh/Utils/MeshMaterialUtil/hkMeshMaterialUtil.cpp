/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/GeometryUtilities/hkGeometryUtilities.h>
#include <Common/GeometryUtilities/Mesh/Utils/MeshMaterialUtil/hkMeshMaterialUtil.h>
#include <Common/GeometryUtilities/Mesh/hkMeshShape.h>

void hkMeshMaterialUtil::convert(const hkMeshMaterial* source, hkMeshMaterial* destination)
{
	destination->setName(source->getName());

	for(int i = 0; i < source->getNumTextures(); ++i)
	{
		hkMeshTexture* texture = source->getTexture(i);
		destination->addTexture(texture);		
	}
	
	// If there are no textures, set material colors
	if( source->getNumTextures() == 0 )
	{
		hkVector4 diffuse;
		hkVector4 ambient;
		hkVector4 specular;
		hkVector4 emmisive;
		source->getColors(diffuse, ambient, specular, emmisive);
		destination->setColors(diffuse, ambient, specular, emmisive);
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
