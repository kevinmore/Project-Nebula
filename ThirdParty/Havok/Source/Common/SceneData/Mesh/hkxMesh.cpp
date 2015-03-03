/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/SceneData/hkSceneData.h>
#include <Common/SceneData/Mesh/hkxMesh.h>
#include <Common/Base/Types/Geometry/hkGeometry.h>

hkxMesh::hkxMesh(hkFinishLoadedObjectFlag f ) : hkReferencedObject(f), m_sections(f), m_userChannelInfos(f)
{
}

hkxMesh::~hkxMesh()
{
	
}

void hkxMesh::collectVertexPositions (hkArray<hkVector4>& verticesInOut) const
{
	for (int si=0; si < m_sections.getSize(); ++si)
	{
		hkxMeshSection* section = m_sections[si];
		section->collectVertexPositions(verticesInOut);
	}
}

void hkxMesh::appendGeometry(hkGeometry& geometryInOut, hkArray<hkxMaterial*>* materialsInOut) const
{
	int materialIndex = -1;

	for (int si=0; si < m_sections.getSize(); ++si)
	{
		int indexOffset = geometryInOut.m_vertices.getSize();

		if (materialsInOut)
		{
			hkxMaterial* material = m_sections[si]->m_material;
			materialIndex = materialsInOut->indexOf(material);
			if (materialIndex == -1)
			{
				materialIndex = materialsInOut->getSize();
				materialsInOut->pushBack(material);
			}
		}

		hkGeometry sectionGeometry;
		m_sections[si]->appendGeometry(sectionGeometry, materialIndex);

		// Increment indices for added geometries
		for (int t=0; t<sectionGeometry.m_triangles.getSize(); t++)
		{
			sectionGeometry.m_triangles[t].m_a += indexOffset;
			sectionGeometry.m_triangles[t].m_b += indexOffset;
			sectionGeometry.m_triangles[t].m_c += indexOffset;
		}

		// Merge arrays
		geometryInOut.m_vertices.insertAt(indexOffset, sectionGeometry.m_vertices.begin(), sectionGeometry.m_vertices.getSize());
		geometryInOut.m_triangles.insertAt(geometryInOut.m_triangles.getSize(), sectionGeometry.m_triangles.begin(), sectionGeometry.m_triangles.getSize());
	}	
}

//
//	Removes the given user channel

void hkxMesh::removeUserChannel(int userChannelIndex)
{
	const int numChannels = m_userChannelInfos.getSize();
	if ( (userChannelIndex < 0) || (userChannelIndex >= numChannels) )
	{
		return;	// Invalid user channel index!
	}

	// Remove info
	m_userChannelInfos.removeAt(userChannelIndex);

	// Remove actual data
	for (int si = m_sections.getSize() - 1; si >= 0; si--)
	{
		hkxMeshSection* section = m_sections[si];
		section->m_userChannels.removeAt(userChannelIndex);
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
