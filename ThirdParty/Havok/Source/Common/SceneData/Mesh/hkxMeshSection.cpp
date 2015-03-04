/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/SceneData/hkSceneData.h>
#include <Common/SceneData/Mesh/hkxMeshSection.h>
#include <Common/Base/Types/Geometry/hkGeometry.h>

hkxMeshSection::hkxMeshSection(const hkxMeshSection& other)
	: hkReferencedObject(other)
{
	for(int i = 0; i < other.m_indexBuffers.getSize(); i++)
	{
		m_indexBuffers.pushBack(other.m_indexBuffers[i]);
	}
	
	for(int i = 0; i < other.m_userChannels.getSize(); i++)
	{
		m_userChannels.pushBack(other.m_userChannels[i]);	
	}
	

	m_material = other.m_material;
	m_vertexBuffer = other.m_vertexBuffer;
}

/// Returns the total number of triangles in all index buffers
hkUint32 hkxMeshSection::getNumTriangles () const
{
	hkUint32 nTriangles = 0;
	
	for (int n=0; n<m_indexBuffers.getSize(); n++)
	{
		hkxIndexBuffer* ibuffer = m_indexBuffers[n];

		nTriangles += ibuffer->getNumTriangles();
	}
	
	return nTriangles;
}

/// Explore the index buffers for the indices of the triIndex'th triangle
void hkxMeshSection::getTriangleIndices (hkUint32 triIndex, hkUint32& indexAOut, hkUint32& indexBOut, hkUint32& indexCOut) const
{
	hkUint32 nTriangles = 0;	
	for (int n=0; n<m_indexBuffers.getSize(); n++)
	{
		hkxIndexBuffer* ibuffer = m_indexBuffers[n];
		hkUint32 nBuffer = ibuffer->getNumTriangles();
	
		if ( triIndex < nTriangles + nBuffer )
		{
			ibuffer->getTriangleIndices( triIndex-nTriangles, indexAOut, indexBOut, indexCOut );
			return;
		}
	
		nTriangles += nBuffer;
	}
}

void hkxMeshSection::collectVertexPositions (hkArray<hkVector4>& verticesInOut) const
{
	hkxVertexBuffer* vertices = m_vertexBuffer;
	if (vertices )
	{
		const hkxVertexDescription& vDesc = vertices->getVertexDesc();
		const hkxVertexDescription::ElementDecl* vdecl = vDesc.getElementDecl(hkxVertexDescription::HKX_DU_POSITION, 0);
		int numVerts = vertices->getNumVertices(); 	
		if (vdecl && (numVerts > 0))
		{
			void* data = vertices->getVertexDataPtr(*vdecl);
			hkUint32 posStride = vdecl->m_byteStride;

			int bufOffset = verticesInOut.getSize();
			verticesInOut.setSize( bufOffset + numVerts );
			for (int vi=0; vi < numVerts; ++vi)
			{
				verticesInOut[vi + bufOffset].load<4,HK_IO_NATIVE_ALIGNED>((const hkFloat32*)data);
				data = hkAddByteOffset(data, posStride);
			}
		}
	}
}

void hkxMeshSection::appendGeometry (struct hkGeometry& geometryInOut, int materialIndex)
{
	collectVertexPositions(geometryInOut.m_vertices);

	// Now, check the index buffer
	for (int ib=0; ib < m_indexBuffers.getSize(); ++ib)
	{
		hkxIndexBuffer* ibuffer = m_indexBuffers[ib];

		const int numIndices = ibuffer->m_indices16.getSize() | ibuffer->m_indices32.getSize();
		bool smallIndices = ibuffer->m_indices16.getSize() > 0;
		int index = 0;

		while (index < numIndices)
		{
			hkGeometry::Triangle newTriangle;

			newTriangle.m_material = materialIndex;

			switch (ibuffer->m_indexType)
			{
			case hkxIndexBuffer::INDEX_TYPE_TRI_LIST:
				{

					newTriangle.m_a = smallIndices ? ibuffer->m_indices16[index] : ibuffer->m_indices32[index];
					newTriangle.m_b = smallIndices ? ibuffer->m_indices16[index+1] : ibuffer->m_indices32[index+1];
					newTriangle.m_c = smallIndices ? ibuffer->m_indices16[index+2] : ibuffer->m_indices32[index+2];

					index += 3;
					break;
				}
			case hkxIndexBuffer::INDEX_TYPE_TRI_STRIP:
				{
					if (index<2)
					{
						index++;
						continue;
					}

					if (index==2)
					{
						newTriangle.m_a = smallIndices ? ibuffer->m_indices16[0] : ibuffer->m_indices32[0];
						newTriangle.m_b = smallIndices ? ibuffer->m_indices16[1] : ibuffer->m_indices32[1];
						newTriangle.m_c = smallIndices ? ibuffer->m_indices16[2] : ibuffer->m_indices32[2];						

						index ++;
						break;
					}

					const hkGeometry::Triangle &previousTriangle = geometryInOut.m_triangles[geometryInOut.m_triangles.getSize()-1];

					newTriangle.m_a = previousTriangle.m_c;
					newTriangle.m_b = previousTriangle.m_b;
					newTriangle.m_c = smallIndices? ibuffer->m_indices16[index] : ibuffer->m_indices32[index];

					index ++;
					break;
				}
			default:
				{
					HK_WARN_ALWAYS(0xabbaa883, "Unsupported index buffer type - Ignoring");
					index = numIndices;
					continue;
				}
			}

			geometryInOut.m_triangles.pushBack(newTriangle);

		}
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
