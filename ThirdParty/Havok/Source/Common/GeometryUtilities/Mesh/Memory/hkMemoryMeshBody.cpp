/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>
#include <Common/GeometryUtilities/Mesh/Memory/hkMemoryMeshBody.h>
#include <Common/Base/Reflection/hkClass.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Common/GeometryUtilities/Mesh/hkMeshShape.h>
#include <Common/GeometryUtilities/Mesh/hkMeshVertexBuffer.h>

hkMemoryMeshBody::hkMemoryMeshBody(hkMeshSystem* meshSystem, const hkMeshShape* shape, const hkMatrix4& transform, hkIndexedTransformSetCinfo* transformSet)
{
	m_name = "";

    if (transformSet)
    {
        m_transformSet = new hkIndexedTransformSet(*transformSet);
        m_transformSet->removeReference();
    }

	if (shape != HK_NULL)
	{
		const int numSections = shape->getNumSections();
		hkLocalArray<hkMeshVertexBuffer*> buffers(numSections);
		hkLocalArray<hkMeshVertexBuffer*> uniqueBuffers(numSections);
		hkLocalArray<hkMeshVertexBuffer*> dynamicBuffers(numSections);

		{
			hkMeshSection section;
			for (int i = 0; i < numSections; i++)
			{
				shape->lockSection(i, hkMeshShape::ACCESS_VERTEX_BUFFER, section);
				section.m_vertexBuffer->addReference();
				buffers.pushBack(section.m_vertexBuffer);
				shape->unlockSection(section);
			}
		}
					
		{
			for (int i = 0; i < numSections; i++)
			{
				hkMeshVertexBuffer* buffer = buffers[i];
				if (uniqueBuffers.indexOf(buffer) < 0)
				{
					uniqueBuffers.pushBack(buffer);
				}
			}
		}

		const int numBuffers = uniqueBuffers.getSize();
		dynamicBuffers.setSize(numBuffers);

		{
			hkVertexFormat dstFormat;
			for (int i = 0; i < numBuffers; i++)
			{
				hkMeshVertexBuffer* srcBuffer = uniqueBuffers[i];
				dynamicBuffers[i] = srcBuffer->clone();
			}
		}

		// Set up each of the buffers
		m_vertexBuffers.setSize(numSections);
		for (int i = 0; i < numSections; i++)
		{
			int index = uniqueBuffers.indexOf(buffers[i]);
			HK_ASSERT(0x7bba546c, index >= 0);

			hkMeshVertexBuffer* vertexBuffer = dynamicBuffers[index];
			vertexBuffer->addReference();

			m_vertexBuffers[i] = vertexBuffer;
		}

		hkReferencedObject::removeReferences(buffers.begin(), buffers.getSize());
		hkReferencedObject::removeReferences(dynamicBuffers.begin(), dynamicBuffers.getSize());
	}

    m_transform = transform;
    m_shape = shape;
}

hkMemoryMeshBody::hkMemoryMeshBody( hkFinishLoadedObjectFlag flag )
: hkMeshBody(flag)
, m_transformSet(flag)
, m_shape(flag)
, m_vertexBuffers(flag)
, m_name(flag)
{
}

hkMemoryMeshBody::~hkMemoryMeshBody()
{
    for (int i = 0; i < m_vertexBuffers.getSize(); i++)
    {
        hkMeshVertexBuffer* buffer = m_vertexBuffers[i];
        buffer->removeReference();
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
