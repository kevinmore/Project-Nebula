/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/GeometryUtilities/hkGeometryUtilities.h>
#include <Common/GeometryUtilities/Mesh/hkMeshSystem.h>
#include <Common/Base/Reflection/hkClass.h>
#include <Common/Base/Reflection/hkTypeInfo.h>
#include <Common/Base/Container/LocalArray/hkLocalBuffer.h>
#include <Common/GeometryUtilities/Mesh/hkMeshVertexBuffer.h>

HK_REFLECTION_DEFINE_STUB_VIRTUAL(hkMeshSystem, hkReferencedObject);

//
//	Create a vertex buffer. The vertex format is retrieved from the given sourceVertexBuffer

hkMeshVertexBuffer* hkMeshSystem::createVertexBuffer(const hkMeshVertexBuffer* templateVertexBuffer, int numVertices)
{
	hkMeshVertexBuffer* meshVb = const_cast<hkMeshVertexBuffer*>(templateVertexBuffer);
	hkVertexFormat fmt;
	meshVb->getVertexFormat(fmt);
	return createVertexBuffer(fmt, numVertices);
}

//
//	Creates a compound shape. The default implementation calls the hkMatrix4 based createCompoundShape().

hkMeshShape* hkMeshSystem::createCompoundShape(const hkMeshShape*const* shapes, const hkQTransform* transforms, int numShapes)
{
	hkLocalBuffer<hkMatrix4> temp(numShapes);
	for (int k = 0; k < numShapes; k++)
	{
		const hkQTransform& qtm = transforms[k];
		
		hkTransform tempTm;
		tempTm.setRotation(qtm.m_rotation);
		tempTm.setTranslation(qtm.m_translation);

		temp[k].set(tempTm);
	}

	return createCompoundShape(shapes, temp.begin(), numShapes);
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
