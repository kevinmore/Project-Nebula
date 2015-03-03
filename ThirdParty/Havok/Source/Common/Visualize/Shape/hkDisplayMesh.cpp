/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Visualize/hkVisualize.h>
#include <Common/Visualize/Shape/hkDisplayMesh.h>
#include <Common/Base/Reflection/hkClass.h>
#include <Common/Base/Reflection/Registry/hkVtableClassRegistry.h>
#include <Common/GeometryUtilities/Mesh/hkMeshBody.h>
#include <Common/Serialize/Data/Native/hkDataObjectNative.h>
#include <Common/Serialize/Tagfile/Binary/hkBinaryTagfileWriter.h>
#include <Common/Serialize/Util/hkBuiltinTypeRegistry.h>

hkDisplayMesh::hkDisplayMesh(hkMeshBody* mesh)
:hkDisplayGeometry(HK_DISPLAY_MESH),
m_mesh(mesh)
{	
	hkMatrix4 matrix;
	mesh->getTransform(matrix);
	
	hkMatrixDecomposition::Decomposition decomp;
	hkMatrixDecomposition::decomposeMatrix(matrix, decomp);
	
	hkTransform transform; transform.set(decomp.m_rotation, decomp.m_translation);
	setTransform(transform);		
}

hkMeshBody* hkDisplayMesh::getMesh() const
{
	return m_mesh;
}

void hkDisplayMesh::buildGeometry()
{	
}

void hkDisplayMesh::getWireframeGeometry( hkArrayBase<hkVector4>& lines, hkMemoryAllocator& a )
{
}

void hkDisplayMesh::serializeMeshData()
{
	m_meshAsTagfile.clear();
	hkOstream buffer(m_meshAsTagfile);				
	
	hkVtableClassRegistry* registry = hkBuiltinTypeRegistry::getInstance().getVtableClassRegistry();
	const hkClass* meshClass = registry->getClassFromVirtualInstance(m_mesh);

	HK_ASSERT( 0x5b6c1414, meshClass != HK_NULL );				

	hkDataWorldNative world;
	world.setContents(m_mesh, *meshClass);
	hkBinaryTagfileWriter().save(world.getContents(), buffer.getStreamWriter(), HK_NULL);

	//hkOstream debugBuffer( "meshtest.txt" );
	//hkSerializeUtil::saveTagfile(mesh, *meshClass, debugBuffer.getStreamWriter(), false, HK_NULL);		
}

void hkDisplayMesh::freeSerializedMeshData()
{
	m_meshAsTagfile.clearAndDeallocate();
}

int hkDisplayMesh::getSeriaizedMeshDataSize() const
{
	return m_meshAsTagfile.getSize();
}

const char* hkDisplayMesh::getSeriaizedMeshData() const
{
	return m_meshAsTagfile.begin();
}

void hkForwardingDisplayGeometryBuilder::buildDisplayGeometries( const hkReferencedObject* source, hkArray<hkDisplayGeometry*>& displayGeometries )
{
	const hkClass* klass = hkBuiltinTypeRegistry::getInstance().getVtableClassRegistry()->getClassFromVirtualInstance(source);
	if( hkMeshBodyClass.isSuperClass(*klass) )
	{
		hkMeshBody* meshBody = const_cast<hkMeshBody*>(reinterpret_cast<const hkMeshBody*>(source));
		displayGeometries.pushBack(new hkDisplayMesh(meshBody));
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
