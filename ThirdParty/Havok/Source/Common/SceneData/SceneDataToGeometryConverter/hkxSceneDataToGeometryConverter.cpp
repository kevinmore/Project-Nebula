/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/SceneData/hkSceneData.h>
#include <Common/SceneData/SceneDataToGeometryConverter/hkxSceneDataToGeometryConverter.h>
#include <Common/Base/Types/Geometry/hkGeometry.h>
#include <Common/SceneData/Scene/hkxScene.h>
#include <Common/SceneData/Scene/hkxSceneUtils.h>

void HK_CALL hkxSceneDataToGeometryConverter::convertToSingleGeometryRecursive( const hkxNode* node, const hkMatrix4& transform, hkGeometry& geometryInOut, hkArray<hkxMaterial*>& materialsInOut )
{
	HK_ASSERT(0x5f269290, node);

	// Convert this node if it's a mesh
	hkxMesh* mesh = hkxSceneUtils::getMeshFromNode(node);
	if (mesh)
	{
		// Build the geometry in local space
		int numVerticesBefore = geometryInOut.m_vertices.getSize();
		mesh->appendGeometry(geometryInOut, &materialsInOut);
		int numVerticesAfter = geometryInOut.m_vertices.getSize();

		// Transform it to world space
		for (int i = numVerticesBefore; i < numVerticesAfter; i++)
		{
			transform.transformPosition(geometryInOut.m_vertices[i], geometryInOut.m_vertices[i]);
		}
	}

	// Convert and merge all the child nodes
	for (int i = 0; i < node->m_children.getSize(); i++)
	{
		// Get the child node's local-to-world transform
		hkMatrix4 childTransform;
		childTransform.setMul(transform, node->m_children[i]->m_keyFrames[0]);

		convertToSingleGeometryRecursive(node->m_children[i], childTransform, geometryInOut, materialsInOut);
	}
}

void HK_CALL hkxSceneDataToGeometryConverter::convertToSingleGeometry( const hkxScene* scene, const hkxNode* root, hkGeometry& geometryInOut, hkArray<hkxMaterial*>& materialsInOut )
{
	hkMatrix4 transform;
	HK_ON_DEBUG( hkResult result = ) scene->getWorldFromNodeTransform(root, transform);
	HK_ASSERT(0x3d9edf5f, result == HK_SUCCESS);

	convertToSingleGeometryRecursive(root, transform, geometryInOut, materialsInOut);
}

void HK_CALL hkxSceneDataToGeometryConverter::convertToSingleGeometry( const hkxScene* scene, const hkArray< hkRefPtr<hkxNode> >& roots, hkGeometry& geometryInOut, hkArray<hkxMaterial*>& materialsInOut )
{
	for (int i = 0; i < roots.getSize(); i++)
	{
		convertToSingleGeometry(scene, roots[i], geometryInOut, materialsInOut);
	}
}

void HK_CALL hkxSceneDataToGeometryConverter::convertToGeometryInstancesRecursive( const hkxNode* node, const hkMatrix4& transform, GeometryInstances& geometriesOut, hkArray<hkxMaterial*>& materialsInOut, MeshToGeometryIndexMap& meshToGeometryIndexMap )
{
	HK_ASSERT(0x1781e10c, node);

	// Convert this node if it's a mesh
	hkxMesh* mesh = hkxSceneUtils::getMeshFromNode(node);
	if (mesh)
	{
		// If the mesh has been encountered before then fetch the corresponding geometry index from the map.
		// Otherwise, build a new geometry (in local space) and insert its index into the map.
		int geometryIndex = 0;
		if(meshToGeometryIndexMap.get(mesh, &geometryIndex) == HK_FAILURE)
		{
			geometryIndex = geometriesOut.m_geometries.getSize();

			hkGeometry& geometry = geometriesOut.m_geometries.expandOne();
			mesh->appendGeometry(geometry, &materialsInOut);

			meshToGeometryIndexMap.insert(mesh, geometryIndex);
		}

		// Add a new instance
		GeometryInstances::Instance& instance = geometriesOut.m_instances.expandOne();
		instance.m_geometryIdx = geometryIndex;
		instance.m_worldFromLocal = transform;
	}

	// Convert all the child nodes
	for (int i = 0; i < node->m_children.getSize(); i++)
	{
		// Get the child node's local-to-world transform
		hkMatrix4 childTransform;
		childTransform.setMul(transform, node->m_children[i]->m_keyFrames[0]);

		convertToGeometryInstancesRecursive(node->m_children[i], childTransform, geometriesOut, materialsInOut, meshToGeometryIndexMap);
	}
}

void HK_CALL hkxSceneDataToGeometryConverter::convertToGeometryInstances( const hkxScene* scene, const hkxNode* root, GeometryInstances& geometriesOut, hkArray<hkxMaterial*>& materialsInOut )
{
	hkMatrix4 transform;
	HK_ON_DEBUG( hkResult result = ) scene->getWorldFromNodeTransform(root, transform);
	HK_ASSERT(0x3d9edf5f, result == HK_SUCCESS);

	// Map used to detect instanced meshes
	MeshToGeometryIndexMap meshToGeometryIndexMap;

	convertToGeometryInstancesRecursive(root, transform, geometriesOut, materialsInOut, meshToGeometryIndexMap);
}

void HK_CALL hkxSceneDataToGeometryConverter::mergeInstances( const GeometryInstances& geometriesIn, hkGeometry& geometryInOut )
{
	for (int i = 0; i < geometriesIn.m_instances.getSize(); i++)
	{
		const hkGeometry& geometry = geometriesIn.m_geometries[ geometriesIn.m_instances[i].m_geometryIdx ];
		HK_ASSERT(0x520fc822, &geometry != &geometryInOut);

		const hkMatrix4& transform = geometriesIn.m_instances[i].m_worldFromLocal;

		int offset = geometryInOut.m_vertices.getSize();

		// Expand geometryInOut by the required number of vertices and copy them over while transforming them to world space
		int numVertices = geometry.m_vertices.getSize();
		hkVector4* vertices = geometryInOut.m_vertices.expandBy(numVertices);
		for (int v = 0; v < numVertices; v++)
		{
			transform.transformPosition(geometry.m_vertices[v], vertices[v]);
		}

		// Expand geometryInOut by the required number of triangles and copy them over while
		// offsetting the original vertex indices to point to the vertices copied above
		int numTriangles = geometry.m_triangles.getSize();
		hkGeometry::Triangle* triangles = geometryInOut.m_triangles.expandBy(numTriangles);
		for (int t = 0; t < numTriangles; t++)
		{
			triangles[t].set( geometry.m_triangles[t].m_a+offset,
							  geometry.m_triangles[t].m_b+offset,
							  geometry.m_triangles[t].m_c+offset,
							  geometry.m_triangles[t].m_material );
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
