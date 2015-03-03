/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/SceneData/hkSceneData.h>

#include <Common/GeometryUtilities/Mesh/Converters/SceneDataToMesh/hkSceneDataToMeshConverter.h>
#include <Common/GeometryUtilities/Mesh/Memory/hkMemoryMeshShape.h>
#include <Common/GeometryUtilities/Mesh/Memory/hkMemoryMeshSystem.h>
#include <Common/GeometryUtilities/Mesh/Skin/hkSkinnedMeshShapeBuilder.h>
#include <Common/GeometryUtilities/Mesh/Utils/MeshSectionMergeUtil/hkMeshSectionMergeUtil.h>
#include <Common/SceneData/Scene/hkxScene.h>
#include <Common/SceneData/Scene/hkxSceneUtils.h>

// this
#include <Common/SceneData/Mesh/MemoryMeshFactory/hkxMemoryMeshFactory.h>

hkxMemoryMeshFactory::hkxMemoryMeshFactory()
{
	m_meshSystem = new hkMemoryMeshSystem;
}

hkxMemoryMeshFactory::~hkxMemoryMeshFactory()
{
	// 	m_meshSystem->removeReference();
}

hkMeshSystem* hkxMemoryMeshFactory::getMeshSystem()
{
	return m_meshSystem;
}

void hkxMemoryMeshFactory::extractShapes( hkRootLevelContainer* rootLevelContainer, hkStringMap<int>& extraGraphicsNodes, hkStringMap<hkMeshShape*>& shapesOut )
{
	extractShapes( rootLevelContainer, extraGraphicsNodes, m_meshSystem, shapesOut );
}

/// Use to accumulate all rigid body nodes and physics systems from the list of \a rootLevelContainers.
/// This function depends on the serialization type, so it is likely that this function has to be reimplemented for each game engines
/// This implementation was moved here from the hkdAssetProcessingUtil
void hkxMemoryMeshFactory::extractShapes( hkRootLevelContainer* rootLevelContainer, hkStringMap<int>& extraGraphicsNodes, hkMemoryMeshSystem *meshSystem, hkStringMap<hkMeshShape*>& shapesOut )
{
	//
	//	Find all the physics nodes with the original hkxNode data
	//

	hkxScene*			scene = reinterpret_cast<hkxScene*>( rootLevelContainer->findObjectByType( hkxSceneClass.getName() ));
	hkArray<hkxSceneUtils::GraphicsNode> nodes;
	hkxSceneUtils::findAllGraphicsNodes(false, false, extraGraphicsNodes, scene->m_rootNode, nodes);

	// Add all the newly found rigid bodies
	for (int n = 0; n < nodes.getSize(); n++)
	{
		const hkxSceneUtils::GraphicsNode* node = &nodes[n];

		if ( shapesOut.getWithDefault( node->m_name, HK_NULL ) )
		{
			HK_WARN_ALWAYS( 0xabba43de, "Object: '" << node->m_name <<"' exists twice, deleting second version" );
			continue;
		}

		hkMeshShape* meshShape =  hkSceneDataToMeshConverter::convert( meshSystem, HK_NULL, scene, node->m_node, hkSceneDataToMeshConverter::SPACE_ONLY_USING_SCALE_SKEW );
		meshShape->setName( node->m_name );

		if (!hkMeshSectionMergeUtil::hasUniqueMaterials(meshShape))
		{
			// We may want to merge materials -> to improve performance etc
			// <<js.todo Really this merging should be controlled by parameters available in the editor

			const bool mergeMaterialSections = false;

			if (mergeMaterialSections)
			{
				hkMeshShape* mergedShape = hkMeshSectionMergeUtil::mergeShapeSectionsByMaterial(meshSystem, meshShape);
				if (mergedShape)
				{
					meshShape->removeReference();
					meshShape = mergedShape;
				}
				else
				{
					HK_WARN(0x8d7292bb, "Unable to merge sections with the same material");
				}

			}
			else
			{
				HK_WARN(0x24234a21, "Mesh shape '" << node->m_name << "' has multiple sections with the same material: impacts graphics performance");
			}
		}

		if ( meshShape != HK_NULL)
		{
			shapesOut.insert( node->m_name, meshShape);
		}
	}
}

// Nothing to do here
void hkxMemoryMeshFactory::storeShapes( hkRootLevelContainer* rootLevelContainer, hkArray<const hkMeshShape*>& shapes )
{
	// Add a reference to all shapes, to have the same behavior as the Vision mesh factory
	hkReferencedObject::addReferences(shapes.begin(), shapes.getSize());
}

const hkMatrix3& hkxMemoryMeshFactory::getTransform() const
{
	return hkMatrix3::getIdentity();
}

void hkxMemoryMeshFactory::setTransform( const hkMatrix3& transform )
{
}

//
//	Creates a skinned mesh shape from the given mesh shapes and transforms. A single bone will drive each of the provided shapes.
//	Currently, the implementation returns HK_NULL.

hkSkinnedMeshShape* hkxMemoryMeshFactory::createSkinnedMesh(const char* skinnedMeshName, const hkMeshShape** meshShapesIn, const hkQTransform* transformsIn, int numMeshes, int maxBonesPerSection)
{
	// If you implement this, make sure to implement and call removeShapes properly, so you don't end-up with both the skinned and original graphics
	// inside the asset file.

	// Create skinned mesh shape
	hkStorageSkinnedMeshShape* skinnedMeshShape = new hkStorageSkinnedMeshShape;

	// Set name
	skinnedMeshShape->setName( skinnedMeshName );

	// Build the skinned mesh shape from the given meshes
	hkSkinnedMeshBuilder builder( skinnedMeshShape, m_meshSystem, maxBonesPerSection );
	for ( int i = 0; i < numMeshes; i++ )
	{
		builder.addMesh( meshShapesIn[i], transformsIn[i], 1 );
	}
	builder.build();

	return skinnedMeshShape;
}

//
//	Removes all the given shapes from the root level container

void hkxMemoryMeshFactory::removeShapes(hkRootLevelContainer* rootLevelContainer, hkArray<const hkMeshShape*>& shapes)
{}

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
