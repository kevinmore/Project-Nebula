/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/SceneData/hkSceneData.h>
#include <Common/SceneData/VisualDebugger/Viewer/hkxSceneViewer.h>
#include <Common/GeometryUtilities/Mesh/Converters/SceneDataToMesh/hkSceneDataToMeshConverter.h>
#include <Common/GeometryUtilities/Mesh/hkMeshBody.h>
#include <Common/GeometryUtilities/Mesh/Memory/hkMemoryMeshSystem.h>
#include <Common/SceneData/Scene/hkxScene.h>
#include <Common/SceneData/Scene/hkxSceneUtils.h>
#include <Common/Visualize/hkProcessFactory.h>
#include <Common/Visualize/Shape/hkDisplayMesh.h>

int hkxSceneViewer::m_tag = 0;

hkProcess* HK_CALL hkxSceneViewer::create(const hkArray<hkProcessContext*>& contexts)
{	
	hkxSceneDataContext* context = HK_NULL;

	for( hkInt32 i = 0; i < contexts.getSize(); ++i )
	{
		if( hkString::strCmp( contexts[i]->getType(), HK_SCENE_DATA_CONTEXT_TYPE_STRING) == 0 )
		{
			context = static_cast<hkxSceneDataContext*>(contexts[i]);						
		}
	}

	return (context != HK_NULL ? new hkxSceneViewer( context ) : HK_NULL );	
}

void HK_CALL hkxSceneViewer::registerViewer()
{
	m_tag = hkProcessFactory::getInstance().registerProcess( getName(), create );
}

hkxSceneViewer::hkxSceneViewer( hkxSceneDataContext* context )
:hkProcess(true),
m_context(context)
{
	HK_ASSERT2(0x5d5b8242, m_context != HK_NULL, "Cannot initialize viewer with null context." );
	
	m_context->addListener( this );

	m_displayGeometryBuilder = new hkForwardingDisplayGeometryBuilder();
}

hkxSceneViewer::~hkxSceneViewer()
{
	for( hkInt32 i = 0; i < m_context->getScenes().getSize(); ++i )
	{
		sceneRemovedCallback( m_context->getScenes()[i] );
	}

	// clear out the texture paths
	if ( m_displayHandler )
	{
		m_displayHandler->clearTextureSearchPaths();
	}

	m_context->removeListener( this );
	m_displayGeometryBuilder->removeReference();
}

void hkxSceneViewer::sceneAddedCallback( hkxScene* scene )
{
	// Initialize memory mesh system to hold onto converted data
	hkMemoryMeshSystem meshSystem;

	hkArray< hkRefPtr< hkxNode > > nodesWithMeshes;
	hkxSceneUtils::findAllMeshNodes( scene, scene->m_rootNode, nodesWithMeshes );

	// Only create dummy nodes if the root node is HK_NULL.  This can happen if the scene nodes were
	// pruned but skins and meshes were left in.
	if( scene->m_rootNode == HK_NULL )
	{
		for( hkInt32 i = 0; i < scene->m_meshes.getSize(); ++i )
		{	
			hkxNode* dummyNode = new hkxNode();
			{
				dummyNode->m_object.set(scene->m_meshes[i], &hkxMeshClass);
				dummyNode->m_keyFrames.pushBack(hkMatrix4::getIdentity());
			}

			nodesWithMeshes.pushBack(dummyNode);
		}
	}
	
	// Add all meshes to the debug display
	for(hkInt32 i = 0; i < nodesWithMeshes.getSize(); ++i)
	{		
		// Get the node
		hkxNode* node = nodesWithMeshes[i];

		// Get the mesh associated with the node
		hkxMesh* mesh = hkxSceneUtils::getMeshFromNode(nodesWithMeshes[i]);

		// Convert mesh to hkMeshShape
		hkMeshShape* genericMeshShape = hkSceneDataToMeshConverter::convert(&meshSystem, HK_NULL, hkMatrix4::getIdentity(), mesh, m_context->getAllowTextureMipmap());
				
		// Wrap in hkMeshBody
		hkMeshBody* genericMeshBody = meshSystem.createBody(genericMeshShape, nodesWithMeshes[i]->m_keyFrames[0], HK_NULL);
		genericMeshShape->removeReference();

		// Add to the debug display handler
		if( isLocalViewer() )
		{
			hkDisplayMesh geometry(genericMeshBody);
			m_displayHandler->addGeometry(&geometry, (hkUlong)(node), m_tag, 0);
		}
		else
		{
			m_displayHandler->addGeometryHash( genericMeshBody, m_displayGeometryBuilder, hkUlong(mesh), hkUlong(node), m_tag );
		}
		genericMeshBody->removeReference();
	}
}

void hkxSceneViewer::sceneRemovedCallback( hkxScene* scene )
{
	hkArray< hkRefPtr< hkxNode > > nodesWithMeshes;
	hkxSceneUtils::findAllMeshNodes( scene, scene->m_rootNode, nodesWithMeshes );

	// Only create dummy nodes if the root node is HK_NULL.  This can happen if the scene nodes were
	// pruned but skins and meshes were left in.
	if( scene->m_rootNode == HK_NULL )
	{
		for( hkInt32 i = 0; i < scene->m_meshes.getSize(); ++i )
		{	
			hkxNode* dummyNode = new hkxNode();
			{
				dummyNode->m_object.set(scene->m_meshes[i], &hkxMeshClass);
				dummyNode->m_keyFrames.pushBack(hkMatrix4::getIdentity());
			}

			nodesWithMeshes.pushBack(dummyNode);
		}
	}

	// Remove all meshes from the debug display
	for(hkInt32 i = 0; i < nodesWithMeshes.getSize(); ++i)
	{
		// Get the node
		hkxNode* node = nodesWithMeshes[i];

		// Remove from the debug display handler
		m_displayHandler->removeGeometry((hkUlong)(node), m_tag, 0);
	}
}

void hkxSceneViewer::init()
{
	if( m_context )
	{
		if ( m_displayHandler )
		{
			const hkArray<const char*>& searchPaths = m_context->getTextureSearchPaths();
			for ( int texturePathsIter = 0; texturePathsIter < searchPaths.getSize(); texturePathsIter++ )
			{
				m_displayHandler->addTextureSearchPath( searchPaths[texturePathsIter] );
			}
		}

		for( hkInt32 i = 0; i < m_context->getScenes().getSize(); ++i )
		{
			sceneAddedCallback( m_context->getScenes()[i] );
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
