/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/PhysicsSystem/hknpPhysicsToSceneDataBridge.h>

#include <Common/SceneData/hkSceneData.h>
#include <Common/SceneData/VisualDebugger/hkxSceneDataContext.h>
#include <Common/SceneData/Scene/hkxScene.h>
#include <Common/SceneData/Scene/hkxSceneUtils.h>
#include <Common/Serialize/Util/hkRootLevelContainer.h>
#include <Common/Visualize/hkDebugDisplay.h>
#include <Common/Visualize/hkVisualDebugger.h>

#include <Physics/Physics/Extensions/Viewers/Shape/hknpShapeViewer.h>
#include <Physics/Physics/Extensions/PhysicsSystem/hknpPhysicsSceneData.h>

hknpPhysicsToSceneDataBridge::hknpPhysicsToSceneDataBridge( hknpWorld* physicsWorld, hkxSceneDataContext* sceneDataContext, hkVisualDebugger* debugger )
:	m_sceneDataContext(sceneDataContext),
	m_physicsWorld(physicsWorld),
	m_debugger(debugger)
{
}

hknpPhysicsToSceneDataBridge::~hknpPhysicsToSceneDataBridge()
{
	for( int containersIt = 0; containersIt < m_loadedContainers.getSize(); containersIt++ )
	{
		removeRootLevelContainer( m_loadedContainers[containersIt].m_name );
	}

	for( hkPointerMap<hkUlong, hkMatrix4*>::Iterator iter = m_meshIdToScaleAndSkewTransformMap.getIterator(); m_meshIdToScaleAndSkewTransformMap.isValid(iter); iter = m_meshIdToScaleAndSkewTransformMap.getNext(iter))
	{
		delete m_meshIdToScaleAndSkewTransformMap.getValue(iter);
	}

	for( hkPointerMap<hknpBodyId, hkTransform*>::Iterator iter = m_rigidBodyToTransformMap.getIterator(); m_rigidBodyToMeshIdMap.isValid(iter); iter = m_rigidBodyToTransformMap.getNext(iter) )
	{
		delete m_rigidBodyToTransformMap.getValue(iter);
	}
}

void hknpPhysicsToSceneDataBridge::setSceneDataContext( hkxSceneDataContext* sceneDataContext )
{
	if ( sceneDataContext != HK_NULL )
	{
		m_sceneDataContext = sceneDataContext;
	}
}

void hknpPhysicsToSceneDataBridge::addRootLevelContainer( const char* name, const hkRootLevelContainer* rootLevelContainer, bool addToWorld )
{
	// Our entry for this container
	RootLevelContainer& container = m_loadedContainers.expandOne();
	container.m_container = *rootLevelContainer;
	container.m_name = name;

	// Get the meshes and physics world objects from the root level container
	if(rootLevelContainer != HK_NULL )
	{
		// Add the scene to the world
		hkxScene* scene = reinterpret_cast<hkxScene*>( rootLevelContainer->findObjectByType( hkxSceneClass.getName() ) );
		if( scene != HK_NULL )
		{
			m_sceneDataContext->addScene( scene );
		}

		// Find all mesh nodes
		hkArray< hkRefPtr<hkxNode> > meshNodes;
		hkArray< hkMatrix4 > meshNodeWorldFromLocalTransforms;

		hkMatrix4 worldTransform; worldTransform.setIdentity();
		if ( scene != HK_NULL )
		{
			hkxSceneUtils::findAllMeshNodes( scene, scene->m_rootNode, meshNodes, &worldTransform, &meshNodeWorldFromLocalTransforms );
		}

		// Add our rigid bodies to the world, converting old physics if available/necessary
		{
			hknpPhysicsSceneData* physicsData = HK_NULL;

			while( 1 )
			{
				physicsData = static_cast<hknpPhysicsSceneData*>( rootLevelContainer->findObjectByType( hknpPhysicsSceneDataClass.getName(), physicsData ) );

				if( physicsData == HK_NULL )
				{
					break;
				}

				// Get the physics objects from the physics systems, after getting the physics systems from the physics data
				for ( int dataIt = 0; dataIt < physicsData->m_systemDatas.getSize(); ++dataIt )
				{
					const hknpPhysicsSystemData* physicsSystemData = physicsData->m_systemDatas[dataIt];

					if ( addToWorld )
					{
						container.m_systems.pushBack(
							new hknpPhysicsSystem(
								physicsSystemData,
								m_physicsWorld,
								hkTransform::getIdentity(),
								hknpWorld::ADD_BODY_NOW ) );
					}

					// Build our name to runtime id map
					for ( int rbIt = 0; rbIt < physicsSystemData->m_bodyCinfos.getSize(); rbIt++ )
					{
						const char* bodyName = physicsSystemData->m_bodyCinfos[rbIt].m_name;
						hknpBodyId bodyId = m_physicsWorld->m_bodyManager.findBodyByName( bodyName );
						if ( bodyId.isValid() )
						{
							// We use the body name stored in the body manager so it is still valid if our system data gets cleaned up.
							const char* storedBodyName = m_physicsWorld->m_bodyManager.getBodyName( bodyId );
							m_rigidBodyNameToIdMap.insert( storedBodyName, bodyId.value() );
						}
					}
				}
			}
		}

		// Iterate over all meshes and physics world objects, finding matching pairs (by name) and adding their corresponding ID's
		for( int i = 0; i < meshNodes.getSize(); ++i )
		{
			hknpBodyId rigidBody( m_rigidBodyNameToIdMap.getWithDefault( meshNodes[i]->m_name, hknpBodyId::INVALID ) );

			if ( rigidBody.isValid() )
			{
				hkUlong meshId = reinterpret_cast<hkUlong>(meshNodes[i].val());

				m_rigidBodyToMeshIdMap.insert( rigidBody.value(), meshId );
				m_rigidBodyToTransformMap.insert( rigidBody.value(), new hkTransform( m_physicsWorld->getBodyTransform( rigidBody ) ) );

				// The mesh associated with the rigid body may have its own scaling term we need to make sure that when
				// we update the mesh from the rigid body transforms its scale is preserved. Hence we store the
				// scale and skew term of the mesh.
				{
					::hkMatrixDecomposition::Decomposition decomposition;
					::hkMatrixDecomposition::decomposeMatrix(meshNodeWorldFromLocalTransforms[i], decomposition);

					hkMatrix4* scaleAndSkew = new hkMatrix4();
					*scaleAndSkew = decomposition.m_scaleAndSkew;

					m_meshIdToScaleAndSkewTransformMap.insert( meshId, scaleAndSkew );
				}
			}
		}
	}
}

void hknpPhysicsToSceneDataBridge::removeRootLevelContainer( const char* name )
{
	// Find the loadedContainer
	RootLevelContainer* loadedContainer = HK_NULL;
	int containerIdx = -1;
	const int numberOfEntries = m_loadedContainers.getSize();
	for ( int i = 0; i < numberOfEntries; ++i )
	{
		if ( hkString::strCmp( m_loadedContainers[i].m_name, name ) == 0 )
		{
			loadedContainer = &m_loadedContainers[i];
			containerIdx = i;
			break;
		}
	}

	// Ensure that this loadedContainer has been loaded
	if ( loadedContainer != HK_NULL )
	{
		// Remove our rigid bodies from the world
		{
			hknpPhysicsSceneData* physicsData = HK_NULL;

			while( 1 )
			{
				physicsData = static_cast<hknpPhysicsSceneData*>( loadedContainer->m_container.findObjectByType( hknpPhysicsSceneDataClass.getName(), physicsData ) );

				if( physicsData == HK_NULL )
				{
					break;
				}

				// Get the physics objects from the physics systems, after getting the physics systems from the physics data
				for ( int dataIt = 0; dataIt < physicsData->m_systemDatas.getSize(); ++dataIt )
				{
					const hknpPhysicsSystemData* physicsSystemData = physicsData->m_systemDatas[dataIt];
					for ( int rbIt = 0; rbIt < physicsSystemData->m_bodyCinfos.getSize(); rbIt++ )
					{
						const hknpBodyCinfo& cinfo = physicsSystemData->m_bodyCinfos[rbIt];
						hknpBodyId::Type bodyId = m_rigidBodyNameToIdMap.getWithDefault( cinfo.m_name, hknpBodyId::INVALID );
						if ( bodyId != hknpBodyId::INVALID )
						{
							m_rigidBodyNameToIdMap.remove( cinfo.m_name );

							hkUlong meshId = m_rigidBodyToMeshIdMap.getWithDefault( bodyId, (hkUlong)-1 );

							m_rigidBodyToMeshIdMap.remove( bodyId );

							hkTransform* transform = m_rigidBodyToTransformMap.getWithDefault( bodyId, HK_NULL );
							delete transform;
							m_rigidBodyToTransformMap.remove( bodyId );

							if ( meshId != (hkUlong)-1 )
							{
								hkMatrix4* scaleAndSkew = m_meshIdToScaleAndSkewTransformMap.getWithDefault( meshId, HK_NULL );
								delete scaleAndSkew;
								m_meshIdToScaleAndSkewTransformMap.remove( meshId );
							}
						}
					}
				}
			}
		}

		// Remove the scene from the context
		hkxScene* scene = reinterpret_cast<hkxScene*>( loadedContainer->m_container.findObjectByType( hkxSceneClass.getName() ) );
		if( scene != HK_NULL )
		{
			m_sceneDataContext->removeScene( scene );
		}

		// Now cleanup our loadedContainer
		{
			hkReferencedObject::removeReferences<hknpPhysicsSystem>( loadedContainer->m_systems.begin(), loadedContainer->m_systems.getSize() );
			m_loadedContainers.removeAt( containerIdx );
		}
	}
}

void hknpPhysicsToSceneDataBridge::getLoadedScenes( hkArray<hkxScene*>& scenes )
{
	for ( int i = 0; i < m_loadedContainers.getSize(); ++i )
	{
		hkxScene* scene = reinterpret_cast<hkxScene*>(m_loadedContainers[i].m_container.findObjectByType( hkxSceneClass.getName() ) );
		if ( scene != HK_NULL )
		{
			scenes.pushBack( scene );
		}
	}
}

void hknpPhysicsToSceneDataBridge::resetPhysicsTransforms() const
{
	hkPointerMap<hknpBodyId, hkUlong>::Iterator iter = m_rigidBodyToMeshIdMap.getIterator();
	for( ; m_rigidBodyToMeshIdMap.isValid( iter ); iter = m_rigidBodyToMeshIdMap.getNext( iter ) )
	{
		hknpBodyId rigidBody( m_rigidBodyToMeshIdMap.getKey(iter) );
		hkTransform* transform = m_rigidBodyToTransformMap.getWithDefault( rigidBody.value(), HK_NULL );

		if( rigidBody.isValid() && transform != HK_NULL )
		{
			m_physicsWorld->setBodyTransform( rigidBody, *transform );

			const hknpBody& body = m_physicsWorld->getBody( rigidBody );
			if ( !body.isStatic() )
			{
				m_physicsWorld->setBodyLinearVelocity( rigidBody, hkVector4::getZero() );
				m_physicsWorld->setBodyAngularVelocity( rigidBody, hkVector4::getZero() );
			}

			// refresh display
			hknpShapeViewer* shapeViewer = getShapeViewer();
			if ( shapeViewer != HK_NULL )
			{
				shapeViewer->refreshBody( m_physicsWorld, rigidBody );
			}
		}
	}
}

void hknpPhysicsToSceneDataBridge::syncSceneDataToPhysics()
{
	if(m_physicsWorld != HK_NULL)
	{
		// Iterate over the physics objects in the array, updating their corresponding mesh
		hkPointerMap<hknpBodyId, hkUlong>::Iterator iter = m_rigidBodyToMeshIdMap.getIterator();
		for( ; m_rigidBodyToMeshIdMap.isValid( iter ); iter = m_rigidBodyToMeshIdMap.getNext( iter ) )
		{
			// Attempt to cast the array object as a rigid body.
			hknpBodyId rigidBody( m_rigidBodyToMeshIdMap.getKey(iter) );
			if( rigidBody.isValid() )
			{
				
				const hkTransform& transform = m_physicsWorld->getBodyTransform( rigidBody );
				hkMatrix4 matrix;
				matrix.set(transform);

				hkUlong meshId = m_rigidBodyToMeshIdMap.getWithDefault(rigidBody.value(), (hkUlong)-1);

				// When syncing we need to preserve the scale term for the meshes so that it can be rendered properly.
				if ( meshId != (hkUlong)-1 )
				{
					hkMatrix4* scaleAndSkewTransform = m_meshIdToScaleAndSkewTransformMap.getWithDefault(meshId, HK_NULL);
					matrix.mul( *scaleAndSkewTransform );
				}

				hkDebugDisplay::getInstance().updateGeometry( matrix, m_rigidBodyToMeshIdMap.getValue(iter), 0 );
			}
		}
	}
}

hknpPhysicsToSceneDataBridge::RootLevelContainer* hknpPhysicsToSceneDataBridge::getLoadedContainerInfo( const char* name )
{
	const int numberOfEntries = m_loadedContainers.getSize();
	for ( int i = 0; i < numberOfEntries; ++i )
	{
		if ( hkString::strCmp( m_loadedContainers[i].m_name, name ) == 0 )
		{
			return &m_loadedContainers[i];
		}
	}

	return HK_NULL;
}

void hknpPhysicsToSceneDataBridge::setPhysicsWorld( hknpWorld* physicsWorld )
{
	m_physicsWorld = physicsWorld;
}

hknpShapeViewer* hknpPhysicsToSceneDataBridge::getShapeViewer() const
{
	if ( m_debugger != HK_NULL )
	{
		return static_cast<hknpShapeViewer*>( m_debugger->getCurrentProcessByName( hknpShapeViewer::getName() ) );
	}

	return HK_NULL;
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
