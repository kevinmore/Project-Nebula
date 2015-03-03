/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/SceneData/hkSceneData.h>
#include <Common/SceneData/VisualDebugger/hkxSceneDataContext.h>
#include <Common/SceneData/Scene/hkxScene.h>
#include <Common/SceneData/Scene/hkxSceneUtils.h>
#include <Common/Serialize/Util/hkRootLevelContainer.h>
#include <Common/Visualize/hkDebugDisplay.h>

#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Dynamics/World/hkpPhysicsSystem.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Utilities/Serialize/hkpPhysicsData.h>
#include <Physics2012/Utilities/Serialize/hkpPhysicsToSceneDataBridge.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/Collide/hkpShapeDisplayViewer.h>

hkpPhysicsToSceneDataBridge::hkpPhysicsToSceneDataBridge( hkpWorld* physicsWorld, hkxSceneDataContext* sceneDataContext )
:	m_sceneDataContext(sceneDataContext),
	m_physicsWorld(physicsWorld)
{
}

hkpPhysicsToSceneDataBridge::~hkpPhysicsToSceneDataBridge()
{
	for( hkPointerMap<hkUlong, hkMatrix4*>::Iterator iter = m_meshIdToScaleAndSkewTransformMap.getIterator(); m_meshIdToScaleAndSkewTransformMap.isValid(iter); iter = m_meshIdToScaleAndSkewTransformMap.getNext(iter))
	{
		delete m_meshIdToScaleAndSkewTransformMap.getValue(iter);
	}

	for( hkPointerMap<hkpRigidBody*, hkTransform*>::Iterator iter = m_rigidBodyToTransformMap.getIterator(); m_rigidBodyToTransformMap.isValid(iter); iter = m_rigidBodyToTransformMap.getNext(iter) )
	{
		delete m_rigidBodyToTransformMap.getValue(iter);
	}
}

void hkpPhysicsToSceneDataBridge::setSceneDataContext( hkxSceneDataContext* sceneDataContext )
{
	if ( sceneDataContext != HK_NULL )
	{
		m_sceneDataContext = sceneDataContext;
	}
}

void hkpPhysicsToSceneDataBridge::addRootLevelContainer( const char* name, const hkRootLevelContainer* rootLevelContainer)
{
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

		// Find all rigid bodies
		hkArray< hkpRigidBody* > rigidBodies;
		findAllRigidBodies(rootLevelContainer, rigidBodies);

		// Iterate over all meshes and physics world objects, finding matching pairs (by name) and adding their corresponding ID's		
		for( int i = 0; i < meshNodes.getSize(); ++i )
		{			
			for( int j = 0; j < rigidBodies.getSize(); ++j )
			{
				hkpRigidBody* rigidBody = rigidBodies[j];

				if( hkString::strCmp( meshNodes[i]->m_name, rigidBody->getName()  ) == 0 )
				{
					hkUlong meshId = reinterpret_cast<hkUlong>(meshNodes[i].val());

					m_rigidBodyToMeshIdMap.insert( rigidBody, meshId );
					m_rigidBodyToTransformMap.insert(rigidBody, new hkTransform(rigidBody->getTransform()));
					
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

		// Add this root level container to the list of loaded containers
		RootLevelContainer container;
		container.m_container = *rootLevelContainer;
		container.m_name = name;
		m_loadedContainers.pushBack( container );
	}
}

void hkpPhysicsToSceneDataBridge::removeRootLevelContainer( const char* name )
{
	// Ensure that this container has been loaded
	if ( getLoadedContainer( name ) != HK_NULL )
	{
		// Get the container with the given name from the array of loaded container
		// and remove it
		hkRootLevelContainer rootLevelContainer;
		const int numberOfEntries = m_loadedContainers.getSize();
		for ( int i = 0; i < numberOfEntries; ++i )
		{
			if ( hkString::strCmp( m_loadedContainers[i].m_name, name ) == 0 )
			{
				rootLevelContainer = m_loadedContainers[i].m_container;
				
				// Remove this container from the array of loaded containers
				m_loadedContainers.removeAt( i );
				
				break;
			}
		}

		// Remove paired physics object and mesh IDs from parallel arrays
		hkArray<hkpRigidBody*> rigidBodies;
		findAllRigidBodies( &rootLevelContainer, rigidBodies );
		for (int i = 0; i < rigidBodies.getSize(); ++i )
		{
			hkUlong meshId = m_rigidBodyToMeshIdMap.getWithDefault( rigidBodies[i], (hkUlong)-1 );

			m_rigidBodyToMeshIdMap.remove( rigidBodies[i] );
			
			hkTransform* transform = m_rigidBodyToTransformMap.getWithDefault( rigidBodies[i], HK_NULL );
			delete transform;
			m_rigidBodyToTransformMap.remove(rigidBodies[i]);

			if ( meshId != (hkUlong)-1 )
			{
				hkMatrix4* scaleAndSkew = m_meshIdToScaleAndSkewTransformMap.getWithDefault( meshId, HK_NULL );
				delete scaleAndSkew;
				m_meshIdToScaleAndSkewTransformMap.remove( meshId );
			}
		}

		// Remove the scene from the context 
		hkxScene* scene = reinterpret_cast<hkxScene*>( rootLevelContainer.findObjectByType( hkxSceneClass.getName() ) );
		if( scene != HK_NULL )
		{
			m_sceneDataContext->removeScene( scene );
		}
	}
}

void hkpPhysicsToSceneDataBridge::getLoadedScenes( hkArray<hkxScene*>& scenes )
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

void hkpPhysicsToSceneDataBridge::resetPhysicsTransforms() const
{
	hkPointerMap<hkpWorldObject*, hkUlong>::Iterator iter = m_rigidBodyToMeshIdMap.getIterator();
	for( ; m_rigidBodyToMeshIdMap.isValid( iter ); iter = m_rigidBodyToMeshIdMap.getNext( iter ) )
	{		
		hkpRigidBody* rigidBody = hkpGetRigidBody(m_rigidBodyToMeshIdMap.getKey(iter)->getCollidable());
		hkTransform* transform = m_rigidBodyToTransformMap.getWithDefault(rigidBody, HK_NULL);

		if( rigidBody != HK_NULL && transform != HK_NULL )
		{
			hkpWorld* world = rigidBody->getWorld();

			// Remove and re-add the body so that VDB viewers are updated.
			if( world != HK_NULL )
			{
				world->removeEntity(rigidBody);
			}

			rigidBody->setTransform(*transform);

			// Only reset velocities on non-fixed bodies
			if( rigidBody->getMotionType() != hkpMotion::MOTION_FIXED )
			{
				rigidBody->setLinearVelocity(hkVector4::getZero());
				rigidBody->setAngularVelocity(hkVector4::getZero());
			}

			if( world != HK_NULL )
			{
				world->addEntity(rigidBody);
			}
		}
	}
}

void hkpPhysicsToSceneDataBridge::syncSceneDataToPhysics()
{	
	if(m_physicsWorld != HK_NULL)
	{
		m_physicsWorld->markForRead();
		// Iterate over the physics objects in the array, updating their corresponding mesh
		hkPointerMap<hkpWorldObject*, hkUlong>::Iterator iter = m_rigidBodyToMeshIdMap.getIterator();
		for( ; m_rigidBodyToMeshIdMap.isValid( iter ); iter = m_rigidBodyToMeshIdMap.getNext( iter ) )
		{
			// Attempt to cast the array object as a rigid body.
			hkpRigidBody* rigidBody = hkpGetRigidBody(m_rigidBodyToMeshIdMap.getKey(iter)->getCollidable());
			if( rigidBody != HK_NULL )
			{
				hkTransform transform;
				rigidBody->approxCurrentTransform( transform );
				hkMatrix4 matrix;
				matrix.set(transform);

				hkUlong meshId = m_rigidBodyToMeshIdMap.getWithDefault(rigidBody, (hkUlong)-1);
				
				// When syncing we need to preserve the scale term for the meshes so that it can be rendered properly.
				if ( meshId != (hkUlong)-1 )
				{
					hkMatrix4* scaleAndSkewTransform = m_meshIdToScaleAndSkewTransformMap.getWithDefault(meshId, HK_NULL);
					matrix.mul( *scaleAndSkewTransform );
				}

				hkDebugDisplay::getInstance().updateGeometry( matrix, m_rigidBodyToMeshIdMap.getValue(iter), 0 );
			}
		}
		
		m_physicsWorld->unmarkForRead();
	}	
}

hkRootLevelContainer* hkpPhysicsToSceneDataBridge::getLoadedContainer( const char* name )
{
	const int numberOfEntries = m_loadedContainers.getSize();
	for ( int i = 0; i < numberOfEntries; ++i )
	{
		if ( hkString::strCmp( m_loadedContainers[i].m_name, name ) == 0 )
		{
			return &m_loadedContainers[i].m_container;
		}
	}

	return HK_NULL;
}

void hkpPhysicsToSceneDataBridge::setPhysicsWorld( hkpWorld* physicsWorld )
{
	m_physicsWorld = physicsWorld;
}

void hkpPhysicsToSceneDataBridge::findAllRigidBodies( const hkRootLevelContainer* rootLevelContainer, hkArray<hkpRigidBody*>& rigidBodiesOut )
{
	hkpPhysicsData* physicsData = HK_NULL;

	while( 1 )
	{
		physicsData = static_cast<hkpPhysicsData*>( rootLevelContainer->findObjectByType( hkpPhysicsDataClass.getName(), physicsData) );

		if( physicsData == HK_NULL )
		{
			break;
		}
		
		// Get the physics objects from the physics systems, after getting the physics systems from the physics data
		const hkArray<hkpPhysicsSystem*>& physicsSystems = physicsData->getPhysicsSystems();
		for ( int i = 0; i < physicsSystems.getSize(); ++i )
		{			
			const hkpPhysicsSystem* physicsSystem = physicsSystems[i];
			if ( physicsSystem != HK_NULL )
			{
				hkArray<hkpRigidBody*> rigidBodies;
				rigidBodies = physicsSystem->getRigidBodies();
				for ( int j = 0; j < rigidBodies.getSize(); ++j )
				{
					rigidBodiesOut.pushBack( rigidBodies[j] );
				}
			}
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
