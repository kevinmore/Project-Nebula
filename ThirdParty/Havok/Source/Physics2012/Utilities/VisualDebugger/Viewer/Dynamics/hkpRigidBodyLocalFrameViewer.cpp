/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Monitor/hkMonitorStream.h>
#include <Common/Base/hkBase.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/Dynamics/hkpRigidBodyLocalFrameViewer.h>
#include <Common/Visualize/hkDebugDisplayHandler.h>
#include <Common/Visualize/hkProcessFactory.h>

#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>

#include <Common/Visualize/hkDrawUtil.h>

int hkpRigidBodyLocalFrameViewer::m_tag = 0;
hkReal hkpRigidBodyLocalFrameViewer::m_scale = 1.0f;

hkProcess* HK_CALL hkpRigidBodyLocalFrameViewer::create(const hkArray<hkProcessContext*>& contexts)
{
	return new hkpRigidBodyLocalFrameViewer( contexts );	
}

void HK_CALL hkpRigidBodyLocalFrameViewer::registerViewer()
{
	m_tag = hkProcessFactory::getInstance().registerProcess( getName(), create );
}

hkpRigidBodyLocalFrameViewer::hkpRigidBodyLocalFrameViewer(const hkArray<hkProcessContext*>& contexts)
:hkpWorldViewerBase(contexts)
{	
}

void hkpRigidBodyLocalFrameViewer::init()
{
	if (m_context)
	{
		for( int i=0; i < m_context->getNumWorlds(); ++i)
		{
			worldAddedCallback( m_context->getWorld(i) );
		}
	}
}

hkpRigidBodyLocalFrameViewer::~hkpRigidBodyLocalFrameViewer()
{
	if (m_context)
	{
		for( int i=0; i < m_context->getNumWorlds(); ++i)
		{
			worldRemovedCallback( m_context->getWorld(i) );
		}
	}
}

void hkpRigidBodyLocalFrameViewer::worldRemovedCallback( hkpWorld* world )
{
	world->markForWrite();
	world->removeWorldPostSimulationListener( this );
	world->unmarkForWrite();
}

void hkpRigidBodyLocalFrameViewer::worldAddedCallback( hkpWorld* world )
{
	world->markForWrite();
	world->addWorldPostSimulationListener( this );
	world->unmarkForWrite();
}

void hkpRigidBodyLocalFrameViewer::postSimulationCallback( hkpWorld* physicsWorld )
{
	physicsWorld->markForRead();

	hkArray<hkpRigidBody*> rigidBodiesWithLocalFrames;
	{			
		const hkArray<hkpSimulationIsland*>& activeSimulationIslands = physicsWorld->getActiveSimulationIslands();

		for( int j = 0; j < activeSimulationIslands.getSize(); ++j )
		{
			for( int k = 0; k < activeSimulationIslands[j]->getEntities().getSize(); ++k )
			{
				hkpEntity* entity = activeSimulationIslands[j]->getEntities()[k];

				if( entity->m_localFrame != HK_NULL )
				{
					rigidBodiesWithLocalFrames.pushBack(static_cast<hkpRigidBody*>(entity));
				}
			}
		}

		const hkArray<hkpSimulationIsland*>& inactiveSimulationIslands = physicsWorld->getInactiveSimulationIslands();

		for( int j = 0; j < inactiveSimulationIslands.getSize(); ++j )
		{
			for( int k = 0; k < inactiveSimulationIslands[j]->getEntities().getSize(); ++k )
			{
				hkpEntity* entity = inactiveSimulationIslands[j]->getEntities()[k];

				if( entity->m_localFrame != HK_NULL )
				{
					rigidBodiesWithLocalFrames.pushBack(static_cast<hkpRigidBody*>(entity));
				}
			}
		}

		const hkpSimulationIsland* fixedIsland = physicsWorld->getFixedIsland();

		for( int j = 0; j < fixedIsland->getEntities().getSize(); ++j )
		{
			hkpEntity* entity = fixedIsland->getEntities()[j];

			if( entity->m_localFrame != HK_NULL )
			{
				rigidBodiesWithLocalFrames.pushBack(static_cast<hkpRigidBody*>(entity));
			}					
		}
	}

	for( int j = 0; j < rigidBodiesWithLocalFrames.getSize(); ++j )
	{
		hkTransform transform;
		rigidBodiesWithLocalFrames[j]->approxCurrentTransform(transform);
		hkDrawUtil::displayLocalFrame( *rigidBodiesWithLocalFrames[j]->m_localFrame, transform, m_scale, true, 0xFF4982B8, " (local frame)"  );
	}

	physicsWorld->unmarkForRead();
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
