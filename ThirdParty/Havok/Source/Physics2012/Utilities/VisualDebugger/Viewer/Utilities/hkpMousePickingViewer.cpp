/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/Utilities/hkpMousePickingViewer.h>

#include <Common/Visualize/hkProcessFactory.h>
#include <Common/Visualize/Serialize/hkDisplaySerializeIStream.h>
#include <Common/Visualize/Serialize/hkDisplaySerializeOStream.h>
#include <Common/Visualize/hkVisualDebuggerProtocol.h>

#include <Physics2012/Collide/Agent/Collidable/hkpCollidable.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Utilities/Actions/MouseSpring/hkpMouseSpringAction.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>

int hkpMousePickingViewer::m_tag = 0;

hkProcess* HK_CALL hkpMousePickingViewer::create(const hkArray<hkProcessContext*>& contexts )
{
	return new hkpMousePickingViewer(contexts);
}

void HK_CALL hkpMousePickingViewer::registerViewer()
{
	m_tag = hkProcessFactory::getInstance().registerProcess( getName(), create );
}

hkpMousePickingViewer::hkpMousePickingViewer( const hkArray<hkProcessContext*>& contexts )
:	hkpWorldViewerBase(contexts),
	m_currentWorld(HK_NULL),
	m_mouseSpring(HK_NULL),
	m_mouseSpringMaxRelativeForce(1000.0f)
{

}

void hkpMousePickingViewer::worldRemovedCallback( hkpWorld* world )
{
	if (world == m_currentWorld)
		releaseObject();
}

hkpMousePickingViewer::~hkpMousePickingViewer()
{
	releaseObject();
}

static hkUint8 _cmds[] = { 
	hkVisualDebuggerProtocol::HK_PICK_OBJECT, 
	hkVisualDebuggerProtocol::HK_DRAG_OBJECT, 
	hkVisualDebuggerProtocol::HK_RELEASE_OBJECT };

void hkpMousePickingViewer::getConsumableCommands( hkUint8*& commands, int& numCommands )
{
	commands = _cmds;
	numCommands	= 3;
}

void hkpMousePickingViewer::consumeCommand( hkUint8 command  )
{
	switch (command)
	{
		case hkVisualDebuggerProtocol::HK_PICK_OBJECT:
		{
			hkVector4 worldPosition;
			m_inStream->readQuadVector4(worldPosition);
			hkUint64 id = m_inStream->read64u();
			if(m_inStream->isOk())
			{
				pickObject(id, worldPosition);
			}
		}
		break;
		case hkVisualDebuggerProtocol::HK_DRAG_OBJECT:
		{
			hkVector4 newWorldPosition;
			m_inStream->readQuadVector4(newWorldPosition);
			if(m_inStream->isOk())
			{
				dragObject(newWorldPosition);
			}
		}
		break;
		case hkVisualDebuggerProtocol::HK_RELEASE_OBJECT:
		{
			releaseObject();
		}
		break;
	}
}

static hkpRigidBody* _findCollidableInWorld(hkUint64 id, hkpWorld* world)
{
	for( int i = 0; i < world->getActiveSimulationIslands().getSize(); i++ )
	{
		const hkpSimulationIsland* island = world->getActiveSimulationIslands()[i];
		for( int r = 0 ; r < island->getEntities().getSize(); r++ )
		{
			hkpEntity* ent = island->getEntities()[r];
			if( (hkUint64)ent->getCollidable() == id )
			{
				return static_cast<hkpRigidBody*>(ent);
			}
		}
	}

	for( int i = 0; i < world->getInactiveSimulationIslands().getSize(); i++ )
	{
		const hkpSimulationIsland* island = world->getInactiveSimulationIslands()[i];
		for( int r = 0 ; r < island->getEntities().getSize(); r++ )
		{
			hkpEntity* ent = island->getEntities()[r];
			if( (hkUint64)ent->getCollidable() == id )
			{
				return static_cast<hkpRigidBody*>(ent);
			}
		}
	}

	return HK_NULL;
}


hkBool hkpMousePickingViewer::pickObject( hkUint64 id, const hkVector4& worldPosition )
{
	if( !m_context )
	{
		return false;
	}

	// HACK!  We know the id is actually the address of the Collidable
	// !! NOT 64 BIT SAFE !!
	
	if ((id & 0x03) == 0x03) // 0x1 == swept transform from, 0x2 = swept transform to, 0x3 = convex radius (ok to pick)
	{
		id = hkClearBits(id, 0x03);
	}
	else if ((id % 4) != 0)
	{
		return false;
	}

	hkpRigidBody* rb = HK_NULL;
	for( int w = 0; w < m_context->getNumWorlds() && rb == HK_NULL; w++ )
	{
		hkpWorld* world = m_context->getWorld(w);
		world->markForRead();
		rb = _findCollidableInWorld(id, world);
		world->unmarkForRead();
	}

	if( rb ) // may not be a collidable, or the collidable may belong to a phantom.
	{
		m_currentWorld = rb->getWorld();
		if( rb && !rb->isFixed() && rb->getWorld() == m_currentWorld)
		{
			hkVector4 positionAinA;
			positionAinA.setTransformedInversePos( rb->getTransform(), worldPosition );

			const hkReal springDamping = 0.5f;
			const hkReal springElasticity = 0.3f;
			const hkReal objectDamping = 0.95f;
			m_currentWorld->markForWrite();
			
				// addreference requires write lock on world for entities in the world
				m_mouseSpring = new hkpMouseSpringAction( positionAinA, worldPosition, springDamping, springElasticity, objectDamping, rb );

				m_mouseSpring->setMaxRelativeForce(m_mouseSpringMaxRelativeForce);
				m_currentWorld->addAction( m_mouseSpring );

			m_currentWorld->unmarkForWrite();

			return true;
		}
	}
	return true;
}

void hkpMousePickingViewer::dragObject( const hkVector4& newWorldSpacePoint )
{
	if( m_mouseSpring != HK_NULL )
	{
		if( m_mouseSpring->getWorld() )
		{
			m_currentWorld->markForWrite();
				m_mouseSpring->setMousePosition( newWorldSpacePoint );
			m_currentWorld->unmarkForWrite();
		}
	}
}

void hkpMousePickingViewer::releaseObject()
{
	if( m_mouseSpring != HK_NULL)
	{
		if( m_mouseSpring->getWorld() )
		{
			m_currentWorld->markForWrite();
				m_currentWorld->removeAction( m_mouseSpring );
				static_cast<hkpRigidBody*>( m_mouseSpring->getEntity() )->activate();	
				m_mouseSpring->removeReference();
			m_currentWorld->unmarkForWrite();
		}
		else
		{
			// no world, no lock or action added (shouldn't really happen anyway)
			m_mouseSpring->removeReference();
		}
		m_mouseSpring = HK_NULL;
	}
	m_currentWorld = HK_NULL;
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
