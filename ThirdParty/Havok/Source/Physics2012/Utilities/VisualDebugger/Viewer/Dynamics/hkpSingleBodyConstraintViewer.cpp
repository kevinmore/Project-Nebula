/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/Dynamics/hkpSingleBodyConstraintViewer.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/Dynamics/hkpConstraintViewer.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>

#include <Common/Visualize/hkProcessFactory.h>
#include <Common/Visualize/Serialize/hkDisplaySerializeIStream.h>
#include <Common/Visualize/Serialize/hkDisplaySerializeOStream.h>
#include <Common/Visualize/hkVisualDebuggerProtocol.h>

#include <Physics2012/Collide/Agent/Collidable/hkpCollidable.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>

int hkpSingleBodyConstraintViewer::m_tag = 0;

hkProcess* HK_CALL hkpSingleBodyConstraintViewer::create(const hkArray<hkProcessContext*>& contexts )
{
	return new hkpSingleBodyConstraintViewer(contexts);
}

void HK_CALL hkpSingleBodyConstraintViewer::registerViewer()
{
	m_tag = hkProcessFactory::getInstance().registerProcess( getName(), create );
}

hkpSingleBodyConstraintViewer::hkpSingleBodyConstraintViewer( const hkArray<hkProcessContext*>& contexts )
:	hkpWorldViewerBase(contexts),
	m_currentWorld(HK_NULL),
	m_pickedBody(HK_NULL)
{
	if (m_context)
	{
		for (int i = 0; i < m_context->getNumWorlds(); ++i)
		{
			hkpWorld* world = m_context->getWorld(i);
			world->markForWrite();
			world->addWorldPostSimulationListener( this );
			world->unmarkForWrite();
		}
	}
}

hkpSingleBodyConstraintViewer::~hkpSingleBodyConstraintViewer()
{
	releaseObject();

	if (m_context)
	{
		for (int i=0; i < m_context->getNumWorlds(); ++i)
		{
			hkpWorld* world = m_context->getWorld(i);
			world->markForWrite();
			world->removeWorldPostSimulationListener( this );
			world->unmarkForWrite();
		}
	}
}

void hkpSingleBodyConstraintViewer::worldRemovedCallback( hkpWorld* world )
{
	if (world == m_currentWorld)
		releaseObject();
}

void hkpSingleBodyConstraintViewer::entityRemovedCallback( hkpEntity* entity )
{
	if (entity == m_pickedBody)
		releaseObject();
}

void hkpSingleBodyConstraintViewer::entityDeletedCallback( hkpEntity* entity )
{
	entityRemovedCallback(entity);
}

void hkpSingleBodyConstraintViewer::postSimulationCallback( hkpWorld* world )
{
	if (m_pickedBody)
	{
		hkLocalArray<hkpConstraintInstance*> constraints(10);
		m_pickedBody->getAllConstraints(constraints);
		for (hkLocalArray<hkpConstraintInstance*>::iterator it = constraints.begin(), last = constraints.end(); it != last; it++)
		{
			hkpConstraintInstance* constraint = *it;
			hkpConstraintViewer::draw(constraint, constraint->getMasterEntity(), m_displayHandler);
		}
	}
}

static hkUint8 _cmds[] = { 
	hkVisualDebuggerProtocol::HK_PICK_OBJECT };

void hkpSingleBodyConstraintViewer::getConsumableCommands( hkUint8*& commands, int& numCommands )
{
	commands = _cmds;
	numCommands	= 1;
}

void hkpSingleBodyConstraintViewer::consumeCommand( hkUint8 command  )
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
	}
}

hkBool hkpSingleBodyConstraintViewer::pickObject( hkUint64 id, const hkVector4& worldPosition )
{
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

	hkpCollidable* col = reinterpret_cast<hkpCollidable*>( (hkUlong) id);
	hkpRigidBody * rb = hkpGetRigidBody(col);
	if(rb && rb != m_pickedBody) // may not be, may be a phantom for instance
	{
		releaseObject();
		m_pickedBody = rb;
		m_currentWorld = rb->getWorld();

		m_currentWorld->markForWrite();

		// addEntityListener requires write lock on world for entities in the world
		m_pickedBody->addEntityListener(this);

		m_currentWorld->unmarkForWrite();
	}
	return true;
}

void hkpSingleBodyConstraintViewer::releaseObject()
{
	if(m_pickedBody)
	{
		m_currentWorld->markForWrite();
		m_pickedBody->removeEntityListener(this);
		m_currentWorld->unmarkForWrite();
		m_pickedBody = HK_NULL;
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
