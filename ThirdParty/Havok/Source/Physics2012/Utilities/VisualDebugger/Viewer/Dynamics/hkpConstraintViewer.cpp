/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Monitor/hkMonitorStream.h>

#include <Physics2012/Utilities/VisualDebugger/Viewer/Dynamics/hkpConstraintViewer.h>
#include <Common/Visualize/hkDebugDisplayHandler.h>
#include <Common/Visualize/hkProcessFactory.h>

#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>

#include <Common/Visualize/hkDebugDisplay.h>
#include <Common/Base/Types/Color/hkColor.h>
#include <Physics2012/Dynamics/Constraint/hkpConstraintInstance.h>
#include <Physics2012/Dynamics/Constraint/Breakable/hkpBreakableConstraintData.h>
#include <Physics2012/Dynamics/Constraint/Malleable/hkpMalleableConstraintData.h>
#include <Physics2012/Dynamics/Constraint/Chain/hkpConstraintChainInstance.h>

#include <Physics/Constraint/Visualize/DrawDispatcher/hkpDrawDispatcher.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/Dynamics/Constraint/Drawer/hkpConstraintChainDrawer.h>

int hkpConstraintViewer::m_tag = 0;
hkReal hkpConstraintViewer::m_scale = 0.25f;

void HK_CALL hkpConstraintViewer::registerViewer()
{
	m_tag = hkProcessFactory::getInstance().registerProcess( getName(), create );
}

hkProcess* HK_CALL hkpConstraintViewer::create(const hkArray<hkProcessContext*>& contexts)
{
	return new hkpConstraintViewer(contexts);
}

hkpConstraintViewer::hkpConstraintViewer(const hkArray<hkProcessContext*>& contexts)
: hkpWorldViewerBase( contexts )
{
	if (m_context)
	{
		for (int i=0; i < m_context->getNumWorlds(); ++i)
		{
			hkpWorld* world = m_context->getWorld(i);
	//		world->addConstraintListener( this );
			world->markForWrite();
			world->addWorldPostSimulationListener( this );
			world->unmarkForWrite();
		}
	}
}

hkpConstraintViewer::~hkpConstraintViewer()
{
	if (m_context)
	{
		for (int i=0; i < m_context->getNumWorlds(); ++i)
		{
			hkpWorld* world = m_context->getWorld(i);
	//		world->removeConstraintListener( this );
			world->markForWrite();
			world->removeWorldPostSimulationListener( this );
			world->unmarkForWrite();
		}
	}
}



void hkpConstraintViewer::worldAddedCallback( hkpWorld* world)
{
//	world->addConstraintListener( this );
	world->markForWrite();
	world->addWorldPostSimulationListener( this );
	world->unmarkForWrite();
}

void hkpConstraintViewer::worldRemovedCallback( hkpWorld* world)
{
//	world->removeConstraintListener( this );
	world->markForWrite();
	world->removeWorldPostSimulationListener( this );
	world->unmarkForWrite();
}

void hkpConstraintViewer::postSimulationCallback( hkpWorld* world )
{
	HK_TIMER_BEGIN("hkpConstraintViewer", this);

	{
		const hkArray<hkpSimulationIsland*>& islands = world->getActiveSimulationIslands();
		for ( int i = 0; i < islands.getSize(); ++i )
		{
			for ( int e = 0; e < islands[i]->getEntities().getSize(); e++)
			{
				hkpEntity* entity = islands[i]->getEntities()[e];
				const hkSmallArray<struct hkConstraintInternal>&  constraintMasters = entity->getConstraintMasters();

				for ( int c = 0; c < constraintMasters.getSize(); c++)
				{
					draw ( constraintMasters[c].m_constraint, entity, m_displayHandler );
				}
			}
		}
	}

	{
		const hkArray<hkpSimulationIsland*>& islands = world->getInactiveSimulationIslands();
		for ( int i = 0; i < islands.getSize(); ++i )
		{
			for ( int e = 0; e < islands[i]->getEntities().getSize(); e++)
			{
				hkpEntity* entity = islands[i]->getEntities()[e];
				const hkSmallArray<struct hkConstraintInternal>&  constraintMasters = entity->getConstraintMasters();
				for ( int c = 0; c < constraintMasters.getSize(); c++)
				{
					draw ( constraintMasters[c].m_constraint, entity, m_displayHandler );
				}
			}
		}
	}

	HK_TIMER_END();
}


void hkpConstraintViewer::draw(hkpConstraintInstance* constraint, const hkpEntity* masterEntity, hkDebugDisplayHandler* displayHandler)
{
	HK_TIMER_BEGIN("draw", this);

	int type = constraint->getData()->getType();

	HK_ASSERT2(0x21d74fd9, displayHandler,"displayHandler is NULL");

	
	hkReferencedObject::lockAll();

	// Get the rigid bodies from the constraint
	hkpRigidBody* bodyA = reinterpret_cast<hkpRigidBody*>(constraint->getEntityA());
	hkpRigidBody* bodyB = reinterpret_cast<hkpRigidBody*>(constraint->getEntityB());

	// Get their local to world transforms
	const hkTransform& transformA = bodyA->getTransform();
	const hkTransform& transformB = bodyB->getTransform();

	// associate the drawings with the collidable
	const int id = (int)(hkUlong)(masterEntity->getCollidable());

	switch(type)
	{
	case hkpConstraintData::CONSTRAINT_TYPE_BREAKABLE:
		{
			hkpBreakableConstraintData* breakableConstraint = static_cast<hkpBreakableConstraintData*>(constraint->getDataRw());
			hkpConstraintInstance fakeConstraint(constraint->getEntityA(), constraint->getEntityB(), breakableConstraint->getWrappedConstraintData() );
			draw(&fakeConstraint, masterEntity, displayHandler);
		}
		break;
	case hkpConstraintData::CONSTRAINT_TYPE_MALLEABLE:
		{
			hkpMalleableConstraintData* malleableConstraint = static_cast<hkpMalleableConstraintData*>(constraint->getDataRw());
			hkpConstraintInstance fakeConstraint(constraint->getEntityA(), constraint->getEntityB(), malleableConstraint->getWrappedConstraintData() );
			draw(&fakeConstraint, masterEntity, displayHandler);
		}
		break;
	case hkpConstraintData::CONSTRAINT_TYPE_STIFF_SPRING_CHAIN:
	case hkpConstraintData::CONSTRAINT_TYPE_BALL_SOCKET_CHAIN:
	case hkpConstraintData::CONSTRAINT_TYPE_POWERED_CHAIN:
		{
			hkpConstraintChainDrawer drawer;
			drawer.setScale(m_scale);
			const hkpConstraintChainData* data = static_cast<const hkpConstraintChainData*>(constraint->getData());
			drawer.drawConstraint(data, transformA, transformB, displayHandler, id, m_tag, static_cast<const hkpConstraintChainInstance*>(constraint)->m_chainedEntities );
		}
		break;
	default:
		hkpDispatchDraw(constraint->getData(), transformA, transformB, displayHandler, id, m_tag, m_scale );
		break;
	}

	hkReferencedObject::unlockAll();

	HK_TIMER_END();
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
