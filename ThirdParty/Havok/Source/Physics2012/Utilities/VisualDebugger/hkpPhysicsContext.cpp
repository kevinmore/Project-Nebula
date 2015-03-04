/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Dynamics/World/hkpPhysicsSystem.h>
#include <Physics2012/Dynamics/World/Listener/hkpWorldDeletionListener.h>
#include <Physics/Constraint/Data/hkpConstraintData.h>

#include <Physics2012/Utilities/VisualDebugger/hkpPhysicsContext.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/Collide/hkpBroadphaseViewer.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/Collide/hkpMidphaseViewer.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/Dynamics/hkpConstraintViewer.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/Collide/hkpActiveContactPointViewer.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/Collide/hkpInactiveContactPointViewer.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/Collide/hkpToiContactPointViewer.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/Collide/hkpToiCountViewer.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/Dynamics/hkpPhantomDisplayViewer.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/Dynamics/hkpRigidBodyCentreOfMassViewer.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/Dynamics/hkpRigidBodyLocalFrameViewer.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/Dynamics/hkpRigidBodyInertiaViewer.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/Collide/hkpShapeDisplayViewer.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/Collide/hkpConvexRadiusViewer.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/Dynamics/hkpSweptTransformDisplayViewer.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/Dynamics/hkpSimulationIslandViewer.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/Utilities/hkpMousePickingViewer.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/Dynamics/hkpWorldSnapshotViewer.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/Vehicle/hkpVehicleViewer.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/Collide/hkpWeldingViewer.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/Collide/hkpInconsistentWindingViewer.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/Dynamics/hkpSingleBodyConstraintViewer.h>

#include <Common/Visualize/hkVisualDebugger.h>

void HK_CALL hkpPhysicsContext::registerAllPhysicsProcesses()
{
	hkpBroadphaseViewer::registerViewer();
	hkpMidphaseViewer::registerViewer();
	hkpRigidBodyCentreOfMassViewer::registerViewer();
	hkpRigidBodyLocalFrameViewer::registerViewer();
	hkpConstraintViewer::registerViewer();
	hkpConvexRadiusViewer::registerViewer();
	hkpActiveContactPointViewer::registerViewer();
	hkpInactiveContactPointViewer::registerViewer();
	hkpToiContactPointViewer::registerViewer();
	hkpToiCountViewer::registerViewer();
	hkpRigidBodyInertiaViewer::registerViewer();
	hkpMousePickingViewer::registerViewer();
	hkpPhantomDisplayViewer::registerViewer();
	hkpShapeDisplayViewer::registerViewer();
	hkpSimulationIslandViewer::registerViewer();
	hkpSweptTransformDisplayViewer::registerViewer();
	hkpVehicleViewer::registerViewer();
	hkpWorldSnapshotViewer::registerViewer();
	hkpWeldingViewer::registerViewer();
	hkpInconsistentWindingViewer::registerViewer();
	hkpSingleBodyConstraintViewer::registerViewer();
}

hkpPhysicsContext::hkpPhysicsContext()
{
	
}

hkpPhysicsContext::~hkpPhysicsContext()
{
	for (int w=(m_worlds.getSize()-1); w >=0 ; --w)
	{
		removeWorld( m_worlds[w] );
	}
}

void hkpPhysicsContext::setOwner(hkVisualDebugger* vdb)
{
	if (m_owner)
	{
		// iterate in reverse to avoid null dereference
		// inside removeFromInspection.
		// see HVK-5577.
		for ( int wi = m_worlds.getSize()-1; wi >= 0; --wi )
			removeFromInspection(m_worlds[wi]);	
	}

	m_owner = vdb;

	if (vdb)
	{
		for ( int i = 0; i < m_worlds.getSize(); ++i )
			addForInspection( m_worlds[i] );	
	}
}

void hkpPhysicsContext::removeWorld( hkpWorld* oldWorld )
{
	int wi = m_worlds.indexOf(oldWorld);
	if (wi >= 0)
	{
		oldWorld->removeWorldDeletionListener(this);
		for (int i=0; i < m_addListeners.getSize(); ++i)
		{
			m_addListeners[i]->worldRemovedCallback(oldWorld);
		}		

		removeFromInspection( oldWorld );

		m_worlds.removeAt(wi);
	}
}

void hkpPhysicsContext::addWorld( hkpWorld* newWorld )
{
	// make sure we don't have it already
	if (m_worlds.indexOf(newWorld) < 0)
	{
		newWorld->markForWrite();
			newWorld->addWorldDeletionListener(this);
		newWorld->unmarkForWrite();

		m_worlds.pushBack(newWorld);

		for (int i=0; i < m_addListeners.getSize(); ++i)
		{
			m_addListeners[i]->worldAddedCallback( newWorld );
		}

		addForInspection( newWorld );
	}
}

// XXX get this from a reg so that we don't have to always affect the code footprint..
extern const hkClass hkpEntityClass;
extern const hkClass hkpPhantomClass;
extern const hkClass hkpActionClass;
extern const hkClass hkpConstraintInstanceClass;

void hkpPhysicsContext::addForInspection(hkpWorld* w)
{
	if (m_owner && w)
	{
		w->lock();

		w->addEntityListener(this);
		w->addPhantomListener(this);
		w->addActionListener(this);
		w->addConstraintListener(this);

		hkpWorldCinfo& cinfo = m_worldCinfos.expandOne();
		w->getCinfo( cinfo );
		if (m_owner)
		{
			m_owner->addTrackedObject(&cinfo, hkpWorldCinfoClass, "hkpWorldCinfo", HK_NULL);
		}

		// easiest to get the world to give us the info
		// (world itself is not a valid ptr for inspection
		//  as it has no class.. literally ;)
		hkpPhysicsSystem* sys = w->getWorldAsOneSystem();
		const hkArray<hkpRigidBody*>& rbs = sys->getRigidBodies();
		for (int ri=0; ri < rbs.getSize(); ++ri)
			entityAddedCallback( const_cast<hkpRigidBody*>(rbs[ri]) );
		
		const hkArray<hkpPhantom*>& phantoms = sys->getPhantoms();
		for (int pi=0; pi < phantoms.getSize(); ++pi)
			phantomAddedCallback( const_cast<hkpPhantom*>(phantoms[pi]) );

		const hkArray<hkpAction*>& actions = sys->getActions();
		for (int ai=0; ai < actions.getSize(); ++ai)
			actionAddedCallback( const_cast<hkpAction*>(actions[ai]) );

		const hkArray<hkpConstraintInstance*>& constraints = sys->getConstraints();
		for (int ci=0; ci < constraints.getSize(); ++ci)
			constraintAddedCallback( const_cast<hkpConstraintInstance*>(constraints[ci]) );

		sys->removeReference();	

		w->unlock();
	}
}

void hkpPhysicsContext::removeFromInspection(hkpWorld* w)
{
	if (m_owner && w)
	{
		w->removeEntityListener(this);
		w->removePhantomListener(this);
		w->removeActionListener(this);
		w->removeConstraintListener(this);

		int idx = m_worlds.indexOf( w );
		if (m_owner && idx != -1)
		{
			m_owner->removeTrackedObject( m_worldCinfos.begin() + idx );
			m_worldCinfos.removeAt(idx);
		}

		hkpPhysicsSystem* sys = w->getWorldAsOneSystem();
		const hkArray<hkpRigidBody*>& rbs = sys->getRigidBodies();
		for (int ri=0; ri < rbs.getSize(); ++ri)
			entityRemovedCallback(const_cast<hkpRigidBody*>(rbs[ri]) );

		const hkArray<hkpPhantom*>& phantoms = sys->getPhantoms();
		for (int pi=0; pi < phantoms.getSize(); ++pi)
			phantomRemovedCallback(const_cast<hkpPhantom*>(phantoms[pi]) );

		const hkArray<hkpAction*>& actions = sys->getActions();
		for (int ai=0; ai < actions.getSize(); ++ai)
			actionRemovedCallback(const_cast<hkpAction*>(actions[ai]) );

		const hkArray<hkpConstraintInstance*>& constraints = sys->getConstraints();
		for (int ci=0; ci < constraints.getSize(); ++ci)
			constraintRemovedCallback(const_cast<hkpConstraintInstance*>(constraints[ci]) );

		sys->removeReference();
	}
}

int hkpPhysicsContext::findWorld(hkpWorld* world)
{
	return m_worlds.indexOf(world);
}

void hkpPhysicsContext::worldDeletedCallback( hkpWorld* world )
{
	removeWorld(world);
}

void hkpPhysicsContext::entityAddedCallback( hkpEntity* entity )
{
	if (m_owner)
		m_owner->addTrackedObject(entity, hkpEntityClass, "Entities", (hkUlong)entity->getCollidable());
}

void hkpPhysicsContext::entityRemovedCallback( hkpEntity* entity )
{
	if (m_owner)
		m_owner->removeTrackedObject(entity);
}

void hkpPhysicsContext::phantomAddedCallback( hkpPhantom* phantom )
{
	if (m_owner)
		m_owner->addTrackedObject(phantom, hkpPhantomClass, "Phantoms", (hkUlong)phantom->getCollidable());
}

void hkpPhysicsContext::phantomRemovedCallback( hkpPhantom* phantom )
{
	if (m_owner)
		m_owner->removeTrackedObject(phantom);
}

void hkpPhysicsContext::constraintAddedCallback( hkpConstraintInstance* constraint )
{
	if (m_owner && constraint->getData() && constraint->getData()->getType() != hkpConstraintData::CONSTRAINT_TYPE_CONTACT)
	{
		m_owner->addTrackedObject(constraint, hkpConstraintInstanceClass, "Constraints", HK_NULL /*.. no ID for constraint*/);
	}
}

void hkpPhysicsContext::constraintRemovedCallback( hkpConstraintInstance* constraint )
{
	if (m_owner && constraint->getData() && constraint->getData()->getType() != hkpConstraintData::CONSTRAINT_TYPE_CONTACT)
		m_owner->removeTrackedObject(constraint);
}

void hkpPhysicsContext::actionAddedCallback( hkpAction* action )
{
	if (m_owner)
		m_owner->addTrackedObject(action, hkpActionClass, "Actions", HK_NULL /*no ID for action*/);
}

void hkpPhysicsContext::actionRemovedCallback( hkpAction* action )
{
	if (m_owner)
		m_owner->removeTrackedObject(action);
}


void hkpPhysicsContext::addWorldAddedListener( hkpPhysicsContextWorldListener* cb )
{
	if (m_addListeners.indexOf(cb)< 0)
	{
		m_addListeners.pushBack( cb );
	}
}

void hkpPhysicsContext::removeWorldAddedListener( hkpPhysicsContextWorldListener* cb )
{
	int index = m_addListeners.indexOf(cb);
	if (index >= 0 )
	{
		m_addListeners.removeAt( index );
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
