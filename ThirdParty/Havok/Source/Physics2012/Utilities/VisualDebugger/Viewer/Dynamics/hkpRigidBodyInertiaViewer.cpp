/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Monitor/hkMonitorStream.h>
#include <Common/Base/Math/Matrix/hkMatrix3Util.h>
#include <Common/Base/Types/Color/hkColor.h>
#include <Common/Visualize/hkProcessFactory.h>

#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/Dynamics/hkpRigidBodyInertiaViewer.h>

static const hkColor::Argb HK_DEFAULT_INERTIA_TENSOR_COLOR = hkColor::MAGENTA;

int hkpRigidBodyInertiaViewer::m_tag = 0;

hkProcess* HK_CALL hkpRigidBodyInertiaViewer::create(const hkArray<hkProcessContext*>& contexts)
{
	return new hkpRigidBodyInertiaViewer(contexts);
}

void HK_CALL hkpRigidBodyInertiaViewer::registerViewer()
{
	m_tag = hkProcessFactory::getInstance().registerProcess( getName(), create );
}

hkpRigidBodyInertiaViewer::hkpRigidBodyInertiaViewer(const hkArray<hkProcessContext*>& contexts)
: hkpWorldViewerBase(contexts)
{

}

void hkpRigidBodyInertiaViewer::init()
{
	if (m_context) // have a physics context
	{
		int nw = m_context->getNumWorlds();
		for (int i=0; i < nw; ++i)
		{
			addWorld(m_context->getWorld(i));
		}
	}
}

hkpRigidBodyInertiaViewer::~hkpRigidBodyInertiaViewer()
{
	if (m_context) // have a physics context
	{
		int nw = m_context->getNumWorlds();
		for (int i=0; i < nw; ++i)
		{
			removeWorld(m_context->getWorld(i));
		}
	}
}

void hkpRigidBodyInertiaViewer::worldRemovedCallback( hkpWorld* world )
{
	removeWorld(world);
}

void hkpRigidBodyInertiaViewer::worldAddedCallback( hkpWorld* world )
{
	addWorld(world);
}

void hkpRigidBodyInertiaViewer::removeWorld(hkpWorld* world)
{
	world->markForWrite();

	world->removeEntityListener( this );
	world->removeWorldPostSimulationListener( this );

	const hkArray<hkpSimulationIsland*>& activeIslands = world->getActiveSimulationIslands();

	{
		for(int i = 0; i < activeIslands.getSize(); i++)
		{
			const hkArray<hkpEntity*>& activeEntities = activeIslands[i]->getEntities();
			for(int j = 0; j < activeEntities.getSize(); j++)
			{
				entityRemovedCallback( activeEntities[j] );
			}
		}
	}

	// get all the inactive entities from the inactive simulation islands
	const hkArray<hkpSimulationIsland*>& inactiveIslands = world->getInactiveSimulationIslands();

	{
		for(int i = 0; i < inactiveIslands.getSize(); i++)
		{
			const hkArray<hkpEntity*>& activeEntities = inactiveIslands[i]->getEntities();
			for(int j = 0; j < activeEntities.getSize(); j++)
			{
				entityRemovedCallback( activeEntities[j] );
			}
		}
	}

	// get all the fixed bodies in the world
	if (world->getFixedIsland())
	{
		const hkArray<hkpEntity*>& fixedEntities = world->getFixedIsland()->getEntities();
		for(int j = 0; j < fixedEntities.getSize(); j++)
		{
			entityRemovedCallback( fixedEntities[j] );
		}
	}

	world->unmarkForWrite();
}

void hkpRigidBodyInertiaViewer::addWorld(hkpWorld *world)
{
	world->markForWrite();

	world->addEntityListener( this );
	world->addWorldPostSimulationListener( this );
	
	// get all the active entities from the active simulation islands
	{
		const hkArray<hkpSimulationIsland*>& activeIslands = world->getActiveSimulationIslands();

		for(int i = 0; i < activeIslands.getSize(); i++)
		{
			const hkArray<hkpEntity*>& activeEntities = activeIslands[i]->getEntities();
			for(int j = 0; j < activeEntities.getSize(); j++)
			{
				entityAddedCallback( activeEntities[j] );
			}
		}
	}

	// get all the inactive entities from the inactive simulation islands
	{
		const hkArray<hkpSimulationIsland*>& inactiveIslands = world->getInactiveSimulationIslands();

		for(int i = 0; i < inactiveIslands.getSize(); i++)
		{
			const hkArray<hkpEntity*>& activeEntities = inactiveIslands[i]->getEntities();
			for(int j = 0; j < activeEntities.getSize(); j++)
			{
				entityAddedCallback( activeEntities[j] );
			}
		}
	}


	// get all the fixed bodies in the world
	if (world->getFixedIsland())
	{
		const hkArray<hkpEntity*>& fixedEntities = world->getFixedIsland()->getEntities();
		for(int j = 0; j < fixedEntities.getSize(); j++)
		{
			entityAddedCallback( fixedEntities[j] );
		}
	}

	world->unmarkForWrite();
}

void hkpRigidBodyInertiaViewer::entityAddedCallback( hkpEntity* entity )
{
//	if(entity->getType() == HK_RIGID_BODY)
//	{
		hkpRigidBody* rigidBody = static_cast<hkpRigidBody*>(entity);
		m_entitiesCreated.pushBack(rigidBody);
//	}
}

void hkpRigidBodyInertiaViewer::entityRemovedCallback( hkpEntity* entity )
{
//	if( entity->getType() == HK_RIGID_BODY )
//	{
		hkpRigidBody* rigidBody = static_cast<hkpRigidBody*>(entity);

		// remove the id from the list of 'owned' created entities
		const int index = m_entitiesCreated.indexOf(rigidBody);
	//	HK_ASSERT2(0x312dbfd7, index != -1, "Trying to remove body which hkpRigidBodyInertiaViewer does not think has been added!");
		if(index >= 0)
		{
			m_entitiesCreated.removeAt(index);
		}
//	}
}

void hkpRigidBodyInertiaViewer::postSimulationCallback( hkpWorld* world )
{
	HK_TIMER_BEGIN("hkpRigidBodyInertiaViewer", this);

	HK_TIMER_BEGIN("getProps", this);
	m_displayBoxes.setSize(m_entitiesCreated.getSize());

	int displayBoxCount = 0;
	for(int i = 0; i < m_entitiesCreated.getSize(); i++)
	{
		const hkSimdReal invMass = m_entitiesCreated[i]->getRigidMotion()->getMassInv();
		if (invMass.isNotEqualZero())
		{
			// Want the inertia tensor to be displayed as a "box". This requires "diagonalising" the IT,
			// but for the moment we assume the IT is already diagonalised...

			// Box IT has elements given by:
			/*	hkReal alpha = halfSize(0) * 2.0f;
				hkReal beta = halfSize(1) * 2.0f;
				hkReal gamma = halfSize(2) * 2.0f;

				hkReal ixx = 1.0f / 6.0f;
				hkReal iyy = 1.0f / 6.0f;
				hkReal izz = 1.0f / 6.0f;


				hkReal ixxP = ( (beta*beta) / 12 )
								+ (   (gamma*gamma) / 12 ) * mass;

				hkReal iyyP = ( (alpha*alpha) / 12)
								+ ( (gamma*gamma) / 12 ) * mass;

				hkReal izzP = ( (alpha*alpha) / 12 )
								+ (  (beta*beta) / 12 ) * mass;

			*/
			// So ixxP/mass * 12 = (beta*beta) + (gamma*gamma);
			//    iyyP/mass * 12 = (alpha*alpha) + (gamma*gamma);
			//    izzP/mass * 12 = (alpha*alpha) + (beta*beta);

			// Therefore ixxP/mass * 12 - iyyP/mass * 12 + izzP/mass * 12 = 2 (beta*beta) etc.

			hkMatrix3 m;
			m_entitiesCreated[i]->getInertiaLocal(m);
			hkVector4 diag;
			hkMatrix3Util::_getDiagonal(m, diag);

			const hkSimdReal twelve = hkSimdReal::fromFloat(12.0f);

			hkSimdReal betaSqrd = (diag.getComponent<0>() - diag.getComponent<1>() + diag.getComponent<2>()) * invMass * hkSimdReal_6;
			hkSimdReal beta = betaSqrd.sqrt<HK_ACC_23_BIT,HK_SQRT_SET_ZERO>();	// Safety check for zeroed elements!

			hkSimdReal gammaSqrd = diag.getComponent<0>() * invMass * twelve - betaSqrd;
			hkSimdReal gamma = gammaSqrd.sqrt<HK_ACC_23_BIT,HK_SQRT_SET_ZERO>();	// Safety check for zeroed elements!

			hkSimdReal alphaSqrd = diag.getComponent<2>() * invMass * twelve - betaSqrd;
			hkSimdReal alpha = alphaSqrd.sqrt<HK_ACC_23_BIT,HK_SQRT_SET_ZERO>();	// Safety check for zeroed elements!

			const hkSimdReal scale = hkSimdReal::fromFloat(1.01f * 0.5f);

			hkVector4 halfExtents;
			halfExtents.set(alpha,beta,gamma,hkSimdReal_0);
			halfExtents.mul(scale);
		
			// We need to construct the transform to the *centre* of the OBB/AABB, since the drawObbLines() method
			// uses a (halfExtents vector4) and (transform). 
			hkTransform t = m_entitiesCreated[i]->getTransform();
			
				// calc transform as t + (vector to centre of OBB/AABB in world space)
			hkVector4 centre = m_entitiesCreated[i]->getCenterOfMassLocal();			
			hkVector4 offset;
			offset.setRotatedDir(t.getRotation(), centre);
			offset.add(t.getTranslation());
			t.setTranslation(offset);
		
			
			m_displayBoxes[displayBoxCount++].setParameters(halfExtents, t);	
			
			//hkpCollideDebugUtil::drawObbLines(t, halfExtents, HK_DEFAULT_INERTIA_TENSOR_COLOR, m_displayHandler);
		}
	}

	HK_TIMER_END();
	HK_TIMER_BEGIN("sendProps", this);

	m_displayBoxes.setSize(displayBoxCount);

	
	hkArray<hkDisplayGeometry*> geometries;
	geometries.setSize( m_displayBoxes.getSize() );

	for(int j = m_displayBoxes.getSize() - 1; j >= 0; j--)
	{
		geometries[j] = &m_displayBoxes[j];	
	}
	
	m_displayHandler->displayGeometry(geometries, HK_DEFAULT_INERTIA_TENSOR_COLOR, 0, m_tag);

	HK_TIMER_END();
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
