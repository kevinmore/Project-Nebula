/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Monitor/hkMonitorStream.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/Collide/hkpConvexRadiusViewer.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/Collide/hkpConvexRadiusBuilder.h>

#include <Common/Visualize/Shape/hkDisplayGeometry.h>
#include <Common/Visualize/hkDebugDisplayHandler.h>
#include <Common/Visualize/hkProcessFactory.h>

#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>

#include <Common/Base/Types/Color/hkColor.h>

#include <Physics2012/Utilities/VisualDebugger/ShapeHash/hkpShapeHashUtil.h>
#include <Physics2012/Collide/BroadPhase/hkpBroadPhase.h>

static const hkColor::Argb HK_DEFAULT_OBJECT_COLOR = hkColor::rgbFromChars( 140, 240, 140, 140 ); // transparent green
static const hkColor::Argb HK_DEFAULT_FIXED_OBJECT_COLOR = hkColor::rgbFromChars( 70, 200, 70, 140 ); // darker transparent green

#define ID_OFFSET 3

#define HK_CONVEX_RADIUS_HASH_START_VALUE 0x6a89f6ee50900b12ull

int hkpConvexRadiusViewer::m_tag = 0;

hkProcess* HK_CALL hkpConvexRadiusViewer::create(const hkArray<hkProcessContext*>& contexts )
{
	return new hkpConvexRadiusViewer(contexts);
}

void HK_CALL hkpConvexRadiusViewer::registerViewer()
{
	m_tag = hkProcessFactory::getInstance().registerProcess( getName(), create );
}

hkpConvexRadiusViewer::hkpConvexRadiusViewer( const hkArray<hkProcessContext*>& contexts )
: hkpWorldViewerBase( contexts )
, m_builder( new hkpConvexRadiusBuilder( hkpConvexRadiusBuilder::hkpConvexRadiusBuilderEnvironment() ) )
{	
	m_fixedObjectColor = HK_DEFAULT_FIXED_OBJECT_COLOR;
	m_movableObjectColor = HK_DEFAULT_OBJECT_COLOR;
}

void hkpConvexRadiusViewer::init()
{
	if (m_context)
	{
		for (int i=0; i < m_context->getNumWorlds(); ++i)
		{
			addWorld( m_context->getWorld(i) );
		}
	}
}

hkpConvexRadiusViewer::~hkpConvexRadiusViewer()
{
	m_builder->removeReference();
	int ne = m_worldEntities.getSize();
	for (int i=(ne-1); i >= 0; --i) // backwards as remove alters array
	{
		removeWorld(i);
	}
}

void hkpConvexRadiusViewer::worldRemovedCallback( hkpWorld* world ) 
{ 
	int de = findWorld(world);
	if (de >= 0)
	{	
		removeWorld(de);
	}	
}

//World added listener. Should impl this in sub class, but call up to this one to get the listener reg'd.
void hkpConvexRadiusViewer::worldAddedCallback( hkpWorld* world )
{
	addWorld(world);	
}

void hkpConvexRadiusViewer::addWorld(hkpWorld* world)
{
	world->markForWrite();

	world->addEntityListener( this );
	world->addWorldPostSimulationListener( this );

	WorldToEntityData* wed = new WorldToEntityData;
	wed->world = world;
	m_worldEntities.pushBack(wed);

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

void hkpConvexRadiusViewer::entityAddedCallback( hkpEntity* entity )
{	
	hkBool hasHash = false;
	hkDebugDisplayHandler::Hash hash;

	if(entity->getCollidable()->getShape() == HK_NULL)
	{
		return;
	}

	// figure out the right world list for it
	// We should defo have the world in our list
	const hkpWorld *const world = entity->getWorld();
	const int index = findWorld(world);
	if (index < 0)
	{
		return;
	}

	WorldToEntityData* wed = m_worldEntities[index];

	
	hkpRigidBody* rigidBody = static_cast<hkpRigidBody*>(entity);

	const hkpShape *const shape = rigidBody->getCollidable()->getShape();

	if ( !shape )
	{
		return;
	}

	if ( !isLocalViewer() && m_displayHandler->doesSupportHashes() )
	{
		hkpShapeHashUtil hashUtil( HK_CONVEX_RADIUS_HASH_START_VALUE );
		if ( rigidBody->hasProperty( HK_PROPERTY_OVERRIDE_SHAPE_HASH ) )
		{
			hashUtil.writeUint64( rigidBody->getProperty( HK_PROPERTY_OVERRIDE_SHAPE_HASH ).m_data );
		}
		else
		{
			hashUtil.writeShape( shape, hkpShapeHashUtil::USE_CONVEX_RADIUS );
		}
		hasHash = hashUtil.getHash( hash );
	}

	const hkUlong id = (hkUlong)(rigidBody->getCollidable());
	const hkUlong displayId = id + ID_OFFSET; // odd number(!== collidable), so will not be pickable.
	const hkColor::Argb color = rigidBody->isFixed() ? m_fixedObjectColor : m_movableObjectColor;

	if ( !hasHash )
	{
		// We add the geometry lazily so that the handler can opt to build it in parts.
		if ( m_displayHandler->addGeometryLazily( shape, m_builder, rigidBody->getTransform(), displayId, m_tag, 0 ) == HK_SUCCESS )
		{
			wed->entitiesCreated.pushBack( displayId );
		
			m_displayHandler->setGeometryColor( color, displayId, m_tag );
		}
	}
	else
	{
		const hkTransform& transform = rigidBody->getTransform();
		// Get the aabb from the broadphase.
		hkAabb aabb;
		{
			const hkpCollidable *const collidable = entity->getCollidable();
			const hkpBroadPhaseHandle *const broadPhaseHandle = collidable->getBroadPhaseHandle();
			const hkpBroadPhase *const broadPhase = world->getBroadPhase();
			broadPhase->markForRead();
			broadPhase->getAabb( broadPhaseHandle, aabb );
			broadPhase->unmarkForRead();
		}

		wed->entitiesCreated.pushBack( displayId );
		
		m_displayHandler->addGeometryHash( shape, m_builder, hash, aabb, color, transform, displayId, m_tag );
	}
}

void hkpConvexRadiusViewer::entityRemovedCallback( hkpEntity* entity )
{
	if( entity->getCollidable()->getShape() == HK_NULL )
	{
		return;
	}

	hkpWorld* world = entity->getWorld();
	int worldIndex = findWorld(world);
	if (worldIndex >= 0)
	{
		WorldToEntityData* wed = m_worldEntities[worldIndex];

		hkpRigidBody* rigidBody = static_cast<hkpRigidBody*>(entity);
		hkUlong id = (hkUlong)rigidBody->getCollidable();

		// remove the geometry from the displayHandler
		hkUlong displayId = id + ID_OFFSET;
		m_displayHandler->removeGeometry(displayId, m_tag, 0);

		// remove the id from the list of 'owned' created entities
		const int index = wed->entitiesCreated.indexOf(displayId);
		if(index >= 0)
		{
			wed->entitiesCreated.removeAt(index);
		}
	}
}



void hkpConvexRadiusViewer::postSimulationCallback( hkpWorld* world )
{
	HK_TIMER_BEGIN("hkpConvexRadiusViewer", this);

	// update the transform for all active entities (in all the active simulation islands)
	{
		const hkArray<hkpSimulationIsland*>& activeIslands = world->getActiveSimulationIslands();

		for(int i = 0; i < activeIslands.getSize(); i++)
		{
			const hkArray<hkpEntity*>& activeEntities = activeIslands[i]->getEntities();
			for(int j = 0; j < activeEntities.getSize(); j++)
			{
				hkpRigidBody* rigidBody = static_cast<hkpRigidBody*>(activeEntities[j]);
				hkUlong id = hkUlong( rigidBody->getCollidable() );
				hkUlong displayId = id + ID_OFFSET;

				hkTransform transform;
				rigidBody->approxTransformAt( world->getCurrentTime(), transform );
				m_displayHandler->updateGeometry( transform, displayId , m_tag );
			}
		}
	}

	HK_TIMER_END();

}

void hkpConvexRadiusViewer::removeAllGeometries(int worldIndex)
{
	WorldToEntityData* wed = m_worldEntities[worldIndex];
	for(int i = 0; i < wed->entitiesCreated.getSize(); i++)
	{
		m_displayHandler->removeGeometry(wed->entitiesCreated[i], m_tag, 0);
	}
	wed->entitiesCreated.setSize(0);
}

int hkpConvexRadiusViewer::findWorld( const hkpWorld* world )
{
	int ne = m_worldEntities.getSize();
	for (int i=0; i < ne; ++i)
	{
		if (m_worldEntities[i]->world == world)
			return i;
	}
	return -1;
}

void hkpConvexRadiusViewer::removeWorld( int i )
{
	m_worldEntities[i]->world->markForWrite();

		m_worldEntities[i]->world->removeEntityListener( this );
		m_worldEntities[i]->world->removeWorldPostSimulationListener( this );
		removeAllGeometries(i);

	m_worldEntities[i]->world->unmarkForWrite();

	delete m_worldEntities[i];
	m_worldEntities.removeAt(i);
	// other base listeners handled in worldviewerbase
}

void hkpConvexRadiusViewer::inactiveEntityMovedCallback( hkpEntity* entity )
{
	hkpRigidBody* rigidBody = static_cast<hkpRigidBody*>(entity);
	hkUlong id =  (hkUlong)rigidBody->getCollidable();
	id += ID_OFFSET;
	m_displayHandler->updateGeometry(rigidBody->getTransform(), id, m_tag);
}

// Hack: internal method used by Destruction in the Preview Tool. Search for [0xaf5511e1] or see COM-1312
// for more details.
void hkpConvexRadiusViewer::setMinimumVisibleRadius(hkReal radius)
{
	m_builder->setMinimumVisibleRadius(radius);
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
