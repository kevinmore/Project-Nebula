/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Common/Base/Monitor/hkMonitorStream.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/Collide/hkpShapeDisplayViewer.h>
#include <Physics2012/Utilities/VisualDebugger/Viewer/hkpShapeDisplayBuilder.h>

#include <Common/Visualize/Shape/hkDisplayGeometry.h>
#include <Common/Visualize/hkDebugDisplayHandler.h>
#include <Common/Visualize/hkProcessFactory.h>
#include <Common/Visualize/hkCommandRouter.h>

#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>
#include <Physics2012/Dynamics/World/hkpWorld.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>

#include <Common/Base/Types/Color/hkColor.h>
#include <Common/Visualize/Shape/hkDisplayConvex.h>

#include <Physics2012/Utilities/VisualDebugger/ShapeHash/hkpShapeHashUtil.h>
#include <Physics2012/Collide/BroadPhase/hkpBroadPhase.h>
#include <Common/Visualize/hkVisualDebuggerProtocol.h>
#include <Common/Visualize/hkVersionReporter.h>

#if defined( HK_PLATFORM_XBOX360 ) || defined( HK_PLATFORM_PS3_PPU ) || defined (HK_PLATFORM_WIN32 ) || defined( HK_PLATFORM_LINUX) || defined(HK_PLATFORM_MAC) || defined(HK_PLATFORM_IOS) || defined(HK_PLATFORM_PSVITA) || defined(HK_PLATFORM_ANDROID) || defined(HK_PLATFORM_CTR) || defined(HK_PLATFORM_WIIU) || defined(HK_PLATFORM_PS4) 
#define PLATFORM_SUPPORTS_INSTANCING 1
#endif

#define HK_SHAPE_DISPLAY_HASH_START_VALUE 0xabe76d3e90b1e6a8ull

extern const hkColor::Argb HK_SHAPE_DISPLAY_VIEWER_DEFAULT_OBJECT_COLOR = hkColor::rgbFromChars( 240, 240, 240, 255 );
extern const hkColor::Argb HK_SHAPE_DISPLAY_VIEWER_DEFAULT_FIXED_OBJECT_COLOR = hkColor::rgbFromChars( 120, 120, 120, 255 );

int hkpShapeDisplayViewer::s_tagShapeViewer = 0;
int hkpShapeDisplayViewer::s_tagTriggerVolumeViewer = 0;


hkProcess* HK_CALL hkpShapeDisplayViewer::createShapeViewer(const hkArray<hkProcessContext*>& contexts )
{
	return new hkpShapeDisplayViewer(contexts, s_tagShapeViewer, isShape);
}

hkProcess* HK_CALL hkpShapeDisplayViewer::createTriggerVolumeViewer(const hkArray<hkProcessContext*>& contexts )
{
	return new hkpShapeDisplayViewer(contexts, s_tagTriggerVolumeViewer, isTriggerVolume);
}

void HK_CALL hkpShapeDisplayViewer::registerViewer()
{
	s_tagShapeViewer = hkProcessFactory::getInstance().registerProcess( "Shapes", createShapeViewer );
	s_tagTriggerVolumeViewer = hkProcessFactory::getInstance().registerProcess( "Trigger Volumes", createTriggerVolumeViewer );
}


hkpShapeDisplayViewer::hkpShapeDisplayViewer( const hkArray<hkProcessContext*>& contexts, int processTag, IsDisplayableFunction display )
: hkpWorldViewerBase( contexts )
, m_builder( new hkpShapeDisplayBuilder( hkpShapeDisplayBuilder::hkpShapeDisplayBuilderEnvironment() ) )
{
	m_tag = processTag;
	m_isDisplayable = display;
	m_enableShapeTransformUpdate = true;
	m_enableInstancing = false;
	m_enableDisplayCaching = false;
	m_enableDisplayCreation = true;
	m_autoGeometryCreation = true;
	m_enableAutoColor = true;
	m_enableAutoColorOverride = false;
	
	m_fixedObjectColor = HK_SHAPE_DISPLAY_VIEWER_DEFAULT_FIXED_OBJECT_COLOR;
	m_movableObjectColor = HK_SHAPE_DISPLAY_VIEWER_DEFAULT_OBJECT_COLOR;

	m_timeForDisplay = -1.f;

	int nc = contexts.getSize();
	for (int i=0; i < nc; ++i)
	{
		if ( hkString::strCmp(HK_DISPLAY_VIEWER_OPTIONS_CONTEXT, contexts[i]->getType() ) ==0 )
		{
			ShapeDisplayViewerOptions* options = static_cast<ShapeDisplayViewerOptions*>(contexts[i] );
			m_enableShapeTransformUpdate = options->m_enableShapeTransformUpdate;
			break;
		}
	}
}

void hkpShapeDisplayViewer::init()
{
	// We can't check if we are a local viewer until this point as input/output streams have not yet been setup
#if 0
	m_enableInstancing = isLocalViewer();
#endif
	if (m_context)
	{
		for (int i=0; i < m_context->getNumWorlds(); ++i)
		{
			addWorld( m_context->getWorld(i) );
		}
	}
}

hkpShapeDisplayViewer::~hkpShapeDisplayViewer()
{
	m_builder->removeReference();
	int ne = m_worldEntities.getSize();
	for (int i=(ne-1); i >= 0; --i) // backwards as remove alters array
	{
		removeWorld(i);
	}
}

void hkpShapeDisplayViewer::worldRemovedCallback( hkpWorld* world ) 
{ 
	int de = findWorld(world);
	if (de >= 0)
	{	
		removeWorld(de);
	}	
}

//World added listener. Should impl this in sub class, but call up to this one to get the listener reg'd.
void hkpShapeDisplayViewer::worldAddedCallback( hkpWorld* world )
{
	addWorld(world);	
}

void hkpShapeDisplayViewer::addWorld(hkpWorld* world)
{
#ifdef HK_DEBUG
	// For remote connections, we check that the shapeHashUtil is up-to-date here.
	if ( ( m_inStream != HK_NULL ) || ( m_outStream != HK_NULL ) )
	{
		hkpShapeHashUtil::assertShapesUpToDate();
	}
#endif

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

void hkpShapeDisplayViewer::setInstancingEnabled( bool on )
{
#ifdef PLATFORM_SUPPORTS_INSTANCING
	m_enableInstancing = on; 
#endif
}

void hkpShapeDisplayViewer::setAutoColorMode( bool on )
{
	m_enableAutoColor = on; 
}


void hkpShapeDisplayViewer::setDisplayBodyCachingEnabled( bool on )
{
	m_enableDisplayCaching = on; 
}

void hkpShapeDisplayViewer::setDisplayBodyCreateEnabled( bool on )
{
	m_enableDisplayCreation = on;
}

void hkpShapeDisplayViewer::setAutoGeometryCreation( bool on )
{
	m_autoGeometryCreation = on;
}

static const hkpShape* _getProxyShape( hkpEntity* entity, bool forLocal, bool& ignoreWholeObject )
{
	const hkpShape* shapeInProperty = reinterpret_cast<const hkpShape*>( entity->getProperty( HK_PROPERTY_DISPLAY_SHAPE ).getPtr() );
	const hkpShape* vdbShape = reinterpret_cast<const hkpShape*>( entity->getProperty( HK_PROPERTY_VDB_DISPLAY_PTR ).getPtr() );
	const hkpShape* localViewerShape  = reinterpret_cast<const hkpShape*>( entity->getProperty( HK_PROPERTY_DISPLAY_PTR ).getPtr() );

	// The shape this viewer will return, which will be set to one of the above or null.
	const hkpShape* displayShape = HK_NULL;

	ignoreWholeObject = false;
	if (forLocal)
	{
		// Local viewer (not a VDB viewer). 
		if (entity->hasProperty(HK_PROPERTY_DISPLAY_PTR))
		{
			if (localViewerShape == HK_NULL)
			{
				ignoreWholeObject = true;
				return HK_NULL; // Asked for Null local display
			}
			displayShape = localViewerShape;
		}
	}

	if (displayShape == HK_NULL)
	{
		if (entity->hasProperty(HK_PROPERTY_VDB_DISPLAY_PTR))
		{
			if (vdbShape == HK_NULL)
			{
				ignoreWholeObject = true;
				return HK_NULL; // Asked for Null vdb display (will apply to local too unless Local specified separately)
			}
			displayShape = vdbShape;
		}
		else if (entity->hasProperty(HK_PROPERTY_DISPLAY_SHAPE))
		{
			if (shapeInProperty == HK_NULL)
			{
				ignoreWholeObject = true;
				return HK_NULL; // Asked for Null shape prop 
			}
			displayShape = shapeInProperty;
		}
	}
	
	return displayShape;
}

void hkpShapeDisplayViewer::entityShapeSetCallback( hkpEntity* entity)
{
	entityRemovedCallback(entity);
	entityAddedCallback(entity);
}

void hkpShapeDisplayViewer::entityAddedCallback( hkpEntity* entity )
{
	HK_TIME_CODE_BLOCK("hkpShapeDisplayViewer", this);

	// Check if the entity should be displayed by this viewer
	if (!m_isDisplayable(entity))
	{
		return;
	}

	// if we are a local viewer (not a vdb viewer) then we 
	// will ignore the bodies that have a display ptr already or a zero HK_PROPERTY_DISPLAY_PTR	
	bool ignoreObject;
	const hkpShape* displayShape = _getProxyShape(entity, isLocalViewer(), ignoreObject);
	if (ignoreObject) 
	{
		return;
	}

	// figure out the right world list for it
	// We should have the world in our list
	const hkpWorld* world = entity->getWorld();
	int index = findWorld(world);
	if (index < 0)
	{
		return;
	}

	WorldToEntityData* wed = m_worldEntities[index];

	
	hkpRigidBody* rigidBody = static_cast<hkpRigidBody*>(entity);

	const hkpShape* shape = displayShape;
	if (!shape)
	{
		shape = rigidBody->getCollidable()->getShape();
	}
	if (!shape)
	{
		return; // nothing to build from
	}

	bool displayGeomAlreadyCreated = false;
	if ( m_enableDisplayCaching )
	{
		ShapeMap::Iterator i = m_cachedShapes.findOrInsertKey( shape, 0 );
		hkUlong shapeAlreadyUsed = m_cachedShapes.getValue( i );
		displayGeomAlreadyCreated = ( shapeAlreadyUsed > 0 );
		if ( shapeAlreadyUsed == 0 )
		{
			m_cachedShapes.setValue( i, 1 );
		}
	}

	hkUlong instanceID = 0;
	if (m_enableInstancing)
	{
		instanceID = m_instancedShapeToGeomID.getWithDefault( shape, 0);
	}

	// Has a geometry been stored in the body's properties?
	hkDisplayGeometry* displayGeometryForStoredGeometry = HK_NULL;
	// If there is a stored geometry, should we keep it, or delete it after addition?
	bool keepStoredGeometry = false;
	// Do geometries need to be built from the shape?
	bool needToBuildGeometries = false;
	// The display handler can handle hashes, and the shape has a hash.
	bool hasHash = false;
	hkDebugDisplayHandler::Hash hash;

	if (!instanceID && !displayGeomAlreadyCreated && m_enableDisplayCreation && m_autoGeometryCreation)
	{
		hkGeometry* sharedGeom = reinterpret_cast<hkGeometry*>(rigidBody->getProperty(HK_PROPERTY_OVERRIDE_DEBUG_DISPLAY_GEOMETRY_NO_DELETE).getPtr());
		hkGeometry* onceOffGeom = reinterpret_cast<hkGeometry*>(rigidBody->getProperty(HK_PROPERTY_OVERRIDE_DEBUG_DISPLAY_GEOMETRY).getPtr());
		// Is there a stored geometry?
		if (sharedGeom || onceOffGeom )
		{
			keepStoredGeometry = sharedGeom ? true : false;
			if (onceOffGeom)
			{
				rigidBody->removeProperty( HK_PROPERTY_OVERRIDE_DEBUG_DISPLAY_GEOMETRY );
			}
			displayGeometryForStoredGeometry = new hkDisplayConvex(sharedGeom? sharedGeom : onceOffGeom);
		} 
		else
		{
			if ( !isLocalViewer() && m_displayHandler->doesSupportHashes() )
			{
				hkpShapeHashUtil hashUtil( HK_SHAPE_DISPLAY_HASH_START_VALUE );
				if ( rigidBody->hasProperty( HK_PROPERTY_OVERRIDE_SHAPE_HASH ) )
				{
					hashUtil.writeUint64( rigidBody->getProperty( HK_PROPERTY_OVERRIDE_SHAPE_HASH ).m_data );
				}
				else
				{
					hashUtil.writeShape( shape, hkpShapeHashUtil::IGNORE_CONVEX_RADIUS );
				}
				hasHash = hashUtil.getHash( hash );
			}

			// Either we're a local viewer, or there was no hash for the shape.
			if ( !hasHash )
			{
				needToBuildGeometries = true;
			}

			
			const hkpShape* shapeInProperty = reinterpret_cast<const hkpShape*>( entity->getProperty( HK_PROPERTY_DISPLAY_SHAPE ).getPtr() );
			if (shapeInProperty && (shapeInProperty == displayShape))
			{
				shapeInProperty->removeReference();
				entity->removeProperty( HK_PROPERTY_DISPLAY_SHAPE );
			}
		}
	}

	// send the display geometries off to the display handler
	hkUlong id = getId( rigidBody ); 
	wed->entitiesCreated.pushBack( id );
	if (!instanceID && ( ( displayGeometryForStoredGeometry || displayGeomAlreadyCreated || needToBuildGeometries ) || !m_autoGeometryCreation /* precreated */ ) )
	{
		if ( displayGeometryForStoredGeometry )
		{
			// Add the display geometry
			hkInplaceArray<hkDisplayGeometry*, 1> arrayOfGeometries;
			arrayOfGeometries.pushBackUnchecked( displayGeometryForStoredGeometry );
			m_displayHandler->addGeometry( arrayOfGeometries, rigidBody->getTransform(), id, m_tag, (hkUlong)shape );
			// Tidy up.
			if ( keepStoredGeometry )
			{
				displayGeometryForStoredGeometry->releaseGeometry();
			}
			displayGeometryForStoredGeometry->removeReference();
		}
		else if ( displayGeomAlreadyCreated )
		{
			const hkArrayBase<hkDisplayGeometry*> dummyArrayOfGeometries;
			m_displayHandler->addGeometry( dummyArrayOfGeometries, rigidBody->getTransform(), id, m_tag, (hkUlong)shape );
		}
		else if ( needToBuildGeometries )
		{
			// A geometry needs to be built. 
			// We add the geometry lazily so that the handler can opt to build it in parts.
			m_displayHandler->addGeometryLazily( shape, m_builder, rigidBody->getTransform(), id, m_tag, (hkUlong)shape );
		}
		
		if (m_enableInstancing)
		{
			m_instancedShapeToGeomID.insert( shape, id ); 
			m_instancedShapeToUsageCount.insert( shape, 1 );
		}

		hkColor::Argb color = rigidBody->getProperty( HK_PROPERTY_DEBUG_DISPLAY_COLOR ).getInt();
		if ( m_enableAutoColor && ( m_enableAutoColorOverride || (0 == color)) )
		{
			color = rigidBody->isFixed() ? m_fixedObjectColor : m_movableObjectColor;
		}
		if (color)
		{
			m_displayHandler->setGeometryColor( color, id, getProcessTag() );
		}
	}
	else if (instanceID)
	{
		hkUlong numUsingInstance = m_instancedShapeToUsageCount.getWithDefault( shape, 1 );
		numUsingInstance++;
		m_instancedShapeToUsageCount.insert( shape, numUsingInstance );
		m_displayHandler->addGeometryInstance( instanceID, rigidBody->getTransform(), id, getProcessTag() , (hkUlong)shape);

	}

	// Send the hash to the vdb.
	if ( hasHash )
	{
		const hkTransform& transform = rigidBody->getTransform();
		// Get the aabb from the shape directly.
		hkAabb aabb;
		{
			shape->getAabb(transform, 0.0f, aabb);
		}

		hkColor::Argb color = rigidBody->getProperty( HK_PROPERTY_DEBUG_DISPLAY_COLOR ).getInt();
		if ( m_enableAutoColor && ( m_enableAutoColorOverride || (0 == color)) )
		{
			color = rigidBody->isFixed() ? m_fixedObjectColor : m_movableObjectColor;
		}

		m_displayHandler->addGeometryHash( shape, m_builder, hash, aabb, color, transform, id, m_tag );
	}
}

void hkpShapeDisplayViewer::entityRemovedCallback( hkpEntity* entity )
{
	bool ignoreObject;	
	const hkpShape* displayShape = _getProxyShape(entity, isLocalViewer(), ignoreObject);
	//if (ignoreObject) return;

	hkpWorld* world = entity->getWorld();
	int worldIndex = findWorld(world);
	if (worldIndex < 0)
	{
		return;
	}

	WorldToEntityData* wed = m_worldEntities[worldIndex];

	hkpRigidBody* rigidBody = static_cast<hkpRigidBody*>(entity);
	const hkpShape* shape = displayShape;
	if (!shape)
	{
		shape = rigidBody->getCollidable()->getShape();
	}
	
	hkUlong id = getId( rigidBody );

	// remove the geometry from the displayHandler
	m_displayHandler->removeGeometry(id, m_tag, (hkUlong)shape);

	if (m_enableInstancing && shape)
	{
		hkUlong currentNum = m_instancedShapeToUsageCount.getWithDefault( shape, 0);
		currentNum--;
		if (currentNum == 0)
		{
			m_instancedShapeToGeomID.remove( shape ); // no longer an instanced display for it.
			m_instancedShapeToUsageCount.remove( shape );
		}
		else
		{
			m_instancedShapeToUsageCount.insert( shape, currentNum );
		}
	}

	// remove the id from the list of 'owned' created entities
	const int index = wed->entitiesCreated.indexOf(id);
	if(index >= 0)
	{
		wed->entitiesCreated.removeAt(index);
	}
}

void hkpShapeDisplayViewer::clearCache()
{
	m_cachedShapes.clear();
}


void hkpShapeDisplayViewer::synchronizeTransforms(hkDebugDisplayHandler* displayHandler, hkpWorld* world )
{
	hkReal timeForDisplay = m_timeForDisplay > 0 ? m_timeForDisplay : world->getCurrentTime() ;

	displayHandler->lockForUpdate();

	// update the transform for all active entities (in all the active simulation islands)
	{
		const hkArray<hkpSimulationIsland*>& activeIslands = world->getActiveSimulationIslands();

		for(int i = 0; i < activeIslands.getSize(); i++)
		{
			const hkArray<hkpEntity*>& activeEntities = activeIslands[i]->getEntities();
			for(int j = 0; j < activeEntities.getSize(); j++)
			{
				hkpRigidBody* rigidBody = static_cast<hkpRigidBody*>(activeEntities[j]);
				hkUlong id = getId( rigidBody );
				
				hkTransform transform;
				rigidBody->approxTransformAt( timeForDisplay, transform );

				displayHandler->updateGeometry(transform, id, m_tag);

/*
				const char* name = rigidBody->getName();
				if ( name )
				{
					displayHandler->display3dText( name, transform.getTranslation(), hkColor::WHITE, m_tag );
				}
*/
			}
		}
	}

	// update the transform for all inactive entities (in all the inactive simulation islands)
	if(0)
	{
		const hkArray<hkpSimulationIsland*>& inactiveIslands = world->getInactiveSimulationIslands();

		for(int i = 0; i < inactiveIslands.getSize(); i++)
		{
			const hkArray<hkpEntity*>& inactiveEntities = inactiveIslands[i]->getEntities();
			for(int j = 0; j < inactiveEntities.getSize(); j++)
			{
				hkpRigidBody* rigidBody = static_cast<hkpRigidBody*>(inactiveEntities[j]);
				hkUlong id = getId( rigidBody );
				displayHandler->updateGeometry(rigidBody->getTransform(), id, m_tag);
			}
		}
	}

	// update the transform for all fixed entities 
	if(0)
	{
		const hkArray<hkpEntity*>& fixedEntities = world->getFixedIsland()->getEntities();
		for(int j = 0; j < fixedEntities.getSize(); j++)
		{
			hkpRigidBody* rigidBody = static_cast<hkpRigidBody*>(fixedEntities[j]);
			hkUlong id = getId( rigidBody );
			displayHandler->updateGeometry(rigidBody->getTransform(), id, m_tag);
		}
	}

	displayHandler->unlockForUpdate();

}

void hkpShapeDisplayViewer::synchronizeTransforms(hkpWorld* world )
{
	synchronizeTransforms( m_displayHandler, world );
}

void hkpShapeDisplayViewer::postSimulationCallback( hkpWorld* world )
{
	HK_TIME_CODE_BLOCK("hkpShapeDisplayViewer", this);

	if ( !m_enableShapeTransformUpdate)
	{
		return;
	}

	synchronizeTransforms( m_displayHandler, world );
}

void hkpShapeDisplayViewer::removeAllGeometries(int worldIndex)
{
	WorldToEntityData* wed = m_worldEntities[worldIndex];
	for(int i = 0; i < wed->entitiesCreated.getSize(); i++)
	{
		const hkpShape* s = ((hkpCollidable*)wed->entitiesCreated[i])->getShape();
		m_displayHandler->removeGeometry(wed->entitiesCreated[i], m_tag, (hkUlong)s);
	}
	wed->entitiesCreated.setSize(0);
}

int hkpShapeDisplayViewer::findWorld( const hkpWorld* world )
{
	int ne = m_worldEntities.getSize();
	for (int i=0; i < ne; ++i)
	{
		if (m_worldEntities[i]->world == world)
			return i;
	}
	return -1;
}

void hkpShapeDisplayViewer::removeWorld( int i )
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

void hkpShapeDisplayViewer::inactiveEntityMovedCallback( hkpEntity* entity )
{
	hkpRigidBody* rigidBody = static_cast<hkpRigidBody*>(entity);
	hkUlong id = getId( rigidBody );
	m_displayHandler->updateGeometry(rigidBody->getTransform(), id, getProcessTag());
}

hkBool HK_CALL hkpShapeDisplayViewer::isTriggerVolume( hkpEntity* entity )
{
	return entity->hasProperty(HK_PROPERTY_TRIGGER_VOLUME);
}

hkBool HK_CALL hkpShapeDisplayViewer::isShape( hkpEntity* entity )
{
	return !entity->hasProperty(HK_PROPERTY_TRIGGER_VOLUME);
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
