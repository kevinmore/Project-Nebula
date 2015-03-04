/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/Viewers/Shape/hknpShapeViewer.h>

#include <Common/Visualize/hkProcessFactory.h>
#include <Common/Visualize/hkVisualDebuggerProtocol.h>
#include <Common/Visualize/Shape/hkDisplayCapsule.h>
#include <Common/Visualize/Shape/hkDisplayConvex.h>
#include <Common/Visualize/Shape/hkDisplaySphere.h>
#include <Common/Visualize/Serialize/hkDisplaySerializeIStream.h>

#include <Physics/Physics/Collide/Shape/Convex/Capsule/hknpCapsuleShape.h>
#include <Physics/Physics/Collide/Shape/Convex/Scaled/hknpScaledConvexShape.h>
#include <Physics/Physics/Collide/Shape/Convex/Sphere/hknpSphereShape.h>
#include <Physics/Physics/Collide/Shape/Composite/Compound/hknpCompoundShape.h>

// Force explicit template instantiation
#include <Common/Base/Container/PointerMap/hkMap.cxx>
template class hkMapBase< hknpShapeViewer::GroupKey, int, hknpShapeViewer::GroupMapOperations >;


/*static*/ int hknpShapeViewer::s_tag = 0;
/*static*/ int hknpShapeViewer::s_tagNoRadius = 0;

void HK_CALL hknpShapeViewer::registerViewer( hkProcessFactory& factory )
{
	s_tag = factory.registerProcess( getName(), create );

	// Register a second time with unexpanded convex radii (workaround for slow building/sending, HNP-294)
	s_tagNoRadius = factory.registerProcess( getLowDetailName(), createWithoutRadius );
}

hkProcess* HK_CALL hknpShapeViewer::create( const hkArray<hkProcessContext*>& contexts )
{
	hknpShapeViewer* viewer = new hknpShapeViewer( contexts );
	viewer->m_tag = s_tag;
	return viewer;
}

hkProcess* HK_CALL hknpShapeViewer::createWithoutRadius( const hkArray<hkProcessContext*>& contexts )
{
	hknpShapeViewer* viewer = new hknpShapeViewer( contexts );
	viewer->m_tag = s_tagNoRadius;
	viewer->setConvexRadiusDisplayMode( hknpShape::CONVEX_RADIUS_DISPLAY_NONE );
	return viewer;
}

void hknpShapeViewer::precreateDisplayGeometryForShape( const hknpShape* shape )
{
	if( !m_precreatedGeometryMap.hasKey( shape ) )
	{
		m_precreatedGeometryMap.insert( shape, m_precreatedGeometries.getSize() );
		hknpShapeUtil::buildShapeDisplayGeometries(
			shape, hkTransform::getIdentity(), hkVector4::getConstant(HK_QUADREAL_1), m_radiusDisplayMode,
			m_precreatedGeometries.expandOne() );
	}
}


hknpShapeViewer::hknpShapeViewer( const hkArray<hkProcessContext*>& contexts )
:	hknpViewer( contexts ),
	m_radiusDisplayMode( hknpShape::CONVEX_RADIUS_DISPLAY_ROUNDED ),
	m_instancingEnabled( true ),
	m_pickedWorld( HK_NULL ),
	m_pickedBodyId( hknpBodyId::invalid() )
{
}

hknpShapeViewer::~hknpShapeViewer()
{
	if ( m_context )
	{
		for ( int i=0; i < m_context->getNumWorlds(); i++ )
		{
			worldRemovedCallback( m_context->getWorld(i) );
		}
	}

	for( int i = m_worldDatas.getSize() - 1; i >= 0 ; i-- )
	{
		removeWorld(i);
	}
	m_worldDatas.clear();

	// Delete any cached display geometries
	for ( int i = 0; i < m_precreatedGeometries.getSize(); i++ )
	{
		for ( int j = 0; j < m_precreatedGeometries[i].getSize(); j++ )
		{
			m_precreatedGeometries[i][j]->removeReference();
		}
	}
}

void hknpShapeViewer::worldAddedCallback( hknpWorld* world )
{
	addWorld( world );
}

void hknpShapeViewer::worldRemovedCallback( hknpWorld* world )
{
	int worldIndex = getWorldIndex( world );
	if ( worldIndex >= 0 )
	{
		removeWorld( worldIndex );
	}
}

void hknpShapeViewer::getConsumableCommands( hkUint8*& commands, int& numCommands )
{
	if( !m_context )
	{
		commands = HK_NULL;
		numCommands = 0;
		return;
	}

	static hkUint8 _cmds[] = {
		hkVisualDebuggerProtocol::HK_PICK_OBJECT,
		hkVisualDebuggerProtocol::HK_DRAG_OBJECT,
		hkVisualDebuggerProtocol::HK_RELEASE_OBJECT
	};

	commands = _cmds;
	numCommands	= 3;
}

void hknpShapeViewer::consumeCommand( hkUint8 command )
{
	switch ( command )
	{
		case hkVisualDebuggerProtocol::HK_PICK_OBJECT:
			{
				hkVector4 worldPosition;
				m_inStream->readQuadVector4( worldPosition );
				hkUint64 id = m_inStream->read64u();
				if( m_inStream->isOk() )
				{
					pickObject( id, worldPosition );
				}
			}
			break;

		case hkVisualDebuggerProtocol::HK_DRAG_OBJECT:
			{
				hkVector4 newWorldPosition;
				m_inStream->readQuadVector4( newWorldPosition );
				if( m_inStream->isOk() )
				{
					dragObject( newWorldPosition );
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


int hknpShapeViewer::addWorld( hknpWorld* world )
{
	HK_ASSERT(0x578d43bc, getWorldIndex( world ) == -1 );

	// Add a tracking structure
	WorldData* worldData = HK_NULL;
	int worldIndex;
	{
		worldData = new WorldData();
		worldData->m_world = world;

		worldIndex = m_worldDatas.indexOf( HK_NULL );
		if( worldIndex != -1 )
		{
			m_worldDatas[ worldIndex ] = worldData;
		}
		else
		{
			worldIndex = m_worldDatas.getSize();
			m_worldDatas.pushBack( worldData );
		}
	}

	// Add existing bodies
	{
		worldData->m_dynamicBodyIds.reserve( world->m_bodyManager.getPeakBodyId().value() + 1 );
		for( hknpBodyIterator it = world->getBodyIterator(); it.isValid(); it.next() )
		{
			const hknpBody& body = it.getBody();
			if( body.isAddedToWorld() && body.m_shape )
			{
				addBody( worldIndex, body.m_id );
			}
		}
	}

	// Listen for changes
	{
		HK_SUBSCRIBE_TO_SIGNAL( world->m_signals.m_bodyAdded, this, hknpShapeViewer );
		HK_SUBSCRIBE_TO_SIGNAL( world->m_signals.m_bodyRemoved, this, hknpShapeViewer );
		HK_SUBSCRIBE_TO_SIGNAL( world->m_signals.m_bodyShapeChanged, this, hknpShapeViewer );
		HK_SUBSCRIBE_TO_SIGNAL( world->m_signals.m_staticBodyMoved, this, hknpShapeViewer );
		HK_SUBSCRIBE_TO_SIGNAL( world->m_signals.m_bodySwitchStaticDynamic, this, hknpShapeViewer );
	}

	return worldIndex;
}

void hknpShapeViewer::removeWorld( int worldIndex )
{
	if ( worldIndex < 0 )
	{
		return;
	}

	if( m_worldDatas[ worldIndex ] )
	{
		hknpWorld* world = m_worldDatas[ worldIndex ]->m_world;
		if( world )
		{
			if( m_pickedWorld == world )
			{
				m_pickedWorld = HK_NULL;
			}

			// Stop listening for changes
			{
				world->m_signals.m_bodyAdded.unsubscribeAll(this);
				world->m_signals.m_bodyRemoved.unsubscribeAll(this);
				world->m_signals.m_bodyShapeChanged.unsubscribeAll(this);
				world->m_signals.m_staticBodyMoved.unsubscribeAll(this);
				world->m_signals.m_bodySwitchStaticDynamic.unsubscribeAll(this);
			}

			// Destroy all body display objects
			{
				hkArray< hknpBodyId >* bodyIdLists[2] = {
					&m_worldDatas[ worldIndex ]->m_dynamicBodyIds,
					&m_worldDatas[ worldIndex ]->m_staticBodyIds
				};

				for( int i=0; i<2; ++i )
				{
					const int numBodies = bodyIdLists[i]->getSize();
					for ( int j = 0; j < numBodies; ++j )
					{
						destroyDisplayObject( worldIndex, (*bodyIdLists[i])[j] );
					}
				}
			}
		}

		// Free the tracking structure
		delete m_worldDatas[ worldIndex ];
		m_worldDatas[ worldIndex ] = HK_NULL;
	}
}

void hknpShapeViewer::addBody( hknpWorld* world, hknpBodyId bodyId )
{
	// Find or add a world tracking structure
	int worldIndex = getWorldIndex( world );
	if( worldIndex == -1 )
	{
		WorldData* worldData = new WorldData();
		worldData->m_world = world;
		worldIndex = m_worldDatas.getSize();
		m_worldDatas.pushBack( worldData );
	}

	addBody( worldIndex, bodyId );
}

void hknpShapeViewer::addBody( int worldIndex, hknpBodyId bodyId )
{
	WorldData* worldData = m_worldDatas[worldIndex];
	hknpWorld* world = worldData->m_world;

	// Ignore invisible / self-managing bodies
	if ( !m_context->getColorScheme()->isBodyVisible(world, bodyId, this) )
	{
		return;
	}

	if( world->m_simulationStage != hknpWorld::SIMULATION_DONE )
	{
		// Queue the body for display object generation.
		// We don't do it here because it would interfere with the step performance.
		worldData->m_bodiesToAdd.pushBack( bodyId );
	}
	else
	{
		// Create the display object now
		const hknpBody& body = world->getBody( bodyId );
		hkArray< hknpBodyId >* bodies = body.isStatic() ? &worldData->m_staticBodyIds : &worldData->m_dynamicBodyIds;
		if( bodies->indexOf( bodyId ) == -1 )
		{
			bodies->pushBack( bodyId );
			createDisplayObject( worldIndex, bodyId );
		}
	}
}

void hknpShapeViewer::removeBody( const hknpWorld* world, hknpBodyId bodyId )
{
	if( m_pickedBodyId == bodyId )
	{
		m_pickedBodyId = hknpBodyId::invalid();
	}

	// Ignore invisible / self-managing bodies
	if ( !m_context->getColorScheme()->isBodyVisible(world, bodyId, this) )
	{
		return;
	}

	int worldIndex = getWorldIndex( world );
	HK_ASSERT( 0x62b01e1, worldIndex != -1 );
	WorldData& worldData = *m_worldDatas[worldIndex];

	// Order the body lists. Usually we will get a hit in the first one,
	// but just in case a motion type change hasn't been propagated we check both.
	hkArray< hknpBodyId >* bodyLists[2];
	{
		const hknpBody& body = world->getBody( bodyId );
		if ( body.isStatic() )
		{
			bodyLists[0] = &worldData.m_staticBodyIds;
			bodyLists[1] = &worldData.m_dynamicBodyIds;
		}
		else
		{
			bodyLists[0] = &worldData.m_dynamicBodyIds;
			bodyLists[1] = &worldData.m_staticBodyIds;
		}
	}

	// Find and remove the body
	for( int i=0; i<2; ++i )
	{
		const int index = bodyLists[i]->indexOf( bodyId );
		if ( index != -1 )
		{
			bodyLists[i]->removeAt( index );
			destroyDisplayObject( worldIndex, bodyId );
			return;
		}
	}

	// Might be pending addition, so check that queue too
	{
		const int index = worldData.m_bodiesToAdd.indexOf( bodyId );
		if( index != -1 )
		{
			worldData.m_bodiesToAdd.removeAt( index );
		}
	}
}

void hknpShapeViewer::refreshBody( const hknpWorld* world, hknpBodyId bodyId, hknpShapeInstanceId* childIds, int numChildIds )
{
	const hkUlong displayId = composeDisplayObjectId( world, bodyId );

	if( childIds && ( numChildIds > 0 ) )
	{
		// Set visibility of compound body children
		const hknpShape* shape = world->getBody( bodyId ).m_shape;
		switch( shape->getType() )
		{
			case hknpShapeType::STATIC_COMPOUND:
			case hknpShapeType::DYNAMIC_COMPOUND:
			{
				const hknpCompoundShape* cmp = static_cast<const hknpCompoundShape*>( shape );
				for( int i=0; i<numChildIds; ++i )
				{
					const bool isActive = ( cmp->getInstance( childIds[i] ).getFlags() & hknpShapeInstance::IS_ENABLED );
					m_displayHandler->setGeometryVisibility( childIds[i].value(), isActive, displayId, m_tag );
				}
			}
			break;

			default:
				break;
		}
	}
	else
	{
		// Refresh whole body
		const int worldIndex = getWorldIndex( world );
		destroyDisplayObject( worldIndex, bodyId );
		createDisplayObject( worldIndex, bodyId );
	}
}

void hknpShapeViewer::refreshAllBodies( const hknpWorld* world, BodyType bodyTypes )
{
	const int worldIndex = getWorldIndex( world );
	HK_ASSERT(0x419fb7e5, worldIndex != -1 );
	WorldData& worldData = *m_worldDatas[worldIndex];

	if ( bodyTypes & DYNAMIC_BODY )
	{
		for ( int i = worldData.m_dynamicBodyIds.getSize()-1; i >= 0; i-- )
		{
			destroyDisplayObject( worldIndex, worldData.m_dynamicBodyIds[i] );
			createDisplayObject( worldIndex, worldData.m_dynamicBodyIds[i] );
		}
	}

	if ( bodyTypes & STATIC_BODY )
	{
		for ( int i = worldData.m_staticBodyIds.getSize()-1; i >= 0; i-- )
		{
			destroyDisplayObject( worldIndex, worldData.m_staticBodyIds[i] );
			createDisplayObject( worldIndex, worldData.m_staticBodyIds[i] );
		}
	}
}

void hknpShapeViewer::onBodyAddedSignal( hknpWorld* world, hknpBodyId bodyId )
{
	addBody( world, bodyId );
}

void hknpShapeViewer::onBodyRemovedSignal( hknpWorld* world, hknpBodyId bodyId )
{
	removeBody( world, bodyId );
}

void hknpShapeViewer::onBodyShapeSetSignal( hknpWorld* world, hknpBodyId bodyId )
{
	removeBody( world, bodyId );
	addBody( world, bodyId );
}

void hknpShapeViewer::onBodySwitchStaticDynamicSignal( hknpWorld* world, hknpBodyId bodyId, bool isStatic )
{
	// Ignore invisible / self-managing bodies
	if ( !m_context->getColorScheme()->isBodyVisible(world, bodyId, this) )
	{
		return;
	}

	const hkUlong displayId = composeDisplayObjectId( world, bodyId );
	if( m_displayIdToGroupMap.hasKey( displayId ) )
	{
		// it is instanced, so we need to fully re-add it (so that it changes color)
		removeBody( world, bodyId );
		addBody( world, bodyId );
	}
	else
	{
		// otherwise just move it to the appropriate list
		{
			const int worldIndex = getWorldIndex( world );
			HK_ASSERT(0x419fb7e7, worldIndex != -1 );
			WorldData& worldData = *m_worldDatas[worldIndex];

			const int staticIndex = worldData.m_staticBodyIds.indexOf( bodyId );
			const int dynamicIndex = worldData.m_dynamicBodyIds.indexOf( bodyId );

			// If it hasn't been added yet, don't try to move it.
			if ( (staticIndex == -1) && ( dynamicIndex == -1 ) )
			{
				return;
			}

			if ( isStatic )
			{
				if( staticIndex == -1 )
				{
					worldData.m_staticBodyIds.pushBack( bodyId );
				}
				if( dynamicIndex != -1 )
				{
					worldData.m_dynamicBodyIds.removeAt( dynamicIndex );
				}
			}
			else
			{
				if( staticIndex != -1 )
				{
					worldData.m_staticBodyIds.removeAt( staticIndex );
				}
				if( dynamicIndex == -1 )
				{
					worldData.m_dynamicBodyIds.pushBack( bodyId );
				}
			}
		}

		// and update the color
		{
			const hkColor::Argb color = m_context->getColorScheme()->getBodyColor( world, bodyId );
			m_displayHandler->setGeometryColor( color, displayId, m_tag );
		}
	}
}

void hknpShapeViewer::onStaticBodyMovedSignal( hknpWorld* world, hknpBodyId bodyId )
{
	const hknpBody& body = world->getBody( bodyId );
	const hkUlong displayId = composeDisplayObjectId( world, bodyId );

	m_displayHandler->updateGeometry( body.getTransform(), displayId, m_tag );
}

void hknpShapeViewer::createDisplayObject( int worldIndex, hknpBodyId bodyId, bool useInstancing )
{
	const hknpWorld* world = m_worldDatas[ worldIndex ]->m_world;
	const hknpBody& body = world->getBody( bodyId );
	const hkUlong displayId = composeDisplayObjectId( worldIndex, bodyId );
	const hkColor::Argb color = m_context->getColorScheme()->getBodyColor( world, bodyId );

	//
	// Try to instance it
	//
	{
		useInstancing &= m_instancingEnabled;

		
		
		useInstancing &= ( body.m_shape->asConvexShape() != HK_NULL );

		if( useInstancing )
		{
			// Create the map key
			GroupKey groupKey;
			{
				groupKey.m_shape = body.m_shape;
				
				groupKey.m_color = color;
			}

			// Get or create the display ID group
			int groupIndex;
			if( m_keyToGroupMap.get( groupKey, &groupIndex ) == HK_FAILURE )
			{
				groupIndex = m_groups.getSize();
				m_groups.expandOne();
				m_keyToGroupMap.insert( groupKey, groupIndex );
			}

			// Add our ID
			hkArray< hkUlong >& displayIds = m_groups[ groupIndex ].m_displayIds;
			HK_ASSERT(0x55cbfc93, displayIds.indexOf( displayId ) == -1 );
			displayIds.pushBack( displayId );
			m_displayIdToGroupMap.insert( displayId, groupIndex );

			// If we already had a display object there, try to instance it
			if( displayIds.getSize() > 1 )
			{
				hkResult res = m_displayHandler->addGeometryInstance( displayIds[0], body.getTransform(), displayId, m_tag, 0 );
				if( res == HK_SUCCESS )
				{
					HK_ON_DEBUG( res = ) m_displayHandler->setGeometryPickable( true, displayId, m_tag );
					HK_ASSERT( 0x24cc9c48, res == HK_SUCCESS );
					return;
				}
			}
		}
	}

	//
	// Otherwise create a new display object
	//
	{
		// Get/create some display geometry(s) for the shape
		hkArray<hkDisplayGeometry*>* displayGeometries;
		hkArray<hkDisplayGeometry*> localDisplayGeometries;
		int precreatedGeometryIndex = m_precreatedGeometryMap.getWithDefault( body.m_shape, -1 );
		if( precreatedGeometryIndex == -1 )
		{
			hknpShapeUtil::buildShapeDisplayGeometries(
				body.m_shape, hkTransform::getIdentity(), hkVector4::getConstant(HK_QUADREAL_1), m_radiusDisplayMode,
				localDisplayGeometries );
			displayGeometries = &localDisplayGeometries;
		}
		else
		{
			displayGeometries = &m_precreatedGeometries[ precreatedGeometryIndex ];
		}

		// Send to the display handler
		m_displayHandler->addGeometry( *displayGeometries, body.getTransform(), displayId, m_tag, 0 );
		m_displayHandler->setGeometryPickable( true, displayId, m_tag );
		m_displayHandler->setGeometryColor( color, displayId, m_tag );

		// Delete any local display geometries
		for ( int j = 0; j < localDisplayGeometries.getSize(); j++ )
		{
			localDisplayGeometries[j]->removeReference();
		}
	}
}

void hknpShapeViewer::setBodyColor( const hknpWorld* world, hknpBodyId bodyId, hkColor::Argb color )
{
	// Set in the color scheme
	m_context->getColorScheme()->setBodyColor( world, bodyId, color );

	//
	// Update display object
	//

	const int worldIndex = getWorldIndex( world );
	const hkUlong displayId = composeDisplayObjectId( worldIndex, bodyId );

	// If it is instanced we must un-instance it to be able to set a unique color
	if( m_displayIdToGroupMap.hasKey( displayId ) )
	{
		destroyDisplayObject( worldIndex, bodyId );
		createDisplayObject( worldIndex, bodyId, false );
	}

	// (Try to) set the color
	m_displayHandler->setGeometryColor( color, displayId, m_tag );
}

void hknpShapeViewer::destroyDisplayObject( int worldIndex, hknpBodyId bodyId )
{
	const hkUlong displayId = composeDisplayObjectId( worldIndex, bodyId );

	// Remove it from the display handler
	m_displayHandler->removeGeometry( displayId, m_tag, 0 );

	// Remove it from its instance group
	int groupIndex = -1;
	if( m_displayIdToGroupMap.get( displayId, &groupIndex ) == HK_SUCCESS )
	{
		int index = m_groups[groupIndex].m_displayIds.indexOf( displayId );
		HK_ASSERT(0x354e9df3, index != -1 );
		m_groups[groupIndex].m_displayIds.removeAt( index );
		m_displayIdToGroupMap.remove( displayId );
	}
}

void hknpShapeViewer::step( hkReal deltaTime )
{
	HK_TIME_CODE_BLOCK( "ShapeViewer", this );

	// Update each world
	for( int wi=0; wi < m_worldDatas.getSize(); ++wi )
	{
		WorldData* data = m_worldDatas[wi];
		if( data )
		{
			// Add any pending bodies
			for( int i = 0; i < data->m_bodiesToAdd.getSize(); ++i )
			{
				const hknpBodyId bodyId = data->m_bodiesToAdd[i];
				const hknpBody& body = data->m_world->getBody( bodyId );

				hkArray< hknpBodyId >* bodies = body.isStatic() ? &data->m_staticBodyIds : &data->m_dynamicBodyIds;
				if( bodies->indexOf( bodyId ) == -1 )
				{
					bodies->pushBack( bodyId );
					createDisplayObject( wi, bodyId );
				}
			}
			data->m_bodiesToAdd.clear();

			// Update transform of dynamic bodies
			for( int i = 0; i < data->m_dynamicBodyIds.getSize(); ++i )
			{
				const hknpBodyId bodyId = data->m_dynamicBodyIds[i];
				const hkUlong displayId = composeDisplayObjectId( wi, bodyId );
				const hknpBody& body = data->m_world->getBody( bodyId );
				m_displayHandler->updateGeometry( body.getTransform(), displayId, m_tag );
			}

			
			// Update transform of (potentially) instanced static bodies,
			// because hkgInstancedDisplayObject expects as many calls to updateGeometry() as there are instances
			if( m_instancingEnabled )
			{
				for( int i = 0; i < data->m_staticBodyIds.getSize(); ++i )
				{
					const hknpBodyId bodyId = data->m_staticBodyIds[i];
					const hkUlong displayId = composeDisplayObjectId( wi, bodyId );
					const hknpBody& body = data->m_world->getBody( bodyId );
					m_displayHandler->updateGeometry( body.getTransform(), displayId, m_tag );
				}
			}
		}
	}

	// Apply mouse spring
	if( m_pickedWorld && m_pickedBodyId.isValid() && m_pickedWorld->isBodyAdded( m_pickedBodyId ) )
	{
		const hknpBody* body = &m_pickedWorld->getBody( m_pickedBodyId );

		if( body->isStaticOrKeyframed() )
		{
			return;
		}

		if( body->isInactive() )
		{
			// Request an activation. We can't modify the motion until it is active.
			m_pickedWorld->activateBody( m_pickedBodyId );
			return;
		}

		const hkReal springElasticity = 0.1f;
		const hkReal springDamping = .5f;
		const hkReal maxRelativeForce = 250.0f;
		const hkReal velocityGain = 0.95f;

		hknpMotion* dynamicMotion = &m_pickedWorld->accessMotion( body->m_motionId );

		hkVector4 pRb;
		pRb.setTransformedPos( body->getTransform(), m_mouseSpringLocalPosition );

		// calculate an impulse
		hkVector4 impulse;
		{
			hkVector4 ptDiff;
			ptDiff.setSub( pRb, m_mouseSpringWorldPosition );

			hkMatrix3 effMassMatrix;
			dynamicMotion->buildEffMassMatrixAt( pRb, effMassMatrix );

			// calculate the velocity delta
			hkVector4 delta;
			{
				hkVector4 relVel;
				dynamicMotion->getPointVelocity( pRb, relVel );
				hkSimdReal springElasticitySimd; springElasticitySimd.setFromFloat(springElasticity);
				hkSimdReal springDampingSimd;    springDampingSimd.setFromFloat(springDamping);

				delta.setMul( springElasticitySimd * m_pickedWorld->m_solverInfo.m_invDeltaTime, ptDiff );
				delta.addMul( springDampingSimd, relVel );
			}

			impulse._setRotatedDir( effMassMatrix, delta );
			impulse.setNeg<4>( impulse );

			// clip it
			hkSimdReal impulseLen2 = impulse.lengthSquared<3>();
			hkSimdReal mass; mass.setReciprocal<HK_ACC_23_BIT, HK_DIV_SET_HIGH>(dynamicMotion->getInverseMass());
			hkSimdReal maxRelativeForceSimd; maxRelativeForceSimd.setFromFloat( maxRelativeForce );
			hkSimdReal maxImpulse = mass * m_pickedWorld->m_solverInfo.m_deltaTime * maxRelativeForceSimd;
			if ( impulseLen2 > (maxImpulse*maxImpulse) )
			{
				hkSimdReal factor= maxImpulse * impulseLen2.sqrtInverse();
				impulse.mul( factor );
			}
		}

		// apply velocity damping
		{
			hkSimdReal gain; gain.setFromFloat( velocityGain );

			hkVector4 linearVelocity;
			linearVelocity = dynamicMotion->getLinearVelocity();
			linearVelocity.mul(gain);
			dynamicMotion->setLinearVelocity(linearVelocity);

			hkVector4 angularVelocity;
			dynamicMotion->getAngularVelocity(angularVelocity);
			angularVelocity.mul(gain);
			dynamicMotion->setAngularVelocity(angularVelocity);
		}

		// apply the impulse
		m_pickedWorld->applyBodyImpulseAt( m_pickedBodyId, impulse, pRb );
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
