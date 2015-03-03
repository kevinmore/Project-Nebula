/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Vehicle/hkpVehicle.h>

#include <Physics2012/Vehicle/WheelCollide/LinearCast/hkpVehicleLinearCastWheelCollide.h>

#include <Physics2012/Collide/Query/Multithreaded/CollisionQuery/hkpCollisionQueryJobs.h>
#include <Physics2012/Collide/Query/Collector/PointCollector/hkpRootCdPoint.h>
#include <Physics2012/Collide/Query/CastUtil/hkpLinearCastInput.h>

#include <Physics2012/Collide/Query/Collector/PointCollector/hkpClosestCdPointCollector.h>
#include <Physics2012/Collide/Shape/Convex/Cylinder/hkpCylinderShape.h>

#include <Physics2012/Dynamics/Phantom/hkpAabbPhantom.h>

#include <Common/Base/Container/LocalArray/hkLocalArray.h>

// Debug and vizualization:

#include <Common/Visualize/hkDebugDisplay.h>
#include <Common/Visualize/Shape/hkDisplayBox.h>
#define HK_WHEEL_START_COLOR hkColor::GREEN
#define HK_WHEEL_END_COLOR hkColor::RED
#define HK_WHEEL_CONTACT_COLOR hkColor::WHITE
// Visualize the wheel casts in start and end state
//#define HK_DEBUG_LINEAR_CAST_WHEELS
// Visualize the wheel contact points.
//#define HK_DEBUG_WHEEL_CONTACT_POINTS


hkpVehicleLinearCastWheelCollide::hkpVehicleLinearCastWheelCollide()
:	m_maxExtraPenetration( HK_REAL_EPSILON ),
	m_startPointTolerance( HK_REAL_EPSILON )
{
	m_alreadyUsed = false;
	m_type = LINEAR_CAST_WHEEL_COLLIDE;
}


void hkpVehicleLinearCastWheelCollide::setWheelShapes( const hkpVehicleInstance* vehicle, const hkArray<hkpShape*>& shapes )
{
	const int numWheels = vehicle->getNumWheels();
	HK_WARN_ON_DEBUG_IF( m_wheelStates.getSize(), 0x5efe3917, "Wheel shapes probably already provided." );
	HK_ASSERT2( 0x5efe3917, shapes.getSize() >= numWheels, "Not enough shape pointers provided." );

	// The resizing of the array is taken as an indicating that wheelShapes have been assigned.
	m_wheelStates.setSize( numWheels );

	// Create a wheel shape for each wheel.
	for ( hkUint8 i = 0; i < numWheels; ++i )
	{
		m_wheelStates[i].m_shape = shapes[i];
		shapes[i]->addReference();
	}
}


hkpCylinderShape* hkpVehicleLinearCastWheelCollide::createWheelShape( hkReal width, hkReal radius )
{
	hkVector4 vertA;
	{
		vertA.set( 0.0f, 0.0f, 0.5f * width );
	}
	hkVector4 vertB;
	{
		vertB.setNeg<3>( vertA );
	}
	return new hkpCylinderShape( vertA, vertB, radius, 0.0f );
}


void hkpVehicleLinearCastWheelCollide::init( const hkpVehicleInstance* vehicle )
{
	const int numWheels = vehicle->getNumWheels();
	
	// The wheel states array has zero size unless wheel shapes have been provided.
	if ( m_wheelStates.getSize() == 0 )
	{
		m_wheelStates.setSize( numWheels );
		for ( hkUint8 i = 0; i < numWheels; ++i )
		{
			const hkpVehicleData::WheelComponentParams& wheelParams = vehicle->m_data->m_wheelParams[i];
			m_wheelStates[i].m_shape = createWheelShape( wheelParams.m_width, wheelParams.m_radius );
		}
	}
#ifdef HK_DEBUG
	// In the case of provided cylinder shapes, check they are consistent with the vehicle's parameters.
	else
	{
		HK_ASSERT2( 0x9fee4145, m_wheelStates.getSize() == numWheels, "Wheel states set to the wrong size." );
		for ( hkUint8 i = 0; i < numWheels; ++i )
		{
			const hkpShape *const shape = m_wheelStates[i].m_shape;
			if ( shape->getType() == hkcdShapeType::CYLINDER )
			{
				const hkpCylinderShape *const cylinder = static_cast<const hkpCylinderShape*>( shape );
				const hkReal radius = cylinder->getCylinderRadius();
				const hkReal width = cylinder->getVertex<0>().distanceTo( cylinder->getVertex<1>() ).getReal();
				const hkpVehicleData::WheelComponentParams& wheelParams = vehicle->m_data->m_wheelParams[i];
				HK_ASSERT2( 0x9fee4145, hkMath::fabs( radius - wheelParams.m_radius ) < HK_REAL_EPSILON, "Provided wheel shape has wrong radius." );
				HK_ASSERT2( 0x9fee4145, hkMath::fabs( width - wheelParams.m_width ) < HK_REAL_EPSILON, "Provided wheel shape has wrong width." );
			}
		}
	}
#endif

	m_rejectChassisListener.m_chassis = vehicle->getChassis()->getCollidable();
	
	// Set up the other wheel state parameters.
	for ( hkUint8 i = 0; i < numWheels; ++i )
	{
		// Set the other parameters.
		updateWheelState( vehicle, i );

		// Initialize the phantom
		{
			hkAabb aabb;
			{
				calcAabbOfWheel( vehicle, i, aabb );
			}

			m_wheelStates[i].m_phantom = new hkpAabbPhantom( aabb, m_wheelCollisionFilterInfo );
			m_wheelStates[i].m_phantom->addPhantomOverlapListener( &m_rejectChassisListener );
		}
	}
}


hkpVehicleLinearCastWheelCollide::~hkpVehicleLinearCastWheelCollide()
{
	const int numWheels = m_wheelStates.getSize();
	for ( int i = 0; i < numWheels; ++i )
	{
		WheelState& wheelState = m_wheelStates[i];
		wheelState.m_shape->removeReference();
		hkpAabbPhantom *const phantom = wheelState.m_phantom;
		if ( m_memSizeAndFlags != 0 && phantom->m_memSizeAndFlags != 0)
		{
			// can't remove m_rejectRayChassisListener from phantom's overlap listeners list
			// because if the objects were serialized they can already be destroyed (cleanup procedure)
			// this is suggested as a temporary solution, need to find a better way
			phantom->removePhantomOverlapListener( &m_rejectChassisListener );
		}
		phantom->removeReference();
	}
}


void hkpVehicleLinearCastWheelCollide::collideWheels( const hkReal deltaTime, const hkpVehicleInstance* vehicle, CollisionDetectionWheelOutput* cdInfoOut )
{
	const hkUint8 numWheels = vehicle->m_data->m_numWheels;
	for ( hkUint8 i = 0; i < numWheels; ++i )
	{
		CollisionDetectionWheelOutput& cd_wheelInfo = cdInfoOut[i];

		hkpRootCdPoint linearCastOutput;
		
		if ( castSingleWheel( vehicle, i, linearCastOutput ) )
		{
			getCollisionOutputFromCastResult( vehicle, i, linearCastOutput, cd_wheelInfo );
		}
		else
		{
			getCollisionOutputWithoutHit( vehicle, i, cd_wheelInfo );
		}
		wheelCollideCallback( vehicle, i, cd_wheelInfo );
	}
}


void hkpVehicleLinearCastWheelCollide::updateBeforeCollisionDetection( const hkpVehicleInstance* vehicle )
{
	const int numWheels = m_wheelStates.getSize();

	// Set up the wheel states.
	for ( hkUint8 i = 0; i < numWheels; ++i )
	{
		// Adjust the wheel state parameters.
		updateWheelState( vehicle, i );

		// Adjust the phantom
		hkAabb aabb;
		{
			calcAabbOfWheel( vehicle, i, aabb );
		}

		hkpAabbPhantom *const phantom = m_wheelStates[i].m_phantom;
		phantom->markForWrite();
		phantom->setAabb( aabb );
		phantom->unmarkForWrite();
	}
}


hkpVehicleWheelCollide* hkpVehicleLinearCastWheelCollide::clone( const hkpRigidBody& newChassis, const hkArray<hkpPhantom*>& newPhantoms ) const
{
	hkpVehicleLinearCastWheelCollide* newC = new hkpVehicleLinearCastWheelCollide();

	const int numWheels = m_wheelStates.getSize();
	newC->m_wheelStates.setSize( numWheels );
	
	HK_ASSERT2( 0x1ffe7ea3, newPhantoms.getSize() >= numWheels, "Not enough phantoms provided to clone." );

	for ( hkUint8 i = 0; i < numWheels; ++i )
	{
		const WheelState& oldWheelState = m_wheelStates[i];
		WheelState& newWheelState = newC->m_wheelStates[i];

		// Pointers to the wheel shapes are copied.
		{
			newWheelState.m_shape = oldWheelState.m_shape;
			newWheelState.m_shape->addReference();
		}

		// The provided phantoms are used.
		{
			HK_ASSERT2( 0x1ffe7ea3, newPhantoms[i]->getType() == HK_PHANTOM_AABB, "Phantom of wrong phantom type provided to clone." );
			hkpAabbPhantom* newPhantom = static_cast<hkpAabbPhantom*>( newPhantoms[i] );

			newWheelState.m_phantom = newPhantom;
			
			newPhantom->addReference();

			// Fix up the rejectChassisListeners.
			newPhantom->removePhantomOverlapListener( const_cast<hkpRejectChassisListener*>( &this->m_rejectChassisListener ) );
			newPhantom->addPhantomOverlapListener( &newC->m_rejectChassisListener );
			
			newC->m_rejectChassisListener.m_chassis = newChassis.getCollidable();
		}
	}
	newC->m_wheelCollisionFilterInfo = m_wheelCollisionFilterInfo;
	newC->m_maxExtraPenetration = m_maxExtraPenetration;
	newC->m_startPointTolerance = m_startPointTolerance;

	return newC;
}


#ifdef HK_DEBUG_LINEAR_CAST_WHEELS

// Vizualize the start and end position of the wheels. This currently only works for the cylinder shape.
// Since wheel collision detection is done at the start of the frame and rendering is done
// at the end, the rendered wheels will appear one frame behind.
void debugLinearCastWheel( const hkpVehicleLinearCastWheelCollide::WheelState& wheelState )
{
	const hkpShape *const shape = wheelState.m_shape;
	if ( shape->getType() == hkcdShapeType::CYLINDER )
	{
		const hkpCylinderShape *const wheelShape = static_cast<const hkpCylinderShape*>( shape );

		// Visualization of cylinders is not currently implemented, so we use a box.
		hkVector4 halfExtents;
		{
			const hkReal radius = wheelShape->getCylinderRadius();
			const hkReal thickness = wheelShape->getVertex( 0 ).distanceTo3( wheelShape->getVertex( 1 ) );
			halfExtents.set( radius, radius, 0.5f * thickness );
		}
		hkDisplayBox db( halfExtents );

		hkInplaceArray<hkDisplayGeometry*, 1> geometries;
		geometries.pushBack( &db );

		hkDebugDisplay::getInstance().displayGeometry( geometries, wheelState.m_transform, HK_WHEEL_START_COLOR, 0, 0 );

		hkTransform trans( wheelState.m_transform.getRotation(), wheelState.m_to );

		hkDebugDisplay::getInstance().displayGeometry( geometries, trans, HK_WHEEL_END_COLOR, 0, 0 );
	}
}

#endif


int hkpVehicleLinearCastWheelCollide::getTotalNumCommands() const
{
	int sum = 0;
	const int numCollidables = m_wheelStates.getSize();
	for ( hkUint8 i = 0; i < numCollidables; ++i )
	{
		sum += getNumCommands( i );
	}
	return sum;
}


int hkpVehicleLinearCastWheelCollide::getNumCommands( hkUint8 numWheel ) const
{
	hkpAabbPhantom *const phantom = m_wheelStates[numWheel].m_phantom;
	phantom->ensureDeterministicOrder();
	return phantom->getOverlappingCollidables().getSize();
}


int hkpVehicleLinearCastWheelCollide::buildLinearCastCommands( const hkpVehicleInstance* vehicle, const hkpCollisionFilter* collisionFilter, hkpCollidable* collidableStorage, hkpPairLinearCastCommand* commandStorage, hkpRootCdPoint* outputStorage ) const
{
	int numCommandsIssued = 0;
	const int numWheels = m_wheelStates.getSize();
	for ( hkInt8 i = 0; i < numWheels; ++i )
	{
		const WheelState& wheelState = m_wheelStates[i];
		// Create a collidable to represent the wheel.
		new (collidableStorage) hkpCollidable( wheelState.m_shape, &wheelState.m_transform );
		wheelState.m_phantom->ensureDeterministicOrder();
		const hkArray<hkpCollidable*>& overlappingCollidables = wheelState.m_phantom->getOverlappingCollidables();
		const int numCollidables = overlappingCollidables.getSize();		
		for ( hkInt8 j = 0; j < numCollidables; ++j )
		{
			commandStorage->m_collidableA = collidableStorage;
			commandStorage->m_collidableB = overlappingCollidables[j] ;
			commandStorage->m_from = wheelState.m_transform.getTranslation();
			commandStorage->m_to = wheelState.m_to;
			commandStorage->m_results = outputStorage;
			commandStorage->m_resultsCapacity = 1;
			
			commandStorage->m_numResultsOut = 0;

			// Ignore start points.
			commandStorage->m_startPointResultsCapacity = 0;

			// Move the storage pointers
			++commandStorage;
			++outputStorage;
			// Increment the command counter.
			++numCommandsIssued;
		}
		++collidableStorage;
	}
	return numCommandsIssued;			
}


hkBool hkpVehicleLinearCastWheelCollide::castSingleWheel( const hkpVehicleInstance* vehicle, hkUint8 wheelNum, hkpRootCdPoint& linearCastOutput ) const
{
	const WheelState& wheelState = m_wheelStates[wheelNum];

	hkpCollidable wheel( wheelState.m_shape, &wheelState.m_transform );
	{
		wheel.setCollisionFilterInfo( m_wheelCollisionFilterInfo );
	}

	hkpLinearCastInput input;
	{
		input.m_to = wheelState.m_to;
		input.m_maxExtraPenetration = m_maxExtraPenetration;
		input.m_startPointTolerance = m_startPointTolerance;
	}

#ifdef HK_DEBUG_LINEAR_CAST_WHEELS
	debugLinearCastWheel( wheelState );
#endif

	hkpClosestCdPointCollector collector;
	{
		// CAST THE SHAPE
		wheelState.m_phantom->linearCast( &wheel, input, collector, HK_NULL );
	}
	
	if ( collector.hasHit() )
	{
		linearCastOutput = collector.getHit();

#ifdef HK_DEBUG_WHEEL_CONTACT_POINTS
		const hkContactPoint cp = collector.getHitContact();
		// Don't use the contact points distance to render the contact plane.
		hkVector4 normal = cp.getNormal();
		normal( 3 ) = 0.0f;
		HK_DISPLAY_PLANE( normal, cp.getPosition(), 0.5f, HK_WHEEL_CONTACT_COLOR );
#endif

		return true;
	}
	else
	{
		return false;
	}
}


const hkpRootCdPoint* hkpVehicleLinearCastWheelCollide::determineNearestHit( hkUint8 numWheel, const hkpPairLinearCastCommand* commandStorageOutput ) const
{
	const int numCollidables = getNumCommands( numWheel );
	const hkpRootCdPoint* nearestSoFar = HK_NULL;
	for ( int i = 0; i < numCollidables; ++i )
	{
		const hkpPairLinearCastCommand& command = commandStorageOutput[i];
		// If there was a hit
		if ( command.m_numResultsOut )
		{
			const hkpRootCdPoint *const current = command.m_results;
			// If this hit is the nearest so far.
			if ( !nearestSoFar || ( *current < *nearestSoFar ) )
			{
				nearestSoFar = current;
			}
		}
	}
	return nearestSoFar;
}


void hkpVehicleLinearCastWheelCollide::getCollisionOutputFromCastResult( const hkpVehicleInstance* vehicle, hkUint8 wheelNum, const hkpRootCdPoint& linearCastOutput, CollisionDetectionWheelOutput& output ) const
{
	const hkReal suspensionLength = vehicle->m_suspension->m_wheelParams[wheelNum].m_length;
	const hkpVehicleInstance::WheelInfo& wheel_info = vehicle->m_wheelsInfo[wheelNum];

	output.m_contactPoint = linearCastOutput.m_contact;
	output.m_contactBody = hkpGetRigidBody( linearCastOutput.m_rootCollidableB );
	HK_ASSERT2( 0x1f3b75d4,  output.m_contactBody, "The wheel hit a phantom object." );
	output.m_contactFriction = output.m_contactBody->getMaterial().getFriction();
	// The full shape key hierarchy is not available.
	output.m_contactShapeKey[0] = linearCastOutput.m_shapeKeyB;
	output.m_contactShapeKey[1] = HK_INVALID_SHAPE_KEY;
	output.m_currentSuspensionLength = suspensionLength * linearCastOutput.m_contact.getDistance();
	
	// Let theta be the angle between the contact normal and the suspension direction.
	hkSimdReal cosTheta = output.m_contactPoint.getNormal().dot<3>( wheel_info.m_suspensionDirectionWs );
	//HK_ASSERT( 0x66b55978, cosTheta < 0.f );

	if ( cosTheta < -hkSimdReal::fromFloat(vehicle->m_data->m_normalClippingAngleCos) )
	{
		//
		// calculate the suspension velocity
		// 
		hkVector4 chassis_velocity_at_contactPoint;
		vehicle->getChassis()->getPointVelocity(output.m_contactPoint.getPosition(), chassis_velocity_at_contactPoint);

		hkVector4 groundVelocityAtContactPoint;
		output.m_contactBody->getPointVelocity( output.m_contactPoint.getPosition(), groundVelocityAtContactPoint);

		hkVector4 chassisRelativeVelocity; chassisRelativeVelocity.setSub( chassis_velocity_at_contactPoint, groundVelocityAtContactPoint);

		hkSimdReal projVel = output.m_contactPoint.getNormal().dot<3>( chassisRelativeVelocity );

		hkSimdReal inv; inv.setReciprocal(cosTheta); inv = -inv;
		output.m_suspensionClosingSpeed = (projVel * inv).getReal();
		output.m_suspensionScalingFactor = inv.getReal();
	}
	else
	{
		output.m_suspensionClosingSpeed = 0.0f;
		output.m_suspensionScalingFactor = 1.0f / vehicle->m_data->m_normalClippingAngleCos;
	}
}


void hkpVehicleLinearCastWheelCollide::getCollisionOutputWithoutHit( const hkpVehicleInstance* vehicle, hkUint8 wheelNum, CollisionDetectionWheelOutput& cdInfo ) const
{
	const hkReal suspensionLength = vehicle->m_suspension->m_wheelParams[wheelNum].m_length;
	const hkpVehicleInstance::WheelInfo& wheel_info = vehicle->m_wheelsInfo[wheelNum];

	cdInfo.m_contactBody = HK_NULL; 
	cdInfo.m_currentSuspensionLength = suspensionLength;
	cdInfo.m_suspensionClosingSpeed = 0.0f;
	cdInfo.m_contactPoint.setPosition( wheel_info.m_rayEndPointWs );

	hkVector4 contactPointWsNormal; contactPointWsNormal.setNeg<4>( wheel_info.m_suspensionDirectionWs );
	cdInfo.m_contactPoint.setNormalOnly( contactPointWsNormal );
	cdInfo.m_contactFriction = 0.0f;
	cdInfo.m_suspensionScalingFactor = 1.0f;
	cdInfo.m_contactPoint.setDistance( suspensionLength );
}


void hkpVehicleLinearCastWheelCollide::calcAabbOfWheel( const hkpVehicleInstance* vehicle, hkUint8 wheelNum, hkAabb& aabbOut ) const
{
	const hkpVehicleData::WheelComponentParams& wheelParam = vehicle->m_data->m_wheelParams[wheelNum];

	hkVector4 halfExtents;
	{
		halfExtents.set( wheelParam.m_radius, wheelParam.m_radius, 0.5f * wheelParam.m_width );
	}

	hkTransform trans( m_wheelStates[wheelNum].m_transform );
	{
		trans.setTranslation( m_wheelStates[wheelNum].m_to );
	}

	hkAabbUtil::calcAabb( trans, halfExtents, hkSimdReal_0, aabbOut );
}


void hkpVehicleLinearCastWheelCollide::updateWheelState( const hkpVehicleInstance* vehicle, hkUint8 wheelNum )
{
	const hkpVehicleInstance::WheelInfo& wheelInfo = vehicle->m_wheelsInfo[wheelNum];
	WheelState& wheelState = m_wheelStates[wheelNum];
	hkQuaternion rotation;
	{
		rotation.setMul( vehicle->getChassis()->getRotation(), wheelInfo.m_steeringOrientationChassisSpace );

		// Get chassis forward orientation relative to its rigid body
		hkRotation& chassisOrientation = vehicle->m_data->m_chassisOrientation;
		hkRotation rot; rot.setCols( chassisOrientation.getColumn<1>(), chassisOrientation.getColumn<0>(), chassisOrientation.getColumn<2>() );
		hkQuaternion orient; orient.set( rot );

		rotation.setMul( rotation, orient );
	}
	wheelState.m_transform.set( rotation, wheelInfo.m_hardPointWs );
	wheelState.m_to.setAddMul( wheelInfo.m_hardPointWs, wheelInfo.m_suspensionDirectionWs, hkSimdReal::fromFloat(vehicle->m_suspension->m_wheelParams[wheelNum].m_length) );
}


void hkpVehicleLinearCastWheelCollide::getPhantoms( hkArray<hkpPhantom*>& phantomsOut )
{
	const int numWheels = m_wheelStates.getSize();
	for ( int i = 0; i < numWheels; ++i )
	{
		phantomsOut.pushBack( m_wheelStates[i].m_phantom );
	}
}


void hkpVehicleLinearCastWheelCollide::addToWorld( hkpWorld* world )
{
	const int numWheels = m_wheelStates.getSize();
	for ( int i = 0; i < numWheels; ++i )
	{
		world->addPhantom( m_wheelStates[i].m_phantom );
	}
}


void hkpVehicleLinearCastWheelCollide::removeFromWorld()
{
	const int numWheels = m_wheelStates.getSize();
	for ( int i = 0; i < numWheels; ++i )
	{
		HK_ASSERT2( 0xe441bc9e, m_wheelStates[i].m_phantom->isAddedToWorld(), "Phantom not added to world." );
		hkpWorld* world = m_wheelStates[i].m_phantom->getWorld();
		world->removePhantom( m_wheelStates[i].m_phantom );
	}
}


void hkpVehicleLinearCastWheelCollide::setCollisionFilterInfo( hkUint32 filterInfo )
{
	m_wheelCollisionFilterInfo = filterInfo;
	const int numWheels = m_wheelStates.getSize();
	for ( int i = 0; i < numWheels; ++i )
	{
		hkpAabbPhantom *const phantom = m_wheelStates[i].m_phantom;
		if ( phantom )
		{
			phantom->getCollidableRw()->setCollisionFilterInfo( filterInfo );
		}
	}	
}


void hkpVehicleLinearCastWheelCollide::wheelCollideCallback( const hkpVehicleInstance* vehicle, hkUint8 wheelIndex, CollisionDetectionWheelOutput& cdInfo )
{
	centerWheelContactPoint( vehicle, wheelIndex, cdInfo );
}


void hkpVehicleLinearCastWheelCollide::centerWheelContactPoint( const hkpVehicleInstance* vehicle, hkUint8 wheelIndex, CollisionDetectionWheelOutput& cdInfo ) const
{
	// Move the contact point position to the center plane of the wheel by translating it to wheel space,
	// zeroing its z coordinate and translating it back to world space.
	const hkTransform& transformToWorld = m_wheelStates[wheelIndex].m_transform;
	hkVector4 positionWheelSpace;
	{
		positionWheelSpace.setTransformedInversePos( transformToWorld, cdInfo.m_contactPoint.getPosition() );
		positionWheelSpace.zeroComponent<2>();
	}
	hkVector4 newPosition;
	{
		newPosition.setTransformedPos( transformToWorld, positionWheelSpace );
	}
	cdInfo.m_contactPoint.setPosition( newPosition );
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
