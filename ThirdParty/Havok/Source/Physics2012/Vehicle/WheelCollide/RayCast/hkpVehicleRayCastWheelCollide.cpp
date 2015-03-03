/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Vehicle/hkpVehicle.h>

#include <Physics2012/Vehicle/hkpVehicleInstance.h>
#include <Physics2012/Vehicle/WheelCollide/RayCast/hkpVehicleRayCastWheelCollide.h>
#include <Physics2012/Collide/Filter/hkpCollisionFilter.h>

#include <Physics2012/Dynamics/Phantom/hkpAabbPhantom.h>
#include <Physics2012/Collide/Query/CastUtil/hkpWorldRayCastInput.h>
#include <Physics2012/Collide/Query/CastUtil/hkpWorldRayCastOutput.h>
#include <Physics2012/Collide/Query/Multithreaded/RayCastQuery/hkpRayCastQueryJobs.h>

hkpVehicleRayCastWheelCollide::hkpVehicleRayCastWheelCollide()
:	m_wheelCollisionFilterInfo( 0 ),
	m_phantom( HK_NULL )
{
	m_alreadyUsed = false;
	m_type = RAY_CAST_WHEEL_COLLIDE;
}


void hkpVehicleRayCastWheelCollide::init( const hkpVehicleInstance* vehicle )
{
	HK_ASSERT2(0x70660634, m_phantom == HK_NULL, "init() will create a new phantom. The phantom must be HK_NULL when init is called");

	// Initialize the phantom
	hkAabb aabb;
	{
		calcWheelsAABB( vehicle, aabb );
	}

	m_phantom = new hkpAabbPhantom( aabb, m_wheelCollisionFilterInfo );

	m_rejectRayChassisListener.m_chassis = vehicle->getChassis()->getCollidable();
	m_phantom->addPhantomOverlapListener( &m_rejectRayChassisListener );
}


void hkpVehicleRayCastWheelCollide::getPhantoms( hkArray<hkpPhantom*>& phantomsOut )
{
	phantomsOut.pushBack( m_phantom );
}


hkpVehicleWheelCollide* hkpVehicleRayCastWheelCollide::clone( const hkpRigidBody& newChassis, const hkArray<hkpPhantom*>& newPhantoms ) const
{
	hkpVehicleRayCastWheelCollide* newC = new hkpVehicleRayCastWheelCollide();
	HK_ASSERT2( 0x1ffe7ea3, newPhantoms[0]->getType() == HK_PHANTOM_AABB, "Phantom of wrong phantom type provided to clone." );
	newC->m_phantom = (hkpAabbPhantom*) newPhantoms[0];
	newC->m_phantom->addReference();

	newC->m_phantom->removePhantomOverlapListener(const_cast<hkpRejectChassisListener*>(&this->m_rejectRayChassisListener));
	newC->m_phantom->addPhantomOverlapListener(&newC->m_rejectRayChassisListener);
	newC->m_rejectRayChassisListener.m_chassis = newChassis.getCollidable();
	newC->m_wheelCollisionFilterInfo = m_wheelCollisionFilterInfo;

	return newC;
}


hkpVehicleRayCastWheelCollide::~hkpVehicleRayCastWheelCollide()
{
	// Vehicle has a phantom yet? 
	if( m_phantom != HK_NULL ) 
	{
		// can't remove m_rejectRayChassisListener from phantom's overlap listeners list
		// because if the objects were serialized they can already be destroyed (cleanup procedure)
		// this is suggested as a temporary solution, need to find a better way
		if( m_rejectRayChassisListener.m_chassis != HK_NULL )
		{
			m_phantom->removePhantomOverlapListener( &m_rejectRayChassisListener );
		}
		m_phantom->removeReference();
	}
}


void hkpVehicleRayCastWheelCollide::calcWheelsAABB( const hkpVehicleInstance* vehicle, hkAabb& aabbOut ) const
{
	aabbOut.m_min.setMin( vehicle->m_wheelsInfo[0].m_hardPointWs, vehicle->m_wheelsInfo[0].m_rayEndPointWs );
	aabbOut.m_max.setMax( vehicle->m_wheelsInfo[0].m_hardPointWs, vehicle->m_wheelsInfo[0].m_rayEndPointWs );

	for (int w_it=1; w_it<vehicle->m_data->m_numWheels; w_it ++)
	{
		const hkpVehicleInstance::WheelInfo &wheel_info = vehicle->m_wheelsInfo[w_it];
		aabbOut.m_min.setMin( aabbOut.m_min, wheel_info.m_rayEndPointWs );
		aabbOut.m_min.setMin( aabbOut.m_min, wheel_info.m_hardPointWs );
		aabbOut.m_max.setMax( aabbOut.m_max, wheel_info.m_rayEndPointWs );
		aabbOut.m_max.setMax( aabbOut.m_max, wheel_info.m_hardPointWs );
	}
}


void hkpVehicleRayCastWheelCollide::updateBeforeCollisionDetection( const hkpVehicleInstance* vehicle )
{
	hkAabb aabb;
	calcWheelsAABB( vehicle, aabb );

	// we know that we are the only user of the phantom right now
	m_phantom->markForWrite();
	m_phantom->setAabb( aabb );
	m_phantom->unmarkForWrite();
}


void hkpVehicleRayCastWheelCollide::castSingleWheel( const hkpVehicleInstance::WheelInfo& wheelInfo, hkpWorldRayCastOutput& output ) const
{
	hkpWorldRayCastInput input;
	input.m_from = wheelInfo.m_hardPointWs;
	input.m_to = wheelInfo.m_rayEndPointWs;
	input.m_enableShapeCollectionFilter = true;
	input.m_filterInfo = m_wheelCollisionFilterInfo;
	m_phantom->castRay( input, output );
}


void hkpVehicleRayCastWheelCollide::collideWheels( const hkReal deltaTime, const hkpVehicleInstance* vehicle, CollisionDetectionWheelOutput* cdInfoOut )
{
	// Check if the phantom has been added to the world.
	HK_ASSERT2(0x676876e3, m_phantom->getWorld() != HK_NULL, "The phantom for the wheelCollide component must be added to the world before using a hkpVehicleInstance.");

	const hkUint8 numWheels = vehicle->m_data->m_numWheels;
	for ( hkUint8 i = 0; i < numWheels; ++i )
	{
		CollisionDetectionWheelOutput& cd_wheelInfo = cdInfoOut[i];
		const hkpVehicleInstance::WheelInfo& wheel_info = vehicle->m_wheelsInfo[i];

		hkpWorldRayCastOutput rayCastOutput;
		castSingleWheel( wheel_info, rayCastOutput );

		if ( rayCastOutput.hasHit() )
		{
			getCollisionOutputFromCastResult( vehicle, i, rayCastOutput, cd_wheelInfo );
		}
		else
		{
			getCollisionOutputWithoutHit( vehicle, i, cd_wheelInfo );
		}
		wheelCollideCallback( vehicle, i, cd_wheelInfo );
	}
}


void hkpVehicleRayCastWheelCollide::getCollisionOutputFromCastResult( const hkpVehicleInstance* vehicle, hkUint8 wheelInfoNum, const hkpWorldRayCastOutput& rayCastOutput, CollisionDetectionWheelOutput& cdInfo ) const
{
	HK_ASSERT2( 0x1ee3f67b, rayCastOutput.hasHit(), "Cannot convert to collision output when there is no hit." );
	
	const hkReal suspensionLength = vehicle->m_suspension->m_wheelParams[wheelInfoNum].m_length;
	const hkpVehicleInstance::WheelInfo& wheel_info = vehicle->m_wheelsInfo[wheelInfoNum];

	cdInfo.m_contactPoint.setNormalOnly( rayCastOutput.m_normal );
	int numKeys = hkMath::min2( (int) hkpShapeRayCastOutput::MAX_HIERARCHY_DEPTH, (int) hkpVehicleInstance::WheelInfo::MAX_NUM_SHAPE_KEYS );
	for( int i = 0; i < numKeys; ++i )
	{
		cdInfo.m_contactShapeKey[i] = rayCastOutput.m_shapeKeys[i];
	}

	hkpRigidBody* groundRigidBody = hkpGetRigidBody( rayCastOutput.m_rootCollidable );
	HK_ASSERT2(0x1f3b75d4,  groundRigidBody, 
		"Your car raycast hit a phantom object. If you don't want this to happen, disable collisions between the wheel raycast phantom and phantoms.\
		\nTo do this, change hkpVehicleRayCastWheelCollide::m_wheelCollisionFilterInfo or hkpCollisionFilter::isCollisionEnabled( const hkpCollidable& a, const hkpCollidable& b )");

	cdInfo.m_contactBody = groundRigidBody; 

	const hkReal wheelRadius = vehicle->m_data->m_wheelParams[wheelInfoNum].m_radius;
	hkReal hitDistance = rayCastOutput.m_hitFraction * ( suspensionLength + wheelRadius );
	cdInfo.m_currentSuspensionLength = hitDistance - wheelRadius;
	
	hkVector4 contactPointWsPosition; contactPointWsPosition.setAddMul( wheel_info.m_hardPointWs, wheel_info.m_suspensionDirectionWs, hkSimdReal::fromFloat(hitDistance) );
	cdInfo.m_contactPoint.setPosition( contactPointWsPosition );
	cdInfo.m_contactPoint.setDistance( cdInfo.m_currentSuspensionLength );
	cdInfo.m_contactFriction = cdInfo.m_contactBody->getMaterial().getFriction();

	// Let theta be the angle between the contact normal and the suspension direction.
	hkSimdReal cosTheta = cdInfo.m_contactPoint.getNormal().dot<3>( wheel_info.m_suspensionDirectionWs );
	HK_ASSERT( 0x66b55978,  cosTheta.isLessZero() );

	if ( cosTheta < -hkSimdReal::fromFloat(vehicle->m_data->m_normalClippingAngleCos) )
	{
		//
		// calculate the suspension velocity
		// 
		hkVector4	 chassis_velocity_at_contactPoint;
		vehicle->getChassis()->getPointVelocity( cdInfo.m_contactPoint.getPosition(), chassis_velocity_at_contactPoint );

		hkVector4 groundVelocityAtContactPoint;
		groundRigidBody->getPointVelocity( cdInfo.m_contactPoint.getPosition(), groundVelocityAtContactPoint );

		hkVector4 chassisRelativeVelocity; chassisRelativeVelocity.setSub( chassis_velocity_at_contactPoint, groundVelocityAtContactPoint );

		hkSimdReal projVel = cdInfo.m_contactPoint.getNormal().dot<3>( chassisRelativeVelocity );

		hkSimdReal inv; inv.setReciprocal(cosTheta); inv = -inv;
		cdInfo.m_suspensionClosingSpeed = (projVel * inv).getReal();
		cdInfo.m_suspensionScalingFactor = inv.getReal();
	}
	else
	{
		cdInfo.m_suspensionClosingSpeed = 0.0f;
		cdInfo.m_suspensionScalingFactor = 1.0f / vehicle->m_data->m_normalClippingAngleCos;
	}
}


void hkpVehicleRayCastWheelCollide::getCollisionOutputWithoutHit( const hkpVehicleInstance* vehicle, hkUint8 wheelNum, CollisionDetectionWheelOutput& cdInfo ) const
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


int hkpVehicleRayCastWheelCollide::buildRayCastCommands( const hkpVehicleInstance* vehicle, const hkpCollisionFilter* collisionFilter, hkInt32 filterSize, hkpShapeRayCastCommand* commandStorage, hkpWorldRayCastOutput* outputStorage ) const
{
	m_phantom->ensureDeterministicOrder();
	const int numOfCollidables = m_phantom->getOverlappingCollidables().getSize();

	// Only issue commands if there are collidables.
	if ( numOfCollidables > 0 )
	{
		const hkInt8 numWheels = vehicle->m_data->m_numWheels;

		for ( hkInt8 i = 0; i < numWheels; ++i, ++commandStorage, ++outputStorage )
		{
			const hkpVehicleInstance::WheelInfo& wheelInfo = vehicle->m_wheelsInfo[i];
			const hkVector4& hardPointWs = wheelInfo.m_hardPointWs;
			const hkVector4& rayEntPointWs = wheelInfo.m_rayEndPointWs;

			HK_ASSERT2( 0x16656fd2, hardPointWs.isOk<3>(), "Input 'from' vector is invalid." );
			HK_ASSERT2( 0x1d9b6f86, rayEntPointWs.isOk<3>(), "Input 'to' vector is invalid." );

#ifdef HK_DEBUG
			//
			//	Check whether our ray is inside the AABB
			//
			{
				hkAabb rayAabb;
				rayAabb.m_min.setMin( hardPointWs, rayEntPointWs );
				rayAabb.m_max.setMax( hardPointWs, rayEntPointWs );
				HK_ASSERT2(0x26c49b67,  m_phantom->getAabb().contains( rayAabb ), "The aabb of the hkpAabbPhantom does not include the ray. Did you forget to call setAabb");
			}
#endif

			// Setup Input
			commandStorage->m_rayInput.m_from = hardPointWs;
			commandStorage->m_rayInput.m_to = rayEntPointWs;
 			commandStorage->m_rayInput.m_filterInfo = m_wheelCollisionFilterInfo;
 			commandStorage->m_rayInput.m_rayShapeCollectionFilter = collisionFilter;
 			commandStorage->m_filterType = collisionFilter->m_type;
 			commandStorage->m_filterSize = filterSize;
			commandStorage->m_useCollector = false;

			// Setup Output
			commandStorage->m_collidables = const_cast<const hkpCollidable**>( m_phantom->getOverlappingCollidables().begin() );
			commandStorage->m_numCollidables = numOfCollidables;
			// We accept only one hit as a result.
			commandStorage->m_resultsCapacity = 1;
			commandStorage->m_numResultsOut = 0;
			commandStorage->m_results = outputStorage;
		}
		return numWheels;
	}
	else
	{
		return 0;
	}
}


void hkpVehicleRayCastWheelCollide::addToWorld( hkpWorld* world )
{
	world->addPhantom( m_phantom );
}


void hkpVehicleRayCastWheelCollide::removeFromWorld()
{
	HK_ASSERT2( 0xe441bc9e, m_phantom->isAddedToWorld(), "Phantom not added to world." );
	hkpWorld* world = m_phantom->getWorld();
	world->removePhantom( m_phantom );
}

void hkpVehicleRayCastWheelCollide::setCollisionFilterInfo( hkUint32 filterInfo )
{
	m_wheelCollisionFilterInfo = filterInfo;
	// If this is called before init, the phantom will not yet exist.
	if ( m_phantom )
	{
		m_phantom->getCollidableRw()->setCollisionFilterInfo( filterInfo );
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
