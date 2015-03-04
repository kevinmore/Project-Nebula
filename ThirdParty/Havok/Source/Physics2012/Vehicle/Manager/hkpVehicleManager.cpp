/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Vehicle/hkpVehicle.h>
#include <Physics2012/Vehicle/Manager/hkpVehicleManager.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>
#include <Physics2012/Vehicle/Manager/MultithreadedVehicle/hkpVehicleJobs.h>

hkpVehicleManager::~hkpVehicleManager()
{
	// Surrender any references held to vehicleInstances.
	const int numVehicles = m_registeredVehicles.getSize();
	for ( int i = 0; i < numVehicles; ++i )
	{
		m_registeredVehicles[i]->removeReference();
	}
}

void hkpVehicleManager::addVehicle( hkpVehicleInstance* vehicle )
{
	HK_ASSERT2( 0xa299bca0, m_registeredVehicles.indexOf( vehicle ) < 0, "Trying to register the same vehicle twice." );
	HK_ASSERT2( 0xa299bca0, vehicle->getChassis()->isAddedToWorld(), "Trying to register a vehicle which isn't added to a world");
	// Adding action based vehicles to the manager is probably, but not necessarily, an error.
	HK_WARN_ON_DEBUG_IF( vehicle->getWorld(), 0xa299bca1, "Adding a vehicle instance to a manager when it is already added to a world as an action.");
	vehicle->addReference();
	m_registeredVehicles.pushBack( vehicle );
}

void hkpVehicleManager::removeVehicle( hkpVehicleInstance* vehicle )
{
	const int index = m_registeredVehicles.indexOf( vehicle );
	HK_ASSERT2( 0xa299bca0, index >= 0, "Trying to remove a vehicle which is not registered in the manager." );
	m_registeredVehicles.removeAt( index );
	vehicle->removeReference();
}

void hkpVehicleManager::stepVehicles( const hkStepInfo& updatedStepInfo )
{
	hkLocalArray< hkpVehicleInstance* > activeVehicles( m_registeredVehicles.getSize() );
	getActiveVehicles( activeVehicles );

	if ( activeVehicles.getSize() )
	{
		stepVehicleArray( activeVehicles, updatedStepInfo );
	}
}

void hkpVehicleManager::stepVehicleArray( hkArray<hkpVehicleInstance*>& activeVehicles, const hkStepInfo& updatedStepInfo )
{
	hkLocalArray< hkpVehicleWheelCollide::CollisionDetectionWheelOutput > cdInfo( hkpVehicleJobResults::s_maxNumWheels );
	hkpVehicleAerodynamics::AerodynamicsDragOutput aerodynamicsDragInfo;
	hkLocalArray< hkReal > suspensionForceAtWheel( hkpVehicleJobResults::s_maxNumWheels ); 
	hkLocalArray< hkReal > totalLinearForceAtWheel( hkpVehicleJobResults::s_maxNumWheels );

	const int numVehicles = activeVehicles.getSize();
	for ( int i = 0; i < numVehicles; ++i )
	{
		hkpVehicleInstance *const vehicle = activeVehicles[i];
		HK_ASSERT2( 0x148e0b8d, ( vehicle->getNumWheels() <= hkpVehicleJobResults::s_maxNumWheels ), "Max number of wheels used should be set appropriately" );

		cdInfo.setSize( vehicle->getNumWheels() );
		suspensionForceAtWheel.setSize( vehicle->getNumWheels() );
		totalLinearForceAtWheel.setSize( vehicle->getNumWheels() );

		vehicle->updateBeforeCollisionDetection();

		HK_TIMER_BEGIN( "VehicleJob", HK_NULL );

		vehicle->m_wheelCollide->collideWheels( updatedStepInfo.m_deltaTime, vehicle, cdInfo.begin() );

		vehicle->updateComponents( updatedStepInfo, cdInfo.begin(), aerodynamicsDragInfo, suspensionForceAtWheel, totalLinearForceAtWheel );

		vehicle->simulateVehicle( updatedStepInfo, aerodynamicsDragInfo, suspensionForceAtWheel, totalLinearForceAtWheel/*, vehicleResults*/ );

		HK_TIMER_END();
	}
}

void hkpVehicleManager::getActiveVehicles( hkArray< hkpVehicleInstance* >& activeVehicles ) const
{
	for ( int i = 0; i < m_registeredVehicles.getSize(); ++i )
	{
		hkpVehicleInstance* vehicle = m_registeredVehicles[i];
		if ( vehicle->getChassis()->isActive() )
		{
			activeVehicles.pushBack( vehicle );
		}
	}
}

void hkpVehicleManager::addToWorld( hkpWorld* world )
{
	const int numVehicles = getNumVehicles();
	for ( int i = 0; i < numVehicles; ++i )
	{
		getVehicle( i ).addToWorld( world );
	}
}

void hkpVehicleManager::removeFromWorld()
{
	const int numVehicles = getNumVehicles();
	for ( int i = 0; i < numVehicles; ++i )
	{
		getVehicle( i ).removeFromWorld();
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
