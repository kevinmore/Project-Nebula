/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Vehicle/hkpVehicle.h>
#include <Physics2012/Vehicle/Manager/MultithreadedVehicle/hkpVehicleJobs.h>

#if !defined(HK_PLATFORM_SPU)
void hkpVehicleIntegrateJob::setRunsOnSpuOrPpu()
{
#if defined (HK_PLATFORM_HAS_SPU)
 	m_jobSpuType = HK_JOB_SPU_TYPE_DISABLED;
	return;
#endif
}
#endif

void hkpVehicleJobResults::applyForcesFromStep( const hkpVehicleInstance& vehicleInstance )
{
	HK_TIMER_BEGIN( "ApplyVehicleForces", HK_NULL );

	const int numWheels = vehicleInstance.getNumWheels();
	// Apply suspensions impulses per wheel to ground bodies
	for ( int w_it = 0; w_it < numWheels; ++w_it )
	{
		if ( !m_groundBodyImpulses[w_it].allEqualZero<3>(hkSimdReal::fromFloat(1e-3f)) )
		{
			// Apply impulses to rigid body (not to motion) such that the entity is activated
			vehicleInstance.m_wheelsInfo[w_it].m_contactBody->applyPointImpulse( m_groundBodyImpulses[w_it], vehicleInstance.m_wheelsInfo[w_it].m_hardPointWs );
		}
	}

	// Set vehicle velocity
	// Velocities are applied to hkpRigidBody such that the vehicle will be activated
	hkpRigidBody* chassis = vehicleInstance.getChassis();
	chassis->setAngularVelocity( m_chassisAngularVel );
	chassis->setLinearVelocity( m_chassisLinearVel );

	// Set velocity of ground body per axle
	// Suspension impulses are not applied to these bodies (see above), as the impulses are already applied to the corresponding accumulators
	for ( int ax_it = 0; ax_it < 2; ++ax_it )
	{
		if ( m_groundBodyPtr[ax_it] )
		{
			// Set velocities in rigid body (not in rigid motion) such that the entity is activated
			m_groundBodyPtr[ax_it]->setAngularVelocity( m_groundBodyAngularVel[ax_it] );
			m_groundBodyPtr[ax_it]->setLinearVelocity( m_groundBodyLinearVel[ax_it] );
		}
	}

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
