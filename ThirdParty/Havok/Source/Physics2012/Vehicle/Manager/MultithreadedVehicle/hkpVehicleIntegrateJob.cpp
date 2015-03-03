/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Vehicle/hkpVehicle.h>
#include <Physics2012/Vehicle/Manager/MultithreadedVehicle/hkpVehicleIntegrateJob.h>
#include <Common/Base/Container/LocalArray/hkLocalArray.h>

hkJobQueue::JobStatus HK_CALL hkVehicleIntegrateJob( hkJobQueue& jobQueue, hkJobQueue::JobQueueEntry& nextJobOut )
{
	const hkpVehicleIntegrateJob& vehicleJob = reinterpret_cast< hkpVehicleIntegrateJob& >( nextJobOut );

	hkVehicleIntegrateImplementation( vehicleJob, const_cast< hkpVehicleCommand* >( vehicleJob.m_commandArray ) );

	hkJobQueue::JobStatus status = jobQueue.finishJobAndGetNextJob( &nextJobOut, nextJobOut, hkJobQueue::WAIT_FOR_NEXT_JOB );

	return status;
}

void hkVehicleIntegrateImplementation( const hkpVehicleIntegrateJob& vehicleJob, hkpVehicleCommand* commandsBase )
{
	HK_TIMER_BEGIN( "VehicleJob", HK_NULL );

	hkInplaceArray< hkpVehicleWheelCollide::CollisionDetectionWheelOutput, hkpVehicleInstance::s_maxNumLocalWheels > cdInfo;
	hkpVehicleInstance* vehiclePtr;

	hkpVehicleAerodynamics::AerodynamicsDragOutput aerodynamicsDragInfo;
	hkInplaceArray< hkReal, hkpVehicleInstance::s_maxNumLocalWheels > suspensionForceAtWheel; 
	hkInplaceArray< hkReal, hkpVehicleInstance::s_maxNumLocalWheels > totalLinearForceAtWheel;

	hkpVehicleCommand* command = commandsBase;
	hkpVehicleInstance** vehiclePtrPtr = vehicleJob.m_vehicleArrayPtr;

	for ( int v_it = 0; v_it < vehicleJob.m_numCommands; ++v_it )
	{
		vehiclePtr = (*vehiclePtrPtr);

		HK_ASSERT2( 0x155bffe2, vehiclePtr->getChassis()->isAddedToWorld(), "Vehicle chassis is not added to world.");

		cdInfo.setSize( vehiclePtr->getNumWheels() );
		suspensionForceAtWheel.setSize( vehiclePtr->getNumWheels() );
		totalLinearForceAtWheel.setSize( vehiclePtr->getNumWheels() );

		vehiclePtr->m_wheelCollide->collideWheels( vehicleJob.m_stepInfo.m_deltaTime, vehiclePtr, cdInfo.begin() );

		vehiclePtr->updateComponents( vehicleJob.m_stepInfo, cdInfo.begin(), aerodynamicsDragInfo, suspensionForceAtWheel, totalLinearForceAtWheel );

		vehiclePtr->simulateVehicle( vehicleJob.m_stepInfo, aerodynamicsDragInfo, suspensionForceAtWheel, totalLinearForceAtWheel, command->m_jobResults );

		vehiclePtrPtr++;
		command++;
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
