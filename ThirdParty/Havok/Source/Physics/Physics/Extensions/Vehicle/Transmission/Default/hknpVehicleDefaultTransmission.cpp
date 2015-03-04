/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Extensions/Vehicle/hknpVehicleInstance.h>
#include <Physics/Physics/Extensions/Vehicle/Transmission/Default/hknpVehicleDefaultTransmission.h>

hknpVehicleDefaultTransmission::hknpVehicleDefaultTransmission()
{
	m_downshiftRPM = 0;
	m_upshiftRPM = 0;
	m_primaryTransmissionRatio = 0;
	m_clutchDelayTime = 0;
	m_reverseGearRatio = 0;
}


// Returns the gear ratio of the current gear
hkReal hknpVehicleDefaultTransmission::getCurrentRatio( const hknpVehicleInstance* vehicle, TransmissionOutput& transmissionOut ) const
{
	const hkReal cur_gear_ratio = transmissionOut.m_isReversing ? -m_reverseGearRatio : m_gearsRatio[ transmissionOut.m_currentGear ];

	const hkReal total_ratio = cur_gear_ratio * m_primaryTransmissionRatio;

	return total_ratio;
}

hkReal HK_CALL hknpVehicleDefaultTransmission::calculatePrimaryTransmissionRatio( const hkReal vehicleTopSpeed,
																			   const hkReal wheelRadius,
																			   const hkReal maxEngineRpm,
																			   const hkReal topGearRatio )
{
	HK_ASSERT2(0x2a408c39, vehicleTopSpeed	> 0.0f, "Invalid vehicleTopSpeed passed to hknpVehicleDefaultTransmission::calculatePrimaryTransmissionRatio");
	HK_ASSERT2(0x413737ff, wheelRadius		> 0.0f, "Invalid wheelRadius passed to hknpVehicleDefaultTransmission::calculatePrimaryTransmissionRatio");
	HK_ASSERT2(0x5fdc2f33, maxEngineRpm		> 0.0f, "Invalid maxEngineRpm passed to hknpVehicleDefaultTransmission::calculatePrimaryTransmissionRatio");
	HK_ASSERT2(0x79faa2d6, topGearRatio		> 0.0f, "Invalid topGearRatio passed to hknpVehicleDefaultTransmission::calculatePrimaryTransmissionRatio");

	const hkReal topSpeedCar = ( vehicleTopSpeed * 1.605f ) / 3.6f;
	const hkReal maxWheelAngularVel = topSpeedCar / wheelRadius;
	const hkReal maxWheelRpm = ( maxWheelAngularVel * 60.0f ) / ( 2.0f * HK_REAL_PI );

	return ( ( maxEngineRpm / maxWheelRpm ) / topGearRatio );
}



void hknpVehicleDefaultTransmission::calcTransmission( const hkReal deltaTime, const hknpVehicleInstance* vehicle, TransmissionOutput& transmissionOut )
{
#if defined HK_DEBUG
	checkTotalWheelTorqueRatio( vehicle );
#endif // #if defined HK_DEBUG

	transmissionOut.m_isReversing = calcIsReversing( vehicle, transmissionOut );
	transmissionOut.m_mainTransmittedTorque = calcMainTransmittedTorque( vehicle, transmissionOut );
	transmissionOut.m_transmissionRPM = calcTransmissionRPM( vehicle, transmissionOut );

	int w_it;
	for (w_it=0; w_it< m_wheelsTorqueRatio.getSize(); w_it++)
	{
		transmissionOut.m_wheelsTransmittedTorque[w_it] = transmissionOut.m_mainTransmittedTorque * m_wheelsTorqueRatio[w_it];
	}

	updateCurrentGear( deltaTime, vehicle, transmissionOut );
}

hkReal hknpVehicleDefaultTransmission::calcMainTransmittedTorque( const hknpVehicleInstance* vehicle, TransmissionOutput& transmissionOut ) const
{
	if (transmissionOut.m_delayed)
	{
		return 0.0f;
	}

	const hkReal engine_torque = vehicle->m_torque;
	const hkReal cur_ratio = getCurrentRatio( vehicle, transmissionOut );

	const hkReal transmitted_torque = engine_torque * cur_ratio;

	return transmitted_torque;
}

hkReal hknpVehicleDefaultTransmission::calcTransmissionRPM( const hknpVehicleInstance* vehicle, TransmissionOutput& transmissionOut ) const
{
#if defined HK_DEBUG
	checkTotalWheelTorqueRatio( vehicle );
#endif // #if defined HK_DEBUG

	const hkReal sec_min = 60.0f;
	const hkReal rev_rad = 1.0f / (2.0f * HK_REAL_PI);

	hkReal average_wheel_rpm = 0.0f;

	int w_it;
	for (w_it=0; w_it< vehicle->m_data->m_numWheels; w_it++)
	{
		const hknpVehicleInstance::WheelInfo &wheel_info = vehicle->m_wheelsInfo[ w_it ];
		const hkReal wheel_rpm = wheel_info.m_noSlipIdealSpinVelocity * sec_min * rev_rad;
		average_wheel_rpm += wheel_rpm * m_wheelsTorqueRatio[w_it];
	}

	const hkReal cur_ratio = getCurrentRatio( vehicle, transmissionOut );
	// RPM cannot be negative
	const hkReal current_rpm = hkMath::max2(hkReal(0.f), average_wheel_rpm * cur_ratio);

	return current_rpm;
}

void hknpVehicleDefaultTransmission::updateCurrentGear( const hkReal deltaTime, const hknpVehicleInstance* vehicle, TransmissionOutput& transmissionOut )
{
	// End clutch delay if time is up
	transmissionOut.m_clutchDelayCountdown -= deltaTime;
	if ((transmissionOut.m_delayed) && ( transmissionOut.m_clutchDelayCountdown <= 0.0f ))
	{
		transmissionOut.m_delayed=false;
	}

	if ( transmissionOut.m_isReversing ) return;

	if ((transmissionOut.m_transmissionRPM < m_downshiftRPM) && (transmissionOut.m_currentGear>0))
	{
		transmissionOut.m_currentGear--;
		// start clutch delay
		transmissionOut.m_clutchDelayCountdown = m_clutchDelayTime;
		transmissionOut.m_delayed=true;
	}

	if ((transmissionOut.m_transmissionRPM>m_upshiftRPM) && ((transmissionOut.m_currentGear+1) < m_gearsRatio.getSize()))
	{
		transmissionOut.m_currentGear++;
		// start clutch delay
		transmissionOut.m_clutchDelayCountdown = m_clutchDelayTime;
		transmissionOut.m_delayed=true;
	}
}

// Tries to reverse; returns false if current gear >1
hkBool hknpVehicleDefaultTransmission::calcIsReversing( const hknpVehicleInstance* vehicle, TransmissionOutput& transmissionOut ) const
{
	if (! vehicle->m_tryingToReverse ) return false;
	if ( transmissionOut.m_currentGear > 0 ) return false;

	return true;
}

#if defined HK_DEBUG
void hknpVehicleDefaultTransmission::checkTotalWheelTorqueRatio( const hknpVehicleInstance* vehicle ) const
{
	hkReal wheelTotalTorque = 0.0f;

	// Sum the wheel ratios.
	for ( int wheelIterator = 0; wheelIterator < m_wheelsTorqueRatio.getSize(); wheelIterator++ )
	{
		wheelTotalTorque += m_wheelsTorqueRatio[wheelIterator];
	}

	// Check wheel ratio total and warn if necessary.
	const hkReal torqueDifference = wheelTotalTorque - 1.0f;
	const hkReal torqueEpsilon = 1e-3f;
	if ( hkMath::fabs( torqueDifference ) > torqueEpsilon )
	{
		HK_WARN( 0x62d0b819, "Total wheel torque ratio differs from recommended value of 1.0f." );
	}
}
#endif // #if defined HK_DEBUG

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
