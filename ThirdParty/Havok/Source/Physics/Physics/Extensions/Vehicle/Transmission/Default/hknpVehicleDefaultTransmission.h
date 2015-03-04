/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_VEHICLE_DEFAULT_TRANSMISSION_H
#define HKNP_VEHICLE_DEFAULT_TRANSMISSION_H

#include <Physics/Physics/Extensions/Vehicle/Transmission/hknpVehicleTransmission.h>

extern const class hkClass hknpVehicleDefaultTransmissionClass;

/// Implements an automatic transmission, with N gears and a reverse gear, different torque ratios for each wheel
/// (so FWD - RWD - AWD can be specified) and clutch delay.
class hknpVehicleDefaultTransmission : public hknpVehicleTransmission
{
	public:

		HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_VEHICLE);
		HK_DECLARE_REFLECTION();

			/// Default constructor
		hknpVehicleDefaultTransmission();

			/// Serialization constructor.
		hknpVehicleDefaultTransmission(hkFinishLoadedObjectFlag f) :
			hknpVehicleTransmission(f), m_gearsRatio(f), m_wheelsTorqueRatio(f) {}

			/// Calculates information about the effects of transmission on the vehicle.
		virtual void calcTransmission(
			const hkReal deltaTime, const hknpVehicleInstance* vehicle, TransmissionOutput& TransmissionOutputOut );

			/// A utility function to calculate the primaryTransmissionRatio for a vehicle. The answer
			/// returned is intended only as a guideline. It will probably need to be tweaked a little
			/// to give the exact desired performance. See the "Tuning tips" of the Havok User Guide
			/// for more details.
			/// The formula is copied here:
			/// TopSpeedCar[m/s] = TopSpeedOfCar[mph] * 1.605 / 3.6
			/// MaxWheelAngularVel = TopSpeedCar[m/s] / wheelRadius
			/// MaxWheelRPM = MaxWheelAngularVel * 60 / 2 * PI
			/// PrimirayTransmissionRatio = maxEngineRPM / MaxWheelRPM
		static hkReal HK_CALL calculatePrimaryTransmissionRatio(
			const hkReal vehicleTopSpeed, const hkReal wheelRadius, const hkReal maxEngineRpm, const hkReal topGearRatio );

		//
		// The following functions are all used by calcTransmission.
		//

			/// Calculates the transmitted torque.
		virtual hkReal calcMainTransmittedTorque(const hknpVehicleInstance* vehicle, TransmissionOutput& transmissionOut) const;

			/// Calculates the transmission RPM.
		virtual hkReal calcTransmissionRPM(const hknpVehicleInstance* vehicle, TransmissionOutput& transmissionOut) const;

			/// Calculates if the vehicle is currently reversing.
		virtual hkBool calcIsReversing(const hknpVehicleInstance* vehicle, TransmissionOutput& transmissionOut) const;

			/// Calculates the current gear.
		virtual void updateCurrentGear( const hkReal deltaTime, const hknpVehicleInstance* vehicle, TransmissionOutput& transmissionOut );

			/// Returns the gear ratio of the current gear.
		virtual hkReal getCurrentRatio(const hknpVehicleInstance* vehicle, TransmissionOutput& transmissionOut) const;

	protected:

#if defined HK_DEBUG

		// Check that the sum of all the wheel torque ratios equals 1.0f. Fire a warning if it does not.
		void checkTotalWheelTorqueRatio( const hknpVehicleInstance* vehicle ) const;

#endif

	public:

			/// The RPM of the engine the transmission shifts down.
		hkReal m_downshiftRPM;

			/// The RPM of the engine the transmission shifts up. Note: m_upshiftRpm > m_downshiftRpm.
		hkReal m_upshiftRPM;

			/// An extra factor to the gear ratio.
		hkReal m_primaryTransmissionRatio;

			/// The time needed [seconds] to shift a gear.
		hkReal m_clutchDelayTime;

			/// The back gear ratio.
		hkReal m_reverseGearRatio;

			/// The ratio of the forward gears.
		hkArray<hkReal> m_gearsRatio;

			/// The transmission ratio for every wheel
		hkArray<hkReal> m_wheelsTorqueRatio;
};

#endif // HKNP_VEHICLE_DEFAULT_TRANSMISSION_H

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
