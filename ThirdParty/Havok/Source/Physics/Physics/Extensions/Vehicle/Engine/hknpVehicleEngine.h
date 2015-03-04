/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_VEHICLE_ENGINE_H
#define HKNP_VEHICLE_ENGINE_H

#include <Physics/Physics/Extensions/Vehicle/DriverInput/hknpVehicleDriverInput.h>
#include <Physics/Physics/Extensions/Vehicle/Transmission/hknpVehicleTransmission.h>

extern const class hkClass hknpVehicleEngineClass;

class hknpVehicleInstance;


/// Responsible for calculating values related to the engine of the vehicle, in particular the engine torque and RPM.
/// A typical hknpVehicleEngine implementation would collaborate with:
///  - the vehicle's hknpVehicleDriverInput (for the accelerator pedal input)
///  - the vehicle's hknpVehicleTransmission (for RPM and torque transmission)
class hknpVehicleEngine : public hkReferencedObject
{
	public:

		/// Container for data output by the engine calculations.
		/// Note that each of these members can be accessed through the hknpVehicleInstance.
		struct EngineOutput
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_VEHICLE, hknpVehicleEngine::EngineOutput );

			hkReal m_torque;	///< The torque currently supplied by the engine.
			hkReal m_rpm;		///< The RPM the engine is currently running at.
		};

	public:

		HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_VEHICLE);
		HK_DECLARE_REFLECTION();

			/// Serialization constructor.
		hknpVehicleEngine(hkFinishLoadedObjectFlag flag) : hkReferencedObject(flag) {}

			/// Sets the current values of the torque and rpm.
		virtual void calcEngineInfo(
			const hkReal deltaTime, const hknpVehicleInstance* vehicle,
			const hknpVehicleDriverInput::FilteredDriverInputOutput& FilteredDriverInputOutput,
			const hknpVehicleTransmission::TransmissionOutput& TransmissionOutput, EngineOutput& engineOutput ) = 0;

	protected:

		hknpVehicleEngine() {}
};

#endif // HKNP_VEHICLE_ENGINE_H

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
