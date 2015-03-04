/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_VEHICLE_AERODYNAMICS_H
#define HKNP_VEHICLE_AERODYNAMICS_H

extern const class hkClass hknpVehicleAerodynamicsClass;

class hknpVehicleInstance;

/// Responsible for providing the total force and torque applied to the chassis by environmental forces.
/// Examples of environmental forces include aerodynamic drag and aerodynamic lift.
class hknpVehicleAerodynamics : public hkReferencedObject
{
	public:

		/// Container for data output by the aerodynamics calculations.
		struct AerodynamicsDragOutput
		{
		public:
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_VEHICLE, hknpVehicleAerodynamics::AerodynamicsDragOutput );
			/// The total force applied by all aerodynamics forces.
			hkVector4 m_aerodynamicsForce;

			/// The total torque applied by all aerodynamics forces
			hkVector4 m_aerodynamicsTorque;
		};

	public:

		HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_VEHICLE);
		HK_DECLARE_REFLECTION();

			/// Serialization constructor.
		hknpVehicleAerodynamics(hkFinishLoadedObjectFlag flag) : hkReferencedObject(flag) {}

			/// Calculates the effect of aerodynamic forces on the vehicle.
		virtual void calcAerodynamics(const hkReal deltaTime, const hknpVehicleInstance* vehicle, AerodynamicsDragOutput& dragInfoOut) = 0;

	protected:

		hknpVehicleAerodynamics() {}
};

#endif // HKNP_VEHICLE_AERODYNAMICS_H

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
