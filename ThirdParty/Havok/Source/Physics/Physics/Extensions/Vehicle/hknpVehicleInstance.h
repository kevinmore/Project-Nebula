/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HKNP_VEHICLE_INSTANCE_H
#define HKNP_VEHICLE_INSTANCE_H

#include <Physics/Physics/Dynamics/World/hknpWorld.h>
#include <Physics/Physics/Dynamics/Motion/hknpMotion.h>
#include <Physics/Physics/Dynamics/Action/hknpAction.h>

#include <Physics/Physics/Extensions/Vehicle/hknpVehicleData.h>
#include <Physics/Physics/Extensions/Vehicle/Brake/hknpVehicleBrake.h>
#include <Physics/Physics/Extensions/Vehicle/DriverInput/hknpVehicleDriverInput.h>
#include <Physics/Physics/Extensions/Vehicle/AeroDynamics/hknpVehicleAerodynamics.h>
#include <Physics/Physics/Extensions/Vehicle/Transmission/hknpVehicleTransmission.h>
#include <Physics/Physics/Extensions/Vehicle/WheelCollide/hknpVehicleWheelCollide.h>
#include <Physics/Physics/Extensions/Vehicle/VelocityDamper/hknpVehicleVelocityDamper.h>
#include <Physics/ConstraintSolver/VehicleFriction/hkpVehicleFriction.h>

extern const class hkClass hknpVehicleInstanceWheelInfoClass;
extern const class hkClass hknpVehicleInstanceClass;

struct hkpVehicleFrictionSolverParams;
struct hkpVehicleFrictionSolverAxleParams;
class hkpVelocityAccumulator;
class hkStepInfo;
class hkContactPoint;
class hknpVehicleEngine;
class hknpVehicleSteering;
class hknpVehicleSuspension;
class hknpTyremarksInfo;


/// This is a helper struct which contains velocities to apply to the vehicle and ground objects (per axle) and also
/// impulses to apply to other objects that the vehicle is in contact with
struct hknpVehicleJobResults
{
	HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR(HK_MEMORY_CLASS_BASE,hknpVehicleJobResults);
	static const int s_maxNumWheels = 32;

	hkVector4 m_chassisLinearVel;
	hkVector4 m_chassisAngularVel;
	hkVector4 m_groundBodyLinearVel[2];
	hkVector4 m_groundBodyAngularVel[2];
	hknpBodyId m_groundBodyPtr[2];
	hkVector4 m_groundBodyImpulses[s_maxNumWheels];
};


/// This is the main class for a vehicle - it is a container for all the runtime data the vehicle needs, and also
/// contains pointers to all the components that can be shared between different vehicles.
/// This class cannot be shared between different vehicles.
class hknpVehicleInstance : public hknpUnaryAction
{
	public:

		/// This structure stores all data that is useful to the user and is shared among vehicle components.
		struct WheelInfo
		{
			HK_DECLARE_NONVIRTUAL_CLASS_ALLOCATOR( HK_MEMORY_CLASS_VEHICLE, hknpVehicleInstance::WheelInfo );
			HK_DECLARE_REFLECTION();

			void init();

			/// The point of contact of the wheel with the ground (if the wheel is in contact with the ground).
			hkContactPoint m_contactPoint;

			/// The friction coefficient at the point of contact.
			hkReal m_contactFriction;

			/// The ground body the vehicle is in contact. This value is HK_NULL if none of the wheels are in contact
			/// with the ground.
			hknpBodyId m_contactBodyId; //+nosave //+overridetype(hkUint32)

			/// The shape key hierarchy of the object at the point of contact.
			hknpShapeKey m_contactShapeKey;

			/// Location of the wheel hardPoint in world space
			hkVector4 m_hardPointWs;

			/// Farthest point on the wheel from the vehicle in the direction of the suspension.
			hkVector4 m_rayEndPointWs;

			/// Current length of the suspension.
			hkReal m_currentSuspensionLength;

			/// Current direction of the suspension (in world space).
			hkVector4 m_suspensionDirectionWs;

			/// Axis relative to the chassis that the wheel is spinning around.
			hkVector4 m_spinAxisChassisSpace;

			/// Axis in world space that the wheel is spinning around.
			hkVector4 m_spinAxisWs;

			/// The current rotation of the steering wheel.
			hkQuaternion m_steeringOrientationChassisSpace;

			/// The current spin velocity of the wheel in rad/s.
			hkReal m_spinVelocity;

			/// The spin velocity of the wheel assuming no slipping.
			hkReal m_noSlipIdealSpinVelocity;

			/// The current spin angle of the wheel in rads.
			hkReal m_spinAngle;

			/// The energy density lost when skidding (useful to implement tire marks or skid sounds).
			hkReal m_skidEnergyDensity;

			/// The side force at the particular wheel.
			hkReal m_sideForce;

			/// The forward velocity lost by this particular wheel when slipping. This is the difference between the
			/// linear velocity of the wheels (angular velocity projected) and the actual velocity of the vehicle.
			hkReal m_forwardSlipVelocity;

			/// The slip velocity in the side direction for the particular wheel.
			hkReal m_sideSlipVelocity;
		};

	public:

		HK_DECLARE_CLASS_ALLOCATOR(HK_MEMORY_CLASS_VEHICLE);
		HK_DECLARE_REFLECTION();

		/// Default constructor, the chassis rigid body must be constructed already.
		hknpVehicleInstance( hknpBodyId chassis, hknpWorld* world );

		/// Serialization constructor.
		hknpVehicleInstance( hkFinishLoadedObjectFlag f )
			: hknpUnaryAction(f), m_wheelsInfo(f), m_frictionStatus(f), m_isFixed(f), m_wheelsSteeringAngle(f), m_world(HK_NULL) {}

		/// Destructor
		virtual ~hknpVehicleInstance();

		//
		// Methods
		//

		/// Initialize any data that is derived from the initially setup data, such as the number of wheels on each axle.
		virtual void init();

		/// Applies the vehicle controller.
		/// Calls stepVehicle.
		virtual ApplyActionResult applyAction(
			const hknpSimulationThreadContext& tl, const hknpSolverInfo& stepInfo,
			hknpCdPairWriter* HK_RESTRICT pairWriter );

		/// Called when the world coordinate system has been shifted by adding 'offset' to all positions.
		virtual void onShiftWorld( hkVector4Parameter offset );

		/// Update and simulate the vehicle.
		void stepVehicle( const hkStepInfo& stepInfo );

		/// Update the wheels' hardpoints and the wheelCollide component's with the vehicle current transform.
		/// This should be called before wheel collision detection.
		void updateBeforeCollisionDetection();

		/// Apply the calculated forces to the chassis and bodies in contact with the wheels
		void applyForcesFromStep( hknpVehicleJobResults& vehicleResults );

		/// Update and simulate the vehicle given the collision information.
		void stepVehicleUsingWheelCollideOutput( const hkStepInfo& stepInfo, const hknpVehicleWheelCollide::CollisionDetectionWheelOutput* cdInfo, hknpVehicleJobResults& vehicleResults );

		//
		// Adding and removing the vehicle to and from the world.
		//

		/// Add the vehicle's chassis body to the world.
		/// Does not add the vehicle instance as an action to the world.
		virtual void addToWorld( hknpWorld* world );

		/// Removes the vehicle's chassis body from the world.
		/// This hknpVehicleInstance is also removed since it's an action attached to the chassis body.
		virtual void removeFromWorld();

		//
		// Functions to calculate useful information.
		//

		/// Calculate the current position and rotation of a wheel for the graphics engine.
		virtual void calcCurrentPositionAndRotation( const hkTransform& chassisTransform, const hknpVehicleSuspension* suspension, int wheelNo, hkVector4& posOut, hkQuaternion& rotOut );

		/// Retrieves the current RPM of the vehicle
		virtual hkReal calcRPM() const;

		/// Retrieves the current speed of the vehicle in KM/H
		virtual hkReal calcKMPH() const;

		/// Retrieves the current speed of the vehicle in MP/H
		virtual hkReal calcMPH() const;

		/// Retrieve the velocity for a fixed ground object.
		/// You can override this function if you want to assign velocity to ground objects like speed pads.
		virtual void handleFixedGroundAccum( hknpBodyId ground, hkpVelocityAccumulator& accum );

		/// Returns the number of wheels.
		inline hkUint8 getNumWheels() const;

		//
		// Internal functions
		//

		/// Update the components of the vehicle, given the collision detection information.
		/// This puts data in the provided aerodynamicsDragInfo structure and the suspensionForceAtWheel and totalLinearForceAtWheel arrays which is needed
		/// by the simulation step.
		void updateComponents( const hkStepInfo& stepInfo, const hknpVehicleWheelCollide::CollisionDetectionWheelOutput* cdInfo, hknpVehicleAerodynamics::AerodynamicsDragOutput& aerodynamicsDragInfo, hkArray<hkReal>& suspensionForceAtWheel, hkArray<hkReal>& totalLinearForceAtWheel );

		/// Calculate and apply forces to the chassis and the rigid bodies the vehicle is riding on.
		void simulateVehicle( const hkStepInfo& stepInfo, const hknpVehicleAerodynamics::AerodynamicsDragOutput& aerodynamicsDragInfo, const hkArray<hkReal>& suspensionForceAtWheel, const hkArray<hkReal>& totalLinearForceAtWheel, hknpVehicleJobResults& vehicleResults );

	protected:

		// These methods update the state of the components prior to simulation.

		void updateWheels( hkReal deltaTime, const hknpVehicleWheelCollide::CollisionDetectionWheelOutput* cdInfo );
		void updateDriverInput( hkReal deltaTime, hknpVehicleDriverInput::FilteredDriverInputOutput& filteredDriverInputInfo );
		void updateSteering( hkReal deltaTime, const hknpVehicleDriverInput::FilteredDriverInputOutput& filteredDriverInputInfo );
		void updateTransmission( hkReal deltaTime, hknpVehicleTransmission::TransmissionOutput& transmissionInfo );
		void updateEngine( hkReal deltaTime, const hknpVehicleDriverInput::FilteredDriverInputOutput& filteredDriverInputInfo, const hknpVehicleTransmission::TransmissionOutput& transmissionInfo );
		void updateBrake( hkReal deltaTime, const hknpVehicleDriverInput::FilteredDriverInputOutput& filteredDriverInputInfo, hknpVehicleBrake::WheelBreakingOutput& wheelBreakingInfo );
		void updateSuspension( hkReal deltaTime, const hknpVehicleWheelCollide::CollisionDetectionWheelOutput* cdInfo, hkArray<hkReal>& suspensionForces );
		void updateAerodynamics( hkReal deltaTime, hknpVehicleAerodynamics::AerodynamicsDragOutput& aerodynamicsDragInfo );

		// Setup accumulators for the simulation step

		void prepareChassisParams( const hkStepInfo& stepInfo, hkpVehicleFrictionSolverParams& frictionParams );
		void prepareAxleParams( hkReal deltaTime, const hkArray<hkReal>& suspensionForceAtWheel, const hkArray<hkReal>& totalLinearForceAtWheel, hknpBodyId groundBody[], hkpVehicleFrictionSolverParams& frictionParams, const hkStepInfo& stepInfo, hkpVelocityAccumulator groundAccum[], hkpVelocityAccumulator groundAccumAtIntegration[] );
		void getAxleParamsFromWheel( int w_it, hkReal linearForceAtWheel, hkReal suspensionForceAtWheel, hkReal estimatedCarSpeed, hkpVehicleFrictionSolverAxleParams& axle_params );

		// Apply forces to accumulators

		void applyAerodynamicDrag( const hknpVehicleAerodynamics::AerodynamicsDragOutput& aerodynamicsDragInfo, hkpVehicleFrictionSolverParams& frictionParams, const hkReal deltaTime );
		void applyVelocityDamping( const hkReal deltaTime, hkpVehicleFrictionSolverParams& frictionParams, const hknpVehicleVelocityDamper* damper );
		void applySuspensionForces( hkReal deltaTime, const hkArray<hkReal>& suspensionForceAtWheel, hknpBodyId groundBody[], hkpVehicleFrictionSolverParams& frictionParams, hkVector4* suspensionForcesOnGround );
		void applyDampingToAxleAccumulators( const hkStepInfo& stepInfo, hknpBodyId* groundBodies, hkpVehicleFrictionSolverParams& frictionParams, hkVector4* originalLinearVels, hkVector4* originalAngularVels );
		void applyDampingToChassisAccumulator( const hkStepInfo& stepInfo, hkpVelocityAccumulator& accumulator, const hknpMotion& motion );
		void getExtraTorqueFactor( hkReal deltaTime, hkpVelocityAccumulator& accumulatorForChassis ) const;
		void applyFrictionSolver( const hkStepInfo& stepInfo, hkpVehicleFrictionSolverParams& frictionParams );

		// Apply results of the friction solver.

		void calcChassisVelocities( hkpVelocityAccumulator& accumulatorForChassis, hknpVehicleJobResults& vehicleResults );
		void calcGroundBodyVelocities( hkReal deltaTime, const hkpVehicleFrictionSolverAxleParams axleParams[], hknpBodyId groundBody[], hknpVehicleJobResults& vehicleResults, hkVector4* originalLinearVels, hkVector4* originalAngularVels );
		void applyResultsToWheelInfo( hkReal deltaTime, hknpBodyId groundBody[], const hkpVehicleFrictionSolverParams& frictionParams );

		void buildAccumulator(const hkStepInfo& info, const hknpMotion& motion, const hkTransform& transform, hkpVelocityAccumulator& accumulatorsOut);

	public:

		/// An upper bound useful for creating local arrays for wheels.
		static const hkUint8 s_maxNumLocalWheels = 16;

		//
		// Members
		//

		/// Contains data about the vehicle that can be shared across several vehicles.
		hknpVehicleData* m_data;

		/// The DriverInput for the vehicle.
		hknpVehicleDriverInput* m_driverInput;

		/// The Steering for the vehicle.
		hknpVehicleSteering* m_steering;

		/// The Engine for the vehicle.
		hknpVehicleEngine* m_engine;

		/// The Transmission for the vehicle.
		hknpVehicleTransmission* m_transmission;

		/// The Brake for the vehicle.
		hknpVehicleBrake* m_brake;

		/// The Suspension for the vehicle.
		hknpVehicleSuspension* m_suspension;

		/// The Aerodynamics for the vehicle.
		hknpVehicleAerodynamics* m_aerodynamics;

		/// The Collision for the vehicle.
		hknpVehicleWheelCollide* m_wheelCollide;

		/// The Tyremarks controller for the vehicle.
		hknpTyremarksInfo* m_tyreMarks;

		/// The list of external vehicle controllers.
		hknpVehicleVelocityDamper* m_velocityDamper;

		/// The WheelInfo class holds all wheel information generated externally (from the physics engine)
		/// such as each wheel's ground contact, sliding state, forces, contact friction etc.
		hkArray<struct WheelInfo> m_wheelsInfo;

		///
		hkpVehicleFrictionStatus m_frictionStatus;


		//
		// Variables used by the components to cache data.
		// This is slightly ugly, but necessary until it is possible until a
		// decent runtime cache manager can be constructed.
		//


		///	Current controller input state.
		hknpVehicleDriverInputStatus* m_deviceStatus;

		// from brake
		hkArray<hkBool> m_isFixed;
		hkReal m_wheelsTimeSinceMaxPedalInput;

		// from driver input
		hkBool m_tryingToReverse;

		// from engine
		hkReal m_torque;
		hkReal m_rpm;

		// from steering
		hkReal m_mainSteeringAngle;
		hkReal m_mainSteeringAngleAssumingNoReduction;

		hkArray<hkReal> m_wheelsSteeringAngle;

		// from transmission
		hkBool m_isReversing;
		hkInt8 m_currentGear;
		hkBool m_delayed;
		hkReal m_clutchDelayCountdown;

	public:

		/// Get a reference to the vehicle's chassis transform.
		HK_FORCE_INLINE const hkTransform& getChassisTransform( void ) const;

		/// Get a const reference to the vehicle's chassis motion.
		HK_FORCE_INLINE const hknpMotion& getChassisMotion ( void ) const;

		/// Get a reference to the vehicle's chassis motion.
		HK_FORCE_INLINE hknpMotion& accessChassisMotion ( void );

		hknpWorld* const m_world; //+nosave
};

#include <Physics/Physics/Extensions/Vehicle/hknpVehicleInstance.inl>

#endif // HKNP_VEHICLE_INSTANCE_H

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
