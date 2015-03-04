/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#include <Physics2012/Vehicle/hkpVehicle.h>
static const char s_libraryName[] = "hkpVehicle";
#include <Common/Base/Memory/Tracker/hkTrackerClassDefinition.h>

void HK_CALL hkpVehicleRegister() {}

#include <Physics2012/Vehicle/AeroDynamics/Default/hkpVehicleDefaultAerodynamics.h>


// hkpVehicleDefaultAerodynamics ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleDefaultAerodynamics)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleDefaultAerodynamics)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpVehicleDefaultAerodynamics, s_libraryName, hkpVehicleAerodynamics)

#include <Physics2012/Vehicle/AeroDynamics/hkpVehicleAerodynamics.h>


// hkpVehicleAerodynamics ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleAerodynamics)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(AerodynamicsDragOutput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleAerodynamics)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpVehicleAerodynamics, s_libraryName, hkReferencedObject)


// AerodynamicsDragOutput hkpVehicleAerodynamics
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpVehicleAerodynamics, AerodynamicsDragOutput, s_libraryName)

#include <Physics2012/Vehicle/Brake/Default/hkpVehicleDefaultBrake.h>


// hkpVehicleDefaultBrake ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleDefaultBrake)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(WheelBrakingProperties)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleDefaultBrake)
    HK_TRACKER_MEMBER(hkpVehicleDefaultBrake, m_wheelBrakingProperties, 0, "hkArray<hkpVehicleDefaultBrake::WheelBrakingProperties, hkContainerHeapAllocator>") // hkArray< struct hkpVehicleDefaultBrake::WheelBrakingProperties, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpVehicleDefaultBrake, s_libraryName, hkpVehicleBrake)


// WheelBrakingProperties hkpVehicleDefaultBrake
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpVehicleDefaultBrake, WheelBrakingProperties, s_libraryName)

#include <Physics2012/Vehicle/Brake/hkpVehicleBrake.h>


// hkpVehicleBrake ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleBrake)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(WheelBreakingOutput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleBrake)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpVehicleBrake, s_libraryName, hkReferencedObject)


// WheelBreakingOutput hkpVehicleBrake

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleBrake::WheelBreakingOutput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleBrake::WheelBreakingOutput)
    HK_TRACKER_MEMBER(hkpVehicleBrake::WheelBreakingOutput, m_brakingTorque, 0, "hkInplaceArray<float, 32, hkContainerHeapAllocator>") // class hkInplaceArray< float, 32, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpVehicleBrake::WheelBreakingOutput, m_isFixed, 0, "hkInplaceArray<hkBool, 32, hkContainerHeapAllocator>") // class hkInplaceArray< hkBool, 32, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpVehicleBrake::WheelBreakingOutput, s_libraryName)

#include <Physics2012/Vehicle/Camera/hkp1dAngularFollowCam.h>


// hkp1dAngularFollowCam ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkp1dAngularFollowCam)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CameraInput)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CameraOutput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkp1dAngularFollowCam)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkp1dAngularFollowCam, s_libraryName, hkReferencedObject)


// CameraInput hkp1dAngularFollowCam
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkp1dAngularFollowCam, CameraInput, s_libraryName)


// CameraOutput hkp1dAngularFollowCam
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkp1dAngularFollowCam, CameraOutput, s_libraryName)

#include <Physics2012/Vehicle/Camera/hkp1dAngularFollowCamCinfo.h>


// hkp1dAngularFollowCamCinfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkp1dAngularFollowCamCinfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CameraSet)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkp1dAngularFollowCamCinfo, s_libraryName)


// CameraSet hkp1dAngularFollowCamCinfo
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkp1dAngularFollowCamCinfo, CameraSet, s_libraryName)

#include <Physics2012/Vehicle/DriverInput/Default/hkpVehicleDefaultAnalogDriverInput.h>


// hkpVehicleDriverInputAnalogStatus ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleDriverInputAnalogStatus)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleDriverInputAnalogStatus)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpVehicleDriverInputAnalogStatus, s_libraryName, hkpVehicleDriverInputStatus)


// hkpVehicleDefaultAnalogDriverInput ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleDefaultAnalogDriverInput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleDefaultAnalogDriverInput)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpVehicleDefaultAnalogDriverInput, s_libraryName, hkpVehicleDriverInput)

#include <Physics2012/Vehicle/DriverInput/hkpVehicleDriverInput.h>


// hkpVehicleDriverInputStatus ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleDriverInputStatus)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleDriverInputStatus)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpVehicleDriverInputStatus, s_libraryName, hkReferencedObject)


// hkpVehicleDriverInput ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleDriverInput)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(FilteredDriverInputOutput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleDriverInput)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpVehicleDriverInput, s_libraryName, hkReferencedObject)


// FilteredDriverInputOutput hkpVehicleDriverInput
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpVehicleDriverInput, FilteredDriverInputOutput, s_libraryName)

#include <Physics2012/Vehicle/Engine/Default/hkpVehicleDefaultEngine.h>


// hkpVehicleDefaultEngine ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleDefaultEngine)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleDefaultEngine)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpVehicleDefaultEngine, s_libraryName, hkpVehicleEngine)

#include <Physics2012/Vehicle/Engine/hkpVehicleEngine.h>


// hkpVehicleEngine ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleEngine)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(EngineOutput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleEngine)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpVehicleEngine, s_libraryName, hkReferencedObject)


// EngineOutput hkpVehicleEngine
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpVehicleEngine, EngineOutput, s_libraryName)

#include <Physics2012/Vehicle/Manager/LinearCastBatchingManager/hkpVehicleLinearCastBatchingManager.h>


// hkpVehicleLinearCastBatchingManager ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleLinearCastBatchingManager)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(LinearCastBatch)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleLinearCastBatchingManager)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpVehicleLinearCastBatchingManager, s_libraryName, hkpVehicleCastBatchingManager)


// LinearCastBatch hkpVehicleLinearCastBatchingManager

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleLinearCastBatchingManager::LinearCastBatch)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleLinearCastBatchingManager::LinearCastBatch)
    HK_TRACKER_MEMBER(hkpVehicleLinearCastBatchingManager::LinearCastBatch, m_collidableStorage, 0, "hkpCollidable*") // class hkpCollidable*
    HK_TRACKER_MEMBER(hkpVehicleLinearCastBatchingManager::LinearCastBatch, m_commandStorage, 0, "hkpPairLinearCastCommand*") // struct hkpPairLinearCastCommand*
    HK_TRACKER_MEMBER(hkpVehicleLinearCastBatchingManager::LinearCastBatch, m_outputStorage, 0, "hkpRootCdPoint*") // struct hkpRootCdPoint*
    HK_TRACKER_MEMBER(hkpVehicleLinearCastBatchingManager::LinearCastBatch, m_jobHeaders, 0, "hkpCollisionQueryJobHeader*") // struct hkpCollisionQueryJobHeader*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpVehicleLinearCastBatchingManager::LinearCastBatch, s_libraryName)

#include <Physics2012/Vehicle/Manager/MultithreadedVehicle/hkpMultithreadedVehicleManager.h>


// hkpMultithreadedVehicleManager ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMultithreadedVehicleManager)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(VehicleCommandBatch)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMultithreadedVehicleManager)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpMultithreadedVehicleManager, s_libraryName, hkpVehicleManager)


// VehicleCommandBatch hkpMultithreadedVehicleManager

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMultithreadedVehicleManager::VehicleCommandBatch)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMultithreadedVehicleManager::VehicleCommandBatch)
    HK_TRACKER_MEMBER(hkpMultithreadedVehicleManager::VehicleCommandBatch, m_commandStorage, 0, "hkpVehicleCommand*") // struct hkpVehicleCommand*
    HK_TRACKER_MEMBER(hkpMultithreadedVehicleManager::VehicleCommandBatch, m_outputStorage, 0, "hkpVehicleJobResults*") // struct hkpVehicleJobResults*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpMultithreadedVehicleManager::VehicleCommandBatch, s_libraryName)

#include <Physics2012/Vehicle/Manager/MultithreadedVehicle/hkpVehicleJobQueueUtils.h>


// hkpVehicleJobQueueUtils ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpVehicleJobQueueUtils, s_libraryName)

#include <Physics2012/Vehicle/Manager/MultithreadedVehicle/hkpVehicleJobs.h>


// hkpVehicleJob ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpVehicleJob, s_libraryName)


// hkpVehicleJobResults ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleJobResults)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleJobResults)
    HK_TRACKER_MEMBER(hkpVehicleJobResults, m_groundBodyPtr, 0, "hkpRigidBody* [2]") // class hkpRigidBody* [2]
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpVehicleJobResults, s_libraryName)


// hkpVehicleCommand ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleCommand)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleCommand)
    HK_TRACKER_MEMBER(hkpVehicleCommand, m_jobResults, 0, "hkpVehicleJobResults*") // struct hkpVehicleJobResults*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpVehicleCommand, s_libraryName)


// hkpVehicleIntegrateJob ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleIntegrateJob)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleIntegrateJob)
    HK_TRACKER_MEMBER(hkpVehicleIntegrateJob, m_commandArray, 0, "hkpVehicleCommand*") // const struct hkpVehicleCommand*
    HK_TRACKER_MEMBER(hkpVehicleIntegrateJob, m_vehicleArrayPtr, 0, "hkpVehicleInstance**") // class hkpVehicleInstance**
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpVehicleIntegrateJob, s_libraryName, hkpVehicleJob)

#include <Physics2012/Vehicle/Manager/RayCastBatchingManager/hkpVehicleRayCastBatchingManager.h>


// hkpVehicleRayCastBatchingManager ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleRayCastBatchingManager)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(RaycastBatch)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleRayCastBatchingManager)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpVehicleRayCastBatchingManager, s_libraryName, hkpVehicleCastBatchingManager)


// RaycastBatch hkpVehicleRayCastBatchingManager

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleRayCastBatchingManager::RaycastBatch)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleRayCastBatchingManager::RaycastBatch)
    HK_TRACKER_MEMBER(hkpVehicleRayCastBatchingManager::RaycastBatch, m_commandStorage, 0, "hkpShapeRayCastCommand*") // struct hkpShapeRayCastCommand*
    HK_TRACKER_MEMBER(hkpVehicleRayCastBatchingManager::RaycastBatch, m_outputStorage, 0, "hkpWorldRayCastOutput*") // struct hkpWorldRayCastOutput*
    HK_TRACKER_MEMBER(hkpVehicleRayCastBatchingManager::RaycastBatch, m_index, 0, "hkUint8*") // hkUint8*
    HK_TRACKER_MEMBER(hkpVehicleRayCastBatchingManager::RaycastBatch, m_jobHeaders, 0, "hkpCollisionQueryJobHeader*") // struct hkpCollisionQueryJobHeader*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpVehicleRayCastBatchingManager::RaycastBatch, s_libraryName)

#include <Physics2012/Vehicle/Manager/hkpVehicleCastBatchingManager.h>


// hkpVehicleCastBatchingManager ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleCastBatchingManager)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleCastBatchingManager)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpVehicleCastBatchingManager, s_libraryName, hkpVehicleManager)

#include <Physics2012/Vehicle/Manager/hkpVehicleManager.h>


// hkpVehicleManager ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleManager)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleManager)
    HK_TRACKER_MEMBER(hkpVehicleManager, m_registeredVehicles, 0, "hkArray<hkpVehicleInstance*, hkContainerHeapAllocator>") // hkArray< class hkpVehicleInstance*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpVehicleManager, s_libraryName, hkReferencedObject)

#include <Physics2012/Vehicle/Simulation/Default/hkpVehicleDefaultSimulation.h>


// hkpVehicleDefaultSimulation ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleDefaultSimulation)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleDefaultSimulation)
    HK_TRACKER_MEMBER(hkpVehicleDefaultSimulation, m_frictionDescription, 0, "hkpVehicleFrictionDescription *") // class hkpVehicleFrictionDescription *
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpVehicleDefaultSimulation, s_libraryName, hkpVehicleSimulation)

#include <Physics2012/Vehicle/Simulation/PerWheel/hkpVehiclePerWheelSimulation.h>


// hkpVehiclePerWheelSimulation ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehiclePerWheelSimulation)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(WheelData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehiclePerWheelSimulation)
    HK_TRACKER_MEMBER(hkpVehiclePerWheelSimulation, m_instance, 0, "hkpVehicleInstance*") // class hkpVehicleInstance*
    HK_TRACKER_MEMBER(hkpVehiclePerWheelSimulation, m_world, 0, "hkpWorld*") // class hkpWorld*
    HK_TRACKER_MEMBER(hkpVehiclePerWheelSimulation, m_wheelData, 0, "hkArray<hkpVehiclePerWheelSimulation::WheelData, hkContainerHeapAllocator>") // hkArray< struct hkpVehiclePerWheelSimulation::WheelData, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpVehiclePerWheelSimulation, s_libraryName, hkpVehicleSimulation)


// WheelData hkpVehiclePerWheelSimulation

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehiclePerWheelSimulation::WheelData)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehiclePerWheelSimulation::WheelData)
    HK_TRACKER_MEMBER(hkpVehiclePerWheelSimulation::WheelData, m_frictionData, 0, "hkpWheelFrictionConstraintData") // class hkpWheelFrictionConstraintData
    HK_TRACKER_MEMBER(hkpVehiclePerWheelSimulation::WheelData, m_frictionConstraint, 0, "hkpConstraintInstance*") // class hkpConstraintInstance*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpVehiclePerWheelSimulation::WheelData, s_libraryName)

#include <Physics2012/Vehicle/Simulation/hkpVehicleSimulation.h>


// hkpVehicleSimulation ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleSimulation)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SimulationInput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleSimulation)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpVehicleSimulation, s_libraryName, hkReferencedObject)


// SimulationInput hkpVehicleSimulation
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpVehicleSimulation, SimulationInput, s_libraryName)

#include <Physics2012/Vehicle/Steering/Ackerman/hkpVehicleSteeringAckerman.h>


// hkpVehicleSteeringAckerman ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleSteeringAckerman)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleSteeringAckerman)
    HK_TRACKER_MEMBER(hkpVehicleSteeringAckerman, m_doesWheelSteer, 0, "hkArray<hkBool, hkContainerHeapAllocator>") // hkArray< hkBool, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpVehicleSteeringAckerman, s_libraryName, hkpVehicleSteering)

#include <Physics2012/Vehicle/Steering/Default/hkpVehicleDefaultSteering.h>


// hkpVehicleDefaultSteering ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleDefaultSteering)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleDefaultSteering)
    HK_TRACKER_MEMBER(hkpVehicleDefaultSteering, m_doesWheelSteer, 0, "hkArray<hkBool, hkContainerHeapAllocator>") // hkArray< hkBool, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpVehicleDefaultSteering, s_libraryName, hkpVehicleSteering)

#include <Physics2012/Vehicle/Steering/hkpVehicleSteering.h>


// hkpVehicleSteering ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleSteering)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SteeringAnglesOutput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleSteering)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpVehicleSteering, s_libraryName, hkReferencedObject)


// SteeringAnglesOutput hkpVehicleSteering

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleSteering::SteeringAnglesOutput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleSteering::SteeringAnglesOutput)
    HK_TRACKER_MEMBER(hkpVehicleSteering::SteeringAnglesOutput, m_wheelsSteeringAngle, 0, "hkInplaceArray<float, 32, hkContainerHeapAllocator>") // class hkInplaceArray< float, 32, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpVehicleSteering::SteeringAnglesOutput, s_libraryName)

#include <Physics2012/Vehicle/Suspension/Default/hkpVehicleDefaultSuspension.h>


// hkpVehicleDefaultSuspension ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleDefaultSuspension)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(WheelSpringSuspensionParameters)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleDefaultSuspension)
    HK_TRACKER_MEMBER(hkpVehicleDefaultSuspension, m_wheelSpringParams, 0, "hkArray<hkpVehicleDefaultSuspension::WheelSpringSuspensionParameters, hkContainerHeapAllocator>") // hkArray< struct hkpVehicleDefaultSuspension::WheelSpringSuspensionParameters, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpVehicleDefaultSuspension, s_libraryName, hkpVehicleSuspension)


// WheelSpringSuspensionParameters hkpVehicleDefaultSuspension
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpVehicleDefaultSuspension, WheelSpringSuspensionParameters, s_libraryName)

#include <Physics2012/Vehicle/Suspension/hkpVehicleSuspension.h>


// hkpVehicleSuspension ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleSuspension)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(SuspensionWheelParameters)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleSuspension)
    HK_TRACKER_MEMBER(hkpVehicleSuspension, m_wheelParams, 0, "hkArray<hkpVehicleSuspension::SuspensionWheelParameters, hkContainerHeapAllocator>") // hkArray< struct hkpVehicleSuspension::SuspensionWheelParameters, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpVehicleSuspension, s_libraryName, hkReferencedObject)


// SuspensionWheelParameters hkpVehicleSuspension
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpVehicleSuspension, SuspensionWheelParameters, s_libraryName)

#include <Physics2012/Vehicle/Transmission/Default/hkpVehicleDefaultTransmission.h>


// hkpVehicleDefaultTransmission ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleDefaultTransmission)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleDefaultTransmission)
    HK_TRACKER_MEMBER(hkpVehicleDefaultTransmission, m_gearsRatio, 0, "hkArray<float, hkContainerHeapAllocator>") // hkArray< float, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpVehicleDefaultTransmission, m_wheelsTorqueRatio, 0, "hkArray<float, hkContainerHeapAllocator>") // hkArray< float, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpVehicleDefaultTransmission, s_libraryName, hkpVehicleTransmission)

#include <Physics2012/Vehicle/Transmission/hkpVehicleTransmission.h>


// hkpVehicleTransmission ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleTransmission)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(TransmissionOutput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleTransmission)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpVehicleTransmission, s_libraryName, hkReferencedObject)


// TransmissionOutput hkpVehicleTransmission

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleTransmission::TransmissionOutput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleTransmission::TransmissionOutput)
    HK_TRACKER_MEMBER(hkpVehicleTransmission::TransmissionOutput, m_wheelsTransmittedTorque, 0, "float*") // float*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpVehicleTransmission::TransmissionOutput, s_libraryName)

#include <Physics2012/Vehicle/TyreMarks/hkpTyremarksInfo.h>


// hkpTyremarkPoint ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpTyremarkPoint, s_libraryName)


// hkpTyremarksWheel ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpTyremarksWheel)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpTyremarksWheel)
    HK_TRACKER_MEMBER(hkpTyremarksWheel, m_tyremarkPoints, 0, "hkArray<hkpTyremarkPoint, hkContainerHeapAllocator>") // hkArray< struct hkpTyremarkPoint, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpTyremarksWheel, s_libraryName, hkReferencedObject)


// hkpTyremarksInfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpTyremarksInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpTyremarksInfo)
    HK_TRACKER_MEMBER(hkpTyremarksInfo, m_tyremarksWheel, 0, "hkArray<hkpTyremarksWheel*, hkContainerHeapAllocator>") // hkArray< class hkpTyremarksWheel*, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpTyremarksInfo, s_libraryName, hkReferencedObject)

#include <Physics2012/Vehicle/VelocityDamper/Default/hkpVehicleDefaultVelocityDamper.h>


// hkpVehicleDefaultVelocityDamper ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleDefaultVelocityDamper)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleDefaultVelocityDamper)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpVehicleDefaultVelocityDamper, s_libraryName, hkpVehicleVelocityDamper)

#include <Physics2012/Vehicle/VelocityDamper/hkpVehicleVelocityDamper.h>


// hkpVehicleVelocityDamper ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleVelocityDamper)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleVelocityDamper)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpVehicleVelocityDamper, s_libraryName, hkReferencedObject)

#include <Physics2012/Vehicle/Wheel/hkpLinearCastWheel.h>


// hkpLinearCastWheel ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpLinearCastWheel)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpLinearCastWheel)
    HK_TRACKER_MEMBER(hkpLinearCastWheel, m_shape, 0, "hkpShape *") // class hkpShape *
    HK_TRACKER_MEMBER(hkpLinearCastWheel, m_collector, 0, "hkpClosestCdPointCollector*") // class hkpClosestCdPointCollector*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpLinearCastWheel, s_libraryName, hkpWheel)

#include <Physics2012/Vehicle/Wheel/hkpRaycastWheel.h>


// hkpRaycastWheel ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpRaycastWheel)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpRaycastWheel)
    HK_TRACKER_MEMBER(hkpRaycastWheel, m_collector, 0, "hkpClosestRayHitCollector*") // class hkpClosestRayHitCollector*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpRaycastWheel, s_libraryName, hkpWheel)

#include <Physics2012/Vehicle/Wheel/hkpWheel.h>


// hkpWheel ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpWheel)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpWheel)
    HK_TRACKER_MEMBER(hkpWheel, m_frictionData, 0, "hkpWheelFrictionConstraintData") // class hkpWheelFrictionConstraintData
    HK_TRACKER_MEMBER(hkpWheel, m_frictionConstraint, 0, "hkpConstraintInstance*") // class hkpConstraintInstance*
    HK_TRACKER_MEMBER(hkpWheel, m_axle, 0, "hkpWheelFrictionConstraintAtom::Axle*") // struct hkpWheelFrictionConstraintAtom::Axle*
    HK_TRACKER_MEMBER(hkpWheel, m_chassis, 0, "hkpRigidBody *") // class hkpRigidBody *
    HK_TRACKER_MEMBER(hkpWheel, m_phantom, 0, "hkpAabbPhantom*") // class hkpAabbPhantom*
    HK_TRACKER_MEMBER(hkpWheel, m_rejectChassisListener, 0, "hkpRejectChassisListener") // class hkpRejectChassisListener
    HK_TRACKER_MEMBER(hkpWheel, m_contactBody, 0, "hkpRigidBody*") // class hkpRigidBody*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpWheel, s_libraryName, hkReferencedObject)

#include <Physics2012/Vehicle/WheelCollide/LinearCast/hkpVehicleLinearCastWheelCollide.h>


// hkpVehicleLinearCastWheelCollide ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleLinearCastWheelCollide)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(WheelState)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleLinearCastWheelCollide)
    HK_TRACKER_MEMBER(hkpVehicleLinearCastWheelCollide, m_wheelStates, 0, "hkArray<hkpVehicleLinearCastWheelCollide::WheelState, hkContainerHeapAllocator>") // hkArray< struct hkpVehicleLinearCastWheelCollide::WheelState, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpVehicleLinearCastWheelCollide, m_rejectChassisListener, 0, "hkpRejectChassisListener") // class hkpRejectChassisListener
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpVehicleLinearCastWheelCollide, s_libraryName, hkpVehicleWheelCollide)


// WheelState hkpVehicleLinearCastWheelCollide

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleLinearCastWheelCollide::WheelState)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleLinearCastWheelCollide::WheelState)
    HK_TRACKER_MEMBER(hkpVehicleLinearCastWheelCollide::WheelState, m_phantom, 0, "hkpAabbPhantom*") // class hkpAabbPhantom*
    HK_TRACKER_MEMBER(hkpVehicleLinearCastWheelCollide::WheelState, m_shape, 0, "hkpShape*") // const class hkpShape*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpVehicleLinearCastWheelCollide::WheelState, s_libraryName)

#include <Physics2012/Vehicle/WheelCollide/RayCast/hkpVehicleRayCastWheelCollide.h>


// hkpVehicleRayCastWheelCollide ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleRayCastWheelCollide)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleRayCastWheelCollide)
    HK_TRACKER_MEMBER(hkpVehicleRayCastWheelCollide, m_phantom, 0, "hkpAabbPhantom*") // class hkpAabbPhantom*
    HK_TRACKER_MEMBER(hkpVehicleRayCastWheelCollide, m_rejectRayChassisListener, 0, "hkpRejectChassisListener") // class hkpRejectChassisListener
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpVehicleRayCastWheelCollide, s_libraryName, hkpVehicleWheelCollide)

#include <Physics2012/Vehicle/WheelCollide/RejectChassisListener/hkpRejectChassisListener.h>


// hkpRejectChassisListener ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpRejectChassisListener)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpRejectChassisListener)
    HK_TRACKER_MEMBER(hkpRejectChassisListener, m_chassis, 0, "hkpCollidable*") // const class hkpCollidable*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpRejectChassisListener, s_libraryName, hkReferencedObject)

#include <Physics2012/Vehicle/WheelCollide/hkpVehicleWheelCollide.h>


// hkpVehicleWheelCollide ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleWheelCollide)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CollisionDetectionWheelOutput)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(WheelCollideType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleWheelCollide)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpVehicleWheelCollide, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpVehicleWheelCollide, WheelCollideType, s_libraryName)


// CollisionDetectionWheelOutput hkpVehicleWheelCollide

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleWheelCollide::CollisionDetectionWheelOutput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleWheelCollide::CollisionDetectionWheelOutput)
    HK_TRACKER_MEMBER(hkpVehicleWheelCollide::CollisionDetectionWheelOutput, m_contactBody, 0, "hkpRigidBody*") // class hkpRigidBody*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpVehicleWheelCollide::CollisionDetectionWheelOutput, s_libraryName)

#include <Physics2012/Vehicle/hkpVehicleData.h>


// hkpVehicleData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleData)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(WheelComponentParams)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleData)
    HK_TRACKER_MEMBER(hkpVehicleData, m_wheelParams, 0, "hkArray<hkpVehicleData::WheelComponentParams, hkContainerHeapAllocator>") // hkArray< struct hkpVehicleData::WheelComponentParams, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpVehicleData, m_numWheelsPerAxle, 0, "hkArray<hkInt8, hkContainerHeapAllocator>") // hkArray< hkInt8, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpVehicleData, s_libraryName, hkReferencedObject)


// WheelComponentParams hkpVehicleData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpVehicleData, WheelComponentParams, s_libraryName)

#include <Physics2012/Vehicle/hkpVehicleInstance.h>


// hkpVehicleInstance ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleInstance)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(WheelInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleInstance)
    HK_TRACKER_MEMBER(hkpVehicleInstance, m_data, 0, "hkpVehicleData*") // class hkpVehicleData*
    HK_TRACKER_MEMBER(hkpVehicleInstance, m_driverInput, 0, "hkpVehicleDriverInput*") // class hkpVehicleDriverInput*
    HK_TRACKER_MEMBER(hkpVehicleInstance, m_steering, 0, "hkpVehicleSteering*") // class hkpVehicleSteering*
    HK_TRACKER_MEMBER(hkpVehicleInstance, m_engine, 0, "hkpVehicleEngine*") // class hkpVehicleEngine*
    HK_TRACKER_MEMBER(hkpVehicleInstance, m_transmission, 0, "hkpVehicleTransmission*") // class hkpVehicleTransmission*
    HK_TRACKER_MEMBER(hkpVehicleInstance, m_brake, 0, "hkpVehicleBrake*") // class hkpVehicleBrake*
    HK_TRACKER_MEMBER(hkpVehicleInstance, m_suspension, 0, "hkpVehicleSuspension*") // class hkpVehicleSuspension*
    HK_TRACKER_MEMBER(hkpVehicleInstance, m_aerodynamics, 0, "hkpVehicleAerodynamics*") // class hkpVehicleAerodynamics*
    HK_TRACKER_MEMBER(hkpVehicleInstance, m_wheelCollide, 0, "hkpVehicleWheelCollide*") // class hkpVehicleWheelCollide*
    HK_TRACKER_MEMBER(hkpVehicleInstance, m_tyreMarks, 0, "hkpTyremarksInfo*") // class hkpTyremarksInfo*
    HK_TRACKER_MEMBER(hkpVehicleInstance, m_velocityDamper, 0, "hkpVehicleVelocityDamper*") // class hkpVehicleVelocityDamper*
    HK_TRACKER_MEMBER(hkpVehicleInstance, m_vehicleSimulation, 0, "hkpVehicleSimulation*") // class hkpVehicleSimulation*
    HK_TRACKER_MEMBER(hkpVehicleInstance, m_wheelsInfo, 0, "hkArray<hkpVehicleInstance::WheelInfo, hkContainerHeapAllocator>") // hkArray< struct hkpVehicleInstance::WheelInfo, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpVehicleInstance, m_deviceStatus, 0, "hkpVehicleDriverInputStatus*") // class hkpVehicleDriverInputStatus*
    HK_TRACKER_MEMBER(hkpVehicleInstance, m_isFixed, 0, "hkArray<hkBool, hkContainerHeapAllocator>") // hkArray< hkBool, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpVehicleInstance, m_wheelsSteeringAngle, 0, "hkArray<float, hkContainerHeapAllocator>") // hkArray< float, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpVehicleInstance, s_libraryName, hkpUnaryAction)


// WheelInfo hkpVehicleInstance

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleInstance::WheelInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleInstance::WheelInfo)
    HK_TRACKER_MEMBER(hkpVehicleInstance::WheelInfo, m_contactBody, 0, "hkpRigidBody*") // class hkpRigidBody*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpVehicleInstance::WheelInfo, s_libraryName)

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
