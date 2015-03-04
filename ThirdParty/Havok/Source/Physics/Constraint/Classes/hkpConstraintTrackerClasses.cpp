/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#include <Common/Base/hkBase.h>
static const char s_libraryName[] = "hkpConstraint";
#include <Common/Base/Memory/Tracker/hkTrackerClassDefinition.h>

void HK_CALL hkpConstraintRegister() {}

#include <Physics/Constraint/Atom/Bridge/hkpBridgeConstraintAtom.h>


// hkpBridgeConstraintAtom ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBridgeConstraintAtom)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBridgeConstraintAtom)
    HK_TRACKER_MEMBER(hkpBridgeConstraintAtom, m_constraintData, 0, "hkpConstraintData*") // class hkpConstraintData*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpBridgeConstraintAtom, s_libraryName, hkpConstraintAtom)


// hkpBridgeAtoms ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBridgeAtoms)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBridgeAtoms)
    HK_TRACKER_MEMBER(hkpBridgeAtoms, m_bridgeAtom, 0, "hkpBridgeConstraintAtom") // struct hkpBridgeConstraintAtom
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpBridgeAtoms, s_libraryName)

#include <Physics/Constraint/Atom/hkpConstraintAtom.h>


// hkpConstraintAtom ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpConstraintAtom, s_libraryName)


// hkpSetupStabilizationAtom ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpSetupStabilizationAtom, s_libraryName)


// hkp3dAngConstraintAtom ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkp3dAngConstraintAtom, s_libraryName)


// hkpDeformableLinConstraintAtom ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpDeformableLinConstraintAtom, s_libraryName)


// hkpDeformableAngConstraintAtom ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpDeformableAngConstraintAtom, s_libraryName)


// hkpBallSocketConstraintAtom ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpBallSocketConstraintAtom, s_libraryName)


// hkpStiffSpringConstraintAtom ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpStiffSpringConstraintAtom, s_libraryName)


// hkpSetLocalTransformsConstraintAtom ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpSetLocalTransformsConstraintAtom, s_libraryName)


// hkpSetLocalTranslationsConstraintAtom ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpSetLocalTranslationsConstraintAtom, s_libraryName)


// hkpSetLocalRotationsConstraintAtom ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpSetLocalRotationsConstraintAtom, s_libraryName)


// hkpOverwritePivotConstraintAtom ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpOverwritePivotConstraintAtom, s_libraryName)


// hkpLinConstraintAtom ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpLinConstraintAtom, s_libraryName)


// hkpLinSoftConstraintAtom ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpLinSoftConstraintAtom, s_libraryName)


// hkpLinLimitConstraintAtom ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpLinLimitConstraintAtom, s_libraryName)


// hkp2dAngConstraintAtom ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkp2dAngConstraintAtom, s_libraryName)


// hkpAngConstraintAtom ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpAngConstraintAtom, s_libraryName)


// hkpAngLimitConstraintAtom ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpAngLimitConstraintAtom, s_libraryName)


// hkpTwistLimitConstraintAtom ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpTwistLimitConstraintAtom, s_libraryName)


// hkpConeLimitConstraintAtom ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpConeLimitConstraintAtom, s_libraryName)


// hkpAngFrictionConstraintAtom ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpAngFrictionConstraintAtom, s_libraryName)


// hkpAngMotorConstraintAtom ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpAngMotorConstraintAtom)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpAngMotorConstraintAtom)
    HK_TRACKER_MEMBER(hkpAngMotorConstraintAtom, m_motor, 0, "hkpConstraintMotor*") // class hkpConstraintMotor*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpAngMotorConstraintAtom, s_libraryName, hkpConstraintAtom)


// hkpRagdollMotorConstraintAtom ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpRagdollMotorConstraintAtom)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpRagdollMotorConstraintAtom)
    HK_TRACKER_MEMBER(hkpRagdollMotorConstraintAtom, m_motors, 0, "hkpConstraintMotor* [3]") // class hkpConstraintMotor* [3]
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpRagdollMotorConstraintAtom, s_libraryName, hkpConstraintAtom)


// hkpLinFrictionConstraintAtom ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpLinFrictionConstraintAtom, s_libraryName)


// hkpWheelFrictionConstraintAtom ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpWheelFrictionConstraintAtom)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Axle)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpWheelFrictionConstraintAtom)
    HK_TRACKER_MEMBER(hkpWheelFrictionConstraintAtom, m_axle, 0, "hkpWheelFrictionConstraintAtom::Axle*") // struct hkpWheelFrictionConstraintAtom::Axle*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpWheelFrictionConstraintAtom, s_libraryName, hkpConstraintAtom)


// Axle hkpWheelFrictionConstraintAtom
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpWheelFrictionConstraintAtom, Axle, s_libraryName)


// hkpLinMotorConstraintAtom ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpLinMotorConstraintAtom)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpLinMotorConstraintAtom)
    HK_TRACKER_MEMBER(hkpLinMotorConstraintAtom, m_motor, 0, "hkpConstraintMotor*") // class hkpConstraintMotor*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpLinMotorConstraintAtom, s_libraryName, hkpConstraintAtom)


// hkpPulleyConstraintAtom ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpPulleyConstraintAtom, s_libraryName)


// hkpRackAndPinionConstraintAtom ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpRackAndPinionConstraintAtom, s_libraryName)


// hkpCogWheelConstraintAtom ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpCogWheelConstraintAtom, s_libraryName)

#include <Physics/Constraint/Data/AngularFriction/hkpAngularFrictionConstraintData.h>


// hkpAngularFrictionConstraintData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpAngularFrictionConstraintData)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Runtime)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Atoms)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpAngularFrictionConstraintData)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpAngularFrictionConstraintData, s_libraryName, hkpConstraintData)


// Runtime hkpAngularFrictionConstraintData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpAngularFrictionConstraintData, Runtime, s_libraryName)


// Atoms hkpAngularFrictionConstraintData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpAngularFrictionConstraintData, Atoms, s_libraryName)

#include <Physics/Constraint/Data/BallAndSocket/hkpBallAndSocketConstraintData.h>


// hkpBallAndSocketConstraintData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpBallAndSocketConstraintData)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Runtime)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Atoms)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpBallAndSocketConstraintData)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpBallAndSocketConstraintData, s_libraryName, hkpConstraintData)


// Runtime hkpBallAndSocketConstraintData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpBallAndSocketConstraintData, Runtime, s_libraryName)


// Atoms hkpBallAndSocketConstraintData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpBallAndSocketConstraintData, Atoms, s_libraryName)

#include <Physics/Constraint/Data/Clearance/hkpLinearClearanceConstraintData.h>


// hkpLinearClearanceConstraintData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpLinearClearanceConstraintData)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Runtime)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Atoms)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Type)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpLinearClearanceConstraintData)
    HK_TRACKER_MEMBER(hkpLinearClearanceConstraintData, m_atoms, 0, "hkpLinearClearanceConstraintData::Atoms") // struct hkpLinearClearanceConstraintData::Atoms
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpLinearClearanceConstraintData, s_libraryName, hkpConstraintData)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpLinearClearanceConstraintData, Type, s_libraryName)


// Runtime hkpLinearClearanceConstraintData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpLinearClearanceConstraintData, Runtime, s_libraryName)


// Atoms hkpLinearClearanceConstraintData

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpLinearClearanceConstraintData::Atoms)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Axis)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpLinearClearanceConstraintData::Atoms)
    HK_TRACKER_MEMBER(hkpLinearClearanceConstraintData::Atoms, m_motor, 0, "hkpLinMotorConstraintAtom") // struct hkpLinMotorConstraintAtom
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpLinearClearanceConstraintData::Atoms, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpLinearClearanceConstraintData::Atoms, Axis, s_libraryName)

#include <Physics/Constraint/Data/CogWheel/hkpCogWheelConstraintData.h>


// hkpCogWheelConstraintData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCogWheelConstraintData)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Runtime)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Atoms)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCogWheelConstraintData)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpCogWheelConstraintData, s_libraryName, hkpConstraintData)


// Runtime hkpCogWheelConstraintData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpCogWheelConstraintData, Runtime, s_libraryName)


// Atoms hkpCogWheelConstraintData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpCogWheelConstraintData, Atoms, s_libraryName)

#include <Physics/Constraint/Data/DeformableFixed/hkpDeformableFixedConstraintData.h>


// hkpDeformableFixedConstraintData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpDeformableFixedConstraintData)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Runtime)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Atoms)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpDeformableFixedConstraintData)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpDeformableFixedConstraintData, s_libraryName, hkpConstraintData)


// Runtime hkpDeformableFixedConstraintData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpDeformableFixedConstraintData, Runtime, s_libraryName)


// Atoms hkpDeformableFixedConstraintData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpDeformableFixedConstraintData, Atoms, s_libraryName)

#include <Physics/Constraint/Data/Fixed/hkpFixedConstraintData.h>


// hkpFixedConstraintData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpFixedConstraintData)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Runtime)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Atoms)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpFixedConstraintData)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpFixedConstraintData, s_libraryName, hkpConstraintData)


// Runtime hkpFixedConstraintData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpFixedConstraintData, Runtime, s_libraryName)


// Atoms hkpFixedConstraintData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpFixedConstraintData, Atoms, s_libraryName)

#include <Physics/Constraint/Data/Hinge/hkpHingeConstraintData.h>


// hkpHingeConstraintData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpHingeConstraintData)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Runtime)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Atoms)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpHingeConstraintData)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpHingeConstraintData, s_libraryName, hkpConstraintData)


// Runtime hkpHingeConstraintData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpHingeConstraintData, Runtime, s_libraryName)


// Atoms hkpHingeConstraintData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpHingeConstraintData, Atoms, s_libraryName)

#include <Physics/Constraint/Data/HingeLimits/hkpHingeLimitsData.h>


// hkpHingeLimitsData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpHingeLimitsData)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Runtime)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Atoms)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpHingeLimitsData)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpHingeLimitsData, s_libraryName, hkpConstraintData)


// Runtime hkpHingeLimitsData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpHingeLimitsData, Runtime, s_libraryName)


// Atoms hkpHingeLimitsData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpHingeLimitsData, Atoms, s_libraryName)

#include <Physics/Constraint/Data/LimitedHinge/hkpLimitedHingeConstraintData.h>


// hkpLimitedHingeConstraintData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpLimitedHingeConstraintData)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Runtime)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Atoms)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpLimitedHingeConstraintData)
    HK_TRACKER_MEMBER(hkpLimitedHingeConstraintData, m_atoms, 0, "hkpLimitedHingeConstraintData::Atoms") // struct hkpLimitedHingeConstraintData::Atoms
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpLimitedHingeConstraintData, s_libraryName, hkpConstraintData)


// Runtime hkpLimitedHingeConstraintData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpLimitedHingeConstraintData, Runtime, s_libraryName)


// Atoms hkpLimitedHingeConstraintData

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpLimitedHingeConstraintData::Atoms)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Axis)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpLimitedHingeConstraintData::Atoms)
    HK_TRACKER_MEMBER(hkpLimitedHingeConstraintData::Atoms, m_angMotor, 0, "hkpAngMotorConstraintAtom") // struct hkpAngMotorConstraintAtom
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpLimitedHingeConstraintData::Atoms, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpLimitedHingeConstraintData::Atoms, Axis, s_libraryName)

#include <Physics/Constraint/Data/PointToPath/hkpLinearParametricCurve.h>


// hkpLinearParametricCurve ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpLinearParametricCurve)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpLinearParametricCurve)
    HK_TRACKER_MEMBER(hkpLinearParametricCurve, m_points, 0, "hkArray<hkVector4f, hkContainerHeapAllocator>") // hkArray< hkVector4f, struct hkContainerHeapAllocator >
    HK_TRACKER_MEMBER(hkpLinearParametricCurve, m_distance, 0, "hkArray<float, hkContainerHeapAllocator>") // hkArray< float, struct hkContainerHeapAllocator >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpLinearParametricCurve, s_libraryName, hkpParametricCurve)

#include <Physics/Constraint/Data/PointToPath/hkpParametricCurve.h>


// hkpParametricCurve ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpParametricCurve)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpParametricCurve)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpParametricCurve, s_libraryName, hkReferencedObject)

#include <Physics/Constraint/Data/PointToPath/hkpPointToPathConstraintData.h>


// hkpPointToPathConstraintData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpPointToPathConstraintData)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Runtime)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(OrientationConstraintType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpPointToPathConstraintData)
    HK_TRACKER_MEMBER(hkpPointToPathConstraintData, m_atoms, 0, "hkpBridgeAtoms") // struct hkpBridgeAtoms
    HK_TRACKER_MEMBER(hkpPointToPathConstraintData, m_path, 0, "hkpParametricCurve*") // class hkpParametricCurve*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpPointToPathConstraintData, s_libraryName, hkpConstraintData)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpPointToPathConstraintData, OrientationConstraintType, s_libraryName)


// Runtime hkpPointToPathConstraintData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpPointToPathConstraintData, Runtime, s_libraryName)

#include <Physics/Constraint/Data/PointToPlane/hkpPointToPlaneConstraintData.h>


// hkpPointToPlaneConstraintData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpPointToPlaneConstraintData)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Runtime)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Atoms)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpPointToPlaneConstraintData)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpPointToPlaneConstraintData, s_libraryName, hkpConstraintData)


// Runtime hkpPointToPlaneConstraintData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpPointToPlaneConstraintData, Runtime, s_libraryName)


// Atoms hkpPointToPlaneConstraintData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpPointToPlaneConstraintData, Atoms, s_libraryName)

#include <Physics/Constraint/Data/Prismatic/hkpPrismaticConstraintData.h>


// hkpPrismaticConstraintData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpPrismaticConstraintData)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Runtime)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Atoms)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpPrismaticConstraintData)
    HK_TRACKER_MEMBER(hkpPrismaticConstraintData, m_atoms, 0, "hkpPrismaticConstraintData::Atoms") // struct hkpPrismaticConstraintData::Atoms
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpPrismaticConstraintData, s_libraryName, hkpConstraintData)


// Runtime hkpPrismaticConstraintData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpPrismaticConstraintData, Runtime, s_libraryName)


// Atoms hkpPrismaticConstraintData

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpPrismaticConstraintData::Atoms)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Axis)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpPrismaticConstraintData::Atoms)
    HK_TRACKER_MEMBER(hkpPrismaticConstraintData::Atoms, m_motor, 0, "hkpLinMotorConstraintAtom") // struct hkpLinMotorConstraintAtom
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpPrismaticConstraintData::Atoms, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpPrismaticConstraintData::Atoms, Axis, s_libraryName)

#include <Physics/Constraint/Data/Pulley/hkpPulleyConstraintData.h>


// hkpPulleyConstraintData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpPulleyConstraintData)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Runtime)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Atoms)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpPulleyConstraintData)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpPulleyConstraintData, s_libraryName, hkpConstraintData)


// Runtime hkpPulleyConstraintData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpPulleyConstraintData, Runtime, s_libraryName)


// Atoms hkpPulleyConstraintData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpPulleyConstraintData, Atoms, s_libraryName)

#include <Physics/Constraint/Data/RackAndPinion/hkpRackAndPinionConstraintData.h>


// hkpRackAndPinionConstraintData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpRackAndPinionConstraintData)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Runtime)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Atoms)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Type)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpRackAndPinionConstraintData)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpRackAndPinionConstraintData, s_libraryName, hkpConstraintData)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpRackAndPinionConstraintData, Type, s_libraryName)


// Runtime hkpRackAndPinionConstraintData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpRackAndPinionConstraintData, Runtime, s_libraryName)


// Atoms hkpRackAndPinionConstraintData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpRackAndPinionConstraintData, Atoms, s_libraryName)

#include <Physics/Constraint/Data/Ragdoll/hkpRagdollConstraintData.h>


// hkpRagdollConstraintData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpRagdollConstraintData)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Runtime)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Atoms)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MotorIndex)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpRagdollConstraintData)
    HK_TRACKER_MEMBER(hkpRagdollConstraintData, m_atoms, 0, "hkpRagdollConstraintData::Atoms") // struct hkpRagdollConstraintData::Atoms
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpRagdollConstraintData, s_libraryName, hkpConstraintData)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpRagdollConstraintData, MotorIndex, s_libraryName)


// Runtime hkpRagdollConstraintData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpRagdollConstraintData, Runtime, s_libraryName)


// Atoms hkpRagdollConstraintData

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpRagdollConstraintData::Atoms)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Axis)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpRagdollConstraintData::Atoms)
    HK_TRACKER_MEMBER(hkpRagdollConstraintData::Atoms, m_ragdollMotors, 0, "hkpRagdollMotorConstraintAtom") // struct hkpRagdollMotorConstraintAtom
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpRagdollConstraintData::Atoms, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpRagdollConstraintData::Atoms, Axis, s_libraryName)

#include <Physics/Constraint/Data/RagdollLimits/hkpRagdollLimitsData.h>


// hkpRagdollLimitsData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpRagdollLimitsData)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Runtime)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Atoms)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpRagdollLimitsData)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpRagdollLimitsData, s_libraryName, hkpConstraintData)


// Runtime hkpRagdollLimitsData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpRagdollLimitsData, Runtime, s_libraryName)


// Atoms hkpRagdollLimitsData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpRagdollLimitsData, Atoms, s_libraryName)

#include <Physics/Constraint/Data/Rotational/hkpRotationalConstraintData.h>


// hkpRotationalConstraintData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpRotationalConstraintData)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Runtime)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Atoms)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpRotationalConstraintData)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpRotationalConstraintData, s_libraryName, hkpConstraintData)


// Runtime hkpRotationalConstraintData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpRotationalConstraintData, Runtime, s_libraryName)


// Atoms hkpRotationalConstraintData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpRotationalConstraintData, Atoms, s_libraryName)

#include <Physics/Constraint/Data/StiffSpring/hkpStiffSpringConstraintData.h>


// hkpStiffSpringConstraintData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpStiffSpringConstraintData)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Runtime)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Atoms)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpStiffSpringConstraintData)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpStiffSpringConstraintData, s_libraryName, hkpConstraintData)


// Runtime hkpStiffSpringConstraintData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpStiffSpringConstraintData, Runtime, s_libraryName)


// Atoms hkpStiffSpringConstraintData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpStiffSpringConstraintData, Atoms, s_libraryName)

#include <Physics/Constraint/Data/Wheel/hkpWheelConstraintData.h>


// hkpWheelConstraintData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpWheelConstraintData)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Runtime)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Atoms)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpWheelConstraintData)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpWheelConstraintData, s_libraryName, hkpConstraintData)


// Runtime hkpWheelConstraintData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpWheelConstraintData, Runtime, s_libraryName)


// Atoms hkpWheelConstraintData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpWheelConstraintData, Atoms, s_libraryName)

#include <Physics/Constraint/Data/WheelFriction/hkpWheelFrictionConstraintData.h>


// hkpWheelFrictionConstraintData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpWheelFrictionConstraintData)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Runtime)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Atoms)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpWheelFrictionConstraintData)
    HK_TRACKER_MEMBER(hkpWheelFrictionConstraintData, m_atoms, 0, "hkpWheelFrictionConstraintData::Atoms") // struct hkpWheelFrictionConstraintData::Atoms
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpWheelFrictionConstraintData, s_libraryName, hkpConstraintData)


// Runtime hkpWheelFrictionConstraintData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpWheelFrictionConstraintData, Runtime, s_libraryName)


// Atoms hkpWheelFrictionConstraintData

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpWheelFrictionConstraintData::Atoms)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpWheelFrictionConstraintData::Atoms)
    HK_TRACKER_MEMBER(hkpWheelFrictionConstraintData::Atoms, m_friction, 0, "hkpWheelFrictionConstraintAtom") // struct hkpWheelFrictionConstraintAtom
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpWheelFrictionConstraintData::Atoms, s_libraryName)

#include <Physics/Constraint/Data/hkpConstraintData.h>


// hkpConstraintData ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConstraintData)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ConstraintInfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(RuntimeInfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(ConstraintType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConstraintData)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpConstraintData, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpConstraintData, ConstraintType, s_libraryName)


// ConstraintInfo hkpConstraintData

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConstraintData::ConstraintInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConstraintData::ConstraintInfo)
    HK_TRACKER_MEMBER(hkpConstraintData::ConstraintInfo, m_atoms, 0, "hkpConstraintAtom*") // struct hkpConstraintAtom*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpConstraintData::ConstraintInfo, s_libraryName, hkpConstraintInfo)


// RuntimeInfo hkpConstraintData
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpConstraintData, RuntimeInfo, s_libraryName)

#include <Physics/Constraint/Data/hkpConstraintInfo.h>


// End hkpJacobianSchemaInfo
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkpJacobianSchemaInfo::End, s_libraryName, hkpJacobianSchemaInfo_End)


// Header hkpJacobianSchemaInfo
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkpJacobianSchemaInfo::Header, s_libraryName, hkpJacobianSchemaInfo_Header)


// Goto hkpJacobianSchemaInfo
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkpJacobianSchemaInfo::Goto, s_libraryName, hkpJacobianSchemaInfo_Goto)


// ShiftSolverResults hkpJacobianSchemaInfo
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkpJacobianSchemaInfo::ShiftSolverResults, s_libraryName, hkpJacobianSchemaInfo_ShiftSolverResults)


// Bilateral1D hkpJacobianSchemaInfo
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkpJacobianSchemaInfo::Bilateral1D, s_libraryName, hkpJacobianSchemaInfo_Bilateral1D)


// BilateralUserTau1D hkpJacobianSchemaInfo
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkpJacobianSchemaInfo::BilateralUserTau1D, s_libraryName, hkpJacobianSchemaInfo_BilateralUserTau1D)


// LinearLimits1D hkpJacobianSchemaInfo
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkpJacobianSchemaInfo::LinearLimits1D, s_libraryName, hkpJacobianSchemaInfo_LinearLimits1D)


// Friction1D hkpJacobianSchemaInfo
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkpJacobianSchemaInfo::Friction1D, s_libraryName, hkpJacobianSchemaInfo_Friction1D)


// LinearMotor1D hkpJacobianSchemaInfo
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkpJacobianSchemaInfo::LinearMotor1D, s_libraryName, hkpJacobianSchemaInfo_LinearMotor1D)


// StableBallSocket hkpJacobianSchemaInfo
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkpJacobianSchemaInfo::StableBallSocket, s_libraryName, hkpJacobianSchemaInfo_StableBallSocket)


// NpStableBallSocket hkpJacobianSchemaInfo
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkpJacobianSchemaInfo::NpStableBallSocket, s_libraryName, hkpJacobianSchemaInfo_NpStableBallSocket)


// StableAngular3D hkpJacobianSchemaInfo
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkpJacobianSchemaInfo::StableAngular3D, s_libraryName, hkpJacobianSchemaInfo_StableAngular3D)


// NpStableAngular3D hkpJacobianSchemaInfo
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkpJacobianSchemaInfo::NpStableAngular3D, s_libraryName, hkpJacobianSchemaInfo_NpStableAngular3D)


// Pulley1D hkpJacobianSchemaInfo
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkpJacobianSchemaInfo::Pulley1D, s_libraryName, hkpJacobianSchemaInfo_Pulley1D)


// Angular1D hkpJacobianSchemaInfo
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkpJacobianSchemaInfo::Angular1D, s_libraryName, hkpJacobianSchemaInfo_Angular1D)


// AngularLimits1D hkpJacobianSchemaInfo
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkpJacobianSchemaInfo::AngularLimits1D, s_libraryName, hkpJacobianSchemaInfo_AngularLimits1D)


// AngularFriction1D hkpJacobianSchemaInfo
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkpJacobianSchemaInfo::AngularFriction1D, s_libraryName, hkpJacobianSchemaInfo_AngularFriction1D)


// AngularMotor1D hkpJacobianSchemaInfo
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkpJacobianSchemaInfo::AngularMotor1D, s_libraryName, hkpJacobianSchemaInfo_AngularMotor1D)


// SingleContact hkpJacobianSchemaInfo
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkpJacobianSchemaInfo::SingleContact, s_libraryName, hkpJacobianSchemaInfo_SingleContact)


// PairContact hkpJacobianSchemaInfo
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkpJacobianSchemaInfo::PairContact, s_libraryName, hkpJacobianSchemaInfo_PairContact)


// WheelFriction hkpJacobianSchemaInfo
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkpJacobianSchemaInfo::WheelFriction, s_libraryName, hkpJacobianSchemaInfo_WheelFriction)


// Friction2D hkpJacobianSchemaInfo
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkpJacobianSchemaInfo::Friction2D, s_libraryName, hkpJacobianSchemaInfo_Friction2D)


// Friction3D hkpJacobianSchemaInfo
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkpJacobianSchemaInfo::Friction3D, s_libraryName, hkpJacobianSchemaInfo_Friction3D)


// RollingFriction2D hkpJacobianSchemaInfo
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkpJacobianSchemaInfo::RollingFriction2D, s_libraryName, hkpJacobianSchemaInfo_RollingFriction2D)


// SetMass hkpJacobianSchemaInfo
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkpJacobianSchemaInfo::SetMass, s_libraryName, hkpJacobianSchemaInfo_SetMass)


// AddVelocity hkpJacobianSchemaInfo
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkpJacobianSchemaInfo::AddVelocity, s_libraryName, hkpJacobianSchemaInfo_AddVelocity)


// SetCenterOfMass hkpJacobianSchemaInfo
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkpJacobianSchemaInfo::SetCenterOfMass, s_libraryName, hkpJacobianSchemaInfo_SetCenterOfMass)


// StiffSpringChain hkpJacobianSchemaInfo
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkpJacobianSchemaInfo::StiffSpringChain, s_libraryName, hkpJacobianSchemaInfo_StiffSpringChain)


// BallSocketChain hkpJacobianSchemaInfo
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkpJacobianSchemaInfo::BallSocketChain, s_libraryName, hkpJacobianSchemaInfo_BallSocketChain)


// StabilizedBallSocketChain hkpJacobianSchemaInfo
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkpJacobianSchemaInfo::StabilizedBallSocketChain, s_libraryName, hkpJacobianSchemaInfo_StabilizedBallSocketChain)


// PoweredChain hkpJacobianSchemaInfo
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkpJacobianSchemaInfo::PoweredChain, s_libraryName, hkpJacobianSchemaInfo_PoweredChain)


// StableStiffSpring hkpJacobianSchemaInfo
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkpJacobianSchemaInfo::StableStiffSpring, s_libraryName, hkpJacobianSchemaInfo_StableStiffSpring)


// NpStableStiffSpring hkpJacobianSchemaInfo
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkpJacobianSchemaInfo::NpStableStiffSpring, s_libraryName, hkpJacobianSchemaInfo_NpStableStiffSpring)


// DeformableLinear3D hkpJacobianSchemaInfo
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkpJacobianSchemaInfo::DeformableLinear3D, s_libraryName, hkpJacobianSchemaInfo_DeformableLinear3D)


// NpDeformableLinear3D hkpJacobianSchemaInfo
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkpJacobianSchemaInfo::NpDeformableLinear3D, s_libraryName, hkpJacobianSchemaInfo_NpDeformableLinear3D)


// DeformableAngular3D hkpJacobianSchemaInfo
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkpJacobianSchemaInfo::DeformableAngular3D, s_libraryName, hkpJacobianSchemaInfo_DeformableAngular3D)


// NpDeformableAngular3D hkpJacobianSchemaInfo
HK_TRACKER_IMPLEMENT_NAMESPACE_SIMPLE(hkpJacobianSchemaInfo::NpDeformableAngular3D, s_libraryName, hkpJacobianSchemaInfo_NpDeformableAngular3D)


// hkpConstraintInfo ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpConstraintInfo, s_libraryName)

#include <Physics/Constraint/Motor/Callback/hkpCallbackConstraintMotor.h>


// hkpCallbackConstraintMotor ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCallbackConstraintMotor)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(CallbackType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCallbackConstraintMotor)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpCallbackConstraintMotor, s_libraryName, hkpLimitedForceConstraintMotor)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpCallbackConstraintMotor, CallbackType, s_libraryName)

#include <Physics/Constraint/Motor/LimitedForce/hkpLimitedForceConstraintMotor.h>


// hkpLimitedForceConstraintMotor ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpLimitedForceConstraintMotor)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpLimitedForceConstraintMotor)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpLimitedForceConstraintMotor, s_libraryName, hkpConstraintMotor)

#include <Physics/Constraint/Motor/Position/hkpPositionConstraintMotor.h>


// hkpPositionConstraintMotor ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpPositionConstraintMotor)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpPositionConstraintMotor)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpPositionConstraintMotor, s_libraryName, hkpLimitedForceConstraintMotor)

#include <Physics/Constraint/Motor/SpringDamper/hkpSpringDamperConstraintMotor.h>


// hkpSpringDamperConstraintMotor ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSpringDamperConstraintMotor)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpSpringDamperConstraintMotor)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpSpringDamperConstraintMotor, s_libraryName, hkpLimitedForceConstraintMotor)

#include <Physics/Constraint/Motor/Velocity/hkpVelocityConstraintMotor.h>


// hkpVelocityConstraintMotor ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVelocityConstraintMotor)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVelocityConstraintMotor)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpVelocityConstraintMotor, s_libraryName, hkpLimitedForceConstraintMotor)

#include <Physics/Constraint/Motor/hkpConstraintMotor.h>


// hkpConstraintMotor ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConstraintMotor)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(MotorType)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConstraintMotor)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_ABSTRACT(hkpConstraintMotor, s_libraryName, hkReferencedObject)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpConstraintMotor, MotorType, s_libraryName)


// hkpMaxSizeConstraintMotor ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpMaxSizeConstraintMotor)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpMaxSizeConstraintMotor)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpMaxSizeConstraintMotor, s_libraryName, hkpConstraintMotor)

#include <Physics/Constraint/Visualize/Drawer/hkpBallSocketDrawer.h>


// hkpBallSocketDrawer ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpBallSocketDrawer, s_libraryName)

#include <Physics/Constraint/Visualize/Drawer/hkpCogWheelDrawer.h>


// hkpCogWheelDrawer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpCogWheelDrawer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpCogWheelDrawer)
    HK_TRACKER_MEMBER(hkpCogWheelDrawer, m_cogWheels, 0, "hkDisplaySemiCircle [2]") // class hkDisplaySemiCircle [2]
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpCogWheelDrawer, s_libraryName, hkpConstraintDrawer)

#include <Physics/Constraint/Visualize/Drawer/hkpConstraintDrawer.h>


// hkpConstraintDrawer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConstraintDrawer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConstraintDrawer)
    HK_TRACKER_MEMBER(hkpConstraintDrawer, m_primitiveDrawer, 0, "hkpPrimitiveDrawer") // class hkpPrimitiveDrawer
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpConstraintDrawer, s_libraryName)

#include <Physics/Constraint/Visualize/Drawer/hkpDeformableFixedConstraintDrawer.h>


// hkpDeformableFixedConstraintDrawer ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpDeformableFixedConstraintDrawer, s_libraryName)

#include <Physics/Constraint/Visualize/Drawer/hkpFixedConstraintDrawer.h>


// hkpFixedConstraintDrawer ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpFixedConstraintDrawer, s_libraryName)

#include <Physics/Constraint/Visualize/Drawer/hkpHingeDrawer.h>


// hkpHingeDrawer ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpHingeDrawer, s_libraryName)

#include <Physics/Constraint/Visualize/Drawer/hkpHingeLimitsDrawer.h>


// hkpHingeLimitsDrawer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpHingeLimitsDrawer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpHingeLimitsDrawer)
    HK_TRACKER_MEMBER(hkpHingeLimitsDrawer, m_angularLimit, 0, "hkDisplaySemiCircle") // class hkDisplaySemiCircle
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpHingeLimitsDrawer, s_libraryName, hkpConstraintDrawer)

#include <Physics/Constraint/Visualize/Drawer/hkpLimitedHingeDrawer.h>


// hkpLimitedHingeDrawer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpLimitedHingeDrawer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpLimitedHingeDrawer)
    HK_TRACKER_MEMBER(hkpLimitedHingeDrawer, m_angularLimit, 0, "hkDisplaySemiCircle") // class hkDisplaySemiCircle
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpLimitedHingeDrawer, s_libraryName, hkpConstraintDrawer)

#include <Physics/Constraint/Visualize/Drawer/hkpPointToPathDrawer.h>


// hkpPointToPathDrawer ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpPointToPathDrawer, s_libraryName)

#include <Physics/Constraint/Visualize/Drawer/hkpPointToPlaneDrawer.h>


// hkpPointToPlaneDrawer ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpPointToPlaneDrawer, s_libraryName)

#include <Physics/Constraint/Visualize/Drawer/hkpPrimitiveDrawer.h>


// hkpPrimitiveDrawer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpPrimitiveDrawer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpPrimitiveDrawer)
    HK_TRACKER_MEMBER(hkpPrimitiveDrawer, m_displayHandler, 0, "hkDebugDisplayHandler*") // class hkDebugDisplayHandler*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpPrimitiveDrawer, s_libraryName)

#include <Physics/Constraint/Visualize/Drawer/hkpPrismaticDrawer.h>


// hkpPrismaticDrawer ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpPrismaticDrawer, s_libraryName)

#include <Physics/Constraint/Visualize/Drawer/hkpPulleyDrawer.h>


// hkpPulleyDrawer ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpPulleyDrawer, s_libraryName)

#include <Physics/Constraint/Visualize/Drawer/hkpRackAndPinionDrawer.h>


// hkpRackAndPinionDrawer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpRackAndPinionDrawer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpRackAndPinionDrawer)
    HK_TRACKER_MEMBER(hkpRackAndPinionDrawer, m_cogWheel, 0, "hkDisplaySemiCircle") // class hkDisplaySemiCircle
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpRackAndPinionDrawer, s_libraryName, hkpConstraintDrawer)

#include <Physics/Constraint/Visualize/Drawer/hkpRagdollDrawer.h>


// hkpRagdollDrawer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpRagdollDrawer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpRagdollDrawer)
    HK_TRACKER_MEMBER(hkpRagdollDrawer, m_twistCone, 0, "hkDisplayCone") // class hkDisplayCone
    HK_TRACKER_MEMBER(hkpRagdollDrawer, m_planeCone1, 0, "hkDisplayCone") // class hkDisplayCone
    HK_TRACKER_MEMBER(hkpRagdollDrawer, m_planeCone2, 0, "hkDisplayCone") // class hkDisplayCone
    HK_TRACKER_MEMBER(hkpRagdollDrawer, m_plane, 0, "hkDisplayPlane") // class hkDisplayPlane
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpRagdollDrawer, s_libraryName, hkpConstraintDrawer)

#include <Physics/Constraint/Visualize/Drawer/hkpRagdollLimitsDrawer.h>


// hkpRagdollLimitsDrawer ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpRagdollLimitsDrawer)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpRagdollLimitsDrawer)
    HK_TRACKER_MEMBER(hkpRagdollLimitsDrawer, m_twistCone, 0, "hkDisplayCone") // class hkDisplayCone
    HK_TRACKER_MEMBER(hkpRagdollLimitsDrawer, m_planeCone1, 0, "hkDisplayCone") // class hkDisplayCone
    HK_TRACKER_MEMBER(hkpRagdollLimitsDrawer, m_planeCone2, 0, "hkDisplayCone") // class hkDisplayCone
    HK_TRACKER_MEMBER(hkpRagdollLimitsDrawer, m_plane, 0, "hkDisplayPlane") // class hkDisplayPlane
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpRagdollLimitsDrawer, s_libraryName, hkpConstraintDrawer)

#include <Physics/Constraint/Visualize/Drawer/hkpStiffSpringDrawer.h>


// hkpStiffSpringDrawer ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpStiffSpringDrawer, s_libraryName)

#include <Physics/Constraint/Visualize/Drawer/hkpWheelDrawer.h>


// hkpWheelDrawer ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpWheelDrawer, s_libraryName)

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
