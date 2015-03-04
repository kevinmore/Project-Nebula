/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#include <Common/Base/hkBase.h>
static const char s_libraryName[] = "hkpConstraintSolver";
#include <Common/Base/Memory/Tracker/hkTrackerClassDefinition.h>

void HK_CALL hkpConstraintSolverRegister() {}

#include <Physics/ConstraintSolver/Accumulator/hkpVelocityAccumulator.h>


// hkpVelocityAccumulator ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpVelocityAccumulator, s_libraryName)


// hkpVelocityAccumulator2 ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpVelocityAccumulator2, s_libraryName)

#include <Physics/ConstraintSolver/Constraint/Motor/hkpMotorConstraintInfo.h>


// hkp1dBilateralConstraintStatus ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkp1dBilateralConstraintStatus, s_libraryName)


// hkp1dConstraintMotorInfo ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkp1dConstraintMotorInfo, s_libraryName)


// hkpConstraintMotorInput ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConstraintMotorInput)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConstraintMotorInput)
    HK_TRACKER_MEMBER(hkpConstraintMotorInput, m_stepInfo, 0, "hkPadSpu<hkpConstraintQueryStepInfo*>") // class hkPadSpu< const class hkpConstraintQueryStepInfo* >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpConstraintMotorInput, s_libraryName, hkp1dBilateralConstraintStatus)


// hkpConstraintMotorOutput ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpConstraintMotorOutput, s_libraryName)

#include <Physics/ConstraintSolver/Constraint/hkpConstraintQueryIn.h>


// hkpConstraintQueryStepInfo ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpConstraintQueryStepInfo, s_libraryName)


// hkpConstraintQueryIn ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConstraintQueryIn)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConstraintQueryIn)
    HK_TRACKER_MEMBER(hkpConstraintQueryIn, m_bodyA, 0, "hkPadSpu<hkpVelocityAccumulator*>") // class hkPadSpu< const class hkpVelocityAccumulator* >
    HK_TRACKER_MEMBER(hkpConstraintQueryIn, m_bodyB, 0, "hkPadSpu<hkpVelocityAccumulator*>") // class hkPadSpu< const class hkpVelocityAccumulator* >
    HK_TRACKER_MEMBER(hkpConstraintQueryIn, m_transformA, 0, "hkPadSpu<hkTransformf*>") // class hkPadSpu< const hkTransformf* >
    HK_TRACKER_MEMBER(hkpConstraintQueryIn, m_transformB, 0, "hkPadSpu<hkTransformf*>") // class hkPadSpu< const hkTransformf* >
    HK_TRACKER_MEMBER(hkpConstraintQueryIn, m_constraintInstance, 0, "hkPadSpu<hkpConstraintInstance*>") // class hkPadSpu< class hkpConstraintInstance* >
    HK_TRACKER_MEMBER(hkpConstraintQueryIn, m_violatedConstraints, 0, "hkPadSpu<hkpViolatedConstraintArray*>") // class hkPadSpu< struct hkpViolatedConstraintArray* >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpConstraintQueryIn, s_libraryName, hkpConstraintQueryStepInfo)

#include <Physics/ConstraintSolver/Constraint/hkpConstraintQueryOut.h>


// hkpConstraintQueryOut ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpConstraintQueryOut)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpConstraintQueryOut)
    HK_TRACKER_MEMBER(hkpConstraintQueryOut, m_jacobianSchemas, 0, "hkPadSpu<hkpJacobianSchema*>") // class hkPadSpu< class hkpJacobianSchema* >
    HK_TRACKER_MEMBER(hkpConstraintQueryOut, m_constraintRuntime, 0, "hkPadSpu<hkpConstraintRuntime*>") // class hkPadSpu< struct hkpConstraintRuntime* >
    HK_TRACKER_MEMBER(hkpConstraintQueryOut, m_constraintRuntimeInMainMemory, 0, "hkPadSpu<void*>") // class hkPadSpu< void* >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpConstraintQueryOut, s_libraryName)

#include <Physics/ConstraintSolver/Solve/hkpSolve.h>


// hkpImpulseLimitBreachedElem ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpImpulseLimitBreachedElem)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpImpulseLimitBreachedElem)
    HK_TRACKER_MEMBER(hkpImpulseLimitBreachedElem, m_constraintInstance, 0, "hkpConstraintInstance*") // class hkpConstraintInstance*
    HK_TRACKER_MEMBER(hkpImpulseLimitBreachedElem, m_solverResult, 0, "hkpSolverResults*") // class hkpSolverResults*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpImpulseLimitBreachedElem, s_libraryName)


// hkpImpulseLimitBreachedHeader ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpImpulseLimitBreachedHeader)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpImpulseLimitBreachedHeader)
    HK_TRACKER_MEMBER(hkpImpulseLimitBreachedHeader, m_next, 0, "hkpImpulseLimitBreachedHeader*") // struct hkpImpulseLimitBreachedHeader*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpImpulseLimitBreachedHeader, s_libraryName)

#include <Physics/ConstraintSolver/Solve/hkpSolverElemTemp.h>


// hkpSolverElemTemp ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpSolverElemTemp, s_libraryName)

#include <Physics/ConstraintSolver/Solve/hkpSolverInfo.h>


// hkpSolverInfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpSolverInfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(DeactivationInfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(DeactivationClass)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkpSolverInfo, s_libraryName)

HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpSolverInfo, DeactivationClass, s_libraryName)


// DeactivationInfo hkpSolverInfo
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpSolverInfo, DeactivationInfo, s_libraryName)


// hkp1dMotorSolverInfo ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkp1dMotorSolverInfo, s_libraryName)


// hkp3dAngularMotorSolverInfo ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkp3dAngularMotorSolverInfo, s_libraryName)


// hkpViolatedConstraintArray ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpViolatedConstraintArray)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpViolatedConstraintArray)
    HK_TRACKER_MEMBER(hkpViolatedConstraintArray, m_constraints, 0, "hkpConstraintInstance* [128]") // class hkpConstraintInstance* [128]
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpViolatedConstraintArray, s_libraryName)

#include <Physics/ConstraintSolver/Solve/hkpSolverResults.h>


// hkpSolverResults ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpSolverResults, s_libraryName)

#include <Physics/ConstraintSolver/VehicleFriction/hkpVehicleFriction.h>


// hkpVehicleStepInfo ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpVehicleStepInfo, s_libraryName)


// hkpVehicleFrictionDescription ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleFrictionDescription)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(Cinfo)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(AxisDescription)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpVehicleFrictionDescription)
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS(hkpVehicleFrictionDescription, s_libraryName, hkReferencedObject)


// Cinfo hkpVehicleFrictionDescription
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpVehicleFrictionDescription, Cinfo, s_libraryName)


// AxisDescription hkpVehicleFrictionDescription
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpVehicleFrictionDescription, AxisDescription, s_libraryName)


// hkpVehicleFrictionStatus ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpVehicleFrictionStatus)
    HK_TRACKER_DECLARE_CHILD_SIMPLE(AxisStatus)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_IMPLEMENT_SIMPLE(hkpVehicleFrictionStatus, s_libraryName)


// AxisStatus hkpVehicleFrictionStatus
HK_TRACKER_IMPLEMENT_CHILD_SIMPLE(hkpVehicleFrictionStatus, AxisStatus, s_libraryName)

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
