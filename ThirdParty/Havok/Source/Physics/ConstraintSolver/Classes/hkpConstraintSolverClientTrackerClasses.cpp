/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
//HK_REFLECTION_PARSER_EXCLUDE_FILE

#include <Common/Base/hkBase.h>
static const char s_libraryName[] = "hkpConstraintSolverClient";
#include <Common/Base/Memory/Tracker/hkTrackerClassDefinition.h>

void HK_CALL hkpConstraintSolverClientRegister() {}

#include <Physics/ConstraintSolver/Constraint/Bilateral/hkp1dBilateralConstraintInfo.h>


// hkp1dLinearBilateralConstraintInfo ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkp1dLinearBilateralConstraintInfo, s_libraryName)


// hkp1dLinearBilateralUserTauConstraintInfo ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkp1dLinearBilateralUserTauConstraintInfo, s_libraryName)


// hkp1dAngularBilateralConstraintInfo ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkp1dAngularBilateralConstraintInfo, s_libraryName)


// hkp1dLinearLimitInfo ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkp1dLinearLimitInfo, s_libraryName)


// hkp1dAngularFrictionInfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkp1dAngularFrictionInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkp1dAngularFrictionInfo)
    HK_TRACKER_MEMBER(hkp1dAngularFrictionInfo, m_constrainedDofW, 0, "hkPadSpu<hkVector4f*>") // class hkPadSpu< const hkVector4f* >
    HK_TRACKER_MEMBER(hkp1dAngularFrictionInfo, m_lastSolverResults, 0, "hkPadSpu<hkpSolverResults*>") // class hkPadSpu< class hkpSolverResults* >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkp1dAngularFrictionInfo, s_libraryName)


// hkp1dLinearFrictionInfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkp1dLinearFrictionInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkp1dLinearFrictionInfo)
    HK_TRACKER_MEMBER(hkp1dLinearFrictionInfo, m_lastSolverResults, 0, "hkPadSpu<hkpSolverResults*>") // class hkPadSpu< class hkpSolverResults* >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkp1dLinearFrictionInfo, s_libraryName)


// hkp1dAngularLimitInfo ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkp1dAngularLimitInfo, s_libraryName)


// hkpPulleyConstraintInfo ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpPulleyConstraintInfo, s_libraryName)

#include <Physics/ConstraintSolver/Constraint/Chain/hkpChainConstraintInfo.h>


// hkpConstraintChainTriple ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpConstraintChainTriple, s_libraryName)


// hkpConstraintChainMatrixTriple ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpConstraintChainMatrixTriple, s_libraryName)


// hkpVelocityAccumulatorOffset ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpVelocityAccumulatorOffset, s_libraryName)

#include <Physics/ConstraintSolver/Constraint/Chain/hkpPoweredChainSolverUtil.h>


// hkpConstraintChainMatrix6Triple ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpConstraintChainMatrix6Triple, s_libraryName)


// hkpChainSolverInfo ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpChainSolverInfo)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpChainSolverInfo)
    HK_TRACKER_MEMBER(hkpChainSolverInfo, m_j, 0, "hkPadSpu<hkp1Lin2AngJacobian*>") // class hkPadSpu< class hkp1Lin2AngJacobian* >
    HK_TRACKER_MEMBER(hkpChainSolverInfo, m_jAng, 0, "hkPadSpu<hkp2AngJacobian*>") // class hkPadSpu< class hkp2AngJacobian* >
    HK_TRACKER_MEMBER(hkpChainSolverInfo, m_va, 0, "hkPadSpu<hkpVelocityAccumulatorOffset*>") // class hkPadSpu< class hkpVelocityAccumulatorOffset* >
    HK_TRACKER_MEMBER(hkpChainSolverInfo, m_accumsBase, 0, "hkPadSpu<hkpVelocityAccumulator*>") // class hkPadSpu< class hkpVelocityAccumulator* >
    HK_TRACKER_MEMBER(hkpChainSolverInfo, m_triples, 0, "hkPadSpu<hkpConstraintChainMatrix6Triple*>") // class hkPadSpu< struct hkpConstraintChainMatrix6Triple* >
    HK_TRACKER_MEMBER(hkpChainSolverInfo, m_motorsState, 0, "hkPadSpu<hkp3dAngularMotorSolverInfo*>") // class hkPadSpu< struct hkp3dAngularMotorSolverInfo* >
    HK_TRACKER_MEMBER(hkpChainSolverInfo, m_gTempBuffer, 0, "hkPadSpu<hkVector8f*>") // class hkPadSpu< class hkVector8f* >
    HK_TRACKER_MEMBER(hkpChainSolverInfo, m_velocitiesBuffer, 0, "hkPadSpu<hkVector8f*>") // class hkPadSpu< class hkVector8f* >
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpChainSolverInfo, s_libraryName)


// hkpCfmParam ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpCfmParam, s_libraryName)


// hkpPoweredChainBuildJacobianParams ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpPoweredChainBuildJacobianParams)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpPoweredChainBuildJacobianParams)
    HK_TRACKER_MEMBER(hkpPoweredChainBuildJacobianParams, m_accumulators, 0, "hkpVelocityAccumulatorOffset*") // class hkpVelocityAccumulatorOffset*
    HK_TRACKER_MEMBER(hkpPoweredChainBuildJacobianParams, m_accumsBase, 0, "hkpVelocityAccumulator*") // const class hkpVelocityAccumulator*
    HK_TRACKER_MEMBER(hkpPoweredChainBuildJacobianParams, m_motorsState, 0, "hkp3dAngularMotorSolverInfo*") // struct hkp3dAngularMotorSolverInfo*
    HK_TRACKER_MEMBER(hkpPoweredChainBuildJacobianParams, m_childConstraintStatusWriteBackBuffer, 0, "hkUint8*") // hkUint8*
    HK_TRACKER_MEMBER(hkpPoweredChainBuildJacobianParams, m_jacobiansEnd, 0, "hkp2AngJacobian*") // class hkp2AngJacobian*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpPoweredChainBuildJacobianParams, s_libraryName)

#include <Physics/ConstraintSolver/Jacobian/hkpJacobianElement.h>


// hkp2AngJacobian ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkp2AngJacobian, s_libraryName)


// hkp1Lin2AngJacobian ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkp1Lin2AngJacobian, s_libraryName)


// hkpJacDouble2Bil ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpJacDouble2Bil, s_libraryName)


// hkpJacTriple2Bil1Ang ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpJacTriple2Bil1Ang, s_libraryName)


// hkp2Lin2AngJacobian ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkp2Lin2AngJacobian, s_libraryName)

#include <Physics/ConstraintSolver/Jacobian/hkpJacobianHeaderSchema.h>


// hkpJacobianHeaderSchema ::

HK_TRACKER_DECLARE_CLASS_BEGIN(hkpJacobianHeaderSchema)
HK_TRACKER_DECLARE_CLASS_END

HK_TRACKER_CLASS_MEMBERS_BEGIN(hkpJacobianHeaderSchema)
    HK_TRACKER_MEMBER(hkpJacobianHeaderSchema, m_solverResultInMainMemory, 0, "hkpSolverResults*") // class hkpSolverResults*
    HK_TRACKER_MEMBER(hkpJacobianHeaderSchema, m_constraintInstance, 0, "hkpConstraintInstance*") // class hkpConstraintInstance*
HK_TRACKER_CLASS_MEMBERS_END()
HK_TRACKER_IMPLEMENT_CLASS_BASE(hkpJacobianHeaderSchema, s_libraryName)


// hkpJacobianGotoSchema ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpJacobianGotoSchema, s_libraryName)

#include <Physics/ConstraintSolver/Jacobian/hkpJacobianSchema.h>


// hkpJacobianSchema ::
HK_TRACKER_IMPLEMENT_SIMPLE(hkpJacobianSchema, s_libraryName)

/*
 * Havok SDK - Product file, BUILD(#20130912)
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
