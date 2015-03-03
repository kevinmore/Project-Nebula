/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#ifndef HK_E_CONTACT_SOLVER_LOG_H
#define HK_E_CONTACT_SOLVER_LOG_H

#include <Physics/Physics/hknpTypes.h>

#include <Common/Base/Container/BlockStream/hkBlockStream.h>


struct hknpSolverBody;
struct hknpSolverJac;
class hknpSolverVelocity;
class hknpSolverSumVelocity;
class hknpConstraintSolverJacobianReader;

extern hknpBodyId* g_npBodiesToDebugPrint;
extern int g_npNumBodiesToDebugPrint;

class hkpRigidBody;
extern hkArray<hkpRigidBody*>* g_npRigidBodiesToDebugPrint;

class hknpContactSolverLog
{
	public:

		//static void HK_CALL debugPrintJacobian(hknpSolverVelocity* solverVel, hknpSolverSumVelocity* solverSumVel, const hknpMxContactJacobian* jac, int jacobianIdx, int manifoldIndexInJacobian);
		//static void HK_CALL debugPrintJacobians(hknpSolverVelocity* solverVel, hknpSolverSumVelocity* solverSumVel, hkBlockStream<hknpMxContactJacobian>::Reader& reader );
		//static void HK_CALL debugPrintSolverVelocities(hknpSolverVelocity* solverVel, hknpSolverSumVelocity* solverSumVel, int numSolverVelocities, int stepsDone, int numSteps, hkReal invIntegrateVelocityFactor );
		//static void HK_CALL debugPrintImpulsesApplied(int jacobianId, hknpSolverVelocity* solverVel, hknpSolverSumVelocity* solverSumVel, const hkInt32* solverVelAOffsets, const hkInt32* solverVelBOffsets, hkReal impulses[7][4]);

	private:
		static void HK_CALL debugPrintJacobian(const hkArray<int>& solverVelToMotion, hknpMotion* motions, const hknpMxContactJacobian* jac, int jacobianIdx, int manifoldIndexInJacobian);
		static void HK_CALL debugPrintJacobians(const hkArray<int>& solverVelToMotion, hknpMotion* motions, hknpConstraintSolverJacobianReader& reader );
		static void HK_CALL debugPrintMotions(hknpMotion* motions, int numMotions, int stepsDone, int numSteps, hkReal invIntegrateVelocityFactor );
		static void HK_CALL debugPrintImpulsesApplied(const hkArray<int>& solverVelToMotion, int jacobianId, hknpMotion* motions, const hkInt32* solverVelAOffsets, const hkInt32* solverVelBOffsets, hkReal impulses[7][4]);
};




#endif // HK_E_CONTACT_SOLVER_LOG_H

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
