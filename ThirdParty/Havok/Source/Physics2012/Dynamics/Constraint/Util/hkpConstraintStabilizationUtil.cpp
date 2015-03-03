/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Dynamics/World/hkpPhysicsSystem.h>
#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>
#include <Physics/Constraint/Atom/Bridge/hkpBridgeConstraintAtom.h>
#include <Physics2012/Dynamics/Constraint/Atom/hkpSimpleContactConstraintAtom.h>
#include <Physics2012/Dynamics/Constraint/Atom/hkpModifierConstraintAtom.h>
#include <Physics/Constraint/Data/hkpConstraintDataUtils.h>
#include <Physics2012/Dynamics/Constraint/Util/hkpConstraintStabilizationUtil.h>
#include <Common/Base/Math/Matrix/hkMatrix3Util.h>

//
//	Collects all rigid bodies in the given simulation island

static void HK_CALL collectRigidBodies(const hkpSimulationIsland* island, hkArray<hkpRigidBody*>& rigidBodiesOut)
{
	const hkArray<hkpEntity*>& entities = island->getEntities();
	for (int ei = entities.getSize() - 1; ei >= 0; ei--)
	{
		hkpEntity* entity = const_cast<hkpEntity*>(entities[ei]);
		hkpRigidBody* rigidBody = (hkpRigidBody*)entity;

		// Try to locate it in the output
		if ( rigidBodiesOut.indexOf(rigidBody) >= 0 )
		{
			continue;
		}

		// Must add
		rigidBodiesOut.pushBack(rigidBody);
	}
}

//
//	Collects all rigid bodies from the world. Assumes the world is locked

static void HK_CALL collectRigidBodies(hkpWorld* physicsWorld, hkArray<hkpRigidBody*>& rigidBodiesOut)
{
	// Get bodies in the active islands
	const hkArray<hkpSimulationIsland*>& activeIslands = physicsWorld->getActiveSimulationIslands();
	for (int i = activeIslands.getSize() - 1; i >= 0; i--)
	{
		const hkpSimulationIsland* island = activeIslands[i];
		collectRigidBodies(island, rigidBodiesOut);
	}

	// Get bodies in the inactive islands
	const hkArray<hkpSimulationIsland*>& inactiveIslands = physicsWorld->getInactiveSimulationIslands();
	for (int i = inactiveIslands.getSize() - 1; i >= 0; i--)
	{
		const hkpSimulationIsland* island = inactiveIslands[i];
		collectRigidBodies(island, rigidBodiesOut);
	}

	// Get fixed bodies
	collectRigidBodies(physicsWorld->getFixedIsland(), rigidBodiesOut);
}

//
//	Collects all constraints from the world. Assumes the world is locked

static void HK_CALL collectConstraints(hkArray<hkpRigidBody*>& rigidBodies, hkArray<hkpConstraintInstance*>& constraintsOut)
{
	for (int bi = rigidBodies.getSize() - 1; bi >= 0; bi--)
	{
		// Get body
		hkpRigidBody* rigidBody = rigidBodies[bi];

		// Get its constraints
		const int numConstraints = rigidBody->getNumConstraints();
		for (int ci = 0; ci < numConstraints; ci++)
		{
			hkpConstraintInstance* constraint = const_cast<hkpConstraintInstance*>(rigidBody->getConstraint(ci));
			const hkpConstraintData* data = constraint->getData();

			switch ( data->getType() )
			{
			case hkpConstraintData::CONSTRAINT_TYPE_CONTACT:
				{
					// Ignore contacts
				}
				break;

			default:
				{
					// See if the constraint has already been added
					int jointIdx = constraintsOut.indexOf(constraint);
					if ( jointIdx < 0 )
					{
						// Must add a new one!
						constraintsOut.pushBack(constraint);
					}
				}
				break;
			}
		}
	}
}

//
//	Returns the center of mass and inverse inertia in local space

static void HK_CALL getRigidBodyInfo(hkpRigidBody* body, hkVector4& invInertia, hkVector4& vCom)
{
	if ( body )
	{
		hkMatrix3 temp;
		body->getInertiaInvLocal(temp);
		hkMatrix3Util::_getDiagonal(temp, invInertia);
		invInertia.setComponent<3>(body->getRigidMotion()->getMassInv());
		vCom = body->getCenterOfMassLocal();
	}
	else
	{
		invInertia.setZero();
		vCom.setZero();
	}
}

//
//	Computes a pair of scaling factors for the current inverse inertia tensors of the constrained rigid bodies, that will stabilize the
//	ball and socket part of the given constraint. The amount of stabilization is controlled by stabilizationAmount, ranging from 0 (no
//	stabilization) to 1 (full stabilization).

void HK_CALL hkpConstraintStabilizationUtil::computeBallSocketInertiaStabilizationFactors(	const class hkpConstraintInstance* constraint, const hkSimdReal& stabilizationAmount,
																							hkSimdReal& inertiaScaleOutA, hkSimdReal& inertiaScaleOutB)
{
	// Constants
	const hkSimdReal eps = hkSimdReal::getConstant<HK_QUADREAL_EPS>();
	const hkSimdReal one = hkSimdReal::getConstant<HK_QUADREAL_1>();

	// Get constraint data
	const hkpConstraintData* cData = constraint->getData();

	// Get constrained body masses and centers of mass
	hkVector4 invInertiaA, invInertiaB;
	hkVector4 vComA, vComB;
	getRigidBodyInfo(constraint->getRigidBodyA(), invInertiaA, vComA);
	getRigidBodyInfo(constraint->getRigidBodyB(), invInertiaB, vComB);

	// Compute the dominant inverse inertia
	const hkSimdReal maxInvInertiaA = invInertiaA.horizontalMax<3>();
	const hkSimdReal maxInvInertiaB = invInertiaB.horizontalMax<3>();

	// Get constraint pivots
	const hkVector4& vPivotA = hkpConstraintDataUtils::getPivotA(cData);
	const hkVector4& vPivotB = hkpConstraintDataUtils::getPivotB(cData);

	// Compute arms
	hkVector4 vArmA;	vArmA.setSub(vPivotA, vComA);
	hkVector4 vArmB;	vArmB.setSub(vPivotB, vComB);

	// Compute arm lengths
	const hkSimdReal armLenA = vArmA.length<3>();
	const hkSimdReal armLenB = vArmB.length<3>();

	// Compute inertia scaling factors
	hkSimdReal fA = maxInvInertiaA * armLenA * stabilizationAmount;
	hkSimdReal fB = maxInvInertiaB * armLenB * stabilizationAmount;

	const hkVector4Comparison cmpA = fA.greater(eps);
	const hkVector4Comparison cmpB = fB.greater(eps);
	fA.setDiv(invInertiaA.getComponent<3>(), fA);
	fB.setDiv(invInertiaB.getComponent<3>(), fB);

	fA.setSelect(cmpA, fA, one);
	fB.setSelect(cmpB, fB, one);

	// Choose minimum, and sub-unitary factor. If > 1 we are not modifying the problem.
	inertiaScaleOutA.setMin(fA, one);
	inertiaScaleOutB.setMin(fB, one);
}

//
//	Constraint atom iterator (typed)

#define SKIP_ATOM_TYPE(TYPE, CLASS_NAME)\
			case hkpConstraintAtom::TYPE:\
{\
	CLASS_NAME* atom = static_cast<CLASS_NAME*>(currentAtom);\
	currentAtom = atom->next();\
	break;\
}

//
//	Constraint atom iterator (untyped)

#define SKIP_ATOM(TYPE, CLASS_NAME)\
			case hkpConstraintAtom::TYPE:\
{\
	CLASS_NAME* atom = static_cast<CLASS_NAME*>(currentAtom);\
	currentAtom = atom + 1;\
	break;\
}

//
//	Changes the inertia of the given rigid body so that the constraints attached to it are stable. The amount of stabilization is controlled by
//	stabilizationAmount, ranging from 0 (no stabilization) to 1 (full stabilization). In the case where the rigid body inertia has been stabilized,
//	there is no need for the solver to further stabilize the constraint, so the solverStabilizationAmount can be set to 0. However, you may want to
//	distribute part of the stabilization in the inertia and delegate the rest to the solver step, e.g., set solverStabilizationAmount to 1 and
//	use a stabilizationAmount < 1. You should note that visual artifacts may appear in this latter case, especially when collisions are involved.
//	Returns the number of stabilized bodies, i.e. 1 if the body was stabilized or 0 otherwise.


int HK_CALL hkpConstraintStabilizationUtil::stabilizeRigidBodyInertia(hkpRigidBody* rigidBody, hkArray<hkpConstraintInstance*>& constraints, const hkReal stabilizationAmount, const hkReal solverStabilizationAmount)
{
	// Ignore fixed objects
	if ( !rigidBody || (rigidBody->getMotionType() == hkpMotion::MOTION_FIXED) )
	{
		return 0;	// Nothing to do!
	}

	// Constants
	const hkSimdReal one = hkSimdReal_1;
	const hkSimdReal scaleAmount = hkSimdReal::fromFloat(stabilizationAmount);

	// Initialize scale
	hkSimdReal fScale = one;

	// Get body constraints
	const int numConstraints = constraints.getSize();
	for (int i = 0; i < numConstraints; i++)
	{
		// Get constraint
		const hkpConstraintInstance* cInstance = constraints[i];
		const hkpConstraintData* cData = cInstance->getData();

		// Compute a scaling factor for the inertia
		hkSimdReal fC = one;

		// Get constraint atoms
		hkpConstraintData::ConstraintInfo cInfo;
		cData->getConstraintInfo(cInfo);
		hkpConstraintAtom* currentAtom = cInfo.m_atoms;
		hkpConstraintAtom* atomsEnd = hkAddByteOffset(currentAtom, cInfo.m_sizeOfAllAtoms);
		while ( currentAtom < atomsEnd )
		{
			switch ( currentAtom->getType() )
			{
			case hkpConstraintAtom::TYPE_INVALID:
				{
					currentAtom = reinterpret_cast<hkpConstraintAtom*>( HK_NEXT_MULTIPLE_OF(HK_REAL_ALIGNMENT, hkUlong(currentAtom)) );
				}
				break;

			case hkpConstraintAtom::TYPE_BALL_SOCKET:
				{
					hkpBallSocketConstraintAtom* atom = reinterpret_cast<hkpBallSocketConstraintAtom*>(currentAtom);
					currentAtom = atom->next();

					hkSimdReal fA, fB;
					computeBallSocketInertiaStabilizationFactors(cInstance, scaleAmount, fA, fB);
					fC = (cInstance->getRigidBodyA() == rigidBody) ? fA : fB;

					atom->setInertiaStabilizationFactor(solverStabilizationAmount);
				}
				break;

				SKIP_ATOM_TYPE(TYPE_BRIDGE,							hkpBridgeConstraintAtom);
				SKIP_ATOM_TYPE(TYPE_SET_LOCAL_TRANSFORMS,			hkpSetLocalTransformsConstraintAtom);
				SKIP_ATOM_TYPE(TYPE_SET_LOCAL_TRANSLATIONS,			hkpSetLocalTranslationsConstraintAtom);
				SKIP_ATOM_TYPE(TYPE_SET_LOCAL_ROTATIONS,			hkpSetLocalRotationsConstraintAtom);
				SKIP_ATOM_TYPE(TYPE_STIFF_SPRING,					hkpStiffSpringConstraintAtom);
				SKIP_ATOM_TYPE(TYPE_LIN,							hkpLinConstraintAtom);
				SKIP_ATOM_TYPE(TYPE_LIN_SOFT,						hkpLinSoftConstraintAtom);
				SKIP_ATOM_TYPE(TYPE_LIN_LIMIT,						hkpLinLimitConstraintAtom);
				SKIP_ATOM_TYPE(TYPE_LIN_FRICTION,					hkpLinFrictionConstraintAtom);
				SKIP_ATOM_TYPE(TYPE_LIN_MOTOR,						hkpLinMotorConstraintAtom);
				SKIP_ATOM_TYPE(TYPE_2D_ANG,							hkp2dAngConstraintAtom);
				SKIP_ATOM_TYPE(TYPE_ANG,							hkpAngConstraintAtom);
				SKIP_ATOM_TYPE(TYPE_ANG_LIMIT,						hkpAngLimitConstraintAtom);
				SKIP_ATOM_TYPE(TYPE_TWIST_LIMIT,					hkpTwistLimitConstraintAtom);
				SKIP_ATOM_TYPE(TYPE_CONE_LIMIT,						hkpConeLimitConstraintAtom);
				SKIP_ATOM_TYPE(TYPE_ANG_FRICTION,					hkpAngFrictionConstraintAtom);
				SKIP_ATOM_TYPE(TYPE_ANG_MOTOR,						hkpAngMotorConstraintAtom);
				SKIP_ATOM_TYPE(TYPE_RAGDOLL_MOTOR,					hkpRagdollMotorConstraintAtom);
				SKIP_ATOM_TYPE(TYPE_PULLEY,							hkpPulleyConstraintAtom);
				SKIP_ATOM_TYPE(TYPE_RACK_AND_PINION,				hkpRackAndPinionConstraintAtom);
				SKIP_ATOM_TYPE(TYPE_COG_WHEEL,						hkpCogWheelConstraintAtom);
				SKIP_ATOM_TYPE(TYPE_SETUP_STABILIZATION,			hkpSetupStabilizationAtom);
				SKIP_ATOM_TYPE( TYPE_3D_ANG,						hkp3dAngConstraintAtom);
				SKIP_ATOM_TYPE(TYPE_OVERWRITE_PIVOT,				hkpOverwritePivotConstraintAtom);
				SKIP_ATOM_TYPE(TYPE_CONTACT,						hkpSimpleContactConstraintAtom);
				SKIP_ATOM(TYPE_MODIFIER_SOFT_CONTACT,				hkpSoftContactModifierConstraintAtom);
				SKIP_ATOM(TYPE_MODIFIER_MASS_CHANGER,				hkpMassChangerModifierConstraintAtom);
				SKIP_ATOM(TYPE_MODIFIER_VISCOUS_SURFACE,			hkpViscousSurfaceModifierConstraintAtom);
				SKIP_ATOM(TYPE_MODIFIER_MOVING_SURFACE,				hkpMovingSurfaceModifierConstraintAtom);
				SKIP_ATOM(TYPE_MODIFIER_IGNORE_CONSTRAINT,			hkpIgnoreModifierConstraintAtom);
				SKIP_ATOM(TYPE_MODIFIER_CENTER_OF_MASS_CHANGER,		hkpCenterOfMassChangerModifierConstraintAtom);

			default:
				{
					currentAtom = atomsEnd;	// Unknown atom, stop!
				}
				break;
			}
		}

		// Pick the minimum factor
		fScale.setMin(fScale, fC);
	}

	// Scale the inverse inertia
	if ( fScale < hkSimdReal::fromFloat(1.0f - HK_REAL_EPSILON) )
	{
		// Get inverse inertia in local space
		hkMatrix3 invI;
		rigidBody->getInertiaInvLocal(invI);

		// Get inverse inertias
		hkVector4 dd; hkMatrix3Util::_getDiagonal(invI, dd);

		// Compute maximum scaled inverse inertia
		hkSimdReal dMax = dd.horizontalMax<3>();

		// Check whether our max inverse inertia is non-zero
		const hkSimdReal eps = hkSimdReal_Eps;
		const hkVector4Comparison cmp = dMax.greater(eps);
		dMax.setSelect(cmp, dMax * fScale, eps);

		// Clamp inertias to the maximum value
		hkVector4 dMaxV; dMaxV.setAll(dMax);
		dd.setMin(dd, dMaxV);

		hkMatrix3Util::_setDiagonal(dd, invI);

		// Set it back
		rigidBody->setInertiaInvLocal(invI);

		// Body was stabilized
		return 1;
	}

	// Body was not stabilized (Was not necessary)
	return 0;
}

//
//	Changes the inertia of the given rigid body so that the constraints attached to it are stable. The amount of stabilization is controlled by
//	stabilizationAmount, ranging from 0 (no stabilization) to 1 (full stabilization). In the case where the rigid body inertia has been stabilized,
//	there is no need for the solver to further stabilize the constraint, so the solverStabilizationAmount can be set to 0. However, you may want to
//	distribute part of the stabilization in the inertia and delegate the rest to the solver step, e.g., set solverStabilizationAmount to 1 and
//	use a stabilizationAmount < 1. You should note that visual artifacts may appear in this latter case, especially when collisions are involved.


int HK_CALL hkpConstraintStabilizationUtil::stabilizeRigidBodyInertia(hkpRigidBody* rigidBody, const hkReal stabilizationAmount, const hkReal solverStabilizationAmount)
{
	if ( !rigidBody )
	{
		return 0;	// No body provided, nothing to do!
	}

	// Gather rigid body constraints in an array
	hkArray<hkpConstraintInstance*> constraints;
	const int numConstraints = rigidBody->getNumConstraints();
	constraints.setSize(numConstraints);
	for (int ci = 0; ci < numConstraints; ci++)
	{
		hkpConstraintInstance* cInstance = const_cast<hkpConstraintInstance*>(rigidBody->getConstraint(ci));
		constraints[ci] = cInstance;
	}

	// Stabilize inertia for all constraints
	return stabilizeRigidBodyInertia(rigidBody, constraints, stabilizationAmount, solverStabilizationAmount);
}

//
//	Calls stabilizeRigidBodyInertia for all bodies in the given physics system.

int HK_CALL hkpConstraintStabilizationUtil::stabilizePhysicsSystemInertias(hkpPhysicsSystem* physicsSystem, const hkReal stabilizationAmount, const hkReal solverStabilizationAmount)
{
	if ( !physicsSystem )
	{
		return 0;	// No system provided, nothing to do!
	}

	const hkArray<hkpRigidBody*>& bodies = physicsSystem->getRigidBodies();
	const int numBodies = bodies.getSize();

	int numStabilized = 0;
	for (int bi = 0; bi < numBodies; bi++)
	{
		hkpRigidBody* body = const_cast<hkpRigidBody*>(bodies[bi]);
		numStabilized += stabilizeRigidBodyInertia(body, stabilizationAmount, solverStabilizationAmount);
	}

	// Return the number of stabilized bodies
	return numStabilized;
}

//
//	Calls stabilizeRigidBodyInertia for all bodies in the given physics world. Returns the number of stabilized bodies.

int HK_CALL hkpConstraintStabilizationUtil::stabilizePhysicsWorldInertias(hkpWorld* physicsWorld, const hkReal stabilizationAmount, const hkReal solverStabilizationAmount)
{
	if ( !physicsWorld )
	{
		return 0;	// No world provided, nothing to do!
	}

	int numStabilized = 0;

	physicsWorld->lock();

	// Collect bodies
	hkArray<hkpRigidBody*> rigidBodies;
	collectRigidBodies(physicsWorld, rigidBodies);

	// Stabilize each body
	const int numBodies = rigidBodies.getSize();
	for (int bi = 0; bi < numBodies; bi++)
	{
		hkpRigidBody* body = rigidBodies[bi];
		numStabilized += stabilizeRigidBodyInertia(body, stabilizationAmount, solverStabilizationAmount);
	}

	physicsWorld->unlock();

	return numStabilized;
}

//
//	Sets the solving method for all given constraints.

void HK_CALL hkpConstraintStabilizationUtil::setConstraintsSolvingMethod(const hkArray<hkpConstraintInstance*>& constraints, hkpConstraintAtom::SolvingMethod method)
{
	const int numConstraints = constraints.getSize();
	for (int ci = 0; ci < numConstraints; ci++)
	{
		hkpConstraintInstance* cInstance = constraints[ci];
		hkpConstraintData* cData = const_cast<hkpConstraintData*>(cInstance->getData());
		cData->setSolvingMethod(method);
	}
}

//
//	Sets the solving method for all constraints attached to the given rigid body.

void HK_CALL hkpConstraintStabilizationUtil::setConstraintsSolvingMethod(hkpRigidBody* rigidBody, hkpConstraintAtom::SolvingMethod method)
{
	if ( !rigidBody )
	{
		return;	// No body provided, nothing to do!
	}

	// Gather constraints
	hkArray<hkpConstraintInstance*> constraints;
	const int numConstraints = rigidBody->getNumConstraints();
	constraints.setSize(numConstraints);
	for (int ci = 0; ci < numConstraints; ci++)
	{
		constraints[ci] = const_cast<hkpConstraintInstance*>(rigidBody->getConstraint(ci));
	}

	// Set solving mode on gathered constraints
	setConstraintsSolvingMethod(constraints, method);
}

//
//	Sets the solving method for all constraints in the given physics system.

void HK_CALL hkpConstraintStabilizationUtil::setConstraintsSolvingMethod(hkpPhysicsSystem* physicsSystem, hkpConstraintAtom::SolvingMethod method)
{
	if ( !physicsSystem )
	{
		return;	// No system provided, nothing to do!
	}

	const hkArray<hkpConstraintInstance*>& systemConstraints = physicsSystem->getConstraints();
	const hkArray<hkpRigidBody*>& systemBodies = physicsSystem->getRigidBodies();

	// Set method on physics system constraints
	setConstraintsSolvingMethod(systemConstraints, method);

	// Set method for constraints attached on the bodies, to make sure we haven't missed a constraint
	const int numBodies = systemBodies.getSize();
	for (int bi = 0; bi < numBodies; bi++)
	{
		hkpRigidBody* rigidBody = systemBodies[bi];
		setConstraintsSolvingMethod(rigidBody, method);
	}
}

//
//	Sets the solving method for all constraints in the given physics world.

void HK_CALL hkpConstraintStabilizationUtil::setConstraintsSolvingMethod(hkpWorld* physicsWorld, hkpConstraintAtom::SolvingMethod method)
{
	if ( !physicsWorld )
	{
		return;	// No world provided, nothing to do!
	}

	physicsWorld->lock();

	// Gather all bodies in the world
	hkArray<hkpRigidBody*> rigidBodies;
	collectRigidBodies(physicsWorld, rigidBodies);

	// Gather all constraints
	hkArray<hkpConstraintInstance*> constraints;
	collectConstraints(rigidBodies, constraints);

	// Set solving method
	setConstraintsSolvingMethod(constraints, method);

	physicsWorld->unlock();
}

//
//	END!
//

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
