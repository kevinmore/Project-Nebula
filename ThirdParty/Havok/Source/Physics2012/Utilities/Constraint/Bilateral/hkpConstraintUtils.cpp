/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Common/Base/hkBase.h>
#include <Physics2012/Utilities/Constraint/Bilateral/hkpConstraintUtils.h>

#include <Physics/Constraint/Data/BallAndSocket/hkpBallAndSocketConstraintData.h>
#include <Physics/Constraint/Data/Fixed/hkpFixedConstraintData.h>
#include <Physics/Constraint/Data/DeformableFixed/hkpDeformableFixedConstraintData.h>
#include <Physics/Constraint/Data/Hinge/hkpHingeConstraintData.h>
#include <Physics/Constraint/Data/LimitedHinge/hkpLimitedHingeConstraintData.h>
#include <Physics/Constraint/Data/Ragdoll/hkpRagdollConstraintData.h>
#include <Physics/Constraint/Data/CogWheel/hkpCogWheelConstraintData.h>
#include <Physics/Constraint/Data/RackAndPinion/hkpRackAndPinionConstraintData.h>
#include <Physics/Constraint/Data/Prismatic/hkpPrismaticConstraintData.h>
#include <Physics/Constraint/Data/RagdollLimits/hkpRagdollLimitsData.h>
#include <Physics/Constraint/Data/HingeLimits/hkpHingeLimitsData.h>
#include <Physics/Constraint/Data/hkpConstraintDataUtils.h>
#include <Physics/Constraint/Motor/hkpConstraintMotor.h>

#include <Physics/ConstraintSolver/Constraint/Bilateral/hkpInternalConstraintUtils.h>
#include <Physics2012/Dynamics/Constraint/Atom/hkpSimpleContactConstraintAtom.h>

#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Dynamics/Constraint/hkpConstraintInstance.h>
#include <Physics2012/Dynamics/Constraint/Chain/BallSocket/hkpBallSocketChainData.h>
#include <Physics2012/Dynamics/Constraint/Chain/Powered/hkpPoweredChainData.h>
#include <Physics2012/Dynamics/Constraint/Chain/BallSocket/hkpBallSocketChainData.h>
#include <Physics2012/Dynamics/Constraint/Chain/hkpConstraintChainInstance.h>

#include <Physics2012/Dynamics/World/hkpSimulationIsland.h>

// EXP-1839 : Straight atom copy doesn't respect reference counting
static void _copyAtoms (const hkpRagdollConstraintData::Atoms& atomsIn, hkpRagdollConstraintData::Atoms& atomsOut)
{
	atomsOut = atomsIn;
	for (int i=0; i<3; i++)
	{
		if (atomsOut.m_ragdollMotors.m_motors[i])
		{
			atomsOut.m_ragdollMotors.m_motors[i]->addReference();
		}
	}
}

// EXP-1839 : Straight atom copy doesn't respect reference counting
static void _copyAtoms (const hkpLimitedHingeConstraintData::Atoms& atomsIn, hkpLimitedHingeConstraintData::Atoms& atomsOut)
{
	atomsOut = atomsIn;
	if(atomsOut.m_angMotor.m_motor)
	{
		atomsOut.m_angMotor.m_motor->addReference();
	}
}
hkpConstraintInstance* hkpConstraintUtils::convertToPowered (const hkpConstraintInstance* originalConstraint, hkpConstraintMotor* constraintMotor, hkBool enableMotors)
{
	hkpConstraintInstance* newConstraint = HK_NULL;
	const hkpConstraintData* constraintData = originalConstraint->getData();
	hkpConstraintData* newConstraintData = HK_NULL;

	hkpEntity* entityA = originalConstraint->getEntityA();
	hkpEntity* entityB = originalConstraint->getEntityB();

	switch (constraintData->getType())
	{
		case hkpConstraintData::CONSTRAINT_TYPE_LIMITEDHINGE:
		{
			newConstraintData = new hkpLimitedHingeConstraintData();
			_copyAtoms(static_cast<const hkpLimitedHingeConstraintData*>(constraintData)->m_atoms, static_cast<hkpLimitedHingeConstraintData*>(newConstraintData)->m_atoms);
			break;
		}
		case hkpConstraintData::CONSTRAINT_TYPE_RAGDOLL:
		{
			newConstraintData = new hkpRagdollConstraintData();	
			_copyAtoms(static_cast<const hkpRagdollConstraintData*>(constraintData)->m_atoms, static_cast<hkpRagdollConstraintData*>(newConstraintData)->m_atoms);
			break;
		}

		default:
		{
			HK_WARN_ALWAYS (0xabba1b34, "Cannot convert constraint \""<<originalConstraint->getName()<<"\" to a powered constraint.");
			HK_WARN_ALWAYS (0xabba1b34, "Only limited hinges and ragdoll constraints can be powered.");
			return HK_NULL;
		}
	}

	hkpConstraintDataUtils::convertToPowered(newConstraintData, constraintMotor, enableMotors);
	newConstraint = new hkpConstraintInstance (entityA, entityB, newConstraintData, originalConstraint->getPriority());
	newConstraintData->removeReference();
	newConstraint->setName( originalConstraint->getName() );

	return newConstraint;
}

hkpConstraintInstance* HK_CALL hkpConstraintUtils::convertToLimits (hkpConstraintInstance* originalConstraint)
{
	const hkpConstraintData* originalData = originalConstraint->getData();
	hkpConstraintData* limitData = hkpConstraintDataUtils::createLimited(originalData);

	if (limitData)
	{
		hkpConstraintInstance* limitInstance = new hkpConstraintInstance( originalConstraint->getEntityA(), originalConstraint->getEntityB(), limitData, originalConstraint->getPriority() );
		limitData->removeReference();

		return limitInstance;
	}
	return HK_NULL;
}


hkBool hkpConstraintUtils::checkAndFixConstraint (const hkpConstraintInstance* constraint, hkReal maxAllowedError, hkReal relativeFixupOnError)
{
	hkVector4 childPivotInChild;
	hkVector4 parentPivotInParent;

	hkResult res = 	hkpConstraintDataUtils::getConstraintPivots(constraint->getData(), childPivotInChild, parentPivotInParent);

	// Unsupported constraint type? return false
	if (res!=HK_SUCCESS)
	{
		return false;
	}

	// The pivotInA should be 0,0,0 if the pivots are properly aligned, warn otherwise
	if (childPivotInChild.lengthSquared<3>().isGreater(hkSimdReal::fromFloat(1e-6f)))
	{
		HK_WARN_ALWAYS (0xabba5dff, "Pivot of child rigid body (A) is expected to be aligned with the constraint at setup time.");
	}

	hkpRigidBody* parentRigidBody = constraint->getRigidBodyB();
	hkpRigidBody* childRigidBody = constraint->getRigidBodyA();

	const hkTransform& worldFromParent = parentRigidBody->getTransform();
	const hkTransform& worldFromChild = childRigidBody->getTransform();

	hkVector4 parentPivotInWorld; parentPivotInWorld._setTransformedPos(worldFromParent, parentPivotInParent);
	hkVector4 childPivotInWorld; childPivotInWorld._setTransformedPos(worldFromChild, childPivotInChild);

	hkVector4 error; error.setSub(parentPivotInWorld, childPivotInWorld);

	// Are they aligned ?
	hkSimdReal maxError; maxError.setFromFloat(maxAllowedError);
	if (error.lengthSquared<3>() > maxError*maxError)
	{
		// NO

		// Interpolate the new position between the desired position and the position of the parent
		const hkVector4& parentPositionInWorld = parentRigidBody->getPosition();
		hkVector4 newChildPositionInWorld; newChildPositionInWorld.setInterpolate(parentPivotInWorld, parentPositionInWorld, hkSimdReal::fromFloat(relativeFixupOnError));

		childRigidBody->setPosition(newChildPositionInWorld);

		// Set the velocity to match the parent
		childRigidBody->setLinearVelocity(parentRigidBody->getLinearVelocity());
		childRigidBody->setAngularVelocity(parentRigidBody->getAngularVelocity());

		return true; // fix up done
	}
	else
	{
		// YES

		return false; // no fix up done
	}

}

void hkpConstraintUtils::setHingePivotToOptimalPosition(hkpConstraintInstance* constraint)
{
	// Only support hkpHingeConstraintData and hkpLimitedHingeConstraintData.
	//
	hkpSetLocalTransformsConstraintAtom* transformsAtom = HK_NULL;
	{
		hkpConstraintData* data = constraint->getDataRw();
		switch(data->getType())
		{
		case hkpConstraintData::CONSTRAINT_TYPE_HINGE: 
			transformsAtom = &static_cast<hkpHingeConstraintData*>(data)->m_atoms.m_transforms;
			break;
		case hkpConstraintData::CONSTRAINT_TYPE_LIMITEDHINGE: 
			transformsAtom = &static_cast<hkpLimitedHingeConstraintData*>(data)->m_atoms.m_transforms;
			break;
		default:
			HK_ASSERT2(0xad909031, false, "hkpConstraintUtils::setHingePivotToOptimalPosition() only supports hinge and limited-hinge constraints.");
			return;
		}
	}

	// Calculate projection of the centers of mass of the connected bodies onto the constraint axis.
	// Calculations need to be performed in local space.
	//
	hkpRigidBody* bodyA = static_cast<hkpRigidBody*>(constraint->getEntityA());
	hkpRigidBody* bodyB = static_cast<hkpRigidBody*>(constraint->getEntityB());

	
	const bool useRelativeInertia = true;
	const hkSimdReal eps = hkSimdReal::getConstant<HK_QUADREAL_EPS>();

	// calculate inertia to mass ratios
	hkSimdReal invInertiaA; invInertiaA.setZero();
	{
		hkMatrix3 inertiaLocA; bodyA->getInertiaInvLocal(inertiaLocA);
		hkRotation invRot = transformsAtom->m_transformA.getRotation();
		invRot.transpose();
		inertiaLocA.changeBasis(invRot);
		hkVector4 colSum; 
		colSum.setAdd(inertiaLocA.getColumn<1>(), inertiaLocA.getColumn<2>());
		colSum.zeroComponent<0>();
		invInertiaA = colSum.horizontalAdd<3>();
		

		const hkSimdReal bodyMassInv = bodyA->getRigidMotion()->getMassInv();
		if (useRelativeInertia && bodyMassInv > eps)
		{
			invInertiaA.div(bodyMassInv);
		}
	}

	hkSimdReal invInertiaB; invInertiaB.setZero();
	{
		hkMatrix3 inertiaLocB; bodyB->getInertiaInvLocal(inertiaLocB);
		hkRotation invRot = transformsAtom->m_transformB.getRotation();
		invRot.transpose();
		inertiaLocB.changeBasis(invRot);
		hkVector4 colSum; 
		colSum.setAdd(inertiaLocB.getColumn<1>(), inertiaLocB.getColumn<2>());
		colSum.zeroComponent<0>();
		invInertiaB = colSum.horizontalAdd<3>();

		const hkSimdReal bodyMassInv = bodyB->getRigidMotion()->getMassInv();
		if (useRelativeInertia && bodyMassInv > eps)
		{
			invInertiaB.div(bodyMassInv);
		}
	}

	// calculate pivots and their components along axis
	const hkVector4& axisA = transformsAtom->m_transformA.getRotation().getColumn<0>();
	const hkVector4& bodyAlocalCOM = bodyA->getCenterOfMassLocal();
	hkVector4 pivotA; pivotA.setSub(transformsAtom->m_transformA.getTranslation(), bodyAlocalCOM);
	const hkSimdReal pivotLengthAlongAxisA = pivotA.dot<3>(axisA);

	const hkVector4& axisB = transformsAtom->m_transformB.getRotation().getColumn<0>();
	const hkVector4& bodyBlocalCOM = bodyB->getCenterOfMassLocal();
	hkVector4 pivotB; pivotB.setSub(transformsAtom->m_transformB.getTranslation(), bodyBlocalCOM);
	const hkSimdReal pivotLengthAlongAxisB = pivotB.dot<3>(axisB);

	// interpolate the projection points using the bodies' inertias.
	hkSimdReal invSumInertias; invSumInertias.setReciprocal(invInertiaA + invInertiaB);
	const hkSimdReal totalPivotLengthAlongAxis = pivotLengthAlongAxisA - pivotLengthAlongAxisB;
	const hkSimdReal newPivotLenghtAlongAxisA =   totalPivotLengthAlongAxis * invInertiaB * invSumInertias;
	const hkSimdReal newPivotLenghtAlongAxisB = - totalPivotLengthAlongAxis * invInertiaA * invSumInertias;

	pivotA.addMul(newPivotLenghtAlongAxisA - pivotLengthAlongAxisA, axisA);
	pivotB.addMul(newPivotLenghtAlongAxisB - pivotLengthAlongAxisB, axisB);

	// Adjust transforms stored in the atoms.
	transformsAtom->m_transformA.getTranslation().setAdd(pivotA, bodyAlocalCOM);
	transformsAtom->m_transformB.getTranslation().setAdd(pivotB, bodyBlocalCOM);
}

void hkpConstraintUtils::collectConstraints(const hkArray<hkpEntity*>& entities, hkArray<hkpConstraintInstance*>& constraintsOut, hkpConstraintUtils::CollectConstraintsFilter* collectionFilter)
{
	// use temp pointer map to check if constraint has been added.
	// faster than linear search through array.
	hkPointerMap<hkpConstraintInstance*,bool>::Temp constraintsMap;
	for (int i = 0; i < entities.getSize(); i++)
	{
		hkpEntity* srcEntity = entities[i];
		int numConstraints = srcEntity->getNumConstraints();
		for (int j = 0; j < numConstraints; j++)
		{
			hkpConstraintInstance* srcConstraint = srcEntity->getConstraint(j);
			if ((!collectionFilter || collectionFilter->collectConstraint(srcConstraint)) && !constraintsMap.hasKey(srcConstraint))
			{
				constraintsMap.insert(srcConstraint, true);
				constraintsOut.pushBack(srcConstraint);
			}
		}
	}
}


hkpConstraintInstance* hkpConstraintUtils::createMatchingHingeConstraintFromCogWheelConstraint(
	const hkpConstraintInstance* constraint, int bodyIndex, class hkpRigidBody* anotherBody, bool createLimitedHinge /*= false*/ )
{
	HK_ASSERT2(0x40d1e0fd, constraint->getData()->getType() == hkpConstraintData::CONSTRAINT_TYPE_COG_WHEEL,
		"Input constraint must be a cog wheel constraint" );
	const hkpCogWheelConstraintData* cwcd = static_cast<const hkpCogWheelConstraintData*>( constraint->getData() );

	hkpRigidBody* body = static_cast<hkpRigidBody*>(bodyIndex ? constraint->getEntityA() : constraint->getEntityB());
	const hkTransform& localTransform = bodyIndex ? cwcd->m_atoms.m_transforms.m_transformB : cwcd->m_atoms.m_transforms.m_transformA;

	hkpConstraintData* data;

	if (!createLimitedHinge)
	{
		hkpHingeConstraintData* hingeData = new hkpHingeConstraintData();
		hkVector4 rotationPivot; rotationPivot._setTransformedPos(body->getTransform(), localTransform.getTranslation());
		hkVector4 rotationAxis;  rotationAxis._setRotatedDir(body->getTransform().getRotation(), localTransform.getRotation().getColumn<0>());
		hingeData->setInWorldSpace(body->getTransform(), anotherBody->getTransform(), rotationPivot, rotationAxis);
		data = hingeData;
	}
	else
	{
		hkpLimitedHingeConstraintData* hingeData = new hkpLimitedHingeConstraintData();
		hkVector4 rotationPivot; rotationPivot._setTransformedPos(body->getTransform(), localTransform.getTranslation());
		hkVector4 rotationAxis;  rotationAxis._setRotatedDir(body->getTransform().getRotation(), localTransform.getRotation().getColumn<0>());
		hingeData->setInWorldSpace(body->getTransform(), anotherBody->getTransform(), rotationPivot, rotationAxis);
		data = hingeData;
	}

	hkpConstraintInstance* instance = new hkpConstraintInstance(body, anotherBody, data);
	data->removeReference();

	return instance;
}


hkpConstraintInstance* hkpConstraintUtils::createMatchingPrismaticConstraintFromRackAndPinionConstraint(
	hkpConstraintInstance* constraint, hkpRigidBody* anotherBody)
{
	HK_ASSERT2(0x762a359b, constraint->getData()->getType() == hkpConstraintData::CONSTRAINT_TYPE_RACK_AND_PINION,
		"Input constraint must be a rack and pinion constraint" );
	const hkpRackAndPinionConstraintData* pdcd = static_cast<const hkpRackAndPinionConstraintData*>( constraint->getData() );

	hkpRigidBody* bodyB = static_cast<hkpRigidBody*>(constraint->getEntityB());
	hkVector4 pivot;     pivot._setTransformedPos(bodyB->getTransform(), 
		pdcd->m_atoms.m_transforms.m_transformB.getTranslation());
	hkVector4 shiftAxis; shiftAxis._setRotatedDir(bodyB->getTransform().getRotation(), 
		pdcd->m_atoms.m_transforms.m_transformB.getRotation().getColumn<0>());

	if (!pdcd->m_atoms.m_rackAndPinion.m_isScrew)
	{
		// For the rack-and-pinion constraint we shift the pivot point
		hkpRigidBody* bodyA = static_cast<hkpRigidBody*>(constraint->getEntityA());
		hkVector4 pinionAxis; pinionAxis._setRotatedDir(bodyA->getTransform().getRotation(), 
			pdcd->m_atoms.m_transforms.m_transformA.getRotation().getColumn<0>());
		hkVector4 cross; cross.setCross(shiftAxis, pinionAxis);
		if (cross.lengthSquared<3>() < hkSimdReal::getConstant<HK_QUADREAL_EPS_SQRD>() )
		{
			HK_WARN(0xad332882, "Constraint's shift & pinion axes are parallel. Cannot offset the pivot point properly "
				"when generating matching prismatic constraint. The generated constraint will be less "
				"stable");
		}
		else
		{
			cross.normalize<3>();
			cross.mul(hkSimdReal::fromFloat(pdcd->m_atoms.m_rackAndPinion.m_pinionRadiusOrScrewPitch));
			pivot.add(cross);
		}
	}

	hkpPrismaticConstraintData* data = new hkpPrismaticConstraintData();
	data->setInWorldSpace(bodyB->getTransform(), anotherBody->getTransform(), pivot, shiftAxis);

	hkpConstraintInstance* instance = new hkpConstraintInstance(bodyB, anotherBody, data);
	data->removeReference();

	return instance;
}

hkpConstraintInstance* hkpConstraintUtils::createMatchingHingeConstraintFromRackAndPinionConstraint(
	hkpConstraintInstance* constraint, hkpRigidBody* anotherBody, bool createLimitedHinge )
{
	HK_ASSERT2(0x28620018, constraint->getData()->getType() == hkpConstraintData::CONSTRAINT_TYPE_RACK_AND_PINION,
		"Input constraint must be a rack and pinion constraint" );
	const hkpRackAndPinionConstraintData* pdcd = static_cast<const hkpRackAndPinionConstraintData*>( constraint->getData() );

	hkpRigidBody* bodyA = static_cast<hkpRigidBody*>(constraint->getEntityA());

	hkpConstraintData* data;

	if (!createLimitedHinge)
	{
		hkpHingeConstraintData* hingeData = new hkpHingeConstraintData();
		hkVector4 rotationPivot; rotationPivot._setTransformedPos(bodyA->getTransform(), 
			pdcd->m_atoms.m_transforms.m_transformA.getTranslation());
		hkVector4 rotationAxis;  rotationAxis._setRotatedDir(bodyA->getTransform().getRotation(), 
			pdcd->m_atoms.m_transforms.m_transformA.getRotation().getColumn<0>());
		hingeData->setInWorldSpace(bodyA->getTransform(), anotherBody->getTransform(), rotationPivot, rotationAxis);
		data = hingeData;
	}
	else
	{
		hkpLimitedHingeConstraintData* hingeData = new hkpLimitedHingeConstraintData();
		hkVector4 rotationPivot; rotationPivot._setTransformedPos(bodyA->getTransform(), 
			pdcd->m_atoms.m_transforms.m_transformA.getTranslation());
		hkVector4 rotationAxis;  rotationAxis._setRotatedDir(bodyA->getTransform().getRotation(), 
			pdcd->m_atoms.m_transforms.m_transformA.getRotation().getColumn<0>());
		hingeData->setInWorldSpace(bodyA->getTransform(), anotherBody->getTransform(), rotationPivot, rotationAxis);
		data = hingeData;
	}

	hkpConstraintInstance* instance = new hkpConstraintInstance(bodyA, anotherBody, data);
	data->removeReference();

	return instance;
}

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
