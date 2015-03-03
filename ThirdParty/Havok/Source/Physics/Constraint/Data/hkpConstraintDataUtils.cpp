/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Constraint/hkpConstraint.h>
#include <Common/Base/KeyCode.h>

#include <Physics/Constraint/Data/hkpConstraintDataUtils.h>

#include <Physics/Constraint/Data/hkpConstraintData.h>
#include <Physics/Constraint/Data/LimitedHinge/hkpLimitedHingeConstraintData.h>
#include <Physics/Constraint/Data/Ragdoll/hkpRagdollConstraintData.h>
#include <Physics/Constraint/Data/BallAndSocket/hkpBallAndSocketConstraintData.h>
#include <Physics/Constraint/Data/Fixed/hkpFixedConstraintData.h>
#include <Physics/Constraint/Data/DeformableFixed/hkpDeformableFixedConstraintData.h>
#include <Physics/Constraint/Data/Hinge/hkpHingeConstraintData.h>
#include <Physics/Constraint/Data/RagdollLimits/hkpRagdollLimitsData.h>
#include <Physics/Constraint/Data/HingeLimits/hkpHingeLimitsData.h>
#include <Physics/Constraint/Data/Prismatic/hkpPrismaticConstraintData.h>

// Constraints
#include <Physics/ConstraintSolver/Solve/hkpSolverResults.h>
#include <Physics/Constraint/Data/StiffSpring/hkpStiffSpringConstraintData.h>
#include <Physics/Constraint/Data/BallAndSocket/hkpBallAndSocketConstraintData.h>
#include <Physics/Constraint/Data/Wheel/hkpWheelConstraintData.h>
#include <Physics/Constraint/Data/Hinge/hkpHingeConstraintData.h>
#include <Physics/Constraint/Data/Prismatic/hkpPrismaticConstraintData.h>
#include <Physics/Constraint/Data/Ragdoll/hkpRagdollConstraintData.h>
#include <Physics/Constraint/Data/PointToPlane/hkpPointToPlaneConstraintData.h>
#include <Physics/Constraint/Data/Pulley/hkpPulleyConstraintData.h>
#include <Physics/Constraint/Data/PointToPath/hkpPointToPathConstraintData.h>

#if !defined ( HK_FEATURE_PRODUCT_PHYSICS )
#	include <Physics2012/Dynamics/Constraint/Breakable/hkpBreakableConstraintData.h>
#	include <Physics2012/Dynamics/Constraint/Malleable/hkpMalleableConstraintData.h>
#endif

#include <Physics/Constraint/Motor/hkpConstraintMotor.h>

#include <Physics/ConstraintSolver/Constraint/Bilateral/hkpInternalConstraintUtils.h>
namespace hkpConstraintDataUtils
{
#ifndef HK_PLATFORM_SPU

	static const hkReal smallTolerance = 0.01f;

	// Sets the constraint to a powered version. Only works with hinges and ragdolls (and will warn if it's not the case)
	void HK_CALL convertToPowered( hkpConstraintData* data, hkpConstraintMotor* motor, hkBool enableMotors )
	{
		switch (data->getType())
		{
			case hkpConstraintData::CONSTRAINT_TYPE_LIMITEDHINGE:
			{
				hkpLimitedHingeConstraintData* hingeData = static_cast<hkpLimitedHingeConstraintData*>(data);
				hingeData->setMotor( motor );
				hingeData->setMotorEnabled(HK_NULL, enableMotors);
				break;
			}

			case hkpConstraintData::CONSTRAINT_TYPE_RAGDOLL:
			{
				hkpRagdollConstraintData* ragdollData = static_cast<hkpRagdollConstraintData*>(data);
				ragdollData->setTwistMotor(motor);
				ragdollData->setPlaneMotor(motor);
				ragdollData->setConeMotor(motor);
				ragdollData->setMotorsEnabled(HK_NULL, enableMotors);
				break;
			}

			default:
			{
				HK_WARN_ALWAYS (0xabba1b34, "Cannot convert constraint data to a powered constraint.");
			}
		}
	}


	hkpConstraintData* HK_CALL createLimited( const hkpConstraintData* data )
	{
		// We have to create a new data.
		hkpConstraintData * result = HK_NULL;

		switch(data->getType())
		{
			case hkpConstraintData::CONSTRAINT_TYPE_RAGDOLL:
			{
				hkpRagdollLimitsData* n = new hkpRagdollLimitsData();
				const hkpRagdollConstraintData* o = static_cast<const hkpRagdollConstraintData*>(data);
				n->m_atoms.m_rotations.m_rotationA = o->m_atoms.m_transforms.m_transformA.getRotation();
				n->m_atoms.m_rotations.m_rotationB = o->m_atoms.m_transforms.m_transformB.getRotation();
				n->m_atoms.m_twistLimit  = o->m_atoms.m_twistLimit;
				n->m_atoms.m_planesLimit = o->m_atoms.m_planesLimit;
				n->m_atoms.m_coneLimit   = o->m_atoms.m_coneLimit;
				hkBool stabilizationEnabled = o->getConeLimitStabilization();
				n->setConeLimitStabilization(stabilizationEnabled);
				result = n;


				break;
			}

			case hkpConstraintData::CONSTRAINT_TYPE_LIMITEDHINGE:
			{
				hkpHingeLimitsData* n = new hkpHingeLimitsData();
				const hkpLimitedHingeConstraintData* o = static_cast<const hkpLimitedHingeConstraintData*>(data);

				n->m_atoms.m_rotations.m_rotationA = o->m_atoms.m_transforms.m_transformA.getRotation();
				n->m_atoms.m_rotations.m_rotationB = o->m_atoms.m_transforms.m_transformB.getRotation();
				n->m_atoms.m_angLimit = o->m_atoms.m_angLimit;
				n->m_atoms.m_2dAng     = o->m_atoms.m_2dAng;
				result = n;

				break;
			}

			case hkpConstraintData::CONSTRAINT_TYPE_RAGDOLL_LIMITS:
			case hkpConstraintData::CONSTRAINT_TYPE_HINGE_LIMITS:
			{
				// Just return the original constraint.
				// Remember to add a reference.
				HK_WARN(0xad67d7db, "The original constraint already is of the 'limits' type.");
				result->addReference();
				return result;

			}

			default:
			{
				HK_WARN_ALWAYS(0xabbad7da, "Unsupported constraint type. Cannot generate limits constraints.");
				return HK_NULL;
			}
		}
		return result;
	}


	hkResult HK_CALL getConstraintPivots( const hkpConstraintData* data, hkVector4& pivotInAOut, hkVector4& pivotInBOut )
	{
		switch (data->getType())
		{
			case hkpConstraintData::CONSTRAINT_TYPE_BALLANDSOCKET:
			{
				const hkpBallAndSocketConstraintData* bsConstraint = static_cast<const hkpBallAndSocketConstraintData*> (data);
				pivotInAOut = bsConstraint->m_atoms.m_pivots.m_translationA;
				pivotInBOut = bsConstraint->m_atoms.m_pivots.m_translationB;
				break;
			}

				case hkpConstraintData::CONSTRAINT_TYPE_FIXED:
			{
				const hkpFixedConstraintData* fixedConstraint = static_cast<const hkpFixedConstraintData*> (data);
				pivotInAOut = fixedConstraint->m_atoms.m_transforms.m_transformA.getTranslation();
				pivotInBOut = fixedConstraint->m_atoms.m_transforms.m_transformB.getTranslation();
				break;
			}

			case hkpConstraintData::CONSTRAINT_TYPE_DEFORMABLE_FIXED:
			{
				const hkpDeformableFixedConstraintData* dcd = static_cast<const hkpDeformableFixedConstraintData*> (data);
				pivotInAOut = dcd->m_atoms.m_transforms.m_transformA.getTranslation();
				pivotInBOut = dcd->m_atoms.m_transforms.m_transformB.getTranslation();
				break;
			}

			case hkpConstraintData::CONSTRAINT_TYPE_HINGE:
			{
				const hkpHingeConstraintData* hConstraint = static_cast<const hkpHingeConstraintData*> (data);
				pivotInAOut = hConstraint->m_atoms.m_transforms.m_transformA.getTranslation();
				pivotInBOut = hConstraint->m_atoms.m_transforms.m_transformB.getTranslation();
				break;
			}

			case hkpConstraintData::CONSTRAINT_TYPE_LIMITEDHINGE:
			{
				const hkpLimitedHingeConstraintData* hConstraint = static_cast<const hkpLimitedHingeConstraintData*> (data);
				pivotInAOut = hConstraint->m_atoms.m_transforms.m_transformA.getTranslation();
				pivotInBOut = hConstraint->m_atoms.m_transforms.m_transformB.getTranslation();
				break;
			}

			case hkpConstraintData::CONSTRAINT_TYPE_RAGDOLL:
			{
				const hkpRagdollConstraintData* rConstraint = static_cast<const hkpRagdollConstraintData*> (data);
				pivotInAOut = rConstraint->m_atoms.m_transforms.m_transformA.getTranslation();
				pivotInBOut = rConstraint->m_atoms.m_transforms.m_transformB.getTranslation();
				break;
			}

			default:
			{
				HK_WARN_ALWAYS (0xabbabf3b, "Unsupported type of constraint in getConstraintPivots");
				pivotInAOut.setZero();
				pivotInBOut.setZero();
				return HK_FAILURE;
			}
		}

		return HK_SUCCESS;
	}


	hkResult HK_CALL getConstraintMotors( const hkpConstraintData* data, hkpConstraintMotor*& motor0, hkpConstraintMotor*& motor1, hkpConstraintMotor*& motor2 )
	{
		switch (data->getType())
		{
			case hkpConstraintData::CONSTRAINT_TYPE_LIMITEDHINGE:
			{
				const hkpLimitedHingeConstraintData* hConstraint = static_cast<const hkpLimitedHingeConstraintData*> (data);
				motor0 = hConstraint->getMotor();
				motor1 = HK_NULL;
				motor2 = HK_NULL;
				break;
			}

			case hkpConstraintData::CONSTRAINT_TYPE_RAGDOLL:
			{
				const hkpRagdollConstraintData* rConstraint = static_cast<const hkpRagdollConstraintData*> (data);
				// Possibly, those motors should be extracted in a different order.
				motor0 = rConstraint->getTwistMotor();
				motor1 = rConstraint->getConeMotor();
				motor2 = rConstraint->getPlaneMotor();
				break;
			}

			default:
			{
				motor0 = motor1 = motor2 = HK_NULL;
				HK_WARN_ALWAYS (0xabbae233, "This type of constraint does not have motors");
				return HK_FAILURE;
			}
		}

		return HK_SUCCESS;
	}


	hkResult HK_CALL loosenConstraintLimits(hkpConstraintData* data, const hkTransform& bodyATransform, const hkTransform& bodyBTransform)
	{
		switch(data->getType())
		{
			case hkpConstraintData::CONSTRAINT_TYPE_RAGDOLL:
			{
				// NOTE: Body B is the reference object, A is the constrained object

				hkpRagdollConstraintData* ragdollData = static_cast<hkpRagdollConstraintData*>(data);
				hkpRagdollConstraintData::Atoms& atoms = ragdollData->m_atoms;

				// Base transforms
				const hkTransform &pivotTransformA = atoms.m_transforms.m_transformA; // parent
				const hkTransform &pivotTransformB = atoms.m_transforms.m_transformB; // child

				// Get axes information for cone limit
				hkVector4 twistAxisInA_WS;	twistAxisInA_WS._setRotatedDir(bodyATransform.getRotation(), pivotTransformA.getColumn(atoms.m_coneLimit.m_twistAxisInA));
				hkVector4 refAxisInB_WS;	refAxisInB_WS._setRotatedDir(bodyBTransform.getRotation(), pivotTransformB.getColumn(atoms.m_coneLimit.m_refAxisInB));

				// Get axes information for plane limit
				hkVector4 twistAxisInA_WS2;	twistAxisInA_WS2._setRotatedDir(bodyATransform.getRotation(), pivotTransformA.getColumn(atoms.m_planesLimit.m_twistAxisInA));
				hkVector4 refAxisInB_WS2;	refAxisInB_WS2._setRotatedDir(bodyBTransform.getRotation(), pivotTransformB.getColumn(atoms.m_planesLimit.m_refAxisInB));

#if defined(HK_PLATFORM_WIN32) && (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED) && !defined(HK_ARCH_ARM)
				hkVector4 acosDots; acosDots.set(twistAxisInA_WS.dot<3>(refAxisInB_WS), twistAxisInA_WS2.dot<3>(refAxisInB_WS2), hkSimdReal_0, hkSimdReal_0);
#	if defined(HK_REAL_IS_DOUBLE)
				acosDots.m_quad.xy = hkMath::twoAcos(acosDots.m_quad.xy);
#	else
				acosDots.m_quad = hkMath::quadAcos(acosDots.m_quad);
#	endif
				hkReal currentConeAngle = acosDots(0) + hkpConstraintDataUtils::smallTolerance;
				hkReal currentPlaneAngle = acosDots(1);
#else
				hkReal currentConeAngle = hkMath::acos(twistAxisInA_WS.dot<3>(refAxisInB_WS).getReal()) + hkpConstraintDataUtils::smallTolerance;
				hkReal currentPlaneAngle = hkMath::acos(twistAxisInA_WS2.dot<3>(refAxisInB_WS2).getReal());
#endif

				// Loosen cone constraint to accommodate if needed
				{
					hkReal maxConeAngle = atoms.m_coneLimit.m_maxAngle;
					atoms.m_coneLimit.m_maxAngle = hkMath::max2(maxConeAngle, currentConeAngle);
				}

				// Loosen plane constraint to accommodate if needed
				{
					hkReal maxPlaneAngle = atoms.m_planesLimit.m_maxAngle;
					hkReal minPlaneAngle = atoms.m_planesLimit.m_minAngle;
					const hkReal piDivBy2 = (HK_REAL_PI * hkReal(0.5f));
					if (currentPlaneAngle <= piDivBy2)
					{
						atoms.m_planesLimit.m_maxAngle = hkMath::max2(maxPlaneAngle, piDivBy2 - currentPlaneAngle + hkpConstraintDataUtils::smallTolerance);
					}
					else
					{
						atoms.m_planesLimit.m_minAngle = hkMath::min2(minPlaneAngle, -(currentPlaneAngle - piDivBy2) - hkpConstraintDataUtils::smallTolerance);
					}
				}

				// Get axes information for cone limit
				twistAxisInA_WS._setRotatedDir(bodyATransform.getRotation(), pivotTransformA.getColumn(atoms.m_twistLimit.m_twistAxis));
				refAxisInB_WS._setRotatedDir(bodyBTransform.getRotation(), pivotTransformB.getColumn(atoms.m_twistLimit.m_twistAxis)); // reused for twist axis
				hkVector4 planeAxisA_WS;	planeAxisA_WS._setRotatedDir(bodyATransform.getRotation(), pivotTransformA.getColumn(atoms.m_twistLimit.m_refAxis));
				hkVector4 planeAxisB_WS;	planeAxisB_WS._setRotatedDir(bodyBTransform.getRotation(), pivotTransformB.getColumn(atoms.m_twistLimit.m_refAxis));
				hkVector4 currentTwistAxis_WS;
				hkPadSpu<hkReal> currentTwistAngle;
				hkInternalConstraintUtils_calcRelativeAngle( twistAxisInA_WS, refAxisInB_WS, planeAxisA_WS, planeAxisB_WS, currentTwistAxis_WS, currentTwistAngle );

				// Loosen twist constraint to accommodate if needed
				{
					hkReal minTwistAngle = atoms.m_twistLimit.m_minAngle;
					hkReal maxTwistAngle = atoms.m_twistLimit.m_maxAngle;
					if (currentTwistAngle >= hkReal(0))
					{
						atoms.m_twistLimit.m_maxAngle = hkMath::max2(maxTwistAngle, currentTwistAngle + hkpConstraintDataUtils::smallTolerance);
					}
					else
					{
						atoms.m_twistLimit.m_minAngle = hkMath::min2(minTwistAngle, currentTwistAngle - hkpConstraintDataUtils::smallTolerance);
					}
				}
				break;
			}
			case hkpConstraintData::CONSTRAINT_TYPE_LIMITEDHINGE:
			{
				hkpLimitedHingeConstraintData* hingeData = static_cast<hkpLimitedHingeConstraintData*>(data);
				hkpLimitedHingeConstraintData::Atoms& atoms = hingeData->m_atoms;

				// Base transforms
				const hkTransform &pivotTransformA = atoms.m_transforms.m_transformA; // parent
				const hkTransform &pivotTransformB = atoms.m_transforms.m_transformB; // child

				// Get limited axis in world space
				// NOTE: looking in hkpLimitedHingeConstraintData, found that column(1) contains perpAxis, column(2) contains a second perpAxis
				hkVector4 perpAxisA_WS;  perpAxisA_WS._setRotatedDir(bodyATransform.getRotation(), pivotTransformA.getColumn<1>());
				hkVector4 perpAxisB_WS;  perpAxisB_WS._setRotatedDir(bodyBTransform.getRotation(), pivotTransformB.getColumn<1>());
				hkVector4 perpAxisBPerp_WS;   perpAxisBPerp_WS._setRotatedDir(bodyBTransform.getRotation(), pivotTransformB.getColumn<2>());

				// Loosen bend constraint to accommodate if needed
				{
					hkReal minBendAngle = atoms.m_angLimit.m_minAngle;
					hkReal maxBendAngle = atoms.m_angLimit.m_maxAngle;
#if defined(HK_PLATFORM_WIN32) && (HK_CONFIG_SIMD == HK_CONFIG_SIMD_ENABLED) && !defined(HK_ARCH_ARM)
					hkSimdReal acosDot; acosDot.setClamped( perpAxisA_WS.dot<3>(perpAxisB_WS), hkSimdReal_Minus1, hkSimdReal_1 );
#	if defined(HK_REAL_IS_DOUBLE)
					acosDot.m_real = hkMath::twoAcos(acosDot.m_real);
#	else
					acosDot.m_real = hkMath::quadAcos(acosDot.m_real);
#	endif
					acosDot.setFlipSign(acosDot, perpAxisA_WS.dot<3>(perpAxisBPerp_WS).lessZero());
					hkReal currentBendAngle = acosDot.getReal();
#else
					hkReal currentBendAngleSign = perpAxisA_WS.dot<3>(perpAxisBPerp_WS).isGreaterEqualZero() ? hkReal(1) : hkReal(-1);
					hkReal currentBendAngle = currentBendAngleSign * hkMath::acos(perpAxisA_WS.dot<3>(perpAxisB_WS).getReal());
#endif
					if (currentBendAngle >= hkReal(0))
					{
						atoms.m_angLimit.m_maxAngle = hkMath::max2(maxBendAngle, currentBendAngle + hkpConstraintDataUtils::smallTolerance);
					}
					else
					{
						atoms.m_angLimit.m_minAngle = hkMath::min2(minBendAngle, currentBendAngle - hkpConstraintDataUtils::smallTolerance);
					}
				}
			}
			break;

		default:
			HK_WARN_ALWAYS (0x7a501290, "This type of constraint is not supported for loosening.");
			return HK_FAILURE;
		}

		return HK_SUCCESS;
	}


#endif

	/// Returns the maximum size of the runtime for the given constraint data
	int HK_CALL getSizeOfRuntime(const hkpConstraintData* dataIn)
	{
		switch ( dataIn->getType() )
		{
		case hkpConstraintData::CONSTRAINT_TYPE_BALLANDSOCKET:		return sizeof(hkpBallAndSocketConstraintData::Runtime);
		case hkpConstraintData::CONSTRAINT_TYPE_DEFORMABLE_FIXED:	return sizeof(hkpDeformableFixedConstraintData::Runtime);
		case hkpConstraintData::CONSTRAINT_TYPE_FIXED:				return sizeof(hkpFixedConstraintData::Runtime);
		case hkpConstraintData::CONSTRAINT_TYPE_HINGE:				return sizeof(hkpHingeConstraintData::Runtime);
		case hkpConstraintData::CONSTRAINT_TYPE_LIMITEDHINGE:		return sizeof(hkpLimitedHingeConstraintData::Runtime);
		case hkpConstraintData::CONSTRAINT_TYPE_RAGDOLL:			return sizeof(hkpRagdollConstraintData::Runtime);
		case hkpConstraintData::CONSTRAINT_TYPE_RAGDOLL_LIMITS:		return sizeof(hkpRagdollLimitsData::Runtime);
		case hkpConstraintData::CONSTRAINT_TYPE_PRISMATIC:			return sizeof(hkpPrismaticConstraintData::Runtime);
		case hkpConstraintData::CONSTRAINT_TYPE_STIFFSPRING:		return sizeof(hkpStiffSpringConstraintData::Runtime);
		case hkpConstraintData::CONSTRAINT_TYPE_WHEEL:				return sizeof(hkpWheelConstraintData::Runtime);
		case hkpConstraintData::CONSTRAINT_TYPE_PULLEY:				return sizeof(hkpPulleyConstraintData::Runtime);
 		case hkpConstraintData::CONSTRAINT_TYPE_HINGE_LIMITS:		return sizeof(hkpHingeLimitsData::Runtime);
 		case hkpConstraintData::CONSTRAINT_TYPE_POINTTOPATH:		return sizeof(hkpPointToPathConstraintData::Runtime);
 		case hkpConstraintData::CONSTRAINT_TYPE_POINTTOPLANE:		return sizeof(hkpPointToPlaneConstraintData::Runtime);

//		case hkpConstraintData::CONSTRAINT_TYPE_COG_WHEEL:			return sizeof(hkpCogWheelConstraintData::Runtime);
// 		case hkpConstraintData::CONSTRAINT_TYPE_ROTATIONAL:			return sizeof(hkpRotationalConstraintData::Runtime);
// 		case hkpConstraintData::CONSTRAINT_TYPE_RACK_AND_PINION:	return sizeof(hkpRackAndPinionConstraintData::Runtime);
		default:
			HK_ASSERT2(0x40a64513, false, "Unsupported constraint type" );
			break;
		}

		return 0;
	}

	hkVector4 getLargestLinearImpulse( hkpConstraintData::ConstraintType constraintType, const hkpConstraintRuntime* runtimeIn, const hkQTransform& transformA, const hkQTransform& transformB )
	{
		hkVector4 maxImpulse;	maxImpulse.setZero();

		if ( !runtimeIn )
		{
			return maxImpulse;
		}

		switch ( constraintType )
		{
		case hkpConstraintData::CONSTRAINT_TYPE_BALLANDSOCKET:			// hkpBallAndSocketConstraintData
			{
				const hkpBallAndSocketConstraintData::Runtime* runtime = reinterpret_cast<const hkpBallAndSocketConstraintData::Runtime*>(runtimeIn);
				const hkReal* sr	= &runtime->m_solverResults[hkpBallAndSocketConstraintData::SOLVER_RESULT_LIN_0].m_impulseApplied;
				hkVector4 vImpulse;	vImpulse.set(sr[0], sr[2], sr[4]);

				maxImpulse._setRotatedDir(transformA.m_rotation, vImpulse);
			}
			break;

		case hkpConstraintData::CONSTRAINT_TYPE_DEFORMABLE_FIXED:		// hkpDeformableFixedConstraintData
			{
				const hkpDeformableFixedConstraintData::Runtime* runtime = reinterpret_cast<const hkpDeformableFixedConstraintData::Runtime*>(runtimeIn);
				const hkReal* sr	= &runtime->m_solverResults[hkpDeformableFixedConstraintData::SOLVER_RESULT_LIN_0].m_impulseApplied;
				hkVector4 vImpulse;	vImpulse.set(sr[0], sr[2], sr[4]);

				maxImpulse._setRotatedDir(transformA.m_rotation, vImpulse);
			}
			break;

		case hkpConstraintData::CONSTRAINT_TYPE_FIXED:					// hkpFixedConstraintData
			{
				const hkpFixedConstraintData::Runtime* runtime = reinterpret_cast<const hkpFixedConstraintData::Runtime*>(runtimeIn);
				const hkReal* sr	= (const hkReal*)&runtime->m_solverResults[hkpFixedConstraintData::SOLVER_RESULT_LIN_0].m_impulseApplied;
				hkVector4 vImpulse;	vImpulse.set(sr[0], sr[2], sr[4]);

				maxImpulse._setRotatedDir(transformA.m_rotation, vImpulse);
			}
			break;

		case hkpConstraintData::CONSTRAINT_TYPE_HINGE:					// hkpHingeConstraintData
			{
				const hkpHingeConstraintData::Runtime* runtime = reinterpret_cast<const hkpHingeConstraintData::Runtime*>(runtimeIn);
				const hkReal* sr	= (const hkReal*)&runtime->m_solverResults[hkpHingeConstraintData::SOLVER_RESULT_LIN_0].m_impulseApplied;
				hkVector4 vImpulse;	vImpulse.set(sr[0], sr[2], sr[4]);

				maxImpulse._setRotatedDir(transformA.m_rotation, vImpulse);
			}
			break;

		case hkpConstraintData::CONSTRAINT_TYPE_LIMITEDHINGE:			// hkpLimitedHingeConstraintData
			{
				const hkpLimitedHingeConstraintData::Runtime* runtime = reinterpret_cast<const hkpLimitedHingeConstraintData::Runtime*>(runtimeIn);
				const hkReal* sr	= &runtime->m_solverResults[hkpLimitedHingeConstraintData::SOLVER_RESULT_LIN_0].m_impulseApplied;
				hkVector4 vImpulse;	vImpulse.set(sr[0], sr[2], sr[4]);

				maxImpulse._setRotatedDir(transformA.m_rotation, vImpulse);
			}
			break;

		case hkpConstraintData::CONSTRAINT_TYPE_RAGDOLL:				// hkpRagdollConstraintData OR hkpAngularFrictionConstraintData?
			{
				const hkpRagdollConstraintData::Runtime* runtime = reinterpret_cast<const hkpRagdollConstraintData::Runtime*>(runtimeIn);
				const hkReal* sr	= &runtime->m_solverResults[hkpRagdollConstraintData::SOLVER_RESULT_LIN_0].m_impulseApplied;
				hkVector4 vImpulse;	vImpulse.set(sr[0], sr[2], sr[4]);

				maxImpulse._setRotatedDir(transformA.m_rotation, vImpulse);
			}
			break;

		case hkpConstraintData::CONSTRAINT_TYPE_RAGDOLL_LIMITS:			// hkpRagdollLimitsData
		case hkpConstraintData::CONSTRAINT_TYPE_ROTATIONAL:				// hkpRotationalConstraintData
			// No linear impulses!
			break;

			
/*		case hkpConstraintData::CONSTRAINT_TYPE_PRISMATIC:				// hkpPrismaticConstraintData
			{
				const hkpPrismaticConstraintData* prismatic = static_cast<const hkpPrismaticConstraintData*>(dataIn);
				const hkpPrismaticConstraintData::Runtime* runtime = reinterpret_cast<const hkpPrismaticConstraintData::Runtime*>(runtimeIn);


				for ( int i=7; i<=8; i++ )
				{
					hkVector4 impulse = prismatic->m_atoms.m_transforms.m_transformA.getRotation().getColumn(2);
					impulse.mul( hkSimdReal::fromFloat( runtime->m_solverResults[i].m_impulseApplied ) );
					if ( impulse.length<3>() > maxImpulse.length<3>() )
					{
						maxImpulse = impulse;
					}
				}
			}
			break;

		case hkpConstraintData::CONSTRAINT_TYPE_STIFFSPRING:			// hkpStiffSpringConstraintData
			{
				const hkpStiffSpringConstraintData* spring = static_cast<const hkpStiffSpringConstraintData*>(dataIn);
				const hkpStiffSpringConstraintData::Runtime* runtime = reinterpret_cast<const hkpStiffSpringConstraintData::Runtime*>(runtimeIn);

				hkVector4 p1; p1.setTransformedPos( transformA, spring->m_atoms.m_pivots.m_translationA );
				hkVector4 p2; p2.setTransformedPos( transformB, spring->m_atoms.m_pivots.m_translationB );

				hkVector4 impulse; impulse.setSub( p2, p1 );
				impulse.mul( hkSimdReal::fromFloat( runtime->m_solverResults[0].m_impulseApplied ) );
				if ( impulse.length<3>() > maxImpulse.length<3>() )
				{
					maxImpulse = impulse;
				}
			}
			break;

		case hkpConstraintData::CONSTRAINT_TYPE_WHEEL:					// hkpWheelConstraintData
			{
				const hkpWheelConstraintData* wheel = static_cast<const hkpWheelConstraintData*>(dataIn);
				const hkpWheelConstraintData::Runtime* runtime = reinterpret_cast<const hkpWheelConstraintData::Runtime*>(runtimeIn);
				hkRotation rotation; rotation.set( transformA.getRotation() );
				for ( int i=2; i<=4; i++ )
				{
					hkVector4 impulse;
					switch ( i )
					{
					case 2: impulse = rotation.getColumn( wheel->m_atoms.m_lin0Limit.m_axisIndex ); break;
					case 3: impulse = rotation.getColumn( wheel->m_atoms.m_lin1.m_axisIndex ); break;
					case 4: impulse = rotation.getColumn( wheel->m_atoms.m_lin2.m_axisIndex ); break;
					}
					impulse.mul( hkSimdReal::fromFloat( runtime->m_solverResults[i].m_impulseApplied ) );
					if ( impulse.length<3>() > maxImpulse.length<3>() )
					{
						maxImpulse = impulse;
					}
				}
			}
			break;*/

//		case hkpConstraintData::CONSTRAINT_TYPE_PULLEY:					// hkpPulleyConstraintData
//		case hkpConstraintData::CONSTRAINT_TYPE_RACK_AND_PINION:		// hkpRackAndPinionConstraintData
// 		case hkpConstraintData::CONSTRAINT_TYPE_COG_WHEEL:				// hkpCogWheelConstraintData
// 		case hkpConstraintData::CONSTRAINT_TYPE_HINGE_LIMITS:			// hkpHingeLimitsData
// 		case hkpConstraintData::CONSTRAINT_TYPE_POINTTOPATH:			// hkpPointToPathConstraintData
// 		case hkpConstraintData::CONSTRAINT_TYPE_POINTTOPLANE:			// hkpPointToPlaneConstraintData
		default:
			HK_ASSERT2(0x40a64513, false, "Unsupported constraint type" );
			break;
		}

		return maxImpulse;
	}
}

#ifndef HK_PLATFORM_SPU

hkBool HK_CALL hkpConstraintDataUtils::constraintSupportsPivotGetSet(const hkpConstraintData* data)
{
	switch(data->getType())
	{
	case hkpConstraintData::CONSTRAINT_TYPE_BALLANDSOCKET:
	case hkpConstraintData::CONSTRAINT_TYPE_FIXED:
	case hkpConstraintData::CONSTRAINT_TYPE_DEFORMABLE_FIXED:
	case hkpConstraintData::CONSTRAINT_TYPE_HINGE:
	case hkpConstraintData::CONSTRAINT_TYPE_LIMITEDHINGE:
	case hkpConstraintData::CONSTRAINT_TYPE_PRISMATIC:
	case hkpConstraintData::CONSTRAINT_TYPE_RAGDOLL:
	case hkpConstraintData::CONSTRAINT_TYPE_STIFFSPRING:
	case hkpConstraintData::CONSTRAINT_TYPE_WHEEL:
	case hkpConstraintData::CONSTRAINT_TYPE_POINTTOPLANE:
	case hkpConstraintData::CONSTRAINT_TYPE_PULLEY:
	case hkpConstraintData::CONSTRAINT_TYPE_BREAKABLE:
	case hkpConstraintData::CONSTRAINT_TYPE_MALLEABLE:
		return true;

	default:
		return false;
	}
}

//
//	Get the constraint's pivot for body i = 0 / 1.

hkVector4Parameter HK_CALL hkpConstraintDataUtils::getPivot(const hkpConstraintData* data, int pivotIndex)
{
	switch ( data->getType() )
	{
	case hkpConstraintData::CONSTRAINT_TYPE_BALLANDSOCKET:		{ const hkpBallAndSocketConstraintData*		d = static_cast<const hkpBallAndSocketConstraintData*>(data);		return (&d->m_atoms.m_pivots.m_translationA)[pivotIndex]; }
	case hkpConstraintData::CONSTRAINT_TYPE_FIXED:				{ const hkpFixedConstraintData*				d = static_cast<const hkpFixedConstraintData*>(data);				return (&d->m_atoms.m_transforms.m_transformA)[pivotIndex].getTranslation(); }
	case hkpConstraintData::CONSTRAINT_TYPE_DEFORMABLE_FIXED:	{ const hkpDeformableFixedConstraintData*	d = static_cast<const hkpDeformableFixedConstraintData*>(data);		return (&d->m_atoms.m_transforms.m_transformA)[pivotIndex].getTranslation(); }
	case hkpConstraintData::CONSTRAINT_TYPE_HINGE:				{ const hkpHingeConstraintData*				d = static_cast<const hkpHingeConstraintData*>(data);				return (&d->m_atoms.m_transforms.m_transformA)[pivotIndex].getTranslation(); }
	case hkpConstraintData::CONSTRAINT_TYPE_LIMITEDHINGE:		{ const hkpLimitedHingeConstraintData*		d = static_cast<const hkpLimitedHingeConstraintData*>(data);		return (&d->m_atoms.m_transforms.m_transformA)[pivotIndex].getTranslation(); }
	case hkpConstraintData::CONSTRAINT_TYPE_PRISMATIC:			{ const hkpPrismaticConstraintData*			d = static_cast<const hkpPrismaticConstraintData*>(data);			return (&d->m_atoms.m_transforms.m_transformA)[pivotIndex].getTranslation(); }
	case hkpConstraintData::CONSTRAINT_TYPE_RAGDOLL:			{ const hkpRagdollConstraintData*			d = static_cast<const hkpRagdollConstraintData*>(data);				return (&d->m_atoms.m_transforms.m_transformA)[pivotIndex].getTranslation(); }
	case hkpConstraintData::CONSTRAINT_TYPE_STIFFSPRING:		{ const hkpStiffSpringConstraintData*		d = static_cast<const hkpStiffSpringConstraintData*>(data);			return (&d->m_atoms.m_pivots.m_translationA)[pivotIndex]; }
	case hkpConstraintData::CONSTRAINT_TYPE_WHEEL:				{ const hkpWheelConstraintData*				d = static_cast<const hkpWheelConstraintData*>(data);				return (&d->m_atoms.m_suspensionBase.m_transformA)[pivotIndex].getTranslation(); }
	case hkpConstraintData::CONSTRAINT_TYPE_POINTTOPLANE:		{ const hkpPointToPlaneConstraintData*		d = static_cast<const hkpPointToPlaneConstraintData*>(data);		return (&d->m_atoms.m_transforms.m_transformA)[pivotIndex].getTranslation(); }
	case hkpConstraintData::CONSTRAINT_TYPE_PULLEY:				{ const hkpPulleyConstraintData*			d = static_cast<const hkpPulleyConstraintData*>(data);				return (&d->m_atoms.m_translations.m_translationA)[pivotIndex]; }

#if !defined ( HK_FEATURE_PRODUCT_PHYSICS )
	case hkpConstraintData::CONSTRAINT_TYPE_BREAKABLE:			return getPivot(static_cast<const hkpBreakableConstraintData*>(data)->m_constraintData, pivotIndex);
	case hkpConstraintData::CONSTRAINT_TYPE_MALLEABLE:			return getPivot(static_cast<const hkpMalleableConstraintData*>(data)->getWrappedConstraintData(), pivotIndex);
#endif

	default:
		{
			HK_WARN(0xad873344, "This constraint doesn't support getPivot() functionality.");
			return hkVector4::getZero();
		}
	}
}

const hkVector4& HK_CALL hkpConstraintDataUtils::getPivotA(const hkpConstraintData* data)
{
	switch(data->getType())
	{
		case hkpConstraintData::CONSTRAINT_TYPE_BALLANDSOCKET:		return static_cast<const hkpBallAndSocketConstraintData*>(	data)->m_atoms.m_pivots.m_translationA;
		case hkpConstraintData::CONSTRAINT_TYPE_FIXED:				return static_cast<const hkpFixedConstraintData*>(			data)->m_atoms.m_transforms.m_transformA.getTranslation();
		case hkpConstraintData::CONSTRAINT_TYPE_DEFORMABLE_FIXED:	return static_cast<const hkpDeformableFixedConstraintData*>(data)->m_atoms.m_transforms.m_transformA.getTranslation();
		case hkpConstraintData::CONSTRAINT_TYPE_HINGE:				return static_cast<const hkpHingeConstraintData*>(			data)->m_atoms.m_transforms.m_transformA.getTranslation();
		case hkpConstraintData::CONSTRAINT_TYPE_LIMITEDHINGE:		return static_cast<const hkpLimitedHingeConstraintData*>(	data)->m_atoms.m_transforms.m_transformA.getTranslation();
		case hkpConstraintData::CONSTRAINT_TYPE_PRISMATIC:			return static_cast<const hkpPrismaticConstraintData*>(		data)->m_atoms.m_transforms.m_transformA.getTranslation();
		case hkpConstraintData::CONSTRAINT_TYPE_RAGDOLL:			return static_cast<const hkpRagdollConstraintData*>(		data)->m_atoms.m_transforms.m_transformA.getTranslation();
		case hkpConstraintData::CONSTRAINT_TYPE_STIFFSPRING:		return static_cast<const hkpStiffSpringConstraintData*>(	data)->m_atoms.m_pivots.m_translationA;
		case hkpConstraintData::CONSTRAINT_TYPE_WHEEL:				return static_cast<const hkpWheelConstraintData*>(			data)->m_atoms.m_suspensionBase.m_transformA.getTranslation();
		case hkpConstraintData::CONSTRAINT_TYPE_POINTTOPLANE:		return static_cast<const hkpPointToPlaneConstraintData*>(	data)->m_atoms.m_transforms.m_transformA.getTranslation();
		case hkpConstraintData::CONSTRAINT_TYPE_PULLEY:				return static_cast<const hkpPulleyConstraintData*>(			data)->m_atoms.m_translations.m_translationA;

#if !defined ( HK_FEATURE_PRODUCT_PHYSICS )
		case hkpConstraintData::CONSTRAINT_TYPE_BREAKABLE:			return getPivotA( static_cast<const hkpBreakableConstraintData*>(data)->m_constraintData );
		case hkpConstraintData::CONSTRAINT_TYPE_MALLEABLE:			return getPivotA( static_cast<const hkpMalleableConstraintData*>(data)->getWrappedConstraintData() );
#endif

		//
		// Those constraints are a contact, or don't have pivots (rotational only), or have multiple pivots (chains), are just left out for the moment
		//

		//case hkpConstraintData::CONSTRAINT_TYPE_CONTACT:

		//case hkpConstraintData::CONSTRAINT_TYPE_ROTATIONAL:
		//case hkpConstraintData::CONSTRAINT_TYPE_HINGE_LIMITS:
		//case hkpConstraintData::CONSTRAINT_TYPE_RAGDOLL_LIMITS:

		//case hkpConstraintData::CONSTRAINT_TYPE_STIFF_SPRING_CHAIN:
		//case hkpConstraintData::CONSTRAINT_TYPE_BALL_SOCKET_CHAIN:
		//case hkpConstraintData::CONSTRAINT_TYPE_POWERED_CHAIN:

		//case hkpConstraintData::CONSTRAINT_TYPE_POINTTOPATH:
		//case hkpConstraintData::CONSTRAINT_TYPE_CUSTOM:
		//case hkpConstraintData::CONSTRAINT_TYPE_GENERIC:

		default:
		{
			HK_WARN(0xad873344, "This constraint doesn't support getPivot() functionality.");
			return hkVector4::getZero();
		}
	}
}

const hkVector4& HK_CALL hkpConstraintDataUtils::getPivotB(const hkpConstraintData* data)
{
	switch(data->getType())
	{
		case hkpConstraintData::CONSTRAINT_TYPE_BALLANDSOCKET:		return static_cast<const hkpBallAndSocketConstraintData*>(	data)->m_atoms.m_pivots.m_translationB;
		case hkpConstraintData::CONSTRAINT_TYPE_FIXED:				return static_cast<const hkpFixedConstraintData*>(			data)->m_atoms.m_transforms.m_transformB.getTranslation();
		case hkpConstraintData::CONSTRAINT_TYPE_DEFORMABLE_FIXED:	return static_cast<const hkpDeformableFixedConstraintData*>(data)->m_atoms.m_transforms.m_transformB.getTranslation();
		case hkpConstraintData::CONSTRAINT_TYPE_HINGE:				return static_cast<const hkpHingeConstraintData*>(			data)->m_atoms.m_transforms.m_transformB.getTranslation();
		case hkpConstraintData::CONSTRAINT_TYPE_LIMITEDHINGE:		return static_cast<const hkpLimitedHingeConstraintData*>(	data)->m_atoms.m_transforms.m_transformB.getTranslation();
		case hkpConstraintData::CONSTRAINT_TYPE_PRISMATIC:			return static_cast<const hkpPrismaticConstraintData*>(		data)->m_atoms.m_transforms.m_transformB.getTranslation();
		case hkpConstraintData::CONSTRAINT_TYPE_RAGDOLL:			return static_cast<const hkpRagdollConstraintData*>(		data)->m_atoms.m_transforms.m_transformB.getTranslation();
		case hkpConstraintData::CONSTRAINT_TYPE_STIFFSPRING:		return static_cast<const hkpStiffSpringConstraintData*>(	data)->m_atoms.m_pivots.m_translationB;
		case hkpConstraintData::CONSTRAINT_TYPE_WHEEL:				return static_cast<const hkpWheelConstraintData*>(			data)->m_atoms.m_suspensionBase.m_transformB.getTranslation();
		case hkpConstraintData::CONSTRAINT_TYPE_POINTTOPLANE:		return static_cast<const hkpPointToPlaneConstraintData*>(	data)->m_atoms.m_transforms.m_transformB.getTranslation();
		case hkpConstraintData::CONSTRAINT_TYPE_PULLEY:				return static_cast<const hkpPulleyConstraintData*>(			data)->m_atoms.m_translations.m_translationB;

#if !defined ( HK_FEATURE_PRODUCT_PHYSICS )
		case hkpConstraintData::CONSTRAINT_TYPE_BREAKABLE:			return getPivotB( static_cast<const hkpBreakableConstraintData*>(data)->m_constraintData );
		case hkpConstraintData::CONSTRAINT_TYPE_MALLEABLE:			return getPivotB( static_cast<const hkpMalleableConstraintData*>(data)->getWrappedConstraintData() );
#endif

		//
		// Those constraints are a contact, or don't have pivots (rotational only), or have multiple pivots (chains), are just left out for the moment
		//

		//case hkpConstraintData::CONSTRAINT_TYPE_CONTACT:

		//case hkpConstraintData::CONSTRAINT_TYPE_ROTATIONAL:
		//case hkpConstraintData::CONSTRAINT_TYPE_HINGE_LIMITS:
		//case hkpConstraintData::CONSTRAINT_TYPE_RAGDOLL_LIMITS:

		//case hkpConstraintData::CONSTRAINT_TYPE_STIFF_SPRING_CHAIN:
		//case hkpConstraintData::CONSTRAINT_TYPE_BALL_SOCKET_CHAIN:
		//case hkpConstraintData::CONSTRAINT_TYPE_POWERED_CHAIN:

		//case hkpConstraintData::CONSTRAINT_TYPE_POINTTOPATH:
		//case hkpConstraintData::CONSTRAINT_TYPE_CUSTOM:
		//case hkpConstraintData::CONSTRAINT_TYPE_GENERIC:

		default:
		{
			HK_WARN(0xad873344, "This constraint doesn't support getPivot() functionality.");
			return hkVector4::getZero();
		}
	}
}


void HK_CALL hkpConstraintDataUtils::setPivot(hkpConstraintData* data, const hkVector4& pivot, int index)
{
	switch(data->getType())
	{
	case hkpConstraintData::CONSTRAINT_TYPE_HINGE:				{ hkpHingeConstraintData* d				= static_cast<hkpHingeConstraintData*>(				data); (&d->m_atoms.m_transforms.m_transformA)[index].setTranslation(pivot); break; }
	case hkpConstraintData::CONSTRAINT_TYPE_LIMITEDHINGE:		{ hkpLimitedHingeConstraintData* d		= static_cast<hkpLimitedHingeConstraintData*>(		data); (&d->m_atoms.m_transforms.m_transformA)[index].setTranslation(pivot); break; }
	case hkpConstraintData::CONSTRAINT_TYPE_PRISMATIC:			{ hkpPrismaticConstraintData* d			= static_cast<hkpPrismaticConstraintData*>(			data); (&d->m_atoms.m_transforms.m_transformA)[index].setTranslation(pivot); break; }
	case hkpConstraintData::CONSTRAINT_TYPE_RAGDOLL:			{ hkpRagdollConstraintData* d			= static_cast<hkpRagdollConstraintData*>(			data); (&d->m_atoms.m_transforms.m_transformA)[index].setTranslation(pivot); break; }
	case hkpConstraintData::CONSTRAINT_TYPE_WHEEL:				{ hkpWheelConstraintData* d				= static_cast<hkpWheelConstraintData*>(				data); (&d->m_atoms.m_suspensionBase.m_transformA)[index].setTranslation(pivot); break; }
	case hkpConstraintData::CONSTRAINT_TYPE_POINTTOPLANE:		{ hkpPointToPlaneConstraintData* d		= static_cast<hkpPointToPlaneConstraintData*>(		data); (&d->m_atoms.m_transforms.m_transformA)[index].setTranslation(pivot); break; }
	case hkpConstraintData::CONSTRAINT_TYPE_PULLEY:				{ hkpPulleyConstraintData* d			= static_cast<hkpPulleyConstraintData*>(			data); (&d->m_atoms.m_translations.m_translationA)[index] = pivot; break; }
	case hkpConstraintData::CONSTRAINT_TYPE_BALLANDSOCKET:		{ hkpBallAndSocketConstraintData* d		= static_cast<hkpBallAndSocketConstraintData*>(		data); (&d->m_atoms.m_pivots.m_translationA)[index] = pivot; break; }
	case hkpConstraintData::CONSTRAINT_TYPE_FIXED:				{ hkpFixedConstraintData* d				= static_cast<hkpFixedConstraintData*>(		  		data); (&d->m_atoms.m_transforms.m_transformA)[index].setTranslation(pivot); break; }
	case hkpConstraintData::CONSTRAINT_TYPE_DEFORMABLE_FIXED:	{ hkpDeformableFixedConstraintData* d	= static_cast<hkpDeformableFixedConstraintData*>(	data); (&d->m_atoms.m_transforms.m_transformA)[index].setTranslation(pivot); break; }
	case hkpConstraintData::CONSTRAINT_TYPE_STIFFSPRING:		{ hkpStiffSpringConstraintData* d		= static_cast<hkpStiffSpringConstraintData*>(		data); (&d->m_atoms.m_pivots.m_translationA)[index] = pivot; break; }

#if !defined ( HK_FEATURE_PRODUCT_PHYSICS )
	case hkpConstraintData::CONSTRAINT_TYPE_BREAKABLE:			{ setPivot( static_cast<hkpBreakableConstraintData*>(data)->m_constraintData, pivot, index ); break; }
	case hkpConstraintData::CONSTRAINT_TYPE_MALLEABLE:			{ setPivot( static_cast<hkpMalleableConstraintData*>(data)->getWrappedConstraintData(), pivot, index ); break; }
#endif

															//
															// Those constraints are a contact, or don't have pivots (rotational only), or have multiple pivots (chains), are just left out for the moment
															//

															//case hkpConstraintData::CONSTRAINT_TYPE_CONTACT:

															//case hkpConstraintData::CONSTRAINT_TYPE_ROTATIONAL:
															//case hkpConstraintData::CONSTRAINT_TYPE_HINGE_LIMITS:
															//case hkpConstraintData::CONSTRAINT_TYPE_RAGDOLL_LIMITS:

															//case hkpConstraintData::CONSTRAINT_TYPE_STIFF_SPRING_CHAIN:
															//case hkpConstraintData::CONSTRAINT_TYPE_BALL_SOCKET_CHAIN:
															//case hkpConstraintData::CONSTRAINT_TYPE_POWERED_CHAIN:

															//case hkpConstraintData::CONSTRAINT_TYPE_POINTTOPATH:
															//case hkpConstraintData::CONSTRAINT_TYPE_CUSTOM:
															//case hkpConstraintData::CONSTRAINT_TYPE_GENERIC:

	default:
		{
			HK_WARN(0xad873344, "This constraint doesn't support setPivot() functionality.");
		}
	}
}

void HK_CALL hkpConstraintDataUtils::setPivotTransform(hkpConstraintData* data, const hkTransform& pivot, int index)
{
	switch(data->getType())
	{
	case hkpConstraintData::CONSTRAINT_TYPE_HINGE:				{ hkpHingeConstraintData* d				= static_cast<hkpHingeConstraintData*>(        data); (&d->m_atoms.m_transforms.m_transformA)[index] = pivot; break; }
	case hkpConstraintData::CONSTRAINT_TYPE_LIMITEDHINGE:		{ hkpLimitedHingeConstraintData* d		= static_cast<hkpLimitedHingeConstraintData*>( data); (&d->m_atoms.m_transforms.m_transformA)[index] = pivot; break; }
	case hkpConstraintData::CONSTRAINT_TYPE_PRISMATIC:			{ hkpPrismaticConstraintData* d			= static_cast<hkpPrismaticConstraintData*>(    data); (&d->m_atoms.m_transforms.m_transformA)[index] = pivot; break; }
	case hkpConstraintData::CONSTRAINT_TYPE_RAGDOLL:			{
																	hkpRagdollConstraintData* d			= static_cast<hkpRagdollConstraintData*>(		data);
																	(&d->m_atoms.m_transforms.m_transformA)[index] = pivot;
																	break;
																}
	case hkpConstraintData::CONSTRAINT_TYPE_POINTTOPLANE:		{ hkpPointToPlaneConstraintData* d		= static_cast<hkpPointToPlaneConstraintData*>(		data); (&d->m_atoms.m_transforms.	m_transformA)[index]	= pivot;					break; }
	case hkpConstraintData::CONSTRAINT_TYPE_WHEEL:				{ hkpWheelConstraintData* d				= static_cast<hkpWheelConstraintData*>(				data); (&d->m_atoms.m_suspensionBase.m_transformA)[index]	= pivot;					break; }
	case hkpConstraintData::CONSTRAINT_TYPE_FIXED:				{ hkpFixedConstraintData* d				= static_cast<hkpFixedConstraintData*>(		   		data); (&d->m_atoms.m_transforms.	m_transformA)[index]	= pivot;					break; }
	case hkpConstraintData::CONSTRAINT_TYPE_DEFORMABLE_FIXED:	{ hkpDeformableFixedConstraintData* d	= static_cast<hkpDeformableFixedConstraintData*>(	data); (&d->m_atoms.m_transforms.	m_transformA)[index]	= pivot;					break; }
	case hkpConstraintData::CONSTRAINT_TYPE_BALLANDSOCKET:		{ hkpBallAndSocketConstraintData* d		= static_cast<hkpBallAndSocketConstraintData*>(		data); (&d->m_atoms.m_pivots.		m_translationA)[index]	= pivot.getTranslation();	break; }
	case hkpConstraintData::CONSTRAINT_TYPE_STIFFSPRING:		{ hkpStiffSpringConstraintData* d		= static_cast<hkpStiffSpringConstraintData*>(		data); (&d->m_atoms.m_pivots.		m_translationA)[index]	= pivot.getTranslation();	break; }
	case hkpConstraintData::CONSTRAINT_TYPE_PULLEY:				{ hkpPulleyConstraintData* d			= static_cast<hkpPulleyConstraintData*>(			data); (&d->m_atoms.m_translations.	m_translationA)[index]	= pivot.getTranslation();	break; }

#if !defined ( HK_FEATURE_PRODUCT_PHYSICS )
	case hkpConstraintData::CONSTRAINT_TYPE_BREAKABLE:			{ setPivotTransform( static_cast<hkpBreakableConstraintData*>(data)->m_constraintData, pivot, index ); break; }
	case hkpConstraintData::CONSTRAINT_TYPE_MALLEABLE:			{ setPivotTransform( static_cast<hkpMalleableConstraintData*>(data)->getWrappedConstraintData(), pivot, index ); break; }
#endif

															//
															// Those constraints are a contact, or don't have pivots (rotational only), or have multiple pivots (chains), are just left out for the moment
															//

															//case hkpConstraintData::CONSTRAINT_TYPE_CONTACT:

															//case hkpConstraintData::CONSTRAINT_TYPE_ROTATIONAL:
															//case hkpConstraintData::CONSTRAINT_TYPE_HINGE_LIMITS:
															//case hkpConstraintData::CONSTRAINT_TYPE_RAGDOLL_LIMITS:

															//case hkpConstraintData::CONSTRAINT_TYPE_STIFF_SPRING_CHAIN:
															//case hkpConstraintData::CONSTRAINT_TYPE_BALL_SOCKET_CHAIN:
															//case hkpConstraintData::CONSTRAINT_TYPE_POWERED_CHAIN:

															//case hkpConstraintData::CONSTRAINT_TYPE_POINTTOPATH:
															//case hkpConstraintData::CONSTRAINT_TYPE_CUSTOM:
															//case hkpConstraintData::CONSTRAINT_TYPE_GENERIC:

	default:
		{
			HK_WARN(0xad873344, "This constraint doesn't support setPivot() functionality.");
		}
	}
}

void HK_CALL hkpConstraintDataUtils::getPivotTransform( const hkpConstraintData* data, int index, hkTransform& pivot )
{
	pivot.setIdentity();

	switch(data->getType())
	{
	case hkpConstraintData::CONSTRAINT_TYPE_HINGE:				{ const hkpHingeConstraintData* d			= static_cast<const hkpHingeConstraintData*>(			data); pivot = (&d->m_atoms.m_transforms.m_transformA)[index]; break; }
	case hkpConstraintData::CONSTRAINT_TYPE_LIMITEDHINGE:		{ const hkpLimitedHingeConstraintData* d	= static_cast<const hkpLimitedHingeConstraintData*>(	data); pivot = (&d->m_atoms.m_transforms.m_transformA)[index]; break; }
	case hkpConstraintData::CONSTRAINT_TYPE_PRISMATIC:			{ const hkpPrismaticConstraintData* d		= static_cast<const hkpPrismaticConstraintData*>(		data); pivot = (&d->m_atoms.m_transforms.m_transformA)[index]; break; }
	case hkpConstraintData::CONSTRAINT_TYPE_RAGDOLL:			{ const hkpRagdollConstraintData* d			= static_cast<const hkpRagdollConstraintData*>(			data); pivot = (&d->m_atoms.m_transforms.m_transformA)[index]; break; }
	case hkpConstraintData::CONSTRAINT_TYPE_POINTTOPLANE:		{ const hkpPointToPlaneConstraintData* d	= static_cast<const hkpPointToPlaneConstraintData*>(	data); pivot = (&d->m_atoms.m_transforms.m_transformA)[index]; break; }
	case hkpConstraintData::CONSTRAINT_TYPE_FIXED:				{ const hkpFixedConstraintData* d		 	= static_cast<const hkpFixedConstraintData*>(			data); pivot = (&d->m_atoms.m_transforms.m_transformA)[index]; break; }
	case hkpConstraintData::CONSTRAINT_TYPE_DEFORMABLE_FIXED:	{ const hkpDeformableFixedConstraintData* d	= static_cast<const hkpDeformableFixedConstraintData*>(	data); pivot = (&d->m_atoms.m_transforms.m_transformA)[index]; break; }
	case hkpConstraintData::CONSTRAINT_TYPE_WHEEL:				{ const hkpWheelConstraintData* d			= static_cast<const hkpWheelConstraintData*>(			data); pivot = (&d->m_atoms.m_suspensionBase.m_transformA)[index]; break; }
	case hkpConstraintData::CONSTRAINT_TYPE_BALLANDSOCKET:		{ const hkpBallAndSocketConstraintData* d	= static_cast<const hkpBallAndSocketConstraintData*>(	data); pivot.setTranslation((&d->m_atoms.m_pivots.m_translationA)[index] ); break; }
	case hkpConstraintData::CONSTRAINT_TYPE_STIFFSPRING:		{ const hkpStiffSpringConstraintData* d		= static_cast<const hkpStiffSpringConstraintData*>(		data); pivot.setTranslation((&d->m_atoms.m_pivots.m_translationA)[index] ); break; }
	case hkpConstraintData::CONSTRAINT_TYPE_PULLEY:				{ const hkpPulleyConstraintData* d			= static_cast<const hkpPulleyConstraintData*>(			data); pivot.setTranslation((&d->m_atoms.m_translations.m_translationA)[index] ); break; }

#if !defined ( HK_FEATURE_PRODUCT_PHYSICS )
	case hkpConstraintData::CONSTRAINT_TYPE_BREAKABLE:			{ getPivotTransform( static_cast<const hkpBreakableConstraintData*>(data)->m_constraintData, index, pivot); break; }
	case hkpConstraintData::CONSTRAINT_TYPE_MALLEABLE:			{ getPivotTransform( static_cast<const hkpMalleableConstraintData*>(data)->getWrappedConstraintData(), index, pivot ); break; }
#endif

															//
															// Those constraints are a contact, or don't have pivots (rotational only), or have multiple pivots (chains), are just left out for the moment
															//

															//case hkpConstraintData::CONSTRAINT_TYPE_CONTACT:

															//case hkpConstraintData::CONSTRAINT_TYPE_ROTATIONAL:
															//case hkpConstraintData::CONSTRAINT_TYPE_HINGE_LIMITS:
															//case hkpConstraintData::CONSTRAINT_TYPE_RAGDOLL_LIMITS:

															//case hkpConstraintData::CONSTRAINT_TYPE_STIFF_SPRING_CHAIN:
															//case hkpConstraintData::CONSTRAINT_TYPE_BALL_SOCKET_CHAIN:
															//case hkpConstraintData::CONSTRAINT_TYPE_POWERED_CHAIN:

															//case hkpConstraintData::CONSTRAINT_TYPE_POINTTOPATH:
															//case hkpConstraintData::CONSTRAINT_TYPE_CUSTOM:
															//case hkpConstraintData::CONSTRAINT_TYPE_GENERIC:

	default:
		{
			HK_WARN(0xad873344, "This constraint doesn't support setPivot() functionality.");
		}
	}
}


hkpConstraintData* HK_CALL hkpConstraintDataUtils::cloneIfCanHaveMotors(const hkpConstraintData* data)
{
	hkpConstraintData::ConstraintType type = (hkpConstraintData::ConstraintType)data->getType();

	switch(type)
	{
	case hkpConstraintData::CONSTRAINT_TYPE_LIMITEDHINGE:
		{
			const hkpLimitedHingeConstraintData* oldData = static_cast<const hkpLimitedHingeConstraintData*>(data);

			// Create new instance
			hkpLimitedHingeConstraintData* newData = new hkpLimitedHingeConstraintData();

			// Copy all data
			newData->m_atoms = oldData->m_atoms;
			newData->m_userData = oldData->m_userData;

			// Clone motors
			HK_CPU_PTR(class hkpConstraintMotor*)& motor = newData->m_atoms.m_angMotor.m_motor;
			if (motor)
			{
				motor = motor->clone();
			}

			// All done
			return newData;
		}

	case hkpConstraintData::CONSTRAINT_TYPE_PRISMATIC:
		{
			const hkpPrismaticConstraintData* oldData = static_cast<const hkpPrismaticConstraintData*>(data);

			// Create new instance
			hkpPrismaticConstraintData* newData = new hkpPrismaticConstraintData();

			// Copy all data
			newData->m_atoms = oldData->m_atoms;
			newData->m_userData = oldData->m_userData;

			// Clone motors
			HK_CPU_PTR(class hkpConstraintMotor*)& motor = newData->m_atoms.m_motor.m_motor;
			if (motor)
			{
				motor = motor->clone();
			}

			// All done
			return newData;
		}

	case hkpConstraintData::CONSTRAINT_TYPE_RAGDOLL:
		{
			const hkpRagdollConstraintData* oldData = static_cast<const hkpRagdollConstraintData*>(data);

			// Create new instance
			hkpRagdollConstraintData* newData = new hkpRagdollConstraintData();

			// Copy all data
			newData->m_atoms = oldData->m_atoms;
			newData->m_userData = oldData->m_userData;

			// Clone motors
			HK_CPU_PTR(class hkpConstraintMotor*)* motors = newData->m_atoms.m_ragdollMotors.m_motors;
			for (int i = 0; i < 3; i++)
			{
				if (motors[i])
				{
					motors[i] = motors[i]->clone();
				}
			}

			// All done
			return newData;
		}

	default:
		{
			// Constraint doesn't have motors
			return HK_NULL;
		}
	}
}

template <class ConstraintDataType>
static ConstraintDataType* HK_CALL hkpConstraintDataUtils_copyAtoms(const hkpConstraintData* constraintIn)
{
	ConstraintDataType* constraintOut = new ConstraintDataType();

	const int size	= static_cast<const ConstraintDataType*>(constraintIn)->m_atoms.getSizeOfAllAtoms();
	const hkpConstraintAtom* src_atom = static_cast<const ConstraintDataType*>(constraintIn)->m_atoms.getAtoms();
	const hkpConstraintAtom* dst_atom = static_cast<ConstraintDataType*>(constraintOut)->m_atoms.getAtoms();
	void* dst = (void*)dst_atom;

	hkString::memCpy(dst, src_atom, size);

	return constraintOut;
}

hkpConstraintData* HK_CALL hkpConstraintDataUtils::deepClone(const hkpConstraintData* data)
{
	HK_ASSERT2(0x38dbcef2, data, "Constraint data is null");

	switch(data->getType())
	{
	case hkpConstraintData::CONSTRAINT_TYPE_BALLANDSOCKET:		return hkpConstraintDataUtils_copyAtoms<hkpBallAndSocketConstraintData>(data);
	case hkpConstraintData::CONSTRAINT_TYPE_FIXED:				return hkpConstraintDataUtils_copyAtoms<hkpFixedConstraintData>(data);
	case hkpConstraintData::CONSTRAINT_TYPE_DEFORMABLE_FIXED:	return hkpConstraintDataUtils_copyAtoms<hkpDeformableFixedConstraintData>(data);
	case hkpConstraintData::CONSTRAINT_TYPE_HINGE:				return hkpConstraintDataUtils_copyAtoms<hkpHingeConstraintData>(data);
	case hkpConstraintData::CONSTRAINT_TYPE_STIFFSPRING:		return hkpConstraintDataUtils_copyAtoms<hkpStiffSpringConstraintData>(data);
	case hkpConstraintData::CONSTRAINT_TYPE_WHEEL:				return hkpConstraintDataUtils_copyAtoms<hkpWheelConstraintData>(data);
	case hkpConstraintData::CONSTRAINT_TYPE_POINTTOPLANE:		return hkpConstraintDataUtils_copyAtoms<hkpPointToPlaneConstraintData>(data);
	case hkpConstraintData::CONSTRAINT_TYPE_PULLEY:				return hkpConstraintDataUtils_copyAtoms<hkpPulleyConstraintData>(data);
	case hkpConstraintData::CONSTRAINT_TYPE_HINGE_LIMITS:		return hkpConstraintDataUtils_copyAtoms<hkpHingeLimitsData>(data);
	case hkpConstraintData::CONSTRAINT_TYPE_RAGDOLL_LIMITS:		return hkpConstraintDataUtils_copyAtoms<hkpRagdollLimitsData>(data);

	case hkpConstraintData::CONSTRAINT_TYPE_LIMITEDHINGE:
		{
			hkpLimitedHingeConstraintData* limitedHingeData = hkpConstraintDataUtils_copyAtoms<hkpLimitedHingeConstraintData>(data);

			// Add a reference to the motor, if it exists. If you wish to deepClone this motor, use cloneIfCanHaveMotors()
			
			if( limitedHingeData->getMotor() )
			{
				limitedHingeData->getMotor()->addReference();
			}

			return limitedHingeData;
		}

	case hkpConstraintData::CONSTRAINT_TYPE_POINTTOPATH:
		{
			hkpPointToPathConstraintData* copy = hkpConstraintDataUtils_copyAtoms<hkpPointToPathConstraintData>(data);
			hkpParametricCurve* path = static_cast<const hkpPointToPathConstraintData*>(data)->getPath();

			hkpParametricCurve* clone = path->clone();

			copy->setPath(path);
			clone->removeReference();

			return copy;
		}

	case hkpConstraintData::CONSTRAINT_TYPE_PRISMATIC:
		{
			hkpPrismaticConstraintData* prismaticData = hkpConstraintDataUtils_copyAtoms<hkpPrismaticConstraintData>(data);

			// Add a reference to the motor, if it exists. If you wish to deepClone this motor, use cloneIfCanHaveMotors()
			
			if( prismaticData->getMotor() )
			{
				prismaticData->getMotor()->addReference();
			}

			return prismaticData;
		}

	case hkpConstraintData::CONSTRAINT_TYPE_RAGDOLL:
		{
			hkpRagdollConstraintData* ragdollData = hkpConstraintDataUtils_copyAtoms<hkpRagdollConstraintData>(data);

			// Add a reference to the motors, if they exist. If you wish to deepClone this motor, use cloneIfCanHaveMotors()
			
			if( ragdollData->getTwistMotor() )
			{
				ragdollData->getTwistMotor()->addReference();
			}

			if( ragdollData->getConeMotor() )
			{
				ragdollData->getConeMotor()->addReference();
			}

			if( ragdollData->getPlaneMotor() )
			{
				ragdollData->getPlaneMotor()->addReference();
			}

			return ragdollData;
		}

		// Generic & contact constrainsts
	case hkpConstraintData::CONSTRAINT_TYPE_GENERIC:
	case hkpConstraintData::CONSTRAINT_TYPE_CONTACT:
	case hkpConstraintData::CONSTRAINT_TYPE_STIFF_SPRING_CHAIN:
	case hkpConstraintData::CONSTRAINT_TYPE_BALL_SOCKET_CHAIN:
	case hkpConstraintData::CONSTRAINT_TYPE_POWERED_CHAIN:
	default:
		{
			HK_ASSERT2(0xad8754dd, false, "Cloning of chain/generic/simplecontact costraints not supported.");
			return HK_NULL;
		}
	}
}

#endif

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
