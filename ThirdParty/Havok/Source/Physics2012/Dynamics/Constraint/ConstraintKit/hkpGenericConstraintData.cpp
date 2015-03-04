/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/Constraint/ConstraintKit/hkpGenericConstraintData.h>
#include <Physics2012/Dynamics/Constraint/ConstraintKit/hkpConstraintModifier.h>
#include <Physics2012/Dynamics/Constraint/ConstraintKit/hkpGenericConstraintParameters.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics/ConstraintSolver/Constraint/Bilateral/hkp1dBilateralConstraintInfo.h>
#include <Physics/Constraint/Motor/hkpConstraintMotor.h>
#include <Physics/ConstraintSolver/Constraint/hkpConstraintQueryIn.h>
#include <Physics/ConstraintSolver/Constraint/hkpConstraintQueryOut.h>
#include <Physics/ConstraintSolver/Constraint/Bilateral/hkpInternalConstraintUtils.h>
#include <Physics/ConstraintSolver/Constraint/Motor/hkpMotorConstraintInfo.h>
#include <Physics/ConstraintSolver/Accumulator/hkpVelocityAccumulator.h>

// adding to use the HK_SCHEMA_INIT macro
#include <Physics/ConstraintSolver/Jacobian/hkpJacobianSchema.h>

hkpGenericConstraintData::hkpGenericConstraintData() 
{
	m_scheme.m_info.clear();
	m_scheme.m_info.addHeader();
	m_atoms.m_bridgeAtom.init( this );
}

hkpGenericConstraintData::hkpGenericConstraintData(hkFinishLoadedObjectFlag f) : hkpConstraintData(f), m_atoms(f), m_scheme(f)
{
	if( f.m_finishing )
	{
		m_atoms.m_bridgeAtom.init( this );
	}
}


hkpGenericConstraintData::~hkpGenericConstraintData()
{
	int i;
	for( i = 0; i < m_scheme.m_motors.getSize(); i++ )
	{
		m_scheme.m_motors[i]->removeReference();
	}

	//!me should constraint modifiers be reference counted as well?
}

void hkpGenericConstraintData::getConstraintInfo( hkpConstraintData::ConstraintInfo& info ) const
{
	info.m_atoms = const_cast<hkpConstraintAtom*>(m_atoms.getAtoms());
	info.m_sizeOfAllAtoms = m_atoms.getSizeOfAllAtoms();
	info.clear();
	(hkpConstraintInfo&)info = m_scheme.m_info;
}

void hkpGenericConstraintData::getRuntimeInfo( hkBool wantRuntime, hkpConstraintData::RuntimeInfo& infoOut ) const 
{
	// we need runtime data to be able to support lastAngle and friction
	infoOut.m_numSolverResults = m_scheme.m_info.m_numSolverResults;
	infoOut.m_sizeOfExternalRuntime = sizeof( hkpSolverResults) * infoOut.m_numSolverResults;
}

void hkpGenericConstraintData::buildJacobian( const hkpConstraintQueryIn &in, hkpConstraintQueryOut &out )
{
	hkpSolverResults* solverResults = reinterpret_cast<hkpSolverResults*>( out.m_constraintRuntime.val() );
	in.beginConstraints( out, solverResults, sizeof(hkpSolverResults) );
	hatchScheme( &m_scheme, in, out );
	hkEndConstraints();
}


hkVector4* hkpGenericConstraintData::getParameters( hkpParameterIndex parameterIndex )
{
	return &m_scheme.m_data[ parameterIndex ];
}


void hkpGenericConstraintData::setParameters( hkpParameterIndex parameterIndex, int numParameters, const hkVector4* newValues )
{
	int i;
	for( i = parameterIndex; i < parameterIndex + numParameters; i++, newValues++ )
	{
		m_scheme.m_data[ i ] = *newValues;
	}
}

//
// commands 
//

// linear constraints

void hkpGenericConstraintData::constrainAllLinearW( hkArray<int>::iterator& currentCommand, hkArray<hkVector4>::iterator& currentData, const hkpGenericConstraintDataScheme& scheme, hkpGenericConstraintDataParameters& vars, const hkpConstraintQueryIn &in, hkpConstraintQueryOut &out ) const
{
	hkp1dLinearBilateralConstraintInfo bp;
	bp.m_pivotA = vars.m_pivotAw; 
	bp.m_pivotB = vars.m_pivotBw;
	bp.m_constrainedDofW = hkVector4::getConstant<HK_QUADREAL_1000>();
	hk1dLinearBilateralConstraintBuildJacobian( bp, in, out );
	bp.m_constrainedDofW = hkVector4::getConstant<HK_QUADREAL_0100>();
	hk1dLinearBilateralConstraintBuildJacobian( bp, in, out );
	bp.m_constrainedDofW = hkVector4::getConstant<HK_QUADREAL_0010>();
	hk1dLinearBilateralConstraintBuildJacobian( bp, in, out );
	vars.m_currentResult += 3;
}

inline void hkpGenericConstraintData::constrainLinearW( hkArray<int>::iterator& currentCommand, hkArray<hkVector4>::iterator& currentData, const hkpGenericConstraintDataScheme& scheme, hkpGenericConstraintDataParameters& vars, const hkpConstraintQueryIn &in, hkpConstraintQueryOut &out ) const
{
	hkp1dLinearBilateralConstraintInfo bp;
	currentCommand++;
	bp.m_constrainedDofW = vars.m_linearBasisW.getColumn( *currentCommand );
	bp.m_pivotA = vars.m_pivotAw; 
	bp.m_pivotB = vars.m_pivotBw;
	hk1dLinearBilateralConstraintBuildJacobian( bp, in, out );
	vars.m_currentResult++;
}

static int hkGenericConstraintDataAxisOrder[5] = { 0, 1, 2, 0, 1 };  // use to avoid any conditionals or modulus

static hkReal HK_CALL calcDeltaAngleAroundAxis( int axis, const hkpGenericConstraintDataParameters& vars )
{
	const hkVector4& zeroErrorAxisAinW = vars.m_angularBasisAw.getColumn( hkGenericConstraintDataAxisOrder[axis+1] );
	const hkVector4& negZeroErrorAxisBinW = vars.m_angularBasisBw.getColumn( hkGenericConstraintDataAxisOrder[axis+2] );

	hkReal sinTheta = negZeroErrorAxisBinW.dot<3>( zeroErrorAxisAinW ).getReal();
	const hkVector4& cosCheck = vars.m_angularBasisBw.getColumn( hkGenericConstraintDataAxisOrder[axis+1] );
	hkReal cosTheta = cosCheck.dot<3>( zeroErrorAxisAinW ).getReal();
	
	//!me do I need atan2???  is it just to determine quadrant?  does small angle formula work here?
	return hkMath::atan2Approximation( sinTheta, cosTheta );
}



inline void hkpGenericConstraintData::constrainToAngularW( hkArray<int>::iterator& currentCommand, hkArray<hkVector4>::iterator& currentData, const hkpGenericConstraintDataScheme& scheme, hkpGenericConstraintDataParameters& vars, const hkpConstraintQueryIn &in, hkpConstraintQueryOut &out ) const
{
	currentCommand++;
	int axis = *currentCommand;

	hkp1dAngularBilateralConstraintInfo bp;
	bp.m_constrainedDofW = vars.m_angularBasisAw.getColumn( hkGenericConstraintDataAxisOrder[axis+1] );
	bp.m_zeroErrorAxisAinW = vars.m_angularBasisAw.getColumn( hkGenericConstraintDataAxisOrder[axis+2] );
	bp.m_perpZeroErrorAxisBinW = vars.m_angularBasisBw.getColumn( axis );
	hk1dAngularBilateralConstraintBuildJacobian( bp, in, out );	
	
	// keep non-constrained axis as the axis from B from which error is measured
	// so we need to negate one of the other vectors to keep AxB = C
	hkVector4 negated;
	negated.setNeg<4>( bp.m_constrainedDofW );
	bp.m_constrainedDofW = bp.m_zeroErrorAxisAinW;
	bp.m_zeroErrorAxisAinW = negated;
	hk1dAngularBilateralConstraintBuildJacobian( bp, in, out );	
	vars.m_currentResult += 2;

}



inline void hkpGenericConstraintData::constrainAllAngularW( hkArray<int>::iterator& currentCommand, hkArray<hkVector4>::iterator& currentData, const hkpGenericConstraintDataScheme& scheme, hkpGenericConstraintDataParameters& vars, const hkpConstraintQueryIn &in, hkpConstraintQueryOut &out ) const
{
	hkp1dAngularBilateralConstraintInfo bp;

	bp.m_zeroErrorAxisAinW = vars.m_angularBasisAw.getColumn<0>();
	bp.m_perpZeroErrorAxisBinW = vars.m_angularBasisBw.getColumn<1>();
	bp.m_constrainedDofW = vars.m_angularBasisAw.getColumn<2>();
	hk1dAngularBilateralConstraintBuildJacobian( bp, in, out );	
	
	hkp1dAngularBilateralConstraintInfo bp2;
	bp2.m_zeroErrorAxisAinW = vars.m_angularBasisAw.getColumn<1>();
	bp2.m_perpZeroErrorAxisBinW = vars.m_angularBasisBw.getColumn<2>();
	bp2.m_constrainedDofW = bp.m_zeroErrorAxisAinW;
	hk1dAngularBilateralConstraintBuildJacobian( bp2, in, out );

	bp2.m_perpZeroErrorAxisBinW = vars.m_angularBasisBw.getColumn<0>();
	bp2.m_constrainedDofW = bp2.m_zeroErrorAxisAinW;
	bp2.m_zeroErrorAxisAinW = bp.m_constrainedDofW;
	hk1dAngularBilateralConstraintBuildJacobian( bp2, in, out );
	vars.m_currentResult += 3;

}


// limits, friction, motors


inline void hkpGenericConstraintData::setLinearLimitW( hkArray<int>::iterator& currentCommand, hkArray<hkVector4>::iterator& currentData, const hkpGenericConstraintDataScheme& scheme, hkpGenericConstraintDataParameters& vars, const hkpConstraintQueryIn &in, hkpConstraintQueryOut &out ) const
{
	HK_ALIGN16( hkp1dLinearLimitInfo bp );
	currentCommand++;
	bp.m_constrainedDofW = vars.m_linearBasisW.getColumn( *currentCommand );
	bp.m_pivotA = vars.m_pivotAw; 
	bp.m_pivotB = vars.m_pivotBw;
	hkReal* HK_RESTRICT limit = (hkReal* HK_RESTRICT)currentData;
	currentData++;
	bp.m_min = limit[0];
	bp.m_max = limit[1];
	hk1dLinearLimitBuildJacobian( bp, in, out );
	vars.m_currentResult++;
}



inline void hkpGenericConstraintData::setAngularLimitW( hkArray<int>::iterator& currentCommand, hkArray<hkVector4>::iterator& currentData, const hkpGenericConstraintDataScheme& scheme, hkpGenericConstraintDataParameters& vars, const hkpConstraintQueryIn &in, hkpConstraintQueryOut &out ) const
{
	currentCommand++;
	int axis = *currentCommand;

	hkp1dAngularLimitInfo ali;
	ali.m_tau = hkReal(0.5f);
	ali.m_constrainedDofW = vars.m_angularBasisAw.getColumn( axis );

	ali.m_computedAngle = calcDeltaAngleAroundAxis( axis, vars );

	hkReal* HK_RESTRICT limit = (hkReal* HK_RESTRICT)currentData;
	currentData++;
	ali.m_min = limit[0];
	ali.m_max = limit[1];
	const hkReal limit_2 = limit[2];

	if(ali.m_computedAngle < hkReal(0) && limit_2 > hkReal(0))
	{
		if((limit_2 - ali.m_computedAngle) > HK_REAL_PI)
		{
			ali.m_computedAngle	= ali.m_computedAngle + hkReal(2) * HK_REAL_PI;
		}
	}

	if(ali.m_computedAngle > hkReal(0) && limit_2 < hkReal(0))
	{
		if((ali.m_computedAngle - limit_2) > HK_REAL_PI)
		{
			ali.m_computedAngle = ali.m_computedAngle - hkReal(2) * HK_REAL_PI;
		}
	}

	limit[2] = ali.m_computedAngle;
	
	
	hk1dAngularLimitBuildJacobian( ali, in, out );
	vars.m_currentResult++;
}


inline void hkpGenericConstraintData::setConeLimitW( hkArray<int>::iterator& currentCommand, hkArray<hkVector4>::iterator& currentData, const hkpGenericConstraintDataScheme& scheme, hkpGenericConstraintDataParameters& vars, const hkpConstraintQueryIn &in, hkpConstraintQueryOut &out ) const
{

	currentCommand++;
	int axis = *currentCommand;

	hkp1dAngularLimitInfo ali;
	ali.m_tau = hkReal(0.5f);
	hkVector4 twist = vars.m_angularBasisAw.getColumn( axis );
	hkVector4 twistRef = vars.m_angularBasisBw.getColumn( axis );

	ali.m_constrainedDofW.setCross( twist, twistRef );

	const hkSimdReal lenSqrd = ali.m_constrainedDofW.lengthSquared<3>();
	
	// we have a dead spot in the middle because we don't know which direction we are going there
	if( lenSqrd < hkSimdReal::getConstant<HK_QUADREAL_EPS>() )
	{
		currentData++;
		return;
	}
	ali.m_constrainedDofW.normalize<3>();

	// cos angle 
	{
		// avoid low precision dot product implementations
		hkVector4 m; m.setMul(twist, twistRef);
		ali.m_computedAngle = m.horizontalAdd<3>().getReal();
	}

	hkReal* HK_RESTRICT limit = (hkReal* HK_RESTRICT)currentData;
	currentData++;
	ali.m_min = limit[0];
	ali.m_max = limit[1];

	hk1dAngularLimitBuildJacobian( ali, in, out );
	vars.m_currentResult++;

}


inline void hkpGenericConstraintData::setTwistLimitW( hkArray<int>::iterator& currentCommand, hkArray<hkVector4>::iterator& currentData, const hkpGenericConstraintDataScheme& scheme, hkpGenericConstraintDataParameters& vars, const hkpConstraintQueryIn &in, hkpConstraintQueryOut &out ) const
{

	currentCommand++;
	int axis = *currentCommand;

	currentCommand++;
	int planeAxis = *currentCommand;

	hkp1dAngularLimitInfo ali;
	ali.m_tau = hkReal(0.5f);
	const hkVector4& twistAxisAinWorld = vars.m_angularBasisAw.getColumn( axis );
	const hkVector4& twistAxisBinWorld = vars.m_angularBasisBw.getColumn( axis );

	// twist

	// Calculate "twist" angle explicitly
	{
		const hkVector4& planeAxisAinWorld = vars.m_angularBasisAw.getColumn( planeAxis );
		const hkVector4& planeAxisBinWorld = vars.m_angularBasisBw.getColumn( planeAxis );

		hkInternalConstraintUtils_calcRelativeAngle( twistAxisAinWorld, twistAxisBinWorld,
			planeAxisAinWorld, planeAxisBinWorld, 
			ali.m_constrainedDofW, ali.m_computedAngle );
	}

	hkReal* HK_RESTRICT limit = (hkReal* HK_RESTRICT)currentData;
	currentData++;
	ali.m_min = limit[0];
	ali.m_max = limit[1];

	hk1dAngularLimitBuildJacobian( ali, in, out );
	vars.m_currentResult++;

}


inline void hkpGenericConstraintData::setAngularMotorW( hkArray<int>::iterator& currentCommand, hkArray<hkVector4>::iterator& currentData, hkpGenericConstraintDataScheme& scheme, hkpGenericConstraintDataParameters& vars, const hkpConstraintQueryIn &in, hkpConstraintQueryOut &out ) const
{

	// extract info from scheme, do some setup. 
	currentCommand++;

	hkReal* HK_RESTRICT motorData = (hkReal* HK_RESTRICT)currentData;
	currentData++;

	int motorIndex = hkMath::hkFloatToInt(motorData[0]);
	hkpConstraintMotor* motor = scheme.m_motors[ motorIndex ];

	// motor control and solver setup
	if( motor )
	{
		int axis = *currentCommand;
		hkReal currentPosition;

		// do the angle extraction and linearization 
		{
			hkReal lastAngle = motorData[1];
			hkReal rotations = motorData[2];

			const hkVector4& zeroErrorAxisAinW    = vars.m_angularBasisAw.getColumn( hkGenericConstraintDataAxisOrder[axis+1] );
			const hkVector4& negZeroErrorAxisBinW = vars.m_angularBasisBw.getColumn( hkGenericConstraintDataAxisOrder[axis+2] );

			hkSimdReal sinTheta = -negZeroErrorAxisBinW.dot<3>( zeroErrorAxisAinW );
			const hkVector4& cosCheck = vars.m_angularBasisBw.getColumn( hkGenericConstraintDataAxisOrder[axis+1] );
			hkSimdReal cosTheta = -cosCheck.dot<3>( zeroErrorAxisAinW );

			// shift continuous domain from [ -PI, PI ] to [ 0, 2PI ] 
			// sin(x+PI) = sin(x)cos(PI) + cos(x)sin(PI) = 0-sin(x)
			// sin(x+PI) = cos(x)cos(PI) - sin(x)sin(PI) = -cos(x)-0
			// discontinuities now at x = n*2PI
			hkSimdReal a = hkVector4Util::atan2Approximation(sinTheta,cosTheta) + hkSimdReal_Pi;
			a.store<1>(&currentPosition);

			// check to see if we pass zero
			if( currentPosition - lastAngle < -HK_REAL_PI )
			{
				rotations+=hkReal(1);
			}
			else if( currentPosition - lastAngle > HK_REAL_PI )
			{
				rotations-=hkReal(1);
			}

			lastAngle = currentPosition;

			// extra full rotations ( may be -ve count )
			currentPosition	+= hkReal(2)*HK_REAL_PI*rotations;

			motorData[1] = lastAngle;
			motorData[2] = rotations;

		}

		hkpSolverResults* solverResults = reinterpret_cast<hkpSolverResults*>( out.m_constraintRuntime.val() );

		const hkVector4& constrainedDofW = vars.m_angularBasisAw.getColumn( axis );

		hkpConstraintMotorInput motorIn;

		HK_SCHEMA_INIT(out.m_jacobianSchemas.val(), hkp2AngJacobian,  jac);
		hk1dAngularVelocityMotorBeginJacobian( constrainedDofW, in, jac, motorIn ); 

		motorIn.m_stepInfo = &in;
		motorIn.m_lastResults = solverResults[vars.m_currentResult];
//		motorIn.m_currentPosition = currentPosition;
//		motorIn.m_targetPosition = 0.0f;
// XXX this is broken
		motorIn.m_deltaTarget = hkReal(0);
		motorIn.m_positionError = - currentPosition;

		
		hkpConstraintMotorOutput motorOut;
		hkCalcMotorData(motor, &motorIn, &motorOut );
		
		hk1dAngularVelocityMotorCommitJacobian( motorOut, in, jac, out );
	}

	vars.m_currentResult++;
}


inline void hkpGenericConstraintData::setLinearMotorW( hkArray<int>::iterator& currentCommand, hkArray<hkVector4>::iterator& currentData, hkpGenericConstraintDataScheme& scheme, hkpGenericConstraintDataParameters& vars, const hkpConstraintQueryIn &in, hkpConstraintQueryOut &out ) const
{

	// extract info from scheme, do some setup. 
	currentCommand++;

	hkReal* HK_RESTRICT motorData = (hkReal* HK_RESTRICT)currentData;
	currentData++;

	int motorIndex = hkMath::hkFloatToInt(motorData[0]);
	hkpConstraintMotor* motor = scheme.m_motors[ motorIndex ];

	// motor control and solver setup
	if( motor )
	{
		hkpSolverResults* solverResults = reinterpret_cast<hkpSolverResults*>( out.m_constraintRuntime.val() );
		int axis = *currentCommand;
		const hkVector4 constrainedDofW = vars.m_linearBasisW.getColumn( axis );

		// do the position calculation 
		hkReal currentPosition;
		{
			hkVector4 diff; diff.setSub( vars.m_pivotAw, vars.m_pivotBw );
			currentPosition = diff.dot<3>( constrainedDofW ).getReal();
		}

		hkpConstraintMotorInput motorIn;

		HK_SCHEMA_INIT(out.m_jacobianSchemas.val(), hkp1Lin2AngJacobian,  jac);
		// I'll use the pivot on body B.  So it is as if the reference body ( B ) is where the motor is anchored
		hk1dLinearVelocityMotorBeginJacobian( constrainedDofW, vars.m_pivotBw, in, jac, motorIn ); 

		motorIn.m_lastResults = solverResults[vars.m_currentResult];
		//motorIn.m_currentPosition = currentPosition;
		//motorIn.m_targetPosition = 0.0f;
// XXX this is broken
		motorIn.m_deltaTarget = hkReal(0);
		motorIn.m_positionError = - currentPosition;

		motorIn.m_stepInfo = &in;
		
		hkpConstraintMotorOutput motorOut;
		hkCalcMotorData( motor, &motorIn, &motorOut );
		
		hk1dLinearVelocityMotorCommitJacobian( motorOut, in, jac, out );
	}

	vars.m_currentResult++;
}


inline void hkpGenericConstraintData::setAngularFrictionW( hkArray<int>::iterator& currentCommand, hkArray<hkVector4>::iterator& currentData, hkpGenericConstraintDataScheme& scheme, hkpGenericConstraintDataParameters& vars, const hkpConstraintQueryIn &in, hkpConstraintQueryOut &out ) const
{
	hkpSolverResults* solverResults = reinterpret_cast<hkpSolverResults*>( out.m_constraintRuntime.val() );

	currentCommand++;
	int axis = *currentCommand;
	hkp1dAngularFrictionInfo afi;
	afi.m_constrainedDofW = &vars.m_angularBasisAw.getColumn( axis );
	hkReal* HK_RESTRICT coef = (hkReal* HK_RESTRICT)currentData;
	currentData++;
	afi.m_maxFrictionTorque = coef[0];

	afi.m_numFriction = 1;
	afi.m_lastSolverResults = &(solverResults[ vars.m_currentResult ]);

	hk1dAngularFrictionBuildJacobian( afi, in, out ); 
	vars.m_currentResult++;
}


void hkpGenericConstraintData::setLinearFrictionW( hkArray<int>::iterator& currentCommand, hkArray<hkVector4>::iterator& currentData, hkpGenericConstraintDataScheme& scheme, hkpGenericConstraintDataParameters& vars, const hkpConstraintQueryIn &in, hkpConstraintQueryOut &out ) const
{
	hkpSolverResults* solverResults = reinterpret_cast<hkpSolverResults*>( out.m_constraintRuntime.val() );

	hkp1dLinearFrictionInfo bp;
	currentCommand++;
	bp.m_constrainedDofW = vars.m_linearBasisW.getColumn( *currentCommand );

	bp.m_pivot.setSub( vars.m_pivotAw, vars.m_pivotBw ); 
	hkReal* HK_RESTRICT coef = (hkReal* HK_RESTRICT)currentData;
	currentData++;
	bp.m_maxFrictionForce = coef[0];
	bp.m_lastSolverResults = &(solverResults[ vars.m_currentResult ]);
	hk1dLinearFrictionBuildJacobian( bp, in, out );
	vars.m_currentResult++;
}

//
// end commands
//

void hkpGenericConstraintData::hatchScheme( hkpGenericConstraintDataScheme* scheme, const hkpConstraintQueryIn &inOrig, hkpConstraintQueryOut &out )
{
	hkpConstraintQueryIn in = inOrig;

	hkpGenericConstraintDataParameters vars;
	vars.m_currentResult = 0;

	vars.m_rbA = in.m_transformA;
	vars.m_rbB = in.m_transformB;

	hkArray<int>::iterator currentCommand = scheme->m_commands.begin(); 
	hkArray<hkVector4>::iterator currentData = scheme->m_data.begin();
	hkArray<hkpConstraintModifier *>::iterator currentModifier = scheme->m_modifiers.begin();

	while( 1 )
	{
		switch( *currentCommand )
		{
			// linear constraints

			case hkpGenericConstraintDataScheme::e_setLinearDofA :
			{
				currentCommand++;
				hkVector4& col = vars.m_linearBasisW.getColumn( *currentCommand );
				col._setRotatedDir( vars.m_rbA->getRotation(), *currentData );
				currentData++;
				break;
			}
			case hkpGenericConstraintDataScheme::e_setLinearDofB :
			{
				currentCommand++;
				hkVector4& col = vars.m_linearBasisW.getColumn( *currentCommand );
				col._setRotatedDir( vars.m_rbB->getRotation(), *currentData );
				currentData++;
				break;
			}

			case hkpGenericConstraintDataScheme::e_setLinearDofW :
			{
				currentCommand++;
				hkVector4& col = vars.m_linearBasisW.getColumn( *currentCommand );
				col = *currentData;
				currentData++;
				break;
			}

			case hkpGenericConstraintDataScheme::e_constrainLinearW :
			{
				constrainLinearW( currentCommand, currentData, *scheme, vars, in, out );
				break;
			}

			case hkpGenericConstraintDataScheme::e_constrainAllLinearW :
			{
				constrainAllLinearW( currentCommand, currentData, *scheme, vars, in, out );
				break;
			}

			// angular constraints

			case hkpGenericConstraintDataScheme::e_setAngularBasisA :
			{
				hkRotation& rot = reinterpret_cast<hkRotation&>( *currentData );
				currentData += 3;
				vars.m_angularBasisAw.setMul( vars.m_rbA->getRotation(), rot );  
				break;
			}

			case hkpGenericConstraintDataScheme::e_setAngularBasisB :
			{
				hkRotation& rot = reinterpret_cast<hkRotation&>( *currentData );
				currentData += 3;
				vars.m_angularBasisBw.setMul( vars.m_rbB->getRotation(), rot );  
				break;
			}

			case hkpGenericConstraintDataScheme::e_setAngularBasisAidentity :
			{
				vars.m_angularBasisAw = vars.m_rbA->getRotation();
				break;
			}

			case hkpGenericConstraintDataScheme::e_setAngularBasisBidentity :
			{
				vars.m_angularBasisBw = vars.m_rbB->getRotation();
				break;
			}

			case hkpGenericConstraintDataScheme::e_constrainToAngularW :
			{
				constrainToAngularW( currentCommand, currentData, *scheme, vars, in, out );
				break;
			}

			case hkpGenericConstraintDataScheme::e_constrainAllAngularW :
			{
				constrainAllAngularW( currentCommand, currentData, *scheme, vars, in, out );
				break;
			}

			// limits, motors, friction
			case hkpGenericConstraintDataScheme::e_setLinearLimit :
			{
				setLinearLimitW( currentCommand, currentData, *scheme, vars, in, out );
				break;
			}

			case hkpGenericConstraintDataScheme::e_setAngularLimit :
			{
				setAngularLimitW( currentCommand, currentData, *scheme, vars, in, out );
				break;
			}

			case hkpGenericConstraintDataScheme::e_setConeLimit :
			{
				setConeLimitW( currentCommand, currentData, *scheme, vars, in, out );
				break;
			}

			case hkpGenericConstraintDataScheme::e_setTwistLimit :
			{
				setTwistLimitW( currentCommand, currentData, *scheme, vars, in, out );
				break;
			}

			case hkpGenericConstraintDataScheme::e_setAngularMotor :
			{
				setAngularMotorW( currentCommand, currentData, *scheme, vars, in, out );
				break;
			}
			
			case hkpGenericConstraintDataScheme::e_setLinearMotor :
			{
				setLinearMotorW( currentCommand, currentData, *scheme, vars, in, out );
				break;
			}

			case hkpGenericConstraintDataScheme::e_setAngularFriction :
			{
				setAngularFrictionW( currentCommand, currentData, *scheme, vars, in, out );
				break;
			}

			case hkpGenericConstraintDataScheme::e_setLinearFriction :
			{
				setLinearFrictionW( currentCommand, currentData, *scheme, vars, in, out );
				break;
			}

			// pivot point

			case hkpGenericConstraintDataScheme::e_setPivotA :
			{
				vars.m_pivotAw.setTransformedPos( *vars.m_rbA, *currentData );
				currentData++;
				break;
			}

			case hkpGenericConstraintDataScheme::e_setPivotB :
			{
				vars.m_pivotBw.setTransformedPos( *vars.m_rbB, *currentData );
				currentData++;
				break;
			}

			case hkpGenericConstraintDataScheme::e_setStrength :
			{
				hkReal* HK_RESTRICT strength = (hkReal* HK_RESTRICT)currentData;
				//hkSetTauAndDamping( tau, damping, out );
				currentData++;
				in.m_virtMassFactor = in.m_virtMassFactor * strength[0];
				break;
			}

			case hkpGenericConstraintDataScheme::e_restoreStrengh:
			{
				//hkRestoreTauAndDamping( out );			
				in.m_virtMassFactor		= inOrig.m_virtMassFactor;
				break;
			}

			case hkpGenericConstraintDataScheme::e_doConstraintModifier :
			{
				currentCommand++;
				(*currentModifier)->modify( vars, *currentCommand );
				currentModifier++;
				break;
			}

			case hkpGenericConstraintDataScheme::e_endScheme :
			{
				return;
			}

			default: 
			{
				HK_ASSERT2(0x1b6cd4a1,  0, "generic constraint: unknown opcode" );
				break;
			}

		}

		// next command
		currentCommand++;

	}

}

hkpGenericConstraintDataScheme* hkpGenericConstraintData::getScheme()
{
	return &m_scheme;
}

hkBool hkpGenericConstraintData::isValid() const
{
	// Not implemented
	return true;
}

int hkpGenericConstraintData::getType() const
{
	return hkpConstraintData::CONSTRAINT_TYPE_GENERIC;
}

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
