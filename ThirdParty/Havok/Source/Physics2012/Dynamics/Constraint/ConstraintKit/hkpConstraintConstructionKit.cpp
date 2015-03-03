/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics2012/Dynamics/hkpDynamics.h>
#include <Physics2012/Dynamics/Constraint/ConstraintKit/hkpConstraintConstructionKit.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics/Constraint/Motor/hkpConstraintMotor.h>

// set up construction kit

void hkpConstraintConstructionKit::begin( hkpGenericConstraintData* constraint )
{
	HK_ASSERT2(0x54566fc5,  hkpGenericConstraintDataScheme::e_numCommands <= 0xFF, "constraint construction kit: too many opcodes" );
	m_scheme = constraint->getScheme();
	m_constraint = constraint;
	m_stiffnessReference = 0;
	m_dampingReference = 0;

	m_linearDofSpecifiedA[0] = false;
	m_linearDofSpecifiedA[1] = false;
	m_linearDofSpecifiedA[2] = false;
	m_linearDofSpecifiedB[0] = false;
	m_linearDofSpecifiedB[1] = false;
	m_linearDofSpecifiedB[2] = false;
	m_pivotSpecifiedA = false;
	m_pivotSpecifiedB = false;

	m_angularBasisSpecifiedA = false;
	m_angularBasisSpecifiedB = false;
}

// linear constraint

hkpGenericConstraintData::hkpParameterIndex hkpConstraintConstructionKit::setLinearDofA( const hkVector4& dof, int axis )
{
	m_scheme->m_commands.pushBack( hkpGenericConstraintDataScheme::e_setLinearDofA );
	m_scheme->m_commands.pushBack( axis );
	const int dataIndex = m_scheme->m_data.getSize(); 
	m_scheme->m_data.pushBack( dof );
	m_linearDofSpecifiedA[axis] = true;
	return dataIndex;
}

hkpGenericConstraintData::hkpParameterIndex hkpConstraintConstructionKit::setLinearDofB( const hkVector4& dof, int axis )
{
	m_scheme->m_commands.pushBack( hkpGenericConstraintDataScheme::e_setLinearDofB );
	m_scheme->m_commands.pushBack( axis );
	const int dataIndex = m_scheme->m_data.getSize(); 
	m_scheme->m_data.pushBack( dof );
	m_linearDofSpecifiedB[axis] = true;
	return dataIndex;
}

hkpGenericConstraintData::hkpParameterIndex hkpConstraintConstructionKit::setLinearDofWorld( const hkVector4& dof, int axis )
{
	m_scheme->m_commands.pushBack( hkpGenericConstraintDataScheme::e_setLinearDofW );
	m_scheme->m_commands.pushBack( axis );
	const int dataIndex = m_scheme->m_data.getSize();
	m_scheme->m_data.pushBack( dof );
	m_linearDofSpecifiedA[axis] = true;
	m_linearDofSpecifiedB[axis] = true;
	return dataIndex;
}

void hkpConstraintConstructionKit::constrainLinearDof( int axis )
{
	HK_ASSERT2(0x1583f683, (m_pivotSpecifiedA ), "Cannot constrain Linear DOF: Pivot not yet specified in A space");
	HK_ASSERT2(0x2404229b, (m_pivotSpecifiedB ), "Cannot constrain Linear DOF: Pivot not yet specified in B space");
	HK_ASSERT2(0x6bf36a59, ((m_linearDofSpecifiedB[axis] ) || (m_linearDofSpecifiedA[axis] )), "Cannot constrain Linear DOF: Axis not yet specified");
	m_scheme->m_commands.pushBack( hkpGenericConstraintDataScheme::e_constrainLinearW );
	m_scheme->m_commands.pushBack( axis );
	m_scheme->m_info.add<hkpJacobianSchemaInfo::Bilateral1D>();
}

void hkpConstraintConstructionKit::constrainAllLinearDof()
{
	HK_ASSERT2(0x57a68a69, (m_pivotSpecifiedA ), "Cannot constrain All Linear DOF: Pivot not yet specified in A space");
	HK_ASSERT2(0x6311d307, (m_pivotSpecifiedB ), "Cannot constrain All Linear DOF: Pivot not yet specified in B space");
	m_scheme->m_commands.pushBack( hkpGenericConstraintDataScheme::e_constrainAllLinearW );
	m_scheme->m_info.addMultiple<hkpJacobianSchemaInfo::Bilateral1D>(3);
}

// angular constraint

hkpGenericConstraintData::hkpParameterIndex hkpConstraintConstructionKit::setAngularBasisA( const hkMatrix3& dofBasis )
{
	m_scheme->m_commands.pushBack( hkpGenericConstraintDataScheme::e_setAngularBasisA );
	const int dataIndex = m_scheme->m_data.getSize(); 
	m_scheme->m_data.pushBack( dofBasis.getColumn<0>() );
	m_scheme->m_data.pushBack( dofBasis.getColumn<1>() );
	m_scheme->m_data.pushBack( dofBasis.getColumn<2>() );
	m_angularBasisSpecifiedA = true;
	return dataIndex;
}

hkpGenericConstraintData::hkpParameterIndex hkpConstraintConstructionKit::setAngularBasisB( const hkMatrix3& dofBasis )
{
	m_scheme->m_commands.pushBack( hkpGenericConstraintDataScheme::e_setAngularBasisB );
	const int dataIndex = m_scheme->m_data.getSize(); 
	m_scheme->m_data.pushBack( dofBasis.getColumn<0>() );
	m_scheme->m_data.pushBack( dofBasis.getColumn<1>() );
	m_scheme->m_data.pushBack( dofBasis.getColumn<2>() );
	m_angularBasisSpecifiedB = true;
	return dataIndex;
}	

void hkpConstraintConstructionKit::setAngularBasisABodyFrame()
{
	m_scheme->m_commands.pushBack( hkpGenericConstraintDataScheme::e_setAngularBasisAidentity );
	m_angularBasisSpecifiedA = true;
}

void hkpConstraintConstructionKit::setAngularBasisBBodyFrame()
{
	m_scheme->m_commands.pushBack( hkpGenericConstraintDataScheme::e_setAngularBasisBidentity );
	m_angularBasisSpecifiedB = true;
}


// constrain object to only rotate around degree of freedom specified.
void hkpConstraintConstructionKit::constrainToAngularDof( int axis )
{
	HK_ASSERT2(0x530d4b4d, (m_angularBasisSpecifiedA ), "Cannot constrain Angular DOF: Basis not yet specified in A space");
	HK_ASSERT2(0x21a81142, (m_angularBasisSpecifiedB ), "Cannot constrain Angular DOF: Basis not yet specified in B space");
	m_scheme->m_commands.pushBack( hkpGenericConstraintDataScheme::e_constrainToAngularW );
	m_scheme->m_commands.pushBack( axis );
	m_scheme->m_info.addMultiple<hkpJacobianSchemaInfo::Bilateral1D>(2);
}

// constrain object so it can not rotate relative to second object
void hkpConstraintConstructionKit::constrainAllAngularDof()
{
	HK_ASSERT2(0x66d851d1, (m_angularBasisSpecifiedA ), "Cannot constrain Angular DOF: Basis not yet specified in A space");
	HK_ASSERT2(0x34708bb6, (m_angularBasisSpecifiedB ), "Cannot constrain Angular DOF: Basis not yet specified in B space");
	m_scheme->m_commands.pushBack( hkpGenericConstraintDataScheme::e_constrainAllAngularW );
	m_scheme->m_info.addMultiple<hkpJacobianSchemaInfo::Bilateral1D>(3);
}

// pivot point

hkpGenericConstraintData::hkpParameterIndex hkpConstraintConstructionKit::setPivotA( const hkVector4& pivot )
{
	m_scheme->m_commands.pushBack( hkpGenericConstraintDataScheme::e_setPivotA );
	const int dataIndex = m_scheme->m_data.getSize();
	m_scheme->m_data.pushBack( pivot );
	m_pivotSpecifiedA = true;
	return dataIndex;
}

hkpGenericConstraintData::hkpParameterIndex hkpConstraintConstructionKit::setPivotB( const hkVector4& pivot )
{
	m_scheme->m_commands.pushBack( hkpGenericConstraintDataScheme::e_setPivotB );
	const int dataIndex = m_scheme->m_data.getSize();
	m_scheme->m_data.pushBack( pivot );
	m_pivotSpecifiedB = true;
	return dataIndex;
}

hkpGenericConstraintData::hkpParameterIndex hkpConstraintConstructionKit::setPivotsHelper( hkpRigidBody* bodyA, hkpRigidBody* bodyB, const hkVector4& pivot )
{

	hkVector4 attachA, attachB;
	attachA._setTransformedInversePos( bodyA->getTransform(), pivot );
	attachB._setTransformedInversePos( bodyB->getTransform(), pivot );
	
	int dataIndex = setPivotA( attachA );
	setPivotB( attachB );

	m_pivotSpecifiedA = true;
 	m_pivotSpecifiedB = true;
 
	return dataIndex;
}

// limits, friction, motors

hkpGenericConstraintData::hkpParameterIndex hkpConstraintConstructionKit::setLinearLimit( int axis, hkReal min, hkReal max )
{
	HK_ASSERT2(0x55ac42ef, (m_pivotSpecifiedA ), "Cannot set limits for Linear DOF: Pivot not yet specified in A space");
	HK_ASSERT2(0x330b7c7e, (m_pivotSpecifiedB ), "Cannot set limits for Linear DOF: Pivot not yet specified in B space");
	HK_ASSERT2(0x6533be94, ((m_linearDofSpecifiedB[axis] ) || (m_linearDofSpecifiedA[axis] )), "Cannot constrain Linear DOF: Axis not yet specified");
	m_scheme->m_commands.pushBack( hkpGenericConstraintDataScheme::e_setLinearLimit );
	m_scheme->m_commands.pushBack( hkUchar(axis) );
	hkVector4 limit; limit.set( min, max, hkReal(0), hkReal(0) );
	const int dataIndex = m_scheme->m_data.getSize();
	m_scheme->m_data.pushBack( limit );
	m_scheme->m_info.add<hkpJacobianSchemaInfo::LinearLimits1D>();
	return dataIndex;
}

hkpGenericConstraintData::hkpParameterIndex hkpConstraintConstructionKit::setAngularLimit( int axis, hkReal min, hkReal max )
{
	HK_ASSERT2(0x1396cee8, (m_angularBasisSpecifiedA ), "Cannot set limits for Angular DOF: Basis not yet specified in A space");
	HK_ASSERT2(0x7c1a15e6, (m_angularBasisSpecifiedB ), "Cannot set limits for Angular DOF: Basis not yet specified in B space");
	m_scheme->m_commands.pushBack( hkpGenericConstraintDataScheme::e_setAngularLimit );
	m_scheme->m_commands.pushBack( hkUchar(axis) );
	hkVector4 limit; limit.set( min, max, hkReal(0), hkReal(0) );  //!me change to 
	//!me class hkLimitParam : public class hkVector4{ hkLimitVector( min,max) getMin() setMin()
	const int dataIndex = m_scheme->m_data.getSize();
	m_scheme->m_data.pushBack( limit );
	m_scheme->m_info.add<hkpJacobianSchemaInfo::AngularLimits1D>();
	return dataIndex;
}


hkpGenericConstraintData::hkpParameterIndex hkpConstraintConstructionKit::setConeLimit( int axis, hkReal angle )
{
	HK_ASSERT2(0x6ff88eba, (m_angularBasisSpecifiedA ), "Cannot set limits for Angular DOF: Basis not yet specified in A space");
	HK_ASSERT2(0x3f73c278, (m_angularBasisSpecifiedB ), "Cannot set limits for Angular DOF: Basis not yet specified in B space");
	m_scheme->m_commands.pushBack( hkpGenericConstraintDataScheme::e_setConeLimit );
	m_scheme->m_commands.pushBack( hkUchar(axis) );
	hkVector4 limit; limit.set( hkMath::cos(angle), hkReal(100), hkReal(0), hkReal(0) );  //!me change to 

	const int dataIndex = m_scheme->m_data.getSize();
	m_scheme->m_data.pushBack( limit );
	m_scheme->m_info.add<hkpJacobianSchemaInfo::AngularLimits1D>();
	return dataIndex;
}


hkpGenericConstraintData::hkpParameterIndex hkpConstraintConstructionKit::setTwistLimit( int twistAxis, int planeAxis, hkReal min, hkReal max )
{

	HK_ASSERT2(0x7d13f7fc, (m_angularBasisSpecifiedA ), "Cannot set limits for Angular DOF: Basis not yet specified in A space");
	HK_ASSERT2(0x2adc9420, (m_angularBasisSpecifiedB ), "Cannot set limits for Angular DOF: Basis not yet specified in B space");
	m_scheme->m_commands.pushBack( hkpGenericConstraintDataScheme::e_setTwistLimit );
	m_scheme->m_commands.pushBack( hkUchar(twistAxis) );
	m_scheme->m_commands.pushBack( hkUchar(planeAxis) );
	hkVector4 limit; limit.set( hkMath::sin(min), hkMath::sin(max), hkReal(0), hkReal(0) );  //!me change to 

	const int dataIndex = m_scheme->m_data.getSize();
	m_scheme->m_data.pushBack( limit );
	m_scheme->m_info.add<hkpJacobianSchemaInfo::AngularLimits1D>();
	return dataIndex;

}


hkpGenericConstraintData::hkpParameterIndex hkpConstraintConstructionKit::setAngularMotor( int axis, hkpConstraintMotor* motor )
{
	m_scheme->m_commands.pushBack( hkpGenericConstraintDataScheme::e_setAngularMotor );
	m_scheme->m_commands.pushBack( hkUchar(axis) );

	const int dataIndex = m_scheme->m_data.getSize();
	hkSimdReal numMotors; numMotors.setFromInt32(m_scheme->m_motors.getSize());
	hkSimdReal zero; zero.setZero();
	hkVector4 motorData; motorData.set( numMotors, zero, zero, zero );
	
	HK_ASSERT2(0x647f8842,  motor, "can't add a null motor with constraint kit" );

	motor->addReference();

	m_scheme->m_data.pushBack( motorData ); 
	m_scheme->m_motors.pushBack( motor );

	m_scheme->m_info.add<hkpJacobianSchemaInfo::AngularMotor1D>();
	return dataIndex;
}

hkpGenericConstraintData::hkpParameterIndex hkpConstraintConstructionKit::setLinearMotor( int axis, hkpConstraintMotor* motor )
{
	m_scheme->m_commands.pushBack( hkpGenericConstraintDataScheme::e_setLinearMotor );
	m_scheme->m_commands.pushBack( hkUchar(axis) );

	const int dataIndex = m_scheme->m_data.getSize();
	hkSimdReal numMotors; numMotors.setFromInt32(m_scheme->m_motors.getSize());
	hkSimdReal zero; zero.setZero();
	hkVector4 motorData; motorData.set( numMotors, zero, zero, zero );
	
	HK_ASSERT2(0x6838c050,  motor, "can't add a null motor with constraint kit" );

	motor->addReference();

	m_scheme->m_data.pushBack( motorData ); 
	m_scheme->m_motors.pushBack( motor );

	m_scheme->m_info.add<hkpJacobianSchemaInfo::LinearMotor1D>();
	return dataIndex;
}


hkpGenericConstraintData::hkpParameterIndex hkpConstraintConstructionKit::setAngularFriction( int axis, hkReal maxImpulse )
{
	HK_ASSERT2(0x1fa2673c, (m_angularBasisSpecifiedA ), "Cannot set friction for Angular DOF: Basis not yet specified in A space");
	HK_ASSERT2(0x526a3b98, (m_angularBasisSpecifiedB ), "Cannot set friction for Angular DOF: Basis not yet specified in B space");
	m_scheme->m_commands.pushBack( hkpGenericConstraintDataScheme::e_setAngularFriction );
	m_scheme->m_commands.pushBack( hkUchar(axis) );
	hkVector4 coef; coef.set( maxImpulse, hkReal(0), hkReal(0), hkReal(0) );
	const int dataIndex = m_scheme->m_data.getSize();
	m_scheme->m_data.pushBack( coef );
	m_scheme->m_info.add<hkpJacobianSchemaInfo::AngularFriction1D>();
	return dataIndex;
}

hkpGenericConstraintData::hkpParameterIndex hkpConstraintConstructionKit::setLinearFriction( int axis, hkReal maxImpulse )
{
	HK_ASSERT2(0x3e9305ff, (m_pivotSpecifiedA ), "Cannot set friction for Linear DOF: Pivot not yet specified in A space");
	HK_ASSERT2(0x32b39556, (m_pivotSpecifiedB ), "Cannot set friction for Linear DOF: Pivot not yet specified in B space");
	HK_ASSERT2(0x2e60e5f2, (m_linearDofSpecifiedA[axis] || m_linearDofSpecifiedB[axis] ), "Cannot set limits for Linear DOF: Axis not yet specified in neither A or B space");
	m_scheme->m_commands.pushBack( hkpGenericConstraintDataScheme::e_setLinearFriction );
	m_scheme->m_commands.pushBack( hkUchar(axis) );
	hkVector4 coef; coef.set( maxImpulse, hkReal(0), hkReal(0), hkReal(0) );  //!me change to hkFrictionParam
	const int dataIndex = m_scheme->m_data.getSize();
	m_scheme->m_data.pushBack( coef );
	m_scheme->m_info.add<hkpJacobianSchemaInfo::Friction1D>();
	return dataIndex;
}

// tau, damping


	/// Sets the stiffness of the subsequent constraints.  remember to restore it later
hkpGenericConstraintData::hkpParameterIndex hkpConstraintConstructionKit::setStrength( hkReal strength )
{
	m_stiffnessReference++;
	m_scheme->m_commands.pushBack( hkpGenericConstraintDataScheme::e_setStrength );
	hkVector4 tauAndDampingV; tauAndDampingV.set( strength, hkReal(0), hkReal(0), hkReal(0) );
	const int dataIndex = m_scheme->m_data.getSize();
	m_scheme->m_data.pushBack( tauAndDampingV );
	return dataIndex;
}


	/// restore the stiffness back to solver defaults
void hkpConstraintConstructionKit::restoreStrength()
{
	m_stiffnessReference--;
	m_scheme->m_commands.pushBack( hkpGenericConstraintDataScheme::e_restoreStrengh );
}

//modifiers

void hkpConstraintConstructionKit::addConstraintModifierCallback( hkpConstraintModifier *cm, int userData )
{
	m_scheme->m_commands.pushBack( hkpGenericConstraintDataScheme::e_doConstraintModifier );
	m_scheme->m_modifiers.pushBack( cm );
	m_scheme->m_commands.pushBack( userData ); 
}

// commands

void hkpConstraintConstructionKit::end()
{
	m_scheme->m_commands.pushBack( hkpGenericConstraintDataScheme::e_endScheme );
	HK_ASSERT(0x61bd39b8,  m_stiffnessReference == 0 );
	HK_ASSERT(0x38259980,  m_dampingReference == 0 );
}


void HK_CALL hkpConstraintConstructionKit::computeConstraintInfo(const hkArray<int>& commands, hkpConstraintInfo& info)
{
	const int* currentCommand = commands.begin();
	const int* const commandEnd = commands.end();

	while(currentCommand < commandEnd)
	{
		switch( *currentCommand )
		{
			// linear constraints

		case hkpGenericConstraintDataScheme::e_setLinearDofA :
		case hkpGenericConstraintDataScheme::e_setLinearDofB :
		case hkpGenericConstraintDataScheme::e_setLinearDofW :
			{
				currentCommand++;
				break;
			}

		case hkpGenericConstraintDataScheme::e_constrainLinearW :
			{
				currentCommand++;
				info.add<hkpJacobianSchemaInfo::Bilateral1D>();
				break;
			}

		case hkpGenericConstraintDataScheme::e_constrainAllLinearW :
			{
				info.addMultiple<hkpJacobianSchemaInfo::Bilateral1D>(3);
				break;
			}

			// angular constraints

		case hkpGenericConstraintDataScheme::e_setAngularBasisA :
		case hkpGenericConstraintDataScheme::e_setAngularBasisB :
		case hkpGenericConstraintDataScheme::e_setAngularBasisAidentity :
		case hkpGenericConstraintDataScheme::e_setAngularBasisBidentity :
			{
				break;
			}

		case hkpGenericConstraintDataScheme::e_constrainToAngularW :
			{
				currentCommand++;
				info.addMultiple<hkpJacobianSchemaInfo::Bilateral1D>(2);
				break;
			}

		case hkpGenericConstraintDataScheme::e_constrainAllAngularW :
			{
				info.addMultiple<hkpJacobianSchemaInfo::Bilateral1D>(3);
				break;
			}

			// pivot point

		case hkpGenericConstraintDataScheme::e_setPivotA :
		case hkpGenericConstraintDataScheme::e_setPivotB :
			{
				break;
			}


			// limits, motors, friction
		case hkpGenericConstraintDataScheme::e_setLinearLimit :
			{
				currentCommand++;
				info.add<hkpJacobianSchemaInfo::LinearLimits1D>();
				break;
			}

		case hkpGenericConstraintDataScheme::e_setAngularLimit :
			{
				currentCommand++;
				info.add<hkpJacobianSchemaInfo::AngularLimits1D>();
				break;
			}

		case hkpGenericConstraintDataScheme::e_setConeLimit :
			{
				currentCommand++;
				info.add<hkpJacobianSchemaInfo::AngularLimits1D>();
				break;
			}

		case hkpGenericConstraintDataScheme::e_setTwistLimit :
			{
				currentCommand++;
				currentCommand++;
				info.add<hkpJacobianSchemaInfo::AngularLimits1D>();
				break;
			}

		case hkpGenericConstraintDataScheme::e_setAngularMotor :
			{
				currentCommand++;
				info.add<hkpJacobianSchemaInfo::AngularMotor1D>();
				break;
			}

		case hkpGenericConstraintDataScheme::e_setLinearMotor :
			{
				currentCommand++;
				info.add<hkpJacobianSchemaInfo::LinearMotor1D>();
				break;
			}

		case hkpGenericConstraintDataScheme::e_setAngularFriction :
			{
				currentCommand++;
				info.add<hkpJacobianSchemaInfo::AngularFriction1D>();
				break;
			}

		case hkpGenericConstraintDataScheme::e_setLinearFriction :
			{
				currentCommand++;
				info.add<hkpJacobianSchemaInfo::Friction1D>();
				break;
			}



		case hkpGenericConstraintDataScheme::e_setStrength :
		case hkpGenericConstraintDataScheme::e_restoreStrengh:
			{
				break;
			}

		case hkpGenericConstraintDataScheme::e_doConstraintModifier :
			{
				currentCommand++;
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
