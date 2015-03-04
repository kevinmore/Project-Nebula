/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Dynamics/Motion/hknpMotionCinfo.h>

#include <Common/GeometryUtilities/Inertia/hkInertiaTensorComputer.h>

namespace
{
	static HK_FORCE_INLINE void getOrBuildMassProperties( const hknpShape* shape, hkMassProperties& massPropertiesOut )
	{
		if( shape->getMassProperties( massPropertiesOut ) == HK_FAILURE )
		{
			// Build low quality mass properties if none were present
			//HK_WARN( 0x3eba1f22, "A shape has no mass properties. Approximating them instead." );
			hknpShape::MassConfig massConfig;
			massConfig.m_quality = hknpShape::MassConfig::QUALITY_LOW;
			hkDiagonalizedMassProperties dmp;
			shape->buildMassProperties( massConfig, dmp );
			dmp.unpack( &massPropertiesOut );
		}
	}

	static HK_FORCE_INLINE void getCombinedMassProperties(
		const hknpBodyCinfo* bodyInfos, int numCinfos, const hkTransform& refTransform,
		hkMassProperties& massPropertiesOut )
	{
		hkArray<hkMassElement> massElements;
		massElements.setSize(numCinfos);
		for( int i = 0 ; i < numCinfos; ++i )
		{
			HK_ASSERT( 0xf034df76, bodyInfos[i].m_shape );
			getOrBuildMassProperties( bodyInfos[i].m_shape, massElements[i].m_properties );
			hkTransform bodyTransform;
			bodyTransform.set( bodyInfos[i].m_orientation, bodyInfos[i].m_position );
			massElements[i].m_transform.setMulMulInverse( bodyTransform, refTransform );
		}
		hkInertiaTensorComputer::combineMassProperties( massElements, massPropertiesOut );
	}
}

hknpMotionCinfo::hknpMotionCinfo()
{
	m_motionPropertiesId = hknpMotionPropertiesId::invalid();
	m_enableDeactivation = true;
	m_inverseMass = 1.0f;
	m_massFactor = 1.0f;
	m_inverseInertiaLocal = hkVector4::getConstant(HK_QUADREAL_1);
	m_maxLinearAccelerationDistancePerStep = HK_REAL_HIGH;
	m_maxRotationToPreventTunneling = HK_REAL_HIGH;
	m_centerOfMassWorld.setZero();
	m_orientation.setIdentity();
	m_linearVelocity.setZero();
	m_angularVelocity.setZero();
}

void hknpMotionCinfo::setOrientationAndInertiaLocal( hkQuaternionParameter bodyOrientation, const hkMatrix3& inertia )
{
	hkRotation eigenVec;
	hkVector4  eigenVal;
	inertia.diagonalizeSymmetricApproximation( eigenVec, eigenVal, 10 );

	hkVector4 invInertiaLocal;	invInertiaLocal.setReciprocal< HK_ACC_FULL, HK_DIV_SET_HIGH >(eigenVal);
	m_inverseInertiaLocal.setXYZ_W(invInertiaLocal, hkSimdReal::getConstant(HK_QUADREAL_1));
	hkQuaternion bodyQmotion;
	bodyQmotion.set( (hkRotation&)eigenVec);
	bodyQmotion.normalize();
	m_orientation.setMul( bodyOrientation, bodyQmotion );
}

void hknpMotionCinfo::setBoxInertia( hkVector4Parameter halfExtents, hkReal mass )
{
	HK_ASSERT2(0x33baaf42,  mass > 0.0f, "Cannot calculate mass properties with zero mass or less." );

	hkReal x = halfExtents(0);
	hkReal y = halfExtents(1);
	hkReal z = halfExtents(2);
	hkReal k = mass * ( 1.0f / 3.0f);

	hkVector4 inertia;	inertia.set( (y*y + z*z) * k, (x*x + z*z) * k, (x*x + y*y) * k, 1.0f );
	m_inverseInertiaLocal.setReciprocal(inertia);
	m_inverseInertiaLocal.setComponent<3>(hkSimdReal::getConstant(HK_QUADREAL_1));

	m_inverseMass = 1.0f / mass;
}


void hknpMotionCinfo::initialize( const hknpBodyCinfo* bodyCinfos, int numCinfos )
{
	HK_ASSERT( 0xb4e656d0, bodyCinfos && numCinfos > 0 );

	// The first body's transform is taken as reference
	hkTransform refTransform;
	const hkQuaternion& refRotation = bodyCinfos[0].m_orientation;
	refTransform.set( refRotation, bodyCinfos[0].m_position );
	refTransform.setElement<3,3>( hkSimdReal_0 );	// avoid propagating uninitialized values

	// Get mass properties
	hkMassProperties massProps;
	if( numCinfos == 1 )
	{
		getOrBuildMassProperties( bodyCinfos->m_shape, massProps );
	}
	else
	{
		getCombinedMassProperties( bodyCinfos, numCinfos, refTransform, massProps );
	}

	// In compound body case, the mass properties will be higher than 1.0 so we need to scale the inertia
	m_inverseMass = 1.0f / massProps.m_mass;
	m_massFactor = 1.0f;

	setOrientationAndInertiaLocal( refRotation, massProps.m_inertiaTensor );
	m_centerOfMassWorld.setTransformedPos( refTransform, massProps.m_centerOfMass );
}

void hknpMotionCinfo::initializeWithMass( const hknpBodyCinfo* bodyCinfos, int numCinfos, hkReal mass )
{
	HK_ASSERT( 0xb4e656d0, bodyCinfos && numCinfos > 0 );
	HK_ASSERT2( 0xb4e656d1, mass > 0.0f, "Mass must be positive and non-zero" );

	// The first body's transform is taken as reference
	hkTransform refTransform;
	const hkQuaternion& refRotation = bodyCinfos[0].m_orientation;
	refTransform.set( refRotation, bodyCinfos[0].m_position );
	refTransform.setElement<3,3>( hkSimdReal_0 );	// avoid propagating uninitialized values

	// Get mass properties
	hkMassProperties massProps;
	if( numCinfos == 1 )
	{
		getOrBuildMassProperties( bodyCinfos->m_shape, massProps );
	}
	else
	{
		getCombinedMassProperties( bodyCinfos, numCinfos, refTransform, massProps );
	}

	m_inverseMass = 1.0f / mass;

	// Calculate mass factor
	hkSimdReal massFactor;
	massFactor.setFromFloat(mass);
	massFactor.div( hkSimdReal::fromFloat(massProps.m_mass) );
	m_massFactor = massFactor.getReal();

	massProps.m_inertiaTensor.mul( massFactor );

	setOrientationAndInertiaLocal( refRotation, massProps.m_inertiaTensor );
	m_centerOfMassWorld.setTransformedPos( refTransform, massProps.m_centerOfMass );
}

void hknpMotionCinfo::initializeWithDensity( const hknpBodyCinfo* bodyCinfos, int numCinfos, hkReal density )
{
	HK_ASSERT( 0xb4e656d0, bodyCinfos && numCinfos > 0 );
	HK_ASSERT2( 0xb4e656d1, density > 0.0f, "Density must be positive and non-zero" );

	// The first body's transform is taken as reference
	hkTransform refTransform;
	const hkQuaternion& refRotation = bodyCinfos[0].m_orientation;
	refTransform.set( refRotation, bodyCinfos[0].m_position );
	refTransform.setElement<3,3>( hkSimdReal_0 );	// avoid propagating uninitialized values

	// Get mass properties
	hkMassProperties massProps;
	if( numCinfos == 1 )
	{
		getOrBuildMassProperties( bodyCinfos->m_shape, massProps );
	}
	else
	{
		getCombinedMassProperties( bodyCinfos, numCinfos, refTransform, massProps );
	}

	hkReal mass = density * massProps.m_volume;
	m_inverseMass = 1.0f / mass;

	// Calculate mass factor
	hkSimdReal massFactor;
	massFactor.setFromFloat( mass );
	massFactor.div( hkSimdReal::fromFloat(massProps.m_mass) );
	m_massFactor = massFactor.getReal();

	massProps.m_inertiaTensor.mul( massFactor );

	setOrientationAndInertiaLocal( refRotation, massProps.m_inertiaTensor );
	m_centerOfMassWorld.setTransformedPos( refTransform, massProps.m_centerOfMass );
}

void hknpMotionCinfo::initializeAsKeyFramed( const hknpBodyCinfo* bodyCinfos, int numCinfos )
{
	HK_ASSERT( 0xb4e656d0, bodyCinfos && numCinfos > 0 );

	// The first body's transform is taken as reference
	hkTransform refTransform;
	refTransform.set( bodyCinfos[0].m_orientation, bodyCinfos[0].m_position );
	refTransform.setElement<3,3>( hkSimdReal_0 );	// avoid propagating uninitialized values

	// Get mass properties (just for the COM)
	hkMassProperties massProps;
	if( numCinfos == 1 )
	{
		getOrBuildMassProperties( bodyCinfos->m_shape, massProps );
	}
	else
	{
		getCombinedMassProperties( bodyCinfos, numCinfos, refTransform, massProps );
	}

	m_centerOfMassWorld.setTransformedPos( refTransform, massProps.m_centerOfMass );
	m_inverseMass = 0.0f;
	m_inverseInertiaLocal.setZero();
	m_motionPropertiesId = hknpMotionPropertiesId::KEYFRAMED;
}


/// Stops any angular movement by setting the invInertia to 0.
void hknpMotionCinfo::fixRotation()
{
	m_inverseInertiaLocal.setZero();
}

/// Stops any linear movement by setting the invMass to 0.
void hknpMotionCinfo::fixPosition()
{
	m_inverseMass = 0.0f;
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
