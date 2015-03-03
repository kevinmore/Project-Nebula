/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */

#include <Physics/Physics/hknpPhysics.h>
#include <Physics/Physics/Dynamics/Action/Spring/hknpSpringAction.h>
#include <Physics/Physics/Dynamics/Simulation/hknpSimulationThreadContext.h>

hknpSpringAction::hknpSpringAction( hknpBodyId idA, hknpBodyId idB, hkUlong userData )
: hknpBinaryAction( idA, idB, userData ),
  m_restLength(1.0f),
  m_strength(1000.0f),
  m_damping(0.1f),
  m_onCompression(true),
  m_onExtension(true)
{
}

void hknpSpringAction::initialize( hknpBodyId idA, hknpBodyId idB, hkUlong userData )
{
	hknpBinaryAction::initialize( idA, idB, userData );
	m_restLength = 1.0f;
	m_strength = 1000.0f;
	m_damping = 0.1f;
	m_onCompression = true;
	m_onExtension = true;
}

void hknpSpringAction::setPositionsInWorldSpace( const hkTransform& bodyATransform, hkVector4Parameter pivotA, const hkTransform& bodyBTransform, hkVector4Parameter pivotB )
{
	m_positionAinA.setTransformedInversePos(bodyATransform,pivotA);
	m_positionBinB.setTransformedInversePos(bodyBTransform,pivotB);
	hkVector4 dist;
	dist.setSub(pivotA,pivotB);
	dist.length<3>().store<1>( &m_restLength );
}

void hknpSpringAction::setPositionsInBodySpace( hkVector4Parameter pivotA, hkVector4Parameter pivotB, hkReal restLength )
{
	m_positionAinA = pivotA;
	m_positionBinB = pivotB;
	m_restLength = restLength;
}

hknpSpringAction::ApplyActionResult hknpSpringAction::applyAction( const hknpSimulationThreadContext& tl, const hknpSolverInfo& stepInfo, hknpCdPairWriter* HK_RESTRICT pairWriter )
{
	hknpWorld* world = tl.m_world;

	const hknpBody* bodyA, *bodyB;
	ApplyActionResult result = getAndCheckBodies( world, bodyA, bodyB );
	if ( result != RESULT_OK )
	{
		return result;
	}

	hknpMotion* HK_RESTRICT motionA = getMotion( world, *bodyA );
	hknpMotion* HK_RESTRICT motionB = getMotion( world, *bodyB );

	// tell the hknpDeactivationManager that both bodies are linked together
	addLink( motionA, motionB, pairWriter );

	hkVector4 posA;	posA.setTransformedPos( bodyA->getTransform(), m_positionAinA );
	hkVector4 posB;	posB.setTransformedPos( bodyB->getTransform(), m_positionBinB );

	hkVector4 dirAB; dirAB.setSub( posB, posA );

	hkSimdReal length = dirAB.normalizeWithLength<3>();
	hkSimdReal restLength; restLength.setFromFloat( m_restLength );
	{
		if( !m_onCompression && (length < restLength) )
		{
			return hknpAction::RESULT_OK;
		}
		if( !m_onExtension && (length > restLength) )
		{
			return hknpAction::RESULT_OK;
		}

		hkVector4 velA;		motionA->getPointVelocity( posA, velA );
		hkVector4 velB;		motionB->getPointVelocity( posB, velB );
		hkVector4 velAB;	velAB.setSub( velB, velA );
		hkSimdReal relVel = velAB.dot<3>( dirAB );

		// project length into the future
		//length = length + relVel * stepInfo.m_deltaTime;

		hkSimdReal strength; strength.setFromFloat( m_strength );
		hkSimdReal damping;  damping.setFromFloat( m_damping );
		hkSimdReal force = (relVel * damping) + ((length - restLength) * strength);

		hkVector4 force4; force4.setMul( force, dirAB );
		m_lastForce = force4;
		hkVector4 impulse; impulse.setMul( stepInfo.m_deltaTime, force4 );

		motionA->applyPointImpulse( impulse, posA );

		impulse.setNeg<4>( impulse );
		motionB->applyPointImpulse( impulse, posB );
	}

	return hknpAction::RESULT_OK;
}

void hknpSpringAction::onShiftWorld( hkVector4Parameter offset )
{
	// Update positions if the body is the ground one
	if (m_bodyA.value() == 0)
	{
		m_positionAinA.add( offset );
	}
	if (m_bodyB.value() == 0)
	{
		m_positionBinB.add( offset );
	}
}

//const hkReal dampingRatio = 0.1f;	// Underdamped (oscillates), C < 1
//const hkReal dampingRatio = 1.f;	// critically damped (equilibrates quickly), C = 1
//const hkReal dampingRatio = 2.f;	// Overdamped (no oscillation, slow equilibration), C > 1
/*static*/ hkReal hknpSpringAction::calcDampingfromDampingRatio( const hkReal& dampingRatio , const hkReal& strength, const hkReal& mass )
{
	HK_ASSERT2(0x54712600, mass > 0.f, "Body mass must be positive" );
	HK_ASSERT2(0x54712601, strength > 0.f, "Spring strength stiffness must be positive" );
	HK_ASSERT2(0x54712602, dampingRatio >= 0.f, "Spring damping ratio must not be negative" );
	// damping c = C(2 sqrt(mk))
	const hkReal damping = dampingRatio * 2.f * hkMath::sqrt( mass * strength );
	return damping;
}

/*static*/ hkReal hknpSpringAction::calcDampingRatiofromDamping( const hkReal& damping, const hkReal& strength, const hkReal& mass )
{
	HK_ASSERT2(0x54712600, mass > 0.f, "Body mass must be positive" );
	HK_ASSERT2(0x54712601, strength > 0.f, "Spring strength stiffness must be positive" );
	HK_ASSERT2(0x54712602, damping >= 0.f, "Spring damping must not be negative" );
	// damping ratio C = c/(2 sqrt(mk))
	const hkReal dampingRatio = damping * 0.5f / hkMath::sqrt( mass * strength );
	return dampingRatio;
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
