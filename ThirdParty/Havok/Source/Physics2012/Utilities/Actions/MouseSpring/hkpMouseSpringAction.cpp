/*
 *
 * Confidential Information of Telekinesys Research Limited (t/a Havok). Not for disclosure or distribution without Havok's
 * prior written consent. This software contains code, techniques and know-how which is confidential and proprietary to Havok.
 * Product and Trade Secret source code contains trade secrets of Havok. Havok Software (C) Copyright 1999-2013 Telekinesys Research Limited t/a Havok. All Rights Reserved. Use of this software is subject to the terms of an end user license agreement.
 *
 */
#include <Common/Base/hkBase.h>
#include <Common/Base/Math/Matrix/hkMatrix3Util.h>
#include <Physics2012/Utilities/Actions/MouseSpring/hkpMouseSpringAction.h>
#include <Physics2012/Dynamics/Entity/hkpRigidBody.h>
#include <Physics2012/Internal/Solver/SimpleConstraints/hkpSimpleConstraintUtil.h>

#include <Physics2012/Collide/Dispatch/hkpCollisionDispatcher.h>
#include <Physics2012/Collide/Agent/hkpProcessCollisionInput.h>
#include <Physics2012/Collide/Shape/Convex/Sphere/hkpSphereShape.h>
#include <Physics2012/Collide/Query/Collector/PointCollector/hkpClosestCdPointCollector.h>

#define HKP_ENABLE_MOUSE_VELOCITY_DAMPING

char hkpMouseSpringAction::s_name[] = "hkpMouseSpringAction";

hkpMouseSpringAction::hkpMouseSpringAction( hkpRigidBody* rb )
: hkpUnaryAction( rb ),
  m_springDamping(0.5f),
  m_springElasticity(0.3f),
  m_maxRelativeForce(250.0f),
  m_objectDamping(0.95f)
{
	m_positionInRbLocal.setZero();
	m_mousePositionInWorld.setZero();
	m_shapeKey = HK_INVALID_SHAPE_KEY;
	m_name = s_name;
}

hkpMouseSpringAction::hkpMouseSpringAction( 
	const hkVector4& positionInRbLocal, const hkVector4& mousePositionInWorld,
	const hkReal springDamping, const hkReal springElasticity, 
	const hkReal objectDamping, hkpRigidBody* rb,
	const hkArray<hkpMouseSpringAction::mouseSpringAppliedCallback>* appliedCallbacks )
:	hkpUnaryAction( rb ),
	m_positionInRbLocal( positionInRbLocal ),
	m_mousePositionInWorld( mousePositionInWorld ),
	m_springDamping( springDamping ),
	m_springElasticity( springElasticity ),
	m_objectDamping( objectDamping )
{
	m_maxRelativeForce = 250.0f; // that's 25 times gravity

	if (appliedCallbacks)
	{
		m_applyCallbacks = *appliedCallbacks;
	}

	hkpWorld* world = rb->getWorld();
	// Get the shapeKey that we attach to
	m_shapeKey = HK_INVALID_SHAPE_KEY;

	if (world)
	{
		// Create a dummy sphere in the place of attachment of our spring mouse
		hkpSphereShape mousePointShape(1.f);
		hkTransform mousePointTransform; mousePointTransform.set((const hkRotation&)hkRotation::getIdentity(), mousePositionInWorld);
		hkpCollidable mousePoint(&mousePointShape, &mousePointTransform);

		hkpShapeType typeA = mousePointShape.getType();
		hkpShapeType typeB = rb->getCollidable()->getShape()->getType();

		hkpCollisionDispatcher::GetClosestPointsFunc getClosestPointFunc = world->m_collisionDispatcher->getGetClosestPointsFunc( typeA, typeB );

		hkpCollisionInput input = *world->m_collisionInput;
		hkpClosestCdPointCollector collector;
		getClosestPointFunc( mousePoint, *rb->getCollidable(), input, collector ); 

		if ( collector.hasHit() )
		{
			m_shapeKey = collector.getHit().m_shapeKeyB;
		}
	}
	m_name = s_name;
}

void hkpMouseSpringAction::setMousePosition( const hkVector4& mousePositionInWorld )
{
	if ( !mousePositionInWorld.allEqual<3>( m_mousePositionInWorld, hkSimdReal::fromFloat(1e-3f) ) )
	{
		hkpRigidBody* rb = getRigidBody();
		if ( rb && rb->isAddedToWorld() )
		{
			rb->activate();
		}
	}
	m_mousePositionInWorld = mousePositionInWorld;
}

void hkpMouseSpringAction::setMaxRelativeForce(hkReal newMax)
{
	m_maxRelativeForce = newMax;
}



void hkpMouseSpringAction::applyAction( const hkStepInfo& stepInfo )
{
	hkpRigidBody* rb = getRigidBody();

	// calculate and apply the rigid spring mouse impluse
	const hkVector4& pMouse = m_mousePositionInWorld;
	//m_positionInRbLocal.setZero(); // for centra picking

	hkVector4 pRb; pRb.setTransformedPos(rb->getTransform(), m_positionInRbLocal);

	hkVector4 ptDiff;
	ptDiff.setSub(pRb, pMouse);

	hkpMotion* dynamicMotion = rb->getStoredDynamicMotion() ?  rb->getStoredDynamicMotion() : rb->getMotion();

	// calculate the jacobian
	hkMatrix3 jacobian;
	{
		const hkSimdReal massInv = dynamicMotion->getMassInv();

		hkVector4 r;
		r.setSub( pRb, rb->getCenterOfMassInWorld() );

		hkMatrix3 rhat;
		rhat.setCrossSkewSymmetric(r);

		hkMatrix3 inertialInvWorld;
		dynamicMotion->getInertiaInvWorld(inertialInvWorld);
	
		//jacobian.setZero(); this is not necessary!
		hkMatrix3Util::_setDiagonal( massInv, jacobian );

		// calculate: jacobian -= (rhat * inertialInvWorld * rhat)
		hkMatrix3 temp;
		temp.setMul(rhat, inertialInvWorld);
		hkMatrix3 temp2;
		temp2.setMul(temp, rhat);
		jacobian.sub(temp2);
	}
	

	// invert the jacobian as: jacobian * impluse = velocityDelta...
	// we want to calculate the impluse
	const hkResult jacInvertResult = jacobian.invert(0.0000001f);
	if ( jacInvertResult != HK_SUCCESS )
	{
		return;
	}

#if defined(HKP_ENABLE_MOUSE_VELOCITY_DAMPING)
	// apply damping
	hkVector4 linearVelocity;
	hkVector4 angularVelocity;
	{
		const hkSimdReal damp = hkSimdReal::fromFloat(m_objectDamping);
		linearVelocity = rb->getLinearVelocity();
		linearVelocity.mul(damp);
		rb->setLinearVelocity(linearVelocity);

		angularVelocity = rb->getAngularVelocity();
		angularVelocity.mul(damp);
		rb->setAngularVelocity(angularVelocity);
	}
#endif

	// calculate the velocity delta
	hkVector4 delta;
	{
		hkVector4 relVel;
		rb->getPointVelocity(pRb, relVel);
		delta.setMul(hkSimdReal::fromFloat(m_springElasticity * stepInfo.m_invDeltaTime), ptDiff);
		delta.addMul(hkSimdReal::fromFloat(m_springDamping), relVel);
	}

	// calculate the impluse
	hkVector4 impulse;
	{
		impulse._setRotatedDir(jacobian, delta);	// jacobian is actually the jacobian inverse here!
		impulse.setNeg<4>(impulse);
	}

	//
	//	clip the impulse
	//
	hkSimdReal impulseLen2 = impulse.lengthSquared<3>();
	hkSimdReal mass;
	{
		const hkSimdReal massInv = dynamicMotion->getMassInv();
		if (massInv.isEqualZero())
		{
			mass.setZero();
		}
		else
		{
			mass.setReciprocal(massInv);
		}
	}
	hkSimdReal maxImpulse  = mass * hkSimdReal::fromFloat(stepInfo.m_deltaTime * m_maxRelativeForce); 
	if ( impulseLen2 > maxImpulse * maxImpulse )
	{
		hkSimdReal factor = maxImpulse * impulseLen2.sqrtInverse();
		impulse.mul( factor );
	}


	rb->applyPointImpulse(impulse, pRb);

	for (int i = 0; i < m_applyCallbacks.getSize(); i++)
	{
		m_applyCallbacks[i](this, stepInfo, impulse);
	}

}

void hkpMouseSpringAction::entityRemovedCallback(hkpEntity* entity) 
{                     
	hkpUnaryAction::entityRemovedCallback(entity);

	// this line is needed for mouse action
	m_entity = HK_NULL;
}

hkpAction* hkpMouseSpringAction::clone( const hkArray<hkpEntity*>& newEntities, const hkArray<hkpPhantom*>& newPhantoms ) const
{
	HK_ASSERT2(0xf578efca, newEntities.getSize() == 1, "Wrong clone parameters given to a mousespring action (needs 1 body).");
	// should have two entities as we are a unary action.
	if (newEntities.getSize() != 1) return HK_NULL;

	HK_ASSERT2(0x5b74e112, newPhantoms.getSize() == 0, "Wrong clone parameters given to a mousespring action (needs 0 phantoms).");
	// should have no phantoms.
	if (newPhantoms.getSize() != 0) return HK_NULL;

	hkpMouseSpringAction* ms = new hkpMouseSpringAction( (hkpRigidBody*)newEntities[0] );
	ms->m_positionInRbLocal = m_positionInRbLocal;
	ms->m_mousePositionInWorld = m_mousePositionInWorld;
	ms->m_springDamping = m_springDamping;
	ms->m_springElasticity = m_springElasticity;
	ms->m_maxRelativeForce = m_maxRelativeForce;
	ms->m_objectDamping = m_objectDamping;
	ms->m_userData = m_userData;
	ms->m_shapeKey = m_shapeKey;
	ms->m_applyCallbacks = m_applyCallbacks;
	return ms;
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
